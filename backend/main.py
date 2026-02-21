from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import yfinance as yf
import os
import asyncio
import requests as _requests
from datetime import datetime
from typing import Optional, List
import time
import pytz
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.output_parsers import StrOutputParser
from prompts import MACRO_ANALYSIS_PROMPT
import json

# Load environment variables
load_dotenv()

# yfinance 세션 설정 (클라우드 환경 차단 방지)
import requests as req_lib
_session = req_lib.Session()
_session.headers.update({'User-Agent': 'Mozilla/5.0 FinAgent/1.0'})

app = FastAPI()

# CORS 설정 (별도 도메인 배포 시 필수)
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 실제 운영 시에는 Vercel 주소만 허용하는 것이 좋습니다.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── In-memory cache for market data (60s TTL) ──
_market_cache: dict = {"data": None, "signals": None, "fetched_at": 0}
CACHE_TTL = 60  # seconds

# ── In-memory cache for FRED liquidity data (6h TTL) ──
_liquidity_cache: dict = {"data": None, "fetched_at": 0}
LIQUIDITY_CACHE_TTL = 6 * 3600  # 6시간 (FRED는 일/주 단위 업데이트)

# ── RAG: Load institutional reports once at startup ──
_institutional_context: str = "No specific reports available."
try:
    _report_path = os.path.join(os.path.dirname(__file__), "data", "institutional_reports.json")
    if os.path.exists(_report_path):
        with open(_report_path, "r", encoding="utf-8") as _f:
            _report_data = json.load(_f)
            _institutional_context = "\n\n".join([
                f"[{r['institution']} - {r['title']}]\nSummary: {r['summary']}\nDetails: {r['content']}"
                for r in _report_data.get("reports", [])
            ])
        print(f"[RAG] Loaded {len(_report_data.get('reports', []))} institutional reports.")
except Exception as _rag_e:
    print(f"[RAG] Load error: {_rag_e}")


class MarketData(BaseModel):
    symbol: str
    price: float
    change_percent: float
    name: str


@app.get("/")
def read_root():
    return {"message": "Financial Macro Agent API is running"}


def fetch_market_data_internal():
    symbols = {
        "KRW=X": "USD/KRW",
        "^KS11": "KOSPI",
        "^KQ11": "KOSDAQ",
        "^TNX": "US 10Y Bond",
        "^IRX": "US 13W Bond",
        "CL=F": "WTI Oil",
        "GC=F": "Gold",
        "BTC-USD": "Bitcoin",
        "ETH-USD": "Ethereum",
        "005930.KS": "Samsung Elec",
        "DX-Y.NYB": "Dollar Index",
    }

    data = []
    signals = []

    try:
        # yf.download에 세션 전달 및 벌크 패치
        ticker_list = list(symbols.keys())
        df = yf.download(ticker_list, period="2d", interval="1d", group_by='ticker')
        
        for symbol, name in symbols.items():
            try:
                if symbol not in df.columns.levels[0]: continue
                hist = df[symbol]
                if hist.empty or len(hist) < 1: continue
                
                current = hist['Close'].iloc[-1]
                prev = hist['Close'].iloc[-2] if len(hist) > 1 else current
                change = ((current - prev) / prev) * 100 if prev != 0 else 0
                
                data.append({
                    "symbol": symbol,
                    "name": name,
                    "price": round(float(current), 2),
                    "change_percent": round(float(change), 2)
                })
            except Exception:
                continue
    except Exception as e:
        print(f"Error in batch fetching: {e}")

    # Yield Curve signal
    try:
        us10y = next((x for x in data if x['symbol'] == '^TNX'), None)
        us13w = next((x for x in data if x['symbol'] == '^IRX'), None)
        if us10y and us13w:
            spread = us10y['price'] - us13w['price']
            signals.append(f"Yield Curve (10Y-13W): {spread:.2f}bp ({'Inverted!' if spread < 0 else 'Normal'})")
    except Exception:
        pass

    return {"data": data, "signals": signals}


# ───────────────────────────────────────────────── #
# ◀ 연준 / 재무부 유동성 지표 (FRED 무료 공개 API)    #
# ───────────────────────────────────────────────── #

FRED_LIQUIDITY_SERIES = {
    # 연준 대차대조표 (QT/QE 구분 핵심) — 단위: 백만달러 → 10억달러($B)로 변환
    "WALCL":     {"name": "Fed Balance Sheet",         "unit": "$B",  "scale": 1e-3,  "desc": "연준 총자산 — QT/QE 판단 핵심 지표"},
    # ON RRP — 단기 유동성 통제 통로 — 단위: 10억달러($B)
    "RRPONTSYD": {"name": "ON RRP Balance",            "unit": "$B",  "scale": 1.0,  "desc": "연준 역레포 잔고 (단기 유동성 흡수량)"},
    # TGA — 재무부 일반계좌 — 단위: 백만달러($M)
    "WTREGEN":   {"name": "Treasury General Account",  "unit": "$M",  "scale": 1.0,  "desc": "재무부 TGA 잔액 — 국채 발행·상환 시 유동성 영향"},
    # SOFR — LIBOR 대체 기준금리 — 단위: %
    "SOFR":      {"name": "SOFR Rate",                 "unit": "%",   "scale": 1.0,  "desc": "단기 담보부 기준금리 (LIBOR 대체)"},
    # 실효 연방기금금리 — 단위: %
    "DFF":       {"name": "Effective Fed Funds Rate",  "unit": "%",   "scale": 1.0,  "desc": "실효 연방기금금리 (FOMC 정책 반영)"},
    # M2 통화량 — 단위: 10억달러($B)
    "M2SL":      {"name": "M2 Money Supply",           "unit": "$B",  "scale": 1.0,  "desc": "M2 광의 통화량 (시중 유동성 총량)"},
}


def fetch_fred_series(series_id: str):
    """FRED 공개 CSV 엔드포인트에서 최신값 1개 반환. API 키 불필요."""
    try:
        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
        resp = _requests.get(url, timeout=10,
                             headers={"User-Agent": "Mozilla/5.0 FinAgent/1.0"})
        resp.raise_for_status()
        lines = [l for l in resp.text.strip().splitlines() if l and not l.startswith("DATE")]
        last_line = lines[-1]  # 최신 데이터
        date_str, value_str = last_line.split(",")
        if value_str.strip() in ("", "."):  # 누락값
            # 누락값이면 이전 데이터 찾기
            for line in reversed(lines):
                d, v = line.split(",")
                if v.strip() not in ("", "."):
                    return float(v.strip()), d.strip()
            return None, None
        return float(value_str.strip()), date_str.strip()
    except Exception as e:
        print(f"[FRED] Error fetching {series_id}: {e}")
        return None, None


def fetch_liquidity_data() -> List[dict]:
    """FRED 6개 지표 패치 효치."""
    results = []
    for series_id, meta in FRED_LIQUIDITY_SERIES.items():
        value, date = fetch_fred_series(series_id)
        if value is not None:
            scaled = value * meta["scale"]
            results.append({
                "series": series_id,
                "name": meta["name"],
                "value": round(scaled, 2),
                "unit": meta["unit"],
                "date": date,
                "desc": meta["desc"],
            })
            print(f"[FRED] {series_id}: {scaled:.2f} {meta['unit']} ({date})")
        else:
            print(f"[FRED] {series_id}: N/A")
    return results


def _get_cached_liquidity() -> List[dict]:
    """FRED 유동성 데이터 케시 (6h TTL)."""
    global _liquidity_cache
    now = time.time()
    if _liquidity_cache["data"] is not None and (now - _liquidity_cache["fetched_at"]) < LIQUIDITY_CACHE_TTL:
        print(f"[Liquidity Cache HIT] age={(now - _liquidity_cache['fetched_at'])/3600:.1f}h")
        return _liquidity_cache["data"]
    print("[Liquidity Cache MISS] Fetching FRED liquidity data...")
    _liquidity_cache["data"] = fetch_liquidity_data()
    _liquidity_cache["fetched_at"] = now
    return _liquidity_cache["data"]


def _get_cached_market_data() -> dict:
    """캐시된 시장 데이터 반환. 만료 시 새로 fetch."""
    global _market_cache
    now = time.time()
    if _market_cache["data"] is not None and (now - _market_cache["fetched_at"]) < CACHE_TTL:
        print(f"[Cache HIT] age={(now - _market_cache['fetched_at']):.1f}s")
        return _market_cache
    print("[Cache MISS] Fetching fresh market data...")
    result = fetch_market_data_internal()
    _market_cache["data"] = result["data"]
    _market_cache["signals"] = result["signals"]
    _market_cache["fetched_at"] = now
    return _market_cache


@app.get("/api/market-data")
def get_market_data():
    cached = _get_cached_market_data()
    return {"data": cached["data"]}


@app.get("/api/liquidity")
def get_liquidity_data():
    """연준/재무부 유동성 지표 반환."""
    data = _get_cached_liquidity()
    return {"data": data}


class AnalysisRequest(BaseModel):
    query: str
    model: str = "gemini"


@app.post("/api/analyze")
async def analyze_macro(request: AnalysisRequest):
    api_key_openai = os.getenv("OPENAI_API_KEY")
    api_key_google = os.getenv("GOOGLE_API_KEY")
    api_key_anthropic = os.getenv("ANTHROPIC_API_KEY")

    llm = None
    model_source = "Unknown"
    requested_model = request.model.lower()

    # ── 모델 구성: 기본 + Provider 내부 Fallback ──
    # Gemini: 2.0-flash → 1.5-flash-8b
    # GPT:    gpt-4o    → gpt-4o-mini
    # Claude: 3.5-sonnet → 3.5-haiku

    GEMINI_MODELS = [
        ("gemini-2.0-flash",        "Google Gemini 2.0 Flash"),
        ("gemini-flash-lite-latest", "Google Gemini Flash Lite (Fallback)"),
    ]
    GPT_MODELS = [
        ("gpt-4o",                 "OpenAI GPT-4o"),
        ("gpt-4o-mini",            "OpenAI GPT-4o Mini (Fallback)"),
    ]
    CLAUDE_MODELS = [
        ("claude-3-5-sonnet-20241022", "Anthropic Claude 3.5 Sonnet"),
        ("claude-3-5-haiku-20241022",  "Anthropic Claude 3.5 Haiku (Fallback)"),
    ]

    def _is_retryable_error(msg: str) -> bool:
        keywords = ["429", "quota", "resourceexhausted", "rate_limit",
                    "rate limit", "too many requests", "overloaded",
                    "credit", "billing", "insufficient_quota", "503", "demand"]
        m = msg.lower()
        return any(k in m for k in keywords)

    def _build_gemini(model_id: str):
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=model_id,
            google_api_key=api_key_google,
            temperature=0.7,
            max_retries=0,  # langchain 내부 retry 비활성화 → 즉시 에러 전파
        )

    def _build_gpt(model_id: str):
        return ChatOpenAI(
            temperature=0.7,
            model_name=model_id,
            openai_api_key=api_key_openai,
            max_retries=0,
        )

    def _build_claude(model_id: str):
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=model_id,
            anthropic_api_key=api_key_anthropic,
            temperature=0.7,
            max_tokens=8096,
            max_retries=0,
        )

    # 선택된 provider의 모델 목록과 빌더 결정
    if "gemini" in requested_model:
        if not api_key_google:
            async def _err():
                yield "> ⚠️ **Google API 키가 없습니다.** `.env`에 `GOOGLE_API_KEY`를 추가해주세요."
            return StreamingResponse(_err(), media_type="text/plain")
        model_candidates = GEMINI_MODELS
        model_builder = _build_gemini

    elif "gpt" in requested_model or "openai" in requested_model:
        if not api_key_openai:
            async def _err():
                yield "> ⚠️ **OpenAI API 키가 없습니다.** `.env`에 `OPENAI_API_KEY`를 추가해주세요."
            return StreamingResponse(_err(), media_type="text/plain")
        model_candidates = GPT_MODELS
        model_builder = _build_gpt

    elif "claude" in requested_model:
        if not api_key_anthropic:
            async def _err():
                yield "> ⚠️ **Anthropic API 키가 없습니다.** `.env`에 `ANTHROPIC_API_KEY`를 추가해주세요."
            return StreamingResponse(_err(), media_type="text/plain")
        model_candidates = CLAUDE_MODELS
        model_builder = _build_claude

    else:
        async def _err():
            yield f"> ⚠️ **알 수 없는 모델**: `{request.model}`. gemini / gpt / claude 중 하나를 선택해주세요."
        return StreamingResponse(_err(), media_type="text/plain")

    # ── 1. 시장 데이터 + 유동성 데이터: 병렬 fetch ──
    try:
        # 시장 데이터와 유동성 데이터를 동시 fetchfetch
        loop = asyncio.get_event_loop()
        cached_task = loop.run_in_executor(None, _get_cached_market_data)
        liquidity_task = loop.run_in_executor(None, _get_cached_liquidity)
        cached, liquidity_data = await asyncio.gather(cached_task, liquidity_task)

        market_data = cached["data"]
        macro_signals = "\n".join(cached["signals"] or [])
        market_str = "\n".join([
            f"{item['name']} ({item['symbol']}): {item['price']} ({item['change_percent']}%) [Momentum: {item.get('momentum', 'N/A')}]"
            for item in market_data
        ])
        print(f"[Market] {len(market_data)} items | [Liquidity] {len(liquidity_data)} indicators")

        # 유동성 컨텍스트 문자열 조립
        if liquidity_data:
            liquidity_context = "\n".join([
                f"{item['name']} ({item['series']}): {item['value']:,.2f} {item['unit']}  [{item['date']}]  — {item['desc']}"
                for item in liquidity_data
            ])
        else:
            liquidity_context = "FRED 유동성 데이터를 가져오지 못했습니다."

    except Exception as e:
        import traceback; traceback.print_exc()
        async def _err():
            yield f"> ⚠️ **시장 데이터 로드 실패**: {str(e)}"
        return StreamingResponse(_err(), media_type="text/plain")

    # ── 2. 검색: asyncio.gather로 병렬 실행 ──
    user_query = request.query

    async def _search(q: str) -> str:
        try:
            loop = asyncio.get_event_loop()
            search = DuckDuckGoSearchRun()
            return await loop.run_in_executor(None, search.run, q)
        except Exception as e:
            print(f"[Search WARN] '{q[:30]}': {e}")
            return "Data unavailable."

    print("[Search] Starting parallel DuckDuckGo searches...")
    results = await asyncio.gather(
        _search(f"{user_query} 최신 글로벌 금융 뉴스 경제 분석 2025"),
        _search("한국은행 기준금리 결정 최신 통화정책 2025"),
        _search("Federal Reserve FOMC meeting outcome interest rate decision 2025"),
        _search("US CPI inflation PCE core 2025 latest"),
        _search("한국 반도체 수출 동향 2025"),
    )
    news_summary, bok_policy, fed_policy, cpi_signal, semi_signal = results
    macro_signals += f"\n{cpi_signal}\n{semi_signal}"
    print("[Search] All parallel searches done.")

    # ── 3. LLM 스트리밍 (Provider 내부 Fallback 포함) ──
    chain_inputs = {
        "market_data": market_str,
        "news_summary": news_summary,
        "bok_policy": bok_policy,
        "fed_policy": fed_policy,
        "macro_signals": macro_signals,
        "institutional_context": _institutional_context,
        "liquidity_context": liquidity_context,
        "user_query": user_query,
    }

    async def generate():
        for attempt, (model_id, model_label) in enumerate(model_candidates):
            is_fallback = attempt > 0
            llm = model_builder(model_id)
            chain = MACRO_ANALYSIS_PROMPT | llm | StrOutputParser()
            print(f"[LLM] Streaming with {model_label} (attempt {attempt + 1})...")

            # fallback 시 사용자에게 알림
            if is_fallback:
                yield f"\n\n> ⚡ **{model_label}** 로 자동 전환하여 재시도합니다.\n\n"
            else:
                yield f"**[{model_label}]**\n\n"

            try:
                accumulated = ""
                async for chunk in chain.astream(chain_inputs):
                    accumulated += chunk
                    yield chunk
                # 스트리밍 완료 — 성공
                print(f"[LLM] Done. model={model_label}, chars={len(accumulated)}")
                return

            except Exception as e:
                err_msg = str(e)
                print(f"[LLM Error] {model_label}: {err_msg[:120]}")

                if _is_retryable_error(err_msg) and attempt < len(model_candidates) - 1:
                    # 다음 fallback 모델로 재시도
                    next_label = model_candidates[attempt + 1][1]
                    print(f"[LLM] Error (retryable) → falling back to {next_label}")
                    yield f"\n\n> ⚠️ **{model_label}** 서비스 일시적 지연 또는 한도 초과. **{next_label}** 으로 자동 전환합니다..."
                    continue  # 다음 모델 시도
                elif _is_retryable_error(err_msg):
                    yield f"\n\n> 🚫 **모든 모델이 현재 지연 중이거나 한도를 초과했습니다.** 잠시 후 다시 시도해주세요."
                elif "401" in err_msg or "authentication" in err_msg.lower():
                    yield f"\n\n> ⚠️ **인증 오류**: API 키가 유효하지 않습니다. `.env` 파일을 확인해주세요."
                else:
                    yield f"\n\n> ⚠️ **분석 중 오류 발생**: {err_msg[:200]}"
                return

    return StreamingResponse(generate(), media_type="text/plain")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
