from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import yfinance as yf
import os
import asyncio
from datetime import datetime
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

app = FastAPI()

# ── In-memory cache for market data (60s TTL) ──
_market_cache: dict = {"data": None, "signals": None, "fetched_at": 0}
CACHE_TTL = 60  # seconds

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

    for symbol, name in symbols.items():
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="1mo")

            if not hist.empty:
                current = hist['Close'].iloc[-1]
                prev = hist['Close'].iloc[-2] if len(hist) > 1 else current
                change = ((current - prev) / prev) * 100

                ma5 = hist['Close'].tail(5).mean()
                ma20 = hist['Close'].tail(20).mean()

                momentum_signal = "Neutral"
                if current > ma5 and current > ma20:
                    momentum_signal = "Bullish (Short-term)"
                elif current < ma5 and current < ma20:
                    momentum_signal = "Bearish (Short-term)"

                data.append({
                    "symbol": symbol,
                    "name": name,
                    "price": round(current, 2),
                    "change_percent": round(change, 2),
                    "momentum": momentum_signal,
                })
        except Exception as e:
            print(f"Error fetching {symbol}: {e}")

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

    # ── 모델 선택 (fallback 없음, 키 없으면 즉시 에러) ──
    if "gemini" in requested_model:
        if not api_key_google:
            async def _err():
                yield "> ⚠️ **Google API 키가 없습니다.** `.env`에 `GOOGLE_API_KEY`를 추가해주세요."
            return StreamingResponse(_err(), media_type="text/plain")
        from langchain_google_genai import ChatGoogleGenerativeAI
        llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=api_key_google, temperature=0.7)
        model_source = "Google Gemini 2.0 Flash"

    elif "gpt" in requested_model or "openai" in requested_model:
        if not api_key_openai:
            async def _err():
                yield "> ⚠️ **OpenAI API 키가 없습니다.** `.env`에 `OPENAI_API_KEY`를 추가해주세요."
            return StreamingResponse(_err(), media_type="text/plain")
        llm = ChatOpenAI(temperature=0.7, model_name="gpt-4o", openai_api_key=api_key_openai)
        model_source = "OpenAI GPT-4o"

    elif "claude" in requested_model:
        if not api_key_anthropic:
            async def _err():
                yield "> ⚠️ **Anthropic API 키가 없습니다.** `.env`에 `ANTHROPIC_API_KEY`를 추가해주세요."
            return StreamingResponse(_err(), media_type="text/plain")
        from langchain_anthropic import ChatAnthropic
        llm = ChatAnthropic(model="claude-3-5-sonnet-20241022", anthropic_api_key=api_key_anthropic, temperature=0.7, max_tokens=8096)
        model_source = "Anthropic Claude 3.5 Sonnet"

    if not llm:
        async def _err():
            yield f"> ⚠️ **알 수 없는 모델**: `{request.model}`. gemini / gpt / claude 중 하나를 선택해주세요."
        return StreamingResponse(_err(), media_type="text/plain")

    # ── 1. 시장 데이터: 캐시 우선 사용 ──
    try:
        cached = _get_cached_market_data()
        market_data = cached["data"]
        macro_signals = "\n".join(cached["signals"] or [])
        market_str = "\n".join([
            f"{item['name']} ({item['symbol']}): {item['price']} ({item['change_percent']}%) [Momentum: {item.get('momentum', 'N/A')}]"
            for item in market_data
        ])
        print(f"[Market] Using {'cached' if cached['fetched_at'] else 'fresh'} data: {len(market_data)} items")
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

    # ── 3. LLM 스트리밍 ──
    chain = MACRO_ANALYSIS_PROMPT | llm | StrOutputParser()
    print(f"[LLM] Streaming with {model_source}...")

    async def generate():
        yield f"**[{model_source}]**\n\n"
        try:
            async for chunk in chain.astream({
                "market_data": market_str,
                "news_summary": news_summary,
                "bok_policy": bok_policy,
                "fed_policy": fed_policy,
                "macro_signals": macro_signals,
                "institutional_context": _institutional_context,
                "user_query": user_query,
            }):
                yield chunk
        except Exception as e:
            err_msg = str(e)
            print(f"[LLM Stream Error] {err_msg}")
            if "429" in err_msg or "quota" in err_msg.lower() or "ResourceExhausted" in err_msg:
                yield f"\n\n> ⚠️ **API 쿼터 초과**: {model_source}의 무료 한도를 초과했습니다. 다른 모델을 선택하거나 잠시 후 다시 시도해주세요."
            elif "401" in err_msg or "authentication" in err_msg.lower() or "credit" in err_msg.lower():
                yield f"\n\n> ⚠️ **인증/크레딧 오류**: API 키가 유효하지 않거나 크레딧이 부족합니다."
            else:
                yield f"\n\n> ⚠️ **분석 중 오류 발생**: {err_msg[:200]}"

    return StreamingResponse(generate(), media_type="text/plain")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
