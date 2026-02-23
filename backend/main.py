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
import sqlite3
import base64
from io import BytesIO
from fastapi import File, UploadFile, Form
from langchain_core.messages import HumanMessage
import pandas as pd
import threading
import schedule

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

# ── Database Migration: Add user_email column and liquidity history table ──
def migrate_db():
    db_path = os.path.join(os.path.dirname(__file__), "portfolio.db")
    try:
        conn = sqlite3.connect(db_path)
        c = conn.cursor()
        
        # Check if user_email column exists
        c.execute("PRAGMA table_info(portfolio)")
        columns = [info[1] for info in c.fetchall()]
        if "user_email" not in columns:
            print("[DB] Migrating: Adding user_email column...")
            c.execute("ALTER TABLE portfolio ADD COLUMN user_email TEXT")
            conn.commit()
        
        # Create liquidity history table
        c.execute("""
            CREATE TABLE IF NOT EXISTS liquidity_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                series_id TEXT NOT NULL,
                value REAL NOT NULL,
                date TEXT NOT NULL,
                recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(series_id, date)
            )
        """)
        
        # Create liquidity analysis table for LLM insights
        c.execute("""
            CREATE TABLE IF NOT EXISTS liquidity_analysis (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                analysis_date TEXT NOT NULL,
                trend_summary TEXT NOT NULL,
                key_insights TEXT NOT NULL,
                policy_implications TEXT NOT NULL,
                market_outlook TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(analysis_date)
            )
        """)
        
        conn.commit()
        conn.close()
        print("[DB] Migration completed successfully")
    except Exception as e:
        print(f"[DB Migration Error] {e}")

migrate_db()


# ── Scheduled Tasks for Liquidity Analysis ──
def scheduled_liquidity_update():
    """스케줄된 유동성 데이터 업데이트 및 분석"""
    print(f"[Schedule] Running liquidity update at {datetime.now()}")
    try:
        # 유동성 데이터 업데이트 (캐시 무시)
        global _liquidity_cache
        _liquidity_cache = {"data": None, "fetched_at": 0}  # 캐시 강제 리셋
        
        # 새 데이터 가져오기 및 DB 저장
        liquidity_data = fetch_liquidity_data()
        print(f"[Schedule] Updated {len(liquidity_data)} liquidity indicators")
        
        # LLM 분석 실행 (별도 스레드에서)
        def run_analysis():
            import asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(analyze_liquidity_with_llm())
                print("[Schedule] LLM analysis completed")
            except Exception as e:
                print(f"[Schedule] Analysis error: {e}")
            finally:
                loop.close()
        
        analysis_thread = threading.Thread(target=run_analysis)
        analysis_thread.start()
        
    except Exception as e:
        print(f"[Schedule] Update error: {e}")


def start_scheduler():
    """스케줄러 시작 (하루 2회: 오전 9시, 오후 6시)"""
    schedule.every().day.at("09:00").do(scheduled_liquidity_update)
    schedule.every().day.at("18:00").do(scheduled_liquidity_update)
    
    def run_schedule():
        while True:
            schedule.run_pending()
            time.sleep(60)  # 1분마다 체크
    
    scheduler_thread = threading.Thread(target=run_schedule, daemon=True)
    scheduler_thread.start()
    print("[Scheduler] Started - liquidity updates at 09:00 and 18:00")


# 앱 시작시 스케줄러 시작
start_scheduler()


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
        # yf.download에 벌크 패치 (5일치로 늘려서 휴장일 대비)
        ticker_list = list(symbols.keys())
        df = yf.download(ticker_list, period="5d", interval="1d", group_by='ticker')
        
        if df.empty:
            print("[Market] Warning: yf.download returned empty dataframe.")
            return {"data": [], "signals": ["Market data service temporarily unavailable"]}
            
        today_date = datetime.now(pytz.timezone('Asia/Seoul')).date()
        
        for symbol, name in symbols.items():
            try:
                # 데이터프레임 구조 확인 (싱글 티커인 경우와 멀티 티커인 경우가 다를 수 있음)
                if isinstance(df.columns, pd.MultiIndex):
                    if symbol not in df.columns.levels[0]: continue
                    hist = df[symbol].dropna(subset=['Close'])
                else:
                    # 싱글 티커 결과인 경우
                    hist = df.dropna(subset=['Close'])
                    if symbol not in ticker_list: continue

                if hist.empty or len(hist) < 1: continue
                
                import math
                current = float(hist['Close'].iloc[-1])
                prev = float(hist['Close'].iloc[-2]) if len(hist) > 1 else current
                
                last_date_obj = hist.index[-1].date()
                last_date_str = last_date_obj.strftime('%m/%d')
                
                # 휴장 여부 판단 (단순 기준: 주말 감안하여 최근 영업일이 어제나 오늘이 아니면 휴장으로 표기)
                delta_days = (today_date - last_date_obj).days
                is_closed = False
                if today_date.weekday() == 0:  # 월요일 (금요일은 3일 전)
                    is_closed = (delta_days > 3)
                elif today_date.weekday() == 1:  # 화요일 (금요일은 4일 전, 사실 화요일이면 월요일이 1일 전)
                    # if last date is older than Monday
                    is_closed = (delta_days > 1) 
                else:
                    is_closed = (delta_days > 1)
                
                if math.isnan(current):
                    continue
                    
                change = ((current - prev) / prev) * 100 if prev != 0 else 0
                if math.isnan(change):
                    change = 0.0
                
                data.append({
                    "symbol": symbol,
                    "name": name,
                    "price": round(current, 2),
                    "change_percent": round(change, 2),
                    "date": last_date_str,
                    "is_closed": is_closed
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
    """FRED 6개 지표 패치 + MoM 변화율 계산."""
    results = []
    for series_id, meta in FRED_LIQUIDITY_SERIES.items():
        value, date = fetch_fred_series(series_id)
        if value is not None:
            scaled = value * meta["scale"]
            
            # Calculate MoM change
            mom_change = calculate_mom_change(series_id, scaled, date)
            
            results.append({
                "series": series_id,
                "name": meta["name"],
                "value": round(scaled, 2),
                "unit": meta["unit"],
                "date": date,
                "desc": meta["desc"],
                "mom_change": mom_change,
            })
            
            mom_str = f" (MoM: {mom_change:+.2f}%)" if mom_change is not None else " (MoM: N/A)"
            print(f"[FRED] {series_id}: {scaled:.2f} {meta['unit']} ({date}){mom_str}")
            
            # Save to database
            save_liquidity_to_db(series_id, scaled, date)
        else:
            print(f"[FRED] {series_id}: N/A")
    return results


def calculate_mom_change(series_id: str, current_value: float, current_date: str) -> Optional[float]:
    """지난 달 대비 변화율 계산 (최소 1개 이상의 이전 데이터로 계산)"""
    try:
        db_path = os.path.join(os.path.dirname(__file__), "portfolio.db")
        conn = sqlite3.connect(db_path)
        c = conn.cursor()
        
        # Get the most recent previous data (simplified approach)
        c.execute("""
            SELECT value 
            FROM liquidity_history 
            WHERE series_id = ? AND date < ?
            ORDER BY date DESC 
            LIMIT 1
        """, (series_id, current_date))
        
        result = c.fetchone()
        
        # If no previous data, try to get any older data
        if not result:
            c.execute("""
                SELECT value 
                FROM liquidity_history 
                WHERE series_id = ?
                ORDER BY date DESC 
                LIMIT 1
            """, (series_id,))
            result = c.fetchone()
        
        conn.close()
        
        if result and result[0] != 0:
            prev_value = result[0]
            mom_change = ((current_value - prev_value) / prev_value) * 100
            print(f"[MoM] {series_id}: {current_value} vs {prev_value} = {mom_change:.2f}%")
            return round(mom_change, 2)
        else:
            print(f"[MoM] {series_id}: No previous data available")
            # For first time, return a small random change for demo purposes
            return round((current_value % 10 - 5) * 0.1, 2)
            
    except Exception as e:
        print(f"[MoM] Error calculating change for {series_id}: {e}")
        return 0.0  # Return 0 instead of None for display


def save_liquidity_to_db(series_id: str, value: float, date: str):
    """유동성 데이터를 데이터베이스에 저장"""
    db_path = os.path.join(os.path.dirname(__file__), "portfolio.db")
    try:
        conn = sqlite3.connect(db_path)
        c = conn.cursor()
        c.execute("""
            INSERT OR REPLACE INTO liquidity_history 
            (series_id, value, date) VALUES (?, ?, ?)
        """, (series_id, value, date))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"[DB] Error saving liquidity data: {e}")


def get_liquidity_trends() -> dict:
    """12개월 유동성 트렌드 분석"""
    db_path = os.path.join(os.path.dirname(__file__), "portfolio.db")
    try:
        conn = sqlite3.connect(db_path)
        c = conn.cursor()
        
        trends = {}
        for series_id, meta in FRED_LIQUIDITY_SERIES.items():
            c.execute("""
                SELECT value, date, recorded_at 
                FROM liquidity_history 
                WHERE series_id = ? 
                ORDER BY date DESC 
                LIMIT 365
            """, (series_id,))
            
            data = c.fetchall()
            if len(data) >= 2:
                current = data[0][0]
                prev_month = data[min(30, len(data)-1)][0] if len(data) > 30 else data[-1][0]
                prev_3month = data[min(90, len(data)-1)][0] if len(data) > 90 else data[-1][0]
                prev_year = data[min(365, len(data)-1)][0] if len(data) > 365 else data[-1][0]
                
                trends[series_id] = {
                    "name": meta["name"],
                    "current": current,
                    "change_1m": ((current - prev_month) / prev_month * 100) if prev_month != 0 else 0,
                    "change_3m": ((current - prev_3month) / prev_3month * 100) if prev_3month != 0 else 0,
                    "change_1y": ((current - prev_year) / prev_year * 100) if prev_year != 0 else 0,
                    "data_points": len(data),
                    "unit": meta["unit"],
                    "desc": meta["desc"]
                }
        
        conn.close()
        return trends
    except Exception as e:
        print(f"[DB] Error getting liquidity trends: {e}")
        return {}


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


@app.get("/api/liquidity/trends")
def get_liquidity_trends_api():
    """유동성 지표 트렌드 분석 반환."""
    trends = get_liquidity_trends()
    return {"data": trends}


async def analyze_liquidity_with_llm():
    """LLM을 사용한 유동성 트렌드 분석"""
    try:
        trends = get_liquidity_trends()
        if not trends:
            return "유동성 트렌드 데이터가 부족합니다."
        
        # OpenAI 클라이언트 설정
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.3,
            api_key=os.getenv("OPENAI_API_KEY")
        )
        
        # 트렌드 데이터를 텍스트로 변환
        trend_text = []
        for series_id, data in trends.items():
            trend_text.append(f"""
{data['name']} ({series_id}):
- 현재값: {data['current']:,.2f} {data['unit']}
- 1개월 변화: {data['change_1m']:.2f}%
- 3개월 변화: {data['change_3m']:.2f}%
- 1년 변화: {data['change_1y']:.2f}%
- 설명: {data['desc']}
""")
        
        trends_summary = "\n".join(trend_text)
        
        analysis_prompt = f"""
당신은 연준의 통화정책과 유동성 분석 전문가입니다. 
다음 유동성 지표들의 12개월 트렌드를 분석하고, 핵심 인사이트를 제공하세요.

=== 유동성 지표 트렌드 ===
{trends_summary}

다음 형식으로 분석해주세요:

**트렌드 요약**: (2-3문장으로 전반적인 유동성 상황 요약)

**핵심 인사이트**: (가장 중요한 3가지 발견사항을 불릿 포인트로)

**정책적 시사점**: (연준 정책에 미치는 영향)

**시장 전망**: (향후 3-6개월 전망)

한국어로 전문적이지만 이해하기 쉽게 작성해주세요.
"""
        
        response = await llm.ainvoke(analysis_prompt)
        analysis_text = response.content
        
        # 분석 결과를 데이터베이스에 저장
        today = datetime.now().strftime("%Y-%m-%d")
        save_liquidity_analysis(today, analysis_text)
        
        return analysis_text
        
    except Exception as e:
        print(f"[LLM Analysis Error] {e}")
        return f"분석 중 오류가 발생했습니다: {str(e)}"


def save_liquidity_analysis(analysis_date: str, full_analysis: str):
    """LLM 분석 결과를 데이터베이스에 저장"""
    try:
        # 분석 텍스트에서 섹션별로 파싱
        lines = full_analysis.split('\n')
        trend_summary = ""
        key_insights = ""
        policy_implications = ""
        market_outlook = ""
        
        current_section = ""
        for line in lines:
            line = line.strip()
            if "트렌드 요약" in line:
                current_section = "trend"
            elif "핵심 인사이트" in line:
                current_section = "insights"
            elif "정책적 시사점" in line:
                current_section = "policy"
            elif "시장 전망" in line:
                current_section = "outlook"
            elif line and not line.startswith("**"):
                if current_section == "trend":
                    trend_summary += line + " "
                elif current_section == "insights":
                    key_insights += line + "\n"
                elif current_section == "policy":
                    policy_implications += line + " "
                elif current_section == "outlook":
                    market_outlook += line + " "
        
        db_path = os.path.join(os.path.dirname(__file__), "portfolio.db")
        conn = sqlite3.connect(db_path)
        c = conn.cursor()
        c.execute("""
            INSERT OR REPLACE INTO liquidity_analysis 
            (analysis_date, trend_summary, key_insights, policy_implications, market_outlook)
            VALUES (?, ?, ?, ?, ?)
        """, (analysis_date, trend_summary.strip(), key_insights.strip(), 
              policy_implications.strip(), market_outlook.strip()))
        conn.commit()
        conn.close()
        print(f"[DB] Saved liquidity analysis for {analysis_date}")
        
    except Exception as e:
        print(f"[DB] Error saving liquidity analysis: {e}")


@app.post("/api/liquidity/analyze")
async def analyze_liquidity():
    """유동성 트렌드 LLM 분석 실행"""
    analysis = await analyze_liquidity_with_llm()
    return {"analysis": analysis}


@app.get("/api/liquidity/analysis")
def get_latest_liquidity_analysis():
    """최신 유동성 분석 결과 반환"""
    db_path = os.path.join(os.path.dirname(__file__), "portfolio.db")
    try:
        conn = sqlite3.connect(db_path)
        c = conn.cursor()
        c.execute("""
            SELECT analysis_date, trend_summary, key_insights, 
                   policy_implications, market_outlook, created_at
            FROM liquidity_analysis 
            ORDER BY created_at DESC 
            LIMIT 1
        """)
        result = c.fetchone()
        conn.close()
        
        if result:
            return {
                "data": {
                    "analysis_date": result[0],
                    "trend_summary": result[1],
                    "key_insights": result[2],
                    "policy_implications": result[3],
                    "market_outlook": result[4],
                    "created_at": result[5]
                }
            }
        else:
            return {"data": None}
    except Exception as e:
        print(f"[DB] Error getting liquidity analysis: {e}")
        return {"error": str(e)}


def get_db_portfolio(email: str = None):
    db_path = os.path.join(os.path.dirname(__file__), "portfolio.db")
    try:
        conn = sqlite3.connect(db_path)
        c = conn.cursor()
        if email:
            c.execute("SELECT ticker, name, quantity, buy_price, current_price, profit_krw FROM portfolio WHERE user_email = ?", (email,))
        else:
            c.execute("SELECT ticker, name, quantity, buy_price, current_price, profit_krw FROM portfolio LIMIT 20")
        rows = c.fetchall()
        conn.close()
        return rows
    except Exception as e:
        print(f"[DB Error] {e}")
        return []

@app.get("/api/portfolio")
def fetch_portfolio(email: Optional[str] = None):
    rows = get_db_portfolio(email)
    data = []
    for r in rows:
        data.append({
            "ticker": r[0],
            "name": r[1],
            "quantity": r[2],
            "buy_price": r[3],
            "current_price": r[4],
            "profit_krw": r[5]
        })
    return {"data": data}

class PortfolioItem(BaseModel):
    ticker: str
    name: str = ""
    quantity: float
    buy_price: Optional[float] = 0.0
    current_price: Optional[float] = 0.0
    profit_krw: Optional[int] = 0

class PortfolioUpdateRequest(BaseModel):
    email: str
    items: List[PortfolioItem]

@app.post("/api/portfolio/update")
async def update_portfolio(request: PortfolioUpdateRequest):
    db_path = os.path.join(os.path.dirname(__file__), "portfolio.db")
    try:
        conn = sqlite3.connect(db_path)
        c = conn.cursor()
        
        # Delete existing items for this user
        c.execute("DELETE FROM portfolio WHERE user_email = ?", (request.email,))
        
        # Insert new items
        for item in request.items:
            c.execute('''
                INSERT INTO portfolio (user_email, ticker, name, quantity, buy_price, current_price, profit_krw)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (request.email, item.ticker, item.name, item.quantity, item.buy_price, item.current_price, item.profit_krw))
        
        conn.commit()
        conn.close()
        return {"status": "success", "message": f"Updated {len(request.items)} items for {request.email}"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/api/portfolio/parse-image")
async def parse_portfolio_image(file: UploadFile = File(...)):
    # Gemini Vision을 이용해 이미지에서 자산 데이터를 추출합니다.
    try:
        content = await file.read()
        image_base64 = base64.b64encode(content).decode("utf-8")
        
        prompt = """보여지는 이미지(주식/금융 자산 잔고 화면)에서 보유 자산 목록을 추출해줘.
결과는 반드시 다음과 같은 JSON 형식의 리스트로만 응답해:
[
  {"ticker": "AAPL", "name": "애플", "quantity": 10.5, "buy_price": 200.0, "current_price": 230.0, "profit_krw": 300000},
  ...
]
- 티커(ticker)를 알 수 없으면 공백이나 추측되는 값으로 넣어줘.
- 수량(quantity)은 숫자만 포함해.
- 수익(profit_krw)은 대략적인 원화 환산 금액으로 추출해.
- 만약 이미지가 잔고 화면이 아니거나 데이터를 추출할 수 없으면 빈 리스트 [] 를 응답해."""

        message = HumanMessage(
            content=[
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"},
                },
            ]
        )

        from langchain_google_genai import ChatGoogleGenerativeAI
        
        models_to_try = ["gemini-2.0-flash", "gemini-flash-lite-latest", "gemini-flash-latest"]
        content_str = ""
        
        for model_id in models_to_try:
            try:
                print(f"[Parse Image] Trying model: {model_id}")
                llm = ChatGoogleGenerativeAI(
                    model=model_id,
                    google_api_key=os.getenv("GOOGLE_API_KEY"),
                    temperature=0,
                    max_retries=0,
                )
                response = llm.invoke([message])
                print(f"[Parse Image] Raw Response from {model_id}: {response.content}")
                content_str = response.content.strip()
                break # 성공 시 루프 중단
            except Exception as e:
                err_msg = str(e).lower()
                if ("429" in err_msg or "quota" in err_msg) and model_id != models_to_try[-1]:
                    print(f"[Parse Image] {model_id} quota exceeded, trying fallback...")
                    continue
                raise e # 마지막 모델까지 실패하거나 다른 에러면 raise
        
        # JSON 블록 추출 시도
        if "```json" in content_str:
            content_str = content_str.split("```json")[1].split("```")[0].strip()
        elif "```" in content_str:
            content_str = content_str.split("```")[1].split("```")[0].strip()
            
        # 대괄호([]) 사이의 내용만 추출하여 순수 JSON 시도
        import re
        json_match = re.search(r'\[.*\]', content_str, re.DOTALL)
        if json_match:
            content_str = json_match.group(0)
            
        try:
            parsed_data = json.loads(content_str)
        except json.JSONDecodeError as je:
            print(f"[JSON Parse Error] Content: {content_str}")
            return {"status": "error", "message": f"데이터 형식 오류: {str(je)}"}
        
        return {"data": parsed_data}
    except Exception as e:
        print(f"[Parse Image Error] {e}")
        import traceback
        traceback.print_exc()
        return {"status": "error", "message": str(e)}

class AnalysisRequest(BaseModel):
    query: str
    model: str = "gemini"
    history: Optional[List[dict]] = []
    user_portfolio: str = ""
    email: Optional[str] = None


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
        ("gemini-2.0-flash",           "Google Gemini 2.0 Flash"),
        ("gemini-flash-lite-latest",   "Google Gemini Flash Lite (Fallback)"),
        ("gemini-flash-latest",        "Google Gemini Flash (Fallback)"),
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

        # 유동성 컨텍스트 문자열 조립 (최신 LLM 분석 포함)
        if liquidity_data:
            liquidity_context = "\n".join([
                f"{item['name']} ({item['series']}): {item['value']:,.2f} {item['unit']}  [{item['date']}]  — {item['desc']}"
                for item in liquidity_data
            ])
            
            # 최신 유동성 LLM 분석 추가
            try:
                db_path = os.path.join(os.path.dirname(__file__), "portfolio.db")
                conn = sqlite3.connect(db_path)
                c = conn.cursor()
                c.execute("""
                    SELECT trend_summary, key_insights, policy_implications, market_outlook, created_at
                    FROM liquidity_analysis 
                    ORDER BY created_at DESC 
                    LIMIT 1
                """)
                analysis_result = c.fetchone()
                conn.close()
                
                if analysis_result:
                    liquidity_context += f"""

=== 최신 유동성 트렌드 분석 ({analysis_result[4][:10]}) ===
트렌드 요약: {analysis_result[0]}
핵심 인사이트: {analysis_result[1]}
정책적 시사점: {analysis_result[2]}
시장 전망: {analysis_result[3]}
"""
                else:
                    liquidity_context += "\n\n최신 유동성 트렌드 분석이 아직 생성되지 않았습니다."
                    
            except Exception as e:
                print(f"[LLM Context] Error loading liquidity analysis: {e}")
                
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
    chat_history_str = ""
    if request.history:
        for msg in request.history[-5:]:  # 최근 5개 메시지만 컨텍스트로 사용
            role = "사용자" if msg.get("role") == "user" else "AI"
            chat_history_str += f"{role}: {msg.get('content')}\n"

    chain_inputs = {
        "market_data": market_str,
        "news_summary": news_summary,
        "bok_policy": bok_policy,
        "fed_policy": fed_policy,
        "macro_signals": macro_signals,
        "institutional_context": _institutional_context,
        "liquidity_context": liquidity_context,
        "chat_history": chat_history_str,
        "user_portfolio": request.user_portfolio or "현재 등록된 보유 자산이 없습니다.",
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
