from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import yfinance as yf
import os
from datetime import datetime
import time
import pytz
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.chains import LLMChain
from prompts import MACRO_ANALYSIS_PROMPT
import json

# Load environment variables
load_dotenv()

app = FastAPI()

# ── In-memory cache for market data (60s TTL) ──
_market_cache: dict = {"data": None, "signals": None, "fetched_at": 0}  # reset on deploy
CACHE_TTL = 60  # seconds

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
        "^IRX": "US 13W Bond", # Proxy for short term rate
        "CL=F": "WTI Oil",
        "GC=F": "Gold",
        "BTC-USD": "Bitcoin",
        "ETH-USD": "Ethereum",
        "005930.KS": "Samsung Elec",
        "DX-Y.NYB": "Dollar Index"
    }
    
    data = []
    signals = []
    
    for symbol, name in symbols.items():
        try:
            ticker = yf.Ticker(symbol)
            # Fetch slightly more history for momentum calculation
            hist = ticker.history(period="1mo")
            
            if not hist.empty:
                current = hist['Close'].iloc[-1]
                prev = hist['Close'].iloc[-2] if len(hist) > 1 else current
                change = ((current - prev) / prev) * 100
                
                # Calculate Momentum (5-day & 20-day)
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
                    "momentum": momentum_signal
                })
        except Exception as e:
            print(f"Error fetching {symbol}: {e}")

    # Calculate Spreads (Yield Curve Proxy) if available
    try:
        us10y = next((x for x in data if x['symbol'] == '^TNX'), None)
        us13w = next((x for x in data if x['symbol'] == '^IRX'), None)
        
        if us10y and us13w:
            spread = us10y['price'] - us13w['price']
            signals.append(f"Yield Curve (10Y-13W): {spread:.2f}bp ({'Inverted!' if spread < 0 else 'Normal'})")
    except Exception:
        pass

    return {"data": data, "signals": signals}

@app.get("/api/market-data")
def get_market_data():
    global _market_cache
    now = time.time()
    # Return cached data if still fresh
    if _market_cache["data"] is not None and (now - _market_cache["fetched_at"]) < CACHE_TTL:
        print(f"[Cache HIT] age={(now - _market_cache['fetched_at']):.1f}s")
        return {"data": _market_cache["data"]}
    # Fetch fresh data
    print("[Cache MISS] Fetching fresh market data...")
    result = fetch_market_data_internal()
    _market_cache["data"] = result["data"]
    _market_cache["signals"] = result["signals"]
    _market_cache["fetched_at"] = now
    return {"data": result["data"]}

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

    from fastapi.responses import StreamingResponse as SR

    if "gemini" in requested_model:
        if not api_key_google:
            async def _err():
                yield "> \u26a0\ufe0f **Google API \ud0a4\uac00 \uc5c6\uc2b5\ub2c8\ub2e4.** `.env`\uc5d0 `GOOGLE_API_KEY`\ub97c \ucd94\uac00\ud574\uc8fc\uc138\uc694."
            return SR(_err(), media_type="text/plain")
        from langchain_google_genai import ChatGoogleGenerativeAI
        llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=api_key_google, temperature=0.7)
        model_source = "Google Gemini 2.0 Flash"

    elif "gpt" in requested_model or "openai" in requested_model:
        if not api_key_openai:
            async def _err():
                yield "> \u26a0\ufe0f **OpenAI API \ud0a4\uac00 \uc5c6\uc2b5\ub2c8\ub2e4.** `.env`\uc5d0 `OPENAI_API_KEY`\ub97c \ucd94\uac00\ud574\uc8fc\uc138\uc694."
            return SR(_err(), media_type="text/plain")
        llm = ChatOpenAI(temperature=0.7, model_name="gpt-4o", openai_api_key=api_key_openai)
        model_source = "OpenAI GPT-4o"

    elif "claude" in requested_model:
        if not api_key_anthropic:
            async def _err():
                yield "> \u26a0\ufe0f **Anthropic API \ud0a4\uac00 \uc5c6\uc2b5\ub2c8\ub2e4.** `.env`\uc5d0 `ANTHROPIC_API_KEY`\ub97c \ucd94\uac00\ud574\uc8fc\uc138\uc694."
            return SR(_err(), media_type="text/plain")
        from langchain_anthropic import ChatAnthropic
        llm = ChatAnthropic(model="claude-3-5-sonnet-20241022", anthropic_api_key=api_key_anthropic, temperature=0.7, max_tokens=8096)
        model_source = "Anthropic Claude 3.5 Sonnet"

    if not llm:
        async def _err():
            yield f"> \u26a0\ufe0f **\uc54c \uc218 \uc5c6\ub294 \ubaa8\ub378**: `{request.model}`. gemini / gpt / claude \uc911 \ud558\ub098\ub97c \uc120\ud0dd\ud574\uc8fc\uc138\uc694."
        return SR(_err(), media_type="text/plain")

    # Prepare data (Synchronous part)
    try:
        # 1. Fetch Market Data Context
        print("Fetching market data...")
        market_result = fetch_market_data_internal()
        market_data = market_result["data"]
        macro_signals = "\n".join(market_result["signals"])
        
        market_str = "\n".join([f"{item['name']} ({item['symbol']}): {item['price']} ({item['change_percent']}%) [Momentum: {item.get('momentum', 'N/A')}]" for item in market_data])
        print(f"Market data fetched: {len(market_data)} items")
        
        # 2. Fetch News & Policy Info via Search
        print("Starting DuckDuckGo search...")
        search = DuckDuckGoSearchRun()
        user_query = request.query
        try:
            # Use user's query to make search more relevant
            news_summary = search.run(f"{user_query} 최신 글로벌 금융 뉴스 경제 분석 2025")
            print("News search done.")
            bok_policy = search.run("한국은행 기준금리 결정 최신 통화정책 2025")
            print("BOK policy search done.")
            fed_policy = search.run("Federal Reserve FOMC meeting outcome interest rate decision 2025")
            print("Fed policy search done.")
            
            # Additional Macro Indicators Search
            macro_signals += "\n" + search.run("US CPI inflation PCE core 2025 latest")
            macro_signals += "\n" + search.run("한국 반도체 수출 동향 2025")
        except Exception as search_e:
            print(f"Search warning: {search_e}")
            news_summary = f"News search failed: {str(search_e)}"
            bok_policy = "Data unavailable."
            fed_policy = "Data unavailable."

    except Exception as e:
        import traceback
        traceback.print_exc()
        from fastapi.responses import JSONResponse
        return JSONResponse(content={
            "analysis": f"Error during data fetch: {str(e)}",
            "sentiment": "Error",
            "timestamp": datetime.now(pytz.utc).isoformat()
        })

    # 3. Stream LLM Analysis
    from fastapi.responses import StreamingResponse
    from langchain_core.output_parsers import StrOutputParser

    print(f"Starting LLM analysis streaming with {model_source}...")
    
    # LCEL Chain
    chain = MACRO_ANALYSIS_PROMPT | llm | StrOutputParser()

    async def generate():
        yield f"**[{model_source}]**\n\n"
        
        # Load Institutional Reports (RAG Context)
        institutional_context = "No specific reports available."
        try:
            report_path = os.path.join(os.path.dirname(__file__), "data", "institutional_reports.json")
            if os.path.exists(report_path):
                with open(report_path, "r", encoding="utf-8") as f:
                    report_data = json.load(f)
                    institutional_context = "\n\n".join([
                        f"[{r['institution']} - {r['title']}]\nSummary: {r['summary']}\nDetails: {r['content']}"
                        for r in report_data.get("reports", [])
                    ])
        except Exception as rag_e:
            print(f"RAG Error: {rag_e}")

        try:
            async for chunk in chain.astream({
                "market_data": market_str,
                "news_summary": news_summary,
                "bok_policy": bok_policy,
                "fed_policy": fed_policy,
                "macro_signals": macro_signals,
                "institutional_context": institutional_context,
                "user_query": user_query
            }):
                yield chunk
        except Exception as e:
            err_msg = str(e)
            print(f"[LLM Stream Error] {err_msg}")
            if "429" in err_msg or "quota" in err_msg.lower() or "ResourceExhausted" in err_msg:
                yield f"\n\n> ⚠️ **API 쿼터 초과**: {model_source} 의 무료 한도를 초과했습니다. 다른 모델을 선택하거나 잠시 후 다시 시도해주세요."
            elif "401" in err_msg or "authentication" in err_msg.lower() or "api_key" in err_msg.lower():
                yield f"\n\n> ⚠️ **인증 오류**: API 키가 유효하지 않습니다. .env 파일을 확인해주세요."
            else:
                yield f"\n\n> ⚠️ **분석 중 오류 발생**: {err_msg[:200]}"

    return StreamingResponse(generate(), media_type="text/plain")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
