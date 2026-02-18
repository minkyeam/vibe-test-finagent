# Financial Macro Research Agent (PoC) for Korean Investors

## Project Objective
Develop an AI Agent that analyzes global and Korean macroeconomic conditions to support investment decisions for Korean investors.

## Key Features
1.  **Macro Dashboard:**
    *   Real-time key indicators: USD/KRW, US 10Y Treasury, KOSPI, Oil (WTI).
    *   Comparison of KR vs US interest rates.
2.  **AI Research Agent:**
    *   Analyzes recent news and economic calendar events.
    *   Generates a "Market Climate" report (e.g., "Risk-On", "Risk-Off", "Neutral").
    *   Provides reasoning based on inflation, growth, and liquidity.
3.  **Korean Context Focus:**
    *   Impact of USD/KRW on Korean exporters/importers.
    *   Bank of Korea (BOK) policy analysis.

## Proposed Tech Stack
-   **Frontend:** Next.js (React) + TailwindCSS (Modern, responsive UI).
-   **Backend:** Python (FastAPI)
    -   **Data:** `yfinance` (Market Data).
    -   **Search:** `duckduckgo-search` (News).
    -   **LLM Integration:** OpenAI API / LangChain (for reasoning).

## Directory Structure
-   `/frontend`: Web application.
-   `/backend`: API server and Agent logic.
