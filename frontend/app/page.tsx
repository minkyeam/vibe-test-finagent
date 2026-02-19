'use client';

import { useState, useEffect, useRef, useCallback } from "react";
import ReactMarkdown from 'react-markdown';

interface MarketData {
  symbol: string;
  name: string;
  price: number;
  change_percent: number;
}

interface LiquidityItem {
  series: string;
  name: string;
  value: number;
  unit: string;
  date: string;
  desc: string;
}

interface AnalysisRecord {
  id: string;
  query: string;
  model: string;
  content: string;
  timestamp: string;
  feedback?: 'helpful' | 'not_helpful' | null;
  saved: boolean;
}

type AIModel = 'gemini' | 'claude' | 'gpt';

const SUGGESTED_QUERIES = [
  "현재 글로벌 매크로 환경에서 한국 투자자의 대응 전략은?",
  "미 연준의 금리 정책이 원/달러 환율에 미치는 영향 분석",
  "반도체 섹터 투자 전망과 리스크 요인",
  "한국은행 통화정책과 KOSPI 상관관계",
  "글로벌 인플레이션 추세와 자산 배분 전략",
];

const MODELS = [
  { id: 'gemini' as AIModel, name: 'Gemini', fullName: 'Google Gemini', active: true, desc: 'v1.5 Flash' },
  { id: 'gpt' as AIModel, name: 'GPT-4o', fullName: 'OpenAI ChatGPT', active: true, desc: 'GPT-4o' },
  { id: 'claude' as AIModel, name: 'Claude', fullName: 'Anthropic Claude', active: true, desc: '3.5 Sonnet' },
];

interface ToastItem { id: string; message: string; type: 'warning' | 'info'; }
interface IssueLogItem { id: string; timestamp: string; message: string; type: 'warning' | 'info'; }

export default function Home() {
  const [data, setData] = useState<MarketData[]>([]);
  const [liquidityData, setLiquidityData] = useState<LiquidityItem[]>([]);
  const [analysis, setAnalysis] = useState<string | null>(null);
  const [isExpanded, setIsExpanded] = useState(false);
  const [analyzing, setAnalyzing] = useState(false);
  const [selectedModel, setSelectedModel] = useState<AIModel>('gemini');
  const [showModelDropdown, setShowModelDropdown] = useState(false);
  const [query, setQuery] = useState('');
  const [showSuggestions, setShowSuggestions] = useState(false);
  const [history, setHistory] = useState<AnalysisRecord[]>([]);
  const [activeHistoryId, setActiveHistoryId] = useState<string | null>(null);
  const [currentQuery, setCurrentQuery] = useState('');
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [headerScrolled, setHeaderScrolled] = useState(false);
  const [toasts, setToasts] = useState<ToastItem[]>([]);
  const [issueLog, setIssueLog] = useState<IssueLogItem[]>([]);
  const mainRef = useRef<HTMLElement>(null);

  const abortControllerRef = useRef<AbortController | null>(null);
  const modelDropdownRef = useRef<HTMLDivElement>(null);
  const queryRef = useRef<HTMLTextAreaElement>(null);
  const [dataFetchedAt, setDataFetchedAt] = useState<string>('');
  const [refreshCountdown, setRefreshCountdown] = useState<number>(30);
  const toastIdCounter = useRef(0);

  const pushToast = useCallback((message: string, type: 'warning' | 'info' = 'warning') => {
    const id = `toast-${Date.now()}-${++toastIdCounter.current}`;
    const timestamp = new Date().toLocaleTimeString('ko-KR', { hour: '2-digit', minute: '2-digit', second: '2-digit' });
    setToasts(prev => [...prev, { id, message, type }]);
    setIssueLog(prev => [{ id, timestamp, message, type }, ...prev]);
    setTimeout(() => setToasts(prev => prev.filter(t => t.id !== id)), 5000);
  }, []);

  const fetchMarketData = useCallback(() => {
    fetch('/api/market-data')
      .then((res) => res.json())
      .then((json) => {
        if (json.data) setData(json.data);
        const now = new Date();
        const pad = (n: number) => String(n).padStart(2, '0');
        setDataFetchedAt(
          `${now.getFullYear()}.${pad(now.getMonth() + 1)}.${pad(now.getDate())} ${pad(now.getHours())}:${pad(now.getMinutes())}`
        );
        setRefreshCountdown(30);
      })
      .catch(console.error);
  }, []);

  const fetchLiquidityData = useCallback(() => {
    fetch('/api/liquidity')
      .then((res) => res.json())
      .then((json) => {
        if (json.data) setLiquidityData(json.data);
      })
      .catch(console.error);
  }, []);

  useEffect(() => {
    fetchMarketData();
    fetchLiquidityData();
    const pollInterval = setInterval(fetchMarketData, 30_000);
    const liquidityPollInterval = setInterval(fetchLiquidityData, 60 * 60_000);
    const countdownInterval = setInterval(() => {
      setRefreshCountdown((prev) => (prev <= 1 ? 30 : prev - 1));
    }, 1_000);

    const handleClickOutside = (event: MouseEvent) => {
      if (modelDropdownRef.current && !modelDropdownRef.current.contains(event.target as Node)) {
        setShowModelDropdown(false);
      }
    };
    document.addEventListener("mousedown", handleClickOutside);

    const saved = localStorage.getItem('analysis_history');
    if (saved) setHistory(JSON.parse(saved));

    return () => {
      clearInterval(pollInterval);
      clearInterval(liquidityPollInterval);
      clearInterval(countdownInterval);
      document.removeEventListener("mousedown", handleClickOutside);
    };
  }, [fetchMarketData, fetchLiquidityData]);

  // Scroll listener — 별도 effect로 분리해 mainRef 마운트 타이밍 보장
  useEffect(() => {
    const mainEl = mainRef.current;
    if (!mainEl) return;
    const handleScroll = () => setHeaderScrolled(mainEl.scrollTop > 60);
    mainEl.addEventListener('scroll', handleScroll);
    return () => mainEl.removeEventListener('scroll', handleScroll);
  });

  const saveHistory = (records: AnalysisRecord[]) => {
    setHistory(records);
    localStorage.setItem('analysis_history', JSON.stringify(records.slice(0, 20)));
  };

  const handleStop = useCallback(() => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      abortControllerRef.current = null;
      setAnalyzing(false);
    }
  }, []);

  // 오건영 이름 필터 (클린 출력)
  const cleanOutput = useCallback((raw: string): string => {
    return raw.replace(/오건영(의|이|은|에서|에게|으로|로|가|이다|입니다)?/g, '');
  }, []);

  const handleAnalyze = useCallback(async () => {
    if (!query.trim() || analyzing) return;
    setAnalyzing(true);
    setAnalysis('');
    const userQuery = query.trim();
    setCurrentQuery(userQuery);

    const controller = new AbortController();
    abortControllerRef.current = controller;

    try {
      const response = await fetch('/api/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: userQuery, model: selectedModel }),
        signal: controller.signal,
      });

      if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);

      const reader = response.body?.getReader();
      const decoder = new TextDecoder();
      let fullContent = '';
      const toastedSet = new Set<string>(); // 중복 토스트 방지

      if (reader) {
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          fullContent += decoder.decode(value, { stream: true });

          // ⚠️/⚡ 이슈 라인 → 토스트 발사
          fullContent.split('\n').forEach(line => {
            const t = line.trim();
            if (t.match(/^>\s*(\u26a0\ufe0f|\u26a1)/) && !toastedSet.has(t)) {
              toastedSet.add(t);
              const msg = t.replace(/^>\s*/, '');
              pushToast(msg, msg.startsWith('\u26a0\ufe0f') ? 'warning' : 'info');
            }
          });

          // 이슈 라인 → 빈줄 치환 후 이름 필터링해서 표시
          const display = cleanOutput(
            fullContent
              .split('\n')
              .map(line => (line.trim().match(/^>\s*(\u26a0\ufe0f|\u26a1)/) ? '' : line))
              .join('\n')
          );
          setAnalysis(display);
        }
      }

      // 히스토리 저장용 최종 정제
      const finalContent = cleanOutput(
        fullContent
          .split('\n')
          .map(line => (line.trim().match(/^>\s*(\u26a0\ufe0f|\u26a1)/) ? '' : line))
          .join('\n')
          .trim()
      );

      if (finalContent) {
        const newRecord: AnalysisRecord = {
          id: Date.now().toString(),
          query: userQuery,
          model: selectedModel,
          content: finalContent,
          timestamp: new Date().toISOString(),
          feedback: null,
          saved: false,
        };
        saveHistory([newRecord, ...history]);
        setActiveHistoryId(newRecord.id);
      }
    } catch (err: unknown) {
      if (err instanceof Error && err.name !== 'AbortError') {
        setAnalysis('분석 중 오류가 발생했습니다. 다시 시도해주세요.');
      }
    } finally {
      setAnalyzing(false);
      abortControllerRef.current = null;
    }
  }, [query, analyzing, selectedModel, history, pushToast, cleanOutput]);


  const handleFeedback = (type: 'helpful' | 'not_helpful') => {
    if (!activeHistoryId) return;
    saveHistory(history.map(r => r.id === activeHistoryId ? { ...r, feedback: type } : r));
  };

  const handleSave = () => {
    if (!activeHistoryId) return;
    saveHistory(history.map(r => r.id === activeHistoryId ? { ...r, saved: !r.saved } : r));
  };

  const loadHistoryItem = (record: AnalysisRecord) => {
    setAnalysis(record.content);
    setCurrentQuery(record.query);
    setActiveHistoryId(record.id);
    setQuery(record.query);
  };

  const currentRecord = history.find(r => r.id === activeHistoryId);
  const selectedModelInfo = MODELS.find(m => m.id === selectedModel) ?? MODELS[0];
  const sidebarWidth = sidebarOpen ? 256 : 56;

  return (
    <div className="h-screen bg-white text-zinc-900 font-sans selection:bg-emerald-100 selection:text-emerald-900 tracking-tight flex flex-col overflow-hidden">
      <style>{`
        @keyframes marquee { 0% { transform: translateX(0); } 100% { transform: translateX(-50%); } }
        .animate-marquee { display: inline-flex; animation: marquee 40s linear infinite; }
        @keyframes blink { 0%, 100% { opacity: 1; } 50% { opacity: 0; } }
        .cursor-blink { display: inline-block; width: 2px; height: 1.1em; background-color: #10b981; margin-left: 2px; vertical-align: middle; animation: blink 1s step-end infinite; }
        .custom-scrollbar::-webkit-scrollbar { width: 4px; }
        .custom-scrollbar::-webkit-scrollbar-track { background: transparent; }
        .custom-scrollbar::-webkit-scrollbar-thumb { background: #f4f4f5; border-radius: 10px; }
        .custom-scrollbar::-webkit-scrollbar-thumb:hover { background: #e4e4e7; }
        textarea:focus { outline: none; }
        @keyframes fadeIn { from { opacity: 0; transform: translateY(6px); } to { opacity: 1; transform: translateY(0); } }
        .fade-in { animation: fadeIn 0.2s ease forwards; }
        @keyframes spin { to { transform: rotate(360deg); } }
        .spin { animation: spin 1s linear infinite; }
        @keyframes slideIn { from { opacity: 0; transform: translateX(-8px); } to { opacity: 1; transform: translateX(0); } }
        .slide-in { animation: slideIn 0.2s ease forwards; }
        @keyframes toastSlideIn { from { opacity: 0; transform: translateX(24px); } to { opacity: 1; transform: translateX(0); } }
        @keyframes toastSlideOut { from { opacity: 1; transform: translateX(0); } to { opacity: 0; transform: translateX(24px); } }
        .toast-in { animation: toastSlideIn 0.3s ease forwards; }
      `}</style>

      {/* ── ① Sticky Ticker Bar (전폭, 최상단 고정) ── */}
      <section className="sticky top-0 z-[100] w-full border-b border-zinc-100 bg-white/95 backdrop-blur-xl shrink-0">
        <div className="flex items-center justify-between h-12 px-4">
          <div className="flex items-center gap-3 shrink-0">
            <div className="w-6 h-6 rounded-lg bg-zinc-950 flex items-center justify-center">
              <span className="text-white text-[10px] font-black">N</span>
            </div>
            <span className="text-[10px] font-bold text-zinc-400 uppercase tracking-[0.25em] hidden sm:block">Market</span>
            <span className="w-px h-4 bg-zinc-100 hidden sm:block"></span>
            <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse"></span>
          </div>

          <div className="flex-1 overflow-hidden relative mx-4">
            <div className="animate-marquee whitespace-nowrap flex items-center gap-12 py-1">
              {[...data, ...data].map((item, idx) => (
                <div key={`${item.symbol}-${idx}`} className="flex items-center gap-3 group cursor-default">
                  <span className="font-semibold text-zinc-800 text-xs group-hover:text-emerald-600 transition-colors">{item.name}</span>
                  <span className="text-zinc-400 font-mono text-xs">{item.price.toLocaleString()}</span>
                  <span className={`text-[11px] font-bold ${item.change_percent >= 0 ? 'text-emerald-600' : 'text-rose-500'}`}>
                    {item.change_percent >= 0 ? '▲' : '▼'} {Math.abs(item.change_percent).toFixed(2)}%
                  </span>
                </div>
              ))}
            </div>
          </div>

          <div className="flex items-center gap-2 shrink-0">
            <div className="hidden sm:flex items-center gap-1.5 text-[10px] font-mono text-zinc-300">
              <span className="tabular-nums">{refreshCountdown}s</span>
              <button onClick={fetchMarketData} title="지금 새로고침"
                className="w-6 h-6 flex items-center justify-center rounded-lg hover:bg-zinc-100 transition-colors text-zinc-300 hover:text-zinc-600">
                <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
                </svg>
              </button>
            </div>
            <div className="w-px h-4 bg-zinc-100 hidden sm:block"></div>
            <button onClick={() => setIsExpanded(!isExpanded)}
              className={`w-8 h-8 flex items-center justify-center rounded-lg transition-all border ${isExpanded ? 'bg-zinc-900 border-zinc-900 text-white' : 'hover:bg-zinc-50 border-transparent text-zinc-400'}`}>
              <svg className={`w-4 h-4 transition-transform duration-500 ${isExpanded ? 'rotate-180' : ''}`} fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M19 9l-7 7-7-7" />
              </svg>
            </button>
          </div>
        </div>
      </section>

      {/* ── ② Body: GNB(좌, 전체 높이) + 우측 컬럼(마켓패널 + 메인) ── */}
      <div className="flex flex-1 min-h-0">

        {/* ── Left GNB — 전체 높이 차지, self-scroll ── */}
        <aside
          className={`flex flex-col shrink-0 border-r border-zinc-100 bg-white transition-all duration-300 overflow-y-auto overflow-x-hidden ${sidebarOpen ? 'w-64' : 'w-14'}`}
        >
          {/* Sidebar toggle */}
          <div className="flex items-center justify-between px-3 py-3 border-b border-zinc-50 shrink-0">
            {sidebarOpen && (
              <span className="text-[10px] font-bold text-zinc-400 uppercase tracking-[0.25em] pl-1">History</span>
            )}
            <button onClick={() => setSidebarOpen(!sidebarOpen)}
              className="w-8 h-8 flex items-center justify-center rounded-lg hover:bg-zinc-100 transition-colors text-zinc-400 ml-auto">
              <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                {sidebarOpen
                  ? <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M11 19l-7-7 7-7M18 19l-7-7 7-7" />
                  : <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 5l7 7-7 7M6 5l7 7-7 7" />
                }
              </svg>
            </button>
          </div>

          {/* New analysis button */}
          <div className="px-2 py-2 border-b border-zinc-50 shrink-0">
            <button
              onClick={() => { setAnalysis(null); setCurrentQuery(''); setActiveHistoryId(null); setQuery(''); }}
              className={`w-full h-9 flex items-center gap-2.5 px-3 rounded-xl transition-all text-xs font-medium text-zinc-500 hover:bg-zinc-50 hover:text-zinc-800 ${!sidebarOpen ? 'justify-center' : ''}`}
              title="새 분석"
            >
              <svg className="w-4 h-4 shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
              </svg>
              {sidebarOpen && <span>새 분석</span>}
            </button>
          </div>

          {/* History list */}
          <div className="flex-1 overflow-y-auto custom-scrollbar py-2">
            {!sidebarOpen ? (
              <div className="flex flex-col items-center gap-1.5 px-2 pt-1">
                {history.slice(0, 10).map((record) => (
                  <button key={record.id} onClick={() => loadHistoryItem(record)}
                    title={record.query}
                    className={`w-8 h-8 rounded-lg flex items-center justify-center transition-all text-[10px] font-bold ${activeHistoryId === record.id ? 'bg-zinc-900 text-white' : 'hover:bg-zinc-100 text-zinc-400'}`}>
                    {record.model === 'gemini' ? 'G' : record.model === 'gpt' ? 'O' : 'C'}
                  </button>
                ))}
              </div>
            ) : (
              history.length === 0 ? (
                <div className="px-4 py-8 text-center">
                  <p className="text-xs text-zinc-300">분석 기록이 없습니다</p>
                </div>
              ) : (
                <div className="px-2">
                  {history.map((record) => (
                    <button key={record.id} onClick={() => loadHistoryItem(record)}
                      className={`w-full px-3 py-2.5 rounded-xl text-left transition-all mb-0.5 group ${activeHistoryId === record.id ? 'bg-zinc-100' : 'hover:bg-zinc-50'}`}>
                      <div className="flex items-start gap-2">
                        <span className={`w-1.5 h-1.5 rounded-full mt-1.5 shrink-0 ${record.model === 'gemini' ? 'bg-blue-400' : record.model === 'gpt' ? 'bg-emerald-400' : 'bg-purple-400'}`}></span>
                        <div className="flex-1 min-w-0">
                          <p className="text-xs font-medium text-zinc-700 line-clamp-2 leading-snug">{record.query}</p>
                          <div className="flex items-center gap-1.5 mt-1">
                            <span className="text-[10px] text-zinc-400 uppercase">{record.model}</span>
                            <span className="text-[10px] text-zinc-300">·</span>
                            <span className="text-[10px] text-zinc-400">
                              {new Date(record.timestamp).toLocaleDateString('ko-KR', { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' })}
                            </span>
                            {record.saved && <span className="text-amber-400 text-[10px]">★</span>}
                          </div>
                        </div>
                      </div>
                    </button>
                  ))}
                  {history.length > 0 && (
                    <button onClick={() => { saveHistory([]); setActiveHistoryId(null); }}
                      className="w-full mt-2 px-3 py-2 text-[10px] text-zinc-300 hover:text-rose-400 transition-colors text-center rounded-xl hover:bg-rose-50">
                      전체 삭제
                    </button>
                  )}
                </div>
              )
            )}
          </div>

          {/* Sidebar Footer */}
          {sidebarOpen && (
            <div className="shrink-0 px-4 py-4 border-t border-zinc-50">
              <p className="text-[9px] font-medium text-zinc-300 uppercase tracking-[0.12em] leading-relaxed">
                © 2026 N Finance Agent · MVP Phase 1
              </p>
              <div className="flex items-center gap-1.5 mt-1">
                <span className="text-[9px] text-zinc-200 uppercase tracking-widest">RAG</span>
                <span className="text-[9px] text-zinc-200">·</span>
                <span className="text-[9px] text-zinc-200 uppercase tracking-widest">Human-in-the-Loop</span>
                <span className="text-[9px] text-zinc-200">·</span>
                <span className="text-[9px] text-zinc-200 uppercase tracking-widest">Grounded</span>
              </div>
            </div>
          )}
        </aside>

        {/* ── 우측 컬럼: 마켓패널(접힘/펼침) + 메인 컨텐츠 ── */}
        <div className="flex flex-col flex-1 min-h-0">

          {/* ── 마켓 패널 — GNB 오른쪽만 차지, 동적 너비 ── */}
          <div className={`overflow-hidden transition-all duration-500 border-b border-zinc-100 bg-white shadow-sm shrink-0 ${isExpanded ? 'max-h-[58vh] opacity-100' : 'max-h-0 opacity-0 pointer-events-none'}`}>
            <div className="px-5 py-4 overflow-y-auto max-h-[56vh] custom-scrollbar">

              {/* 헤더 */}
              <div className="flex items-center justify-between mb-4 pb-3 border-b border-zinc-50">
                <div className="flex items-center gap-3">
                  <p className="text-[10px] font-bold text-zinc-400 uppercase tracking-[0.4em]">Global / {data.length} Assets</p>
                  <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse"></span>
                  <span className="text-[10px] text-zinc-300 font-mono tabular-nums">{dataFetchedAt || '—'}</span>
                </div>
                <span className="text-[10px] text-zinc-300 font-mono tabular-nums">{refreshCountdown}s 후 갱신</span>
              </div>

              {/* 시장 데이터 카드 — 컴팩트 (기본 3열 → xl 5열) */}
              <div className="grid grid-cols-3 sm:grid-cols-4 xl:grid-cols-5 gap-2">
                {data.map((item) => (
                  <div key={item.symbol} className="p-3 rounded-xl border border-zinc-100 bg-zinc-50/40 hover:border-zinc-200 hover:bg-white transition-all group">
                    <div className="flex justify-between items-center mb-1">
                      <span className="text-[9px] font-bold text-zinc-400 uppercase tracking-wide group-hover:text-zinc-700 transition-colors truncate pr-1">{item.name}</span>
                      <span className="text-[8px] text-zinc-300 font-mono shrink-0">{item.symbol}</span>
                    </div>
                    <div className="text-sm font-semibold text-zinc-900 tracking-tight tabular-nums leading-tight">
                      {item.price.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
                    </div>
                    <div className={`text-[10px] font-bold mt-0.5 ${item.change_percent >= 0 ? 'text-emerald-600' : 'text-rose-500'}`}>
                      {item.change_percent >= 0 ? '+' : ''}{item.change_percent.toFixed(2)}%
                    </div>
                  </div>
                ))}
              </div>

              {/* US Liquidity Indicators (FRED) */}
              {liquidityData.length > 0 && (
                <div className="mt-5">
                  <div className="flex items-center gap-2 mb-3 pb-2 border-b border-zinc-50">
                    <span className="text-[9px] font-bold text-zinc-400 uppercase tracking-[0.3em]">US Liquidity / Fed &amp; Treasury (FRED)</span>
                    <span className="w-1.5 h-1.5 rounded-full bg-blue-400 animate-pulse shrink-0"></span>
                  </div>
                  <div className="grid grid-cols-3 sm:grid-cols-4 xl:grid-cols-6 gap-2">
                    {liquidityData.map((item) => (
                      <div key={item.series} className="p-3 rounded-xl border border-blue-50 bg-blue-50/30 hover:border-blue-200 hover:bg-blue-50/60 transition-all group" title={item.desc}>
                        <div className="flex justify-between items-center mb-1">
                          <span className="text-[9px] font-bold text-blue-400 uppercase tracking-wide">{item.series}</span>
                          <span className="text-[8px] text-zinc-300 font-mono shrink-0">{item.date?.slice(0, 7)}</span>
                        </div>
                        <div className="text-sm font-semibold text-zinc-800 tracking-tight tabular-nums leading-tight">
                          {item.value.toLocaleString(undefined, { minimumFractionDigits: item.unit === '%' ? 2 : 0, maximumFractionDigits: item.unit === '%' ? 2 : 0 })}
                          <span className="text-[9px] text-zinc-400 ml-0.5 font-normal">{item.unit}</span>
                        </div>
                        <div className="text-[9px] text-zinc-400 mt-0.5 leading-tight truncate">{item.name}</div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </div>

          {/* ── Main Content ── */}
          <main
            ref={mainRef}
            className="flex-1 overflow-y-auto custom-scrollbar relative"
          >
            {/* Sticky Floating Title Bar (스크롤 시 표시) */}
            {(analysis || analyzing) && (
              <div
                className="sticky top-0 z-50 flex items-center justify-between px-8 transition-all duration-200"
                style={{
                  height: headerScrolled ? '52px' : '0px',
                  opacity: headerScrolled ? 1 : 0,
                  pointerEvents: headerScrolled ? 'auto' : 'none',
                  background: 'rgba(255,255,255,0.92)',
                  backdropFilter: 'blur(20px)',
                  WebkitBackdropFilter: 'blur(20px)',
                  borderBottom: headerScrolled ? '1px solid rgba(0,0,0,0.06)' : '1px solid transparent',
                  boxShadow: headerScrolled ? '0 2px 16px rgba(0,0,0,0.06)' : 'none',
                  overflow: 'hidden',
                }}
              >
                <div className="flex items-center gap-3 min-w-0">
                  <div className="w-0.5 h-4 bg-zinc-900 rounded-full shrink-0"></div>
                  <span className="text-sm font-semibold text-zinc-900 truncate tracking-tight">{currentQuery || '매크로 분석'}</span>
                  <span className="text-zinc-200 shrink-0">·</span>
                  <span className={`w-1.5 h-1.5 rounded-full shrink-0 ${selectedModel === 'gemini' ? 'bg-blue-500' : selectedModel === 'gpt' ? 'bg-emerald-500' : 'bg-purple-500'}`}></span>
                  <span className="text-[11px] text-zinc-400 uppercase tracking-widest shrink-0">{selectedModelInfo.name}</span>
                  {analyzing && (
                    <span className="inline-flex items-center gap-1.5 text-[10px] font-bold text-amber-600 bg-amber-50 px-2 py-0.5 rounded-full border border-amber-100 shrink-0">
                      <span className="w-2 h-2 border border-amber-400 border-t-transparent rounded-full spin"></span>분석 중
                    </span>
                  )}
                </div>
                {!analyzing && analysis && (
                  <div className="flex items-center gap-1.5 shrink-0">
                    <button onClick={() => handleFeedback('helpful')}
                      className={`w-8 h-8 flex items-center justify-center rounded-lg border transition-all text-sm ${currentRecord?.feedback === 'helpful' ? 'bg-emerald-50 border-emerald-200' : 'border-zinc-200 hover:bg-zinc-50'}`}
                      title="도움이 됐어요">👍</button>
                    <button onClick={() => handleFeedback('not_helpful')}
                      className={`w-8 h-8 flex items-center justify-center rounded-lg border transition-all text-sm ${currentRecord?.feedback === 'not_helpful' ? 'bg-rose-50 border-rose-200' : 'border-zinc-200 hover:bg-zinc-50'}`}
                      title="개선이 필요해요">👎</button>
                    <button onClick={handleSave}
                      className={`w-8 h-8 flex items-center justify-center rounded-lg border transition-all text-sm ${currentRecord?.saved ? 'bg-amber-50 border-amber-200 text-amber-500' : 'border-zinc-200 hover:bg-zinc-50 text-zinc-400'}`}
                      title={currentRecord?.saved ? '저장됨' : '저장하기'}>★</button>
                    <button onClick={() => { setAnalysis(null); setCurrentQuery(''); setActiveHistoryId(null); }}
                      className="w-8 h-8 flex items-center justify-center rounded-lg border border-zinc-200 hover:bg-zinc-50 text-zinc-400 transition-all text-xs"
                      title="초기화">✕</button>
                  </div>
                )}
              </div>
            )}

            <div className="px-8 py-8 pb-36">
              <section className="min-h-[calc(100vh-120px)]">
                {!analysis && !analyzing ? (
                  <div className="flex flex-col items-center justify-center min-h-[calc(100vh-180px)] text-zinc-200">
                    <svg className="w-16 h-16 mb-5 opacity-10" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={0.8} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                    </svg>
                    <p className="text-sm font-bold uppercase tracking-[0.4em] text-zinc-300">질문을 입력하고 분석을 시작하세요</p>
                    <p className="text-xs text-zinc-200 mt-2">IB 리포트 + 실시간 데이터 기반 · Zero Hallucination Policy</p>
                  </div>
                ) : (
                  <div className="max-w-none">
                    <div className="flex items-start justify-between gap-4 mb-8 pb-6 border-b border-zinc-200">
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center gap-3 mb-2">
                          <div className="w-1 h-6 bg-zinc-950 rounded-full shrink-0"></div>
                          <h2 className="text-xl font-medium tracking-tight text-zinc-950 truncate">{currentQuery || '매크로 분석'}</h2>
                        </div>
                        <div className="flex items-center gap-3 ml-4 flex-wrap">
                          <span className={`w-2 h-2 rounded-full ${selectedModel === 'gemini' ? 'bg-blue-500' : selectedModel === 'gpt' ? 'bg-emerald-500' : 'bg-purple-500'}`}></span>
                          <span className="text-[11px] font-bold text-zinc-400 uppercase tracking-widest">{selectedModelInfo.fullName}</span>
                          <span className="text-zinc-200">·</span>
                          <span className="text-[11px] text-zinc-400">{new Date().toLocaleDateString('ko-KR', { year: 'numeric', month: 'long', day: 'numeric' })}</span>
                          {!analyzing && (
                            <>
                              <span className="text-zinc-200">·</span>
                              <span className="inline-flex items-center gap-1 text-[10px] font-bold text-emerald-600 bg-emerald-50 px-2 py-0.5 rounded-full border border-emerald-100">
                                <span className="w-1 h-1 rounded-full bg-emerald-500"></span>Grounded
                              </span>
                            </>
                          )}
                          {analyzing && (
                            <>
                              <span className="text-zinc-200">·</span>
                              <span className="inline-flex items-center gap-1.5 text-[10px] font-bold text-amber-600 bg-amber-50 px-2 py-0.5 rounded-full border border-amber-100">
                                <span className="w-2 h-2 border border-amber-400 border-t-transparent rounded-full spin"></span>분석 중
                              </span>
                            </>
                          )}
                        </div>
                      </div>

                      {!analyzing && analysis && (
                        <div className="flex items-center gap-2 shrink-0">
                          <button onClick={() => handleFeedback('helpful')}
                            className={`w-9 h-9 flex items-center justify-center rounded-xl border transition-all text-base ${currentRecord?.feedback === 'helpful' ? 'bg-emerald-50 border-emerald-200' : 'border-zinc-200 hover:bg-zinc-50'}`}
                            title="도움이 됐어요">👍</button>
                          <button onClick={() => handleFeedback('not_helpful')}
                            className={`w-9 h-9 flex items-center justify-center rounded-xl border transition-all text-base ${currentRecord?.feedback === 'not_helpful' ? 'bg-rose-50 border-rose-200' : 'border-zinc-200 hover:bg-zinc-50'}`}
                            title="개선이 필요해요">👎</button>
                          <div className="w-px h-6 bg-zinc-100 mx-1"></div>
                          <button onClick={handleSave}
                            className={`w-9 h-9 flex items-center justify-center rounded-xl border transition-all ${currentRecord?.saved ? 'bg-amber-50 border-amber-200 text-amber-500' : 'border-zinc-200 hover:bg-zinc-50 text-zinc-400'}`}
                            title={currentRecord?.saved ? '저장됨' : '저장하기'}>★</button>
                          <button onClick={() => { setAnalysis(null); setCurrentQuery(''); setActiveHistoryId(null); }}
                            className="w-9 h-9 flex items-center justify-center rounded-xl border border-zinc-200 hover:bg-zinc-50 text-zinc-400 transition-all text-sm"
                            title="초기화">✕</button>
                        </div>
                      )}
                    </div>

                    <article className="prose prose-zinc max-w-none
                      prose-h2:text-zinc-950 prose-h2:font-bold prose-h2:text-2xl prose-h2:mt-12 prose-h2:mb-6 prose-h2:tracking-tight prose-h2:border-b prose-h2:border-zinc-100 prose-h2:pb-4
                      prose-h3:text-zinc-500 prose-h3:text-xs prose-h3:font-bold prose-h3:uppercase prose-h3:tracking-[0.2em] prose-h3:mt-10
                      prose-p:text-zinc-700 prose-p:text-lg prose-p:leading-[1.8] prose-p:mb-6
                      prose-strong:text-black prose-strong:font-black
                      prose-li:text-zinc-700 prose-li:text-base prose-li:my-2
                      prose-blockquote:border-l-0 prose-blockquote:bg-white prose-blockquote:p-5 prose-blockquote:rounded-2xl prose-blockquote:border prose-blockquote:border-zinc-100 prose-blockquote:shadow-sm prose-blockquote:not-italic prose-blockquote:text-zinc-600 prose-blockquote:text-sm prose-blockquote:my-4 prose-blockquote:transition-all prose-blockquote:hover:shadow-md prose-blockquote:hover:border-zinc-200">
                      <ReactMarkdown>{analysis}</ReactMarkdown>
                      {analyzing && <span className="cursor-blink"></span>}
                    </article>

                  </div>
                )}
              </section>

              {/* ── 이슈 이력 로그 ── */}
              {issueLog.length > 0 && (
                <div className="mt-12 pt-6 border-t border-zinc-100">
                  <div className="flex items-center justify-between mb-3">
                    <div className="flex items-center gap-2">
                      <span className="text-[9px] font-bold text-zinc-300 uppercase tracking-[0.3em]">시스템 이슈 이력</span>
                      <span className="text-[9px] font-mono text-zinc-200 bg-zinc-50 px-1.5 py-0.5 rounded-md">{issueLog.length}</span>
                    </div>
                    <button onClick={() => setIssueLog([])}
                      className="text-[9px] text-zinc-200 hover:text-rose-400 transition-colors">
                      지우기
                    </button>
                  </div>
                  <div className="space-y-1">
                    {issueLog.map((item) => (
                      <div key={item.id} className="flex items-start gap-2.5 py-1.5 px-3 rounded-lg bg-zinc-50/60">
                        <span className="text-[10px] shrink-0 mt-px">{item.type === 'warning' ? '⚠️' : '⚡'}</span>
                        <span className="text-[10px] text-zinc-400 font-mono tabular-nums shrink-0">{item.timestamp}</span>
                        <span className="text-[10px] text-zinc-500 leading-relaxed">{item.message.replace(/^[⚠️⚡\s]+/, '')}</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </main>
        </div>
      </div>

      {/* ── Floating Input Bar — fixed, GNB 너비 반영 ── */}
      <div className="fixed bottom-0 z-[200] pb-5 pt-7"
        style={{
          left: `${sidebarWidth}px`,
          right: 0,
          background: 'linear-gradient(to top, rgba(248,248,250,0.98) 55%, rgba(248,248,250,0))',
          transition: 'left 0.3s ease',
        }}>
        <div className="max-w-3xl mx-auto px-4">

          {/* Suggested Queries */}
          {showSuggestions && !analyzing && (
            <div className="mb-3 rounded-2xl px-4 py-3 fade-in"
              style={{
                background: 'rgba(255,255,255,0.75)',
                backdropFilter: 'blur(20px)',
                WebkitBackdropFilter: 'blur(20px)',
                border: '1px solid rgba(255,255,255,0.9)',
                boxShadow: '0 8px 32px rgba(0,0,0,0.08), inset 0 1px 0 rgba(255,255,255,0.9)',
              }}>
              <p className="text-[10px] font-bold text-zinc-400 uppercase tracking-widest mb-2.5">추천 질문</p>
              <div className="flex flex-wrap gap-2">
                {SUGGESTED_QUERIES.map((q, i) => (
                  <button key={i} onMouseDown={() => { setQuery(q); setShowSuggestions(false); }}
                    className="text-xs text-zinc-600 px-3 py-1.5 rounded-xl transition-all hover:text-zinc-900"
                    style={{ background: 'rgba(255,255,255,0.6)', border: '1px solid rgba(0,0,0,0.07)', backdropFilter: 'blur(8px)' }}>
                    {q}
                  </button>
                ))}
              </div>
            </div>
          )}

          {/* Model dropdown */}
          {showModelDropdown && !analyzing && (
            <div className="mb-3 rounded-2xl overflow-hidden py-2 fade-in" ref={modelDropdownRef}
              style={{
                background: 'rgba(255,255,255,0.82)',
                backdropFilter: 'blur(24px)',
                WebkitBackdropFilter: 'blur(24px)',
                border: '1px solid rgba(255,255,255,0.95)',
                boxShadow: '0 12px 40px rgba(0,0,0,0.1), inset 0 1px 0 rgba(255,255,255,1)',
              }}>
              <p className="text-[10px] font-bold text-zinc-400 uppercase tracking-widest px-4 pt-2 pb-2">모델 선택</p>
              {MODELS.map((m) => (
                <button key={m.id}
                  onClick={() => { if (m.active) { setSelectedModel(m.id); setShowModelDropdown(false); } }}
                  className={`w-full px-4 py-3 flex items-center gap-3 transition-all ${!m.active ? 'opacity-30 cursor-not-allowed' : selectedModel === m.id ? 'bg-white/60' : 'hover:bg-white/40'}`}>
                  <span className={`w-2.5 h-2.5 rounded-full shrink-0 ${m.id === 'gemini' ? 'bg-blue-500' : m.id === 'gpt' ? 'bg-emerald-500' : 'bg-purple-500'}`}></span>
                  <div className="flex flex-col items-start flex-1">
                    <span className={`text-sm font-semibold ${selectedModel === m.id ? 'text-zinc-950' : 'text-zinc-600'}`}>{m.fullName}</span>
                    <span className="text-[10px] text-zinc-400">{m.desc}</span>
                  </div>
                  {selectedModel === m.id && (
                    <svg className="w-4 h-4 text-emerald-500 shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M5 13l4 4L19 7" />
                    </svg>
                  )}
                </button>
              ))}
            </div>
          )}

          {/* Main glass input card */}
          <div className="rounded-2xl transition-all duration-300"
            style={{
              background: analyzing ? 'rgba(255,255,255,0.65)' : 'rgba(255,255,255,0.82)',
              backdropFilter: 'blur(28px)',
              WebkitBackdropFilter: 'blur(28px)',
              border: analyzing ? '1px solid rgba(0,0,0,0.06)' : showSuggestions ? '1px solid rgba(0,0,0,0.16)' : '1px solid rgba(255,255,255,0.95)',
              boxShadow: analyzing
                ? '0 4px 24px rgba(0,0,0,0.06), inset 0 1px 0 rgba(255,255,255,0.8)'
                : '0 8px 40px rgba(0,0,0,0.12), 0 2px 8px rgba(0,0,0,0.06), inset 0 1px 0 rgba(255,255,255,1)',
            }}>
            <div className="px-4 pt-3 pb-2">
              <textarea
                ref={queryRef}
                value={query}
                onChange={(e) => { if (!analyzing) setQuery(e.target.value); }}
                onFocus={() => { if (!analyzing) setShowSuggestions(true); }}
                onBlur={() => setTimeout(() => setShowSuggestions(false), 200)}
                onKeyDown={(e) => { if (!analyzing && e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); handleAnalyze(); } }}
                disabled={analyzing}
                placeholder={analyzing ? '⏳ AI가 분석 중입니다...' : '분석하고 싶은 주제를 입력하세요... (Enter로 실행)'}
                rows={1}
                style={{ cursor: analyzing ? 'not-allowed' : 'text', background: 'transparent' }}
                className={`w-full text-sm text-zinc-800 placeholder-zinc-400 resize-none font-normal leading-relaxed ${analyzing ? 'opacity-40' : ''}`}
              />
            </div>
            <div style={{ height: '1px', background: 'rgba(0,0,0,0.05)', margin: '0 12px' }}></div>
            <div className="flex items-center justify-between px-3 py-2.5">
              <button onClick={() => { if (!analyzing) setShowModelDropdown(!showModelDropdown); }} disabled={analyzing}
                className="h-8 flex items-center gap-1.5 px-3 rounded-lg text-xs font-medium transition-all"
                style={{
                  background: analyzing ? 'transparent' : showModelDropdown ? 'rgba(0,0,0,0.07)' : 'rgba(0,0,0,0.03)',
                  border: '1px solid rgba(0,0,0,0.07)',
                  color: analyzing ? 'rgba(0,0,0,0.3)' : 'rgba(0,0,0,0.55)',
                  cursor: analyzing ? 'not-allowed' : 'pointer',
                }}>
                <span className={`w-1.5 h-1.5 rounded-full shrink-0 ${selectedModel === 'gemini' ? 'bg-blue-500' : selectedModel === 'gpt' ? 'bg-emerald-500' : 'bg-purple-500'}`}></span>
                <span>{selectedModelInfo.name}</span>
                <svg className={`w-3 h-3 shrink-0 transition-transform ${showModelDropdown ? 'rotate-180' : ''}`} style={{ color: 'rgba(0,0,0,0.3)' }} fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M19 9l-7 7-7-7" />
                </svg>
              </button>

              <div className="flex items-center gap-2">
                <span className="hidden sm:inline-flex h-8 items-center gap-1.5 px-3 rounded-lg text-[10px] font-bold"
                  style={{ background: 'rgba(16,185,129,0.08)', border: '1px solid rgba(16,185,129,0.18)', color: 'rgb(5,150,105)' }}>
                  <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 shrink-0"></span>
                  RAG · Grounded
                </span>
                {analyzing ? (
                  <button onClick={handleStop}
                    className="h-8 flex items-center gap-1.5 px-3 rounded-lg text-xs font-semibold transition-all"
                    style={{ background: 'rgba(239,68,68,0.08)', border: '1px solid rgba(239,68,68,0.2)', color: 'rgb(220,38,38)' }}>
                    <span className="w-2.5 h-2.5 rounded-sm bg-rose-500 shrink-0"></span>
                    중단
                  </button>
                ) : (
                  <button onClick={handleAnalyze}
                    className="h-8 flex items-center gap-1.5 px-3 rounded-lg text-xs font-semibold transition-all active:scale-95"
                    style={{
                      background: 'rgba(9,9,11,0.88)',
                      border: '1px solid rgba(255,255,255,0.12)',
                      color: 'white',
                      boxShadow: '0 2px 8px rgba(0,0,0,0.18), inset 0 1px 0 rgba(255,255,255,0.1)',
                    }}>
                    <svg className="w-3.5 h-3.5 shrink-0" fill="currentColor" viewBox="0 0 20 20">
                      <path d="M10 18a8 8 0 100-16 8 8 0 000 16zM9.555 7.168A1 1 0 008 8v4a1 1 0 001.555.832l3-2a1 1 0 000-1.664l-3-2z" />
                    </svg>
                    AI 분석 실행
                  </button>
                )}
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* ── 토스트 알림 스택 (우하단 고정) ── */}
      {toasts.length > 0 && (
        <div className="fixed bottom-24 right-5 z-[300] flex flex-col gap-2 items-end">
          {toasts.map((toast) => (
            <div key={toast.id} className="toast-in flex items-start gap-2.5 px-4 py-3 rounded-xl shadow-lg max-w-xs"
              style={{
                background: toast.type === 'warning'
                  ? 'rgba(255,251,235,0.96)' : 'rgba(239,246,255,0.96)',
                border: toast.type === 'warning'
                  ? '1px solid rgba(251,191,36,0.3)' : '1px solid rgba(147,197,253,0.4)',
                backdropFilter: 'blur(16px)',
                boxShadow: '0 8px 32px rgba(0,0,0,0.10)',
              }}>
              <span className="text-sm shrink-0">{toast.type === 'warning' ? '⚠️' : '⚡'}</span>
              <p className="text-[11px] leading-snug"
                style={{ color: toast.type === 'warning' ? 'rgb(146,64,14)' : 'rgb(30,64,175)' }}>
                {toast.message.replace(/^[⚠️⚡\s]+/, '')}
              </p>
              <button onClick={() => setToasts(prev => prev.filter(t => t.id !== toast.id))}
                className="ml-1 text-zinc-300 hover:text-zinc-500 transition-colors shrink-0 text-xs leading-none mt-0.5">✕</button>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
