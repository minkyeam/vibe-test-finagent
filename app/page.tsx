'use client';

import { useState, useEffect, useRef, useCallback, useMemo } from "react";
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { useSession, signIn, signOut } from "next-auth/react";

interface MarketData {
  symbol: string;
  name: string;
  price: number;
  change_percent: number;
  date?: string;
  is_closed?: boolean;
}

interface LiquidityItem {
  series: string;
  name: string;
  value: number;
  unit: string;
  date: string;
  desc: string;
  mom_change?: number;
}

interface ChatMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
}

interface AnalysisRecord {
  id: string;
  title: string;
  model: string;
  messages: ChatMessage[];
  timestamp: string;
  feedback?: 'helpful' | 'not_helpful' | null;
  saved: boolean;
}

type AIModel = 'gemini' | 'claude' | 'gpt';

interface PortfolioItem {
  ticker: string;
  name: string;
  quantity: number;
  buy_price: number;
  current_price: number;
  profit_krw: number;
}

type LiquidityTickerItem = LiquidityItem & { isLiquidity: true };
type MarketTickerItem = MarketData & { isLiquidity?: false };
type TickerItem = LiquidityTickerItem | MarketTickerItem;

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
  const [userPortfolio, setUserPortfolio] = useState('');
  const [isMyPortfolioModalOpen, setIsMyPortfolioModalOpen] = useState(false);
  const [portfolioModalSubView, setPortfolioModalSubView] = useState<'list' | 'edit'>('list');
  const [portfolioItems, setPortfolioItems] = useState<PortfolioItem[]>([]);
  const [isParsingImage, setIsParsingImage] = useState(false);
  const [portfolioCurrency, setPortfolioCurrency] = useState<'USD' | 'KRW'>('USD');
  const [macroAlerts, setMacroAlerts] = useState<any[]>([]);
  const [isStressTesting, setIsStressTesting] = useState(false);
  const [stressTestResult, setStressTestResult] = useState<any>(null);

  const mainRef = useRef<HTMLElement>(null);

  const abortControllerRef = useRef<AbortController | null>(null);
  const modelDropdownRef = useRef<HTMLDivElement>(null);
  const queryRef = useRef<HTMLTextAreaElement>(null);
  const [dataFetchedAt, setDataFetchedAt] = useState<string>('');
  const [refreshCountdown, setRefreshCountdown] = useState<number>(30);
  const toastIdCounter = useRef(0);
  const { data: session, status } = useSession();

  // ── Get USD/KRW exchange rate from market data ──
  const usdKrwRate = useMemo(() => {
    const usdKrwItem = data.find(item => item.symbol === 'KRW=X');
    return usdKrwItem?.price || 1440; // fallback to approximate rate
  }, [data]);

  // ── Currency conversion helpers ──
  const convertCurrency = useCallback((amount: number, fromUSD: boolean = true) => {
    if (portfolioCurrency === 'USD') {
      return fromUSD ? amount : amount / usdKrwRate;
    } else {
      return fromUSD ? amount * usdKrwRate : amount;
    }
  }, [portfolioCurrency, usdKrwRate]);

  const formatCurrency = useCallback((amount: number, fromUSD: boolean = true) => {
    const convertedAmount = convertCurrency(amount, fromUSD);
    if (portfolioCurrency === 'USD') {
      return `$${convertedAmount.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
    } else {
      return `₩${Math.round(convertedAmount).toLocaleString()}`;
    }
  }, [convertCurrency, portfolioCurrency]);

  // ── Group history by date (Apple Notes style) ──
  const groupedHistory = useMemo(() => {
    const now = new Date();
    const todayStart = new Date(now.getFullYear(), now.getMonth(), now.getDate());
    const weekAgo = new Date(todayStart); weekAgo.setDate(todayStart.getDate() - 7);
    const monthAgo = new Date(todayStart); monthAgo.setDate(todayStart.getDate() - 30);

    const buckets: Record<string, AnalysisRecord[]> = {};
    const order: string[] = [];

    history.forEach(record => {
      const d = new Date(record.timestamp);
      let label: string;
      if (d >= todayStart) label = '오늘';
      else if (d >= weekAgo) label = '이번 주';
      else if (d >= monthAgo) label = '이전 30일';
      else {
        const y = d.getFullYear(), m = d.getMonth() + 1;
        label = y === now.getFullYear() ? `${m}월` : `${y}년 ${m}월`;
      }
      if (!buckets[label]) { buckets[label] = []; order.push(label); }
      buckets[label].push(record);
    });

    return order.map(label => ({ label, items: buckets[label] }));
  }, [history]);

  const pushToast = useCallback((message: string, type: 'warning' | 'info' = 'warning') => {
    const id = `toast-${Date.now()}-${++toastIdCounter.current}`;
    const timestamp = new Date().toLocaleTimeString('ko-KR', { hour: '2-digit', minute: '2-digit', second: '2-digit' });
    setToasts(prev => [...prev, { id, message, type }]);
    setIssueLog(prev => [{ id, timestamp, message, type }, ...prev]);
    setTimeout(() => setToasts(prev => prev.filter(t => t.id !== id)), 5000);
  }, []);

  const fetchMarketData = useCallback(() => {
    console.log('Fetching market data...');
    fetch('/api/market-data')
      .then((res) => res.json())
      .then((json) => {
        console.log('Market data received:', json);
        if (json.data) {
          setData(json.data);
          console.log('Data set to state:', json.data.length, 'items');
        }
        const now = new Date();
        const pad = (n: number) => String(n).padStart(2, '0');
        setDataFetchedAt(
          `${now.getFullYear()}.${pad(now.getMonth() + 1)}.${pad(now.getDate())} ${pad(now.getHours())}:${pad(now.getMinutes())}`
        );
        setRefreshCountdown(30);
      })
      .catch((error) => {
        console.error('Market data fetch error:', error);
        pushToast('마켓 데이터 로딩 오류', 'warning');
      });
  }, [pushToast]);

  const fetchLiquidityData = useCallback(() => {
    console.log('Fetching liquidity data...');
    fetch('/api/liquidity')
      .then((res) => res.json())
      .then((json) => {
        console.log('Liquidity data received:', json);
        if (json.data) {
          setLiquidityData(json.data);
          console.log('Liquidity data set to state:', json.data.length, 'items');
        }
      })
      .catch((error) => {
        console.error('Liquidity data fetch error:', error);
        pushToast('유동성 데이터 로딩 오류', 'warning');
      });
  }, [pushToast]);

  const fetchMacroAlerts = useCallback(() => {
    fetch('/api/macro-alerts')
      .then((res) => res.json())
      .then((json) => {
        if (json.data && json.data.length > 0) {
          setMacroAlerts(json.data);
        }
      })
      .catch((error) => console.error('Macro alerts fetch error:', error));
  }, []);

  const fetchDbPortfolio = useCallback(() => {
    const email = session?.user?.email;
    const url = email ? `/api/portfolio?email=${encodeURIComponent(email)}` : '/api/portfolio';

    fetch(url)
      .then((res) => res.json())
      .then((json) => {
        if (json.data && json.data.length > 0) {
          setPortfolioItems(json.data);
          const portStr = json.data.map((i: PortfolioItem) => `${i.name || i.ticker}(${i.ticker}) ${i.quantity.toFixed(2)}주`).join(', ');
          setUserPortfolio(portStr);
          localStorage.setItem('user_portfolio', portStr);
        } else {
          const savedPortfolio = localStorage.getItem('user_portfolio');
          if (savedPortfolio) setUserPortfolio(savedPortfolio);
        }
      })
      .catch((e) => {
        console.error("DB Fetch Error", e);
        const savedPortfolio = localStorage.getItem('user_portfolio');
        if (savedPortfolio) setUserPortfolio(savedPortfolio);
      });
  }, [session?.user?.email]);

  const handleMacroStressTest = async () => {
    if (!session?.user?.email) {
      pushToast("로그인이 필요합니다.", "warning");
      return;
    }
    setIsStressTesting(true);
    setStressTestResult(null);
    try {
      const res = await fetch('/api/portfolio/stress-test', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email: session.user.email })
      });
      const json = await res.json();
      if (json.data) {
        setStressTestResult(json.data);
      } else {
        pushToast(json.error || "분석 실패", "warning");
      }
    } catch (e) {
      pushToast("스트레스 테스트 오류", "warning");
    } finally {
      setIsStressTesting(false);
    }
  };

  useEffect(() => {
    fetchMarketData();
    fetchLiquidityData();
    fetchMacroAlerts();
    if (session) fetchDbPortfolio();
    const pollInterval = setInterval(fetchMarketData, 30_000);
    const liquidityPollInterval = setInterval(fetchLiquidityData, 60 * 60_000);
    const macroPollInterval = setInterval(fetchMacroAlerts, 60 * 60_000);
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
  }, [fetchMarketData, fetchLiquidityData, fetchDbPortfolio, session]);

  // Scroll listener — 별도 effect로 분리해 mainRef 마운트 타이밍 보장
  useEffect(() => {
    const mainEl = mainRef.current;
    if (!mainEl) return;
    const handleScroll = () => setHeaderScrolled(mainEl.scrollTop > 60);
    mainEl.addEventListener('scroll', handleScroll);
    return () => mainEl.removeEventListener('scroll', handleScroll);
  }, []);

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

    const currentSession = history.find(r => r.id === activeHistoryId);
    const apiHistory = currentSession ? currentSession.messages : [];

    try {
      const response = await fetch('/api/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: userQuery,
          model: selectedModel,
          history: apiHistory,
          user_portfolio: userPortfolio,
          email: session?.user?.email
        }),
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
        setQuery('');
        const updatedHistory = [...history];
        const sessionIdx = updatedHistory.findIndex(r => r.id === activeHistoryId);
        if (sessionIdx >= 0) {
          updatedHistory[sessionIdx] = {
            ...updatedHistory[sessionIdx],
            messages: [
              ...updatedHistory[sessionIdx].messages,
              { id: Date.now().toString() + 'u', role: 'user', content: userQuery },
              { id: Date.now().toString() + 'a', role: 'assistant', content: finalContent }
            ],
            timestamp: new Date().toISOString()
          };
          saveHistory(updatedHistory);
        } else {
          const newRecord: AnalysisRecord = {
            id: Date.now().toString(),
            title: userQuery,
            model: selectedModel,
            messages: [
              { id: Date.now().toString() + 'u', role: 'user', content: userQuery },
              { id: Date.now().toString() + 'a', role: 'assistant', content: finalContent }
            ],
            timestamp: new Date().toISOString(),
            feedback: null,
            saved: false,
          };
          saveHistory([newRecord, ...updatedHistory]);
          setActiveHistoryId(newRecord.id);
        }
      }
    } catch (err: unknown) {
      if (err instanceof Error && err.name !== 'AbortError') {
        setAnalysis('분석 중 오류가 발생했습니다. 다시 시도해주세요.');
      }
    } finally {
      setAnalyzing(false);
      abortControllerRef.current = null;
    }
  }, [query, analyzing, selectedModel, history, activeHistoryId, userPortfolio, session, pushToast, cleanOutput]);


  const handleFeedback = (type: 'helpful' | 'not_helpful') => {
    if (!activeHistoryId) return;
    saveHistory(history.map(r => r.id === activeHistoryId ? { ...r, feedback: type } : r));
  };

  const handleSave = () => {
    if (!activeHistoryId) return;
    saveHistory(history.map(r => r.id === activeHistoryId ? { ...r, saved: !r.saved } : r));
  };

  const currentRecord = history.find(r => r.id === activeHistoryId);
  const selectedModelInfo = MODELS.find(m => m.id === selectedModel) ?? MODELS[0];
  const sidebarWidth = sidebarOpen ? 272 : 72; // w-64(256) + m-2(8×2) | w-14(56) + m-2(8×2)

  if (status === "loading") {
    return (
      <div className="flex h-screen w-full items-center justify-center">
        <div className="w-6 h-6 border-2 border-zinc-200 border-t-zinc-900 rounded-full animate-spin"></div>
      </div>
    );
  }

  if (!session) {
    return (
      <div className="flex h-screen w-full flex-col items-center justify-center font-sans selection:bg-emerald-100 selection:text-emerald-900 tracking-tight bg-zinc-50 relative">
        <style>{`
          @keyframes fadeIn { from { opacity: 0; transform: translateY(6px); } to { opacity: 1; transform: translateY(0); } }
        `}</style>
        <div className="absolute inset-0 z-0 bg-gradient-to-br from-indigo-50/50 via-white to-emerald-50/50 pointer-events-none"></div>
        <div className="w-full max-w-sm rounded-[28px] premium-glass p-8 shadow-2xl shadow-zinc-300/30 relative z-50" style={{ animation: 'fadeIn 0.4s ease forwards' }}>
          <div className="mb-10 text-center">
            <h1 className="text-2xl font-black tracking-tight text-zinc-900 mb-2">N Finance Agent</h1>
            <p className="text-sm font-medium text-zinc-400">Institutional Grade AI-powered macro research</p>
          </div>

          <div className="space-y-3 relative z-[60]">
            <button onClick={() => { console.log('Google login clicked'); signIn('google'); }} className="w-full h-12 bg-white hover:bg-zinc-50 text-zinc-700 rounded-xl transition-all border border-zinc-200 flex items-center justify-center gap-3 font-semibold text-sm shadow-sm hover:shadow-md cursor-pointer relative z-[70]">
              <svg className="w-5 h-5" viewBox="0 0 24 24"><path fill="#4285F4" d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z" /><path fill="#34A853" d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z" /><path fill="#FBBC05" d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z" /><path fill="#EA4335" d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z" /></svg>
              Google 계정으로 계속
            </button>

            <button onClick={() => { console.log('Naver login clicked'); signIn('naver'); }} className="w-full h-12 bg-[#03C75A] hover:bg-[#02b350] text-white rounded-xl transition-all shadow-sm hover:shadow-md flex items-center justify-center gap-3 font-semibold text-sm border border-[#03C75A] cursor-pointer relative z-[70]">
              <span className="font-black text-lg -mt-px">N</span>
              네이버 계정으로 계속
            </button>
          </div>

          <p className="mt-8 text-center text-[10px] text-zinc-400 font-medium">
            계속 진행하면 서비스 이용약관 및 개인정보 처리방침에 동의하는 것으로 간주합니다.
          </p>
        </div>
      </div>
    );
  }

  return (
    <div
      className="h-screen text-zinc-900 font-sans selection:bg-emerald-100 selection:text-emerald-900 tracking-tight flex flex-col overflow-hidden relative bg-zinc-50"
    >
      {/* ── Ambient Mesh Gradient Background ── */}
      <div className="absolute inset-0 z-0 overflow-hidden pointer-events-none">
        <div className="absolute -top-[20%] -left-[10%] w-[50%] h-[70%] bg-indigo-300/40 rounded-full mix-blend-multiply filter blur-[120px] animate-float" style={{ animationDuration: '15s' }}></div>
        <div className="absolute top-[10%] -right-[10%] w-[60%] h-[80%] bg-emerald-300/30 rounded-full mix-blend-multiply filter blur-[140px] animate-float" style={{ animationDuration: '20s', animationDelay: '2s' }}></div>
        <div className="absolute -bottom-[20%] left-[20%] w-[70%] h-[60%] bg-blue-300/30 rounded-full mix-blend-multiply filter blur-[130px] animate-float" style={{ animationDuration: '18s', animationDelay: '4s' }}></div>
        <div className="absolute inset-0 bg-white/40 backdrop-blur-[60px]"></div>
      </div>

      <div className="relative z-10 flex flex-col h-full overflow-hidden">

        <style>{`
        @keyframes marquee { 0% { transform: translateX(0); } 100% { transform: translateX(-50%); } }
        .animate-marquee { display: inline-flex; align-items: center; animation: marquee 50s linear infinite; }
        @keyframes blink { 0%, 100% { opacity: 1; } 50% { opacity: 0; } }
        .cursor-blink { display: inline-block; width: 2px; height: 1.1em; background-color: #10b981; margin-left: 2px; vertical-align: middle; animation: blink 1s step-end infinite; }
        .custom-scrollbar::-webkit-scrollbar { width: 4px; }
        .custom-scrollbar::-webkit-scrollbar-track { background: transparent; }
        .custom-scrollbar::-webkit-scrollbar-thumb { background: rgba(0,0,0,0.1); border-radius: 10px; }
        .custom-scrollbar::-webkit-scrollbar-thumb:hover { background: rgba(0,0,0,0.18); }
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

        {/* ── ① Sticky Ticker Bar ── */}
        <section className="sticky top-0 z-[100] w-full premium-glass shrink-0">
          <div className="flex items-center h-12 px-5 gap-3">

            {/* ── Logo ── */}
            <div className="flex items-center gap-2.5 shrink-0 pr-4">
              <div className="w-7 h-7 rounded-lg bg-zinc-950 flex items-center justify-center shadow-lg shadow-black/20">
                <span className="text-white text-[13px] font-black tracking-tight">N</span>
              </div>
              <div className="flex items-center gap-2">
                <span className="text-[14px] font-black text-zinc-900 tracking-tight">Finance</span>
                <div className="flex items-center gap-1.5 px-2 py-0.5 rounded-full bg-emerald-500/10 ring-1 ring-emerald-500/20">
                  <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse"></span>
                  <span className="text-[10px] font-black text-emerald-600 uppercase tracking-widest">Live</span>
                </div>
              </div>
            </div>

            {/* ── Marquee ── */}
            <div className="flex-1 overflow-hidden min-w-0">
              <div className="animate-marquee whitespace-nowrap inline-flex items-center">
                {data.length === 0 && liquidityData.length === 0 ? (
                  <span className="text-[12px] text-zinc-400 font-medium animate-pulse">데이터 로드 중...</span>
                ) : (
                  [...data, ...liquidityData.map(l => ({ ...l, isLiquidity: true as const })), ...data, ...liquidityData.map(l => ({ ...l, isLiquidity: true as const }))].map((item: TickerItem, idx) => (
                    item.isLiquidity ? (
                      <span key={`liq-${item.series}-${idx}`} className="inline-flex items-center gap-2 pr-8">
                        <span className="text-[11px] font-semibold text-indigo-500 font-mono">{item.series}</span>
                        <span className="text-[12px] font-bold text-zinc-700 tabular-nums font-mono">{item.value.toLocaleString(undefined, { maximumFractionDigits: 2 })}</span>
                        <span className="text-[10px] text-zinc-400">{item.unit}</span>
                        {item.mom_change !== undefined && item.mom_change !== null && (
                          <span className={`text-[10px] font-bold ${item.mom_change >= 0 ? 'text-emerald-500' : 'text-rose-500'}`}>
                            {item.mom_change >= 0 ? '+' : ''}{item.mom_change.toFixed(1)}%
                          </span>
                        )}
                        <span className="text-zinc-300 pl-4">·</span>
                      </span>
                    ) : (
                      <span key={`${item.symbol}-${idx}`} className="inline-flex items-center gap-2 pr-8">
                        <span className="text-[12px] font-medium text-zinc-600">{item.name}</span>
                        <span className="text-[12px] font-bold text-zinc-900 tabular-nums font-mono">{item.price.toLocaleString()}</span>
                        <span className={`text-[11px] font-semibold px-1.5 py-0.5 rounded-lg ${item.change_percent >= 0 ? 'bg-emerald-50 text-emerald-600' : 'bg-rose-50 text-rose-600'}`}>
                          {item.change_percent >= 0 ? '+' : ''}{item.change_percent.toFixed(2)}%
                        </span>
                        <span className="text-zinc-300 pl-4">·</span>
                      </span>
                    )
                  ))
                )}
              </div>
            </div>

            {/* ── Right Controls ── */}
            <div className="flex items-center gap-2 shrink-0 pl-4">
              <button
                onClick={fetchMarketData}
                title="새로고침"
                className="flex items-center gap-1.5 h-8 px-3 rounded-xl glass-inner text-zinc-500 hover:text-zinc-900 transition-all"
              >
                <svg className="w-3.5 h-3.5 shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
                </svg>
                <span className="text-[11px] font-mono font-semibold tabular-nums hidden sm:block">{refreshCountdown}s</span>
              </button>

              <button
                onClick={() => setIsExpanded(!isExpanded)}
                className={`h-8 px-3.5 flex items-center gap-1.5 rounded-xl text-[12px] font-semibold transition-all ${isExpanded
                  ? 'bg-zinc-900 text-white shadow-sm'
                  : 'glass-inner text-zinc-600 hover:text-zinc-900'
                  }`}
              >
                <svg
                  className={`w-3.5 h-3.5 transition-transform duration-300 ${isExpanded ? 'rotate-180' : ''}`}
                  fill="none" viewBox="0 0 24 24" stroke="currentColor"
                >
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M19 9l-7 7-7-7" />
                </svg>
                <span className="hidden md:block">{isExpanded ? '닫기' : '터미널'}</span>
              </button>
            </div>

          </div>
        </section>

        {/* ── ② Body: GNB(좌, 전체 높이) + 우측 컬럼(마켓패널 + 메인) ── */}
        <div className="flex flex-1 min-h-0">

          {/* ── Left Sidebar (Premium Glass) ── */}
          <aside
            className={`flex flex-col shrink-0 glass-sidebar rounded-2xl m-2 overflow-hidden transition-all duration-300 ${sidebarOpen ? 'w-64' : 'w-14'}`}
          >
            {/* Sidebar Toggle + Title */}
            <div className="flex items-center gap-3 px-4 py-3.5 shrink-0">
              <button onClick={() => setSidebarOpen(!sidebarOpen)}
                className={`w-8 h-8 flex items-center justify-center rounded-lg hover:bg-black/[0.06] transition-all text-zinc-400 hover:text-zinc-700 ${sidebarOpen ? '' : 'mx-auto'}`}>
                <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  {sidebarOpen
                    ? <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M11 19l-7-7 7-7M18 19l-7-7 7-7" />
                    : <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 5l7 7-7 7M6 5l7 7-7 7" />
                  }
                </svg>
              </button>
              {sidebarOpen && <span className="text-[13px] font-bold text-zinc-900 tracking-tight">N Finance</span>}
            </div>

            {/* Portfolio Button */}
            <div className="px-3 pb-3 shrink-0">
              <button
                onClick={() => {
                  fetchDbPortfolio();
                  setPortfolioModalSubView('list');
                  setIsMyPortfolioModalOpen(true);
                }}
                className={`flex items-center gap-3 rounded-xl transition-all font-medium hover:bg-black/[0.04] active:scale-[0.98] ${sidebarOpen ? 'w-full h-10 px-3 text-[12px] text-zinc-700' : 'w-10 h-10 justify-center text-zinc-500 mx-auto'}`}
                title="내 자산 현황 보기"
              >
                <div className="shrink-0 w-7 h-7 flex items-center justify-center bg-emerald-500 rounded-lg text-white">
                  <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                  </svg>
                </div>
                {sidebarOpen && <span className="flex-1 text-left">마이 포트폴리오</span>}
              </button>
            </div>

            {/* History Section */}
            <div className="flex-1 overflow-y-auto custom-scrollbar px-2 pt-1 pb-2 min-h-0">
              {!sidebarOpen ? (
                <div className="flex flex-col items-center gap-3 py-3">
                  <button
                    onClick={() => { setAnalysis(null); setCurrentQuery(''); setActiveHistoryId(null); setQuery(''); }}
                    className="w-9 h-9 rounded-lg flex items-center justify-center text-zinc-400 hover:text-zinc-700 hover:bg-black/[0.06] transition-all"
                    title="새 리포트"
                  >
                    <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M12 4v16m8-8H4" /></svg>
                  </button>
                </div>
              ) : history.length === 0 ? (
                <div className="px-4 py-12 text-center">
                  <p className="text-[12px] font-medium text-zinc-400 leading-relaxed">분석을 시작하면<br />이곳에 기록됩니다</p>
                </div>
              ) : (
                <div>
                  {groupedHistory.map((group, gi) => (
                    <div key={group.label} className="mb-4">
                      {/* Date section header */}
                      <div className={`flex items-center justify-between px-3 ${gi === 0 ? 'pt-1' : 'pt-2'} pb-2`}>
                        <span className="text-[11px] font-black text-zinc-400 uppercase tracking-widest">{group.label}</span>
                        {gi === 0 && (
                          <button
                            onClick={() => { setAnalysis(null); setCurrentQuery(''); setActiveHistoryId(null); setQuery(''); }}
                            className="w-6 h-6 flex items-center justify-center rounded-md hover:bg-black/[0.06] transition-colors text-zinc-400 hover:text-zinc-700"
                            title="새 리포트"
                          >
                            <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M12 4v16m8-8H4" />
                            </svg>
                          </button>
                        )}
                      </div>
                      <div className="space-y-1">
                        {group.items.map((record) => (
                          <button
                            key={record.id}
                            onClick={() => { setAnalysis(null); setActiveHistoryId(record.id); setCurrentQuery(record.title); setQuery(''); }}
                            className={`w-full px-3 py-2.5 rounded-xl transition-all text-left block relative
                            ${activeHistoryId === record.id
                                ? 'bg-white/60 shadow-sm ring-1 ring-black/5'
                                : 'hover:bg-black/[0.04]'}`}
                          >
                            {/* Title — prominent, Apple Notes style */}
                            <p className={`text-[13px] font-semibold leading-snug line-clamp-1 ${activeHistoryId === record.id ? 'text-zinc-950 font-bold' : 'text-zinc-700'
                              }`}>{record.title || '새 분석'}</p>
                            {/* Meta: date · model — muted secondary line */}
                            <p className="text-[10px] font-medium text-zinc-400 mt-1 truncate">
                              {new Date(record.timestamp).toLocaleDateString('ko-KR', { month: 'short', day: 'numeric' })}
                              <span className="mx-1">·</span>
                              <span className="uppercase tracking-wider">{record.model}</span>
                            </p>
                          </button>
                        ))}
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>

            {/* User & Footer Area */}
            <div className="mt-auto shrink-0 relative pt-2">
              <div className="absolute top-0 left-4 right-4 h-px bg-gradient-to-r from-transparent via-zinc-200 to-transparent"></div>
              {session ? (
                <div className={`p-3 ${sidebarOpen ? '' : 'flex justify-center'}`}>
                  <div className={`flex items-center gap-3 px-2 py-2 rounded-lg hover:bg-black/[0.04] transition-all cursor-default ${sidebarOpen ? '' : 'justify-center p-0'}`}>
                    <div className="relative shrink-0">
                      {session.user?.image ? (
                        <img src={session.user.image} alt="Profile" className="w-8 h-8 rounded-full" />
                      ) : (
                        <div className="w-8 h-8 rounded-full bg-zinc-200 flex items-center justify-center text-xs text-zinc-600 font-bold uppercase">{session.user?.name?.[0] || 'U'}</div>
                      )}
                      <div className="absolute -bottom-0.5 -right-0.5 w-2.5 h-2.5 bg-emerald-500 border-2 border-white rounded-full"></div>
                    </div>
                    {sidebarOpen && (
                      <div className="flex-1 min-w-0">
                        <p className="text-[13px] font-semibold text-zinc-900 truncate">{session.user?.name || "Member"}</p>
                      </div>
                    )}
                    {sidebarOpen && (
                      <button onClick={() => signOut()} className="text-[11px] font-medium text-zinc-400 hover:text-zinc-700 transition-colors shrink-0">
                        로그아웃
                      </button>
                    )}
                  </div>
                </div>
              ) : (
                <div className="p-4">
                  {sidebarOpen ? (
                    <button onClick={() => signIn('google')} className="w-full h-11 glass-card hover:shadow-md transition-all text-zinc-900 rounded-2xl flex items-center gap-3 px-4 text-[13px] font-medium">
                      <svg className="w-4 h-4 shrink-0" viewBox="0 0 24 24"><path fill="#4285F4" d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z" /><path fill="#34A853" d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z" /><path fill="#FBBC05" d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z" /><path fill="#EA4335" d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z" /></svg>
                      Google로 로그인
                    </button>
                  ) : (
                    <button onClick={() => signIn()} className="w-9 h-9 rounded-xl bg-zinc-900 text-white flex items-center justify-center hover:scale-105 active:scale-95 transition-all shadow-lg mx-auto">
                      <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M11 16l-4-4m0 0l4-4m-4 4h14" /></svg>
                    </button>
                  )}
                </div>
              )}
            </div>
          </aside>

          {/* ── 우측 컬럼: 마켓패널(접힘/펼침) + 메인 컨텐츠 ── */}
          <div className="flex flex-col flex-1 min-h-0">

            {/* ── 마켓 패널 — GNB 오른쪽만 차지, 동적 너비 ── */}
            <div className={`overflow-hidden transition-all duration-500 premium-glass shrink-0 ${isExpanded ? 'max-h-[58vh] opacity-100' : 'max-h-0 opacity-0 pointer-events-none'}`}>
              <div className="px-5 py-4 overflow-y-auto max-h-[56vh] custom-scrollbar">

                {/* 헤더 */}
                <div className="flex items-center justify-between mb-5 pb-3">
                  <div className="flex items-center gap-3">
                    <span className="text-[11px] font-black text-zinc-900 uppercase tracking-widest">Market Terminal</span>
                    <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-emerald-500/10 ring-1 ring-emerald-500/20">
                      <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse"></span>
                      <span className="text-[10px] font-black text-emerald-600 uppercase tracking-widest">Live</span>
                    </div>
                    <span className="text-[11px] text-zinc-400 font-mono tabular-nums">{dataFetchedAt || '---'}</span>
                  </div>
                  <span className="text-[11px] text-zinc-400 font-mono">↺ {refreshCountdown}s</span>
                </div>

                {/* 시장 데이터 카드 — Premium Glass Grid */}
                <div className="grid grid-cols-2 sm:grid-cols-3 xl:grid-cols-4 2xl:grid-cols-5 gap-3">
                  {data.map((item) => (
                    <div key={item.symbol} className="p-4 rounded-[20px] glass-card hover:shadow-md transition-all group flex flex-col">
                      <div className="flex items-start justify-between mb-3 gap-2">
                        <span className="text-[11px] font-bold text-zinc-600 truncate min-w-0" title={item.name}>{item.name}</span>
                        <span className={`text-[10px] font-black px-1.5 py-0.5 rounded-md shrink-0 ${item.change_percent >= 0 ? 'bg-emerald-50 text-emerald-600' : 'bg-rose-50 text-rose-600'}`}>
                          {item.change_percent >= 0 ? '+' : ''}{item.change_percent.toFixed(2)}%
                        </span>
                      </div>
                      <div className="text-[17px] font-black text-zinc-950 tabular-nums leading-none tracking-tight">
                        {item.price.toLocaleString(undefined, { maximumFractionDigits: 2 })}
                      </div>
                      <div className="text-[10px] text-zinc-400 mt-2 font-mono uppercase font-bold truncate">{item.symbol}</div>
                    </div>
                  ))}
                </div>

                {/* US Liquidity Indicators (FRED) — Modernized */}
                {liquidityData.length > 0 && (
                  <div className="mt-8">
                    <div className="flex items-center gap-3 mb-4 pb-2">
                      <span className="text-[11px] font-black text-zinc-900 uppercase tracking-widest">Macro Liquidity</span>
                      <span className="text-[10px] font-black text-indigo-500 bg-indigo-50 px-2 py-0.5 rounded-md uppercase tracking-wider">FRED</span>
                    </div>
                    <div className="grid grid-cols-2 sm:grid-cols-3 xl:grid-cols-4 2xl:grid-cols-5 gap-3">
                      {liquidityData.map((item) => (
                        <div key={item.series} className="p-4 rounded-[20px] glass-card hover:shadow-md transition-all flex flex-col" title={item.desc}>
                          <div className="flex items-start justify-between mb-3 gap-2">
                            <span className="text-[11px] font-bold text-indigo-600 font-mono truncate min-w-0">{item.series}</span>
                            <span className="text-[10px] font-bold text-zinc-400 font-mono shrink-0">{item.date?.slice(0, 7)}</span>
                          </div>
                          <div className="flex items-end justify-between gap-2 mb-2">
                            <div className="text-[17px] font-black text-zinc-950 tabular-nums leading-none tracking-tight">
                              {item.value.toLocaleString(undefined, { minimumFractionDigits: item.unit === '%' ? 2 : 0, maximumFractionDigits: item.unit === '%' ? 2 : 0 })}
                              <span className="text-[11px] font-black text-zinc-400 ml-1 shrink-0">{item.unit === '%' ? '%' : item.unit}</span>
                            </div>
                            {item.mom_change !== undefined && item.mom_change !== null && (
                              <div className={`text-[11px] font-bold px-1.5 py-0.5 rounded-md ${item.mom_change >= 0
                                ? 'text-emerald-600 bg-emerald-50'
                                : 'text-rose-600 bg-rose-50'
                                }`}>
                                {item.mom_change >= 0 ? '+' : ''}{item.mom_change.toFixed(1)}%
                              </div>
                            )}
                          </div>
                          <div className="text-[10px] font-bold text-zinc-500 truncate max-w-full">{item.name}</div>
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
              {
                (analysis || analyzing) && (
                  <div
                    className="sticky top-0 z-50 flex items-center justify-between px-10 transition-all duration-300 overflow-hidden premium-glass"
                    style={{
                      height: headerScrolled ? '64px' : '0px',
                      opacity: headerScrolled ? 1 : 0,
                      pointerEvents: headerScrolled ? 'auto' : 'none',
                      borderBottom: headerScrolled ? '1px solid rgba(0,0,0,0.08)' : '1px solid transparent',
                      boxShadow: headerScrolled ? '0 10px 40px -10px rgba(0,0,0,0.05)' : 'none',
                    }}
                  >
                    <div className="flex items-center gap-5 min-w-0">
                      <div className="w-1.5 h-6 bg-zinc-950 rounded-full shrink-0"></div>
                      <span className="text-[15px] font-black text-zinc-950 truncate tracking-tight">{currentQuery || 'Secure Analysis Channel'}</span>
                      <span className="w-px h-4 bg-zinc-200 shrink-0 mx-1"></span>
                      <div className="flex items-center gap-2 px-2.5 py-1 rounded-lg bg-zinc-50 border border-zinc-100">
                        <span className={`w-2 h-2 rounded-full shrink-0 ${selectedModel === 'gemini' ? 'bg-blue-500' : selectedModel === 'gpt' ? 'bg-emerald-500' : 'bg-purple-500'}`}></span>
                        <span className="text-[11px] font-black text-zinc-900 uppercase tracking-wider shrink-0">{selectedModelInfo.name}</span>
                      </div>
                      {analyzing && (
                        <div className="flex items-center gap-2 px-3 py-1 bg-zinc-950 rounded-full shadow-lg shadow-zinc-200">
                          <div className="w-2 h-2 border-2 border-white/30 border-t-white rounded-full spin"></div>
                          <span className="text-[10px] font-black text-white uppercase tracking-widest">Processing</span>
                        </div>
                      )}
                    </div>
                    {!analyzing && analysis && (
                      <div className="flex items-center gap-2 shrink-0">
                        <button onClick={() => handleFeedback('helpful')}
                          className={`glass-button w-10 h-10 flex items-center justify-center transition-all ${currentRecord?.feedback === 'helpful' ? 'bg-emerald-500/90 text-white shadow-lg shadow-emerald-100' : 'text-zinc-600 hover:text-emerald-600'}`}
                          title="Helpful">👍</button>
                        <button onClick={() => handleFeedback('not_helpful')}
                          className={`glass-button w-10 h-10 flex items-center justify-center transition-all ${currentRecord?.feedback === 'not_helpful' ? 'bg-rose-500/90 text-white shadow-lg shadow-rose-100' : 'text-zinc-600 hover:text-rose-600'}`}
                          title="Needs Improvement">👎</button>
                        <button onClick={handleSave}
                          className={`glass-button w-10 h-10 flex items-center justify-center transition-all ${currentRecord?.saved ? 'bg-amber-400/90 text-white shadow-lg shadow-amber-100' : 'text-zinc-600 hover:text-amber-600'}`}
                          title={currentRecord?.saved ? 'Stored in Vault' : 'Store in Vault'}>★</button>
                        <div className="w-px h-6 bg-zinc-100 mx-2"></div>
                        <button onClick={() => { setAnalysis(null); setCurrentQuery(''); setActiveHistoryId(null); }}
                          className="glass-button w-10 h-10 flex items-center justify-center text-zinc-400 hover:text-zinc-900 transition-all"
                          title="Close Stream">
                          <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={3} d="M6 18L18 6M6 6l12 12" /></svg>
                        </button>
                      </div>
                    )}
                  </div>
                )
              }

              <div className="px-8 py-8 pb-36">
                <section className="min-h-[calc(100vh-120px)]">
                  {!analysis && !analyzing ? (
                    <div className="flex flex-col items-center justify-center min-h-[calc(100vh-180px)]">
                      <div className="w-14 h-14 rounded-2xl bg-emerald-50 border border-emerald-100 flex items-center justify-center mb-5">
                        <svg className="w-7 h-7 text-emerald-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                        </svg>
                      </div>
                      <p className="text-[13px] font-bold text-zinc-800 mb-1">질문을 입력하고 분석을 시작하세요</p>
                      <p className="text-[11px] text-zinc-400">IB 리포트 + 실시간 데이터 기반 · Zero Hallucination Policy</p>
                    </div>
                  ) : (
                    <div className="max-w-3xl mx-auto flex flex-col gap-12 w-full">
                      {currentRecord?.messages.map((msg) => (
                        <div key={msg.id} className="fade-in w-full">
                          {msg.role === 'user' ? (
                            <div className="flex items-center gap-4 mb-4">
                              <div className="w-1.5 h-6 bg-zinc-950 rounded-full shrink-0 shadow-sm shadow-zinc-200"></div>
                              <h2 className="text-[15px] font-black tracking-tight text-zinc-950 min-w-0 break-words flex-1 leading-snug">{msg.content}</h2>
                            </div>
                          ) : (
                            <div className="ml-5 pl-6 mb-16 relative">
                              {/* Depth marker instead of full border */}
                              <div className="absolute left-0 top-2 bottom-2 w-0.5 bg-gradient-to-b from-zinc-200/80 to-transparent rounded-full font-sans"></div>

                              <div className="flex items-center gap-3 mb-6 pb-4 flex-wrap">
                                <div className="flex items-center gap-2 px-3 py-1 rounded-lg bg-white/50 ring-1 ring-black/5 shadow-sm">
                                  <span className={`w-2 h-2 rounded-full ${currentRecord?.model === 'gemini' ? 'bg-blue-500' : currentRecord?.model === 'gpt' ? 'bg-emerald-500' : 'bg-purple-500'}`}></span>
                                  <span className="text-[11px] font-black text-zinc-900 uppercase tracking-widest">{currentRecord?.model}</span>
                                </div>
                                <span className="text-[10px] font-black text-zinc-300 uppercase tracking-widest">{currentRecord?.timestamp}</span>
                              </div>
                              <article className="prose prose-zinc max-w-none w-full
                                prose-h2:text-zinc-950 prose-h2:font-black prose-h2:text-[17px] prose-h2:mt-10 prose-h2:mb-5 prose-h2:tracking-tight prose-h2:py-2
                                prose-h3:text-zinc-500 prose-h3:text-[11px] prose-h3:font-black prose-h3:uppercase prose-h3:tracking-[0.2em] prose-h3:mt-8
                                prose-p:text-zinc-800 prose-p:text-[14px] prose-p:leading-[1.7] prose-p:mb-5 prose-p:font-medium
                                prose-strong:text-zinc-950 prose-strong:font-black
                                prose-li:text-zinc-800 prose-li:text-[14px] prose-li:my-1.5 prose-li:font-medium
                                prose-blockquote:ring-1 prose-blockquote:ring-black/5 prose-blockquote:bg-white/60 prose-blockquote:px-6 prose-blockquote:py-4 prose-blockquote:rounded-2xl prose-blockquote:not-italic prose-blockquote:text-zinc-900 prose-blockquote:text-[13px] prose-blockquote:font-bold prose-blockquote:my-6 prose-blockquote:shadow-sm
                                prose-table:w-full prose-table:text-[13px] prose-table:border-collapse prose-table:my-6 prose-table:rounded-2xl prose-table:overflow-hidden prose-table:ring-1 prose-table:ring-black/5 prose-table:shadow-sm
                                prose-th:bg-zinc-950/5 prose-th:text-zinc-900 prose-th:px-4 prose-th:py-2.5 prose-th:font-black prose-th:text-left
                                prose-td:px-4 prose-td:py-3 prose-td:text-zinc-800 prose-td:font-medium group-hover:bg-zinc-50/50">
                                <ReactMarkdown remarkPlugins={[remarkGfm]}>{msg.content}</ReactMarkdown>
                              </article>
                            </div>
                          )}
                        </div>
                      ))}
                      {analyzing && (
                        <div className="fade-in w-full">
                          <div className="flex items-center gap-4 mb-3">
                            <div className="w-1.5 h-6 bg-zinc-950 rounded-full shrink-0 shadow-lg shadow-zinc-200"></div>
                            <h2 className="text-[17px] font-black tracking-tight text-zinc-950 min-w-0 break-words flex-1 leading-snug">{currentQuery}</h2>
                          </div>
                          <div className="ml-5 pl-6 border-l-2 border-zinc-200/60">
                            <div className="flex items-center gap-4 mb-6 pb-4 border-b border-zinc-100/60 flex-wrap">
                              <div className="flex items-center gap-2 px-3 py-1 rounded-lg bg-zinc-50 border border-zinc-100">
                                <span className={`w-2 h-2 rounded-full ${selectedModel === 'gemini' ? 'bg-blue-500 shadow-[0_0_8px_rgba(59,130,246,0.5)]' : selectedModel === 'gpt' ? 'bg-emerald-500 shadow-[0_0_8px_rgba(16,185,129,0.5)]' : 'bg-purple-500 shadow-[0_0_8px_rgba(168,85,247,0.5)]'}`}></span>
                                <span className="text-[11px] font-black text-zinc-900 uppercase tracking-widest">{selectedModelInfo.fullName}</span>
                              </div>
                              <div className="flex items-center gap-2 px-4 py-1 bg-zinc-950 rounded-full shadow-lg shadow-zinc-200 animate-pulse">
                                <div className="w-2 h-2 border-2 border-white/30 border-t-white rounded-full spin"></div>
                                <span className="text-[10px] font-black text-white uppercase tracking-[0.2em]">Synthesizing Analysis</span>
                              </div>
                            </div>
                            <article className="prose prose-zinc max-w-none w-full
                              prose-h2:text-zinc-950 prose-h2:font-black prose-h2:text-[22px] prose-h2:mt-12 prose-h2:mb-6 prose-h2:tracking-tight prose-h2:py-2
                              prose-h3:text-zinc-500 prose-h3:text-[11px] prose-h3:font-black prose-h3:uppercase prose-h3:tracking-[0.3em] prose-h3:mt-10
                              prose-p:text-zinc-800 prose-p:text-[16px] prose-p:leading-[1.8] prose-p:mb-6 prose-p:font-medium
                              prose-strong:text-zinc-950 prose-strong:font-black
                              prose-li:text-zinc-800 prose-li:text-[15px] prose-li:my-2 prose-li:font-medium
                              prose-blockquote:border-l-4 prose-blockquote:border-zinc-900 prose-blockquote:bg-zinc-50/50 prose-blockquote:px-8 prose-blockquote:py-6 prose-blockquote:rounded-2xl prose-blockquote:not-italic prose-blockquote:text-zinc-900 prose-blockquote:text-[15px] prose-blockquote:font-bold prose-blockquote:my-8 prose-blockquote:shadow-sm
                              prose-table:w-full prose-table:text-[13px] prose-table:border-collapse prose-table:my-8 prose-table:rounded-2xl prose-table:overflow-hidden
                              prose-th:bg-zinc-950 prose-th:text-white prose-th:px-5 prose-th:py-3 prose-th:font-black prose-th:text-left prose-th:uppercase prose-th:tracking-widest
                              prose-td:px-5 prose-td:py-4 prose-td:border-b prose-td:border-zinc-100 prose-td:text-zinc-900 prose-td:font-bold">
                              <ReactMarkdown remarkPlugins={[remarkGfm]}>{analysis ?? ""}</ReactMarkdown>
                              <span className="cursor-blink"></span>
                            </article>
                          </div>
                        </div>
                      )}

                      {!analyzing && currentRecord && (
                        <div className="flex items-center gap-2 mt-8 ml-4 shrink-0">
                          <button onClick={() => handleFeedback('helpful')}
                            className={`glass-button w-9 h-9 flex items-center justify-center transition-all text-base ${currentRecord?.feedback === 'helpful' ? 'bg-emerald-500/20 text-emerald-600' : 'text-zinc-600 hover:text-emerald-600'}`}
                            title="도움이 됐어요">👍</button>
                          <button onClick={() => handleFeedback('not_helpful')}
                            className={`glass-button w-9 h-9 flex items-center justify-center transition-all text-base ${currentRecord?.feedback === 'not_helpful' ? 'bg-rose-500/20 text-rose-600' : 'text-zinc-600 hover:text-rose-600'}`}
                            title="개선이 필요해요">👎</button>
                          <div className="w-px h-6 bg-gradient-to-b from-transparent via-zinc-200 to-transparent mx-2"></div>
                          <button onClick={handleSave}
                            className={`glass-button w-9 h-9 flex items-center justify-center transition-all ${currentRecord?.saved ? 'bg-amber-400/20 text-amber-600' : 'text-zinc-600 hover:text-amber-600'}`}
                            title={currentRecord?.saved ? '저장됨' : '저장하기'}>★</button>
                          <button onClick={() => { setAnalysis(null); setCurrentQuery(''); setActiveHistoryId(null); }}
                            className="glass-button w-9 h-9 flex items-center justify-center text-zinc-600 hover:text-zinc-900 transition-all text-sm"
                            title="초기화">✕</button>
                        </div>
                      )}
                    </div>
                  )}
                </section>

                {/* ── 매크로 변동성 경보 (Macro Alerts) ── */}
                {macroAlerts.length > 0 && (
                  <div className="mt-12 pt-8 border-t border-zinc-100/60">
                    <div className="flex items-center justify-between mb-4">
                      <div className="flex items-center gap-2">
                        <div className="w-2 h-2 rounded-full bg-rose-500 shadow-[0_0_8px_rgba(244,63,94,0.6)] animate-pulse"></div>
                        <span className="text-[11px] font-black text-zinc-900 uppercase tracking-[0.2em]">Macro Alerts</span>
                        <span className="text-[10px] font-mono text-zinc-500 bg-zinc-100 px-1.5 py-0.5 rounded-md">{macroAlerts.length}</span>
                      </div>
                    </div>
                    <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
                      {macroAlerts.map((alert: any) => (
                        <div key={alert.id} className="glass-panel p-5 hover-lift border-l-[3px] border-rose-400">
                          <div className="flex items-start justify-between mb-3 gap-4">
                            <h4 className="text-[13px] font-black text-zinc-900 leading-snug">{alert.title}</h4>
                            <span className="text-[10px] font-mono font-bold text-zinc-400 shrink-0 bg-zinc-50 px-2 py-1 rounded-md">{new Date(alert.created_at).toLocaleDateString('ko-KR', { month: 'short', day: 'numeric' })}</span>
                          </div>
                          <p className="text-[12px] font-medium text-zinc-600 mb-4 leading-relaxed">{alert.message}</p>
                          <div className="flex flex-col gap-2 bg-zinc-50/50 p-3 rounded-xl ring-1 ring-zinc-200/50">
                            <div className="flex items-start gap-3">
                              <span className="text-[10px] font-black text-rose-500 uppercase tracking-widest w-12 shrink-0 mt-0.5">Impact</span>
                              <span className="text-[11px] font-bold text-zinc-800 leading-snug">{alert.affected_sectors}</span>
                            </div>
                            <div className="flex items-start gap-3">
                              <span className="text-[10px] font-black text-emerald-600 uppercase tracking-widest w-12 shrink-0 mt-0.5">Action</span>
                              <span className="text-[11px] font-bold text-zinc-800 leading-snug">{alert.recommended_actions}</span>
                            </div>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {/* ── 이슈 이력 로그 ── */}
                {issueLog.length > 0 && (
                  <div className="mt-12 pt-6 border-t border-zinc-100">
                    <div className="flex items-center justify-between mb-3">
                      <div className="flex items-center gap-2">
                        <span className="text-[11px] font-bold text-zinc-400 uppercase tracking-[0.3em]">시스템 이슈 이력</span>
                        <span className="text-[10px] font-mono text-zinc-300 bg-zinc-100 px-1.5 py-0.5 rounded-md">{issueLog.length}</span>
                      </div>
                      <button onClick={() => setIssueLog([])}
                        className="text-[10px] text-zinc-300 hover:text-rose-400 transition-colors">
                        지우기
                      </button>
                    </div>
                    <div className="space-y-1">
                      {issueLog.map((item) => (
                        <div key={item.id} className="flex items-start gap-2.5 py-1.5 px-3 glass-inner">
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

        {/* ── Floating Input Bar (Premium Glass) ── */}
        <div className="fixed bottom-0 z-[200] pb-8 pt-10"
          style={{
            left: `${sidebarWidth}px`,
            right: 0,
            background: 'linear-gradient(to top, rgba(247,249,252,1) 40%, rgba(247,249,252,0.9) 65%, rgba(247,249,252,0))',
            transition: 'left 0.3s ease',
          }}>
          <div className="max-w-3xl mx-auto px-6">
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
              <div className="mb-3 rounded-2xl overflow-hidden py-2 fade-in glass-card shadow-xl shadow-zinc-200/30" ref={modelDropdownRef}>
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

            <div className="premium-glass rounded-[24px] p-2 shadow-xl shadow-zinc-200/40 relative">
              <div className="px-4 pt-3.5 pb-2">
                <textarea
                  ref={queryRef}
                  value={query}
                  onChange={(e) => { if (!analyzing) setQuery(e.target.value); }}
                  onFocus={() => { if (!analyzing) setShowSuggestions(true); }}
                  onBlur={() => setTimeout(() => setShowSuggestions(false), 200)}
                  onKeyDown={(e) => { if (!analyzing && e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); handleAnalyze(); } }}
                  disabled={analyzing}
                  placeholder={analyzing ? '분석 중...' : '매크로 분석 주제를 입력하세요...'}
                  rows={1}
                  className="w-full text-[13px] font-medium text-zinc-900 placeholder-zinc-400 resize-none bg-transparent outline-none leading-relaxed"
                />
              </div>

              <div className="flex items-center justify-between px-2 pb-2 mt-2">
                <div className="flex gap-2">
                  <button
                    onClick={() => { if (!analyzing) setShowModelDropdown(!showModelDropdown); }}
                    className="h-8 flex items-center gap-2 px-3.5 rounded-xl glass-inner text-[11px] font-semibold text-zinc-600 hover:text-zinc-900 transition-all"
                  >
                    <span className={`w-1.5 h-1.5 rounded-full ${selectedModel === 'gemini' ? 'bg-blue-400' : selectedModel === 'gpt' ? 'bg-emerald-400' : 'bg-purple-400'}`}></span>
                    {selectedModelInfo.name}
                  </button>
                  <div className="hidden sm:flex items-center gap-1.5 px-3 py-1.5 rounded-xl bg-emerald-50/60">
                    <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse"></span>
                    <span className="text-[10px] font-semibold text-emerald-700">AI Grounded</span>
                  </div>
                </div>

                <div className="flex items-center gap-2">
                  {analyzing ? (
                    <button onClick={handleStop} className="h-9 px-5 rounded-2xl bg-rose-50/80 text-rose-600 font-semibold text-[12px] hover:bg-rose-100 transition-all flex items-center gap-2">
                      <div className="w-2 h-2 bg-rose-500 rounded-sm"></div>
                      중지
                    </button>
                  ) : (
                    <button onClick={handleAnalyze} className="h-9 px-5 rounded-2xl bg-emerald-500 hover:bg-emerald-600 text-white font-semibold text-[12px] transition-all shadow-lg shadow-emerald-200/40 flex items-center gap-2 group/btn active:scale-[0.97]">
                      분석
                      <svg className="w-4 h-4 transition-transform group-hover/btn:translate-x-0.5" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M13 7l5 5m0 0l-5 5m5-5H6" /></svg>
                    </button>
                  )}
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* ── 토스트 알림 스택 (Premium Glass) ── */}
        {
          toasts.length > 0 && (
            <div className="fixed bottom-32 right-8 z-[300] flex flex-col gap-3 items-end">
              {toasts.map((toast) => (
                <div key={toast.id} className="toast-in premium-glass flex items-center gap-4 px-6 py-4 rounded-[22px] shadow-2xl shadow-zinc-300/30 max-w-sm group">
                  <div className={`w-8 h-8 rounded-full flex items-center justify-center shrink-0 ${toast.type === 'warning' ? 'bg-rose-50 text-rose-500' : 'bg-emerald-50 text-emerald-500 font-black'}`}>
                    {toast.type === 'warning' ? '!' : '✓'}
                  </div>
                  <p className={`text-[13px] font-bold leading-tight flex-1 ${toast.type === 'warning' ? 'text-rose-900' : 'text-zinc-900'}`}>
                    {toast.message.replace(/^[⚠️⚡\s]+/, '')}
                  </p>
                  <button onClick={() => setToasts(prev => prev.filter(t => t.id !== toast.id))}
                    className="w-6 h-6 flex items-center justify-center rounded-lg hover:bg-zinc-100 transition-colors text-zinc-300 hover:text-zinc-600">
                    <svg className="w-3 h-3" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={3} d="M6 18L18 6M6 6l12 12" /></svg>
                  </button>
                </div>
              ))}
            </div>
          )
        }
        {/* ── Unified Portfolio Management Modal ── */}
        {
          isMyPortfolioModalOpen && (
            <div
              onClick={() => setIsMyPortfolioModalOpen(false)}
              className="fixed inset-0 z-[500] flex items-center justify-center p-4 bg-zinc-950/30 backdrop-blur-lg animate-in fade-in duration-300"
            >
              <div
                onClick={(e) => e.stopPropagation()}
                className="bg-white/90 backdrop-blur-2xl ring-1 ring-black/5 w-full max-w-4xl rounded-[32px] shadow-2xl shadow-zinc-300/30 overflow-hidden flex flex-col max-h-[90vh] transition-all duration-500 ease-out isolate"
              >
                {/* Header - Shared but Contextual */}
                <div className="px-8 py-6 mb-2 flex items-center justify-between shrink-0 shadow-[0_4px_24px_rgba(0,0,0,0.02)] z-10 relative">
                  <div>
                    <h2 className="text-xl font-black text-zinc-900 tracking-tight">
                      {portfolioModalSubView === 'list' ? '내 포트폴리오' : '포트폴리오 업데이트'}
                    </h2>
                    <p className="text-[13px] text-zinc-400 mt-1">
                      {portfolioModalSubView === 'list' ? '자산 배분 및 수익 현황을 확인하세요.' : '자산을 자동으로 인식하거나 수동으로 관리하세요.'}
                    </p>
                  </div>
                  <div className="flex items-center gap-4">
                    {/* Currency Toggle */}
                    {portfolioModalSubView === 'list' && portfolioItems.length > 0 && (
                      <div className="flex items-center gap-2">
                        <span className="text-[12px] font-medium text-zinc-500">표시 통화</span>
                        <div className="relative">
                          <button
                            onClick={() => setPortfolioCurrency(portfolioCurrency === 'USD' ? 'KRW' : 'USD')}
                            className={`relative w-16 h-8 rounded-full transition-all duration-300 ${portfolioCurrency === 'USD'
                              ? 'bg-blue-500'
                              : 'bg-emerald-500'
                              }`}
                          >
                            <div className={`absolute top-1 w-6 h-6 bg-white rounded-full transition-all duration-300 flex items-center justify-center ${portfolioCurrency === 'USD' ? 'left-1' : 'left-9'
                              }`}>
                              <span className="text-[10px] font-black">
                                {portfolioCurrency === 'USD' ? '$' : '₩'}
                              </span>
                            </div>
                          </button>
                        </div>
                        <span className="text-[11px] font-mono text-zinc-400">
                          {portfolioCurrency === 'USD' ? 'USD' : 'KRW'}
                        </span>
                      </div>
                    )}
                    <button
                      onClick={() => setIsMyPortfolioModalOpen(false)}
                      className="w-10 h-10 flex items-center justify-center rounded-full hover:bg-zinc-50 transition-colors text-zinc-300 hover:text-zinc-900 text-xl overflow-hidden"
                    >
                      ✕
                    </button>
                  </div>
                </div>

                {/* Content Area */}
                <div className="flex-1 overflow-y-auto p-8 custom-scrollbar bg-white/40">
                  {portfolioModalSubView === 'list' ? (
                    /* --- View 1: List & Dashboard --- */
                    portfolioItems.length === 0 ? (
                      <div className="text-center py-20">
                        <div className="w-16 h-16 bg-zinc-50 rounded-3xl flex items-center justify-center mx-auto mb-4">
                          <svg className="w-8 h-8 text-zinc-200" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                          </svg>
                        </div>
                        <p className="text-sm font-bold text-zinc-400">등록된 자산이 없습니다.</p>
                        <button
                          onClick={() => setPortfolioModalSubView('edit')}
                          className="mt-4 text-sm font-black text-zinc-900 hover:underline flex items-center gap-2 mx-auto"
                        >
                          포트폴리오 시작하기 <span className="text-lg">→</span>
                        </button>
                      </div>
                    ) : (
                      <div className="animate-in fade-in slide-in-from-bottom-2 duration-500">
                        {/* Summary Cards */}
                        <div className="grid grid-cols-2 gap-4 mb-10">
                          <div className="bg-zinc-900 rounded-[28px] p-8 text-white shadow-2xl shadow-zinc-200/50 relative overflow-hidden group">
                            <div className="absolute -right-4 -bottom-4 w-32 h-32 bg-white/5 rounded-full blur-3xl group-hover:bg-white/10 transition-all duration-700"></div>
                            <p className="text-[11px] font-black text-zinc-500 uppercase tracking-[0.1em] mb-2">Total Evaluation</p>
                            <div className="flex items-baseline gap-2">
                              <span className="text-4xl font-black tracking-tight">
                                {formatCurrency(portfolioItems.reduce((acc, i) => acc + (i.quantity * (i.current_price || 0)), 0)).replace(/[$₩]/, '')}
                              </span>
                              <span className="text-xs font-bold text-zinc-600 uppercase">{portfolioCurrency}</span>
                            </div>
                          </div>

                          <div className="bg-white rounded-[28px] p-8 ring-1 ring-black/5 shadow-sm relative overflow-hidden group">
                            <div className={`absolute -right-4 -bottom-4 w-32 h-32 rounded-full blur-3xl opacity-20 transition-all duration-700 ${portfolioItems.reduce((acc, i) => acc + (i.profit_krw || 0), 0) >= 0 ? 'bg-emerald-400' : 'bg-rose-400'}`}></div>
                            <p className="text-[11px] font-black text-zinc-400 uppercase tracking-[0.1em] mb-2">Cumulative Return</p>
                            <div className="flex items-baseline gap-2">
                              <span className={`text-4xl font-black tracking-tight ${portfolioItems.reduce((acc, i) => acc + (i.profit_krw || 0), 0) >= 0 ? 'text-emerald-500' : 'text-rose-500'}`}>
                                {portfolioItems.reduce((acc, i) => acc + (i.profit_krw || 0), 0) >= 0 ? '+' : ''}
                                {formatCurrency(portfolioItems.reduce((acc, i) => acc + (i.profit_krw || 0), 0), false).replace(/[$₩]/, '')}
                              </span>
                              <span className="text-xs font-bold text-zinc-900 uppercase">{portfolioCurrency}</span>
                            </div>
                          </div>
                        </div>

                        {/* ── Macro Stress Test Dashboard ── */}
                        <div className="mb-8">
                          <div className="flex items-center justify-between mb-4">
                            <div className="flex items-center gap-3 px-1">
                              <div className="w-1.5 h-1.5 rounded-full bg-rose-500 shadow-sm shadow-rose-300"></div>
                              <h3 className="text-[13px] font-black text-zinc-900 uppercase tracking-widest">Macro Stress Test</h3>
                            </div>
                            <button
                              onClick={handleMacroStressTest}
                              disabled={isStressTesting}
                              className="glass-button px-4 py-2 text-[11px] font-black text-zinc-700 hover:text-rose-600 uppercase tracking-widest flex items-center gap-2"
                            >
                              {isStressTesting ? (
                                <>
                                  <svg className="animate-spin h-3 w-3" viewBox="0 0 24 24"><circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none"></circle><path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"></path></svg>
                                  Analyzing...
                                </>
                              ) : (
                                <>
                                  <span>⚡ AI 진단 실행</span>
                                </>
                              )}
                            </button>
                          </div>

                          {stressTestResult && (
                            <div className="glass-panel p-6 animate-in fade-in slide-in-from-top-2">
                              <div className="flex flex-col md:flex-row items-start justify-between gap-6">
                                <div className="space-y-4 flex-1">
                                  <div>
                                    <p className="text-[11px] font-black text-zinc-400 uppercase tracking-widest mb-1">Max Drawdown Estimate</p>
                                    <div className="flex items-end gap-2">
                                      <p className="text-4xl font-black text-rose-500 tracking-tight leading-none">{stressTestResult.max_drawdown_estimate}</p>
                                      <span className={`text-[10px] font-bold px-2 py-0.5 rounded uppercase ${stressTestResult.risk_level === 'High' ? 'bg-rose-100 text-rose-700' :
                                        stressTestResult.risk_level === 'Medium' ? 'bg-amber-100 text-amber-700' :
                                          'bg-emerald-100 text-emerald-700'
                                        }`}>
                                        {stressTestResult.risk_level} RISK
                                      </span>
                                    </div>
                                  </div>
                                  <div className="space-y-1">
                                    <p className="text-[11px] font-black text-zinc-400 uppercase tracking-widest">Vulnerable Sectors</p>
                                    <div className="flex gap-2 flex-wrap">
                                      {Array.isArray(stressTestResult.vulnerable_sectors) && stressTestResult.vulnerable_sectors.map((sec: string, i: number) => (
                                        <span key={i} className="text-[10px] font-bold bg-rose-50 text-rose-600 px-2.5 py-1 rounded-md">{sec}</span>
                                      ))}
                                    </div>
                                  </div>
                                </div>
                                <div className="hidden md:block w-px h-28 bg-gradient-to-b from-transparent via-zinc-200 to-transparent shrink-0"></div>
                                <div className="space-y-4 flex-[1.5] w-full">
                                  <div>
                                    <p className="text-[11px] font-black text-zinc-400 uppercase tracking-widest mb-1">Analysis & Reasoning</p>
                                    <p className="text-[13px] font-medium text-zinc-700 leading-relaxed">{stressTestResult.analysis_reasoning}</p>
                                  </div>
                                  <div className="bg-zinc-50/80 rounded-xl p-3.5 ring-1 ring-zinc-200/50 hover-lift">
                                    <p className="text-[11px] font-black text-emerald-600 uppercase tracking-widest mb-1.5 flex items-center gap-1.5"><span className="text-sm">💡</span> Action Plan</p>
                                    <p className="text-[12px] font-bold text-zinc-800 leading-snug">{stressTestResult.rebalancing_suggestion}</p>
                                  </div>
                                </div>
                              </div>
                            </div>
                          )}
                        </div>

                        <div className="flex items-center gap-3 mb-6 px-1">
                          <div className="w-1.5 h-1.5 rounded-full bg-zinc-900 shadow-sm shadow-zinc-300"></div>
                          <h3 className="text-[13px] font-black text-zinc-900 uppercase tracking-widest">Holdings Details</h3>
                          <div className="h-px flex-1 bg-gradient-to-r from-zinc-200 to-transparent"></div>
                          <span className="text-[11px] font-bold text-zinc-400 uppercase tracking-wider">{portfolioItems.length} Assets Registered</span>
                        </div>

                        <div className="glass-card rounded-[24px] overflow-hidden">
                          <table className="w-full text-left border-collapse table-fixed">
                            <thead className="bg-zinc-50/60 shadow-[inset_0_-1px_0_rgba(0,0,0,0.05)]">
                              <tr>
                                <th className="w-[30%] px-6 py-4 text-[11px] font-black text-zinc-500 uppercase tracking-widest">Asset</th>
                                <th className="w-[12%] px-6 py-4 text-[11px] font-black text-zinc-400 uppercase tracking-wider text-center">Ticker</th>
                                <th className="w-[14%] px-6 py-4 text-[11px] font-black text-zinc-400 uppercase tracking-wider text-right">Holdings</th>
                                <th className="w-[14%] px-6 py-4 text-[11px] font-black text-zinc-400 uppercase tracking-wider text-right">Price</th>
                                <th className="w-[15%] px-6 py-4 text-[11px] font-black text-zinc-400 uppercase tracking-wider text-right">Value</th>
                                <th className="w-[15%] px-6 py-4 text-[11px] font-black text-zinc-400 uppercase tracking-wider text-right">Profit</th>
                              </tr>
                            </thead>
                            <tbody className="divide-y divide-zinc-50">
                              {portfolioItems.map((item, idx) => {
                                const isKrx = /^\d+/.test(item.ticker);
                                const currentPrice = item.current_price || 0;
                                const currentValue = item.quantity * currentPrice;
                                const profitAmount = item.profit_krw || 0;

                                return (
                                  <tr key={idx} className="hover:bg-zinc-50/50 transition-all group">
                                    <td className="px-6 py-4">
                                      <div className="flex flex-col min-w-0">
                                        <span className="text-[14px] font-black text-zinc-900 group-hover:text-black transition-colors truncate max-w-[120px] sm:max-w-[200px]" title={item.name}>{item.name || '-'}</span>
                                        {isKrx && <span className="text-[10px] font-bold text-emerald-500 uppercase tracking-tighter mt-0.5">KRX</span>}
                                      </div>
                                    </td>
                                    <td className="px-6 py-4 text-center">
                                      <span className="text-[10px] font-black font-mono text-zinc-400 bg-zinc-50 px-2 py-1 rounded-lg uppercase">{item.ticker}</span>
                                    </td>
                                    <td className="px-6 py-4 text-right">
                                      <span className="text-[13px] font-bold text-zinc-600 font-mono">{item.quantity.toLocaleString(undefined, { maximumFractionDigits: 3 })}</span>
                                    </td>
                                    <td className="px-6 py-4 text-right">
                                      <span className="text-[13px] font-bold text-zinc-400 font-mono">
                                        {formatCurrency(currentPrice, !isKrx)}
                                      </span>
                                    </td>
                                    <td className="px-6 py-4 text-right">
                                      <span className="text-[13px] font-black text-zinc-900 font-mono">
                                        {formatCurrency(currentValue, !isKrx)}
                                      </span>
                                    </td>
                                    <td className="px-6 py-4 text-right">
                                      <span className={`text-[11px] font-black px-3 py-1.5 rounded-full inline-block ${profitAmount >= 0 ? 'bg-emerald-50 text-emerald-600' : 'bg-rose-50 text-rose-600'}`}>
                                        {profitAmount >= 0 ? '+' : ''}
                                        {formatCurrency(profitAmount, false)}
                                      </span>
                                    </td>
                                  </tr>
                                );
                              })}
                            </tbody>
                          </table>
                        </div>
                      </div>
                    )
                  ) : (
                    /* --- View 2: Update & Edit Form --- */
                    <div className="animate-in fade-in slide-in-from-right-4 duration-500">
                      <div className="mb-10">
                        <label className="block text-[11px] font-black text-zinc-400 uppercase tracking-widest mb-4">이미지 분석 (Smart AI Link)</label>
                        <div className="relative group">
                          <input
                            type="file"
                            accept="image/*"
                            onChange={async (e) => {
                              const file = e.target.files?.[0];
                              if (!file) return;
                              setIsParsingImage(true);
                              const formData = new FormData();
                              formData.append('file', file);
                              try {
                                const res = await fetch('/api/portfolio/parse-image', { method: 'POST', body: formData });
                                const json = await res.json();
                                if (json.data && json.data.length > 0) {
                                  setPortfolioItems(json.data);
                                  pushToast(`${json.data.length}개의 자산을 찾았습니다.`, 'info');
                                } else if (json.data && json.data.length === 0) {
                                  pushToast("자산을 찾지 못했습니다. 직접 입력해 주세요.", 'warning');
                                }
                              } catch {
                                pushToast("이미지 전송 중 오류가 발생했습니다.", 'warning');
                              } finally {
                                setIsParsingImage(false);
                              }
                            }}
                            className="absolute inset-0 w-full h-full opacity-0 cursor-pointer z-10"
                          />
                          <div className={`h-40 border-2 border-dashed rounded-[28px] flex flex-col items-center justify-center transition-all ${isParsingImage ? 'border-zinc-200/50 bg-zinc-50/50' : 'border-zinc-200/40 hover:border-zinc-300 hover:bg-zinc-50/50'}`}>
                            {isParsingImage ? (
                              <div className="flex flex-col items-center">
                                <div className="w-8 h-8 border-2 border-zinc-200 border-t-zinc-900 rounded-full animate-spin mb-4"></div>
                                <p className="text-sm font-black text-zinc-900">AI 분석 중...</p>
                              </div>
                            ) : (
                              <div className="flex flex-col items-center">
                                <div className="p-4 bg-zinc-50 rounded-2xl mb-3 group-hover:scale-110 transition-transform">
                                  <svg className="w-6 h-6 text-zinc-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
                                  </svg>
                                </div>
                                <p className="text-sm font-black text-zinc-900">파일 업로드</p>
                                <p className="text-xs text-zinc-400 mt-1">드래그하거나 클릭하여 추가하세요.</p>
                              </div>
                            )}
                          </div>
                        </div>
                      </div>

                      <div className="flex items-center justify-between mb-4">
                        <label className="text-[11px] font-black text-zinc-400 uppercase tracking-widest">직접 관리</label>
                        <button
                          onClick={() => setPortfolioItems([...portfolioItems, { ticker: '', name: '', quantity: 0, buy_price: 0, current_price: 0, profit_krw: 0 }])}
                          className="text-[11px] font-black text-zinc-900 bg-zinc-50 px-3 py-1.5 rounded-lg hover:bg-zinc-100 transition-colors"
                        >
                          + 자산 추가
                        </button>
                      </div>

                      <div className="glass-card rounded-[24px] overflow-hidden">
                        <table className="w-full text-left border-collapse table-fixed">
                          <thead className="bg-zinc-50/60 shadow-[inset_0_-1px_0_rgba(0,0,0,0.05)]">
                            <tr>
                              <th className="px-6 py-4 text-[11px] font-black text-zinc-500 uppercase tracking-widest">티커</th>
                              <th className="px-6 py-4 text-[11px] font-black text-zinc-500 uppercase tracking-widest">자산명</th>
                              <th className="px-6 py-4 text-[11px] font-black text-zinc-500 uppercase tracking-widest text-right">보유 수량</th>
                              <th className="px-6 py-4 text-[11px] font-black text-zinc-500 uppercase tracking-widest text-right w-20"></th>
                            </tr>
                          </thead>
                          <tbody>
                            {portfolioItems.map((item, idx) => (
                              <tr key={idx} className="hover:bg-zinc-50/50 transition-colors group">
                                <td className="px-6 py-3">
                                  <input
                                    className="w-full bg-transparent text-[13px] font-bold text-zinc-900 uppercase focus:outline-none focus:text-emerald-600 transition-colors font-mono"
                                    value={item.ticker}
                                    placeholder="AAPL"
                                    onChange={(e) => {
                                      const newItems = [...portfolioItems];
                                      newItems[idx].ticker = e.target.value.toUpperCase();
                                      setPortfolioItems(newItems);
                                    }}
                                  />
                                </td>
                                <td className="px-6 py-3">
                                  <input
                                    className="w-full bg-transparent text-[13px] font-bold text-zinc-600 focus:outline-none"
                                    value={item.name}
                                    placeholder="애플"
                                    onChange={(e) => {
                                      const newItems = [...portfolioItems];
                                      newItems[idx].name = e.target.value;
                                      setPortfolioItems(newItems);
                                    }}
                                  />
                                </td>
                                <td className="px-6 py-3 text-right">
                                  <input
                                    type="number"
                                    className="w-full bg-transparent text-[13px] font-bold text-zinc-900 focus:outline-none text-right font-mono"
                                    value={item.quantity}
                                    onChange={(e) => {
                                      const newItems = [...portfolioItems];
                                      newItems[idx].quantity = parseFloat(e.target.value);
                                      setPortfolioItems(newItems);
                                    }}
                                  />
                                </td>
                                <td className="px-6 py-3 text-right">
                                  <button
                                    onClick={() => setPortfolioItems(portfolioItems.filter((_, i) => i !== idx))}
                                    className="p-2 text-rose-300 hover:text-rose-500 hover:bg-rose-50 rounded-xl transition-all"
                                  >
                                    <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                                    </svg>
                                  </button>
                                </td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    </div>
                  )}
                </div>

                {/* Footer - shared with view switching logic */}
                <div className="px-8 py-6 bg-white/30 flex items-center justify-between shrink-0 shadow-[0_-4px_24px_rgba(0,0,0,0.02)] z-10 relative">
                  {portfolioModalSubView === 'list' ? (
                    <>
                      <button
                        onClick={() => setPortfolioModalSubView('edit')}
                        className="flex items-center gap-2.5 px-6 h-12 rounded-[20px] bg-white border border-zinc-200 hover:border-zinc-900 hover:text-zinc-900 text-zinc-500 text-[13px] font-black transition-all shadow-sm"
                      >
                        <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12" />
                        </svg>
                        업데이트하기
                      </button>
                      <div className="flex items-center gap-4">
                        <div className="flex flex-col items-end">
                          <div className="flex items-center gap-2">
                            <div className="w-2 h-2 rounded-full bg-emerald-500 shadow-[0_0_8px_rgba(16,185,129,0.5)] animate-pulse"></div>
                            <span className="text-[11px] font-black text-zinc-900">LIVE BRIDGE</span>
                          </div>
                          <span className="text-[10px] text-zinc-400 font-bold uppercase tracking-widest mt-0.5">Real-time Data Active</span>
                        </div>
                        <button
                          onClick={() => setIsMyPortfolioModalOpen(false)}
                          className="px-8 h-12 rounded-[20px] text-sm font-black text-white bg-zinc-900 hover:bg-zinc-800 transition-all shadow-xl shadow-zinc-200"
                        >
                          확인 완료
                        </button>
                      </div>
                    </>
                  ) : (
                    <>
                      <button
                        onClick={() => setPortfolioModalSubView('list')}
                        className="px-6 h-12 rounded-[20px] text-sm font-black text-zinc-400 hover:text-zinc-900 transition-colors"
                      >
                        이전으로
                      </button>
                      <button
                        onClick={async () => {
                          if (!session?.user?.email) { pushToast("로그인이 필요합니다.", 'warning'); return; }
                          try {
                            const res = await fetch('/api/portfolio/update', {
                              method: 'POST',
                              headers: { 'Content-Type': 'application/json' },
                              body: JSON.stringify({ email: session.user.email, items: portfolioItems })
                            });
                            const json = await res.json();
                            if (json.status === 'success') {
                              pushToast("포트폴리오가 업데이트되었습니다.", 'info');
                              fetchDbPortfolio();
                              setPortfolioModalSubView('list');
                            }
                          } catch { pushToast("저장 중 오류가 발생했습니다.", 'warning'); }
                        }}
                        disabled={portfolioItems.length === 0}
                        className="px-10 h-12 rounded-[20px] text-sm font-black text-white bg-zinc-900 hover:bg-zinc-800 disabled:opacity-20 transition-all shadow-xl shadow-zinc-200"
                      >
                        변경사항 저장
                      </button>
                    </>
                  )}
                </div>
              </div>
            </div>
          )
        }
      </div>
    </div>
  );
}
