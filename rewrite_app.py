import re

with open('app/page.tsx', 'r', encoding='utf-8') as f:
    code = f.read()

# 1. Add ChatMessage interface
if 'interface ChatMessage' not in code:
    code = code.replace('interface AnalysisRecord {', '''interface ChatMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
}

interface AnalysisRecord {''')

# 2. Modify AnalysisRecord
code = re.sub(r'interface AnalysisRecord \{[\s\S]*?saved: boolean;\n\}',
'''interface AnalysisRecord {
  id: string;
  title: string;
  model: string;
  messages: ChatMessage[];
  timestamp: string;
  feedback?: 'helpful' | 'not_helpful' | null;
  saved: boolean;
}''', code)

# 3. Update handleAnalyze
handle_analyze_orig = '''
      const response = await fetch('/api/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: userQuery, model: selectedModel }),
        signal: controller.signal,
      });
'''

handle_analyze_new = '''
      const currentSession = history.find(r => r.id === activeHistoryId);
      const apiHistory = currentSession ? currentSession.messages : [];

      const response = await fetch('/api/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: userQuery, model: selectedModel, history: apiHistory }),
        signal: controller.signal,
      });
'''
code = code.replace(handle_analyze_orig, handle_analyze_new)

# 4. Update saveHistory block in handleAnalyze
save_history_orig = '''

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
'''

save_history_new = '''

      if (finalContent) {
        setQuery(''); // 입력창 초기화
        let updatedHistory = [...history];
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
'''
code = code.replace(save_history_orig, save_history_new)

# 5. Fix loadHistoryItem
load_history_orig = '''  const loadHistoryItem = (record: AnalysisRecord) => {
    setAnalysis(record.content);
    setCurrentQuery(record.query);
    setActiveHistoryId(record.id);
    setQuery(record.query);
  };'''

load_history_new = '''  const loadHistoryItem = (record: AnalysisRecord) => {
    setAnalysis(null);
    setCurrentQuery('');
    setActiveHistoryId(record.id);
    setQuery('');
  };'''
code = code.replace(load_history_orig, load_history_new)

with open('app/page.tsx', 'w', encoding='utf-8') as f:
    f.write(code)

print("Part 1 complete.")
