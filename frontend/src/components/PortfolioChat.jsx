import React, { useState, useRef, useEffect } from 'react';
import ChatMessage from './ChatMessage';
import ThinkingTrace from './ThinkingTrace';
import ModeToggle from './ModeToggle';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

// Client-side quick intent detector to smoothly auto-switch the UI pill live
function detectModeFromInput(text) {
  if (!text) return 'rag';
  const q = text.toLowerCase();

  const directPatterns = [
    'who are you', 'introduce yourself', 'tell me about yourself', 'your name',
    'full name', 'college', 'ksrct', 'degree', 'department', 'studying',
    'where are you from', 'where do you live', 'graduation', 'hello', 'hi', 'hey'
  ];

  const ragPatterns = [
    'project', 'built', 'develop', 'deepfake', 'kllm', 'signbridge',
    'jarvis', 'youtube', 'theatre', 'studio', 'security', 'skill',
    'language', 'python', 'java', 'react', 'fastapi', 'tool', 'database',
    'sql', 'pytorch', 'achievement', 'contest', 'hackathon', 'ambassador',
    'cgpa', 'marks', 'intern', 'career', 'goal', 'github', 'link'
  ];

  for (const p of ragPatterns) {
    if (q.includes(p)) return 'rag';
  }
  for (const p of directPatterns) {
    if (q.includes(p)) return 'direct';
  }
  return 'rag';
}

export default function PortfolioChat() {
  const [question, setQuestion] = useState('');
  const [activeMode, setActiveMode] = useState('rag'); // Auto-switched based on input
  const [messages, setMessages] = useState([
    {
      role: 'assistant',
      content:
        "Hello! I am Prem Kumar's AI Portfolio Assistant. Based on your question, I will automatically route between **Direct LLM** (for stable identity & education) and **LLM + RAG** (for dynamic projects, skills & achievements).",
      mode: 'rag',
      sources: []
    }
  ]);
  const [loading, setLoading] = useState(false);
  const [activeProcessSteps, setActiveProcessSteps] = useState([]);
  const [error, setError] = useState(null);

  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, loading, activeProcessSteps]);

  // Dynamically update the mode pill as the user types
  const handleInputChange = (e) => {
    const val = e.target.value;
    setQuestion(val);
    if (val.trim()) {
      setActiveMode(detectModeFromInput(val));
    }
  };

  async function handleSend(qText) {
    const currentQuestion = (qText || question).trim();
    if (!currentQuestion || loading) return;

    // Detect mode for this question
    const predictedMode = detectModeFromInput(currentQuestion);
    setActiveMode(predictedMode);

    setError(null);
    setMessages((prev) => [...prev, { role: 'user', content: currentQuestion }]);
    setQuestion('');
    setLoading(true);

    // Initial speculative progress steps showing automatic routing
    setActiveProcessSteps([
      {
        title: `Auto-Router: Detected intent (${currentQuestion.length} chars)`,
        detail: `Question: "${currentQuestion}"`,
        status: 'completed',
        badge: 'Analyzing'
      },
      {
        title: `Auto-Switching to ${predictedMode === 'rag' ? 'LLM + RAG' : 'Direct LLM'}`,
        detail:
          predictedMode === 'rag'
            ? 'Dynamic portfolio query -> Retrieving PDF context via FAISS vector store...'
            : 'Stable identity/profile query -> Directly querying fine-tuned Qwen parameters...',
        status: 'running',
        badge: predictedMode === 'rag' ? 'RAG' : 'Direct'
      }
    ]);

    try {
      const response = await fetch(`${API_URL}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          question: currentQuestion,
          mode: 'auto', // Backend auto-router handles decision
          top_k: 4,
          threshold: 0.25
        })
      });

      if (!response.ok) {
        throw new Error(`API error ${response.status}: ${response.statusText}`);
      }

      const data = await response.json();

      // Ensure the UI pill aligns with the backend's routing decision
      setActiveMode(data.mode);

      setMessages((prev) => [
        ...prev,
        {
          role: 'assistant',
          content: data.answer,
          mode: data.mode,
          sources: data.sources || [],
          process: data.process || [],
          latencyMs: data.latency_ms,
          routingReason: data.routing_reason
        }
      ]);
    } catch (err) {
      console.error('Chat error:', err);
      setError(err.message || 'Failed to reach backend.');
      setMessages((prev) => [
        ...prev,
        {
          role: 'assistant',
          content:
            "Sorry, I couldn't reach the portfolio assistant backend. Please ensure the backend is running at " +
            API_URL,
          mode: predictedMode,
          sources: []
        }
      ]);
    } finally {
      setLoading(false);
      setActiveProcessSteps([]);
    }
  }

  function handleFormSubmit(e) {
    e.preventDefault();
    handleSend();
  }

  return (
    <div className="w-full max-w-3xl mx-auto flex flex-col h-[85vh] rounded-2xl border border-zinc-800/80 bg-zinc-950/80 shadow-2xl overflow-hidden backdrop-blur-md">
      {/* Header */}
      <div className="flex items-center justify-between px-5 py-3.5 border-b border-zinc-800/80 bg-zinc-900/60">
        <div className="flex items-center gap-3">
          <div className="relative">
            <div className="w-9 h-9 rounded-xl bg-orange-600/20 border border-orange-500/30 flex items-center justify-center text-orange-400 font-bold text-sm">
              PK
            </div>
            <span className="absolute bottom-0 right-0 w-2.5 h-2.5 rounded-full bg-emerald-500 ring-2 ring-zinc-950"></span>
          </div>
          <div>
            <h1 className="text-sm font-semibold text-zinc-100 flex items-center gap-2">
              Prem Kumar
              <span className="text-[11px] font-normal px-2 py-0.5 rounded-full bg-zinc-800 text-zinc-400 border border-zinc-700/60">
                AI Portfolio
              </span>
            </h1>
            <p className="text-[11px] text-zinc-400">
              Qwen2.5-1.5B-Instruct + LoRA • Auto-Routing
            </p>
          </div>
        </div>

        {/* Automatic Switching Mode Pill (matches screenshot style) */}
        <ModeToggle mode={activeMode} autoDetected={true} />
      </div>

      {/* Messages Scroll Area */}
      <div className="flex-1 overflow-y-auto p-5 space-y-2">
        {messages.map((msg, index) => (
          <ChatMessage key={index} message={msg} />
        ))}

        {/* Live Loading / Honing process trace */}
        {loading && (
          <div className="my-3 max-w-xl animate-fade-in">
            <ThinkingTrace steps={activeProcessSteps} isLoading={true} defaultOpen={true} />
          </div>
        )}

        {error && (
          <div className="p-3 text-xs text-rose-400 bg-rose-950/30 border border-rose-900/50 rounded-xl">
            {error}
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Input Bar Form */}
      <div className="p-4 border-t border-zinc-800/80 bg-zinc-900/60">
        <form onSubmit={handleFormSubmit} className="flex items-center gap-2">
          <input
            type="text"
            value={question}
            onChange={handleInputChange}
            placeholder="Ask about projects, skills, education, experience (Auto-switches mode)..."
            maxLength={1000}
            disabled={loading}
            className="flex-1 bg-zinc-950 border border-zinc-800 rounded-xl px-4 py-2.5 text-sm text-zinc-100 placeholder-zinc-500 focus:outline-none focus:border-orange-500 transition-colors"
          />
          <button
            type="submit"
            disabled={loading || !question.trim()}
            className="px-4 py-2.5 bg-orange-600 hover:bg-orange-500 disabled:bg-zinc-800 text-white disabled:text-zinc-500 rounded-xl font-medium text-sm transition-all flex items-center gap-1.5 cursor-pointer disabled:cursor-not-allowed shadow-md"
          >
            <span>Send</span>
            <svg className="w-4 h-4" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" d="M14 5l7 7m0 0l-7 7m7-7H3" />
            </svg>
          </button>
        </form>
        <div className="mt-2 flex items-center justify-between text-[11px] text-zinc-500">
          <span className="flex items-center gap-1.5">
            <span className="w-1.5 h-1.5 rounded-full bg-emerald-500"></span>
            Smart Auto-Switch active: Routes to <strong>{activeMode === 'rag' ? 'LLM + RAG' : 'Direct LLM'}</strong> based on your input.
          </span>
          <button
            type="button"
            onClick={() =>
              setMessages([
                {
                  role: 'assistant',
                  content: "Chat cleared. What would you like to know about Prem's portfolio?",
                  mode: 'rag'
                }
              ])
            }
            className="px-2 py-0.5 rounded text-[11px] text-zinc-500 hover:text-zinc-300 hover:bg-zinc-800 transition-colors cursor-pointer bg-transparent border-0"
          >
            Clear chat
          </button>
        </div>
      </div>
    </div>
  );
}
