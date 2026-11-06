import React from 'react';
import PortfolioChat from './components/PortfolioChat';

export default function App() {
  return (
    <div className="min-h-screen bg-zinc-950 text-zinc-100 flex flex-col justify-between p-4 md:p-8 font-sans">
      {/* Top Banner Navigation */}
      <header className="max-w-4xl w-full mx-auto mb-6 flex flex-wrap items-center justify-between gap-4">
        <div>
          <div className="flex items-center gap-2">
            <span className="w-2.5 h-2.5 rounded-full bg-orange-500"></span>
            <span className="text-xs uppercase font-mono tracking-widest text-orange-400">
              Student Portfolio RAG + LLM Integration
            </span>
          </div>
          <h1 className="text-2xl font-bold text-white tracking-tight mt-1">
            Prem Kumar
          </h1>
          <p className="text-xs text-zinc-400">
            B.Tech Artificial Intelligence & Data Science • KSRCT (Class of 2028)
          </p>
        </div>

        {/* Quick Links */}
        <div className="flex items-center gap-2.5 text-xs">
          <a
            href="https://github.com/Premkumar499"
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center px-3 py-1.5 rounded-lg border border-zinc-800 bg-zinc-900 hover:bg-zinc-800 text-zinc-300 hover:text-white transition-colors"
            style={{ textDecoration: 'none' }}
          >
            GitHub
          </a>
          <a
            href="https://deepfake-flame.vercel.app/"
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center px-3 py-1.5 rounded-lg border border-zinc-800 bg-zinc-900 hover:bg-zinc-800 text-zinc-300 hover:text-white transition-colors"
            style={{ textDecoration: 'none' }}
          >
            Deepfake Project
          </a>
        </div>
      </header>

      {/* Main Interactive Chat Section */}
      <main className="max-w-4xl w-full mx-auto flex-1 flex flex-col justify-center">
        <PortfolioChat />
      </main>

      {/* Footer */}
      <footer className="max-w-4xl w-full mx-auto mt-6 pt-4 border-t border-zinc-850 flex flex-wrap items-center justify-between text-xs text-zinc-500 gap-2">
        <div className="flex items-center gap-2">
          <span>Powered by Qwen2.5-1.5B-Instruct + LoRA & FAISS</span>
          <span>•</span>
          <span>FastAPI Backend</span>
        </div>
        <div>
          <span>Knowledge Source: prem_rag_knowledge_base.pdf</span>
        </div>
      </footer>
    </div>
  );
}

