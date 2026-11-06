import React from 'react';

export default function ModeToggle({ mode, autoDetected = true }) {
  const isRAG = mode === 'rag';

  return (
    <div className="flex items-center gap-2 select-none">
      <div className="flex items-center gap-4 px-3.5 py-1.5 bg-zinc-900 border border-zinc-800 rounded-xl shadow-inner text-xs">
        {/* LLM + RAG */}
        <div>
          <span className={`font-medium tracking-wide transition-colors ${
            isRAG ? 'text-zinc-100 font-semibold' : 'text-zinc-100'
          }`}>
            LLM + RAG
          </span>
        </div>

        {/* Direct LLM */}
        <div>
          <span className={`font-medium tracking-wide transition-colors ${
            !isRAG ? 'text-zinc-100 font-semibold' : 'text-zinc-100'
          }`}>
            Direct LLM
          </span>
        </div>
      </div>

      {autoDetected && (
        <span className="hidden sm:inline-flex items-center gap-1 text-[10px] text-zinc-300 bg-zinc-900 border border-zinc-800 px-2 py-0.5 rounded-full font-mono uppercase tracking-wider">
          <span className="w-1.5 h-1.5 rounded-full bg-emerald-400"></span>
          Auto
        </span>
      )}
    </div>
  );
}
