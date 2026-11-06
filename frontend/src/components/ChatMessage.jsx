import React, { useState } from 'react';
import ThinkingTrace from './ThinkingTrace';

function formatMarkdown(text) {
  if (!text) return null;

  const lines = text.split('\n');
  const elements = [];
  let inList = false;
  let listItems = [];

  const flushList = () => {
    if (inList && listItems.length > 0) {
      elements.push(
        <ul key={`ul-${elements.length}`} className="list-disc pl-5 my-2 space-y-1">
          {listItems.map((item, i) => (
            <li key={i} dangerouslySetInnerHTML={{ __html: item }} />
          ))}
        </ul>
      );
      listItems = [];
      inList = false;
    }
  };

  lines.forEach((line, idx) => {
    const trimmed = line.trim();

    if (trimmed.startsWith('• ') || trimmed.startsWith('- ') || trimmed.startsWith('* ')) {
      inList = true;
      const content = trimmed.substring(2);
      const boldFormatted = content.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
      const linkFormatted = boldFormatted.replace(
        /(https?:\/\/[^\s]+)/g,
        '<a href="$1" target="_blank" rel="noopener noreferrer" class="text-orange-400 hover:underline">$1</a>'
      );
      listItems.push(linkFormatted);
    } else {
      flushList();
      if (trimmed === '') return;
      let formatted = trimmed.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
      formatted = formatted.replace(
        /(https?:\/\/[^\s]+)/g,
        '<a href="$1" target="_blank" rel="noopener noreferrer" class="text-orange-400 hover:underline">$1</a>'
      );
      elements.push(
        <p key={idx} className="my-1.5 leading-relaxed" dangerouslySetInnerHTML={{ __html: formatted }} />
      );
    }
  });

  flushList();
  return elements;
}

export default function ChatMessage({ message }) {
  const isUser = message.role === 'user';
  const [showSources, setShowSources] = useState(false);
  const [copied, setCopied] = useState(false);

  const copyAnswer = () => {
    if (message.content) {
      navigator.clipboard.writeText(message.content);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    }
  };

  if (isUser) {
    return (
      <div className="flex justify-end my-3">
        <div className="max-w-xl rounded-2xl bg-orange-600 text-white px-4 py-2.5 shadow-sm text-[14px] leading-relaxed">
          {message.content}
        </div>
      </div>
    );
  }

  return (
    <div className="my-4 max-w-2xl text-[14px]">
      {/* Honing / Thinking trace */}
      {message.process && message.process.length > 0 && (
        <ThinkingTrace
          steps={message.process}
          latencyMs={message.latencyMs}
          isLoading={false}
          defaultOpen={false}
        />
      )}

      {/* Normal chat bubble */}
      <div className="rounded-2xl border border-zinc-800 bg-zinc-900/80 p-4 shadow-sm text-zinc-200">
        <div className="prose-dark text-[14px]">
          {formatMarkdown(message.content)}
        </div>

        {/* Footer info: Mode & Copy button */}
        <div className="mt-3 pt-2.5 border-t border-zinc-800/60 flex items-center justify-between text-xs text-zinc-500">
          <div className="flex items-center gap-2">
            {message.mode && (
              <span
                className={`px-2 py-0.5 rounded text-[11px] font-medium uppercase tracking-wider ${
                  message.mode === 'rag'
                    ? 'bg-orange-500/10 text-orange-400 border border-orange-500/20'
                    : 'bg-zinc-800 text-zinc-300 border border-zinc-700'
                }`}
              >
                {message.mode === 'rag' ? 'RAG Grounded' : 'Direct LLM'}
              </span>
            )}

            {message.sources && message.sources.length > 0 && (
              <button
                type="button"
                onClick={() => setShowSources(!showSources)}
                className="px-2 py-0.5 rounded text-[11px] text-zinc-400 hover:text-zinc-200 hover:bg-zinc-800 transition-colors flex items-center gap-1 cursor-pointer bg-transparent border-0"
              >
                <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" d="M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.747 0 3.332.477 4.5 1.253v13C19.832 18.477 18.247 18 16.5 18c-1.746 0-3.332.477-4.5 1.253" />
                </svg>
                {showSources ? 'Hide Sources' : `${message.sources.length} Sources`}
              </button>
            )}
          </div>

          <button
            type="button"
            onClick={copyAnswer}
            className="px-2.5 py-1 rounded-md text-[11px] text-zinc-400 hover:text-zinc-200 hover:bg-zinc-800 transition-colors flex items-center gap-1.5 cursor-pointer bg-transparent border-0"
          >
            {copied ? (
              <span className="text-emerald-400 font-medium">✓ Copied</span>
            ) : (
              <>
                <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z" />
                </svg>
                <span>Copy</span>
              </>
            )}
          </button>
        </div>

        {/* Collapsible Sources Drawer */}
        {showSources && message.sources && message.sources.length > 0 && (
          <div className="mt-2.5 p-3 rounded-lg bg-zinc-950 border border-zinc-800 text-xs text-zinc-400 space-y-2 animate-fade-in font-mono">
            <div className="text-[11px] font-semibold text-zinc-300 uppercase tracking-wider">
              Retrieved Context Passages:
            </div>
            {message.sources.map((src, i) => (
              <div key={i} className="p-2 rounded bg-zinc-900 border border-zinc-800 text-[11px] leading-relaxed">
                <span className="text-orange-400 font-bold mr-1">[{i + 1}]</span>
                {src}
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
