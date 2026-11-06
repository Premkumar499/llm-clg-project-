import React, { useState } from 'react';

export default function ThinkingTrace({ steps = [], latencyMs, isLoading = false, defaultOpen = true }) {
  const [isOpen, setIsOpen] = useState(defaultOpen);
  const [expandedStep, setExpandedStep] = useState(null);

  if (!isLoading && (!steps || steps.length === 0)) {
    return null;
  }

  const toggleStep = (idx) => {
    setExpandedStep(expandedStep === idx ? null : idx);
  };

  return (
    <div className="my-2.5 max-w-xl text-xs select-none">
      {/* Accordion Trigger Header */}
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="flex items-center gap-2 px-1 py-1.5 text-zinc-400 hover:text-zinc-200 transition-colors group cursor-pointer focus:outline-none"
        aria-label="Toggle execution trace"
      >
        {/* Coral/Orange Dots Icon (matching user screenshot) */}
        <div className="flex items-center gap-0.5 text-orange-500 animate-pulse-dot">
          <span className="w-1.5 h-1.5 rounded-full bg-orange-500"></span>
          <span className="w-1.5 h-1.5 rounded-full bg-orange-400"></span>
          <span className="w-1 h-1 rounded-full bg-orange-600"></span>
        </div>

        <span className="font-medium text-zinc-300 text-sm tracking-wide">
          {isLoading ? 'Honing...' : 'Honing'}
        </span>

        {latencyMs && !isLoading && (
          <span className="text-[11px] text-zinc-500 font-mono ml-0.5">
            ({latencyMs}ms)
          </span>
        )}

        {/* Chevron v / ^ */}
        <svg
          className={`w-3.5 h-3.5 text-zinc-500 group-hover:text-zinc-300 transition-transform duration-200 ${
            isOpen ? 'rotate-180' : 'rotate-0'
          }`}
          fill="none"
          stroke="currentColor"
          strokeWidth="2"
          viewBox="0 0 24 24"
        >
          <path strokeLinecap="round" strokeLinejoin="round" d="M19 9l-7 7-7-7" />
        </svg>
      </button>

      {/* Accordion Card Container (matching dark card style from screenshot) */}
      {isOpen && (
        <div
          className="mt-1.5 rounded-xl border border-zinc-800/80 bg-zinc-900/90 shadow-lg backdrop-blur overflow-hidden divide-y divide-zinc-800/60 animate-fade-in"
          style={{ backgroundColor: '#16161a' }}
        >
          {steps.map((step, idx) => {
            const hasDetail = Boolean(step.detail);
            const isStepExpanded = expandedStep === idx;

            return (
              <div key={idx} className="transition-colors hover:bg-zinc-800/40">
                <div
                  onClick={() => hasDetail && toggleStep(idx)}
                  className={`flex items-center justify-between px-3.5 py-2.5 text-zinc-300 ${
                    hasDetail ? 'cursor-pointer' : ''
                  }`}
                >
                  <div className="flex items-center gap-2.5 min-w-0 pr-2">
                    <span className="w-1.5 h-1.5 rounded-full bg-zinc-500 shrink-0"></span>
                    <span className="text-zinc-300 font-normal truncate text-[13px]">
                      {step.title}
                    </span>
                  </div>

                  <div className="flex items-center gap-2 shrink-0">
                    {step.badge && (
                      <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-zinc-800 text-zinc-400 border border-zinc-700/60">
                        {step.badge}
                      </span>
                    )}
                    {hasDetail && (
                      <svg
                        className={`w-3.5 h-3.5 text-zinc-500 transition-transform duration-200 ${
                          isStepExpanded ? 'rotate-90' : ''
                        }`}
                        fill="none"
                        stroke="currentColor"
                        strokeWidth="2"
                        viewBox="0 0 24 24"
                      >
                        <path strokeLinecap="round" strokeLinejoin="round" d="M9 5l7 7-7 7" />
                      </svg>
                    )}
                  </div>
                </div>

                {/* Sub-detail drawer */}
                {isStepExpanded && step.detail && (
                  <div className="px-3.5 pb-2.5 pt-0.5 text-zinc-400 text-[12px] font-mono leading-relaxed bg-zinc-950/50 border-t border-zinc-850">
                    {step.detail}
                  </div>
                )}
              </div>
            );
          })}

          {isLoading && (
            <div className="flex items-center gap-2 px-3.5 py-2.5 text-zinc-400 italic text-[12px]">
              <span className="inline-block w-2 h-2 rounded-full bg-orange-500 animate-ping"></span>
              Generating grounded response...
            </div>
          )}
        </div>
      )}
    </div>
  );
}

