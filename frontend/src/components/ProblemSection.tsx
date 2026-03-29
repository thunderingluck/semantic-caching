"use client";

import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
} from "recharts";

// Simulate token accumulation over 419 conversation turns
const data = Array.from({ length: 42 }, (_, i) => {
  const turn = (i + 1) * 10;
  return {
    turn,
    fullHistory: turn * 95,
    naiveRag: Math.min(turn * 0.5 * 95 + 80, 510),
    semanticCache: Math.min(turn * 1.2 + 50, 520),
  };
});

const CustomTooltip = ({ active, payload, label }: any) => {
  if (!active || !payload?.length) return null;
  return (
    <div className="bg-slate-900 border border-slate-700 rounded-lg p-3 text-xs shadow-xl">
      <p className="text-slate-400 mb-2">Turn {label}</p>
      {payload.map((p: any) => (
        <div key={p.name} className="flex items-center gap-2 mb-1">
          <span className="w-2 h-2 rounded-full" style={{ background: p.color }} />
          <span className="text-slate-300">{p.name}:</span>
          <span className="text-white font-mono">{p.value.toLocaleString()} tok</span>
        </div>
      ))}
    </div>
  );
};

const noiseItems = [
  { pct: "42%", label: "Background chitchat", color: "#f97316" },
  { pct: "22%", label: "Repeated information", color: "#eab308" },
  { pct: "18%", label: "Stale / outdated context", color: "#6b7280" },
  { pct: "18%", label: "Relevant facts", color: "#22c55e" },
];

export default function ProblemSection() {
  return (
    <section id="problem" className="py-24 px-6">
      <div className="max-w-6xl mx-auto">
        <div className="text-center mb-16">
          <p className="text-blue-400 text-sm font-medium mb-3 tracking-wide uppercase">
            The Problem
          </p>
          <h2 className="text-3xl md:text-4xl font-bold text-white mb-4">
            Context rot degrades long conversations
          </h2>
          <p className="text-slate-400 max-w-2xl mx-auto">
            Raw history grows unboundedly. After 419 turns, you have ~40,000
            tokens — but only 18% is relevant to any given query. The rest is
            noise that dilutes the signal.
          </p>
        </div>

        <div className="grid md:grid-cols-3 gap-8 items-start">
          {/* Chart */}
          <div className="md:col-span-2 bg-slate-900/50 border border-slate-800 rounded-2xl p-6">
            <h3 className="text-sm font-medium text-slate-300 mb-1">
              Context token growth over a 419-turn conversation
            </h3>
            <p className="text-xs text-slate-500 mb-6">
              conv-26 (Caroline &amp; Melanie) · LoCoMo
            </p>
            <ResponsiveContainer width="100%" height={280}>
              <AreaChart data={data} margin={{ top: 4, right: 4, left: 0, bottom: 0 }}>
                <defs>
                  <linearGradient id="gFull" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#ef4444" stopOpacity={0.3} />
                    <stop offset="95%" stopColor="#ef4444" stopOpacity={0} />
                  </linearGradient>
                  <linearGradient id="gCache" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.3} />
                    <stop offset="95%" stopColor="#3b82f6" stopOpacity={0} />
                  </linearGradient>
                  <linearGradient id="gRag" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#a855f7" stopOpacity={0.2} />
                    <stop offset="95%" stopColor="#a855f7" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <CartesianGrid stroke="#1e293b" strokeDasharray="3 3" />
                <XAxis
                  dataKey="turn"
                  tick={{ fill: "#64748b", fontSize: 11 }}
                  tickLine={false}
                  axisLine={{ stroke: "#1e293b" }}
                  label={{ value: "Conversation turn", position: "insideBottom", offset: -2, fill: "#64748b", fontSize: 11 }}
                />
                <YAxis
                  tick={{ fill: "#64748b", fontSize: 11 }}
                  tickLine={false}
                  axisLine={false}
                  tickFormatter={(v) => v >= 1000 ? `${(v / 1000).toFixed(0)}K` : v}
                />
                <Tooltip content={<CustomTooltip />} />
                <ReferenceLine
                  y={500}
                  stroke="#64748b"
                  strokeDasharray="4 2"
                  label={{ value: "Budget (500)", position: "right", fill: "#64748b", fontSize: 10 }}
                />
                <Area
                  type="monotone"
                  dataKey="fullHistory"
                  name="Full History"
                  stroke="#ef4444"
                  strokeWidth={2}
                  fill="url(#gFull)"
                />
                <Area
                  type="monotone"
                  dataKey="naiveRag"
                  name="Naive RAG"
                  stroke="#a855f7"
                  strokeWidth={1.5}
                  fill="url(#gRag)"
                  strokeDasharray="4 2"
                />
                <Area
                  type="monotone"
                  dataKey="semanticCache"
                  name="Semantic Cache"
                  stroke="#3b82f6"
                  strokeWidth={2.5}
                  fill="url(#gCache)"
                />
              </AreaChart>
            </ResponsiveContainer>
            <div className="flex items-center gap-6 mt-4 justify-center">
              {[
                { color: "#ef4444", label: "Full History" },
                { color: "#a855f7", label: "Naive RAG", dash: true },
                { color: "#3b82f6", label: "Semantic Cache" },
              ].map((l) => (
                <div key={l.label} className="flex items-center gap-1.5">
                  <div
                    className="w-6 h-0.5"
                    style={{
                      background: l.color,
                      borderTop: l.dash ? `2px dashed ${l.color}` : undefined,
                      height: l.dash ? 0 : undefined,
                    }}
                  />
                  <span className="text-xs text-slate-400">{l.label}</span>
                </div>
              ))}
            </div>
          </div>

          {/* Noise breakdown */}
          <div className="space-y-4">
            <div className="bg-slate-900/50 border border-slate-800 rounded-2xl p-6">
              <h3 className="text-sm font-medium text-slate-300 mb-4">
                What&apos;s actually in &quot;full history&quot;
              </h3>
              <div className="space-y-3">
                {noiseItems.map((item) => (
                  <div key={item.label}>
                    <div className="flex justify-between text-xs mb-1">
                      <span className="text-slate-400">{item.label}</span>
                      <span className="font-mono" style={{ color: item.color }}>
                        {item.pct}
                      </span>
                    </div>
                    <div className="h-1.5 bg-slate-800 rounded-full overflow-hidden">
                      <div
                        className="h-full rounded-full"
                        style={{
                          width: item.pct,
                          background: item.color,
                          opacity: 0.8,
                        }}
                      />
                    </div>
                  </div>
                ))}
              </div>
            </div>

            <div className="bg-blue-950/40 border border-blue-900/50 rounded-2xl p-5">
              <p className="text-blue-300 text-sm font-medium mb-2">
                Our approach
              </p>
              <p className="text-slate-400 text-sm leading-relaxed">
                Treat context as a{" "}
                <span className="text-white">caching problem</span>. Extract
                atomic facts, score their future utility, retrieve only what
                matters for each query — all within a fixed token budget.
              </p>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
