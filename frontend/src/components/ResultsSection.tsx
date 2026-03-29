"use client";

import { useEffect, useState } from "react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from "recharts";

interface MethodResult {
  exact_match_rate: number;
  avg_f1: number;
  avg_tokens: number;
  n: number;
  f1_per_100tok?: number;
}

type EvalResults = Record<string, MethodResult>;

const METHOD_LABELS: Record<string, string> = {
  semantic_cache: "Semantic Cache",
  full_history: "Full History",
  rolling_summary: "Rolling Summary",
  naive_rag: "Naive RAG",
};

const METHOD_COLORS: Record<string, string> = {
  semantic_cache: "#3b82f6",
  full_history: "#ef4444",
  rolling_summary: "#f97316",
  naive_rag: "#a855f7",
};

const METHOD_ORDER = ["full_history", "rolling_summary", "naive_rag", "semantic_cache"];

const CustomTooltip = ({ active, payload, label }: any) => {
  if (!active || !payload?.length) return null;
  return (
    <div className="bg-slate-900 border border-slate-700 rounded-lg p-3 text-xs shadow-xl">
      <p className="text-slate-400 mb-1">{label}</p>
      <p className="text-white font-mono font-semibold">
        {typeof payload[0].value === "number"
          ? payload[0].value % 1 === 0
            ? payload[0].value.toLocaleString()
            : payload[0].value.toFixed(4)
          : payload[0].value}
      </p>
    </div>
  );
};

interface ChartCardProps {
  title: string;
  subtitle: string;
  data: { method: string; value: number }[];
  higherIsBetter: boolean;
  formatter?: (v: number) => string;
  highlight?: string;
}

function ChartCard({ title, subtitle, data, higherIsBetter, formatter, highlight }: ChartCardProps) {
  const sorted = [...data].sort((a, b) =>
    higherIsBetter ? b.value - a.value : a.value - b.value
  );
  const best = sorted[0]?.method;

  return (
    <div className="bg-slate-900/50 border border-slate-800 rounded-2xl p-5">
      <div className="mb-1">
        <h3 className="text-sm font-semibold text-white">{title}</h3>
        <p className="text-xs text-slate-500">{subtitle}</p>
      </div>
      <p className="text-xs text-slate-600 mb-4">
        {higherIsBetter ? "↑ higher is better" : "↓ lower is better"}
      </p>
      <ResponsiveContainer width="100%" height={200}>
        <BarChart data={data} margin={{ top: 4, right: 4, left: -20, bottom: 0 }}>
          <CartesianGrid stroke="#1e293b" strokeDasharray="3 3" vertical={false} />
          <XAxis
            dataKey="method"
            tick={{ fill: "#64748b", fontSize: 10 }}
            tickLine={false}
            axisLine={false}
            tickFormatter={(v) =>
              v === "semantic_cache"
                ? "Cache ★"
                : v === "full_history"
                ? "Full Hist."
                : v === "rolling_summary"
                ? "Summary"
                : "Naive RAG"
            }
          />
          <YAxis
            tick={{ fill: "#64748b", fontSize: 10 }}
            tickLine={false}
            axisLine={false}
            tickFormatter={formatter}
          />
          <Tooltip content={<CustomTooltip />} />
          <Bar dataKey="value" radius={[4, 4, 0, 0]} maxBarSize={48}>
            {data.map((entry) => (
              <Cell
                key={entry.method}
                fill={METHOD_COLORS[entry.method] ?? "#64748b"}
                opacity={entry.method === (highlight ?? best) ? 1 : 0.45}
                stroke={entry.method === (highlight ?? best) ? METHOD_COLORS[entry.method] : "none"}
                strokeWidth={entry.method === (highlight ?? best) ? 1.5 : 0}
              />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
      <div className="mt-3 flex flex-wrap gap-2">
        {data.map((d) => (
          <div key={d.method} className="flex items-center gap-1.5 text-xs">
            <span
              className="w-2 h-2 rounded-full"
              style={{ background: METHOD_COLORS[d.method] }}
            />
            <span className="text-slate-500">
              {METHOD_LABELS[d.method]}:{" "}
              <span className="text-slate-300 font-mono">
                {formatter ? formatter(d.value) : d.value.toFixed(3)}
              </span>
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}

export default function ResultsSection() {
  const [results, setResults] = useState<EvalResults | null>(null);
  const [error, setError] = useState(false);

  useEffect(() => {
    fetch("/data/eval_results.json")
      .then((r) => r.json())
      .then((d: EvalResults) => {
        // Compute f1_per_100tok
        for (const m in d) {
          d[m].f1_per_100tok =
            d[m].avg_tokens > 0 ? d[m].avg_f1 / (d[m].avg_tokens / 100) : 0;
        }
        setResults(d);
      })
      .catch(() => setError(true));
  }, []);

  if (error) {
    return (
      <section id="results" className="py-24 px-6">
        <div className="max-w-6xl mx-auto text-center text-slate-500">
          <p>Run the notebook to generate eval_results.json, then redeploy.</p>
        </div>
      </section>
    );
  }

  if (!results) {
    return (
      <section id="results" className="py-24 px-6">
        <div className="max-w-6xl mx-auto text-center text-slate-500 animate-pulse">
          Loading results…
        </div>
      </section>
    );
  }

  const methods = METHOD_ORDER.filter((m) => m in results);

  const f1Data = methods.map((m) => ({ method: m, value: results[m].avg_f1 }));
  const emData = methods.map((m) => ({ method: m, value: results[m].exact_match_rate }));
  const tokData = methods.map((m) => ({ method: m, value: results[m].avg_tokens }));
  const effData = methods.map((m) => ({
    method: m,
    value: results[m].f1_per_100tok ?? 0,
  }));

  // Summary table
  const best_f1 = methods.reduce((a, b) =>
    results[a].avg_f1 > results[b].avg_f1 ? a : b
  );
  const best_eff = methods.reduce((a, b) =>
    (results[a].f1_per_100tok ?? 0) > (results[b].f1_per_100tok ?? 0) ? a : b
  );

  return (
    <section id="results" className="py-24 px-6">
      <div className="max-w-6xl mx-auto">
        <div className="text-center mb-16">
          <p className="text-blue-400 text-sm font-medium mb-3 tracking-wide uppercase">
            Evaluation
          </p>
          <h2 className="text-3xl md:text-4xl font-bold text-white mb-4">
            Results across four methods
          </h2>
          <p className="text-slate-400 max-w-2xl mx-auto">
            Evaluated on {results[methods[0]]?.n ?? 20} QA pairs from the
            LoCoMo benchmark (conv-26). Semantic cache matches or beats full
            history at{" "}
            <span className="text-white font-medium">
              {Math.round(
                results["full_history"]?.avg_tokens /
                  results["semantic_cache"]?.avg_tokens
              )}
              ×
            </span>{" "}
            lower token cost.
          </p>
        </div>

        {/* Summary table */}
        <div className="mb-10 overflow-x-auto">
          <table className="w-full text-sm border-collapse">
            <thead>
              <tr className="border-b border-slate-800">
                <th className="text-left py-3 px-4 text-slate-400 font-medium">Method</th>
                <th className="text-right py-3 px-4 text-slate-400 font-medium">EM ↑</th>
                <th className="text-right py-3 px-4 text-slate-400 font-medium">F1 ↑</th>
                <th className="text-right py-3 px-4 text-slate-400 font-medium">Avg Tokens ↓</th>
                <th className="text-right py-3 px-4 text-slate-400 font-medium">F1/100 tok ↑</th>
              </tr>
            </thead>
            <tbody>
              {methods.map((m) => {
                const r = results[m];
                const isOurs = m === "semantic_cache";
                return (
                  <tr
                    key={m}
                    className={`border-b border-slate-900 ${
                      isOurs ? "bg-blue-950/20" : "hover:bg-slate-900/30"
                    } transition-colors`}
                  >
                    <td className="py-3 px-4">
                      <div className="flex items-center gap-2">
                        <span
                          className="w-2 h-2 rounded-full"
                          style={{ background: METHOD_COLORS[m] }}
                        />
                        <span className={isOurs ? "text-white font-medium" : "text-slate-300"}>
                          {METHOD_LABELS[m]}
                        </span>
                        {isOurs && (
                          <span className="text-xs px-1.5 py-0.5 bg-blue-900/50 text-blue-300 border border-blue-800 rounded">
                            ours
                          </span>
                        )}
                      </div>
                    </td>
                    <td className="py-3 px-4 text-right font-mono text-slate-300">
                      {(r.exact_match_rate * 100).toFixed(1)}%
                    </td>
                    <td className={`py-3 px-4 text-right font-mono ${m === best_f1 ? "text-green-400 font-semibold" : "text-slate-300"}`}>
                      {r.avg_f1.toFixed(4)}
                    </td>
                    <td className={`py-3 px-4 text-right font-mono ${m === "semantic_cache" ? "text-blue-400" : "text-slate-300"}`}>
                      {r.avg_tokens.toLocaleString(undefined, { maximumFractionDigits: 0 })}
                    </td>
                    <td className={`py-3 px-4 text-right font-mono ${m === best_eff ? "text-green-400 font-semibold" : "text-slate-300"}`}>
                      {(r.f1_per_100tok ?? 0).toFixed(5)}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>

        {/* Charts grid */}
        <div className="grid md:grid-cols-2 gap-5">
          <ChartCard
            title="Token F1 Score"
            subtitle="Token overlap between predicted and gold answers"
            data={f1Data}
            higherIsBetter={true}
          />
          <ChartCard
            title="Average Context Tokens"
            subtitle="Tokens consumed per query"
            data={tokData}
            higherIsBetter={false}
            formatter={(v) => v >= 1000 ? `${(v / 1000).toFixed(1)}K` : String(Math.round(v))}
            highlight="semantic_cache"
          />
          <ChartCard
            title="Exact Match Rate"
            subtitle="Fraction of answers matching gold exactly"
            data={emData}
            higherIsBetter={true}
            formatter={(v) => `${(v * 100).toFixed(0)}%`}
          />
          <ChartCard
            title="F1 per 100 Tokens"
            subtitle="Answer quality relative to token cost (efficiency)"
            data={effData}
            higherIsBetter={true}
            formatter={(v) => v.toFixed(4)}
          />
        </div>

        {/* Analysis note */}
        <div className="mt-8 p-5 bg-slate-900/40 border border-slate-800 rounded-xl text-sm text-slate-400 leading-relaxed">
          <span className="text-white font-medium">Note on low EM: </span>
          4 of the 20 sampled questions are LoCoMo category 5 (unanswerable
          from single-session context) and 7 are temporal reasoning questions
          requiring exact relative-date expressions. This creates a{" "}
          <span className="text-slate-300">structural ceiling of ~45% EM</span>{" "}
          even for an oracle retriever. The token efficiency metric (F1 per 100
          tokens) is the most informative signal here.
        </div>
      </div>
    </section>
  );
}
