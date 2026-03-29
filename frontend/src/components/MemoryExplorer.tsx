"use client";

import { useEffect, useState, useMemo } from "react";

interface Memory {
  id: string;
  fact: string;
  type: string;
  importance: number;
  persistence: string;
  scope: string;
  source_turns: number[];
  status: string;
  session_id: string;
}

const TYPE_COLORS: Record<string, string> = {
  fact: "bg-slate-800 text-slate-300 border-slate-700",
  preference: "bg-blue-900/50 text-blue-300 border-blue-800",
  constraint: "bg-red-900/50 text-red-300 border-red-800",
  decision: "bg-green-900/50 text-green-300 border-green-800",
  definition: "bg-yellow-900/50 text-yellow-300 border-yellow-800",
  goal: "bg-violet-900/50 text-violet-300 border-violet-800",
};

const PERSISTENCE_COLORS: Record<string, string> = {
  long_term: "text-green-400",
  medium: "text-yellow-400",
  ephemeral: "text-slate-500",
};

const PAGE_SIZE = 15;

export default function MemoryExplorer() {
  const [memories, setMemories] = useState<Memory[]>([]);
  const [loading, setLoading] = useState(true);
  const [query, setQuery] = useState("");
  const [typeFilter, setTypeFilter] = useState("all");
  const [persistenceFilter, setPersistenceFilter] = useState("all");
  const [sortBy, setSortBy] = useState<"importance" | "turn">("importance");
  const [page, setPage] = useState(0);

  useEffect(() => {
    fetch("/data/memories.json")
      .then((r) => r.json())
      .then((d: Memory[]) => {
        setMemories(d);
        setLoading(false);
      })
      .catch(() => setLoading(false));
  }, []);

  const types = useMemo(
    () => ["all", ...Array.from(new Set(memories.map((m) => m.type))).sort()],
    [memories]
  );

  const persistences = useMemo(
    () => ["all", ...Array.from(new Set(memories.map((m) => m.persistence))).sort()],
    [memories]
  );

  const filtered = useMemo(() => {
    let res = memories;
    if (typeFilter !== "all") res = res.filter((m) => m.type === typeFilter);
    if (persistenceFilter !== "all") res = res.filter((m) => m.persistence === persistenceFilter);
    if (query.trim()) {
      const q = query.toLowerCase();
      res = res.filter((m) => m.fact.toLowerCase().includes(q));
    }
    if (sortBy === "importance") {
      res = [...res].sort((a, b) => b.importance - a.importance);
    } else {
      res = [...res].sort(
        (a, b) => (a.source_turns[0] ?? 0) - (b.source_turns[0] ?? 0)
      );
    }
    return res;
  }, [memories, query, typeFilter, persistenceFilter, sortBy]);

  const paginated = filtered.slice(page * PAGE_SIZE, (page + 1) * PAGE_SIZE);
  const totalPages = Math.ceil(filtered.length / PAGE_SIZE);

  // Reset page when filters change
  useEffect(() => setPage(0), [query, typeFilter, persistenceFilter, sortBy]);

  // Stats
  const avgImportance =
    memories.length > 0
      ? memories.reduce((s, m) => s + m.importance, 0) / memories.length
      : 0;
  const typeCounts = useMemo(
    () =>
      memories.reduce(
        (acc, m) => ({ ...acc, [m.type]: (acc[m.type] ?? 0) + 1 }),
        {} as Record<string, number>
      ),
    [memories]
  );

  return (
    <section id="memories" className="py-24 px-6 bg-slate-950/50">
      <div className="max-w-6xl mx-auto">
        <div className="text-center mb-12">
          <p className="text-blue-400 text-sm font-medium mb-3 tracking-wide uppercase">
            Memory Explorer
          </p>
          <h2 className="text-3xl md:text-4xl font-bold text-white mb-4">
            {memories.length} atomic facts extracted
          </h2>
          <p className="text-slate-400 max-w-xl mx-auto">
            Browse, search, and filter the semantic cache built from the 419-turn
            conversation. Each fact is scored for importance and tagged with
            persistence level.
          </p>
        </div>

        {/* Stats bar */}
        {memories.length > 0 && (
          <div className="grid grid-cols-2 md:grid-cols-5 gap-3 mb-8">
            {Object.entries(typeCounts)
              .sort((a, b) => b[1] - a[1])
              .map(([type, count]) => (
                <button
                  key={type}
                  onClick={() =>
                    setTypeFilter(typeFilter === type ? "all" : type)
                  }
                  className={`flex flex-col items-center p-3 rounded-xl border transition-all text-center ${
                    typeFilter === type
                      ? "border-blue-600 bg-blue-950/40"
                      : "border-slate-800 bg-slate-900/40 hover:border-slate-700"
                  }`}
                >
                  <span className="text-xl font-bold text-white">{count}</span>
                  <span
                    className={`text-xs mt-0.5 px-1.5 py-0.5 rounded border ${
                      TYPE_COLORS[type] ?? "text-slate-400"
                    }`}
                  >
                    {type}
                  </span>
                </button>
              ))}
            <div className="flex flex-col items-center p-3 rounded-xl border border-slate-800 bg-slate-900/40 text-center">
              <span className="text-xl font-bold text-white">
                {avgImportance.toFixed(2)}
              </span>
              <span className="text-xs text-slate-500 mt-0.5">avg importance</span>
            </div>
          </div>
        )}

        {/* Controls */}
        <div className="flex flex-wrap gap-3 mb-6">
          <input
            type="text"
            placeholder="Search facts…"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            className="flex-1 min-w-48 bg-slate-900 border border-slate-700 rounded-lg px-4 py-2 text-sm text-white placeholder-slate-600 focus:outline-none focus:border-blue-600 transition-colors"
          />

          <select
            value={typeFilter}
            onChange={(e) => setTypeFilter(e.target.value)}
            className="bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-sm text-slate-300 focus:outline-none focus:border-blue-600 transition-colors"
          >
            {types.map((t) => (
              <option key={t} value={t}>
                {t === "all" ? "All types" : t}
              </option>
            ))}
          </select>

          <select
            value={persistenceFilter}
            onChange={(e) => setPersistenceFilter(e.target.value)}
            className="bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-sm text-slate-300 focus:outline-none focus:border-blue-600 transition-colors"
          >
            {persistences.map((p) => (
              <option key={p} value={p}>
                {p === "all" ? "All persistence" : p.replace("_", " ")}
              </option>
            ))}
          </select>

          <select
            value={sortBy}
            onChange={(e) => setSortBy(e.target.value as "importance" | "turn")}
            className="bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-sm text-slate-300 focus:outline-none focus:border-blue-600 transition-colors"
          >
            <option value="importance">Sort: importance</option>
            <option value="turn">Sort: turn order</option>
          </select>
        </div>

        {/* Results count */}
        <div className="flex items-center justify-between mb-4">
          <p className="text-sm text-slate-500">
            {filtered.length} result{filtered.length !== 1 ? "s" : ""}
            {query && ` matching "${query}"`}
          </p>
          {filtered.length > PAGE_SIZE && (
            <p className="text-sm text-slate-500">
              Page {page + 1} / {totalPages}
            </p>
          )}
        </div>

        {/* Table */}
        {loading ? (
          <div className="text-center py-16 text-slate-500 animate-pulse">
            Loading memories…
          </div>
        ) : paginated.length === 0 ? (
          <div className="text-center py-16 text-slate-500">No results found.</div>
        ) : (
          <div className="space-y-2">
            {paginated.map((mem) => (
              <div
                key={mem.id}
                className="group bg-slate-900/40 hover:bg-slate-900/70 border border-slate-800 hover:border-slate-700 rounded-xl px-5 py-4 transition-all"
              >
                <div className="flex items-start gap-4">
                  {/* Importance bar */}
                  <div className="flex flex-col items-center gap-1 pt-0.5 flex-shrink-0">
                    <span className="text-xs font-mono text-white font-semibold">
                      {mem.importance.toFixed(1)}
                    </span>
                    <div className="w-1 h-12 bg-slate-800 rounded-full overflow-hidden">
                      <div
                        className="w-full rounded-full bg-blue-500 transition-all"
                        style={{
                          height: `${mem.importance * 100}%`,
                          marginTop: `${(1 - mem.importance) * 100}%`,
                        }}
                      />
                    </div>
                  </div>

                  {/* Content */}
                  <div className="flex-1 min-w-0">
                    <p className="text-slate-200 text-sm leading-relaxed mb-2">
                      {mem.fact}
                    </p>
                    <div className="flex flex-wrap items-center gap-2">
                      <span
                        className={`text-xs px-2 py-0.5 rounded-md border ${
                          TYPE_COLORS[mem.type] ?? "bg-slate-800 text-slate-400 border-slate-700"
                        }`}
                      >
                        {mem.type}
                      </span>
                      <span
                        className={`text-xs font-mono ${
                          PERSISTENCE_COLORS[mem.persistence] ?? "text-slate-500"
                        }`}
                      >
                        {mem.persistence.replace("_", " ")}
                      </span>
                      <span className="text-xs text-slate-600">
                        turns {mem.source_turns[0]}–{mem.source_turns[mem.source_turns.length - 1]}
                      </span>
                      <span className="text-xs text-slate-700 font-mono">{mem.id}</span>
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}

        {/* Pagination */}
        {totalPages > 1 && (
          <div className="flex items-center justify-center gap-2 mt-8">
            <button
              onClick={() => setPage((p) => Math.max(0, p - 1))}
              disabled={page === 0}
              className="px-4 py-2 text-sm rounded-lg border border-slate-700 text-slate-400 hover:text-white hover:border-slate-500 disabled:opacity-30 disabled:cursor-not-allowed transition-all"
            >
              ← Prev
            </button>
            <div className="flex gap-1">
              {Array.from({ length: Math.min(7, totalPages) }, (_, i) => {
                const target =
                  totalPages <= 7
                    ? i
                    : page < 4
                    ? i
                    : page > totalPages - 4
                    ? totalPages - 7 + i
                    : page - 3 + i;
                return (
                  <button
                    key={target}
                    onClick={() => setPage(target)}
                    className={`w-8 h-8 text-sm rounded-lg transition-all ${
                      page === target
                        ? "bg-blue-600 text-white"
                        : "text-slate-500 hover:text-white"
                    }`}
                  >
                    {target + 1}
                  </button>
                );
              })}
            </div>
            <button
              onClick={() => setPage((p) => Math.min(totalPages - 1, p + 1))}
              disabled={page === totalPages - 1}
              className="px-4 py-2 text-sm rounded-lg border border-slate-700 text-slate-400 hover:text-white hover:border-slate-500 disabled:opacity-30 disabled:cursor-not-allowed transition-all"
            >
              Next →
            </button>
          </div>
        )}
      </div>
    </section>
  );
}
