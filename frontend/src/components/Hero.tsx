"use client";

const stats = [
  { value: "405", label: "Atomic facts", sub: "extracted from 419 turns" },
  { value: "80×", label: "Token reduction", sub: "40K → 500 tokens per query" },
  { value: "10", label: "Conversations", sub: "LoCoMo benchmark" },
  { value: "1,986", label: "QA pairs", sub: "with evidence annotations" },
];

export default function Hero() {
  return (
    <section className="relative min-h-screen flex flex-col items-center justify-center px-6 pt-14 overflow-hidden">
      {/* Background grid */}
      <div
        className="absolute inset-0 opacity-[0.03]"
        style={{
          backgroundImage:
            "linear-gradient(#334155 1px, transparent 1px), linear-gradient(90deg, #334155 1px, transparent 1px)",
          backgroundSize: "40px 40px",
        }}
      />

      {/* Glow */}
      <div className="absolute top-1/3 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[600px] h-[400px] bg-blue-600/10 rounded-full blur-3xl pointer-events-none" />

      <div className="relative z-10 max-w-4xl w-full text-center">
        <div className="inline-flex items-center gap-2 px-3 py-1 mb-8 rounded-full border border-blue-500/30 bg-blue-500/10 text-blue-400 text-xs font-medium">
          <span className="w-1.5 h-1.5 rounded-full bg-blue-400 animate-pulse" />
          Hackathon Demo · LoCoMo Benchmark
        </div>

        <h1 className="text-5xl md:text-6xl font-bold tracking-tight text-white mb-6 leading-[1.1]">
          Semantic{" "}
          <span className="text-transparent bg-clip-text bg-gradient-to-r from-blue-400 to-violet-400">
            Memory Cache
          </span>
        </h1>

        <p className="text-lg text-slate-400 max-w-2xl mx-auto mb-4 leading-relaxed">
          Context should be{" "}
          <span className="text-white font-medium">constructed</span>, not{" "}
          <span className="text-slate-500 line-through">accumulated</span>. We
          replace raw conversation history with a learned semantic cache that
          stores and retrieves only what has{" "}
          <span className="text-white font-medium">high future utility</span>.
        </p>

        <p className="text-sm text-slate-500 mb-12 font-mono">
          score(m, q) = 0.5·relevance + 0.3·importance + 0.1·recency −
          0.1·redundancy
        </p>

        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-12">
          {stats.map((s) => (
            <div
              key={s.label}
              className="bg-slate-900/60 border border-slate-800 rounded-xl p-4 backdrop-blur-sm"
            >
              <div className="text-3xl font-bold text-white mb-1">{s.value}</div>
              <div className="text-sm font-medium text-slate-300">{s.label}</div>
              <div className="text-xs text-slate-500 mt-0.5">{s.sub}</div>
            </div>
          ))}
        </div>

        <div className="flex items-center justify-center gap-4">
          <a
            href="#results"
            className="px-6 py-2.5 bg-blue-600 hover:bg-blue-500 text-white text-sm font-medium rounded-lg transition-colors"
          >
            View Results
          </a>
          <a
            href="#memories"
            className="px-6 py-2.5 border border-slate-700 hover:border-slate-500 text-slate-300 hover:text-white text-sm font-medium rounded-lg transition-colors"
          >
            Explore Memories
          </a>
        </div>
      </div>

      {/* Scroll indicator */}
      <div className="absolute bottom-8 left-1/2 -translate-x-1/2 flex flex-col items-center gap-1 text-slate-600">
        <span className="text-xs">scroll</span>
        <div className="w-px h-8 bg-gradient-to-b from-slate-600 to-transparent" />
      </div>
    </section>
  );
}
