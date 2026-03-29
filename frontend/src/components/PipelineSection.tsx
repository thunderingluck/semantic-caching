const steps = [
  {
    num: "01",
    title: "Memory Writer",
    subtitle: "Admission policy",
    description:
      "Slides a 6-turn window over the conversation. Calls GPT-4o-mini to extract atomic facts and score their importance (0–1). Deduplicates at cosine similarity > 0.92.",
    tag: "gpt-4o-mini",
    color: "blue",
    io: { in: "419 turns", out: "687 raw facts" },
  },
  {
    num: "02",
    title: "Deduplication",
    subtitle: "Cosine similarity filter",
    description:
      "Embeds all facts with all-MiniLM-L6-v2. Removes near-duplicates using greedy cosine similarity comparison, keeping the highest-importance version.",
    tag: "sentence-transformers",
    color: "violet",
    io: { in: "687 raw facts", out: "405 unique memories" },
  },
  {
    num: "03",
    title: "Memory Store",
    subtitle: "Vector database",
    description:
      "Stores MemoryObjects in Chroma with embeddings, importance scores, persistence tags (ephemeral / medium / long-term), and source turn IDs.",
    tag: "chromadb",
    color: "cyan",
    io: { in: "405 memories", out: "Chroma collection" },
  },
  {
    num: "04",
    title: "Context Builder",
    subtitle: "Retrieval under budget",
    description:
      "Scores candidates with α·relevance + β·importance + γ·recency − λ·redundancy. Greedy MMR selection fills the 500-token budget with diverse, high-utility facts.",
    tag: "token budget: 500",
    color: "green",
    io: { in: "Query + store", out: "~500 token context" },
  },
];

const colorMap: Record<string, string> = {
  blue: "border-blue-900/60 bg-blue-950/20",
  violet: "border-violet-900/60 bg-violet-950/20",
  cyan: "border-cyan-900/60 bg-cyan-950/20",
  green: "border-green-900/60 bg-green-950/20",
};

const numColorMap: Record<string, string> = {
  blue: "text-blue-400",
  violet: "text-violet-400",
  cyan: "text-cyan-400",
  green: "text-green-400",
};

const tagColorMap: Record<string, string> = {
  blue: "bg-blue-900/50 text-blue-300 border-blue-800",
  violet: "bg-violet-900/50 text-violet-300 border-violet-800",
  cyan: "bg-cyan-900/50 text-cyan-300 border-cyan-800",
  green: "bg-green-900/50 text-green-300 border-green-800",
};

export default function PipelineSection() {
  return (
    <section id="pipeline" className="py-24 px-6 bg-slate-950/50">
      <div className="max-w-6xl mx-auto">
        <div className="text-center mb-16">
          <p className="text-blue-400 text-sm font-medium mb-3 tracking-wide uppercase">
            Architecture
          </p>
          <h2 className="text-3xl md:text-4xl font-bold text-white mb-4">
            Four-stage memory pipeline
          </h2>
          <p className="text-slate-400 max-w-xl mx-auto">
            Each stage is independently replaceable. The core contribution is
            the memory policy — what to store, how to score it, and what to
            retrieve.
          </p>
        </div>

        {/* Formula */}
        <div className="mb-12 bg-slate-900/60 border border-slate-800 rounded-2xl p-6 text-center max-w-2xl mx-auto">
          <p className="text-xs text-slate-500 mb-3 uppercase tracking-widest">
            Retrieval scoring formula
          </p>
          <p className="font-mono text-slate-200 text-sm md:text-base">
            score(m, q) ={" "}
            <span className="text-blue-400">0.5 · relevance(m,q)</span> +{" "}
            <span className="text-violet-400">0.3 · importance(m)</span> +{" "}
            <span className="text-cyan-400">0.1 · recency(m)</span> −{" "}
            <span className="text-red-400">0.1 · redundancy(m)</span>
          </p>
          <p className="text-xs text-slate-600 mt-3">
            Greedy MMR selection — each pick maximizes marginal relevance minus
            similarity to already-selected memories
          </p>
        </div>

        {/* Steps */}
        <div className="grid md:grid-cols-4 gap-4">
          {steps.map((step, i) => (
            <div key={step.num} className="flex flex-col">
              <div
                className={`flex-1 border rounded-2xl p-5 ${colorMap[step.color]}`}
              >
                <div className="flex items-start justify-between mb-4">
                  <span
                    className={`text-3xl font-bold font-mono ${numColorMap[step.color]}`}
                  >
                    {step.num}
                  </span>
                  <span
                    className={`text-xs px-2 py-0.5 rounded-md border font-mono ${tagColorMap[step.color]}`}
                  >
                    {step.tag}
                  </span>
                </div>
                <h3 className="text-white font-semibold mb-0.5">{step.title}</h3>
                <p className="text-xs text-slate-500 mb-3">{step.subtitle}</p>
                <p className="text-sm text-slate-400 leading-relaxed">
                  {step.description}
                </p>
              </div>

              {/* I/O badges */}
              <div className="mt-3 space-y-1.5">
                <div className="flex items-center gap-2 text-xs text-slate-500">
                  <span className="text-slate-600">IN</span>
                  <span className="font-mono text-slate-400">{step.io.in}</span>
                </div>
                <div className="flex items-center gap-2 text-xs text-slate-500">
                  <span className="text-slate-600">OUT</span>
                  <span className={`font-mono ${numColorMap[step.color]}`}>
                    {step.io.out}
                  </span>
                </div>
              </div>

              {/* Arrow connector */}
              {i < steps.length - 1 && (
                <div className="hidden md:block absolute" />
              )}
            </div>
          ))}
        </div>

        {/* RLVR callout */}
        <div className="mt-10 border border-slate-700/60 rounded-2xl p-6 bg-gradient-to-r from-slate-900/60 to-violet-950/20">
          <div className="flex items-start gap-4">
            <div className="w-10 h-10 rounded-xl bg-violet-900/50 border border-violet-800 flex items-center justify-center flex-shrink-0">
              <span className="text-violet-400 text-lg">⟳</span>
            </div>
            <div>
              <h3 className="text-white font-semibold mb-1">
                RLVR learning loop
              </h3>
              <p className="text-slate-400 text-sm leading-relaxed max-w-3xl">
                QA outcomes generate a reward signal (
                <span className="text-green-400 font-mono">+1</span> correct
                answer, <span className="text-orange-400 font-mono">−0.5</span>{" "}
                evidence missed,{" "}
                <span className="text-red-400 font-mono">−1</span> evidence
                never stored) that feeds back into importance scores — enabling
                the admission policy to learn which facts have the highest future
                utility. This is temporal credit assignment over what to
                remember.
              </p>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
