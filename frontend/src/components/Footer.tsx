export default function Footer() {
  return (
    <footer className="border-t border-slate-900 py-10 px-6">
      <div className="max-w-6xl mx-auto flex flex-col md:flex-row items-center justify-between gap-4 text-xs text-slate-600">
        <div className="flex items-center gap-2">
          <span className="w-1.5 h-1.5 rounded-full bg-blue-500" />
          <span>Semantic Memory Cache · Hackathon 2026</span>
        </div>
        <div className="flex items-center gap-6">
          <span>LoCoMo dataset by Snap Research</span>
          <a
            href="https://arxiv.org/abs/2402.17753"
            target="_blank"
            rel="noopener noreferrer"
            className="hover:text-slate-400 transition-colors"
          >
            arxiv:2402.17753 ↗
          </a>
          <span>GPT-4o-mini · ChromaDB · all-MiniLM-L6-v2</span>
        </div>
      </div>
    </footer>
  );
}
