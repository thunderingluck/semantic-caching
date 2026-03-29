"use client";

import { useState, useEffect } from "react";

const links = [
  { href: "#problem", label: "Problem" },
  { href: "#pipeline", label: "Pipeline" },
  { href: "#results", label: "Results" },
  { href: "#memories", label: "Memory Explorer" },
];

export default function Nav() {
  const [scrolled, setScrolled] = useState(false);

  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 20);
    window.addEventListener("scroll", onScroll);
    return () => window.removeEventListener("scroll", onScroll);
  }, []);

  return (
    <nav
      className={`fixed top-0 left-0 right-0 z-50 transition-all duration-300 ${
        scrolled
          ? "bg-slate-950/90 backdrop-blur-md border-b border-slate-800"
          : "bg-transparent"
      }`}
    >
      <div className="max-w-6xl mx-auto px-6 h-14 flex items-center justify-between">
        <a href="#" className="flex items-center gap-2">
          <span className="w-2 h-2 rounded-full bg-blue-500 animate-pulse" />
          <span className="text-sm font-semibold text-white tracking-tight">
            Semantic Memory Cache
          </span>
        </a>
        <div className="flex items-center gap-6">
          {links.map((l) => (
            <a
              key={l.href}
              href={l.href}
              className="text-sm text-slate-400 hover:text-white transition-colors"
            >
              {l.label}
            </a>
          ))}
          <a
            href="https://github.com/snap-research/locomo"
            target="_blank"
            rel="noopener noreferrer"
            className="text-xs px-3 py-1.5 rounded-md border border-slate-700 text-slate-300 hover:border-slate-500 hover:text-white transition-all"
          >
            LoCoMo Dataset ↗
          </a>
        </div>
      </div>
    </nav>
  );
}
