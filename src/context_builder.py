"""
context_builder.py — Retrieve and rank memories under a token budget.

Scoring formula:
  score(m, q) = α·relevance(m,q) + β·importance(m) + γ·recency(m) - λ·redundancy(m)

Default weights: α=0.5, β=0.3, γ=0.1, λ=0.1
Token budget: ~500 tokens
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import numpy as np

from memory_store import MemoryStore
from memory_writer import MemoryObject


@dataclass
class RetrievalConfig:
    alpha: float = 0.5      # relevance weight
    beta: float = 0.3       # importance weight
    gamma: float = 0.1      # recency weight
    lam: float = 0.1        # redundancy penalty
    token_budget: int = 500
    min_importance: float = 0.0
    max_candidates: int = 50  # how many to pull from vector search


class ContextBuilder:
    """
    Builds a curated context string from the memory store for a given query.

    Scoring:
      score = α·relevance + β·importance + γ·recency - λ·max_pairwise_similarity
    """

    def __init__(
        self,
        store: MemoryStore,
        config: Optional[RetrievalConfig] = None,
    ):
        self.store = store
        self.config = config or RetrievalConfig()

    # ── Public API ──────────────────────────────

    def build_context(
        self,
        query: str,
        session_id: Optional[str] = None,
        verbose: bool = False,
    ) -> tuple[str, list[dict]]:
        """
        Retrieve memories and build a context string under token budget.

        Returns:
          (context_string, list of scored memory dicts for debugging)
        """
        cfg = self.config

        # Step 1: vector search for candidates
        candidates = self.store.query(
            query_text=query,
            n_results=cfg.max_candidates,
            session_id=session_id,
            status="active",
            min_importance=cfg.min_importance,
        )
        if not candidates:
            return "", []

        # Step 2: score each candidate
        scored = self._score_candidates(candidates)

        # Step 3: greedy selection under token budget
        selected = self._select_under_budget(scored, cfg.token_budget)

        # Step 4: format context string
        context = self._format_context(selected)

        if verbose:
            print(f"[ContextBuilder] {len(candidates)} candidates → "
                  f"{len(selected)} selected ({_count_tokens(context)} tokens)")

        debug_info = [
            {
                "fact": m.fact,
                "type": m.type,
                "importance": m.importance,
                "score": s,
                "relevance": r,
                "recency": rec,
            }
            for m, s, r, rec in selected
        ]
        return context, debug_info

    def get_scored_memories(
        self,
        query: str,
        session_id: Optional[str] = None,
    ) -> list[tuple[MemoryObject, float]]:
        """
        Return (memory, final_score) sorted by score descending.
        Useful for evaluation and debugging.
        """
        cfg = self.config
        candidates = self.store.query(
            query_text=query,
            n_results=cfg.max_candidates,
            session_id=session_id,
            status="active",
        )
        scored = self._score_candidates(candidates)
        return [(m, s) for m, s, _, _ in scored]

    # ── Scoring ──────────────────────────────────

    def _score_candidates(
        self,
        candidates: list[tuple[MemoryObject, float]],
    ) -> list[tuple[MemoryObject, float, float, float]]:
        """
        Return (memory, final_score, relevance, recency) for each candidate.
        Relevance comes from the vector store (cosine similarity).
        Recency is computed from created_at timestamps.
        """
        cfg = self.config

        if not candidates:
            return []

        mems = [m for m, _ in candidates]
        relevances = [r for _, r in candidates]
        recencies = _compute_recencies([m.created_at for m in mems])

        # Collect embeddings for redundancy computation
        embeddings = []
        for m in mems:
            if m.embedding:
                embeddings.append(np.array(m.embedding))
            else:
                embeddings.append(np.zeros(384))

        scored = []
        for i, (mem, rel, rec) in enumerate(zip(mems, relevances, recencies)):
            redundancy = _max_pairwise_similarity(i, embeddings)
            score = (
                cfg.alpha * rel
                + cfg.beta * mem.importance
                + cfg.gamma * rec
                - cfg.lam * redundancy
            )
            scored.append((mem, score, rel, rec))

        # Sort by score descending
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    def _select_under_budget(
        self,
        scored: list[tuple[MemoryObject, float, float, float]],
        token_budget: int,
    ) -> list[tuple[MemoryObject, float, float, float]]:
        """Greedy selection: add memories in score order until budget is exhausted."""
        selected = []
        used_tokens = 0

        for mem, score, rel, rec in scored:
            tokens = _count_tokens(mem.fact)
            if used_tokens + tokens > token_budget:
                continue
            selected.append((mem, score, rel, rec))
            used_tokens += tokens

        return selected

    def _format_context(
        self, selected: list[tuple[MemoryObject, float, float, float]]
    ) -> str:
        """Format selected memories into a context string."""
        if not selected:
            return ""
        lines = ["[MEMORY CONTEXT]"]
        for i, (mem, score, _, _) in enumerate(selected, 1):
            lines.append(
                f"{i}. [{mem.type.upper()}] {mem.fact}  "
                f"(importance={mem.importance:.2f})"
            )
        return "\n".join(lines)


# ──────────────────────────────────────────────
# Token counting (approximate)
# ──────────────────────────────────────────────

def _count_tokens(text: str) -> int:
    """Approximate token count: ~4 chars per token."""
    return max(1, len(text) // 4)


# ──────────────────────────────────────────────
# Recency scoring
# ──────────────────────────────────────────────

def _compute_recencies(timestamps: list[str]) -> list[float]:
    """
    Convert ISO timestamps to recency scores in [0, 1].
    Most recent = 1.0, oldest = 0.0.
    """
    times = []
    now = datetime.now(timezone.utc)
    for ts in timestamps:
        try:
            dt = datetime.fromisoformat(ts)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            times.append((now - dt).total_seconds())
        except (ValueError, TypeError):
            times.append(0.0)

    if not times:
        return []

    max_age = max(times) or 1.0
    return [1.0 - (t / max_age) for t in times]


# ──────────────────────────────────────────────
# Redundancy penalty
# ──────────────────────────────────────────────

def _max_pairwise_similarity(
    idx: int, embeddings: list[np.ndarray]
) -> float:
    """
    Compute max cosine similarity between memory[idx] and all others.
    Used as a redundancy penalty.
    """
    if len(embeddings) <= 1:
        return 0.0
    emb = embeddings[idx]
    norm = np.linalg.norm(emb)
    if norm < 1e-9:
        return 0.0

    others = [embeddings[j] for j in range(len(embeddings)) if j != idx]
    sims = []
    for other in others:
        other_norm = np.linalg.norm(other)
        if other_norm < 1e-9:
            continue
        sim = float(np.dot(emb, other) / (norm * other_norm))
        sims.append(sim)

    return max(sims) if sims else 0.0
