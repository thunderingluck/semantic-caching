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
import tiktoken

from memory_store import MemoryStore
from memory_writer import MemoryObject

_TOKENIZER = tiktoken.get_encoding("cl100k_base")


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

    Base score: α·relevance + β·importance + γ·recency
    Selection: greedy MMR — each pick maximizes base_score minus λ·max_sim_to_selected,
    ensuring diversity across the token budget.
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

        scored = []
        for mem, rel, rec in zip(mems, relevances, recencies):
            score = (
                cfg.alpha * rel
                + cfg.beta * mem.importance
                + cfg.gamma * rec
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
        """
        Greedy MMR selection under token budget.

        At each step picks the candidate that maximizes:
            mmr_score = base_score - λ · max_cosine_sim_to_already_selected

        This correctly penalizes redundancy with respect to the *selected* set,
        not all other candidates — the standard Maximal Marginal Relevance algorithm
        (Carbonell & Goldstein 1998).
        """
        cfg = self.config
        selected: list[tuple[MemoryObject, float, float, float]] = []
        selected_embeddings: list[np.ndarray] = []
        used_tokens = 0
        remaining = list(scored)

        while remaining:
            best_idx = -1
            best_mmr = -float("inf")

            for i, (mem, base_score, rel, rec) in enumerate(remaining):
                tokens = _count_tokens(mem.fact)
                if used_tokens + tokens > token_budget:
                    continue

                if selected_embeddings and mem.embedding:
                    emb = np.array(mem.embedding)
                    norm = np.linalg.norm(emb)
                    if norm > 1e-9:
                        sims = [
                            float(np.dot(emb, s_emb) / (norm * np.linalg.norm(s_emb)))
                            for s_emb in selected_embeddings
                            if np.linalg.norm(s_emb) > 1e-9
                        ]
                        max_sim = max(sims) if sims else 0.0
                    else:
                        max_sim = 0.0
                else:
                    max_sim = 0.0

                mmr_score = base_score - cfg.lam * max_sim
                if mmr_score > best_mmr:
                    best_mmr = mmr_score
                    best_idx = i

            if best_idx == -1:
                break  # nothing fits in remaining budget

            mem, base_score, rel, rec = remaining.pop(best_idx)
            selected.append((mem, base_score, rel, rec))
            used_tokens += _count_tokens(mem.fact)
            emb = np.array(mem.embedding) if mem.embedding else np.zeros(384)
            selected_embeddings.append(emb)

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
# Token counting
# ──────────────────────────────────────────────

def _count_tokens(text: str) -> int:
    """Accurate token count using tiktoken cl100k_base (gpt-4o / gpt-4o-mini)."""
    return max(1, len(_TOKENIZER.encode(text)))


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


