"""
baselines.py — Baseline context construction strategies for comparison.

Baselines:
  1. FullHistory     — all conversation turns concatenated (truncated to budget)
  2. RollingSummary  — ChatGPT-generated rolling summary of the conversation
  3. NaiveRAG        — turn-level TF-IDF or embedding retrieval, no importance scoring
"""

import json
from typing import Optional
from collections import Counter
import math

from openai import OpenAI
import numpy as np
from sentence_transformers import SentenceTransformer

from ingestor import Conversation, Turn
from context_builder import _count_tokens


# ──────────────────────────────────────────────
# 1. Full History
# ──────────────────────────────────────────────

class FullHistoryBaseline:
    """
    Uses the raw conversation history as context.
    Truncates to the last N tokens if over budget (keeps most recent turns).
    """

    def __init__(self, token_budget: int = 2000):
        self.token_budget = token_budget

    def build_context(self, conversation: Conversation, query: str) -> str:
        all_lines = [
            f"[Turn {t.turn_id}] {t.speaker}: {t.text}"
            for t in conversation.turns
        ]
        # Keep as many turns as fit in budget (from the end, most recent first)
        selected = []
        budget = self.token_budget
        for line in reversed(all_lines):
            tokens = _count_tokens(line)
            if budget - tokens < 0:
                break
            selected.insert(0, line)
            budget -= tokens

        return "\n".join(selected)

    def name(self) -> str:
        return "full_history"


# ──────────────────────────────────────────────
# 2. Rolling Summary
# ──────────────────────────────────────────────

_SUMMARY_PROMPT = """You are summarizing a conversation for future reference.

Write a concise factual summary of the key information from this conversation.
Focus on: decisions made, preferences stated, facts established, goals mentioned.
Ignore greetings and chitchat.

Keep the summary under {max_tokens} tokens.

CONVERSATION:
{conversation}

SUMMARY:"""


class RollingSummaryBaseline:
    """
    Builds a rolling summary of the conversation using ChatGPT.
    Summarizes in chunks, then summarizes the summaries.
    """

    def __init__(
        self,
        client: OpenAI,
        model: str = "gpt-4o-mini",
        token_budget: int = 500,
        chunk_size: int = 20,  # turns per chunk
    ):
        self.client = client
        self.model = model
        self.token_budget = token_budget
        self.chunk_size = chunk_size

    def build_context(self, conversation: Conversation, query: str) -> str:
        """Generate a rolling summary of the conversation."""
        turns = conversation.turns
        if not turns:
            return ""

        # Split into chunks
        chunks = [
            turns[i: i + self.chunk_size]
            for i in range(0, len(turns), self.chunk_size)
        ]

        summaries = []
        for chunk in chunks:
            conv_text = "\n".join(
                f"{t.speaker}: {t.text}" for t in chunk
            )
            prompt = _SUMMARY_PROMPT.format(
                max_tokens=self.token_budget // max(len(chunks), 1),
                conversation=conv_text,
            )
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    max_tokens=self.token_budget // max(len(chunks), 1) * 4,
                    messages=[{"role": "user", "content": prompt}],
                )
                summaries.append(response.choices[0].message.content.strip())
            except Exception as e:
                summaries.append(f"[Summary error: {e}]")

        if len(summaries) == 1:
            return summaries[0]

        # Consolidate multiple summaries
        combined = "\n\n".join(f"Part {i+1}: {s}" for i, s in enumerate(summaries))
        prompt = _SUMMARY_PROMPT.format(
            max_tokens=self.token_budget,
            conversation=combined,
        )
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                max_tokens=self.token_budget * 4,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.choices[0].message.content.strip()
        except Exception:
            return combined[: self.token_budget * 4]

    def name(self) -> str:
        return "rolling_summary"


# ──────────────────────────────────────────────
# 3. Naive RAG
# ──────────────────────────────────────────────

class NaiveRAGBaseline:
    """
    Turn-level retrieval: embed each turn, retrieve top-k most similar to query.
    No importance scoring — pure semantic similarity.
    """

    def __init__(
        self,
        embedder: Optional[SentenceTransformer] = None,
        token_budget: int = 500,
        n_turns: int = 10,
    ):
        self.embedder = embedder or SentenceTransformer("all-MiniLM-L6-v2")
        self.token_budget = token_budget
        self.n_turns = n_turns
        # Cache embeddings per session
        self._cache: dict[str, tuple[list[Turn], np.ndarray]] = {}

    def build_context(self, conversation: Conversation, query: str) -> str:
        turns = conversation.turns
        if not turns:
            return ""

        # Embed turns (cache by session)
        if conversation.session_id not in self._cache:
            texts = [f"{t.speaker}: {t.text}" for t in turns]
            embeddings = self.embedder.encode(texts, normalize_embeddings=True)
            self._cache[conversation.session_id] = (turns, embeddings)

        cached_turns, turn_embeddings = self._cache[conversation.session_id]

        # Embed query
        query_emb = self.embedder.encode([query], normalize_embeddings=True)[0]

        # Score all turns
        scores = turn_embeddings @ query_emb

        # Take top-n by score
        top_indices = np.argsort(scores)[::-1][: self.n_turns]
        # Sort by turn order for readability
        top_indices = sorted(top_indices)

        # Build context under budget
        selected_lines = []
        used = 0
        for idx in top_indices:
            t = cached_turns[idx]
            line = f"[Turn {t.turn_id}] {t.speaker}: {t.text}"
            tokens = _count_tokens(line)
            if used + tokens > self.token_budget:
                continue
            selected_lines.append(line)
            used += tokens

        return "\n".join(selected_lines)

    def name(self) -> str:
        return "naive_rag"
