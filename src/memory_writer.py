"""
memory_writer.py — Extract atomic facts from conversation turns and score their importance.

Pipeline:
  1. Slide a window over conversation turns
  2. Call OpenAI to extract atomic facts (JSON)
  3. Call OpenAI to score importance + persistence
  4. Deduplicate against existing memories via cosine similarity
  5. Return list of MemoryObject
"""

import json
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional

from openai import OpenAI
import numpy as np
from sentence_transformers import SentenceTransformer

from ingestor import Conversation, Turn


# ──────────────────────────────────────────────
# Data model
# ──────────────────────────────────────────────

@dataclass
class MemoryObject:
    id: str
    fact: str
    type: str                  # fact | preference | constraint | decision | definition | goal
    importance: float          # 0.0–1.0
    persistence: str           # ephemeral | medium | long_term
    scope: str                 # session | project | user
    source_turns: list[int]
    source_text: str
    status: str                # active | stale | archived
    created_at: str
    session_id: str
    embedding: Optional[list[float]] = field(default=None, repr=False)

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("embedding", None)
        return d


# ──────────────────────────────────────────────
# Prompt loading
# ──────────────────────────────────────────────

_PROMPTS_DIR = Path(__file__).parent.parent / "prompts"

def _load_prompt(name: str) -> str:
    return (_PROMPTS_DIR / name).read_text(encoding="utf-8")


# ──────────────────────────────────────────────
# Memory writer
# ──────────────────────────────────────────────

class MemoryWriter:
    """
    Extracts and scores memories from conversation windows.

    Args:
        client: OpenAI client instance
        model: OpenAI model to use for extraction/scoring
        embedder: SentenceTransformer model for deduplication
        window_size: number of turns per extraction window
        stride: step size for the sliding window
        importance_threshold: discard memories below this score
        dedup_threshold: cosine similarity above which we consider two facts duplicates
    """

    def __init__(
        self,
        client: OpenAI,
        model: str = "gpt-4o-mini",
        embedder: Optional[SentenceTransformer] = None,
        window_size: int = 4,
        stride: int = 2,
        importance_threshold: float = 0.3,
        dedup_threshold: float = 0.92,
    ):
        self.client = client
        self.model = model
        self.embedder = embedder or SentenceTransformer("all-MiniLM-L6-v2")
        self.window_size = window_size
        self.stride = stride
        self.importance_threshold = importance_threshold
        self.dedup_threshold = dedup_threshold

        self._extraction_prompt = _load_prompt("extraction.txt")
        self._importance_prompt = _load_prompt("importance.txt")

    # ── Public API ──────────────────────────────

    def process_conversation(
        self, conversation: Conversation, verbose: bool = False
    ) -> list[MemoryObject]:
        """
        Extract memories from an entire conversation using a sliding window.
        Returns deduplicated, importance-filtered MemoryObjects.
        """
        all_memories: list[MemoryObject] = []
        turns = conversation.turns

        windows = self._make_windows(turns)
        n_windows = len(windows)

        for i, window_turns in enumerate(windows):
            if verbose:
                print(f"\r  window {i+1}/{n_windows} ...", end="", flush=True)
            raw_facts = self._extract_facts(window_turns, verbose=False)
            if not raw_facts:
                continue

            scored = self._score_importance(raw_facts, verbose=False)
            turn_ids = [t.turn_id for t in window_turns]

            for fact_data, score_data in zip(raw_facts, scored):
                importance = score_data.get("importance", 0.0)
                if importance < self.importance_threshold:
                    continue

                mem = self._build_memory_object(
                    fact_data=fact_data,
                    score_data=score_data,
                    source_turns=turn_ids,
                    session_id=conversation.session_id,
                )
                all_memories.append(mem)

        if verbose:
            print()  # newline after progress line

        # Embed all memories
        self._embed_memories(all_memories)

        # Deduplicate
        unique_memories = self._deduplicate(all_memories)

        if verbose:
            print(f"[MemoryWriter] session={conversation.session_id}: "
                  f"{len(turns)} turns → {len(all_memories)} raw → "
                  f"{len(unique_memories)} after dedup")

        return unique_memories

    # ── Private helpers ─────────────────────────

    def _make_windows(self, turns: list[Turn]) -> list[list[Turn]]:
        """Slide a window over turns with given stride."""
        windows = []
        i = 0
        while i < len(turns):
            window = turns[i: i + self.window_size]
            windows.append(window)
            i += self.stride
        return windows

    def _extract_facts(
        self, turns: list[Turn], verbose: bool = False
    ) -> list[dict]:
        """Call OpenAI to extract atomic facts from a window of turns."""
        formatted = "\n".join(
            f"[Turn {t.turn_id}] {t.speaker}: {t.text}" for t in turns
        )
        prompt = self._extraction_prompt.replace("{turns}", formatted)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}],
            )
            text = response.choices[0].message.content.strip()
            facts = _parse_json_response(text)
            if not isinstance(facts, list):
                return []
            return facts
        except Exception as e:
            if verbose:
                print(f"[MemoryWriter] extraction error: {e}")
            return []

    def _score_importance(
        self, facts: list[dict], verbose: bool = False
    ) -> list[dict]:
        """Call OpenAI to score importance + persistence for each fact."""
        facts_text = json.dumps(
            [{"fact": f.get("fact", ""), "type": f.get("type", "fact")} for f in facts],
            indent=2,
        )
        prompt = self._importance_prompt.replace("{facts}", facts_text)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                max_tokens=512,
                messages=[{"role": "user", "content": prompt}],
            )
            text = response.choices[0].message.content.strip()
            scores = _parse_json_response(text)
            if not isinstance(scores, list) or len(scores) != len(facts):
                # Fallback: default scores
                return [{"importance": 0.5, "persistence": "medium"}] * len(facts)
            return scores
        except Exception as e:
            if verbose:
                print(f"[MemoryWriter] scoring error: {e}")
            return [{"importance": 0.5, "persistence": "medium"}] * len(facts)

    def _build_memory_object(
        self,
        fact_data: dict,
        score_data: dict,
        source_turns: list[int],
        session_id: str,
    ) -> MemoryObject:
        fact_type = fact_data.get("type", "fact")
        persistence = score_data.get("persistence", "medium")

        # Infer scope from persistence
        scope_map = {"long_term": "user", "medium": "project", "ephemeral": "session"}
        scope = scope_map.get(persistence, "session")

        return MemoryObject(
            id=f"mem_{uuid.uuid4().hex[:8]}",
            fact=fact_data.get("fact", ""),
            type=fact_type,
            importance=float(score_data.get("importance", 0.5)),
            persistence=persistence,
            scope=scope,
            source_turns=source_turns,
            source_text=fact_data.get("source_text", ""),
            status="active",
            created_at=datetime.now(timezone.utc).isoformat(),
            session_id=session_id,
        )

    def _embed_memories(self, memories: list[MemoryObject]) -> None:
        """Compute and attach embeddings in-place."""
        if not memories:
            return
        texts = [m.fact for m in memories]
        embeddings = self.embedder.encode(texts, normalize_embeddings=True)
        for mem, emb in zip(memories, embeddings):
            mem.embedding = emb.tolist()

    def _deduplicate(self, memories: list[MemoryObject]) -> list[MemoryObject]:
        """
        Remove near-duplicate memories by cosine similarity.
        When duplicates are found, keep the one with higher importance.
        """
        if not memories:
            return []

        kept: list[MemoryObject] = []
        kept_embeddings: list[np.ndarray] = []

        # Sort by importance descending so we keep the best version
        sorted_mems = sorted(memories, key=lambda m: m.importance, reverse=True)

        for mem in sorted_mems:
            if mem.embedding is None:
                kept.append(mem)
                kept_embeddings.append(np.zeros(384))
                continue

            emb = np.array(mem.embedding)
            if kept_embeddings:
                sims = np.array(kept_embeddings) @ emb
                if sims.max() >= self.dedup_threshold:
                    continue  # duplicate — skip

            kept.append(mem)
            kept_embeddings.append(emb)

        return kept


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def _parse_json_response(text: str) -> any:
    """Extract and parse JSON from an LLM response, tolerating markdown fences."""
    # Strip markdown code fences
    text = re.sub(r"```(?:json)?\s*", "", text).strip().rstrip("```").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to find JSON array or object
        match = re.search(r"(\[.*\]|\{.*\})", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass
    return None
