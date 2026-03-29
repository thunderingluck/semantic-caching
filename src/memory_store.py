"""
memory_store.py — Chroma-backed vector store for MemoryObjects.

Supports:
  - add / batch add
  - similarity search by query text or vector
  - filter by session / status / type
  - mark memories as stale/archived
  - persist to disk (optional)
"""

import json
import uuid
from pathlib import Path
from typing import Optional

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

from memory_writer import MemoryObject


class MemoryStore:
    """
    Wraps a Chroma collection to store and retrieve MemoryObjects.

    Args:
        collection_name: name of the Chroma collection
        embedder: SentenceTransformer used to embed query strings
        persist_dir: if set, persist Chroma to disk at this path (in-memory otherwise)
    """

    def __init__(
        self,
        collection_name: str = "memories",
        embedder: Optional[SentenceTransformer] = None,
        persist_dir: Optional[str | Path] = None,
    ):
        self.embedder = embedder or SentenceTransformer("all-MiniLM-L6-v2")

        if persist_dir:
            self._client = chromadb.PersistentClient(path=str(persist_dir))
        else:
            self._client = chromadb.Client()

        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    # ── Write ────────────────────────────────────

    def add(self, memory: MemoryObject) -> None:
        """Add a single MemoryObject to the store."""
        self._collection.add(
            ids=[memory.id],
            embeddings=[memory.embedding] if memory.embedding else None,
            documents=[memory.fact],
            metadatas=[self._to_metadata(memory)],
        )

    def add_batch(self, memories: list[MemoryObject]) -> None:
        """Add a list of MemoryObjects in one batch operation."""
        if not memories:
            return
        ids = [m.id for m in memories]
        embeddings = [m.embedding for m in memories if m.embedding is not None]
        documents = [m.fact for m in memories]
        metadatas = [self._to_metadata(m) for m in memories]

        # If some memories lack embeddings, compute them
        if len(embeddings) != len(memories):
            texts = [m.fact for m in memories]
            computed = self.embedder.encode(texts, normalize_embeddings=True)
            embeddings = [e.tolist() for e in computed]
            for mem, emb in zip(memories, computed):
                mem.embedding = emb.tolist()

        self._collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

    # ── Read ─────────────────────────────────────

    def query(
        self,
        query_text: str,
        n_results: int = 20,
        session_id: Optional[str] = None,
        status: str = "active",
        min_importance: float = 0.0,
    ) -> list[tuple[MemoryObject, float]]:
        """
        Retrieve top-k most relevant memories for a query.

        Returns list of (MemoryObject, similarity_score) tuples.
        similarity_score is in [0, 1] (cosine, higher = more similar).
        """
        query_emb = self.embedder.encode([query_text], normalize_embeddings=True)[0]

        where_filter = self._build_where(session_id=session_id, status=status)
        total = self._collection.count()
        if total == 0:
            return []

        n = min(n_results, total)
        results = self._collection.query(
            query_embeddings=[query_emb.tolist()],
            n_results=n,
            where=where_filter if where_filter else None,
            include=["metadatas", "documents", "distances"],
        )

        memories_scores = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            mem = self._from_metadata(meta, doc)
            if mem.importance < min_importance:
                continue
            # Chroma cosine distance: 0 = identical, 2 = opposite
            similarity = 1 - (dist / 2)
            memories_scores.append((mem, float(similarity)))

        return memories_scores

    def get_by_session(
        self, session_id: str, status: str = "active"
    ) -> list[MemoryObject]:
        """Get all memories for a session, including embeddings for deduplication."""
        where = {"$and": [{"session_id": session_id}, {"status": status}]}
        total = self._collection.count()
        if total == 0:
            return []
        results = self._collection.get(
            where=where,
            include=["metadatas", "documents", "embeddings"],
        )
        mems = []
        for m, d, e in zip(results["metadatas"], results["documents"], results["embeddings"]):
            mem = self._from_metadata(m, d)
            if e is not None:
                mem.embedding = e if isinstance(e, list) else e.tolist()
            mems.append(mem)
        return mems

    def count(self) -> int:
        return self._collection.count()

    # ── Update ───────────────────────────────────

    def update_status(self, memory_id: str, status: str) -> None:
        """Update the status of a memory (active → stale → archived)."""
        self._collection.update(
            ids=[memory_id],
            metadatas=[{"status": status}],
        )

    def mark_stale(self, memory_id: str) -> None:
        self.update_status(memory_id, "stale")

    def archive(self, memory_id: str) -> None:
        self.update_status(memory_id, "archived")

    # ── Staleness detection ──────────────────────

    def find_contradictions(
        self, new_fact: str, session_id: str, threshold: float = 0.85
    ) -> list[MemoryObject]:
        """
        Find existing memories that are semantically very similar to a new fact
        — potential contradictions that should be marked stale.
        """
        results = self.query(
            query_text=new_fact,
            n_results=5,
            session_id=session_id,
        )
        return [mem for mem, score in results if score >= threshold]

    # ── Helpers ──────────────────────────────────

    def _to_metadata(self, mem: MemoryObject) -> dict:
        """Convert MemoryObject to Chroma metadata dict (all values must be str/int/float/bool)."""
        return {
            "mem_id": mem.id,
            "type": mem.type,
            "importance": mem.importance,
            "persistence": mem.persistence,
            "scope": mem.scope,
            "source_turns": json.dumps(mem.source_turns),
            "source_text": mem.source_text[:500],
            "status": mem.status,
            "created_at": mem.created_at,
            "session_id": mem.session_id,
        }

    def _from_metadata(self, meta: dict, document: str) -> MemoryObject:
        source_turns = []
        try:
            source_turns = json.loads(meta.get("source_turns", "[]"))
        except (json.JSONDecodeError, TypeError):
            pass

        return MemoryObject(
            id=meta.get("mem_id", str(uuid.uuid4())),
            fact=document,
            type=meta.get("type", "fact"),
            importance=float(meta.get("importance", 0.5)),
            persistence=meta.get("persistence", "medium"),
            scope=meta.get("scope", "session"),
            source_turns=source_turns,
            source_text=meta.get("source_text", ""),
            status=meta.get("status", "active"),
            created_at=meta.get("created_at", ""),
            session_id=meta.get("session_id", ""),
        )

    def _build_where(
        self,
        session_id: Optional[str],
        status: Optional[str],
    ) -> Optional[dict]:
        conditions = []
        if session_id:
            conditions.append({"session_id": {"$eq": session_id}})
        if status:
            conditions.append({"status": {"$eq": status}})

        if len(conditions) == 0:
            return None
        if len(conditions) == 1:
            return conditions[0]
        return {"$and": conditions}
