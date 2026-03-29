"""
evaluator.py — Evaluate the memory pipeline against baselines on LoCoMo QA.

Metrics:
  - memory_recall:      fraction of evidence turns with at least one extracted memory
  - retrieval_precision: fraction of retrieved memories that overlap with evidence turns
  - qa_accuracy:        exact match + F1 between predicted and gold answers
  - token_efficiency:   avg context tokens used

Reward signal (RLVR framing):
  +1 if retrieved memory enables correct answer
  -1 if critical memory was missing
  -0.5 if context is polluted (irrelevant memories retrieved)
"""

import json
import re
import string
from collections import Counter
from dataclasses import dataclass, field, asdict
from typing import Optional, Callable

from openai import OpenAI

from ingestor import Conversation, QAPair
from memory_store import MemoryStore
from context_builder import ContextBuilder
from memory_writer import MemoryObject


# ──────────────────────────────────────────────
# Data classes
# ──────────────────────────────────────────────

@dataclass
class QAResult:
    question_id: str
    session_id: str
    question: str
    gold_answer: str
    predicted_answer: str
    context_used: str
    context_tokens: int
    exact_match: bool
    f1: float
    method: str
    memory_recall: Optional[float] = None
    retrieval_precision: Optional[float] = None
    reward: Optional[float] = None


@dataclass
class EvalSummary:
    method: str
    num_questions: int
    exact_match_rate: float
    avg_f1: float
    avg_context_tokens: float
    avg_memory_recall: Optional[float] = None
    avg_retrieval_precision: Optional[float] = None
    avg_reward: Optional[float] = None

    def to_dict(self) -> dict:
        return asdict(self)


# ──────────────────────────────────────────────
# Answering
# ──────────────────────────────────────────────

_ANSWER_PROMPT_PATH = None  # loaded lazily

def _load_answering_prompt() -> str:
    from pathlib import Path
    p = Path(__file__).parent.parent / "prompts" / "answering.txt"
    return p.read_text(encoding="utf-8")


def answer_question(
    client: OpenAI,
    question: str,
    context: str,
    model: str = "gpt-4o-mini",
) -> str:
    """Answer a question given a memory context."""
    global _ANSWER_PROMPT_PATH
    if _ANSWER_PROMPT_PATH is None:
        _ANSWER_PROMPT_PATH = _load_answering_prompt()

    prompt = _ANSWER_PROMPT_PATH.replace("{memory_context}", context or "[No context]")
    prompt = prompt.replace("{question}", question)

    try:
        response = client.chat.completions.create(
            model=model,
            max_tokens=256,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"[Error: {e}]"


# ──────────────────────────────────────────────
# Memory-specific metrics
# ──────────────────────────────────────────────

def compute_memory_recall(
    evidence_turns: list[int],
    memories: list[MemoryObject],
) -> float:
    """
    What fraction of evidence turns have at least one memory covering them?
    A memory 'covers' a turn if that turn is in its source_turns.
    """
    if not evidence_turns:
        return 1.0  # nothing to recall
    covered = sum(
        1 for et in evidence_turns
        if any(et in m.source_turns for m in memories)
    )
    return covered / len(evidence_turns)


def compute_retrieval_precision(
    evidence_turns: list[int],
    retrieved_memories: list[MemoryObject],
) -> float:
    """
    What fraction of retrieved memories are relevant (overlap with evidence turns)?
    """
    if not retrieved_memories:
        return 0.0
    relevant = sum(
        1 for m in retrieved_memories
        if any(et in m.source_turns for et in evidence_turns)
    )
    return relevant / len(retrieved_memories)


def compute_reward(
    qa_result: QAResult,
    evidence_turns: list[int],
    retrieved_memories: list[MemoryObject],
    all_memories: list[MemoryObject],
) -> float:
    """
    RLVR-style reward:
      +1  correct answer AND evidence was retrieved
      -1  incorrect AND critical memory was missing from store
      -0.5 incorrect AND memory existed but wasn't retrieved
       0  otherwise
    """
    is_correct = qa_result.f1 >= 0.5

    # Check if evidence was covered in memory store at all
    store_covered = all(
        any(et in m.source_turns for m in all_memories)
        for et in evidence_turns
    ) if evidence_turns else True

    # Check if retrieved memories covered evidence
    retrieval_covered = any(
        et in m.source_turns
        for m in retrieved_memories
        for et in evidence_turns
    ) if evidence_turns else True

    if is_correct and retrieval_covered:
        return 1.0
    if not is_correct and not store_covered:
        return -1.0
    if not is_correct and store_covered and not retrieval_covered:
        return -0.5
    return 0.0


# ──────────────────────────────────────────────
# Answer quality metrics
# ──────────────────────────────────────────────

def normalize_answer(s: str) -> str:
    """Lower, remove punctuation and articles."""
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = "".join(c for c in s if c not in string.punctuation)
    return " ".join(s.split())


def exact_match(pred: str, gold: str) -> bool:
    return normalize_answer(pred) == normalize_answer(gold)


def token_f1(pred: str, gold: str) -> float:
    pred_tokens = normalize_answer(pred).split()
    gold_tokens = normalize_answer(gold).split()
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)

    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


# ──────────────────────────────────────────────
# Main evaluator
# ──────────────────────────────────────────────

class Evaluator:
    """
    Runs a context-building method over LoCoMo QA pairs and computes metrics.

    Usage:
        evaluator = Evaluator(client, store, context_builder)
        results = evaluator.run(conversations, method_name="semantic_cache")
        summary = evaluator.summarize(results)
    """

    def __init__(
        self,
        client: OpenAI,
        store: Optional[MemoryStore] = None,
        context_builder: Optional[ContextBuilder] = None,
        answer_model: str = "gpt-4o-mini",
    ):
        self.client = client
        self.store = store
        self.context_builder = context_builder
        self.answer_model = answer_model

    def run_semantic_cache(
        self,
        conversations: list[Conversation],
        all_memories_by_session: dict[str, list[MemoryObject]],
        verbose: bool = False,
    ) -> list[QAResult]:
        """Evaluate the semantic cache method."""
        results = []

        for conv in conversations:
            session_memories = all_memories_by_session.get(conv.session_id, [])

            for qa in conv.qa_pairs:
                # Build context
                context, debug = self.context_builder.build_context(
                    query=qa.question,
                    session_id=conv.session_id,
                    verbose=False,
                )

                # Get retrieved memories
                retrieved = self.store.query(
                    query_text=qa.question,
                    session_id=conv.session_id,
                    n_results=20,
                )
                retrieved_mems = [m for m, _ in retrieved]

                # Answer
                predicted = answer_question(
                    self.client, qa.question, context, self.answer_model
                )

                em = exact_match(predicted, qa.answer)
                f1 = token_f1(predicted, qa.answer)
                ctx_tokens = len(context) // 4

                mem_recall = compute_memory_recall(qa.evidence_turn_ids, session_memories)
                ret_prec = compute_retrieval_precision(qa.evidence_turn_ids, retrieved_mems)

                result = QAResult(
                    question_id=qa.question_id,
                    session_id=conv.session_id,
                    question=qa.question,
                    gold_answer=qa.answer,
                    predicted_answer=predicted,
                    context_used=context,
                    context_tokens=ctx_tokens,
                    exact_match=em,
                    f1=f1,
                    method="semantic_cache",
                    memory_recall=mem_recall,
                    retrieval_precision=ret_prec,
                )
                result.reward = compute_reward(
                    result, qa.evidence_turn_ids, retrieved_mems, session_memories
                )
                results.append(result)

                if verbose:
                    print(f"  Q: {qa.question[:60]}...")
                    print(f"  Gold: {qa.answer[:40]} | Pred: {predicted[:40]}")
                    print(f"  EM={em}, F1={f1:.2f}, recall={mem_recall:.2f}, "
                          f"prec={ret_prec:.2f}, reward={result.reward:.1f}")

        return results

    def run_baseline(
        self,
        conversations: list[Conversation],
        context_fn: Callable[[Conversation, str], str],
        method_name: str,
        verbose: bool = False,
    ) -> list[QAResult]:
        """Evaluate any baseline given a context-building function."""
        results = []

        for conv in conversations:
            for qa in conv.qa_pairs:
                context = context_fn(conv, qa.question)
                predicted = answer_question(
                    self.client, qa.question, context, self.answer_model
                )

                em = exact_match(predicted, qa.answer)
                f1 = token_f1(predicted, qa.answer)
                ctx_tokens = len(context) // 4

                result = QAResult(
                    question_id=qa.question_id,
                    session_id=conv.session_id,
                    question=qa.question,
                    gold_answer=qa.answer,
                    predicted_answer=predicted,
                    context_used=context,
                    context_tokens=ctx_tokens,
                    exact_match=em,
                    f1=f1,
                    method=method_name,
                )
                results.append(result)

                if verbose:
                    print(f"  [{method_name}] Q: {qa.question[:60]}...")
                    print(f"  EM={em}, F1={f1:.2f}")

        return results

    @staticmethod
    def summarize(results: list[QAResult]) -> EvalSummary:
        """Aggregate QA results into a summary."""
        if not results:
            return EvalSummary(
                method="unknown", num_questions=0,
                exact_match_rate=0, avg_f1=0, avg_context_tokens=0
            )

        method = results[0].method
        n = len(results)

        def _avg(vals):
            vals = [v for v in vals if v is not None]
            return sum(vals) / len(vals) if vals else None

        return EvalSummary(
            method=method,
            num_questions=n,
            exact_match_rate=sum(r.exact_match for r in results) / n,
            avg_f1=_avg([r.f1 for r in results]),
            avg_context_tokens=_avg([r.context_tokens for r in results]),
            avg_memory_recall=_avg([r.memory_recall for r in results]),
            avg_retrieval_precision=_avg([r.retrieval_precision for r in results]),
            avg_reward=_avg([r.reward for r in results]),
        )

    @staticmethod
    def save_results(results: list[QAResult], path: str) -> None:
        """Save results to JSON."""
        data = [
            {
                "question_id": r.question_id,
                "session_id": r.session_id,
                "question": r.question,
                "gold_answer": r.gold_answer,
                "predicted_answer": r.predicted_answer,
                "context_tokens": r.context_tokens,
                "exact_match": r.exact_match,
                "f1": r.f1,
                "method": r.method,
                "memory_recall": r.memory_recall,
                "retrieval_precision": r.retrieval_precision,
                "reward": r.reward,
            }
            for r in results
        ]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        print(f"Saved {len(results)} results to {path}")
