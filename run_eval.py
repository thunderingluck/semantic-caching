#!/usr/bin/env python3
"""
run_eval.py — Standalone evaluation script.

Evaluates the semantic cache and all three baselines on LoCoMo QA,
saves results to results/eval_results.json, and optionally applies
RLVR importance updates to the memory store.

Usage:
    python run_eval.py
    python run_eval.py --n-convs 5 --token-budget 500 --apply-rlvr
    python run_eval.py --model gpt-4o-mini --n-convs 10
"""

import sys
import json
import argparse
from pathlib import Path

# Anchor src/ relative to this script, regardless of working directory
_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT / "src"))

from dotenv import load_dotenv
load_dotenv(_ROOT / ".env")

from openai import OpenAI
from sentence_transformers import SentenceTransformer

from ingestor import load_locomo
from memory_writer import MemoryWriter
from memory_store import MemoryStore
from context_builder import ContextBuilder, RetrievalConfig
from baselines import FullHistoryBaseline, RollingSummaryBaseline, NaiveRAGBaseline
from evaluator import Evaluator, ImportanceLearner


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run semantic cache evaluation")
    p.add_argument("--n-convs", type=int, default=3,
                   help="Number of conversations to evaluate (default: 3)")
    p.add_argument("--token-budget", type=int, default=500,
                   help="Token budget for context builder (default: 500)")
    p.add_argument("--model", default="gpt-4o-mini",
                   help="OpenAI model for extraction + answering (default: gpt-4o-mini)")
    p.add_argument("--apply-rlvr", action="store_true",
                   help="Apply RLVR importance updates after semantic cache eval")
    p.add_argument("--verbose", action="store_true",
                   help="Print per-question results")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    print(f"Config: n_convs={args.n_convs}, token_budget={args.token_budget}, "
          f"model={args.model}, apply_rlvr={args.apply_rlvr}")

    client = OpenAI()
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    # ── Load data ────────────────────────────────────────────────────────────
    conversations = load_locomo(_ROOT / "data" / "locomo10.json")
    eval_convs = [c for c in conversations[: args.n_convs] if c.qa_pairs]
    total_qa = sum(len(c.qa_pairs) for c in eval_convs)
    print(f"\nEvaluating on {len(eval_convs)} conversations, {total_qa} QA pairs")

    # ── Build memory store ───────────────────────────────────────────────────
    print("\nBuilding memory store...")
    store = MemoryStore(collection_name="eval", embedder=embedder)
    writer = MemoryWriter(
        client=client, embedder=embedder,
        window_size=4, stride=2,
        importance_threshold=0.3,
    )
    all_memories_by_session: dict = {}
    for conv in eval_convs:
        memories = writer.process_conversation(conv, verbose=False)
        all_memories_by_session[conv.session_id] = memories
        store.add_batch(memories)
        print(f"  {conv.session_id}: {len(memories)} memories stored")
    print(f"Total memories: {store.count()}")

    # ── Initialize methods ───────────────────────────────────────────────────
    config = RetrievalConfig(token_budget=args.token_budget)
    builder = ContextBuilder(store=store, config=config)
    evaluator = Evaluator(
        client=client, store=store,
        context_builder=builder, answer_model=args.model,
    )
    full_history = FullHistoryBaseline(token_budget=args.token_budget * 4)
    rolling_summary = RollingSummaryBaseline(client=client, token_budget=args.token_budget)
    naive_rag = NaiveRAGBaseline(embedder=embedder, token_budget=args.token_budget)

    all_results: dict = {}
    summaries = []

    # ── Semantic cache ───────────────────────────────────────────────────────
    print("\nRunning: semantic_cache...")
    results_sc = evaluator.run_semantic_cache(
        eval_convs, all_memories_by_session, verbose=args.verbose
    )
    all_results["semantic_cache"] = results_sc
    summary_sc = Evaluator.summarize(results_sc)
    summaries.append(summary_sc)
    print(f"  EM={summary_sc.exact_match_rate:.3f}  F1={summary_sc.avg_f1:.3f}  "
          f"tokens={summary_sc.avg_context_tokens:.0f}  "
          f"recall={summary_sc.avg_memory_recall:.3f}  "
          f"prec={summary_sc.avg_retrieval_precision:.3f}  "
          f"reward={summary_sc.avg_reward:.3f}")

    # ── Optional RLVR importance update ─────────────────────────────────────
    if args.apply_rlvr:
        print("\nApplying RLVR importance updates...")
        learner = ImportanceLearner(store=store, lr=0.05, penalty=0.5)
        deltas = learner.apply(results_sc)
        learner.summarize_updates(deltas)

    # ── Baselines ────────────────────────────────────────────────────────────
    baselines = [
        ("full_history",    lambda conv, q: full_history.build_context(conv, q)),
        ("rolling_summary", lambda conv, q: rolling_summary.build_context(conv, q)),
        ("naive_rag",       lambda conv, q: naive_rag.build_context(conv, q)),
    ]
    for name, ctx_fn in baselines:
        print(f"\nRunning: {name}...")
        results = evaluator.run_baseline(
            eval_convs, ctx_fn, method_name=name, verbose=args.verbose
        )
        all_results[name] = results
        s = Evaluator.summarize(results)
        summaries.append(s)
        print(f"  EM={s.exact_match_rate:.3f}  F1={s.avg_f1:.3f}  "
              f"tokens={s.avg_context_tokens:.0f}")

    # ── Save results ─────────────────────────────────────────────────────────
    results_dir = _ROOT / "results"
    results_dir.mkdir(exist_ok=True)

    for method, results in all_results.items():
        Evaluator.save_results(results, str(results_dir / f"{method}_results.json"))

    summaries_dict = {s.method: s.to_dict() for s in summaries}
    out_path = results_dir / "eval_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summaries_dict, f, indent=2)
    print(f"\nAll results saved to {out_path}")

    # ── Summary table ────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"{'Method':<20} {'EM':>6} {'F1':>6} {'Tokens':>8} {'F1/100tok':>10}")
    print("-" * 60)
    for s in summaries:
        eff = (s.avg_f1 / s.avg_context_tokens * 100) if s.avg_context_tokens else 0.0
        print(f"{s.method:<20} {s.exact_match_rate:>6.3f} {s.avg_f1:>6.3f} "
              f"{s.avg_context_tokens:>8.0f} {eff:>10.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
