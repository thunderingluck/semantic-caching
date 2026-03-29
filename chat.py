"""
chat.py — Interactive CLI for the semantic memory cache.

Usage:
    python chat.py

Commands during chat:
    /memory         — show all stored memories (sorted by importance)
    /ask <query>    — retrieve context and answer a question
    /clear          — wipe the memory store
    /quit           — exit

Each time you submit a turn, the system extracts facts from the
sliding window and shows what was stored (or skipped).
"""

import sys
import os
from pathlib import Path
from datetime import datetime, timezone

from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, str(Path(__file__).parent / "src"))

from openai import OpenAI
from sentence_transformers import SentenceTransformer

from ingestor import Turn
from memory_writer import MemoryWriter, MemoryObject
from memory_store import MemoryStore
from context_builder import ContextBuilder, RetrievalConfig
from evaluator import answer_question


# ── ANSI colours ─────────────────────────────
class C:
    RESET  = "\033[0m"
    BOLD   = "\033[1m"
    DIM    = "\033[2m"
    GREEN  = "\033[32m"
    YELLOW = "\033[33m"
    CYAN   = "\033[36m"
    RED    = "\033[31m"
    BLUE   = "\033[34m"
    MAGENTA = "\033[35m"

def _importance_colour(score: float) -> str:
    if score >= 0.7:
        return C.GREEN
    if score >= 0.4:
        return C.YELLOW
    return C.DIM

def _type_badge(t: str) -> str:
    badges = {
        "constraint":  f"{C.RED}[constraint]{C.RESET}",
        "decision":    f"{C.MAGENTA}[decision]{C.RESET}",
        "goal":        f"{C.BLUE}[goal]{C.RESET}",
        "preference":  f"{C.CYAN}[preference]{C.RESET}",
        "fact":        f"{C.YELLOW}[fact]{C.RESET}",
        "definition":  f"{C.DIM}[definition]{C.RESET}",
    }
    return badges.get(t, f"[{t}]")


# ── Session state ─────────────────────────────

class ChatSession:
    def __init__(self, client, embedder, verbose_extraction=True):
        self.client = client
        self.embedder = embedder
        self.verbose_extraction = verbose_extraction

        self.store = MemoryStore(collection_name="chat_session", embedder=embedder)
        self.writer = MemoryWriter(
            client=client,
            embedder=embedder,
            window_size=4,
            stride=2,
            importance_threshold=0.3,
            dedup_threshold=0.92,
        )
        self.builder = ContextBuilder(
            store=self.store,
            config=RetrievalConfig(token_budget=500),
        )

        self.turns: list[Turn] = []
        self.session_id = "chat"
        self.turn_counter = 0

    def add_turn(self, speaker: str, text: str) -> list[MemoryObject]:
        """Add a turn and run memory extraction on the current window."""
        self.turns.append(Turn(
            turn_id=self.turn_counter,
            speaker=speaker,
            text=text,
            session_id=self.session_id,
        ))
        self.turn_counter += 1

        # Only extract when we have enough turns to fill the stride
        if len(self.turns) < 2:
            return []

        # Build a fake Conversation-like object for the writer
        from ingestor import Conversation
        conv = Conversation(session_id=self.session_id, turns=self.turns)

        # Extract only from the latest window (avoid re-processing all turns)
        window_start = max(0, len(self.turns) - self.writer.window_size)
        window = self.turns[window_start:]

        raw_facts = self.writer._extract_facts(window, verbose=False)
        if not raw_facts:
            return []

        scored = self.writer._score_importance(raw_facts, verbose=False)
        turn_ids = [t.turn_id for t in window]

        candidates = []
        for fact_data, score_data in zip(raw_facts, scored):
            importance = score_data.get("importance", 0.0)
            mem = self.writer._build_memory_object(
                fact_data=fact_data,
                score_data=score_data,
                source_turns=turn_ids,
                session_id=self.session_id,
            )
            candidates.append((mem, importance))

        # Embed
        self.writer._embed_memories([m for m, _ in candidates])

        # Deduplicate against existing store memories
        stored_mems = self.store.get_by_session(self.session_id)
        stored_embeddings = [
            m.embedding for m in stored_mems if m.embedding
        ]

        added = []
        skipped_low = []
        skipped_dup = []

        import numpy as np
        for mem, importance in candidates:
            if importance < self.writer.importance_threshold:
                skipped_low.append((mem, importance))
                continue
            if mem.embedding and stored_embeddings:
                import numpy as np
                emb = np.array(mem.embedding)
                sims = np.array(stored_embeddings) @ emb
                if sims.max() >= self.writer.dedup_threshold:
                    skipped_dup.append(mem)
                    continue
            self.store.add(mem)
            stored_embeddings.append(mem.embedding)
            added.append(mem)

        return added, skipped_low, skipped_dup

    def ask(self, query: str) -> tuple[str, str, list[dict]]:
        """Build context and answer a question. Returns (answer, context, debug)."""
        context, debug = self.builder.build_context(
            query=query,
            session_id=self.session_id,
            verbose=False,
        )
        answer = answer_question(self.client, query, context)
        return answer, context, debug

    def all_memories(self) -> list[MemoryObject]:
        return self.store.get_by_session(self.session_id)


# ── Display helpers ───────────────────────────

def print_memory(mem: MemoryObject, prefix: str = "  ") -> None:
    col = _importance_colour(mem.importance)
    badge = _type_badge(mem.type)
    print(
        f"{prefix}{badge} {col}{C.BOLD}{mem.importance:.2f}{C.RESET}  "
        f"{mem.fact}"
    )
    print(f"{prefix}{C.DIM}turns={mem.source_turns}  persist={mem.persistence}{C.RESET}")


def print_extraction_result(added, skipped_low, skipped_dup) -> None:
    if not added and not skipped_low and not skipped_dup:
        print(f"  {C.DIM}(no facts extracted){C.RESET}")
        return

    if added:
        print(f"  {C.GREEN}+{len(added)} stored:{C.RESET}")
        for mem in added:
            print_memory(mem, prefix="    ")

    if skipped_dup:
        print(f"  {C.DIM}~{len(skipped_dup)} skipped (duplicate):{C.RESET}")
        for mem in skipped_dup:
            print(f"    {C.DIM}\"{mem.fact[:70]}\"{C.RESET}")

    if skipped_low:
        print(f"  {C.DIM}↓{len(skipped_low)} skipped (low importance < 0.3):{C.RESET}")
        for mem, score in skipped_low:
            print(f"    {C.DIM}[{score:.2f}] \"{mem.fact[:70]}\"{C.RESET}")


def print_memory_store(memories: list[MemoryObject]) -> None:
    if not memories:
        print(f"  {C.DIM}(memory store is empty){C.RESET}")
        return
    by_importance = sorted(memories, key=lambda m: m.importance, reverse=True)
    print(f"  {C.BOLD}{len(memories)} memories:{C.RESET}")
    for mem in by_importance:
        print_memory(mem, prefix="    ")


def print_context(context: str, debug: list[dict]) -> None:
    if not context:
        print(f"  {C.DIM}(no relevant memories retrieved){C.RESET}")
        return
    print(f"  {C.CYAN}Retrieved {len(debug)} memories:{C.RESET}")
    for d in debug:
        col = _importance_colour(d["importance"])
        print(
            f"    score={C.BOLD}{d['score']:.3f}{C.RESET}  "
            f"rel={d['relevance']:.2f}  imp={col}{d['importance']:.2f}{C.RESET}  "
            f"\"{d['fact'][:70]}\""
        )


# ── Main REPL ─────────────────────────────────

HELP_TEXT = f"""
{C.BOLD}Commands:{C.RESET}
  {C.CYAN}/memory{C.RESET}          show all stored memories
  {C.CYAN}/ask <question>{C.RESET}  retrieve context and answer
  {C.CYAN}/clear{C.RESET}           wipe memory store
  {C.CYAN}/verbose{C.RESET}         toggle verbose extraction output
  {C.CYAN}/help{C.RESET}            show this message
  {C.CYAN}/quit{C.RESET}            exit

{C.BOLD}Chat:{C.RESET}
  Just type to add turns. Format: {C.DIM}Speaker: text{C.RESET} or plain text (attributed to 'User').
  After every 2+ turns the window is extracted and new facts are shown.
"""


def main():
    print(f"\n{C.BOLD}{C.CYAN}Semantic Memory Cache — Interactive Chat{C.RESET}")
    print(f"{C.DIM}Type /help for commands{C.RESET}\n")

    # Init
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print(f"{C.RED}Error: OPENAI_API_KEY not set.{C.RESET}")
        sys.exit(1)

    print("Loading models...", end="", flush=True)
    client = OpenAI(api_key=api_key)
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    print(f" {C.GREEN}ready{C.RESET}\n")

    session = ChatSession(client, embedder)
    verbose = True

    while True:
        try:
            raw = input(f"{C.BOLD}> {C.RESET}").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        if not raw:
            continue

        # ── Commands ──────────────────────────
        if raw == "/quit" or raw == "/exit":
            print("Bye.")
            break

        elif raw == "/help":
            print(HELP_TEXT)

        elif raw == "/memory":
            mems = session.all_memories()
            print(f"\n{C.BOLD}Memory store:{C.RESET}")
            print_memory_store(mems)
            print()

        elif raw == "/clear":
            session.store._collection.delete(
                where={"session_id": {"$eq": session.session_id}}
            )
            print(f"  {C.YELLOW}Memory store cleared.{C.RESET}\n")

        elif raw == "/verbose":
            verbose = not verbose
            print(f"  Verbose extraction: {'on' if verbose else 'off'}\n")

        elif raw.startswith("/ask "):
            query = raw[5:].strip()
            if not query:
                print(f"  {C.DIM}Usage: /ask <your question>{C.RESET}\n")
                continue
            print(f"\n{C.BOLD}Query:{C.RESET} {query}")
            answer, context, debug = session.ask(query)
            print(f"\n{C.BOLD}Retrieved context:{C.RESET}")
            print_context(context, debug)
            print(f"\n{C.BOLD}Answer:{C.RESET} {C.GREEN}{answer}{C.RESET}\n")

        # ── Chat turn ─────────────────────────
        else:
            # Parse "Speaker: text" or default to "User"
            if ": " in raw and not raw.startswith("/"):
                speaker, text = raw.split(": ", 1)
            else:
                speaker, text = "User", raw

            result = session.add_turn(speaker, text)

            if verbose:
                print(f"\n{C.BOLD}[Turn {session.turn_counter - 1}]{C.RESET} "
                      f"{C.CYAN}{speaker}{C.RESET}: {text}")
                if result:
                    added, skipped_low, skipped_dup = result
                    print(f"{C.BOLD}Memory extraction:{C.RESET}")
                    print_extraction_result(added, skipped_low, skipped_dup)
                else:
                    print(f"  {C.DIM}(need more turns before extraction){C.RESET}")
                print()
            else:
                if result:
                    added, _, _ = result
                    if added:
                        print(f"  {C.GREEN}+{len(added)} memories stored{C.RESET}")


if __name__ == "__main__":
    main()
