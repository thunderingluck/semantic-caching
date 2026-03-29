"""
ingestor.py — Load and parse LoCoMo dataset into structured conversation objects.

LoCoMo format:
  conversations: list of sessions, each with:
    - session_id
    - turns: [{speaker, text, turn_id}, ...]
  qa_pairs: list of QA annotations with evidence turn IDs
"""

import json
import re
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Turn:
    turn_id: int
    speaker: str
    text: str
    session_id: str


@dataclass
class QAPair:
    question_id: str
    session_id: str
    question: str
    answer: str
    evidence_turn_ids: list[int]
    category: Optional[str] = None


@dataclass
class Conversation:
    session_id: str
    turns: list[Turn] = field(default_factory=list)
    qa_pairs: list[QAPair] = field(default_factory=list)

    def get_turns_text(self, turn_ids: list[int]) -> str:
        """Get formatted text for specific turn IDs."""
        turns = [t for t in self.turns if t.turn_id in turn_ids]
        return "\n".join(f"{t.speaker}: {t.text}" for t in turns)

    def get_window(self, center: int, window: int = 5) -> list[Turn]:
        """Get a sliding window of turns around a center turn."""
        half = window // 2
        start = max(0, center - half)
        end = min(len(self.turns), center + half + 1)
        return self.turns[start:end]

    def format_turns(self, turns: list[Turn]) -> str:
        return "\n".join(f"[Turn {t.turn_id}] {t.speaker}: {t.text}" for t in turns)


def load_locomo(path: str | Path) -> list[Conversation]:
    """
    Load LoCoMo JSON file and return list of Conversation objects.

    LoCoMo10 structure (from the paper):
    The file is a list of conversation objects. Each has:
      - A key like "conversation_1" or numeric index
      - Each conversation has sessions with turns and QA pairs

    We normalize whatever structure we find.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"LoCoMo data not found at {path}. "
                                "Download from https://github.com/snap-research/locomo")

    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    conversations = []

    # Handle list format
    if isinstance(raw, list):
        for i, item in enumerate(raw):
            conv = _parse_conversation_item(item, str(i))
            if conv:
                conversations.append(conv)
    # Handle dict format (keyed by conversation ID)
    elif isinstance(raw, dict):
        for key, item in raw.items():
            conv = _parse_conversation_item(item, key)
            if conv:
                conversations.append(conv)

    print(f"Loaded {len(conversations)} conversations from {path}")
    return conversations


def _parse_conversation_item(item: dict, fallback_id: str) -> Optional[Conversation]:
    """Dispatch to the right parser based on the item's schema."""
    if not isinstance(item, dict):
        return None
    # LoCoMo10 format: has 'sample_id' and 'conversation' dict with session_N keys
    if "sample_id" in item and isinstance(item.get("conversation"), dict):
        return _parse_locomo10_item(item)
    return _parse_generic_item(item, fallback_id)


def _parse_locomo10_item(item: dict) -> Optional[Conversation]:
    """
    Parse the actual LoCoMo10 schema:
      item = {
        "sample_id": "conv-26",
        "conversation": {
          "speaker_a": "...", "speaker_b": "...",
          "session_1": [{speaker, dia_id, text}, ...],
          "session_1_date_time": "...",
          ...
        },
        "qa": [{question, answer, evidence: ["D1:3"], category}, ...]
      }
    Turns are assigned sequential integer IDs; dia_id strings are mapped to them
    so evidence references can be resolved.
    """
    session_id = str(item["sample_id"])
    conv_data = item["conversation"]

    # Collect sessions in numeric order
    session_keys = sorted(
        [k for k in conv_data if re.match(r"^session_\d+$", k)],
        key=lambda k: int(k.split("_")[1]),
    )

    turns: list[Turn] = []
    dia_id_to_turn_id: dict[str, int] = {}
    counter = 0

    for sess_key in session_keys:
        sess_turns = conv_data[sess_key]
        if not sess_turns:
            continue
        for t in sess_turns:
            if not isinstance(t, dict):
                continue
            dia_id = t.get("dia_id", "")
            turn = Turn(
                turn_id=counter,
                speaker=str(t.get("speaker", "unknown")).strip(),
                text=str(t.get("text", "")),
                session_id=session_id,
            )
            turns.append(turn)
            if dia_id:
                dia_id_to_turn_id[dia_id] = counter
            counter += 1

    if not turns:
        return None

    qa_pairs: list[QAPair] = []
    for j, qa in enumerate(item.get("qa", [])):
        if not isinstance(qa, dict):
            continue
        question = str(qa.get("question", "")).strip()
        answer = str(qa.get("answer", "")).strip()
        if not question:
            continue

        # Evidence is a list of dia_id strings like ["D1:3", "D2:7"]
        evidence_turn_ids = [
            dia_id_to_turn_id[ev]
            for ev in qa.get("evidence", [])
            if isinstance(ev, str) and ev in dia_id_to_turn_id
        ]

        category = str(qa["category"]) if qa.get("category") is not None else None

        qa_pairs.append(QAPair(
            question_id=f"{session_id}_q{j}",
            session_id=session_id,
            question=question,
            answer=answer,
            evidence_turn_ids=evidence_turn_ids,
            category=category,
        ))

    return Conversation(session_id=session_id, turns=turns, qa_pairs=qa_pairs)


def _parse_generic_item(item: dict, fallback_id: str) -> Optional[Conversation]:
    """Fallback parser for other LoCoMo schema variants."""
    session_id = str(item.get("session_id") or item.get("id") or fallback_id)
    turns = []

    if "turns" in item:
        for t in item["turns"]:
            turns.append(_parse_turn(t, session_id))
    elif "sessions" in item:
        for session in item["sessions"]:
            for t in session.get("turns", []):
                turns.append(_parse_turn(t, session_id))
    elif "dialogue" in item:
        for t in item["dialogue"]:
            turns.append(_parse_turn(t, session_id))

    turns = [t for t in turns if t is not None]
    if not turns:
        return None

    for i, t in enumerate(turns):
        if t.turn_id < 0:
            t.turn_id = i

    qa_pairs = []
    for j, qa in enumerate(item.get("qa_pairs") or item.get("qa") or []):
        parsed = _parse_qa_generic(qa, session_id, j)
        if parsed:
            qa_pairs.append(parsed)

    return Conversation(session_id=session_id, turns=turns, qa_pairs=qa_pairs)


def _parse_turn(t: dict, session_id: str) -> Optional[Turn]:
    if not isinstance(t, dict):
        return None
    text = t.get("text") or t.get("utterance") or t.get("content") or ""
    speaker = str(t.get("speaker") or t.get("role") or t.get("author") or "unknown").strip()
    turn_id = t.get("turn_id") or t.get("id") or t.get("index") or -1
    try:
        turn_id = int(turn_id)
    except (ValueError, TypeError):
        turn_id = -1
    return Turn(turn_id=turn_id, speaker=speaker, text=str(text), session_id=session_id)


def _parse_qa_generic(qa: dict, session_id: str, index: int) -> Optional[QAPair]:
    if not isinstance(qa, dict):
        return None
    question = qa.get("question") or qa.get("q") or ""
    answer = qa.get("answer") or qa.get("a") or qa.get("gold_answer") or ""
    if not question:
        return None
    question_id = str(qa.get("question_id") or qa.get("id") or f"{session_id}_q{index}")
    raw_evidence = qa.get("evidence_turn_ids") or qa.get("evidence") or []
    evidence_turn_ids = []
    for ev in raw_evidence:
        if isinstance(ev, int):
            evidence_turn_ids.append(ev)
        elif isinstance(ev, str):
            try:
                evidence_turn_ids.append(int(ev))
            except ValueError:
                nums = re.findall(r"\d+", ev)
                evidence_turn_ids.extend(int(n) for n in nums)
    category = str(qa.get("category")) if qa.get("category") is not None else None
    return QAPair(
        question_id=question_id,
        session_id=session_id,
        question=str(question),
        answer=str(answer),
        evidence_turn_ids=evidence_turn_ids,
        category=category,
    )


def get_stats(conversations: list[Conversation]) -> dict:
    """Print and return summary statistics."""
    total_turns = sum(len(c.turns) for c in conversations)
    total_qa = sum(len(c.qa_pairs) for c in conversations)
    avg_turns = total_turns / len(conversations) if conversations else 0

    stats = {
        "num_conversations": len(conversations),
        "total_turns": total_turns,
        "total_qa_pairs": total_qa,
        "avg_turns_per_conversation": round(avg_turns, 1),
    }
    for k, v in stats.items():
        print(f"  {k}: {v}")
    return stats
