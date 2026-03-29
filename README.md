# Semantic Memory Cache

Solving context rot with intelligent memory: a learned memory layer for LLM conversations that replaces raw history with structured, importance-scored memory objects. 

**Core thesis:** context should be *constructed*, not accumulated. Long conversations degrade due to **context rot** — irrelevant tokens accumulate and dilute important signals. This system treats context as a **caching problem**: given a fixed token budget, retrieve only the information with the highest expected future utility.

## How it works

```
Conversation turns
      │
      ▼
 MemoryWriter          ← sliding window extraction via LLM
      │ atomic facts + importance scores
      ▼
 MemoryStore           ← Chroma vector store (in-memory or persisted)
      │
      ▼
 ContextBuilder        ← score = α·relevance + β·importance + γ·recency − λ·redundancy
      │ curated context (≤500 tokens)
      ▼
 Downstream LLM answer
```

Facts are deduplicated, importance-filtered (threshold 0.3), and write-invalidated when contradictions are detected.

## Setup

```bash
pip install -r requirements.txt
```

Set your OpenAI API key:

```bash
# .env file (auto-loaded)
OPENAI_API_KEY=sk-...
```

The system uses `gpt-4o-mini` for extraction/scoring and `all-MiniLM-L6-v2` (local) for embeddings.

## Interactive chat — `chat.py`

```bash
python chat.py
```

### Chat input

Type plain text (attributed to `User`) or use `Speaker: text` format:

```
> Alice: I want to use PostgreSQL, not SQLite
> Bob: Agreed, we need ACID compliance for this workload
```

After every 2+ turns the window is extracted and new memories are displayed.

### Commands

| Command | Description |
|---|---|
| `/memory` | Show all stored memories sorted by importance |
| `/ask <question>` | Retrieve relevant context and answer a question |
| `/clear` | Wipe the memory store for the current session |
| `/verbose` | Toggle verbose extraction output (on by default) |
| `/help` | Show command reference |
| `/quit` | Exit |

### Example session

```
> Alice: We've decided to use React for the frontend
[Turn 1] Alice: We've decided to use React for the frontend
Memory extraction:
  +1 stored:
    [decision] 0.82  We've decided to use React for the frontend
      turns=[0, 1]  persist=long_term

> Bob: The deadline is end of Q2
> /ask What frontend framework are we using?

Query: What frontend framework are we using?

Retrieved context:
  score=0.891  rel=0.94  imp=0.82  "We've decided to use React for the frontend"

Answer: React
```

## Python API

### MemoryWriter

Extracts atomic facts from conversation turns using a sliding window.

```python
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from src.ingestor import Conversation, Turn
from src.memory_writer import MemoryWriter

client = OpenAI()
embedder = SentenceTransformer("all-MiniLM-L6-v2")

writer = MemoryWriter(
    client=client,
    embedder=embedder,
    window_size=4,        # turns per extraction window
    stride=2,             # step between windows
    importance_threshold=0.3,  # discard below this
    dedup_threshold=0.92,      # cosine sim above which facts are duplicates
)

conv = Conversation(
    session_id="my_session",
    turns=[
        Turn(turn_id=0, speaker="Alice", text="Use PostgreSQL.", session_id="my_session"),
        Turn(turn_id=1, speaker="Bob",   text="Agreed.",         session_id="my_session"),
    ],
)

memories = writer.process_conversation(conv, verbose=True)
```

### MemoryStore

Chroma-backed vector store for `MemoryObject` instances.

```python
from src.memory_store import MemoryStore

# In-memory (default)
store = MemoryStore(collection_name="my_store", embedder=embedder)

# Persisted to disk
store = MemoryStore(collection_name="my_store", embedder=embedder, persist_dir="./db")

# Add memories
store.add_batch(memories)

# Similarity search
results = store.query(
    query_text="What database are we using?",
    n_results=10,
    session_id="my_session",   # optional filter
    status="active",           # active | stale | archived
    min_importance=0.3,
)
# returns: list of (MemoryObject, similarity_score)

# Get all memories for a session
all_mems = store.get_by_session("my_session")

# Update memory status
store.mark_stale("mem_abc123", superseded_by="mem_def456")
store.archive("mem_abc123")
```

### ContextBuilder

Scores and selects memories under a token budget.

```python
from src.context_builder import ContextBuilder, RetrievalConfig

config = RetrievalConfig(
    alpha=0.5,          # relevance weight
    beta=0.3,           # importance weight
    gamma=0.1,          # recency weight
    lam=0.1,            # redundancy penalty
    token_budget=500,
)

builder = ContextBuilder(store=store, config=config)

context_str, debug_info = builder.build_context(
    query="What database are we using?",
    session_id="my_session",
    verbose=True,
)

# debug_info: list of dicts with keys: fact, type, importance, score, relevance, recency
```

### MemoryObject schema

```python
@dataclass
class MemoryObject:
    id: str              # "mem_0142"
    fact: str            # "Use PostgreSQL instead of SQLite"
    type: str            # fact | preference | constraint | decision | definition | goal
    importance: float    # 0.0–1.0
    persistence: str     # ephemeral | medium | long_term
    scope: str           # session | project | user
    source_turns: list[int]
    source_text: str
    status: str          # active | stale | archived
    created_at: str      # ISO timestamp
    session_id: str
    embedding: list[float] | None
    access_count: int
    last_accessed_at: str | None
    superseded_by: str | None  # ID of superseding memory
```

## Evaluation

Evaluates the memory pipeline against baselines on the [LoCoMo dataset](https://github.com/snap-research/locomo).

**Baselines:**
1. Full conversation history
2. Rolling summary
3. Naive RAG (turn-level retrieval)
4. Semantic cache (this system)

**Metrics:** memory recall, retrieval precision, QA accuracy (exact match + F1), token efficiency.

See `notebooks/03_eval_comparison.ipynb` for results.

## Project structure

```
├── chat.py                  # Interactive CLI
├── src/
│   ├── ingestor.py          # LoCoMo data loading + Turn/Conversation dataclasses
│   ├── memory_writer.py     # Fact extraction + importance scoring
│   ├── memory_store.py      # Chroma vector store wrapper
│   ├── context_builder.py   # Retrieval scoring + budget selection
│   ├── evaluator.py         # QA evaluation + metrics
│   └── baselines.py         # Baseline context methods
├── prompts/
│   ├── extraction.txt       # LLM prompt for fact extraction
│   ├── importance.txt       # LLM prompt for importance scoring
│   └── answering.txt        # LLM prompt for QA
├── notebooks/
│   ├── 01_explore_locomo.ipynb
│   ├── 02_pipeline_demo.ipynb
│   └── 03_eval_comparison.ipynb
└── results/
    └── eval_results.json
```
