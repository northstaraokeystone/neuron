# NEURON v4: Inference-Integrated Ledger

**volatile state + persistent proof = resilient inference**

## Paradigm Shift

| v3 | v4 |
|----|----|
| Manual append triggers | Inference cycle auto-append |
| Single project scope | Multi-model ensemble |
| Ledger as standalone | Ledger as inference complement |
| Generic entries | Model-aware entries with token_count |
| Replay returns list | Replay returns context-ready format |

## The Integration Insight

> "volatile state + persistent proof = resilient inference"

Grok's context windows are volatile (cleared between sessions). NEURON ledger is persistent. Together: resilient inference that survives interruption.

## Entry Format

```json
{
  "ts": "2025-01-15T14:00:00Z",
  "project": "agentproof|axiom|neuron",
  "model": "grok|claude|gemini|neuron",
  "hash": "sha256:blake3",
  "commit": "git commit hash or null",
  "task": "current task (≤50 chars)",
  "next": "next action (≤50 chars)",
  "salience": 0.85,
  "replay_count": 0,
  "energy": 1.0,
  "token_count": 4500,
  "inference_id": "inf_abc123 or null",
  "context_summary": "compressed context (≤500 chars)"
}
```

### New Fields (v4)

| Field | Type | Purpose | Default |
|-------|------|---------|---------|
| model | str | Which LLM generated this entry | "neuron" |
| token_count | int | Context window utilization at append time | 0 |
| inference_id | str\|null | Unique ID for inference cycle | null |
| context_summary | str | Compressed snapshot of context (≤500 chars) | "" |

## Functions

### New (v4)

- `inference_append(model, task, next, context_summary, token_count, inference_id)` → Auto-append from LLM inference cycles
- `replay_to_context(n, format)` → Format ledger for context injection
- `sync_ledger(remote_path)` → Merge remote ledger (last-write-wins)

### Extended (v4)

- `append(project, task, next, commit, energy, model, token_count, inference_id, context_summary)` → Entry with inference metadata
- `replay(n, since, increment_replay, format)` → format="list"|"context"
- `consolidate(top_k, alpha_threshold)` → Weight by token_count
- `prune(max_age_days, salience_threshold)` → Target >99.5% compression

### Core (unchanged)

- `dual_hash(data)` → SHA256:BLAKE3 per CLAUDEME §8
- `alpha(threshold_minutes)` → Stats with variance, expert_novice_ratio
- `recovery_cost(gap_minutes)` → Non-linear cost model
- `predict_next(n_context)` → Pattern-based prospective memory
- `salience_decay(entry, current_ts)` → Time + replay decay
- `energy_estimate(task, next, token_count)` → Cognitive load from text + tokens

## Constants

```python
# Models
SUPPORTED_MODELS = ["grok", "claude", "gemini", "neuron"]

# Inference integration
INFERENCE_CONTEXT_MAX_TOKENS = 128000
MAX_CONTEXT_SUMMARY_LEN = 500

# Consolidation
HIGH_ALPHA_THRESHOLD = 10.0
REPLAY_STRENGTH_FACTOR = 3

# Pruning v3
SALIENCE_RETENTION_THRESHOLD = 0.8
PRUNING_V3_TARGET = 0.995
MIN_AGE_TO_PRUNE_DAYS = 7

# Sync
SYNC_CONFLICT_RESOLUTION = "last_write_wins"
```

## Biological Grounding

| Mechanism | Neural Basis | Implementation |
|-----------|--------------|----------------|
| Sharp-wave ripples | Hippocampus replays sequences 10-20x faster offline | `consolidate()` |
| Synaptic downscaling | Sleep prunes low-salience traces | `prune()` |
| Task-set inertia | Prefrontal switching cost ~200-500ms baseline | `recovery_cost()` |
| Prospective memory | Predict upcoming action from context | `predict_next()` |
| Context reinstatement | Hippocampal pattern completion | `replay_to_context()` |

## Integration Pattern

```
┌─────────────────────────────────────────────────────────────────┐
│                        INFERENCE LAYER                          │
│                                                                 │
│   ┌─────────┐     ┌─────────┐     ┌─────────┐                  │
│   │  GROK   │     │ CLAUDE  │     │ GEMINI  │                  │
│   │ context │     │ context │     │ context │                  │
│   │ window  │     │ window  │     │ window  │                  │
│   └────┬────┘     └────┬────┘     └────┬────┘                  │
│        │               │               │                        │
│        └───────────────┼───────────────┘                        │
│                        │                                        │
│                        ▼                                        │
│              inference_append()                                 │
│                        │                                        │
└────────────────────────┼────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                    NEURON LEDGER                                │
│                                                                 │
│  receipts.jsonl                                                 │
│  ─────────────────────────────────────────────────────────────  │
│  {"model":"grok", "task":"...", "token_count":45000, ...}      │
│  {"model":"claude", "task":"...", "token_count":25000, ...}    │
│                                                                 │
│  replay_to_context() → formatted string for any model          │
│  sync_ledger() → merge across instances                         │
└─────────────────────────────────────────────────────────────────┘
```

## Usage

```python
from neuron import inference_append, replay_to_context, sync_ledger

# Inference cycle append
inference_append(
    model="grok",
    task="reasoning about Mars colony",
    next="calculate entropy budget",
    context_summary="User asked about Mars colony survival...",
    token_count=45000,
    inference_id="inf_abc123"
)

# Resume with context injection
context = replay_to_context(n=5, format="context")
# Inject into LLM prompt

# Sync with remote instance
sync_ledger("/path/to/remote/receipts.jsonl")
```

## Files

```
neuron/
├── receipts.jsonl       # Active ledger (extended schema)
├── archive.jsonl        # Pruned entries (cold storage)
├── neuron.py            # ~180 lines
├── ledger_schema.json   # JSON schema (v4)
├── spec.md              # This file
└── tests/
    ├── test_neuron.py
    └── test_inference.py
```

## Recovery Flow

```
Session ends → context window cleared (volatile)
                    ↓
New session → replay_to_context(5)
                    ↓
NEURON returns formatted state:
  ## NEURON State Recovery
  ### Recent Context (5 entries)
  [ts] grok: Task: X, Next: Y, Context: Z
  ...
  ### Resume Instruction
  Continue from: Y
                    ↓
LLM resumes with full context (persistent)
```

---

**~180 lines. Multi-model. Inference-native.**

**volatile state + persistent proof = resilient inference**
