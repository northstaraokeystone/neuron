# NEURON v3: Biologically Grounded Ledger

**State reconstruction inevitable and advantageous.**

## Paradigm Shift

| v2 | v3 |
|----|----|
| "State loss impossible" | "State reconstruction inevitable" |
| Infinite append-only ledger | Ledger with synaptic pruning |
| Linear recovery assumption | Non-linear prefrontal task-set inertia |
| α = gap/recovery (static) | α with variance tracking |
| Replay = read last entry | Replay = hippocampal consolidation |

## Biological Grounding

| Mechanism | Neural Basis | Implementation |
|-----------|--------------|----------------|
| Sharp-wave ripples | Hippocampus replays sequences 10-20x faster offline | `consolidate()` |
| Synaptic downscaling | Sleep prunes low-salience traces | `prune()` |
| Task-set inertia | Prefrontal switching cost ~200-500ms baseline | `recovery_cost()` |
| Prospective memory | Predict upcoming action from context | `predict_next()` |

## Entry Format

```json
{
  "ts": "2025-01-15T14:00:00Z",
  "project": "agentproof|axiom|neuron",
  "hash": "sha256:blake3",
  "commit": "git commit hash or null",
  "task": "current task (≤50 chars)",
  "next": "next action (≤50 chars)",
  "salience": 0.85,
  "replay_count": 0,
  "energy": 1.0
}
```

### New Fields

| Field | Type | Purpose | Default |
|-------|------|---------|---------|
| salience | float 0-1 | Entry importance (decays over time, increases on replay) | 1.0 |
| replay_count | int | Times entry was accessed during consolidation | 0 |
| energy | float | Cognitive load proxy (higher = more complex context) | 1.0 |

## Functions

### Core (unchanged signature)

- `dual_hash(data)` → SHA256:BLAKE3 per CLAUDEME §8

### Extended

- `append(project, task, next, commit, energy)` → Entry with salience/energy
- `replay(n, since, increment_replay)` → Entries; optionally bump replay_count
- `alpha(threshold_minutes)` → Stats with variance, expert_novice_ratio

### New (v3)

- `consolidate(top_k, alpha_threshold)` → Hippocampal replay
- `prune(max_age_days, salience_threshold)` → Synaptic downscaling
- `recovery_cost(gap_minutes)` → Non-linear cost model
- `predict_next(n_context)` → Pattern-based prospective memory
- `salience_decay(entry, current_ts)` → Time + replay decay
- `energy_estimate(task, next)` → Cognitive load from text

## Constants

```python
DECAY_RATE_PER_DAY = 0.05      # 5% base decay
REPLAY_DECAY_SLOWDOWN = 0.1    # Each replay slows decay 10%
RECOVERY_K = 4.0               # Maximum additional cost
RECOVERY_TAU = 120.0           # Time constant (minutes)
MIN_REPLAY_TO_PRESERVE = 5     # Never prune high-replay entries
```

## Recovery Cost Model

```
cost = 1.0 + 4.0 * (1 - exp(-gap_minutes / 120.0))

gap = 0 min   → cost = 1.00
gap = 15 min  → cost = 1.47
gap = 60 min  → cost = 2.57
gap = 120 min → cost = 3.53
gap = 240 min → cost = 4.73
```

## Salience Decay Model

```
age_days = (now - entry_ts).days
replay_boost = 1 + 0.1 * replay_count
decayed = salience * exp(-0.05 * age_days / replay_boost)
```

After 30 days:
- No replays: salience ≈ 0.22
- 5 replays: salience ≈ 0.47

## Entry Lifecycle

```
Day 0:  append()      → salience=1.0, replay_count=0
Day 3:  replay(+inc)  → replay_count=1, decay slowed
Day 7:  consolidate() → salience boosted (high-α)
Day 30: prune check   → salience=0.35, kept
Day 90: prune check   → salience=0.05, archived
```

## Calibration Sources

| Mechanism | Source | Finding |
|-----------|--------|---------|
| Sharp-wave ripples | Wilson & McNaughton 1994 | 10-20x faster offline replay |
| Synaptic downscaling | Tononi & Cirelli 2014 | Sleep prunes low-salience |
| Task-set inertia | Monsell 2003 | 200-500ms switch cost baseline |
| Memory-for-goals | Altmann & Trafton 2002 | Non-linear decay |
| Expertise | Ericsson 2006 | 5-10x expert/novice ratio |

## Files

```
neuron/
├── receipts.jsonl       # Active ledger
├── archive.jsonl        # Pruned entries (cold storage)
├── neuron.py            # ~140 lines
├── ledger_schema.json   # JSON schema
├── spec.md              # This file
└── tests/
    └── test_neuron.py
```

## Usage

```python
from neuron import append, replay, consolidate, prune, predict_next

# Work session
append("neuron", "implement federation", "write tests", "abc123")

# End of session: consolidate
consolidate(top_k=10, alpha_threshold=5.0)

# Resume: replay with strengthening
last = replay(1, increment_replay=True)[0]

# Weekly: prune
prune(max_age_days=30, salience_threshold=0.1)

# Predict
suggestion = predict_next(n_context=5)
```

---

**~140 lines. Biologically grounded. Reconstruction > perfection.**
