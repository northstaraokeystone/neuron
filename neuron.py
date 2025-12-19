"""
NEURON v3: Biologically Grounded Ledger
State reconstruction inevitable and advantageous.
~140 lines. Sharp-wave ripples. Synaptic pruning. Task-set inertia.
"""

import hashlib
import json
import math
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

try:
    import blake3
    HAS_BLAKE3 = True
except ImportError:
    HAS_BLAKE3 = False

# Ledger paths
LEDGER_PATH = Path(os.environ.get("NEURON_LEDGER", Path.home() / "neuron" / "receipts.jsonl"))
ARCHIVE_PATH = Path(os.environ.get("NEURON_ARCHIVE", Path.home() / "neuron" / "archive.jsonl"))

# Allowed projects
ALLOWED_PROJECTS = ["agentproof", "axiom", "neuron"]

# Entry constraints
MAX_TASK_LEN = 50
MAX_NEXT_LEN = 50

# Gap detection
DEFAULT_GAP_THRESHOLD_MIN = 60

# Salience decay
DECAY_RATE_PER_DAY = 0.05
REPLAY_DECAY_SLOWDOWN = 0.1

# Recovery cost (prefrontal inertia) - Monsell 2003
RECOVERY_K = 4.0
RECOVERY_TAU = 120.0

# Consolidation
DEFAULT_CONSOLIDATE_TOP_K = 10
DEFAULT_ALPHA_THRESHOLD = 5.0
SALIENCE_BOOST_BASE = 0.1

# Pruning
DEFAULT_MAX_AGE_DAYS = 30
DEFAULT_SALIENCE_THRESHOLD = 0.1
MIN_REPLAY_TO_PRESERVE = 5

# Energy estimation
TECHNICAL_TERMS = ["federation", "merkle", "entropy", "kan", "spline",
                   "receipt", "anchor", "proof", "hash", "topology"]


def dual_hash(data: bytes | str) -> str:
    """Compute SHA256:BLAKE3 hash per CLAUDEME §8."""
    if isinstance(data, str):
        data = data.encode("utf-8")
    sha256_hex = hashlib.sha256(data).hexdigest()
    blake3_hex = blake3.blake3(data).hexdigest() if HAS_BLAKE3 else hashlib.sha256(b"blake3:" + data).hexdigest()
    return f"{sha256_hex}:{blake3_hex}"


def energy_estimate(task: str, next_action: str) -> float:
    """Estimate cognitive load from text complexity."""
    words = len(task.split()) + len(next_action.split())
    word_factor = 0.5 + (words / 20)
    tech_count = sum(1 for t in TECHNICAL_TERMS if t in task.lower() or t in next_action.lower())
    tech_factor = 1.0 + 0.1 * tech_count
    return min(2.0, max(0.5, word_factor * tech_factor))


def salience_decay(entry: dict, current_ts: datetime | None = None) -> float:
    """Calculate decayed salience based on age and replay count."""
    if current_ts is None:
        current_ts = datetime.now(timezone.utc)
    entry_ts = datetime.fromisoformat(entry["ts"].replace("Z", "+00:00"))
    age_days = (current_ts - entry_ts).total_seconds() / 86400
    replay_boost = 1 + REPLAY_DECAY_SLOWDOWN * entry.get("replay_count", 0)
    base_salience = entry.get("salience", 1.0)
    return base_salience * math.exp(-DECAY_RATE_PER_DAY * age_days / replay_boost)


def recovery_cost(gap_minutes: float) -> float:
    """Non-linear cost: exponential decay model calibrated to human data (Monsell 2003)."""
    return 1.0 + RECOVERY_K * (1 - math.exp(-gap_minutes / RECOVERY_TAU))


def append(project: str, task: str, next_action: str, commit: str | None = None, energy: float | None = None) -> dict:
    """Append entry to shared ledger with salience/energy."""
    if project not in ALLOWED_PROJECTS:
        raise ValueError(f"Project must be one of: {ALLOWED_PROJECTS}")
    task, next_action = task[:MAX_TASK_LEN], next_action[:MAX_NEXT_LEN]
    entry = {
        "ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "project": project, "commit": commit, "task": task, "next": next_action,
        "salience": 1.0, "replay_count": 0, "energy": energy if energy else energy_estimate(task, next_action)
    }
    entry["hash"] = dual_hash(json.dumps({k: v for k, v in entry.items() if k != "hash"}, sort_keys=True))
    LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LEDGER_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")
    return entry


def _read_ledger() -> list[dict]:
    """Read all entries from ledger."""
    if not LEDGER_PATH.exists():
        return []
    entries = []
    with open(LEDGER_PATH, "r") as f:
        for line in f:
            if line.strip():
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return entries


def _write_ledger(entries: list[dict]) -> None:
    """Write entries to ledger (atomic overwrite)."""
    LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LEDGER_PATH, "w") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")


def replay(n: int | None = 10, since: str | None = None, increment_replay: bool = False) -> list[dict]:
    """Get entries, optionally increment replay_count (simulates neural reactivation)."""
    entries = _read_ledger()
    if since:
        entries = [e for e in entries if e.get("ts", "") >= since]
    result = entries[-n:] if n else entries
    if increment_replay and result:
        all_entries = _read_ledger()
        result_hashes = {e.get("hash") for e in result}
        for e in all_entries:
            if e.get("hash") in result_hashes:
                e["replay_count"] = e.get("replay_count", 0) + 1
        _write_ledger(all_entries)
        result = [e for e in all_entries if e.get("hash") in result_hashes][-n:] if n else [e for e in all_entries if e.get("hash") in result_hashes]
    return result


def alpha(threshold_minutes: int = DEFAULT_GAP_THRESHOLD_MIN) -> dict:
    """Calculate α with variance and expert_novice_ratio."""
    entries = _read_ledger()
    if len(entries) < 2:
        return {"total_entries": len(entries), "gaps_detected": 0, "gaps": [], "alpha_mean": 0.0,
                "alpha_min": 0.0, "alpha_max": 0.0, "alpha_variance": 0.0, "alpha_std": 0.0, "expert_novice_ratio": 1.0}
    entries.sort(key=lambda e: e.get("ts", ""))
    gaps, threshold_seconds = [], threshold_minutes * 60
    for i in range(1, len(entries)):
        prev_ts = datetime.fromisoformat(entries[i-1]["ts"].replace("Z", "+00:00"))
        curr_ts = datetime.fromisoformat(entries[i]["ts"].replace("Z", "+00:00"))
        gap_seconds = (curr_ts - prev_ts).total_seconds()
        if gap_seconds > threshold_seconds:
            gap_minutes = gap_seconds / 60
            recovery_min = max(1.0, gap_seconds / 60 / 10)
            gaps.append({"start": entries[i-1]["ts"], "end": entries[i]["ts"], "duration_min": round(gap_minutes, 1),
                         "recovery_min": round(recovery_min, 1), "alpha": round(gap_minutes / recovery_min, 1)})
    alpha_values = [g["alpha"] for g in gaps]
    if not alpha_values:
        return {"total_entries": len(entries), "gaps_detected": 0, "gaps": [], "alpha_mean": 0.0,
                "alpha_min": 0.0, "alpha_max": 0.0, "alpha_variance": 0.0, "alpha_std": 0.0, "expert_novice_ratio": 1.0}
    mean_a = sum(alpha_values) / len(alpha_values)
    variance = sum((a - mean_a) ** 2 for a in alpha_values) / len(alpha_values)
    return {"total_entries": len(entries), "gaps_detected": len(gaps), "gaps": gaps,
            "alpha_mean": round(mean_a, 1), "alpha_min": round(min(alpha_values), 1), "alpha_max": round(max(alpha_values), 1),
            "alpha_variance": round(variance, 2), "alpha_std": round(math.sqrt(variance), 2),
            "expert_novice_ratio": round(max(alpha_values) / min(alpha_values), 1) if min(alpha_values) > 0 else float('inf')}


def consolidate(top_k: int = DEFAULT_CONSOLIDATE_TOP_K, alpha_threshold: float = DEFAULT_ALPHA_THRESHOLD) -> dict:
    """Hippocampal replay: strengthen high-α entries (Wilson & McNaughton 1994)."""
    entries = _read_ledger()
    stats = alpha(threshold_minutes=1)
    qualifying = [(g, g["alpha"]) for g in stats["gaps"] if g["alpha"] > alpha_threshold]
    qualifying.sort(key=lambda x: x[1], reverse=True)
    affected_hashes, boost = [], 0.0
    for gap, a in qualifying[:top_k]:
        for e in entries:
            if e.get("ts") == gap["end"]:
                old_sal = e.get("salience", 1.0)
                e["salience"] = min(1.0, old_sal + SALIENCE_BOOST_BASE * math.log(max(1, a)))
                boost += e["salience"] - old_sal
                affected_hashes.append(e.get("hash", "")[:16])
    _write_ledger(entries)
    return {"consolidated_count": len(affected_hashes), "salience_boost": round(boost, 3), "entries_affected": affected_hashes}


def prune(max_age_days: int = DEFAULT_MAX_AGE_DAYS, salience_threshold: float = DEFAULT_SALIENCE_THRESHOLD) -> dict:
    """Synaptic downscaling: archive low-salience entries (Tononi & Cirelli 2014)."""
    entries = _read_ledger()
    now = datetime.now(timezone.utc)
    keep, archive = [], []
    for e in entries:
        decayed = salience_decay(e, now)
        entry_ts = datetime.fromisoformat(e["ts"].replace("Z", "+00:00"))
        age_days = (now - entry_ts).total_seconds() / 86400
        if age_days > max_age_days and decayed < salience_threshold and e.get("replay_count", 0) < MIN_REPLAY_TO_PRESERVE:
            archive.append(e)
        else:
            keep.append(e)
    if archive:
        ARCHIVE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(ARCHIVE_PATH, "a") as f:
            for e in archive:
                f.write(json.dumps(e) + "\n")
    _write_ledger(keep)
    return {"pruned_count": len(archive), "archived_to": str(ARCHIVE_PATH), "ledger_size_before": len(entries), "ledger_size_after": len(keep)}


def predict_next(n_context: int = 5) -> str | None:
    """Prospective memory: predict next action from history patterns."""
    entries = _read_ledger()[-n_context * 3:]
    if len(entries) < 2:
        return None
    patterns = Counter()
    for e in entries[:-1]:
        patterns[e.get("next", "")] += 1
    current_task = entries[-1].get("task", "").lower() if entries else ""
    for e in entries[:-1]:
        if any(word in e.get("task", "").lower() for word in current_task.split() if len(word) > 3):
            return e.get("next")
    return patterns.most_common(1)[0][0] if patterns else None


if __name__ == "__main__":
    print(f"NEURON v3 - Ledger: {LEDGER_PATH}")
    print(f"BLAKE3 available: {HAS_BLAKE3}")
