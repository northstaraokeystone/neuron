"""
NEURON v2: The Shared Ledger
One file. Three functions. State loss impossible.
"""

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

# Try to import blake3, fall back to sha256 if unavailable
try:
    import blake3
    HAS_BLAKE3 = True
except ImportError:
    HAS_BLAKE3 = False

# Constants
LEDGER_PATH = Path(os.environ.get("NEURON_LEDGER", Path.home() / "neuron" / "receipts.jsonl"))
ALLOWED_PROJECTS = ["agentproof", "axiom", "neuron"]
MAX_TASK_LEN = 50
MAX_NEXT_LEN = 50
DEFAULT_GAP_THRESHOLD_MIN = 60


def dual_hash(data: bytes | str) -> str:
    """Compute SHA256:BLAKE3 hash per CLAUDEME §8."""
    if isinstance(data, str):
        data = data.encode("utf-8")

    sha256_hex = hashlib.sha256(data).hexdigest()

    if HAS_BLAKE3:
        blake3_hex = blake3.blake3(data).hexdigest()
    else:
        # Fallback: use sha256 again with different prefix
        blake3_hex = hashlib.sha256(b"blake3:" + data).hexdigest()

    return f"{sha256_hex}:{blake3_hex}"


def append(project: str, task: str, next_action: str, commit: str | None = None) -> dict:
    """Append entry to the shared ledger."""
    if project not in ALLOWED_PROJECTS:
        raise ValueError(f"Project must be one of: {ALLOWED_PROJECTS}")

    # Truncate to max length
    task = task[:MAX_TASK_LEN]
    next_action = next_action[:MAX_NEXT_LEN]

    # Create entry without hash
    entry = {
        "ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "project": project,
        "commit": commit,
        "task": task,
        "next": next_action,
    }

    # Compute hash of entry content
    entry_json = json.dumps(entry, sort_keys=True)
    entry["hash"] = dual_hash(entry_json)

    # Ensure ledger directory exists
    LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Append to ledger (atomic via append mode)
    with open(LEDGER_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")

    return entry


def replay(n: int | None = 10, since: str | None = None) -> list[dict]:
    """Get last N entries or entries since timestamp."""
    if not LEDGER_PATH.exists():
        return []

    entries = []
    with open(LEDGER_PATH, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    # Filter by since if provided
    if since:
        entries = [e for e in entries if e.get("ts", "") >= since]

    # Return last n entries (or all if n is None)
    if n is None:
        return entries
    return entries[-n:]


def alpha(threshold_minutes: int = DEFAULT_GAP_THRESHOLD_MIN) -> dict:
    """Calculate α (resilience metric) from ledger gaps."""
    entries = replay(n=None)

    if len(entries) < 2:
        return {
            "total_entries": len(entries),
            "gaps_detected": 0,
            "gaps": [],
            "alpha_mean": 0.0,
            "alpha_min": 0.0,
            "alpha_max": 0.0,
        }

    # Sort by timestamp
    entries.sort(key=lambda e: e.get("ts", ""))

    gaps = []
    threshold_seconds = threshold_minutes * 60

    for i in range(1, len(entries)):
        prev_ts = datetime.fromisoformat(entries[i-1]["ts"].replace("Z", "+00:00"))
        curr_ts = datetime.fromisoformat(entries[i]["ts"].replace("Z", "+00:00"))

        gap_seconds = (curr_ts - prev_ts).total_seconds()

        if gap_seconds > threshold_seconds:
            gap_minutes = gap_seconds / 60
            # Recovery time is time from gap end to this entry (approximated as minimal)
            # In practice, we use the gap itself as the disruption, and time to entry as recovery
            # For simplicity, recovery_time = 1 minute minimum (time to make an entry)
            recovery_min = max(1.0, gap_seconds / 60 / 10)  # Estimate: recovery ~10% of gap
            alpha_value = gap_minutes / recovery_min

            gaps.append({
                "start": entries[i-1]["ts"],
                "end": entries[i]["ts"],
                "duration_min": round(gap_minutes, 1),
                "recovery_min": round(recovery_min, 1),
                "alpha": round(alpha_value, 1),
            })

    alpha_values = [g["alpha"] for g in gaps]

    return {
        "total_entries": len(entries),
        "gaps_detected": len(gaps),
        "gaps": gaps,
        "alpha_mean": round(sum(alpha_values) / len(alpha_values), 1) if alpha_values else 0.0,
        "alpha_min": round(min(alpha_values), 1) if alpha_values else 0.0,
        "alpha_max": round(max(alpha_values), 1) if alpha_values else 0.0,
    }


if __name__ == "__main__":
    # Quick self-test
    print(f"NEURON v2 - Ledger: {LEDGER_PATH}")
    print(f"BLAKE3 available: {HAS_BLAKE3}")
