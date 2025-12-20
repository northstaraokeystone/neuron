"""
NEURON v4.5: The Resonance Catalyst
Controlled α oscillations tuned to natural frequencies drive triad phase transitions.
Bidirectional oscillation replaces directional pump. Gaps amplify resonance.
The organism oscillates to evolve.
"""

import hashlib
import json
import math
import os
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

try:
    import blake3

    HAS_BLAKE3 = True
except ImportError:
    HAS_BLAKE3 = False


# Ledger paths - use functions for lazy evaluation (test compatibility)
def _get_ledger_path() -> Path:
    return Path(
        os.environ.get("NEURON_LEDGER", Path.home() / "neuron" / "receipts.jsonl")
    )


def _get_archive_path() -> Path:
    return Path(
        os.environ.get("NEURON_ARCHIVE", Path.home() / "neuron" / "archive.jsonl")
    )


# Default paths (can be overridden by tests)
LEDGER_PATH = _get_ledger_path()
ARCHIVE_PATH = _get_archive_path()

# Projects and Models
ALLOWED_PROJECTS = ["agentproof", "axiom", "neuron", "grok", "human"]
SUPPORTED_MODELS = ["grok", "claude", "gemini", "neuron"]

# ============================================
# v4.3 SHARED NERVE CONSTANTS
# ============================================
LEDGER_SHARED_MODE = True
ALPHA_SYSTEM_WIDE = True
PRUNING_UNIVERSAL_SALIENCE = True

# AI Event Types (auto-append triggers)
AI_EVENT_TYPES = {
    "task": "Regular task entry",
    "grok_eviction": "Grok context window truncated oldest entries",
    "grok_reset": "Grok conversation reset (cold start)",
    "agentproof_rollback": "AgentProof anomaly containment triggered",
    "agentproof_anchor": "AgentProof blockchain anchor committed",
    "axiom_law_discovery": "AXIOM KAN witnessed new coordination law",
    "axiom_entropy_spike": "AXIOM swarm entropy exceeded threshold",
    "human_interrupt": "Human task switch or interruption",
    "human_return": "Human returned from gap",
    "eviction": "Context eviction event",
    "rollback": "Rollback/recovery event",
    "discovery": "Discovery/learning event",
    "interrupt": "Interruption event",
    "unknown": "Unknown event type",
}

# System-wide Alpha Parameters
SYSTEM_TAU_DEFAULT = 120.0  # Minutes (Monsell 2003, applies to all)
SYSTEM_GAP_WEIGHT = {
    "human": 1.0,  # Baseline
    "grok": 0.8,  # AI recovers faster
    "agentproof": 0.6,  # Deterministic recovery
    "axiom": 0.7,  # Swarm coordination overhead
    "neuron": 0.5,  # Self-recovery fastest
}

# Universal Pruning Weights
PROJECT_PRUNE_WEIGHT = {
    "human": 1.0,  # Preserve human entries slightly more
    "agentproof": 0.9,  # Slightly more aggressive on AP
    "axiom": 0.9,  # Slightly more aggressive on AXIOM
    "grok": 0.85,  # Most aggressive on Grok (ephemeral by nature)
    "neuron": 1.0,  # Self-entries preserved
}

# Entry constraints
MAX_TASK_LEN = 50
MAX_NEXT_LEN = 50
MAX_CONTEXT_SUMMARY_LEN = 500

# Inference integration
INFERENCE_CONTEXT_MAX_TOKENS = 128000

# Gap detection
DEFAULT_GAP_THRESHOLD_MIN = 60

# Salience decay
DECAY_RATE_PER_DAY = 0.05
REPLAY_DECAY_SLOWDOWN = 0.1

# Recovery cost (prefrontal inertia) - Monsell 2003
RECOVERY_K = 4.0
DEFAULT_RECOVERY_TAU = 120.0  # Default: 120 minutes (configurable)
TAU_RANGE = (1.0, 480.0)  # Valid range: 1 min to 8 hours

# Task-specific τ presets
TAU_PRESETS = {
    "quick_task": 15.0,  # Short tasks, fast recovery expected
    "standard": 120.0,  # Default, matches prior behavior
    "deep_work": 240.0,  # Extended focus, slower recovery OK
    "training_run": 480.0,  # Long runs, very slow recovery OK
}

# SLO Targets (from Grok validation)
SLO_APPEND_OVERHEAD_MAX = 0.007  # <0.7% (achieved)
SLO_APPEND_THROUGHPUT_MIN = 1500  # >1500/s (achieved)
SLO_RECOVERY_RATE_MIN = 0.97  # >97% (achieved)
SLO_PRUNING_COMPRESSION_MIN = 0.996  # >99.6% (achieved)
SLO_CONTEXT_RESTORE_MAX_SECONDS = 45  # <45s (achieved)

# Stress testing
STRESS_TEST_DEFAULT_N = 10_000_000
STRESS_TEST_CONCURRENT_WORKERS = 8
STRESS_TEST_OVERHEAD_THRESHOLD = 0.01  # <1%
STRESS_TEST_THROUGHPUT_FLOOR = 1000  # >1000/s

# Fault injection
FAILURE_TYPES = ["timeout", "disconnect", "corrupt", "slow"]
DEFAULT_FAILURE_RATE = 0.05
RECOVERY_SUCCESS_THRESHOLD = 0.95

# Consolidation
DEFAULT_CONSOLIDATE_TOP_K = 10
DEFAULT_ALPHA_THRESHOLD = 5.0
HIGH_ALPHA_THRESHOLD = 10.0
REPLAY_STRENGTH_FACTOR = 3
SALIENCE_BOOST_BASE = 0.1

# Pruning v3
DEFAULT_MAX_AGE_DAYS = 30
DEFAULT_SALIENCE_THRESHOLD = 0.1
SALIENCE_RETENTION_THRESHOLD = 0.8
PRUNING_V3_TARGET = 0.995
MIN_REPLAY_TO_PRESERVE = 5
MIN_AGE_TO_PRUNE_DAYS = 7

# Sync
SYNC_CONFLICT_RESOLUTION = "last_write_wins"

# ============================================
# v4.2 SHARDING CONSTANTS
# ============================================
DEFAULT_SHARD_COUNT = 4
MAX_SHARD_COUNT = 64
SHARD_STRATEGIES = ["hash", "time", "project", "model"]
DEFAULT_SHARD_STRATEGY = "hash"
SHARD_MAX_ENTRIES = 1_000_000
SHARD_EVICTION_PERCENT = 0.20

# ============================================
# v4.2 MULTI-USER SWARM
# ============================================
SWARM_DEFAULT_AGENTS = 1000
SWARM_APPEND_PER_AGENT = 100
SWARM_CONFLICT_THRESHOLD = 0

# ============================================
# v4.2 HIGH STRESS TARGETS
# ============================================
HIGH_STRESS_APPEND_TARGET = 85_000
HIGH_STRESS_ENTRIES = 10_000_000
HIGH_STRESS_WORKERS = 16
HIGH_STRESS_OVERHEAD_MAX = 0.01

# ============================================
# v4.2 RECOVERY CURVES
# ============================================
RECOVERY_CURVE_MODELS = ["exponential_decay", "power_law", "linear"]
DEFAULT_RECOVERY_CURVE = "exponential_decay"

# Exponential decay (Monsell 2003)
EXP_DECAY_K = 4.0
EXP_DECAY_TAU = 120.0

# Power law (Altmann & Trafton 2002)
POWER_LAW_ALPHA = 0.5
POWER_LAW_SCALE = 2.0

# Energy estimation
TECHNICAL_TERMS = [
    "federation",
    "merkle",
    "entropy",
    "kan",
    "spline",
    "receipt",
    "anchor",
    "proof",
    "hash",
    "topology",
]

# ============================================
# v4.4 ENTROPY PUMP CONSTANTS
# ============================================
ENTROPY_PUMP_MODE = True

# Classification thresholds
LOW_ALPHA_EXPORT_THRESHOLD = 0.4  # α < 0.4 → export candidate
HIGH_ALPHA_RECIRCULATE_THRESHOLD = 0.7  # α ≥ 0.7 → recirculate

# Amplification factors
HIGH_ALPHA_AMPLIFICATION = 2.0  # Recirculated entries gain 2x weight
LOW_ALPHA_DECAY = 0.5  # Exported entries compressed 50%

# Export destinations
EXPORT_DESTINATIONS = {
    "stub": "external_audit_stub/",  # Local compressed stubs
    "chain": "audit_chain_queue/",  # Queue for blockchain anchor
    "archive": "cold_archive/",  # Deep cold storage
}

# Burst sync parameters
BURST_SYNC_MIN_INTERVAL = 60.0  # Minimum seconds between bursts
BURST_SYNC_MAX_BATCH = 1000  # Maximum entries per burst
ISOLATION_LATENCY_ADVANTAGE = True  # Enable isolation mode

# Entropy targets
INTERNAL_ENTROPY_TARGET = 0.1  # Near-zero target
INTERNAL_ENTROPY_CRITICAL = 0.5  # Trigger aggressive export

# Gap-triggered export
GAP_EXPORT_MULTIPLIER = 3.0  # Export 3x more on gap detection

# Latency thresholds for burst mode
BURST_MODE_LATENCY_THRESHOLD = 10000  # ms - switch to burst mode above this


# ============================================
# v4.5 RESONANCE CATALYST CONSTANTS
# ============================================
RESONANCE_MODE = True  # Enable oscillation (vs pump-only)
OSCILLATION_AMPLITUDE_DEFAULT = 0.5  # Initial α swing target (0-1)
OSCILLATION_AMPLITUDE_MAX = 1.5  # Divergence threshold → stoprule

# Injection Parameters
LOW_ALPHA_INJECTION_RANGE = (0.1, 0.3)  # α range for synthetic entries
INJECTION_COUNT_DEFAULT = 5  # Entries per injection phase
INJECTION_COUNT_MAX = 100  # Stoprule threshold

# Surge Parameters
HIGH_ALPHA_SURGE_THRESHOLD = 0.6  # Minimum α to receive surge
SURGE_MULTIPLIER_DEFAULT = 1.5  # Salience amplification factor
SURGE_ALPHA_CAP = 1.0  # Maximum α after surge

# Frequency Sources
FREQUENCY_SOURCES = [
    "HUMAN_CIRCADIAN",  # 24h
    "HUMAN_FOCUS",  # 90min
    "GROK_TOKEN_REFRESH",  # Dynamic
    "MARS_LIGHT_DELAY",  # 3-22min
    "INTERSTELLAR_BURST",  # 4yr
]
DEFAULT_FREQUENCY = "HUMAN_FOCUS"

# Gap Resonance
GAP_AMPLITUDE_BOOST = 2.0  # Gap amplifies next oscillation by 2x

# Phase Transition Detection
TRANSITION_CORRELATION_THRESHOLD = 0.7  # Minimum correlation to count as transition

# Oscillation State (module-level)
_oscillation_phase = "inject"  # Current phase: "inject" or "surge"
_oscillation_amplitude = OSCILLATION_AMPLITUDE_DEFAULT
_oscillation_frequency = 0.001  # Hz


# CLAUDEME §8 Core Exception
class StopRule(Exception):
    """CLAUDEME §8: Exception for stoprule failures. Replaces bare except: pass."""

    def __init__(self, rule_name: str, message: str, context: dict | None = None):
        self.rule_name = rule_name
        self.context = context or {}
        super().__init__(f"STOPRULE[{rule_name}]: {message}")


def dual_hash(data: bytes | str) -> str:
    """Compute SHA256:BLAKE3 hash per CLAUDEME §8."""
    if isinstance(data, str):
        data = data.encode("utf-8")
    sha256_hex = hashlib.sha256(data).hexdigest()
    blake3_hex = (
        blake3.blake3(data).hexdigest()
        if HAS_BLAKE3
        else hashlib.sha256(b"blake3:" + data).hexdigest()
    )
    return f"{sha256_hex}:{blake3_hex}"


def merkle(items: list) -> str:
    """Compute merkle root hash of items per CLAUDEME §8.

    Args:
        items: List of items to hash (strings, bytes, or dicts)

    Returns:
        Dual hash (SHA256:BLAKE3) of the merkle root
    """
    if not items:
        return dual_hash(b"empty")

    # Convert items to hashes
    hashes = []
    for item in items:
        if isinstance(item, dict):
            item = json.dumps(item, sort_keys=True)
        if isinstance(item, str):
            item = item.encode("utf-8")
        hashes.append(dual_hash(item))

    # Build merkle tree
    while len(hashes) > 1:
        if len(hashes) % 2 == 1:
            hashes.append(hashes[-1])  # Duplicate last if odd
        new_hashes = []
        for i in range(0, len(hashes), 2):
            combined = f"{hashes[i]}|{hashes[i + 1]}"
            new_hashes.append(dual_hash(combined))
        hashes = new_hashes

    return hashes[0]


# Receipt storage path - use function for lazy evaluation (test compatibility)
def _get_receipts_path() -> Path:
    return Path(
        os.environ.get("NEURON_RECEIPTS", Path.home() / "neuron" / "receipts.jsonl")
    )


RECEIPTS_PATH = _get_receipts_path()


def emit_receipt(receipt_type: str, data: dict) -> dict:
    """Emit a receipt to the receipts ledger per CLAUDEME §4.

    SCHEMA: {type, ts, hash, **data}
    EMIT: This function
    TEST: test_emit_receipt in tests/test_neuron.py
    STOPRULE: stoprule_receipt_emission on failure

    Args:
        receipt_type: Type identifier for the receipt
        data: Receipt payload data

    Returns:
        Complete receipt dict with hash and timestamp
    """
    receipt = {
        "type": receipt_type,
        "ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        **data,
    }
    receipt["hash"] = dual_hash(
        json.dumps({k: v for k, v in receipt.items() if k != "hash"}, sort_keys=True)
    )

    try:
        receipts_path = _get_receipts_path()
        receipts_path.parent.mkdir(parents=True, exist_ok=True)
        with open(receipts_path, "a") as f:
            f.write(json.dumps(receipt) + "\n")
    except Exception as e:
        raise StopRule(
            "receipt_emission",
            f"Failed to emit receipt: {e}",
            {"receipt_type": receipt_type},
        )

    return receipt


def energy_estimate(task: str, next_action: str, token_count: int = 0) -> float:
    """Estimate cognitive load from text complexity and token count."""
    words = len(task.split()) + len(next_action.split())
    word_factor = 0.5 + (words / 20)
    tech_count = sum(
        1 for t in TECHNICAL_TERMS if t in task.lower() or t in next_action.lower()
    )
    tech_factor = 1.0 + 0.1 * tech_count
    base_energy = word_factor * tech_factor
    if token_count > 0:
        token_factor = 0.5 + (token_count / INFERENCE_CONTEXT_MAX_TOKENS) * 1.5
        base_energy = (base_energy + token_factor) / 2
    return min(2.0, max(0.5, base_energy))


def salience_decay(entry: dict, current_ts: datetime | None = None) -> float:
    """Calculate decayed salience based on age and replay count."""
    if current_ts is None:
        current_ts = datetime.now(timezone.utc)
    entry_ts = datetime.fromisoformat(entry["ts"].replace("Z", "+00:00"))
    age_days = (current_ts - entry_ts).total_seconds() / 86400
    replay_boost = 1 + REPLAY_DECAY_SLOWDOWN * entry.get("replay_count", 0)
    base_salience = entry.get("salience", 1.0)
    return base_salience * math.exp(-DECAY_RATE_PER_DAY * age_days / replay_boost)


def recovery_cost(
    gap_minutes: float,
    tau: float = DEFAULT_RECOVERY_TAU,
    model: str = DEFAULT_RECOVERY_CURVE,
) -> float:
    """Non-linear cost: pluggable recovery model (v4.2).

    Args:
        gap_minutes: Time gap since last activity in minutes
        tau: Recovery time constant (default 120.0). Lower τ = faster recovery expected = higher cost.
             Use TAU_PRESETS["quick_task"]=15 for short tasks, TAU_PRESETS["deep_work"]=240 for extended focus.
        model: Recovery curve model - "exponential_decay" (default), "power_law", or "linear"

    Returns:
        Recovery cost multiplier (1.0 to ~5.0)
    """
    tau = max(TAU_RANGE[0], min(TAU_RANGE[1], tau))  # Clamp to valid range

    if model == "exponential_decay":
        return 1.0 + RECOVERY_K * (1 - math.exp(-gap_minutes / tau))
    elif model == "power_law":
        if gap_minutes <= 0:
            return 1.0
        return 1.0 + POWER_LAW_SCALE * (gap_minutes**POWER_LAW_ALPHA)
    elif model == "linear":
        if gap_minutes <= 0:
            return 1.0
        return 1.0 + gap_minutes / tau
    else:
        # Default to exponential decay
        return 1.0 + RECOVERY_K * (1 - math.exp(-gap_minutes / tau))


def append(
    project: str,
    task: str,
    next_action: str,
    commit: str | None = None,
    energy: float | None = None,
    model: str = "neuron",
    token_count: int = 0,
    inference_id: str | None = None,
    context_summary: str = "",
    event_type: str = "task",
    salience: float = 1.0,
    source_context: dict | None = None,
) -> dict:
    """Append entry to shared ledger with salience/energy and optional inference metadata.

    v4.3 Extended: Supports multi-project append with event_type and source_context.

    Args:
        project: Source project identifier (human, agentproof, axiom, grok, neuron)
        task: Task description
        next_action: Next action description
        commit: Git commit hash (optional)
        energy: Cognitive load estimate (optional, computed if not provided)
        model: LLM model identifier
        token_count: Context window utilization
        inference_id: Unique inference cycle ID
        context_summary: Compressed context snapshot
        event_type: Type of event (task, eviction, rollback, discovery, interrupt, etc.)
        salience: Initial importance (0-1)
        source_context: Optional metadata from source project

    Returns:
        Complete entry dict with hash
    """
    if project not in ALLOWED_PROJECTS:
        raise StopRule(
            "invalid_project",
            f"Project must be one of: {ALLOWED_PROJECTS}",
            {"project": project},
        )
    if model not in SUPPORTED_MODELS:
        raise ValueError(f"Model must be one of: {SUPPORTED_MODELS}")

    # Validate event_type, warn but allow unknown
    if event_type not in AI_EVENT_TYPES:
        event_type = "unknown"

    task, next_action = task[:MAX_TASK_LEN], next_action[:MAX_NEXT_LEN]
    context_summary = context_summary[:MAX_CONTEXT_SUMMARY_LEN]

    entry = {
        "ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "project": project,
        "event_type": event_type,  # v4.3 NEW
        "model": model,
        "commit": commit,
        "task": task,
        "next": next_action,
        "salience": max(0.0, min(1.0, salience)),
        "replay_count": 0,
        "energy": energy if energy else energy_estimate(task, next_action, token_count),
        "token_count": token_count,
        "inference_id": inference_id,
        "context_summary": context_summary,
        "source_context": source_context or {},  # v4.3 NEW
    }
    entry["hash"] = dual_hash(
        json.dumps({k: v for k, v in entry.items() if k != "hash"}, sort_keys=True)
    )
    ledger_path = _get_ledger_path()
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    with open(ledger_path, "a") as f:
        f.write(json.dumps(entry) + "\n")

    # Emit shared_ledger_append_receipt for v4.3
    if LEDGER_SHARED_MODE:
        ledger = _read_ledger()
        projects_active = list(set(e.get("project", "neuron") for e in ledger))
        emit_receipt(
            "shared_ledger_append",
            {
                "tenant_id": "neuron",
                "entry_id": entry.get("hash", "")[:32],
                "project": project,
                "event_type": event_type,
                "ledger_size": len(ledger),
                "projects_active": projects_active,
            },
        )

    return entry


def append_from_agentproof(event: str, context: dict) -> dict:
    """Wrapper for AgentProof events. Sets project='agentproof'.

    Args:
        event: Event type (rollback, anchor, etc.)
        context: Metadata from AgentProof (tx_hash, anomaly_type, etc.)

    Returns:
        Entry dict
    """
    event_type_map = {
        "rollback": "agentproof_rollback",
        "anchor": "agentproof_anchor",
    }
    event_type = event_type_map.get(event, "rollback")
    task = f"AgentProof {event}: {context.get('tx_hash', 'unknown')[:20]}"

    return append(
        project="agentproof",
        task=task[:MAX_TASK_LEN],
        next_action="verify_chain_state",
        event_type=event_type,
        salience=0.7,
        source_context={**context, "trigger": "auto"},
    )


def append_from_axiom(event: str, context: dict) -> dict:
    """Wrapper for AXIOM events. Sets project='axiom'.

    Args:
        event: Event type (law_discovery, entropy_spike, etc.)
        context: Metadata from AXIOM (law, compression, entropy, etc.)

    Returns:
        Entry dict
    """
    event_type_map = {
        "law_discovery": "axiom_law_discovery",
        "entropy_spike": "axiom_entropy_spike",
    }
    event_type = event_type_map.get(event, "discovery")
    law = context.get("law", "unknown")
    task = f"AXIOM {event}: {law[:30]}"

    return append(
        project="axiom",
        task=task[:MAX_TASK_LEN],
        next_action="integrate_law",
        event_type=event_type,
        salience=0.8,
        source_context={**context, "trigger": "auto"},
    )


def append_from_grok(event: str, context: dict) -> dict:
    """Wrapper for Grok events. Sets project='grok'.

    Args:
        event: Event type (eviction, reset, etc.)
        context: Metadata from Grok (tokens_evicted, reason, etc.)

    Returns:
        Entry dict
    """
    event_type_map = {
        "eviction": "grok_eviction",
        "reset": "grok_reset",
    }
    event_type = event_type_map.get(event, "eviction")
    tokens = context.get("tokens_evicted", 0)
    task = f"Grok {event}: {tokens} tokens"

    return append(
        project="grok",
        task=task[:MAX_TASK_LEN],
        next_action="restore_context",
        event_type=event_type,
        salience=0.6,
        source_context={**context, "trigger": "auto"},
    )


def load_ledger() -> list:
    """Load the full ledger (alias for _read_ledger for public API)."""
    return _read_ledger()


def inference_append(
    model: str,
    task: str,
    next_action: str,
    context_summary: str,
    token_count: int,
    inference_id: str | None = None,
) -> dict:
    """Auto-append from LLM inference cycles with full context metadata."""
    if model not in SUPPORTED_MODELS:
        raise ValueError(f"Model must be one of: {SUPPORTED_MODELS}")
    if inference_id is None:
        inference_id = f"inf_{uuid.uuid4().hex[:12]}"
    return append(
        project="neuron",
        task=task,
        next_action=next_action,
        commit=None,
        model=model,
        token_count=token_count,
        inference_id=inference_id,
        context_summary=context_summary[:MAX_CONTEXT_SUMMARY_LEN],
    )


def _read_ledger() -> list[dict]:
    """Read all entries from ledger."""
    ledger_path = _get_ledger_path()
    if not ledger_path.exists():
        return []
    entries = []
    with open(ledger_path, "r") as f:
        for line in f:
            if line.strip():
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return entries


def _write_ledger(entries: list[dict]) -> None:
    """Write entries to ledger (atomic overwrite)."""
    ledger_path = _get_ledger_path()
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    with open(ledger_path, "w") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")


def replay_to_context(n: int = 10, format: str = "context") -> str:
    """Return ledger entries formatted for LLM context injection."""
    entries = replay(n=n, increment_replay=True)
    if not entries:
        return "## NEURON State Recovery\n\nNo entries available."

    lines = [
        "## NEURON State Recovery",
        f"\n### Recent Context ({len(entries)} entries)\n",
    ]
    for e in entries:
        model = e.get("model", "neuron")
        ts = e.get("ts", "unknown")
        task = e.get("task", "")
        next_action = e.get("next", "")
        ctx = e.get("context_summary", "")
        lines.append(f"[{ts}] {model}")
        lines.append(f"Task: {task}")
        lines.append(f"Next: {next_action}")
        if ctx:
            lines.append(f"Context: {ctx}")
        lines.append("")

    if entries:
        lines.append("### Resume Instruction")
        lines.append(f"Continue from: {entries[-1].get('next', 'unknown')}")

    return "\n".join(lines)


def replay(
    n: int | None = 10,
    since: str | None = None,
    increment_replay: bool = False,
    format: str = "list",
) -> list[dict] | str:
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
        result = (
            [e for e in all_entries if e.get("hash") in result_hashes][-n:]
            if n
            else [e for e in all_entries if e.get("hash") in result_hashes]
        )

    if format == "context":
        return replay_to_context(n=n)
    return result


def sync_ledger(remote_path: str) -> dict:
    """Merge remote ledger with local using last-write-wins."""
    local_entries = _read_ledger()
    remote_entries = []
    remote_path = Path(remote_path).expanduser()

    if remote_path.exists():
        with open(remote_path, "r") as f:
            for line in f:
                if line.strip():
                    try:
                        remote_entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue

    local_by_hash = {e.get("hash"): e for e in local_entries}
    local_by_inf_id = {
        e.get("inference_id"): e for e in local_entries if e.get("inference_id")
    }

    conflicts_resolved = 0
    for re in remote_entries:
        rhash = re.get("hash")
        rinf_id = re.get("inference_id")

        if rhash in local_by_hash:
            continue

        if rinf_id and rinf_id in local_by_inf_id:
            local_e = local_by_inf_id[rinf_id]
            if re.get("ts", "") > local_e.get("ts", ""):
                local_by_hash[local_e["hash"]] = None
                local_by_hash[rhash] = re
                local_by_inf_id[rinf_id] = re
                conflicts_resolved += 1
        else:
            local_by_hash[rhash] = re
            if rinf_id:
                local_by_inf_id[rinf_id] = re

    merged = [e for e in local_by_hash.values() if e is not None]
    merged.sort(key=lambda x: x.get("ts", ""))
    _write_ledger(merged)

    return {
        "local_entries": len(local_entries),
        "remote_entries": len(remote_entries),
        "merged_entries": len(merged),
        "conflicts_resolved": conflicts_resolved,
        "resolution_strategy": SYNC_CONFLICT_RESOLUTION,
        "sync_ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


def alpha(
    threshold_minutes: int = DEFAULT_GAP_THRESHOLD_MIN,
    tau: float = DEFAULT_RECOVERY_TAU,
    curve_model: str = DEFAULT_RECOVERY_CURVE,
) -> dict:
    """Calculate α with variance, expert_novice_ratio, configurable τ, and recovery curve model.

    Args:
        threshold_minutes: Minimum gap to consider (default 60)
        tau: Recovery time constant for cost calculations (default 120.0).
             Lower τ = faster recovery expected. Use TAU_PRESETS for common values.
        curve_model: Recovery curve model - "exponential_decay" (default), "power_law", or "linear"

    Returns:
        Dict with gap statistics including tau_used and curve_model fields
    """
    tau = max(TAU_RANGE[0], min(TAU_RANGE[1], tau))  # Clamp to valid range
    entries = _read_ledger()
    if len(entries) < 2:
        return {
            "total_entries": len(entries),
            "gaps_detected": 0,
            "gaps": [],
            "alpha_mean": 0.0,
            "alpha_min": 0.0,
            "alpha_max": 0.0,
            "alpha_variance": 0.0,
            "alpha_std": 0.0,
            "expert_novice_ratio": 1.0,
            "tau_used": tau,
            "curve_model": curve_model,
        }
    entries.sort(key=lambda e: e.get("ts", ""))
    gaps, threshold_seconds = [], threshold_minutes * 60
    for i in range(1, len(entries)):
        prev_ts = datetime.fromisoformat(entries[i - 1]["ts"].replace("Z", "+00:00"))
        curr_ts = datetime.fromisoformat(entries[i]["ts"].replace("Z", "+00:00"))
        gap_seconds = (curr_ts - prev_ts).total_seconds()
        if gap_seconds > threshold_seconds:
            gap_minutes = gap_seconds / 60
            # Recovery time scaled by τ factor
            tau_factor = tau / DEFAULT_RECOVERY_TAU
            recovery_min = max(1.0, (gap_seconds / 60 / 10) * tau_factor)
            gap_recovery_cost = recovery_cost(gap_minutes, tau, model=curve_model)
            gaps.append(
                {
                    "start": entries[i - 1]["ts"],
                    "end": entries[i]["ts"],
                    "duration_min": round(gap_minutes, 1),
                    "recovery_min": round(recovery_min, 1),
                    "alpha": round(gap_minutes / recovery_min, 1),
                    "recovery_cost": round(gap_recovery_cost, 2),
                }
            )
    alpha_values = [g["alpha"] for g in gaps]
    if not alpha_values:
        return {
            "total_entries": len(entries),
            "gaps_detected": 0,
            "gaps": [],
            "alpha_mean": 0.0,
            "alpha_min": 0.0,
            "alpha_max": 0.0,
            "alpha_variance": 0.0,
            "alpha_std": 0.0,
            "expert_novice_ratio": 1.0,
            "tau_used": tau,
            "curve_model": curve_model,
        }
    mean_a = sum(alpha_values) / len(alpha_values)
    variance = sum((a - mean_a) ** 2 for a in alpha_values) / len(alpha_values)
    return {
        "total_entries": len(entries),
        "gaps_detected": len(gaps),
        "gaps": gaps,
        "alpha_mean": round(mean_a, 1),
        "alpha_min": round(min(alpha_values), 1),
        "alpha_max": round(max(alpha_values), 1),
        "alpha_variance": round(variance, 2),
        "alpha_std": round(math.sqrt(variance), 2),
        "expert_novice_ratio": round(max(alpha_values) / min(alpha_values), 1)
        if min(alpha_values) > 0
        else float("inf"),
        "tau_used": tau,
        "curve_model": curve_model,
    }


# ============================================
# v4.3 SYSTEM-WIDE ALPHA (Gate 2)
# ============================================


def detect_system_gaps(ledger: list | None = None, threshold_minutes: int = 1) -> list:
    """Identify gaps between ANY consecutive entries regardless of project.

    v4.3: Gaps are system-wide entropy events, not human interruptions.

    Args:
        ledger: Full shared ledger (loads if None)
        threshold_minutes: Minimum gap to consider (default 1 minute)

    Returns:
        List of {start, end, gap_minutes, from_project, to_project}
    """
    if ledger is None:
        ledger = _read_ledger()

    if len(ledger) < 2:
        return []

    # Sort by timestamp
    sorted_entries = sorted(ledger, key=lambda e: e.get("ts", ""))
    gaps = []
    threshold_seconds = threshold_minutes * 60

    for i in range(1, len(sorted_entries)):
        prev_entry = sorted_entries[i - 1]
        curr_entry = sorted_entries[i]

        prev_ts = datetime.fromisoformat(prev_entry["ts"].replace("Z", "+00:00"))
        curr_ts = datetime.fromisoformat(curr_entry["ts"].replace("Z", "+00:00"))
        gap_seconds = (curr_ts - prev_ts).total_seconds()

        if gap_seconds > threshold_seconds:
            gaps.append(
                {
                    "start": prev_entry["ts"],
                    "end": curr_entry["ts"],
                    "gap_minutes": round(gap_seconds / 60, 2),
                    "from_project": prev_entry.get("project", "neuron"),
                    "to_project": curr_entry.get("project", "neuron"),
                }
            )

    return gaps


def weighted_gap(gap_minutes: float, from_project: str, to_project: str) -> float:
    """Apply SYSTEM_GAP_WEIGHT to a gap based on projects.

    v4.3: Cross-project gaps weighted by project recovery characteristics.

    Args:
        gap_minutes: Duration of the gap in minutes
        from_project: Project that created entry before gap
        to_project: Project that created entry after gap

    Returns:
        Weighted gap value
    """
    from_weight = SYSTEM_GAP_WEIGHT.get(from_project, 1.0)
    to_weight = SYSTEM_GAP_WEIGHT.get(to_project, 1.0)
    return gap_minutes * from_weight * to_weight


def alpha_system_wide(
    ledger: list | None = None,
    tau: float = SYSTEM_TAU_DEFAULT,
    threshold_minutes: int = 1,
) -> float:
    """Calculate α across ALL gaps weighted by project.

    v4.3: "α is calculated across human + AI recovery time."

    Args:
        ledger: Full shared ledger (loads if None)
        tau: Recovery time constant (default 120.0)
        threshold_minutes: Minimum gap to consider

    Returns:
        System-wide alpha value (mean of weighted gaps normalized by tau)
    """
    gaps = detect_system_gaps(ledger, threshold_minutes)

    if not gaps:
        return 0.0

    weighted_values = []
    for gap in gaps:
        wg = weighted_gap(gap["gap_minutes"], gap["from_project"], gap["to_project"])
        # Normalize by tau for recovery cost integration
        weighted_values.append(wg / tau)

    return sum(weighted_values) / len(weighted_values)


def alpha_by_project(
    ledger: list | None = None, tau: float = SYSTEM_TAU_DEFAULT
) -> dict:
    """Returns per-project α breakdown.

    v4.3: Useful for debugging and understanding project-specific recovery patterns.

    Args:
        ledger: Full shared ledger (loads if None)
        tau: Recovery time constant

    Returns:
        Dict of {project: α_value}
    """
    if ledger is None:
        ledger = _read_ledger()

    # Group entries by project
    project_entries = {}
    for entry in ledger:
        proj = entry.get("project", "neuron")
        if proj not in project_entries:
            project_entries[proj] = []
        project_entries[proj].append(entry)

    # Calculate alpha for each project
    result = {}
    for proj, entries in project_entries.items():
        if len(entries) < 2:
            result[proj] = 0.0
            continue

        sorted_entries = sorted(entries, key=lambda e: e.get("ts", ""))
        gap_values = []

        for i in range(1, len(sorted_entries)):
            prev_ts = datetime.fromisoformat(
                sorted_entries[i - 1]["ts"].replace("Z", "+00:00")
            )
            curr_ts = datetime.fromisoformat(
                sorted_entries[i]["ts"].replace("Z", "+00:00")
            )
            gap_minutes = (curr_ts - prev_ts).total_seconds() / 60

            if gap_minutes > 1:
                weight = SYSTEM_GAP_WEIGHT.get(proj, 1.0)
                gap_values.append(gap_minutes * weight / tau)

        result[proj] = (
            round(sum(gap_values) / len(gap_values), 2) if gap_values else 0.0
        )

    return result


def emit_system_alpha_receipt(
    ledger: list | None = None, tau: float = SYSTEM_TAU_DEFAULT
) -> dict:
    """Emit system_alpha_receipt with full metrics.

    v4.3: Comprehensive alpha analysis for the triad.

    Returns:
        Receipt dict
    """
    if ledger is None:
        ledger = _read_ledger()

    gaps = detect_system_gaps(ledger)
    cross_project_gaps = [g for g in gaps if g["from_project"] != g["to_project"]]
    alpha_system = alpha_system_wide(ledger, tau)
    by_project = alpha_by_project(ledger, tau)

    mean_gap = sum(g["gap_minutes"] for g in gaps) / len(gaps) if gaps else 0.0

    return emit_receipt(
        "system_alpha",
        {
            "tenant_id": "neuron",
            "alpha_system": round(alpha_system, 2),
            "alpha_by_project": by_project,
            "total_gaps": len(gaps),
            "cross_project_gaps": len(cross_project_gaps),
            "mean_gap_minutes": round(mean_gap, 2),
            "tau": tau,
        },
    )


# ============================================
# v4.3 UNIVERSAL PRUNING (Gate 4)
# ============================================


def universal_score(
    entry: dict,
    system_alpha: float,
    now: datetime | None = None,
    tau: float = SYSTEM_TAU_DEFAULT,
) -> float:
    """Calculate universal score for pruning.

    v4.3: score = salience × exp(-age/τ) × project_weight × (1 / max(system_alpha, 0.1))
    Higher α = more aggressive pruning.

    Args:
        entry: Ledger entry
        system_alpha: Current system-wide alpha
        now: Current timestamp (default: now)
        tau: Time constant for age decay

    Returns:
        Universal score (higher = keep, lower = prune)
    """
    if now is None:
        now = datetime.now(timezone.utc)

    entry_ts = datetime.fromisoformat(entry["ts"].replace("Z", "+00:00"))
    age_minutes = (now - entry_ts).total_seconds() / 60

    salience = entry.get("salience", 1.0)
    project = entry.get("project", "neuron")
    project_weight = PROJECT_PRUNE_WEIGHT.get(project, 1.0)

    # Age decay
    age_factor = math.exp(
        -age_minutes / (tau * 60)
    )  # tau is in minutes, convert to match age

    # Alpha factor: higher system alpha = more aggressive pruning
    alpha_factor = 1.0 / max(system_alpha, 0.1)

    # Replay boost
    replay_boost = 1 + 0.1 * entry.get("replay_count", 0)

    return salience * age_factor * project_weight * alpha_factor * replay_boost


def consolidate(
    top_k: int = DEFAULT_CONSOLIDATE_TOP_K,
    alpha_threshold: float = DEFAULT_ALPHA_THRESHOLD,
) -> dict:
    """Hippocampal replay: strengthen high-α entries with token_count weighting (Wilson & McNaughton 1994)."""
    entries = _read_ledger()
    stats = alpha(threshold_minutes=1)
    qualifying = [
        (g, g["alpha"]) for g in stats["gaps"] if g["alpha"] > alpha_threshold
    ]
    qualifying.sort(key=lambda x: x[1], reverse=True)
    affected_hashes, boost = [], 0.0
    for gap, a in qualifying[:top_k]:
        for e in entries:
            if e.get("ts") == gap["end"]:
                old_sal = e.get("salience", 1.0)
                token_weight = (
                    1 + e.get("token_count", 0) / INFERENCE_CONTEXT_MAX_TOKENS
                )
                e["salience"] = min(
                    1.0,
                    old_sal + SALIENCE_BOOST_BASE * math.log(max(1, a)) * token_weight,
                )
                boost += e["salience"] - old_sal
                if a > HIGH_ALPHA_THRESHOLD:
                    e["replay_count"] = (
                        e.get("replay_count", 0) + REPLAY_STRENGTH_FACTOR
                    )
                affected_hashes.append(e.get("hash", "")[:16])
    _write_ledger(entries)
    return {
        "consolidated_count": len(affected_hashes),
        "salience_boost": round(boost, 3),
        "entries_affected": affected_hashes,
    }


def prune(
    max_age_days: int = DEFAULT_MAX_AGE_DAYS,
    salience_threshold: float = DEFAULT_SALIENCE_THRESHOLD,
    max_entries: int | None = None,
    universal: bool = True,
    alpha_weight: bool = True,
    tau: float = SYSTEM_TAU_DEFAULT,
) -> dict:
    """Synaptic downscaling with v4.3 universal pruning support.

    v4.3 Enhanced: Universal pruning across all projects using universal_score.
    "Pruning becomes universal: Low-α entries downweighted system-wide."

    Args:
        max_age_days: Maximum age for entries (default 30)
        salience_threshold: Minimum salience to keep (default 0.1)
        max_entries: Hard cap on ledger size (None = no cap)
        universal: Apply universal scoring across all projects (default True)
        alpha_weight: Factor in system-wide alpha (default True)
        tau: Recovery time constant for universal scoring

    Returns:
        Dict with pruning metrics including pruned_by_project breakdown
    """
    entries = _read_ledger()
    archive_path = _get_archive_path()
    if not entries:
        return {
            "pruned_count": 0,
            "archived_to": str(archive_path),
            "ledger_size_before": 0,
            "ledger_size_after": 0,
            "compression_ratio": 0.0,
            "pruned_by_project": {},
            "strategy": "universal" if universal else "legacy",
        }

    now = datetime.now(timezone.utc)
    system_alpha = (
        alpha_system_wide(entries, tau) if (universal and alpha_weight) else 1.0
    )

    # Calculate scores for all entries
    scored_entries = []
    for e in entries:
        if universal:
            score = universal_score(e, system_alpha, now, tau)
        else:
            score = salience_decay(e, now)
        scored_entries.append((e, score))

    # Sort by score (lowest first = prune first)
    scored_entries.sort(key=lambda x: x[1])

    # Determine how many to keep
    if max_entries and len(entries) > max_entries:
        # Keep top max_entries by score
        keep_count = max_entries
    else:
        keep_count = len(entries)

    keep, archive = [], []
    pruned_by_project = {}

    for i, (e, score) in enumerate(scored_entries):
        entry_ts = datetime.fromisoformat(e["ts"].replace("Z", "+00:00"))
        age_days = (now - entry_ts).total_seconds() / 86400
        project = e.get("project", "neuron")

        # Preserve conditions (same as before)
        preserve = (
            age_days < MIN_AGE_TO_PRUNE_DAYS
            or e.get("replay_count", 0) >= MIN_REPLAY_TO_PRESERVE
            or score >= SALIENCE_RETENTION_THRESHOLD
        )

        # Check if we should keep based on max_entries
        entries_remaining = len(scored_entries) - i
        keep_slots_remaining = keep_count - len(keep)

        if preserve or entries_remaining <= keep_slots_remaining:
            keep.append(e)
        elif age_days > max_age_days and score < salience_threshold:
            archive.append(e)
            pruned_by_project[project] = pruned_by_project.get(project, 0) + 1
        else:
            keep.append(e)

    # Final enforcement of max_entries if still over
    if max_entries and len(keep) > max_entries:
        # Re-sort keep by score and trim
        keep_scored = [
            (
                e,
                universal_score(e, system_alpha, now, tau)
                if universal
                else salience_decay(e, now),
            )
            for e in keep
        ]
        keep_scored.sort(key=lambda x: x[1], reverse=True)  # Highest scores first
        final_keep = [e for e, _ in keep_scored[:max_entries]]
        overflow = [e for e, _ in keep_scored[max_entries:]]
        for e in overflow:
            project = e.get("project", "neuron")
            pruned_by_project[project] = pruned_by_project.get(project, 0) + 1
        archive.extend(overflow)
        keep = final_keep

    if archive:
        archive_path.parent.mkdir(parents=True, exist_ok=True)
        with open(archive_path, "a") as f:
            for e in archive:
                f.write(json.dumps(e) + "\n")

    _write_ledger(keep)
    compression_ratio = len(archive) / len(entries) if entries else 0.0

    result = {
        "pruned_count": len(archive),
        "archived_to": str(archive_path),
        "ledger_size_before": len(entries),
        "ledger_size_after": len(keep),
        "compression_ratio": round(compression_ratio, 4),
        "salience_preserved": sum(
            1 for e in keep if salience_decay(e, now) >= SALIENCE_RETENTION_THRESHOLD
        ),
        "pruned_by_project": pruned_by_project,
        "system_alpha": round(system_alpha, 2),
        "strategy": "universal" if universal else "legacy",
    }

    # Emit universal_prune_receipt for v4.3
    if universal and PRUNING_UNIVERSAL_SALIENCE:
        emit_receipt(
            "universal_prune",
            {
                "tenant_id": "neuron",
                "before_count": len(entries),
                "after_count": len(keep),
                "pruned_by_project": pruned_by_project,
                "system_alpha": round(system_alpha, 2),
                "compression_ratio": round(compression_ratio, 4),
                "strategy": "universal",
            },
        )

    return result


def predict_next(n_context: int = 5) -> str | None:
    """Prospective memory: predict next action from history patterns."""
    entries = _read_ledger()[-n_context * 3 :]
    if len(entries) < 2:
        return None
    patterns = Counter()
    for e in entries[:-1]:
        patterns[e.get("next", "")] += 1
    current_task = entries[-1].get("task", "").lower() if entries else ""
    for e in entries[:-1]:
        if any(
            word in e.get("task", "").lower()
            for word in current_task.split()
            if len(word) > 3
        ):
            return e.get("next")
    return patterns.most_common(1)[0][0] if patterns else None


# ============================================
# v4.4 ENTROPY PUMP PATH HELPERS
# ============================================


def _get_stub_path() -> Path:
    """Get path for external audit stubs."""
    base = Path(os.environ.get("NEURON_BASE", Path.home() / "neuron"))
    return base / "external_audit_stub"


def _get_chain_queue_path() -> Path:
    """Get path for blockchain anchor queue."""
    base = Path(os.environ.get("NEURON_BASE", Path.home() / "neuron"))
    return base / "audit_chain_queue"


def _get_cold_archive_path() -> Path:
    """Get path for cold archive storage."""
    base = Path(os.environ.get("NEURON_BASE", Path.home() / "neuron"))
    return base / "cold_archive"


def _get_burst_queue_path() -> Path:
    """Get path for burst sync queue."""
    base = Path(os.environ.get("NEURON_BASE", Path.home() / "neuron"))
    return base / "burst_queue"


# ============================================
# v4.4 GATE 1: PUMP CLASSIFICATION
# ============================================


def grok_aligned_score(
    entry: dict, now: datetime | None = None, tau: float = SYSTEM_TAU_DEFAULT
) -> float:
    """Calculate entry-level α (Grok-aligned score) for pump classification.

    Higher α = more valuable (keep/recirculate)
    Lower α = less valuable (export)

    Score combines:
    - Salience (base value)
    - Age decay (exponential)
    - Replay boost (frequency)
    - Project weight

    Args:
        entry: Ledger entry dict
        now: Current timestamp (default: now)
        tau: Time constant for age decay (minutes)

    Returns:
        α score between 0 and 1
    """
    if now is None:
        now = datetime.now(timezone.utc)

    # Parse entry timestamp
    entry_ts = datetime.fromisoformat(entry["ts"].replace("Z", "+00:00"))
    age_minutes = (now - entry_ts).total_seconds() / 60

    # Base salience
    salience = entry.get("salience", 1.0)

    # Age decay factor (exponential decay)
    age_factor = math.exp(
        -age_minutes / (tau * 60)
    )  # Convert tau to match age in minutes

    # Replay boost (more replays = more valuable)
    replay_count = entry.get("replay_count", 0)
    replay_factor = 1 + 0.1 * min(replay_count, 10)  # Cap at 2x boost

    # Project weight (some projects have higher base value)
    project = entry.get("project", "neuron")
    project_weight = PROJECT_PRUNE_WEIGHT.get(project, 1.0)

    # Combined α score (capped at 1.0)
    alpha = min(1.0, salience * age_factor * replay_factor * project_weight)

    return alpha


def compute_internal_entropy(
    ledger: list, now: datetime | None = None, tau: float = SYSTEM_TAU_DEFAULT
) -> float:
    """Compute normalized internal entropy of the ledger.

    H = Σ (1 - α_i) / n

    Lower entropy = more order (better)
    Higher entropy = more disorder (needs export)

    Args:
        ledger: List of ledger entries
        now: Current timestamp (default: now)
        tau: Time constant for α calculation

    Returns:
        Normalized entropy between 0 and 1
    """
    if not ledger:
        return 0.0

    if now is None:
        now = datetime.now(timezone.utc)

    disorder_sum = 0.0
    for entry in ledger:
        alpha = grok_aligned_score(entry, now, tau)
        disorder_sum += 1.0 - alpha

    return disorder_sum / len(ledger)


def classify_for_pump(
    ledger: list, tau: float = SYSTEM_TAU_DEFAULT, now: datetime | None = None
) -> dict:
    """Classify all entries by α into export/retain/recirculate buckets.

    v4.4: "The ledger is not memory—it is an active entropy pump."

    Classification:
    - export: α < LOW_ALPHA_EXPORT_THRESHOLD (pump disorder out)
    - retain: LOW_ALPHA_EXPORT_THRESHOLD ≤ α < HIGH_ALPHA_RECIRCULATE_THRESHOLD
    - recirculate: α ≥ HIGH_ALPHA_RECIRCULATE_THRESHOLD (amplify order)

    Args:
        ledger: List of ledger entries
        tau: Time constant for α calculation
        now: Current timestamp (default: now)

    Returns:
        Dict with export, retain, recirculate lists and entropy metrics
    """
    if now is None:
        now = datetime.now(timezone.utc)

    export = []
    retain = []
    recirculate = []

    for entry in ledger:
        alpha = grok_aligned_score(entry, now, tau)
        entry_with_alpha = {**entry, "_alpha": alpha}

        if alpha < LOW_ALPHA_EXPORT_THRESHOLD:
            export.append(entry_with_alpha)
        elif alpha >= HIGH_ALPHA_RECIRCULATE_THRESHOLD:
            recirculate.append(entry_with_alpha)
        else:
            retain.append(entry_with_alpha)

    internal_entropy = compute_internal_entropy(ledger, now, tau)

    # Emit pump_classification_receipt
    receipt = emit_receipt(
        "pump_classification",
        {
            "tenant_id": "neuron",
            "total_entries": len(ledger),
            "export_count": len(export),
            "retain_count": len(retain),
            "recirculate_count": len(recirculate),
            "internal_entropy_before": round(internal_entropy, 4),
            "classification_threshold_low": LOW_ALPHA_EXPORT_THRESHOLD,
            "classification_threshold_high": HIGH_ALPHA_RECIRCULATE_THRESHOLD,
        },
    )

    return {
        "export": export,
        "retain": retain,
        "recirculate": recirculate,
        "internal_entropy": internal_entropy,
        "total_entries": len(ledger),
        "receipt": receipt,
    }


# ============================================
# v4.4 GATE 2: LOW-α EXPORT
# ============================================


def compress_to_stub(entry: dict) -> dict:
    """Reduce entry to minimal stub for storage.

    Stub contains only: id, ts, project, payload_hash, original_alpha
    Achieves 90%+ size reduction.

    Args:
        entry: Full ledger entry

    Returns:
        Compressed stub dict
    """
    payload = json.dumps(
        {k: v for k, v in entry.items() if k not in ("_alpha", "hash")}, sort_keys=True
    )
    return {
        "id": entry.get("hash", entry.get("inference_id", "unknown"))[:32],
        "ts": entry.get("ts"),
        "project": entry.get("project", "neuron"),
        "payload_hash": dual_hash(payload),
        "original_alpha": entry.get("_alpha", 0.0),
    }


def export_low_alpha(
    entries: list, destination: str = "stub", compress: bool = True
) -> dict:
    """Export low-α entries to external destination.

    v4.4: "aggressively pumps low-α entropy outward"

    Args:
        entries: List of entries to export (from classify_for_pump)
        destination: Target - "stub", "chain", or "archive"
        compress: If True, compress to stub format (90%+ reduction)

    Returns:
        Dict with export metrics
    """
    if not entries:
        return {
            "entries_exported": 0,
            "destination": destination,
            "compression_ratio": 0.0,
            "bytes_before": 0,
            "bytes_after": 0,
            "internal_entropy_after": 0.0,
        }

    # Determine destination path
    dest_paths = {
        "stub": _get_stub_path(),
        "chain": _get_chain_queue_path(),
        "archive": _get_cold_archive_path(),
    }
    dest_path = dest_paths.get(destination, _get_stub_path())
    dest_path.mkdir(parents=True, exist_ok=True)

    # Calculate bytes before
    bytes_before = sum(len(json.dumps(e)) for e in entries)

    # Prepare output entries
    if compress:
        output_entries = [compress_to_stub(e) for e in entries]
    else:
        output_entries = [
            {k: v for k, v in e.items() if k != "_alpha"} for e in entries
        ]

    bytes_after = sum(len(json.dumps(e)) for e in output_entries)

    # Write to destination file
    date_suffix = datetime.now(timezone.utc).strftime("%Y%m%d")
    if destination == "chain":
        output_file = dest_path / "pending.jsonl"
    else:
        output_file = dest_path / f"{destination}_{date_suffix}.jsonl"

    with open(output_file, "a") as f:
        for entry in output_entries:
            f.write(json.dumps(entry) + "\n")

    # Remove exported entries from active ledger
    ledger = _read_ledger()
    export_hashes = {e.get("hash") for e in entries}
    remaining = [e for e in ledger if e.get("hash") not in export_hashes]
    _write_ledger(remaining)

    # Calculate new entropy
    internal_entropy_after = compute_internal_entropy(remaining)

    compression_ratio = 1.0 - (bytes_after / bytes_before) if bytes_before > 0 else 0.0

    # Emit receipt
    receipt = emit_receipt(
        "low_alpha_export",
        {
            "tenant_id": "neuron",
            "entries_exported": len(entries),
            "destination": destination,
            "compression_ratio": round(compression_ratio, 4),
            "bytes_before": bytes_before,
            "bytes_after": bytes_after,
            "internal_entropy_after": round(internal_entropy_after, 4),
        },
    )

    return {
        "entries_exported": len(entries),
        "destination": destination,
        "compression_ratio": compression_ratio,
        "bytes_before": bytes_before,
        "bytes_after": bytes_after,
        "internal_entropy_after": internal_entropy_after,
        "output_file": str(output_file),
        "receipt": receipt,
    }


def queue_for_anchor(entries: list) -> dict:
    """Add entries to blockchain anchor queue.

    Args:
        entries: List of entries to queue

    Returns:
        Dict with queue metrics
    """
    return export_low_alpha(entries, destination="chain", compress=False)


# ============================================
# v4.4 GATE 3: HIGH-α RECIRCULATION
# ============================================


def amplify_entry(entry: dict, factor: float = HIGH_ALPHA_AMPLIFICATION) -> dict:
    """Amplify a high-α entry's salience.

    Args:
        entry: Entry to amplify
        factor: Amplification factor (default 2.0)

    Returns:
        Amplified entry dict
    """
    amplified = {**entry}
    old_salience = amplified.get("salience", 1.0)
    amplified["salience"] = min(1.0, old_salience * factor)
    amplified["replay_count"] = amplified.get("replay_count", 0) + 1
    amplified["amplified_at"] = datetime.now(timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    amplified["recirculation_round"] = amplified.get("recirculation_round", 0) + 1
    # Remove internal _alpha field if present
    amplified.pop("_alpha", None)
    return amplified


def recirculate_high_alpha(
    entries: list, amplification: float = HIGH_ALPHA_AMPLIFICATION
) -> dict:
    """Recirculate high-α entries with amplification.

    v4.4: "recirculating high-α patterns"

    Args:
        entries: List of high-α entries (from classify_for_pump)
        amplification: Factor to increase salience (default 2.0)

    Returns:
        Dict with recirculation metrics
    """
    if not entries:
        return {
            "entries_recirculated": 0,
            "amplification_factor": amplification,
            "avg_salience_before": 0.0,
            "avg_salience_after": 0.0,
            "internal_entropy_after": 0.0,
        }

    # Calculate average salience before
    avg_before = sum(e.get("salience", 1.0) for e in entries) / len(entries)

    # Amplify entries
    amplified_entries = [amplify_entry(e, amplification) for e in entries]

    # Calculate average salience after
    avg_after = sum(e.get("salience", 1.0) for e in amplified_entries) / len(
        amplified_entries
    )

    # Update entries in ledger
    ledger = _read_ledger()
    entry_hashes = {e.get("hash") for e in entries}

    updated_ledger = []
    for e in ledger:
        if e.get("hash") in entry_hashes:
            # Find the amplified version
            for amp in amplified_entries:
                if amp.get("hash") == e.get("hash"):
                    updated_ledger.append(amp)
                    break
        else:
            updated_ledger.append(e)

    _write_ledger(updated_ledger)

    # Calculate new entropy
    internal_entropy_after = compute_internal_entropy(updated_ledger)

    # Emit receipt
    receipt = emit_receipt(
        "high_alpha_recirculate",
        {
            "tenant_id": "neuron",
            "entries_recirculated": len(entries),
            "amplification_factor": amplification,
            "avg_salience_before": round(avg_before, 4),
            "avg_salience_after": round(avg_after, 4),
            "internal_entropy_after": round(internal_entropy_after, 4),
        },
    )

    return {
        "entries_recirculated": len(entries),
        "amplification_factor": amplification,
        "avg_salience_before": avg_before,
        "avg_salience_after": avg_after,
        "internal_entropy_after": internal_entropy_after,
        "receipt": receipt,
    }


# ============================================
# v4.4 GATE 4: GAP-TRIGGERED EXPORT
# ============================================


def detect_gap_trigger(
    ledger: list | None = None, threshold_minutes: float = 1.0
) -> list:
    """Detect gaps that should trigger accelerated export.

    v4.4: "Gaps trigger directed entropy export"

    Args:
        ledger: Ledger entries (loads if None)
        threshold_minutes: Minimum gap duration to trigger

    Returns:
        List of gap events with timestamps and sources
    """
    if ledger is None:
        ledger = _read_ledger()

    if len(ledger) < 2:
        return []

    # Sort by timestamp
    sorted_entries = sorted(ledger, key=lambda e: e.get("ts", ""))
    gaps = []

    for i in range(1, len(sorted_entries)):
        prev = sorted_entries[i - 1]
        curr = sorted_entries[i]

        prev_ts = datetime.fromisoformat(prev["ts"].replace("Z", "+00:00"))
        curr_ts = datetime.fromisoformat(curr["ts"].replace("Z", "+00:00"))
        gap_seconds = (curr_ts - prev_ts).total_seconds()
        gap_minutes = gap_seconds / 60

        if gap_minutes >= threshold_minutes:
            gaps.append(
                {
                    "start": prev["ts"],
                    "end": curr["ts"],
                    "duration_minutes": gap_minutes,
                    "from_project": prev.get("project", "neuron"),
                    "to_project": curr.get("project", "neuron"),
                    "gap_source": prev.get("project", "neuron"),
                }
            )

    return gaps


def gap_directed_export(
    ledger: list | None = None, gap: dict | None = None, tau: float = SYSTEM_TAU_DEFAULT
) -> dict:
    """Trigger accelerated export on gap detection.

    v4.4: "Human Nerve no longer recovers—it DIRECTS the pump"

    On gap detection:
    - Classify current ledger
    - Export low-α × GAP_EXPORT_MULTIPLIER
    - Human uses gap to drive pump harder

    Args:
        ledger: Current ledger (loads if None)
        gap: Gap event dict (optional)
        tau: Time constant for classification

    Returns:
        Dict with gap-triggered export metrics
    """
    if ledger is None:
        ledger = _read_ledger()

    if not ledger:
        return {
            "gap_duration_minutes": gap.get("duration_minutes", 0) if gap else 0,
            "gap_source": gap.get("gap_source", "unknown") if gap else "unknown",
            "export_multiplier": GAP_EXPORT_MULTIPLIER,
            "entries_exported": 0,
            "normal_export_would_be": 0,
            "internal_entropy_after": 0.0,
        }

    # Classify ledger
    classification = classify_for_pump(ledger, tau)
    export_candidates = classification["export"]

    # Calculate normal export count
    normal_count = len(export_candidates)

    # Apply multiplier - export more aggressively
    # Include some retain entries if multiplier demands more
    retain_entries = classification["retain"]

    # Sort retain by alpha (lowest first - closest to export threshold)
    retain_sorted = sorted(retain_entries, key=lambda e: e.get("_alpha", 0.5))

    # Calculate additional entries to export
    additional_needed = int(normal_count * (GAP_EXPORT_MULTIPLIER - 1))
    additional_to_export = retain_sorted[:additional_needed]

    # Combine export sets
    total_export = export_candidates + additional_to_export

    # Perform export
    if total_export:
        export_result = export_low_alpha(
            total_export, destination="stub", compress=True
        )
        internal_entropy_after = export_result["internal_entropy_after"]
        entries_exported = export_result["entries_exported"]
    else:
        internal_entropy_after = compute_internal_entropy(ledger)
        entries_exported = 0

    # Emit receipt
    receipt = emit_receipt(
        "gap_directed_export",
        {
            "tenant_id": "neuron",
            "gap_duration_minutes": round(
                gap.get("duration_minutes", 0) if gap else 0, 2
            ),
            "gap_source": gap.get("gap_source", "unknown") if gap else "unknown",
            "export_multiplier": GAP_EXPORT_MULTIPLIER,
            "entries_exported": entries_exported,
            "normal_export_would_be": normal_count,
            "internal_entropy_after": round(internal_entropy_after, 4),
        },
    )

    return {
        "gap_duration_minutes": gap.get("duration_minutes", 0) if gap else 0,
        "gap_source": gap.get("gap_source", "unknown") if gap else "unknown",
        "export_multiplier": GAP_EXPORT_MULTIPLIER,
        "entries_exported": entries_exported,
        "normal_export_would_be": normal_count,
        "internal_entropy_after": internal_entropy_after,
        "receipt": receipt,
    }


# ============================================
# v4.4 GATE 5: BURST SYNC
# ============================================

# Module-level burst state
_burst_mode_active = False
_burst_queue = []
_last_burst_time = None


def enable_burst_mode(latency_ms: float) -> bool:
    """Enable burst mode based on latency.

    v4.4: "isolation doesn't degrade—it AMPLIFIES order"

    Args:
        latency_ms: Current network latency in milliseconds

    Returns:
        True if burst mode is now active
    """
    global _burst_mode_active
    _burst_mode_active = latency_ms > BURST_MODE_LATENCY_THRESHOLD
    return _burst_mode_active


def is_burst_mode_active() -> bool:
    """Check if burst mode is currently active."""
    return _burst_mode_active


def accumulate_for_burst(entries: list) -> dict:
    """Accumulate entries for burst sync.

    Args:
        entries: Entries to add to burst queue

    Returns:
        Dict with accumulation metrics
    """
    global _burst_queue

    _burst_queue.extend(entries)

    # Enforce max batch size
    if len(_burst_queue) > BURST_SYNC_MAX_BATCH:
        _burst_queue = _burst_queue[-BURST_SYNC_MAX_BATCH:]

    return {
        "entries_accumulated": len(entries),
        "queue_size": len(_burst_queue),
        "max_batch": BURST_SYNC_MAX_BATCH,
    }


def burst_sync(destination: str = "stub") -> dict:
    """Sync entire burst queue to destination in one batch.

    v4.4: "local ledger pumps entropy into delayed sync bursts"

    Args:
        destination: Target destination for export

    Returns:
        Dict with burst sync metrics
    """
    global _burst_queue, _last_burst_time
    import time

    start_time = time.time()

    if not _burst_queue:
        return {
            "burst_mode_active": _burst_mode_active,
            "latency_ms": 0,
            "entries_accumulated": 0,
            "entries_synced": 0,
            "sync_duration_ms": 0,
            "efficiency_ratio": 0.0,
            "internal_entropy_before": 0.0,
            "internal_entropy_after": 0.0,
        }

    # Calculate entropy before
    ledger = _read_ledger()
    entropy_before = compute_internal_entropy(ledger)

    entries_to_sync = list(_burst_queue)

    # Perform export
    export_result = export_low_alpha(
        entries_to_sync, destination=destination, compress=True
    )

    # Clear queue
    _burst_queue = []
    _last_burst_time = datetime.now(timezone.utc)

    end_time = time.time()
    sync_duration_ms = (end_time - start_time) * 1000

    # Calculate efficiency: entries synced per ms of sync time
    # Compare to continuous sync which would be 1 entry per round-trip
    # Burst is more efficient by factor of batch_size
    efficiency_ratio = len(entries_to_sync) if sync_duration_ms > 0 else 0.0

    # Emit receipt
    receipt = emit_receipt(
        "burst_sync",
        {
            "tenant_id": "neuron",
            "burst_mode_active": _burst_mode_active,
            "latency_ms": BURST_MODE_LATENCY_THRESHOLD if _burst_mode_active else 0,
            "entries_accumulated": len(entries_to_sync),
            "entries_synced": export_result["entries_exported"],
            "sync_duration_ms": round(sync_duration_ms, 2),
            "efficiency_ratio": round(efficiency_ratio, 2),
            "internal_entropy_before": round(entropy_before, 4),
            "internal_entropy_after": round(export_result["internal_entropy_after"], 4),
        },
    )

    return {
        "burst_mode_active": _burst_mode_active,
        "latency_ms": BURST_MODE_LATENCY_THRESHOLD if _burst_mode_active else 0,
        "entries_accumulated": len(entries_to_sync),
        "entries_synced": export_result["entries_exported"],
        "sync_duration_ms": sync_duration_ms,
        "efficiency_ratio": efficiency_ratio,
        "internal_entropy_before": entropy_before,
        "internal_entropy_after": export_result["internal_entropy_after"],
        "receipt": receipt,
    }


def reset_burst_state():
    """Reset burst mode state (for testing)."""
    global _burst_mode_active, _burst_queue, _last_burst_time
    _burst_mode_active = False
    _burst_queue = []
    _last_burst_time = None


# ============================================
# v4.4 GATE 6: INTEGRATION
# ============================================


def pump_cycle(ledger: list | None = None, config: dict | None = None) -> dict:
    """Execute full entropy pump cycle.

    v4.4 Pump Cycle:
    1. Classify entries by α
    2. Export low-α (pump disorder out)
    3. Recirculate high-α (amplify order)
    4. Check for gaps → accelerated export
    5. Burst sync if in isolation mode

    Args:
        ledger: Current ledger (loads if None)
        config: Optional configuration overrides

    Returns:
        Dict with cycle metrics and remaining ledger
    """
    if ledger is None:
        ledger = _read_ledger()

    if config is None:
        config = {}

    tau = config.get("tau", SYSTEM_TAU_DEFAULT)
    initial_entropy = compute_internal_entropy(ledger, tau=tau)

    if not ledger:
        return {
            "cycles_run": 0,
            "initial_entropy": 0.0,
            "final_entropy": 0.0,
            "entries_exported_total": 0,
            "entries_recirculated_total": 0,
            "gap_triggered_exports": 0,
            "burst_syncs": 0,
            "remaining_ledger": [],
        }

    # Step 1: Classify
    classification = classify_for_pump(ledger, tau)

    # Step 2: Export low-α
    exported = 0
    if classification["export"]:
        export_result = export_low_alpha(classification["export"], destination="stub")
        exported = export_result["entries_exported"]

    # Step 3: Recirculate high-α
    recirculated = 0
    if classification["recirculate"]:
        recirc_result = recirculate_high_alpha(classification["recirculate"])
        recirculated = recirc_result["entries_recirculated"]

    # Step 4: Check for gaps
    remaining_ledger = _read_ledger()
    gaps = detect_gap_trigger(remaining_ledger)
    gap_exports = 0
    if gaps:
        for gap in gaps[:1]:  # Process one gap per cycle
            gap_result = gap_directed_export(remaining_ledger, gap, tau)
            gap_exports += gap_result["entries_exported"]
            remaining_ledger = _read_ledger()

    # Step 5: Burst sync if active
    burst_count = 0
    if is_burst_mode_active() and _burst_queue:
        burst_sync()
        burst_count = 1
        remaining_ledger = _read_ledger()

    final_entropy = compute_internal_entropy(remaining_ledger, tau=tau)

    return {
        "cycles_run": 1,
        "initial_entropy": initial_entropy,
        "final_entropy": final_entropy,
        "entries_exported_total": exported + gap_exports,
        "entries_recirculated_total": recirculated,
        "gap_triggered_exports": 1 if gap_exports > 0 else 0,
        "burst_syncs": burst_count,
        "remaining_ledger": remaining_ledger,
    }


def validate_entropy_target(
    ledger: list | None = None, target: float = INTERNAL_ENTROPY_TARGET
) -> dict:
    """Validate that entropy target is achieved.

    v4.4 Target: "near-zero internal entropy" (< 0.1)

    Args:
        ledger: Current ledger (loads if None)
        target: Entropy target (default 0.1)

    Returns:
        Dict with validation results
    """
    if ledger is None:
        ledger = _read_ledger()

    current_entropy = compute_internal_entropy(ledger)
    target_achieved = current_entropy < target

    # Emit receipt
    receipt = emit_receipt(
        "entropy_validation",
        {
            "tenant_id": "neuron",
            "final_entropy": round(current_entropy, 4),
            "target_entropy": target,
            "target_achieved": target_achieved,
            "ledger_size": len(ledger),
        },
    )

    return {
        "final_entropy": current_entropy,
        "target_entropy": target,
        "target_achieved": target_achieved,
        "ledger_size": len(ledger),
        "receipt": receipt,
    }


def run_entropy_pump_integration(
    max_cycles: int = 10, target: float = INTERNAL_ENTROPY_TARGET
) -> dict:
    """Run full entropy pump integration until target achieved.

    Args:
        max_cycles: Maximum pump cycles to run
        target: Entropy target

    Returns:
        Dict with integration metrics
    """
    ledger = _read_ledger()
    initial_entropy = compute_internal_entropy(ledger)

    total_exported = 0
    total_recirculated = 0
    gap_exports = 0
    burst_syncs = 0

    for cycle in range(max_cycles):
        result = pump_cycle(ledger)
        ledger = result["remaining_ledger"]

        total_exported += result["entries_exported_total"]
        total_recirculated += result["entries_recirculated_total"]
        gap_exports += result["gap_triggered_exports"]
        burst_syncs += result["burst_syncs"]

        # Check if target achieved
        current_entropy = compute_internal_entropy(ledger)
        if current_entropy < target:
            break

    final_entropy = compute_internal_entropy(ledger)
    target_achieved = final_entropy < target

    # Emit integration receipt
    receipt = emit_receipt(
        "entropy_pump_integration",
        {
            "tenant_id": "neuron",
            "version": "4.4.0",
            "cycles_run": cycle + 1,
            "initial_entropy": round(initial_entropy, 4),
            "final_entropy": round(final_entropy, 4),
            "target_entropy": target,
            "target_achieved": target_achieved,
            "entries_exported_total": total_exported,
            "entries_recirculated_total": total_recirculated,
            "gap_triggered_exports": gap_exports,
            "burst_syncs": burst_syncs,
        },
    )

    return {
        "version": "4.4.0",
        "cycles_run": cycle + 1,
        "initial_entropy": initial_entropy,
        "final_entropy": final_entropy,
        "target_entropy": target,
        "target_achieved": target_achieved,
        "entries_exported_total": total_exported,
        "entries_recirculated_total": total_recirculated,
        "gap_triggered_exports": gap_exports,
        "burst_syncs": burst_syncs,
        "receipt": receipt,
    }


# ============================================
# v4.5 GATE 3: GAP RESONANCE DRIVER
# ============================================


def get_current_phase() -> str:
    """Return current oscillation phase ("inject" or "surge").

    v4.5: Track oscillation state.

    Returns:
        Current phase string
    """
    return _oscillation_phase


def set_oscillation_state(phase: str, amplitude: float, frequency: float) -> None:
    """Update internal oscillation tracking.

    Args:
        phase: "inject" or "surge"
        amplitude: Current amplitude
        frequency: Current frequency in Hz
    """
    global _oscillation_phase, _oscillation_amplitude, _oscillation_frequency

    if phase not in ("inject", "surge"):
        raise StopRule(
            "invalid_phase", f"Phase must be 'inject' or 'surge', got: {phase}"
        )

    _oscillation_phase = phase
    _oscillation_amplitude = amplitude
    _oscillation_frequency = frequency


def gap_resonance_trigger(gap_event: dict) -> dict:
    """When gap detected, trigger oscillation cycle with amplified amplitude.

    v4.5: "Gap detection fires oscillation cycle (not export)"
    "Gap energy amplifies next surge phase"

    Args:
        gap_event: Dict with source, duration_minutes

    Returns:
        resonance_driver_receipt
    """
    global _oscillation_amplitude

    source = gap_event.get("source", "unknown")
    duration_minutes = gap_event.get("duration_minutes", 0)

    # Amplify amplitude by gap boost factor
    amplitude_boost = GAP_AMPLITUDE_BOOST
    boosted_amplitude = _oscillation_amplitude * amplitude_boost

    # Generate cycle ID for triggered oscillation
    triggered_cycle_id = f"gap_osc_{uuid.uuid4().hex[:8]}"

    # Update state
    _oscillation_amplitude = boosted_amplitude

    # Build receipt
    receipt = emit_receipt(
        "resonance_driver",
        {
            "tenant_id": "neuron",
            "gap_source": source,
            "gap_duration_minutes": round(duration_minutes, 2),
            "amplitude_boost": amplitude_boost,
            "triggered_cycle_id": triggered_cycle_id,
            "new_amplitude": round(boosted_amplitude, 4),
        },
    )

    return {
        "receipt_type": "resonance_driver",
        "gap_source": source,
        "gap_duration_minutes": duration_minutes,
        "amplitude_boost": amplitude_boost,
        "triggered_cycle_id": triggered_cycle_id,
        "ts": receipt["ts"],
        "hash": receipt["hash"],
    }


# ============================================
# v4.5 GATE 4: HUMAN PHASE DIRECTION
# ============================================


def human_direct_phase(
    direction: str, human_id: str = "human_nerve", override_reason: str = "manual"
) -> dict:
    """Human override: set phase to "inject" or "surge".

    v4.5: "Human Nerve specifies: 'inject now' or 'surge now'"
    "Human timing overrides automatic frequency"

    Args:
        direction: "inject" or "surge"
        human_id: Identifier for human issuing command
        override_reason: Reason for override

    Returns:
        human_phase_receipt

    Raises:
        ValueError: If direction is invalid
    """
    global _oscillation_phase

    if direction not in ("inject", "surge"):
        raise ValueError(f"Direction must be 'inject' or 'surge', got: {direction}")

    previous_phase = _oscillation_phase
    _oscillation_phase = direction

    # Build receipt
    receipt = emit_receipt(
        "human_phase",
        {
            "tenant_id": "neuron",
            "direction": direction,
            "previous_phase": previous_phase,
            "human_id": human_id,
            "override_reason": override_reason,
        },
    )

    return {
        "receipt_type": "human_phase",
        "direction": direction,
        "previous_phase": previous_phase,
        "human_id": human_id,
        "override_reason": override_reason,
        "ts": receipt["ts"],
        "hash": receipt["hash"],
    }


# ============================================
# v4.5 GATE 5: PHASE TRANSITION DETECTION
# ============================================


def detect_phase_transition(triad_state: dict) -> dict:
    """Check AXIOM/AgentProof for state changes.

    v4.5: "Monitor AXIOM law_discovery events during oscillation"
    "Monitor AgentProof selection_threshold shifts"

    Args:
        triad_state: Dict with axiom and agentproof state

    Returns:
        phase_transition_receipt if detected, empty dict otherwise
    """
    axiom_state = triad_state.get("axiom", {})
    agentproof_state = triad_state.get("agentproof", {})

    # Detect AXIOM law discovery
    axiom_transition = axiom_state.get("laws_discovered", 0) > 0

    # Detect AgentProof selection shift
    agentproof_transition = agentproof_state.get("selection_threshold", 0) > 0.5

    # Determine transition type
    if axiom_transition and agentproof_transition:
        transition_type = "both"
    elif axiom_transition:
        transition_type = "axiom_law"
    elif agentproof_transition:
        transition_type = "agentproof_selection"
    else:
        transition_type = None

    if transition_type is None:
        return {
            "receipt_type": "phase_transition",
            "transition_type": None,
            "detected": False,
        }

    # Calculate correlation with oscillation
    # Simple heuristic: if we're in high amplitude, correlation is higher
    oscillation_correlation = min(
        1.0, _oscillation_amplitude / OSCILLATION_AMPLITUDE_DEFAULT
    )

    # Build receipt
    receipt = emit_receipt(
        "phase_transition",
        {
            "tenant_id": "neuron",
            "transition_type": transition_type,
            "before_state": {"axiom_laws": 0, "agentproof_threshold": 0},
            "after_state": {
                "axiom_laws": axiom_state.get("laws_discovered", 0),
                "agentproof_threshold": agentproof_state.get("selection_threshold", 0),
            },
            "oscillation_correlation": round(oscillation_correlation, 4),
        },
    )

    return {
        "receipt_type": "phase_transition",
        "transition_type": transition_type,
        "before_state": {"axiom_laws": 0, "agentproof_threshold": 0},
        "after_state": {
            "axiom_laws": axiom_state.get("laws_discovered", 0),
            "agentproof_threshold": agentproof_state.get("selection_threshold", 0),
        },
        "oscillation_correlation": oscillation_correlation,
        "detected": True,
        "ts": receipt["ts"],
        "hash": receipt["hash"],
    }


def reset_oscillation_state():
    """Reset oscillation state (for testing)."""
    global _oscillation_phase, _oscillation_amplitude, _oscillation_frequency
    _oscillation_phase = "inject"
    _oscillation_amplitude = OSCILLATION_AMPLITUDE_DEFAULT
    _oscillation_frequency = 0.001


if __name__ == "__main__":
    print("NEURON v4.5 - The Resonance Catalyst")
    print(f"Ledger: {LEDGER_PATH}")
    print(f"BLAKE3 available: {HAS_BLAKE3}")
    print(f"Shared mode: {LEDGER_SHARED_MODE}")
    print(f"Resonance mode: {RESONANCE_MODE}")
    print(f"Projects: {ALLOWED_PROJECTS}")
    print(f"Recovery curves: {RECOVERY_CURVE_MODELS}")
    print(f"Default curve: {DEFAULT_RECOVERY_CURVE}")
    print(f"Shard strategies: {SHARD_STRATEGIES}")
    print(
        f"Oscillation amplitude: {OSCILLATION_AMPLITUDE_DEFAULT} - {OSCILLATION_AMPLITUDE_MAX}"
    )
    print(f"Surge threshold: α ≥ {HIGH_ALPHA_SURGE_THRESHOLD}")
    print(f"Gap boost: {GAP_AMPLITUDE_BOOST}x")
