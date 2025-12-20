"""
NEURON v4.6: Chain Rhythm Conductor
Gap-derived self-conduction: silence writes the score, survival selects the notes.
The ledger is a self-conducting score written by silence.
~400 lines. The chain plays its own symphony.
"""

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

try:
    import blake3

    HAS_BLAKE3 = True
except ImportError:
    HAS_BLAKE3 = False


# ============================================
# v4.6 CHAIN RHYTHM CONDUCTOR CONSTANTS
# ============================================

# Rhythm Source - ONLY source, no external tuning
RHYTHM_SOURCE = "gaps_live"

# Alpha Mode - survival under gaps = weight
ALPHA_MODE = "persistence_survival"

# Human Role - note, not conductor
HUMAN_ROLE = "meta_steer_optional"

# Self-Conduct - chain conducts itself
SELF_CONDUCT_ENABLED = True

# Persistence Parameters
PERSISTENCE_WINDOW_ENTRIES = 1000  # History window for survival analysis
MIN_GAP_MS_FOR_RHYTHM = 100  # Minimum gap to register in rhythm


# ============================================
# STOPRULE - per CLAUDEME section 8
# ============================================
class ConductorStopRule(Exception):
    """CLAUDEME section 8: Exception for conductor stoprule violations."""

    def __init__(self, rule_name: str, message: str, context: dict | None = None):
        self.rule_name = rule_name
        self.context = context or {}
        super().__init__(f"STOPRULE[{rule_name}]: {message}")


# ============================================
# HASHING - per CLAUDEME section 8
# ============================================
def dual_hash(data: bytes | str) -> str:
    """Compute SHA256:BLAKE3 hash per CLAUDEME section 8."""
    if isinstance(data, str):
        data = data.encode("utf-8")
    sha256_hex = hashlib.sha256(data).hexdigest()
    blake3_hex = (
        blake3.blake3(data).hexdigest()
        if HAS_BLAKE3
        else hashlib.sha256(b"blake3:" + data).hexdigest()
    )
    return f"{sha256_hex}:{blake3_hex}"


def _get_receipts_path() -> Path:
    """Get path to receipts ledger."""
    return Path(
        os.environ.get("NEURON_RECEIPTS", Path.home() / "neuron" / "receipts.jsonl")
    )


def emit_receipt(receipt_type: str, data: dict) -> dict:
    """Emit a receipt to the receipts ledger per CLAUDEME section 4.

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
        raise ConductorStopRule(
            "receipt_emission",
            f"Failed to emit receipt: {e}",
            {"receipt_type": receipt_type},
        )

    return receipt


# ============================================
# GATE 1: SELF-CONDUCT SPEC
# ============================================
def _get_spec_path() -> Path:
    """Get path to self_conduct_spec.json."""
    base = Path(os.environ.get("NEURON_BASE", Path(__file__).parent))
    return base / "data" / "self_conduct_spec.json"


def load_self_conduct_spec(path: str | None = None) -> dict:
    """Load and validate self_conduct_spec.json.

    v4.6: "Load spec, compute dual-hash, emit spec_ingest_receipt"

    Args:
        path: Path to spec file (default: data/self_conduct_spec.json)

    Returns:
        Validated spec dict

    Raises:
        ConductorStopRule: If rhythm_source != "gaps_live" or spec malformed
    """
    if path is None:
        path = str(_get_spec_path())

    spec_path = Path(path)

    if not spec_path.exists():
        raise ConductorStopRule(
            "spec_not_found",
            f"Self-conduct spec not found: {path}",
            {"path": path},
        )

    try:
        with open(spec_path) as f:
            spec = json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        raise ConductorStopRule(
            "spec_malformed",
            f"Failed to parse spec: {e}",
            {"path": path},
        )

    # Validate required fields
    if spec.get("rhythm_source") != "gaps_live":
        raise ConductorStopRule(
            "invalid_rhythm_source",
            f"rhythm_source must be 'gaps_live', got: {spec.get('rhythm_source')}",
            {"rhythm_source": spec.get("rhythm_source")},
        )

    if spec.get("alpha_mode") != "persistence_survival":
        raise ConductorStopRule(
            "invalid_alpha_mode",
            f"alpha_mode must be 'persistence_survival', got: {spec.get('alpha_mode')}",
            {"alpha_mode": spec.get("alpha_mode")},
        )

    if spec.get("human_role") != "meta_steer_optional":
        raise ConductorStopRule(
            "invalid_human_role",
            f"human_role must be 'meta_steer_optional', got: {spec.get('human_role')}",
            {"human_role": spec.get("human_role")},
        )

    if not spec.get("self_conduct_enabled", False):
        raise ConductorStopRule(
            "self_conduct_disabled",
            "self_conduct_enabled must be true",
            {"self_conduct_enabled": spec.get("self_conduct_enabled")},
        )

    # Compute dual-hash of spec
    spec_content = json.dumps(spec, sort_keys=True)
    spec_hash = dual_hash(spec_content)

    # Emit spec_ingest_receipt
    receipt = emit_receipt(
        "spec_ingest",
        {
            "spec_hash": spec_hash,
            "rhythm_source": spec["rhythm_source"],
            "alpha_mode": spec["alpha_mode"],
            "human_role": spec["human_role"],
            "self_conduct_enabled": spec["self_conduct_enabled"],
        },
    )

    spec["_spec_hash"] = spec_hash
    spec["_ingest_receipt"] = receipt

    return spec


# ============================================
# GATE 2: GAP RHYTHM DERIVATION
# ============================================
def derive_rhythm_from_gaps(gap_history: list[dict]) -> dict:
    """Analyze gap timing patterns to derive rhythm score.

    v4.6: "Chain derives rhythm directly from gap patterns"
    "NO external input - gaps compose the score"

    Args:
        gap_history: List of gap events with ts, duration_ms

    Returns:
        Dict with rhythm_pattern, tempo_ms, gap_count

    Emits:
        gap_rhythm_receipt
    """
    if not gap_history:
        result = {
            "rhythm_pattern": "silent",
            "tempo_ms": 0,
            "gap_count": 0,
            "derivation_method": "gaps_live",
        }
        receipt = emit_receipt("gap_rhythm", result)
        result["_receipt"] = receipt
        return result

    # Extract gap durations
    durations = []
    for gap in gap_history:
        duration = gap.get("duration_ms", 0)
        if duration >= MIN_GAP_MS_FOR_RHYTHM:
            durations.append(duration)

    if not durations:
        result = {
            "rhythm_pattern": "continuous",
            "tempo_ms": 0,
            "gap_count": 0,
            "derivation_method": "gaps_live",
        }
        receipt = emit_receipt("gap_rhythm", result)
        result["_receipt"] = receipt
        return result

    # Derive tempo from median gap duration
    sorted_durations = sorted(durations)
    median_idx = len(sorted_durations) // 2
    tempo_ms = sorted_durations[median_idx]

    # Determine rhythm pattern based on variance
    mean_duration = sum(durations) / len(durations)
    variance = sum((d - mean_duration) ** 2 for d in durations) / len(durations)
    std_dev = variance**0.5

    # Coefficient of variation determines pattern
    cv = std_dev / mean_duration if mean_duration > 0 else 0

    if cv < 0.3:
        rhythm_pattern = "regular"  # Low variance - steady beat
    elif cv < 0.7:
        rhythm_pattern = "syncopated"  # Medium variance - varied rhythm
    else:
        rhythm_pattern = "chaotic"  # High variance - unpredictable

    result = {
        "rhythm_pattern": rhythm_pattern,
        "tempo_ms": tempo_ms,
        "gap_count": len(durations),
        "derivation_method": "gaps_live",
        "mean_duration_ms": round(mean_duration, 2),
        "cv": round(cv, 4),
    }

    # Emit gap_rhythm_receipt
    receipt = emit_receipt(
        "gap_rhythm",
        {
            "gap_count": result["gap_count"],
            "rhythm_pattern": result["rhythm_pattern"],
            "tempo_ms": result["tempo_ms"],
            "derivation_method": result["derivation_method"],
        },
    )

    result["_receipt"] = receipt
    return result


# ============================================
# GATE 3: PERSISTENCE ALPHA
# ============================================
def calculate_persistence_alpha(entries: list[dict], gap_history: list[dict]) -> dict:
    """Weight entries by survival under longest gaps.

    v4.6: "alpha calculated as causal persistence"
    "Entries surviving longest gaps weighted highest"
    "Entries that fail under gaps get alpha -> 0"

    Args:
        entries: List of ledger entries with id, ts
        gap_history: List of gap events with ts, duration_ms

    Returns:
        Dict mapping entry_id to alpha_weight

    Emits:
        persistence_alpha_receipt
    """
    if not entries or not gap_history:
        result = {"weights": {}, "max_gap_survived_ms": 0, "entry_count": len(entries)}
        receipt = emit_receipt(
            "persistence_alpha",
            {
                "entry_count": result["entry_count"],
                "max_gap_survived_ms": result["max_gap_survived_ms"],
                "alpha_distribution": {},
            },
        )
        result["_receipt"] = receipt
        return result

    # Sort entries by timestamp
    sorted_entries = sorted(entries, key=lambda e: e.get("ts", ""))

    # Sort gaps by duration (longest first)
    sorted_gaps = sorted(
        gap_history, key=lambda g: g.get("duration_ms", 0), reverse=True
    )

    max_gap_ms = sorted_gaps[0].get("duration_ms", 0) if sorted_gaps else 0

    weights = {}

    for entry in sorted_entries:
        entry_id = entry.get("id", entry.get("hash", "unknown"))[:32]
        entry_ts_str = entry.get("ts", "")

        if not entry_ts_str:
            weights[entry_id] = 0.0
            continue

        # Parse entry timestamp
        try:
            entry_ts = datetime.fromisoformat(entry_ts_str.replace("Z", "+00:00"))
        except ValueError:
            weights[entry_id] = 0.0
            continue

        # Find longest gap this entry survived
        max_survived_gap = 0

        for gap in sorted_gaps:
            gap_ts_str = gap.get("ts", gap.get("end", ""))
            gap_duration = gap.get("duration_ms", 0)

            if not gap_ts_str:
                continue

            try:
                gap_ts = datetime.fromisoformat(gap_ts_str.replace("Z", "+00:00"))
            except ValueError:
                continue

            # Entry survives gap if it was created before gap started
            # and still exists after gap ended
            if entry_ts < gap_ts:
                max_survived_gap = max(max_survived_gap, gap_duration)

        # Alpha = survived gap / max gap (normalized to 0-1)
        if max_gap_ms > 0:
            alpha = max_survived_gap / max_gap_ms
        else:
            alpha = 1.0  # No gaps = all entries have full weight

        weights[entry_id] = round(alpha, 4)

    # Calculate distribution stats
    alpha_values = list(weights.values())
    distribution = {
        "min": min(alpha_values) if alpha_values else 0.0,
        "max": max(alpha_values) if alpha_values else 0.0,
        "mean": sum(alpha_values) / len(alpha_values) if alpha_values else 0.0,
    }

    result = {
        "weights": weights,
        "max_gap_survived_ms": max_gap_ms,
        "entry_count": len(entries),
    }

    # Emit persistence_alpha_receipt
    receipt = emit_receipt(
        "persistence_alpha",
        {
            "entry_count": result["entry_count"],
            "max_gap_survived_ms": result["max_gap_survived_ms"],
            "alpha_distribution": distribution,
        },
    )

    result["_receipt"] = receipt
    return result


# ============================================
# GATE 4: HUMAN META STEERING
# ============================================
def human_meta_append(meta_entry: dict | None) -> dict | None:
    """Accept optional meta-entry from human.

    v4.6: "Human no longer directs—instead observes and appends meta-entries"
    "Meta adds to composition, doesn't direct"
    "System functions identically with meta_entry=None"

    Args:
        meta_entry: Optional meta-entry dict from human, or None

    Returns:
        Appended meta dict or None if no meta provided

    Emits:
        meta_steer_receipt if meta provided
    """
    if meta_entry is None:
        # System functions identically without human input
        return None

    # Validate meta-entry structure
    if not isinstance(meta_entry, dict):
        raise ConductorStopRule(
            "invalid_meta_entry",
            "meta_entry must be a dict",
            {"type": type(meta_entry).__name__},
        )

    # Compute hash of meta entry
    meta_content = json.dumps(meta_entry, sort_keys=True)
    meta_hash = dual_hash(meta_content)

    # Determine steering type (observational, not directive)
    steering_type = meta_entry.get("steering_type", "observation")

    # Human can only steer, not conduct
    if steering_type in ("direct", "conduct", "command"):
        raise ConductorStopRule(
            "human_direction_attempted",
            f"Human role is meta_steer_optional, not conductor. "
            f"steering_type '{steering_type}' is not allowed.",
            {"steering_type": steering_type},
        )

    # Emit meta_steer_receipt
    receipt = emit_receipt(
        "meta_steer",
        {
            "meta_entry_hash": meta_hash,
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "steering_type": steering_type,
        },
    )

    result = {
        **meta_entry,
        "_meta_hash": meta_hash,
        "_receipt": receipt,
    }

    return result


# ============================================
# GATE 5: SELF-CONDUCT EMERGENCE
# ============================================
def verify_self_conduct(rhythm: dict, alpha_weights: dict, entry_history: list) -> dict:
    """Confirm rhythm emerges from gaps alone.

    v4.6: "No external conductor required"
    "The gaps ARE the conductor"

    Args:
        rhythm: Result from derive_rhythm_from_gaps
        alpha_weights: Result from calculate_persistence_alpha
        entry_history: List of recent entries

    Returns:
        Dict with self_conducting: bool, rhythm_metrics: dict

    Raises:
        ConductorStopRule: If external conductor detected

    Emits:
        self_conduct_receipt
    """
    # Check for signs of external conductor
    external_conductor_detected = False
    conductor_type = "self"  # Default to self-conducting

    # Check rhythm derivation method
    if rhythm.get("derivation_method") != "gaps_live":
        external_conductor_detected = True
        conductor_type = "external_rhythm"

    # Check for induced oscillation patterns in entry history
    if detect_induced_oscillation(entry_history):
        external_conductor_detected = True
        conductor_type = "induced_oscillation"

    if external_conductor_detected:
        raise ConductorStopRule(
            "external_conductor_detected",
            f"Self-conduct requires gap-derived rhythm only. "
            f"Detected: {conductor_type}",
            {"conductor_type": conductor_type},
        )

    # Calculate rhythm metrics
    weights = alpha_weights.get("weights", {})
    alpha_values = list(weights.values())

    rhythm_metrics = {
        "pattern": rhythm.get("rhythm_pattern", "unknown"),
        "tempo_ms": rhythm.get("tempo_ms", 0),
        "gap_count": rhythm.get("gap_count", 0),
        "entry_count": len(entry_history),
        "mean_alpha": sum(alpha_values) / len(alpha_values) if alpha_values else 0.0,
    }

    self_conducting = (
        rhythm.get("derivation_method") == "gaps_live"
        and not external_conductor_detected
    )

    result = {
        "self_conducting": self_conducting,
        "rhythm_metrics": rhythm_metrics,
        "conductor_type": conductor_type,
    }

    # Emit self_conduct_receipt
    receipt = emit_receipt(
        "self_conduct",
        {
            "self_conducting": result["self_conducting"],
            "rhythm_metrics": rhythm_metrics,
            "conductor_type": result["conductor_type"],
        },
    )

    result["_receipt"] = receipt
    return result


# ============================================
# RESURRECTION GUARD: INDUCED OSCILLATION DETECTION
# ============================================
def detect_induced_oscillation(entry_history: list) -> bool:
    """Check for signs of induced/pumped oscillation.

    v4.6: "Oscillation, injection, and phase control killed"
    "StopRule if induced oscillation detected"

    Signs of induced oscillation:
    - Entries with event_type "resonance_injection"
    - Entries with "oscillation_phase" in source_context
    - Entries with "surge_count" > 0
    - Entries with "amplified_at" field

    Args:
        entry_history: List of entries to check

    Returns:
        True if induced oscillation detected
    """
    if not entry_history:
        return False

    for entry in entry_history:
        # Check for resonance injection
        if entry.get("event_type") == "resonance_injection":
            return True

        # Check source context for oscillation markers
        source_ctx = entry.get("source_context", {})
        if source_ctx.get("oscillation_phase"):
            return True

        # Check for surge markers
        if entry.get("surge_count", 0) > 0:
            return True

        # Check for amplification markers
        if entry.get("amplified_at"):
            return True

        # Check for recirculation markers (v4.4 pump)
        if entry.get("recirculation_round", 0) > 0:
            return True

    return False


# ============================================
# INTEGRATION: FULL CONDUCTOR CYCLE
# ============================================
def conductor_cycle(
    ledger: list | None = None,
    gap_history: list | None = None,
    meta_entry: dict | None = None,
    spec_path: str | None = None,
) -> dict:
    """Execute full chain rhythm conductor cycle.

    v4.6 Conductor Cycle:
    1. Load and validate self_conduct_spec
    2. Derive rhythm from gaps
    3. Calculate persistence alpha
    4. Accept optional human meta-steering
    5. Verify self-conducting emergence

    Args:
        ledger: Current ledger entries (loads if None)
        gap_history: Gap events (detects if None)
        meta_entry: Optional human meta-entry
        spec_path: Path to spec file

    Returns:
        Dict with conductor cycle metrics
    """
    # Gate 1: Load spec
    spec = load_self_conduct_spec(spec_path)

    # Prepare ledger and gap history
    if ledger is None:
        ledger = []

    if gap_history is None:
        gap_history = _detect_gaps_from_ledger(ledger)

    # Gate 2: Derive rhythm from gaps
    rhythm = derive_rhythm_from_gaps(gap_history)

    # Gate 3: Calculate persistence alpha
    alpha_result = calculate_persistence_alpha(ledger, gap_history)

    # Gate 4: Human meta steering (optional)
    meta_result = human_meta_append(meta_entry)

    # Gate 5: Verify self-conduct
    conduct_result = verify_self_conduct(rhythm, alpha_result, ledger)

    return {
        "version": "4.6.0",
        "spec": {
            "rhythm_source": spec["rhythm_source"],
            "alpha_mode": spec["alpha_mode"],
            "human_role": spec["human_role"],
        },
        "rhythm": rhythm,
        "alpha": alpha_result,
        "meta": meta_result,
        "self_conducting": conduct_result["self_conducting"],
        "rhythm_metrics": conduct_result["rhythm_metrics"],
    }


def _detect_gaps_from_ledger(ledger: list) -> list:
    """Detect gaps from ledger timestamps.

    Args:
        ledger: List of ledger entries

    Returns:
        List of gap events
    """
    if len(ledger) < 2:
        return []

    sorted_entries = sorted(ledger, key=lambda e: e.get("ts", ""))
    gaps = []

    for i in range(1, len(sorted_entries)):
        prev_entry = sorted_entries[i - 1]
        curr_entry = sorted_entries[i]

        prev_ts_str = prev_entry.get("ts", "")
        curr_ts_str = curr_entry.get("ts", "")

        if not prev_ts_str or not curr_ts_str:
            continue

        try:
            prev_ts = datetime.fromisoformat(prev_ts_str.replace("Z", "+00:00"))
            curr_ts = datetime.fromisoformat(curr_ts_str.replace("Z", "+00:00"))
        except ValueError:
            continue

        gap_ms = (curr_ts - prev_ts).total_seconds() * 1000

        if gap_ms >= MIN_GAP_MS_FOR_RHYTHM:
            gaps.append(
                {
                    "ts": curr_ts_str,
                    "duration_ms": gap_ms,
                    "from_entry": prev_entry.get("hash", "")[:16],
                    "to_entry": curr_entry.get("hash", "")[:16],
                }
            )

    return gaps


if __name__ == "__main__":
    print("NEURON v4.6 - Chain Rhythm Conductor")
    print(f"Rhythm source: {RHYTHM_SOURCE}")
    print(f"Alpha mode: {ALPHA_MODE}")
    print(f"Human role: {HUMAN_ROLE}")
    print(f"Self-conduct enabled: {SELF_CONDUCT_ENABLED}")
    print(f"Persistence window: {PERSISTENCE_WINDOW_ENTRIES} entries")
    print(f"Min gap for rhythm: {MIN_GAP_MS_FOR_RHYTHM}ms")
    print()
    print("The ledger is a self-conducting score written by silence.")
    print("Gaps compose the rhythm. Survival selects the notes.")
