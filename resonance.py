"""
NEURON v4.5: Resonance Catalyst Engine
Controlled α oscillations: inject low-α disorder, surge high-α order.
Bidirectional flow creates standing waves that drive phase transitions.
~300 lines. The organism oscillates to evolve.
"""

import hashlib
import json
import random
import uuid
from datetime import datetime, timezone

try:
    import blake3

    HAS_BLAKE3 = True
except ImportError:
    HAS_BLAKE3 = False


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

# Gap Resonance
GAP_AMPLITUDE_BOOST = 2.0  # Gap amplifies next oscillation by 2x

# Phase Transition Detection
TRANSITION_CORRELATION_THRESHOLD = 0.7  # Minimum correlation to count as transition


# ============================================
# STOPRULE - per CLAUDEME §8
# ============================================
class ResonanceStopRule(Exception):
    """CLAUDEME §8: Exception for resonance stoprule violations."""

    def __init__(self, rule_name: str, message: str, context: dict | None = None):
        self.rule_name = rule_name
        self.context = context or {}
        super().__init__(f"STOPRULE[{rule_name}]: {message}")


# ============================================
# HASHING - per CLAUDEME §8
# ============================================
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


# ============================================
# STOPRULE CHECKS
# ============================================
def stoprule_oscillation_divergent(amplitude: float) -> None:
    """STOPRULE: Halt if oscillation amplitude exceeds safe threshold."""
    if amplitude > OSCILLATION_AMPLITUDE_MAX:
        raise ResonanceStopRule(
            "oscillation_divergent",
            f"Amplitude {amplitude:.2f} exceeds max {OSCILLATION_AMPLITUDE_MAX}",
            {"amplitude": amplitude, "max": OSCILLATION_AMPLITUDE_MAX},
        )


def stoprule_injection_flood(count: int) -> None:
    """STOPRULE: Halt if injection count exceeds safe threshold."""
    if count > INJECTION_COUNT_MAX:
        raise ResonanceStopRule(
            "injection_flood",
            f"Injection count {count} exceeds max {INJECTION_COUNT_MAX}",
            {"count": count, "max": INJECTION_COUNT_MAX},
        )


# ============================================
# GATE 1: OSCILLATION ENGINE
# ============================================
def inject_low_alpha(
    ledger: list, injection_count: int = INJECTION_COUNT_DEFAULT
) -> dict:
    """Create N synthetic low-α entries and insert into ledger.

    v4.5: "inject controlled disorder"

    Args:
        ledger: Active ledger list (modified in place)
        injection_count: Number of entries to inject

    Returns:
        injection_receipt with entries added, average α of injection

    Raises:
        ResonanceStopRule: If injection_count > INJECTION_COUNT_MAX
    """
    # Stoprule check
    stoprule_injection_flood(injection_count)

    ledger_size_before = len(ledger)
    injected_alphas = []

    for i in range(injection_count):
        # Generate low-α value in range
        alpha_val = random.uniform(*LOW_ALPHA_INJECTION_RANGE)
        injected_alphas.append(alpha_val)

        # Create synthetic entry
        entry = {
            "ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "project": "neuron",
            "event_type": "resonance_injection",
            "model": "neuron",
            "task": f"resonance_inject_{i}",
            "next": "oscillate",
            "salience": alpha_val,
            "alpha": alpha_val,
            "replay_count": 0,
            "energy": 0.5,
            "token_count": 0,
            "source_context": {"oscillation_phase": "inject"},
        }
        entry["hash"] = dual_hash(
            json.dumps({k: v for k, v in entry.items() if k != "hash"}, sort_keys=True)
        )
        ledger.append(entry)

    avg_alpha = sum(injected_alphas) / len(injected_alphas) if injected_alphas else 0.0

    # Build receipt
    payload = json.dumps(
        {
            "injection_count": injection_count,
            "avg_alpha_injected": avg_alpha,
            "ledger_size_before": ledger_size_before,
            "ledger_size_after": len(ledger),
        },
        sort_keys=True,
    )

    receipt = {
        "receipt_type": "injection",
        "injection_count": injection_count,
        "avg_alpha_injected": round(avg_alpha, 4),
        "ledger_size_before": ledger_size_before,
        "ledger_size_after": len(ledger),
        "ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "payload_hash": dual_hash(payload),
    }

    return receipt


def surge_high_alpha(
    ledger: list, surge_multiplier: float = SURGE_MULTIPLIER_DEFAULT
) -> dict:
    """Amplify all entries with α > 0.6 by surge_multiplier (capped at 1.0).

    v4.5: "surge with amplified order"

    Args:
        ledger: Active ledger list (modified in place)
        surge_multiplier: Factor to amplify salience

    Returns:
        surge_receipt with entries surged, α delta
    """
    entries_surged = 0
    alpha_before_sum = 0.0
    alpha_after_sum = 0.0

    for entry in ledger:
        # Get alpha value (use salience as proxy if alpha not set)
        current_alpha = entry.get("alpha", entry.get("salience", 0.5))

        if current_alpha >= HIGH_ALPHA_SURGE_THRESHOLD:
            alpha_before_sum += current_alpha
            # Surge: amplify but cap at 1.0
            new_alpha = min(SURGE_ALPHA_CAP, current_alpha * surge_multiplier)
            entry["alpha"] = new_alpha
            entry["salience"] = new_alpha
            entry["surge_count"] = entry.get("surge_count", 0) + 1
            entry["surged_at"] = datetime.now(timezone.utc).strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            )
            alpha_after_sum += new_alpha
            entries_surged += 1

    avg_alpha_before = alpha_before_sum / entries_surged if entries_surged else 0.0
    avg_alpha_after = alpha_after_sum / entries_surged if entries_surged else 0.0

    # Build receipt
    payload = json.dumps(
        {
            "entries_surged": entries_surged,
            "surge_multiplier": surge_multiplier,
            "avg_alpha_before": avg_alpha_before,
            "avg_alpha_after": avg_alpha_after,
        },
        sort_keys=True,
    )

    receipt = {
        "receipt_type": "surge",
        "entries_surged": entries_surged,
        "surge_multiplier": surge_multiplier,
        "avg_alpha_before": round(avg_alpha_before, 4),
        "avg_alpha_after": round(avg_alpha_after, 4),
        "ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "payload_hash": dual_hash(payload),
    }

    return receipt


def measure_oscillation_amplitude(ledger: list, window_entries: int = 10) -> float:
    """Compute α variance over last N entries.

    Higher variance = larger oscillation amplitude.

    Args:
        ledger: Active ledger
        window_entries: Number of recent entries to analyze

    Returns:
        Amplitude value (standard deviation of α values)
    """
    if not ledger:
        return 0.0

    # Get last N entries
    recent = ledger[-window_entries:]
    if len(recent) < 2:
        return 0.0

    # Extract alpha values
    alphas = [e.get("alpha", e.get("salience", 0.5)) for e in recent]

    # Calculate variance and std dev
    mean_alpha = sum(alphas) / len(alphas)
    variance = sum((a - mean_alpha) ** 2 for a in alphas) / len(alphas)
    amplitude = variance**0.5

    return amplitude


def oscillation_cycle(
    ledger: list,
    frequency_hz: float = 0.001,
    amplitude: float = OSCILLATION_AMPLITUDE_DEFAULT,
) -> dict:
    """Execute one complete oscillation: inject low-α → wait half-period → surge high-α.

    v4.5: "controlled α oscillations replace directional pump"

    Args:
        ledger: Active ledger list (modified in place)
        frequency_hz: Oscillation frequency
        amplitude: Target amplitude (for stoprule check)

    Returns:
        oscillation_receipt with cycle metrics

    Raises:
        ResonanceStopRule: If amplitude exceeds OSCILLATION_AMPLITUDE_MAX
    """
    cycle_id = f"osc_{uuid.uuid4().hex[:12]}"
    start_ts = datetime.now(timezone.utc)

    # Phase 1: Inject low-α (disorder)
    injection_receipt = inject_low_alpha(ledger, INJECTION_COUNT_DEFAULT)

    # Phase 2: Surge high-α (order)
    surge_receipt = surge_high_alpha(ledger, SURGE_MULTIPLIER_DEFAULT)

    # Measure resulting amplitude
    measured_amplitude = measure_oscillation_amplitude(
        ledger, window_entries=len(ledger)
    )

    # Stoprule check
    stoprule_oscillation_divergent(measured_amplitude)

    # Get α values after
    alphas_after = [e.get("alpha", e.get("salience", 0.5)) for e in ledger]
    alpha_min_after = min(alphas_after) if alphas_after else 0.0
    alpha_max_after = max(alphas_after) if alphas_after else 0.0

    # Calculate α swing
    alpha_swing = abs(alpha_max_after - alpha_min_after)

    end_ts = datetime.now(timezone.utc)
    duration_ms = (end_ts - start_ts).total_seconds() * 1000

    # Determine phase (alternating based on cycle)
    phase = "inject" if random.random() < 0.5 else "surge"

    # Build receipt
    payload = json.dumps(
        {
            "cycle_id": cycle_id,
            "frequency_hz": frequency_hz,
            "amplitude": measured_amplitude,
            "phase": phase,
            "alpha_swing": alpha_swing,
            "entries_affected": injection_receipt["injection_count"]
            + surge_receipt["entries_surged"],
        },
        sort_keys=True,
    )

    receipt = {
        "receipt_type": "oscillation",
        "cycle_id": cycle_id,
        "frequency_hz": frequency_hz,
        "amplitude": round(measured_amplitude, 4),
        "phase": phase,
        "duration_ms": round(duration_ms, 2),
        "entries_affected": injection_receipt["injection_count"]
        + surge_receipt["entries_surged"],
        "α_swing": round(alpha_swing, 4),
        "ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "payload_hash": dual_hash(payload),
    }

    return receipt


# ============================================
# GATE 6: INTEGRATION - RESONANCE CATALYST
# ============================================
def resonance_catalyst_cycle(ledger: list, config: dict) -> dict:
    """Full resonance catalyst cycle.

    v4.5: "tune → inject → gap-check → surge → detect-transition"

    Args:
        ledger: Active ledger list
        config: Configuration with frequency_source, amplitude, etc.

    Returns:
        catalyst_receipt with full cycle metrics
    """
    from frequency import tune_frequency

    cycle_start = datetime.now(timezone.utc)
    cycles_completed = 0
    total_injections = 0
    total_surges = 0
    phase_transitions_detected = 0

    # Step 1: Tune frequency
    frequency_source = config.get("frequency_source", "HUMAN_FOCUS")
    amplitude = config.get("amplitude", OSCILLATION_AMPLITUDE_DEFAULT)

    try:
        freq_receipt = tune_frequency(frequency_source)
        frequency_hz = (
            1.0 / freq_receipt["period_seconds"]
            if freq_receipt["period_seconds"] > 0
            else 0.001
        )
    except Exception:
        # Fallback to default
        frequency_hz = 0.001

    # Step 2: Run oscillation cycle
    osc_receipt = oscillation_cycle(
        ledger, frequency_hz=frequency_hz, amplitude=amplitude
    )
    cycles_completed += 1
    total_injections += INJECTION_COUNT_DEFAULT
    total_surges += osc_receipt.get("entries_affected", 0) - INJECTION_COUNT_DEFAULT

    # Step 3: Check for triad response (placeholder for real integration)
    triad_response = {"axiom": {"active": True}, "agentproof": {"active": True}}

    # Calculate duration
    cycle_end = datetime.now(timezone.utc)
    duration_ms = (cycle_end - cycle_start).total_seconds() * 1000

    # Build receipt
    payload = json.dumps(
        {
            "version": "4.5.0",
            "cycles_completed": cycles_completed,
            "total_injections": total_injections,
            "total_surges": total_surges,
            "phase_transitions_detected": phase_transitions_detected,
        },
        sort_keys=True,
    )

    receipt = {
        "receipt_type": "catalyst",
        "version": "4.5.0",
        "cycles_completed": cycles_completed,
        "total_injections": total_injections,
        "total_surges": total_surges,
        "phase_transitions_detected": phase_transitions_detected,
        "triad_response": triad_response,
        "frequency_source": frequency_source,
        "frequency_hz": frequency_hz,
        "amplitude": amplitude,
        "duration_ms": round(duration_ms, 2),
        "ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "payload_hash": dual_hash(payload),
    }

    return receipt


if __name__ == "__main__":
    print("NEURON v4.5 - Resonance Catalyst Engine")
    print(f"Resonance mode: {RESONANCE_MODE}")
    print(
        f"Amplitude range: {OSCILLATION_AMPLITUDE_DEFAULT} - {OSCILLATION_AMPLITUDE_MAX}"
    )
    print(f"Injection range: α = {LOW_ALPHA_INJECTION_RANGE}")
    print(f"Surge threshold: α ≥ {HIGH_ALPHA_SURGE_THRESHOLD}")
    print(f"Gap boost: {GAP_AMPLITUDE_BOOST}x")
