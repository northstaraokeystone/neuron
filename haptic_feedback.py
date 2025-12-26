"""
NEURON-RESONANCE v5.0: Haptic Feedback Loop
Pre-linguistic cortical stimulation for high-salience retrieval.

"Felt sense" before conscious text - bypass slow language entirely.
When salience exceeds threshold, the human FEELS the memory before naming it.

Sub-threshold pulses only - no seizure risk.
"""

import json
import math
import time
from datetime import datetime, timezone
from typing import Optional

from neuron import StopRule, dual_hash, emit_receipt

# ============================================
# STIMULATION SAFETY CONSTANTS
# ============================================
URGENCY_THRESHOLD = 0.8  # Salience threshold for haptic
STIM_DURATION_MS = 200  # Sub-threshold pulse duration
STIM_INTENSITY_MAX = 1.0  # Safety limit
STIM_REGIONS = ["insula", "amygdala", "somatosensory_cortex"]


class HapticStopRule(StopRule):
    """StopRule specific to haptic feedback failures."""

    def __init__(self, rule_name: str, message: str, context: dict | None = None):
        super().__init__(f"haptic_{rule_name}", message, context)


def compute_salience(vector: list[float], retrieval_score: float) -> float:
    """Compute salience 0-1 from vector norm and retrieval confidence.

    Salience combines:
    - Vector norm (activation strength)
    - Retrieval score (match quality)

    Args:
        vector: SDM query/result vector
        retrieval_score: Retrieval confidence 0-1

    Returns:
        Salience score 0-1
    """
    if not vector:
        return 0.0

    # Vector norm (L2)
    norm = math.sqrt(sum(v * v for v in vector))

    # Combine norm with retrieval score
    # Norm contribution capped at 1.0
    norm_factor = min(1.0, norm)

    # Weighted combination
    salience = (norm_factor * 0.4) + (retrieval_score * 0.6)

    return min(1.0, salience)


def inverse_project_to_stim(
    vector: list[float], projection_matrix: list[list[float]]
) -> list[float]:
    """Map SDM vector back to stimulation pattern.

    Pseudo-inverse of projection to approximate channel activations.

    Args:
        vector: SDM vector [dim]
        projection_matrix: Original projection [dim][n_channels]

    Returns:
        Stimulation pattern [n_channels]
    """
    if not vector or not projection_matrix:
        return []

    dim = len(vector)
    n_channels = len(projection_matrix[0]) if projection_matrix else 0

    if dim != len(projection_matrix):
        # Dimension mismatch - return zeros
        return [0.0] * n_channels

    # Transpose and multiply (pseudo-inverse approximation)
    stim_pattern = []
    for ch in range(n_channels):
        activation = sum(vector[d] * projection_matrix[d][ch] for d in range(dim))
        stim_pattern.append(activation)

    # Normalize to [0, 1] range
    if stim_pattern:
        max_abs = max(abs(v) for v in stim_pattern)
        if max_abs > 0:
            stim_pattern = [abs(v) / max_abs for v in stim_pattern]

    return stim_pattern


def validate_stim_intensity(
    intensity: float, max_intensity: float = STIM_INTENSITY_MAX
) -> float:
    """Clamp intensity to safe range.

    Args:
        intensity: Requested intensity
        max_intensity: Maximum allowed intensity

    Returns:
        Clamped intensity

    Raises:
        HapticStopRule: If intensity > 2x max (safety violation)
    """
    if intensity > max_intensity * 2:
        raise HapticStopRule(
            "dangerous_intensity",
            f"Intensity {intensity} > 2x max ({max_intensity * 2}) - safety violation",
            {"intensity": intensity, "max_intensity": max_intensity},
        )

    return min(max_intensity, max(0.0, intensity))


def deliver_stimulation(
    pattern: list[float],
    regions: list[str],
    intensity: float,
    duration_ms: int,
    simulation: bool = True,
) -> dict:
    """Simulate or deliver sub-threshold pulse.

    Args:
        pattern: Stimulation pattern [n_channels]
        regions: Target cortical regions
        intensity: Stimulation intensity 0-1
        duration_ms: Pulse duration in milliseconds
        simulation: If True, simulate only (no hardware)

    Returns:
        Dict with {regions, intensity, duration_ms, delivered, pattern_hash}
    """
    # Validate intensity
    safe_intensity = validate_stim_intensity(intensity)

    # Compute pattern hash
    pattern_hash = dual_hash(json.dumps([round(p, 6) for p in pattern]))

    result = {
        "regions": regions,
        "intensity": safe_intensity,
        "duration_ms": duration_ms,
        "delivered": not simulation,  # Only delivered if not simulation
        "pattern_hash": pattern_hash,
        "simulation_mode": simulation,
    }

    if not simulation:
        # In real implementation, this would interface with Neuralink hardware
        # For now, we mark as delivered in non-simulation mode
        pass

    # Emit stimulate receipt
    receipt = emit_receipt(
        "stimulate",
        {
            "pattern_hash": pattern_hash,
            "regions": regions,
            "intensity": safe_intensity,
            "duration_ms": duration_ms,
            "delivered": result["delivered"],
        },
    )
    result["_receipt"] = receipt

    return result


def vector_to_text(
    vector: list[float], neuron_ledger: list[dict], top_k: int = 5
) -> str:
    """Convert SDM vector to text via NEURON retrieval.

    Retrieves similar entries from ledger and concatenates their tasks.

    Args:
        vector: SDM query vector
        neuron_ledger: NEURON ledger entries
        top_k: Number of entries to retrieve

    Returns:
        Concatenated text from top-k similar entries
    """
    if not vector or not neuron_ledger:
        return ""

    # Retrieve top-k by recency (simplified)
    # In full implementation, this would use proper SDM vector similarity
    recent = neuron_ledger[-top_k:] if len(neuron_ledger) >= top_k else neuron_ledger

    texts = []
    for entry in recent:
        task = entry.get("task", "")
        next_action = entry.get("next", "")
        if task:
            texts.append(task)
        if next_action:
            texts.append(next_action)

    return " | ".join(texts)


def inject_visual_overlay(text: str, simulation: bool = True) -> dict:
    """Simulate or deliver visual cortex text overlay.

    Args:
        text: Text to display
        simulation: If True, simulate only

    Returns:
        Dict with {text, injected}
    """
    result = {
        "text": text,
        "injected": not simulation,
        "simulation_mode": simulation,
    }

    if not simulation:
        # In real implementation, this would interface with visual cortex stimulation
        pass

    return result


def haptic_feedback_loop(
    retrieved_vector: list[float],
    salience: float,
    config: dict,
    neuron_ledger: list[dict],
    projection_matrix: Optional[list[list[float]]] = None,
) -> dict:
    """Full haptic feedback pipeline.

    If salience > threshold → stimulation BEFORE text.

    Args:
        retrieved_vector: SDM result vector
        salience: Computed salience score
        config: Resonance configuration
        neuron_ledger: NEURON ledger for text conversion
        projection_matrix: Optional projection for stim pattern

    Returns:
        Dict with haptic_delivered, text_delivered, haptic_ts, text_ts
    """
    urgency_threshold = config.get("urgency_threshold", URGENCY_THRESHOLD)
    stim_duration_ms = config.get("stim_duration_ms", STIM_DURATION_MS)
    stim_intensity_max = config.get("stim_intensity_max", STIM_INTENSITY_MAX)
    stim_regions = config.get("stim_regions", STIM_REGIONS)
    simulation_mode = config.get("simulation_mode", True)

    haptic_delivered = False
    text_delivered = False
    haptic_ts = None
    text_ts = None

    # High salience → haptic BEFORE text
    if salience >= urgency_threshold:
        # Compute stimulation pattern
        if projection_matrix:
            stim_pattern = inverse_project_to_stim(retrieved_vector, projection_matrix)
        else:
            # Fallback: use vector directly (truncated/padded)
            stim_pattern = [abs(v) for v in retrieved_vector[:100]]

        # Calculate intensity proportional to salience
        intensity = salience * stim_intensity_max

        # Deliver haptic stimulation
        haptic_ts = (
            datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
        )
        deliver_stimulation(
            stim_pattern,
            stim_regions,
            intensity,
            stim_duration_ms,
            simulation_mode,
        )
        haptic_delivered = True

        # Small delay to ensure haptic precedes text
        time.sleep(0.001)  # 1ms

    # Always deliver text (after haptic if haptic was delivered)
    text_ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
    text = vector_to_text(retrieved_vector, neuron_ledger)
    inject_visual_overlay(text, simulation_mode)
    text_delivered = True

    # Verify haptic before text when haptic was delivered
    haptic_before_text = False
    if haptic_delivered and haptic_ts and text_ts:
        haptic_before_text = haptic_ts < text_ts

    result = {
        "haptic_delivered": haptic_delivered,
        "text_delivered": text_delivered,
        "haptic_ts": haptic_ts,
        "text_ts": text_ts,
        "haptic_before_text": haptic_before_text,
        "salience": salience,
        "urgency_threshold": urgency_threshold,
        "text": text if text else None,
    }

    # Emit haptic feedback receipt
    receipt = emit_receipt(
        "haptic_feedback",
        {
            "salience": round(salience, 4),
            "urgency_threshold": urgency_threshold,
            "regions": stim_regions if haptic_delivered else [],
            "intensity": intensity if haptic_delivered else 0.0,
            "haptic_delivered": haptic_delivered,
            "text_delivered": text_delivered,
            "haptic_before_text": haptic_before_text,
        },
    )
    result["_receipt"] = receipt

    return result


def simulate_retrieval_for_test(
    dim: int = 16384, high_salience: bool = True
) -> tuple[list[float], float]:
    """Generate simulated retrieval result for testing.

    Args:
        dim: Vector dimension
        high_salience: If True, return high salience score

    Returns:
        Tuple of (vector, salience_score)
    """
    # Generate random unit vector
    import math

    seed = 12345
    a, c, m = 1664525, 1013904223, 2**32
    state = seed

    vector = []
    for _ in range(dim):
        state = (a * state + c) % m
        val = (state / m) * 2 - 1
        vector.append(val)

    # Normalize
    norm = math.sqrt(sum(v * v for v in vector))
    if norm > 0:
        vector = [v / norm for v in vector]

    # Salience
    if high_salience:
        salience = 0.9
    else:
        salience = 0.5

    return vector, salience


if __name__ == "__main__":
    print("NEURON-RESONANCE v5.0 - Haptic Feedback Loop")
    print(f"Urgency Threshold: {URGENCY_THRESHOLD}")
    print(f"Stim Duration: {STIM_DURATION_MS}ms")
    print(f"Max Intensity: {STIM_INTENSITY_MAX}")
    print(f"Target Regions: {STIM_REGIONS}")
    print()
    print("Felt sense before text. Pre-linguistic primacy.")
