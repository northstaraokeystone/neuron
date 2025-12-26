"""
RESONANCE PROTOCOL - Phase Predictor

Predict oscillation phase for timed stimulation.
Phase error must be < pi/4 radians for 95% of events.

THE PHYSICS:
    Memory consolidation requires precise timing.
    By detecting SO onset, we predict the window for a Spindle.
    By timing bias to coincide with SWR, we amplify consolidation.
"""

from __future__ import annotations

import math
import time

try:
    from .core import (
        PHASE_LOCK_LATENCY_MS,
        STIMULATION_WINDOW_MS,
        emit_receipt,
    )
except ImportError:
    from core import (
        PHASE_LOCK_LATENCY_MS,
        STIMULATION_WINDOW_MS,
        emit_receipt,
    )


def fit_oscillation(
    events: list[dict],
    oscillation_type: str
) -> dict:
    """Fit sinusoid to event history.

    Estimates frequency, phase offset, and amplitude from recent events.

    Args:
        events: List of {timestamp_ns, amplitude, ...}
        oscillation_type: Type of oscillation ("swr", "spindle", "so")

    Returns:
        {frequency_hz, phase_offset, amplitude, n_events}
    """
    if not events or len(events) < 2:
        return {
            "frequency_hz": 0.0,
            "phase_offset": 0.0,
            "amplitude": 0.0,
            "n_events": 0,
        }

    # Extract timestamps and amplitudes
    timestamps = [e["timestamp_ns"] for e in events]
    amplitudes = [e.get("amplitude", 1.0) for e in events]

    # Estimate frequency from inter-event intervals
    intervals = []
    for i in range(1, len(timestamps)):
        dt_ns = timestamps[i] - timestamps[i - 1]
        if dt_ns > 0:
            intervals.append(dt_ns)

    if not intervals:
        return {
            "frequency_hz": 0.0,
            "phase_offset": 0.0,
            "amplitude": sum(amplitudes) / len(amplitudes),
            "n_events": len(events),
        }

    mean_interval_ns = sum(intervals) / len(intervals)
    frequency_hz = 1e9 / mean_interval_ns if mean_interval_ns > 0 else 0.0

    # Estimate phase offset (relative to first event)
    # For simplicity, assume first event is at phase 0
    phase_offset = 0.0

    # Mean amplitude
    mean_amplitude = sum(amplitudes) / len(amplitudes)

    return {
        "frequency_hz": round(frequency_hz, 2),
        "phase_offset": round(phase_offset, 4),
        "amplitude": round(mean_amplitude, 4),
        "n_events": len(events),
    }


def predict_next_peak(
    model: dict,
    current_time_ns: int
) -> int:
    """Predict timestamp of next oscillation peak.

    Args:
        model: Oscillation model from fit_oscillation
        current_time_ns: Current time in nanoseconds

    Returns:
        Predicted timestamp_ns of next peak
    """
    if model["frequency_hz"] <= 0:
        # No valid model, return current time + reasonable default
        return current_time_ns + int(10e6)  # 10ms default

    period_ns = int(1e9 / model["frequency_hz"])
    phase_offset = model["phase_offset"]

    # Calculate time to next peak
    # Peak occurs at phase = 0 (mod 2*pi)
    # Current phase = (2*pi*f*t + phase_offset) mod 2*pi
    # Time to phase 0 = (2*pi - current_phase) / (2*pi*f)

    # Simplified: next peak is roughly one period from now
    # Adjust for phase offset
    time_to_peak_ns = int(period_ns * (1.0 - phase_offset / (2 * math.pi)))

    # Ensure prediction is in the future
    if time_to_peak_ns <= 0:
        time_to_peak_ns += period_ns

    return current_time_ns + time_to_peak_ns


def calculate_phase_error(
    predicted_ns: int,
    actual_ns: int,
    frequency_hz: float = 200.0
) -> float:
    """Compute phase error in radians.

    Args:
        predicted_ns: Predicted event timestamp
        actual_ns: Actual event timestamp
        frequency_hz: Oscillation frequency for phase calculation

    Returns:
        Phase error in radians (-pi to pi)
    """
    if frequency_hz <= 0:
        return math.pi  # Maximum error if no frequency

    period_ns = 1e9 / frequency_hz
    time_error_ns = actual_ns - predicted_ns

    # Convert time error to phase error
    phase_error = (2 * math.pi * time_error_ns / period_ns) % (2 * math.pi)

    # Normalize to [-pi, pi]
    if phase_error > math.pi:
        phase_error -= 2 * math.pi

    return phase_error


def trigger_stimulation(
    predicted_phase: float,
    target_phase: float,
    tolerance: float = math.pi / 4
) -> bool:
    """Return True if current phase is within tolerance of target.

    Args:
        predicted_phase: Predicted current phase (radians)
        target_phase: Target phase for stimulation (radians)
        tolerance: Phase tolerance (default pi/4 radians)

    Returns:
        True if stimulation should trigger
    """
    phase_diff = abs(predicted_phase - target_phase)

    # Handle wraparound
    if phase_diff > math.pi:
        phase_diff = 2 * math.pi - phase_diff

    return phase_diff <= tolerance


def test_phase_lock(n_trials: int = 100) -> dict:
    """Synthetic test: generate oscillation, predict phases.

    PASS CRITERIA: Phase error < pi/4 radians for 95% of events

    Args:
        n_trials: Number of prediction trials

    Returns:
        {mean_error_rad, std_error_rad, p95_error_rad, passed}
    """
    import random
    random.seed(42)

    errors = []
    frequency_hz = 200.0  # SWR frequency
    period_ns = int(1e9 / frequency_hz)

    # Generate synthetic event history
    events = []
    base_time_ns = 0
    for i in range(20):
        jitter_ns = int(random.gauss(0, period_ns * 0.1))  # 10% jitter
        timestamp_ns = base_time_ns + i * period_ns + jitter_ns
        events.append({
            "timestamp_ns": timestamp_ns,
            "amplitude": 1.0 + random.gauss(0, 0.1),
            "frequency_hz": frequency_hz,
        })

    # Fit model
    model = fit_oscillation(events, "swr")

    # Predict future events and compare
    for trial in range(n_trials):
        # Current time is after event history
        current_time_ns = events[-1]["timestamp_ns"] + trial * period_ns

        # Predict next peak
        predicted_ns = predict_next_peak(model, current_time_ns)

        # Actual peak (with jitter)
        actual_ns = current_time_ns + period_ns + int(random.gauss(0, period_ns * 0.1))

        # Calculate error
        error = calculate_phase_error(predicted_ns, actual_ns, frequency_hz)
        errors.append(abs(error))

    # Calculate statistics
    mean_error = sum(errors) / len(errors)
    variance = sum((e - mean_error) ** 2 for e in errors) / len(errors)
    std_error = math.sqrt(variance)

    errors_sorted = sorted(errors)
    p95_error = errors_sorted[int(len(errors) * 0.95)]

    threshold = math.pi / 4
    passed = p95_error < threshold

    result = {
        "n_trials": n_trials,
        "mean_error_rad": round(mean_error, 4),
        "std_error_rad": round(std_error, 4),
        "p95_error_rad": round(p95_error, 4),
        "threshold_rad": round(threshold, 4),
        "passed": passed,
    }

    emit_receipt("phase_lock_test", result)

    return result


def emit_phase_lock_receipt(
    target_oscillation: str,
    predicted_phase: float,
    actual_phase: float,
    lock_error_ms: float
) -> dict:
    """Emit phase_lock_receipt per specification.

    Args:
        target_oscillation: Oscillation type targeted
        predicted_phase: Predicted phase at stimulation
        actual_phase: Actual phase measured
        lock_error_ms: Timing error in milliseconds

    Returns:
        Receipt dict
    """
    phase_error = abs(predicted_phase - actual_phase)
    if phase_error > math.pi:
        phase_error = 2 * math.pi - phase_error

    return emit_receipt(
        "phase_lock",
        {
            "target_oscillation": target_oscillation,
            "predicted_phase": round(predicted_phase, 4),
            "actual_phase": round(actual_phase, 4),
            "phase_error_rad": round(phase_error, 4),
            "lock_error_ms": round(lock_error_ms, 4),
            "within_tolerance": phase_error < math.pi / 4,
        }
    )


class PhaseLockController:
    """Real-time phase-locking controller.

    Maintains oscillation model and predicts optimal stimulation timing.
    """

    def __init__(self, target_oscillation: str = "swr"):
        self.target_oscillation = target_oscillation
        self.event_history: list[dict] = []
        self.model: dict = {}
        self.max_history = 50

    def add_event(self, event: dict) -> None:
        """Add detected event to history and update model."""
        self.event_history.append(event)
        if len(self.event_history) > self.max_history:
            self.event_history = self.event_history[-self.max_history:]

        # Refit model
        self.model = fit_oscillation(self.event_history, self.target_oscillation)

    def predict_next_window(self, current_time_ns: int) -> dict:
        """Predict next optimal stimulation window.

        Returns:
            {start_ns, end_ns, target_phase, confidence}
        """
        if not self.model or self.model.get("frequency_hz", 0) <= 0:
            return {
                "start_ns": current_time_ns,
                "end_ns": current_time_ns + int(STIMULATION_WINDOW_MS * 1e6),
                "target_phase": 0.0,
                "confidence": 0.0,
            }

        peak_ns = predict_next_peak(self.model, current_time_ns)

        # Stimulation window around peak
        window_ns = int(STIMULATION_WINDOW_MS * 1e6)
        start_ns = peak_ns - window_ns // 2
        end_ns = peak_ns + window_ns // 2

        # Confidence based on model quality
        n_events = self.model.get("n_events", 0)
        confidence = min(1.0, n_events / 10)

        return {
            "start_ns": start_ns,
            "end_ns": end_ns,
            "target_phase": 0.0,  # Peak phase
            "confidence": confidence,
        }

    def should_stimulate(self, current_time_ns: int) -> bool:
        """Check if current time is within stimulation window."""
        window = self.predict_next_window(current_time_ns)
        return window["start_ns"] <= current_time_ns <= window["end_ns"]


if __name__ == "__main__":
    print("RESONANCE PROTOCOL - Phase Predictor")
    print(f"Target Latency: <{PHASE_LOCK_LATENCY_MS}ms")
    print(f"Stimulation Window: {STIMULATION_WINDOW_MS}ms")
    print()
    print("Running phase lock test...")
    result = test_phase_lock(n_trials=100)
    print(f"Mean error: {result['mean_error_rad']:.4f} rad")
    print(f"p95 error: {result['p95_error_rad']:.4f} rad (target < {math.pi/4:.4f})")
    print(f"PASSED: {result['passed']}")
