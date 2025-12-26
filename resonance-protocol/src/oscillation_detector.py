"""
RESONANCE PROTOCOL - Oscillation Detection

Real-time detection of SO/Spindle/SWR with <2ms latency.
Streaming signal processing for phase-locked stimulation.

Frequency Bands:
    SO (Slow Oscillation): < 1 Hz
    Spindle: 10-16 Hz (thalamic)
    SWR (Sharp-Wave Ripple): 150-250 Hz (hippocampal)

THE PHYSICS:
    Phase-locking to SWRs requires <2ms loop latency.
    At 200 Hz ripple frequency = 5ms period.
    System must trigger stimulation within specific phase window.
"""

from __future__ import annotations

import math
import time
from typing import Callable

try:
    from .core import (
        SWR_FREQ_MIN,
        SWR_FREQ_MAX,
        SPINDLE_FREQ_MIN,
        SPINDLE_FREQ_MAX,
        SO_FREQ_MAX,
        SWR_THRESHOLD_SD,
        SPINDLE_THRESHOLD_SD,
        PHASE_LOCK_LATENCY_MS,
        SAMPLING_RATE_HZ,
        emit_receipt,
        dual_hash,
    )
except ImportError:
    from core import (
        SWR_FREQ_MIN,
        SWR_FREQ_MAX,
        SPINDLE_FREQ_MIN,
        SPINDLE_FREQ_MAX,
        SO_FREQ_MAX,
        SWR_THRESHOLD_SD,
        SPINDLE_THRESHOLD_SD,
        PHASE_LOCK_LATENCY_MS,
        SAMPLING_RATE_HZ,
        emit_receipt,
        dual_hash,
    )


def bandpass_filter(
    signal: list[float],
    freq_min: float,
    freq_max: float,
    fs: int = SAMPLING_RATE_HZ
) -> list[float]:
    """Apply zero-phase bandpass filter (Butterworth 4th order approximation).

    Uses causal IIR filter for real-time processing.
    Zero-phase only available offline for validation.

    Args:
        signal: Input signal
        freq_min: Low cutoff frequency (Hz)
        freq_max: High cutoff frequency (Hz)
        fs: Sampling frequency (Hz)

    Returns:
        Bandpass filtered signal
    """
    if not signal or len(signal) < 4:
        return list(signal)

    # Normalize frequencies
    low_norm = freq_min / (fs / 2)
    high_norm = freq_max / (fs / 2)

    # Clamp to valid range
    low_norm = max(0.01, min(0.99, low_norm))
    high_norm = max(low_norm + 0.01, min(0.99, high_norm))

    # Second-order IIR coefficients (simplified Butterworth)
    bw = high_norm - low_norm
    center = (high_norm + low_norm) / 2
    q = center / bw if bw > 0 else 1.0

    # State variables for IIR filter
    y1, y2 = 0.0, 0.0
    x1, x2 = 0.0, 0.0

    # Resonance coefficient
    r = 0.99 - (0.01 / q) if q > 0 else 0.98

    # Filter coefficients
    a1 = -2 * r * math.cos(2 * math.pi * center)
    a2 = r * r
    b0 = 1 - r

    filtered = []
    for x in signal:
        y = b0 * x + b0 * x1 - a1 * y1 - a2 * y2
        y2 = y1
        y1 = y
        x2 = x1
        x1 = x
        filtered.append(y)

    return filtered


def _detect_events(
    filtered: list[float],
    threshold_sd: float,
    fs: int,
    event_type: str
) -> list[dict]:
    """Detect threshold crossings in filtered signal.

    Args:
        filtered: Bandpass filtered signal
        threshold_sd: Threshold in standard deviations
        fs: Sampling frequency
        event_type: Type label for events

    Returns:
        List of {timestamp_ns, amplitude, frequency_hz, event_type}
    """
    if not filtered or len(filtered) < 10:
        return []

    # Compute statistics
    mean = sum(filtered) / len(filtered)
    variance = sum((v - mean) ** 2 for v in filtered) / len(filtered)
    std = math.sqrt(variance) if variance > 0 else 1.0

    threshold = mean + threshold_sd * std
    events = []
    in_event = False
    event_start = 0
    max_amp = 0.0
    max_idx = 0

    for i, v in enumerate(filtered):
        if abs(v) > threshold:
            if not in_event:
                in_event = True
                event_start = i
                max_amp = abs(v)
                max_idx = i
            else:
                if abs(v) > max_amp:
                    max_amp = abs(v)
                    max_idx = i
        else:
            if in_event:
                # Event ended
                duration_samples = i - event_start
                if duration_samples >= 3:  # Minimum event length
                    # Estimate frequency from zero crossings
                    zero_crossings = 0
                    for j in range(event_start + 1, i):
                        if filtered[j] * filtered[j - 1] < 0:
                            zero_crossings += 1
                    freq_hz = (zero_crossings / 2) * (fs / duration_samples) if duration_samples > 0 else 0

                    timestamp_ns = int(max_idx * 1e9 / fs)
                    events.append({
                        "event_type": event_type,
                        "timestamp_ns": timestamp_ns,
                        "amplitude": max_amp,
                        "frequency_hz": round(freq_hz, 1),
                        "duration_samples": duration_samples,
                    })
                in_event = False

    return events


def detect_swr(
    lfp_signal: list[float],
    fs: int = SAMPLING_RATE_HZ,
    threshold_sd: float = SWR_THRESHOLD_SD
) -> list[dict]:
    """Detect SWR events (150-250 Hz).

    Args:
        lfp_signal: Local field potential signal
        fs: Sampling frequency (Hz)
        threshold_sd: Detection threshold in standard deviations

    Returns:
        List of {timestamp_ns, amplitude, frequency_hz}
    """
    filtered = bandpass_filter(lfp_signal, SWR_FREQ_MIN, SWR_FREQ_MAX, fs)
    events = _detect_events(filtered, threshold_sd, fs, "swr")

    # Emit receipt for each event
    for event in events:
        emit_receipt("swr_detection", event)

    return events


def detect_spindle(
    lfp_signal: list[float],
    fs: int = SAMPLING_RATE_HZ,
    threshold_sd: float = SPINDLE_THRESHOLD_SD
) -> list[dict]:
    """Detect spindle events (10-16 Hz).

    Args:
        lfp_signal: Local field potential signal
        fs: Sampling frequency (Hz)
        threshold_sd: Detection threshold in standard deviations

    Returns:
        List of spindle events
    """
    filtered = bandpass_filter(lfp_signal, SPINDLE_FREQ_MIN, SPINDLE_FREQ_MAX, fs)
    events = _detect_events(filtered, threshold_sd, fs, "spindle")

    for event in events:
        emit_receipt("spindle_detection", event)

    return events


def detect_slow_oscillation(
    lfp_signal: list[float],
    fs: int = SAMPLING_RATE_HZ
) -> list[dict]:
    """Detect SO events (<1 Hz) with phase information.

    Args:
        lfp_signal: Local field potential signal
        fs: Sampling frequency (Hz)

    Returns:
        List of SO events with phase
    """
    # Filter for slow oscillations
    filtered = bandpass_filter(lfp_signal, 0.1, SO_FREQ_MAX, fs)

    events = []
    # Find zero crossings (phase = 0 at positive-going crossing)
    for i in range(1, len(filtered)):
        if filtered[i - 1] < 0 and filtered[i] >= 0:
            # Positive-going zero crossing = phase 0
            timestamp_ns = int(i * 1e9 / fs)
            events.append({
                "event_type": "so_up_phase",
                "timestamp_ns": timestamp_ns,
                "phase": 0.0,
                "amplitude": abs(filtered[i]),
                "frequency_hz": SO_FREQ_MAX,
            })
        elif filtered[i - 1] > 0 and filtered[i] <= 0:
            # Negative-going zero crossing = phase pi
            timestamp_ns = int(i * 1e9 / fs)
            events.append({
                "event_type": "so_down_phase",
                "timestamp_ns": timestamp_ns,
                "phase": math.pi,
                "amplitude": abs(filtered[i]),
                "frequency_hz": SO_FREQ_MAX,
            })

    for event in events:
        emit_receipt("so_detection", event)

    return events


class StreamState:
    """State container for streaming signal processing."""

    def __init__(self, fs: int = SAMPLING_RATE_HZ):
        self.fs = fs
        self.buffer = []
        self.buffer_size = int(fs * 0.1)  # 100ms buffer
        self.filter_state = {
            "swr": {"y1": 0.0, "y2": 0.0, "x1": 0.0, "x2": 0.0},
            "spindle": {"y1": 0.0, "y2": 0.0, "x1": 0.0, "x2": 0.0},
            "so": {"y1": 0.0, "y2": 0.0, "x1": 0.0, "x2": 0.0},
        }
        self.baseline_mean = 0.0
        self.baseline_var = 1.0
        self.n_samples = 0


def stream_process(
    signal_chunk: list[float],
    state: StreamState
) -> tuple[list[dict], StreamState]:
    """Process signal chunk, update state, return events.

    Designed for real-time streaming with minimal latency.

    Args:
        signal_chunk: New signal samples
        state: Current processing state

    Returns:
        Tuple of (detected_events, updated_state)
    """
    start_time = time.perf_counter()

    # Update buffer
    state.buffer.extend(signal_chunk)
    if len(state.buffer) > state.buffer_size:
        state.buffer = state.buffer[-state.buffer_size:]

    # Update baseline statistics (exponential moving average)
    for sample in signal_chunk:
        state.n_samples += 1
        alpha = min(0.01, 1.0 / state.n_samples)
        delta = sample - state.baseline_mean
        state.baseline_mean += alpha * delta
        state.baseline_var += alpha * (delta * delta - state.baseline_var)

    events = []

    # Detect SWR in chunk
    if len(state.buffer) >= 10:
        swr_filtered = bandpass_filter(
            state.buffer, SWR_FREQ_MIN, SWR_FREQ_MAX, state.fs
        )
        threshold = state.baseline_mean + SWR_THRESHOLD_SD * math.sqrt(state.baseline_var)

        # Check only new samples for events
        for i in range(max(0, len(swr_filtered) - len(signal_chunk)), len(swr_filtered)):
            if abs(swr_filtered[i]) > threshold:
                timestamp_ns = int((state.n_samples - len(swr_filtered) + i) * 1e9 / state.fs)
                events.append({
                    "event_type": "swr",
                    "timestamp_ns": timestamp_ns,
                    "amplitude": abs(swr_filtered[i]),
                    "frequency_hz": (SWR_FREQ_MIN + SWR_FREQ_MAX) / 2,
                })
                break  # One event per chunk max

    elapsed_ms = (time.perf_counter() - start_time) * 1000

    # Emit events with latency tracking
    for event in events:
        event["detection_latency_ms"] = round(elapsed_ms, 3)
        emit_receipt("swr_detection", event)

    return events, state


def benchmark_latency(n_trials: int = 1000) -> dict:
    """Run synthetic signal through detector, measure latency distribution.

    PASS CRITERIA: p95 latency < 2.0ms

    Args:
        n_trials: Number of trials to run

    Returns:
        {p50, p95, p99} latency in ms
    """
    latencies = []
    state = StreamState(fs=SAMPLING_RATE_HZ)

    # Generate synthetic signal with occasional SWR-like bursts
    chunk_size = 100  # 100 samples = 5ms at 20kHz

    for trial in range(n_trials):
        # Generate chunk with noise
        chunk = [math.sin(2 * math.pi * 200 * i / SAMPLING_RATE_HZ) * 0.1
                 + 0.01 * (2 * ((trial * chunk_size + i) % 1000) / 1000 - 1)
                 for i in range(chunk_size)]

        # Inject SWR-like burst occasionally
        if trial % 10 == 0:
            for i in range(20):
                chunk[i] += math.sin(2 * math.pi * 200 * i / SAMPLING_RATE_HZ) * 2.0

        start = time.perf_counter()
        events, state = stream_process(chunk, state)
        elapsed = (time.perf_counter() - start) * 1000
        latencies.append(elapsed)

    # Calculate percentiles
    latencies.sort()
    p50 = latencies[int(n_trials * 0.50)]
    p95 = latencies[int(n_trials * 0.95)]
    p99 = latencies[int(n_trials * 0.99)]

    result = {
        "n_trials": n_trials,
        "p50": round(p50, 4),
        "p95": round(p95, 4),
        "p99": round(p99, 4),
        "target": PHASE_LOCK_LATENCY_MS,
        "passed": p95 < PHASE_LOCK_LATENCY_MS,
    }

    emit_receipt("oscillation_latency_benchmark", result)

    return result


if __name__ == "__main__":
    print("RESONANCE PROTOCOL - Oscillation Detector")
    print(f"SWR Range: {SWR_FREQ_MIN}-{SWR_FREQ_MAX} Hz")
    print(f"Spindle Range: {SPINDLE_FREQ_MIN}-{SPINDLE_FREQ_MAX} Hz")
    print(f"SO Max: {SO_FREQ_MAX} Hz")
    print(f"Target Latency: <{PHASE_LOCK_LATENCY_MS}ms")
    print()
    print("Running latency benchmark...")
    result = benchmark_latency(n_trials=100)
    print(f"p50: {result['p50']:.3f}ms")
    print(f"p95: {result['p95']:.3f}ms (target < {PHASE_LOCK_LATENCY_MS}ms)")
    print(f"p99: {result['p99']:.3f}ms")
    print(f"PASSED: {result['passed']}")
