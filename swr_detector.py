"""
NEURON-RESONANCE v5.0: Sharp-Wave Ripple Detection
Real-time SWR detection from N1 LFP signals. Triggers consolidation sync.

Sharp-Wave Ripples (150-250 Hz oscillations during quiet wakefulness) are when
the brain consolidates memories. They replay experiences 10-20x faster.
NEURON's consolidation syncs to these biological rhythms.

The exocortex and neocortex resonate.
"""

import math
import time

from neuron import StopRule, emit_receipt

# ============================================
# SHARP-WAVE RIPPLE PARAMETERS
# ============================================
SWR_FREQUENCY_MIN = 150  # Hz
SWR_FREQUENCY_MAX = 250  # Hz
SWR_FREQUENCY_RANGE = [SWR_FREQUENCY_MIN, SWR_FREQUENCY_MAX]
SWR_CONFIDENCE_THRESHOLD = 0.8  # Detection confidence
IDLE_THRESHOLD_MS = 5000  # Fallback trigger (5 seconds)

# Detection parameters
BURST_THRESHOLD_STD = 3.0  # Standard deviations for burst detection
MIN_BURST_DURATION_MS = 25  # Minimum burst length
MIN_BURSTS_FOR_SWR = 3  # Minimum bursts to classify as SWR


class SWRStopRule(StopRule):
    """StopRule specific to SWR detection failures."""

    def __init__(self, rule_name: str, message: str, context: dict | None = None):
        super().__init__(f"swr_{rule_name}", message, context)


def bandpass_filter(
    lfp: list[float], low_hz: int, high_hz: int, sample_rate: int
) -> list[float]:
    """Apply Butterworth bandpass filter in SWR range.

    Simplified frequency-domain filtering for SWR band extraction.

    Args:
        lfp: Local field potential signal
        low_hz: Low cutoff frequency (Hz)
        high_hz: High cutoff frequency (Hz)
        sample_rate: Sampling rate (Hz)

    Returns:
        Bandpass filtered signal
    """
    if not lfp or len(lfp) < 4:
        return list(lfp)

    n = len(lfp)

    # Simple IIR bandpass approximation
    # Compute filter coefficients for Butterworth-like response
    low_norm = low_hz / (sample_rate / 2)
    high_norm = high_hz / (sample_rate / 2)

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
    for x in lfp:
        y = b0 * x + b0 * x1 - a1 * y1 - a2 * y2
        y2 = y1
        y1 = y
        x2 = x1
        x1 = x
        filtered.append(y)

    return filtered


def detect_burst(
    filtered: list[float], threshold_std: float = BURST_THRESHOLD_STD
) -> list[dict]:
    """Detect bursts exceeding threshold in filtered signal.

    Args:
        filtered: Bandpass filtered signal
        threshold_std: Threshold in standard deviations

    Returns:
        List of {start_idx, end_idx, amplitude} for each burst
    """
    if not filtered or len(filtered) < 10:
        return []

    # Compute mean and std
    mean = sum(filtered) / len(filtered)
    variance = sum((v - mean) ** 2 for v in filtered) / len(filtered)
    std = math.sqrt(variance) if variance > 0 else 1.0

    threshold = mean + threshold_std * std

    bursts = []
    in_burst = False
    start_idx = 0
    max_amp = 0.0

    for i, v in enumerate(filtered):
        if v > threshold:
            if not in_burst:
                in_burst = True
                start_idx = i
                max_amp = v
            else:
                max_amp = max(max_amp, v)
        else:
            if in_burst:
                bursts.append(
                    {
                        "start_idx": start_idx,
                        "end_idx": i,
                        "amplitude": max_amp,
                        "duration_samples": i - start_idx,
                    }
                )
                in_burst = False

    # Handle burst at end of signal
    if in_burst:
        bursts.append(
            {
                "start_idx": start_idx,
                "end_idx": len(filtered) - 1,
                "amplitude": max_amp,
                "duration_samples": len(filtered) - 1 - start_idx,
            }
        )

    return bursts


def compute_swr_confidence(
    bursts: list[dict],
    min_bursts: int = MIN_BURSTS_FOR_SWR,
    min_duration_ms: float = MIN_BURST_DURATION_MS,
    sample_rate: int = 1000,
) -> float:
    """Compute confidence 0-1 that SWR is occurring.

    Args:
        bursts: List of detected bursts
        min_bursts: Minimum burst count for SWR
        min_duration_ms: Minimum burst duration in ms
        sample_rate: Sampling rate for duration calculation

    Returns:
        Confidence score 0-1
    """
    if not bursts:
        return 0.0

    # Filter bursts by minimum duration
    min_samples = int(min_duration_ms * sample_rate / 1000)
    valid_bursts = [b for b in bursts if b["duration_samples"] >= min_samples]

    if len(valid_bursts) < min_bursts:
        return 0.0

    # Confidence based on burst count and amplitude consistency
    count_factor = min(1.0, len(valid_bursts) / (min_bursts * 2))

    # Amplitude consistency (lower variance = higher confidence)
    if len(valid_bursts) > 1:
        amps = [b["amplitude"] for b in valid_bursts]
        amp_mean = sum(amps) / len(amps)
        amp_var = sum((a - amp_mean) ** 2 for a in amps) / len(amps)
        amp_std = math.sqrt(amp_var) if amp_var > 0 else 0
        cv = amp_std / amp_mean if amp_mean > 0 else 1.0
        consistency_factor = max(0, 1.0 - cv)
    else:
        consistency_factor = 0.5

    # Combined confidence
    confidence = (count_factor * 0.6) + (consistency_factor * 0.4)

    return min(1.0, confidence)


def detect_biological_swr(
    lfp: list[float], config: dict, sample_rate: int = 1000
) -> dict | None:
    """Full SWR detection pipeline.

    Args:
        lfp: Local field potential signal
        config: Resonance configuration dict
        sample_rate: Sampling rate in Hz

    Returns:
        Dict with {detected, frequency_hz, burst_count, confidence} or None if not detected

    Raises:
        SWRStopRule: If frequency outside physiological range [100, 300] Hz
    """
    start_time = time.time()

    swr_freq = config.get("swr_frequency_hz", SWR_FREQUENCY_RANGE)
    confidence_threshold = config.get(
        "swr_confidence_threshold", SWR_CONFIDENCE_THRESHOLD
    )

    low_hz = swr_freq[0]
    high_hz = swr_freq[1]

    # Validate physiological range
    if low_hz < 100 or high_hz > 300:
        raise SWRStopRule(
            "frequency_out_of_range",
            f"SWR frequency [{low_hz}, {high_hz}] outside physiological range [100, 300] Hz",
            {"low_hz": low_hz, "high_hz": high_hz},
        )

    # Step 1: Bandpass filter in SWR range
    filtered = bandpass_filter(lfp, low_hz, high_hz, sample_rate)

    # Step 2: Detect bursts
    bursts = detect_burst(filtered)

    # Step 3: Compute confidence
    confidence = compute_swr_confidence(bursts, sample_rate=sample_rate)

    # Calculate detection latency
    detection_time_ms = (time.time() - start_time) * 1000

    detected = confidence >= confidence_threshold

    # Estimate dominant frequency from burst spacing
    if len(bursts) >= 2:
        intervals = []
        for i in range(1, len(bursts)):
            interval_samples = bursts[i]["start_idx"] - bursts[i - 1]["start_idx"]
            if interval_samples > 0:
                intervals.append(interval_samples)
        if intervals:
            mean_interval = sum(intervals) / len(intervals)
            frequency_hz = (
                sample_rate / mean_interval
                if mean_interval > 0
                else (low_hz + high_hz) / 2
            )
        else:
            frequency_hz = (low_hz + high_hz) / 2
    else:
        frequency_hz = (low_hz + high_hz) / 2

    result = {
        "detected": detected,
        "frequency_hz": round(frequency_hz, 1),
        "burst_count": len(bursts),
        "confidence": round(confidence, 4),
        "detection_method": "bandpass_burst",
        "detection_latency_ms": round(detection_time_ms, 2),
    }

    # Emit receipt
    receipt = emit_receipt(
        "swr_detect",
        {
            "detected": detected,
            "frequency_hz": result["frequency_hz"],
            "burst_count": result["burst_count"],
            "confidence": result["confidence"],
            "detection_method": result["detection_method"],
        },
    )
    result["_receipt"] = receipt

    if not detected:
        return None

    return result


def idle_threshold_fallback(
    last_activity_ms: int, threshold_ms: int = IDLE_THRESHOLD_MS
) -> bool:
    """Fallback trigger if no SWR detected but idle time exceeded.

    For graceful degradation when biological signals unavailable.

    Args:
        last_activity_ms: Time since last activity in milliseconds
        threshold_ms: Idle threshold for fallback trigger

    Returns:
        True if fallback should trigger
    """
    return last_activity_ms >= threshold_ms


def generate_simulated_lfp(
    duration_ms: int, sample_rate: int = 1000, swr_present: bool = True, seed: int = 42
) -> list[float]:
    """Generate simulated LFP signal for testing.

    Args:
        duration_ms: Signal duration in milliseconds
        sample_rate: Sampling rate in Hz
        swr_present: If True, inject SWR-like oscillations
        seed: Random seed

    Returns:
        Simulated LFP signal
    """
    n_samples = int(duration_ms * sample_rate / 1000)

    # Simple PRNG
    a, c, m = 1664525, 1013904223, 2**32
    state = seed

    lfp = []
    for i in range(n_samples):
        state = (a * state + c) % m

        # Background noise
        noise = ((state / m) * 2 - 1) * 0.3

        # Add SWR-like oscillations if requested
        if swr_present:
            t = i / sample_rate
            # 200 Hz ripple oscillation (SWR characteristic)
            ripple = 0.5 * math.sin(2 * math.pi * 200 * t)
            # Modulate with slower envelope
            envelope = 0.5 * (1 + math.sin(2 * math.pi * 8 * t))
            signal = noise + ripple * envelope
        else:
            signal = noise

        lfp.append(signal)

    return lfp


if __name__ == "__main__":
    print("NEURON-RESONANCE v5.0 - SWR Detection")
    print(f"SWR Frequency Range: {SWR_FREQUENCY_RANGE} Hz")
    print(f"Confidence Threshold: {SWR_CONFIDENCE_THRESHOLD}")
    print(f"Idle Fallback: {IDLE_THRESHOLD_MS}ms")
    print()
    print("Sharp-Wave Ripples synchronize biological and digital consolidation.")
