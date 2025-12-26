"""
RESONANCE PROTOCOL - Thread Quality Monitor

Track electrode quality degradation and update HDC dropout mask.
Thread retraction is inevitable due to brain pulsation.

THE PHYSICS:
    The brain is an entropy-fighting system. Thread retraction is
    mechanically inevitable. This module detects degradation and
    adapts the HDC encoder's dropout mask accordingly.

Functions:
    measure_impedance: Get electrode impedance (simulated for v1)
    measure_snr: Compute signal-to-noise ratio
    estimate_retraction: Estimate thread retraction from impedance trend
    update_dropout_mask: Generate binary mask for HDC encoder
    emit_channel_status: Emit channel quality receipt
"""

from __future__ import annotations

import math
from typing import Callable

try:
    from .core import (
        N1_CHANNELS,
        IMPEDANCE_THRESHOLD_KOHM,
        SNR_THRESHOLD_DB,
        RETRACTION_WARNING_MM,
        emit_receipt,
        validate_channel_quality,
    )
except ImportError:
    from core import (
        N1_CHANNELS,
        IMPEDANCE_THRESHOLD_KOHM,
        SNR_THRESHOLD_DB,
        RETRACTION_WARNING_MM,
        emit_receipt,
        validate_channel_quality,
    )


def measure_impedance(
    channel_id: int,
    simulated: bool = True,
    impedance_map: dict[int, float] | None = None
) -> float:
    """Return impedance in kOhm for channel.

    Simulated for v1; hardware interface in production.

    Args:
        channel_id: Channel identifier (0 to N1_CHANNELS-1)
        simulated: If True, return simulated values
        impedance_map: Optional map of channel_id -> impedance

    Returns:
        Impedance in kiloohms
    """
    if impedance_map is not None and channel_id in impedance_map:
        return impedance_map[channel_id]

    if simulated:
        # Simulate impedance with some variation
        # Most channels good (30-80 kOhm), some degraded (>100 kOhm)
        import random
        random.seed(channel_id)  # Deterministic per channel

        # 90% good channels, 10% degraded
        if random.random() < 0.9:
            return random.uniform(30, 80)
        else:
            return random.uniform(100, 200)

    raise NotImplementedError("Hardware impedance measurement not implemented")


def measure_snr(
    channel_id: int,
    signal: list[float],
    noise: list[float]
) -> float:
    """Compute SNR in dB: 20*log10(signal_rms / noise_rms).

    Args:
        channel_id: Channel identifier
        signal: Signal samples
        noise: Noise samples

    Returns:
        SNR in decibels
    """
    if not signal or not noise:
        return 0.0

    # Calculate RMS
    signal_rms = math.sqrt(sum(s * s for s in signal) / len(signal))
    noise_rms = math.sqrt(sum(n * n for n in noise) / len(noise))

    if noise_rms == 0:
        return float('inf') if signal_rms > 0 else 0.0

    # SNR in dB
    snr_db = 20 * math.log10(signal_rms / noise_rms)

    return snr_db


def estimate_retraction(
    impedance_trend: list[tuple[float, float]]
) -> float:
    """Fit linear trend to impedance over time. Return estimated retraction in mm.

    Heuristic: impedance increase of 50 kOhm ~ 0.1mm retraction

    Args:
        impedance_trend: List of (timestamp_hours, impedance_kohm) tuples

    Returns:
        Estimated retraction in mm
    """
    if len(impedance_trend) < 2:
        return 0.0

    # Extract values
    times = [t for t, _ in impedance_trend]
    impedances = [z for _, z in impedance_trend]

    # Simple linear regression
    n = len(times)
    sum_t = sum(times)
    sum_z = sum(impedances)
    sum_tz = sum(t * z for t, z in zip(times, impedances))
    sum_tt = sum(t * t for t in times)

    # Slope = (n*sum_tz - sum_t*sum_z) / (n*sum_tt - sum_t^2)
    denom = n * sum_tt - sum_t * sum_t
    if denom == 0:
        return 0.0

    slope = (n * sum_tz - sum_t * sum_z) / denom

    # Convert slope (kOhm/hour) to retraction (mm)
    # Heuristic: 50 kOhm increase per 0.1mm retraction per day
    # slope is kOhm/hour, so multiply by 24 for daily rate
    daily_impedance_change = slope * 24

    # 50 kOhm/day = 0.1mm/day retraction
    retraction_mm = daily_impedance_change * 0.1 / 50

    return max(0, retraction_mm)


def update_dropout_mask(
    channel_qualities: dict[int, dict]
) -> list[int]:
    """Generate binary mask: 1 if channel good, 0 if degraded.

    Threshold: impedance > 100 kOhm OR SNR < 6 dB -> 0

    Args:
        channel_qualities: Dict {channel_id: {impedance_kohm, snr_db}}

    Returns:
        Binary dropout mask [N1_CHANNELS]
    """
    mask = [1] * N1_CHANNELS

    for channel_id, quality in channel_qualities.items():
        if channel_id >= N1_CHANNELS:
            continue

        impedance = quality.get("impedance_kohm", 0)
        snr = quality.get("snr_db", float('inf'))

        # Check quality thresholds
        if not validate_channel_quality(impedance, snr):
            mask[channel_id] = 0

    return mask


def emit_channel_status(
    channel_id: int,
    quality_metrics: dict
) -> dict:
    """Emit channel_quality_receipt for monitoring.

    Args:
        channel_id: Channel identifier
        quality_metrics: Dict with impedance_kohm, snr_db, retraction_mm, status

    Returns:
        Receipt dict
    """
    return emit_receipt(
        "channel_quality",
        {
            "channel_id": channel_id,
            "snr_db": round(quality_metrics.get("snr_db", 0), 2),
            "impedance_kohm": round(quality_metrics.get("impedance_kohm", 0), 2),
            "retraction_estimate_mm": round(quality_metrics.get("retraction_mm", 0), 4),
            "status": quality_metrics.get("status", "unknown"),
        }
    )


class ThreadMonitor:
    """Monitors electrode thread quality across all channels.

    Updates at 1 Hz during operation.
    Dropout mask updated every 60 seconds.
    """

    def __init__(self, n_channels: int = N1_CHANNELS):
        self.n_channels = n_channels
        self.channel_states: dict[int, dict] = {}
        self.impedance_history: dict[int, list[tuple[float, float]]] = {}
        self.dropout_mask = [1] * n_channels
        self.update_count = 0

    def update_channel(
        self,
        channel_id: int,
        impedance_kohm: float,
        snr_db: float,
        timestamp_hours: float = 0.0
    ) -> dict:
        """Update channel quality state.

        Args:
            channel_id: Channel to update
            impedance_kohm: Current impedance
            snr_db: Current SNR
            timestamp_hours: Time for trend tracking

        Returns:
            Channel status dict
        """
        # Update impedance history
        if channel_id not in self.impedance_history:
            self.impedance_history[channel_id] = []

        self.impedance_history[channel_id].append((timestamp_hours, impedance_kohm))

        # Keep last 168 hours (1 week)
        if len(self.impedance_history[channel_id]) > 168:
            self.impedance_history[channel_id] = self.impedance_history[channel_id][-168:]

        # Estimate retraction
        retraction_mm = estimate_retraction(self.impedance_history[channel_id])

        # Determine status
        is_good = validate_channel_quality(impedance_kohm, snr_db)
        needs_review = retraction_mm > RETRACTION_WARNING_MM

        if needs_review:
            status = "review_required"
        elif is_good:
            status = "good"
        else:
            status = "degraded"

        # Store state
        state = {
            "impedance_kohm": impedance_kohm,
            "snr_db": snr_db,
            "retraction_mm": retraction_mm,
            "status": status,
        }
        self.channel_states[channel_id] = state

        # Emit receipt
        emit_channel_status(channel_id, state)

        return state

    def get_dropout_mask(self) -> list[int]:
        """Get current dropout mask for HDC encoder."""
        return update_dropout_mask(self.channel_states)

    def update_all_channels(self, timestamp_hours: float = 0.0) -> dict:
        """Update all channels (simulated data for testing).

        Args:
            timestamp_hours: Current timestamp in hours

        Returns:
            Summary statistics
        """
        good_count = 0
        degraded_count = 0
        review_count = 0

        for ch in range(self.n_channels):
            impedance = measure_impedance(ch, simulated=True)

            # Simulate SNR based on impedance
            if impedance < IMPEDANCE_THRESHOLD_KOHM:
                snr = 10 + (IMPEDANCE_THRESHOLD_KOHM - impedance) / 10
            else:
                snr = max(0, 6 - (impedance - IMPEDANCE_THRESHOLD_KOHM) / 50)

            state = self.update_channel(ch, impedance, snr, timestamp_hours)

            if state["status"] == "good":
                good_count += 1
            elif state["status"] == "degraded":
                degraded_count += 1
            else:
                review_count += 1

        self.update_count += 1
        self.dropout_mask = self.get_dropout_mask()

        summary = {
            "total_channels": self.n_channels,
            "good": good_count,
            "degraded": degraded_count,
            "review_required": review_count,
            "dropout_rate": 1.0 - (sum(self.dropout_mask) / self.n_channels),
            "update_count": self.update_count,
        }

        emit_receipt("thread_monitor_update", summary)

        return summary


if __name__ == "__main__":
    print("RESONANCE PROTOCOL - Thread Quality Monitor")
    print(f"N1 Channels: {N1_CHANNELS}")
    print(f"Impedance Threshold: {IMPEDANCE_THRESHOLD_KOHM} kOhm")
    print(f"SNR Threshold: {SNR_THRESHOLD_DB} dB")
    print(f"Retraction Warning: {RETRACTION_WARNING_MM} mm")
    print()
    print("Running channel scan...")
    monitor = ThreadMonitor(n_channels=100)  # Reduced for demo
    summary = monitor.update_all_channels(timestamp_hours=0)
    print(f"Good: {summary['good']}")
    print(f"Degraded: {summary['degraded']}")
    print(f"Review Required: {summary['review_required']}")
    print(f"Dropout Rate: {summary['dropout_rate']:.2%}")
