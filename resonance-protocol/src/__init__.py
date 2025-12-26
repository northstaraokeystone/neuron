"""
RESONANCE PROTOCOL v2.0 - Neurophysiology Research Infrastructure

Extends QED/ProofPack architecture to neurophysiology domain.
Receipts ARE neural state. HDC IS the compression. FL IS the sovereignty guarantee.

KILLED:
    - Cloud-based neural processing (latency physics impossible)
    - Global data aggregation (privacy violation)
    - Fixed CNN architecture (retraction failure mode)
    - Suprathreshold stimulation (too crude)

THE INSIGHT:
    The brain is an entropy-fighting system. Thread retraction is
    mechanically inevitable. HDC's holographic robustness is not
    optimization - it's physical necessity.
"""

from .core import (
    neural_hash,
    validate_channel_quality,
    enforce_thermal_limit,
    StopRule,
    dual_hash,
    emit_receipt,
    # Constants
    HDC_DIMENSION,
    N1_CHANNELS,
    N2_CHANNELS,
    SWR_FREQ_MIN,
    SWR_FREQ_MAX,
    SPINDLE_FREQ_MIN,
    SPINDLE_FREQ_MAX,
    SO_FREQ_MAX,
    PHASE_LOCK_LATENCY_MS,
    STIM_CHARGE_DENSITY_LIMIT,
    THERMAL_LIMIT_DELTA_C,
    FL_UPDATE_INTERVAL_SEC,
    CHANNEL_DROPOUT_TOLERANCE,
    SAMPLING_RATE_HZ,
)

__version__ = "2.0.0"
__all__ = [
    "neural_hash",
    "validate_channel_quality",
    "enforce_thermal_limit",
    "StopRule",
    "dual_hash",
    "emit_receipt",
]
