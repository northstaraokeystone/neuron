"""
RESONANCE PROTOCOL - Core Neural Primitives

Extends CLAUDEME foundation with neural-specific primitives.
CLAUDEME section 8 compliant: dual-hash (SHA256:BLAKE3) required.

Functions:
    neural_hash: Compute dual-hash of spike timing within temporal window
    validate_channel_quality: Check channel meets N1 specs
    enforce_thermal_limit: Raise StopRule if thermal limit exceeded
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

try:
    import blake3
    HAS_BLAKE3 = True
except ImportError:
    HAS_BLAKE3 = False

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

# ============================================
# NEURAL SIGNAL PARAMETERS
# ============================================
HDC_DIMENSION = 10000           # Dimensionality for hyperdimensional vectors
N1_CHANNELS = 1024              # Neuralink N1 electrode count
N2_CHANNELS = 3072              # Future N2 target

# ============================================
# OSCILLATION FREQUENCY BANDS
# ============================================
SWR_FREQ_MIN = 150              # Hz - Sharp-Wave Ripple lower bound
SWR_FREQ_MAX = 250              # Hz - Sharp-Wave Ripple upper bound
SPINDLE_FREQ_MIN = 10           # Hz - Thalamic spindle lower bound
SPINDLE_FREQ_MAX = 16           # Hz - Thalamic spindle upper bound
SO_FREQ_MAX = 1.0               # Hz - Slow oscillation maximum (delta band)

# ============================================
# TIMING CONSTRAINTS
# ============================================
PHASE_LOCK_LATENCY_MS = 2.0     # Maximum allowed loop latency for phase-locking
STIMULATION_WINDOW_MS = 1.0     # Time window for stimulation trigger after prediction

# ============================================
# SAFETY LIMITS
# ============================================
STIM_CHARGE_DENSITY_LIMIT = 30.0    # uC/cm^2 - Shannon limit with safety margin
THERMAL_LIMIT_DELTA_C = 1.0         # C - Maximum allowable temperature rise
IMPEDANCE_THRESHOLD_KOHM = 100.0    # kOhm - Channel quality threshold
SNR_THRESHOLD_DB = 6.0              # dB - Minimum signal-to-noise ratio

# ============================================
# HDC PARAMETERS
# ============================================
CHANNEL_DROPOUT_TOLERANCE = 0.30    # Design target: 30% channel loss tolerance
MIN_CLASSIFICATION_ACCURACY = 0.85  # Accuracy floor with dropout
ORTHOGONALITY_THRESHOLD = 0.1       # Maximum mean cosine similarity for random vectors

# ============================================
# FEDERATED LEARNING
# ============================================
FL_UPDATE_INTERVAL_SEC = 86400      # Daily aggregation cycle (24 hours)
FL_MIN_PARTICIPANTS = 3             # Minimum users for aggregation (privacy)
PRIVACY_BUDGET_EPSILON = 1.0        # Differential privacy parameter

# ============================================
# DETECTION THRESHOLDS
# ============================================
SWR_THRESHOLD_SD = 3.0              # Standard deviations above baseline for SWR detection
SPINDLE_THRESHOLD_SD = 2.5          # Standard deviations for spindle detection
RETRACTION_WARNING_MM = 0.5         # Retraction estimate threshold for alert

# ============================================
# SAMPLING
# ============================================
SAMPLING_RATE_HZ = 20000            # Neuralink N1 sampling frequency
WINDOW_SIZE_MS = 50                 # Processing window size


class StopRule(Exception):
    """CLAUDEME section 8: Exception for stoprule failures."""

    def __init__(self, rule_name: str, message: str, context: dict | None = None):
        self.rule_name = rule_name
        self.context = context or {}
        super().__init__(f"STOPRULE[{rule_name}]: {message}")


def _get_receipts_path() -> Path:
    """Get path for receipts ledger."""
    base = Path(os.environ.get("RESONANCE_BASE", Path(__file__).parent.parent))
    return base / "receipts.jsonl"


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


def emit_receipt(receipt_type: str, data: dict) -> dict:
    """Emit a receipt to the receipts ledger per CLAUDEME section 4.

    SCHEMA: {type, ts, hash, **data}
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


def neural_hash(spike_train: list | "np.ndarray", window_ms: float = 50.0) -> str:
    """Compute dual-hash of spike timing within temporal window.

    Handles variable-length sequences. Deterministic for same spike train.

    Args:
        spike_train: Spike times in seconds (list or numpy array)
        window_ms: Temporal window in milliseconds for binning

    Returns:
        Dual-hash string (SHA256:BLAKE3)
    """
    if HAS_NUMPY and hasattr(spike_train, 'tolist'):
        spike_train = spike_train.tolist()

    # Normalize to list
    if not isinstance(spike_train, list):
        spike_train = list(spike_train)

    # Bin spikes by window for deterministic hashing
    window_sec = window_ms / 1000.0
    if not spike_train:
        binned = []
    else:
        min_t = min(spike_train) if spike_train else 0
        binned = [int((t - min_t) / window_sec) for t in spike_train]

    # Create canonical representation
    canonical = json.dumps(sorted(binned), separators=(',', ':'))
    return dual_hash(canonical)


def validate_channel_quality(impedance_kohm: float, snr_db: float) -> bool:
    """Return True if channel meets N1 specs.

    Thresholds:
        - impedance < 100 kOhm
        - SNR > 6 dB

    Args:
        impedance_kohm: Electrode impedance in kiloohms
        snr_db: Signal-to-noise ratio in decibels

    Returns:
        True if channel is good quality
    """
    return impedance_kohm < IMPEDANCE_THRESHOLD_KOHM and snr_db > SNR_THRESHOLD_DB


def enforce_thermal_limit(
    current_temp_c: float,
    baseline_temp_c: float,
    context: dict | None = None
) -> None:
    """Raise StopRule if temperature rise exceeds thermal limit.

    SAFETY CRITICAL: Temperature rise of 1C can alter neuronal firing.

    Args:
        current_temp_c: Current temperature in Celsius
        baseline_temp_c: Baseline temperature in Celsius
        context: Optional context for receipt

    Raises:
        StopRule: If delta T > THERMAL_LIMIT_DELTA_C (1.0 C)
    """
    delta_t = current_temp_c - baseline_temp_c

    if delta_t > THERMAL_LIMIT_DELTA_C:
        # Emit thermal violation receipt before raising
        emit_receipt(
            "thermal_violation",
            {
                "current_temp_c": current_temp_c,
                "baseline_temp_c": baseline_temp_c,
                "delta_t": delta_t,
                "limit_c": THERMAL_LIMIT_DELTA_C,
                "context": context or {},
            }
        )
        raise StopRule(
            "thermal_limit_exceeded",
            f"Temperature rise {delta_t:.2f}C exceeds limit {THERMAL_LIMIT_DELTA_C}C",
            {
                "delta_t": delta_t,
                "limit": THERMAL_LIMIT_DELTA_C,
                "context": context or {},
            }
        )


def merkle(items: list) -> str:
    """Compute merkle root hash of items per CLAUDEME section 8.

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


if __name__ == "__main__":
    print("RESONANCE PROTOCOL - Core Neural Primitives")
    print(f"HDC Dimension: {HDC_DIMENSION}")
    print(f"N1 Channels: {N1_CHANNELS}")
    print(f"Phase Lock Latency: {PHASE_LOCK_LATENCY_MS}ms")
    print(f"BLAKE3 available: {HAS_BLAKE3}")
    print(f"NumPy available: {HAS_NUMPY}")
