"""
NEURON-RESONANCE v5.0: Bio-Digital SDM Bridge
Core bio-digital translation layer. Maps spike rasters to SDM query vectors.

PARADIGM INVERSION:
  Text translation → Native SDM resonance
  Spike trains ARE native SDM queries
  Brain and ledger share the SAME substrate

THE INSIGHT:
  Kanerva's SDM (1988) and Transformer attention (2017) are the same mathematics.
  Hippocampal place cells fire sparse patterns that ARE hyperdimensional bindings.
  NEURON's persistent hashed KV projections ARE hard SDM locations.
  This is not translation. This is resonance.
"""

import json
import math
import os
from pathlib import Path
from typing import Optional

from neuron import StopRule, dual_hash, emit_receipt

# ============================================
# SDM CONFIGURATION CONSTANTS
# ============================================
SDM_DIM = 16384  # Match Grok latent dimension
N1_CHANNELS_MIN = 1024  # Neuralink N1 minimum
N1_CHANNELS_MAX = 4096  # Neuralink N1 maximum
WINDOW_MS = 100  # Population vector integration window

# ============================================
# BIOLOGICAL CONSTRAINTS
# ============================================
SPARSITY_TARGET = 0.01  # ~1% active neurons (biological)
SPARSITY_MAX = 0.05  # Hard limit (5%)


class ResonanceStopRule(StopRule):
    """StopRule specific to resonance bridge failures."""

    def __init__(self, rule_name: str, message: str, context: dict | None = None):
        super().__init__(f"resonance_{rule_name}", message, context)


def _get_resonance_spec_path() -> Path:
    """Get path to resonance_spec.json."""
    base = Path(os.environ.get("NEURON_BASE", Path(__file__).parent))
    return base / "data" / "resonance_spec.json"


def load_resonance_spec(path: str | None = None) -> dict:
    """Load and validate resonance_spec.json. Compute dual-hash.

    Args:
        path: Optional path to spec file. Uses default if None.

    Returns:
        Validated spec dict with _spec_hash and _ingest_receipt.

    Raises:
        ResonanceStopRule: If spec is invalid or constraints violated.
    """
    if path is None:
        path = str(_get_resonance_spec_path())

    try:
        with open(path, "r") as f:
            spec = json.load(f)
    except FileNotFoundError:
        raise ResonanceStopRule(
            "spec_not_found",
            f"resonance_spec.json not found at {path}",
            {"path": path},
        )
    except json.JSONDecodeError as e:
        raise ResonanceStopRule(
            "spec_malformed",
            f"Invalid JSON in resonance_spec.json: {e}",
            {"path": path},
        )

    # Validate required fields
    required = [
        "sdm_dim",
        "sparsity_target",
        "sparsity_max",
        "swr_frequency_hz",
        "stim_intensity_max",
    ]
    for field in required:
        if field not in spec:
            raise ResonanceStopRule(
                "spec_missing_field",
                f"Missing required field: {field}",
                {"field": field},
            )

    # Validate constraints
    if spec["sparsity_target"] > spec["sparsity_max"]:
        raise ResonanceStopRule(
            "spec_constraint_violation",
            f"sparsity_target ({spec['sparsity_target']}) > sparsity_max ({spec['sparsity_max']})",
            {
                "sparsity_target": spec["sparsity_target"],
                "sparsity_max": spec["sparsity_max"],
            },
        )

    if spec["stim_intensity_max"] > 1.0:
        raise ResonanceStopRule(
            "spec_safety_violation",
            f"stim_intensity_max ({spec['stim_intensity_max']}) > 1.0 (safety limit)",
            {"stim_intensity_max": spec["stim_intensity_max"]},
        )

    swr_freq = spec["swr_frequency_hz"]
    if not (100 <= swr_freq[0] and swr_freq[1] <= 300):
        raise ResonanceStopRule(
            "spec_physiological_violation",
            f"swr_frequency_hz {swr_freq} outside physiological range [100, 300]",
            {"swr_frequency_hz": swr_freq},
        )

    # Compute dual-hash
    spec_json = json.dumps(
        {k: v for k, v in spec.items() if not k.startswith("_")}, sort_keys=True
    )
    spec["_spec_hash"] = dual_hash(spec_json)

    # Emit receipt
    receipt = emit_receipt(
        "spec_ingest",
        {
            "spec_hash": spec["_spec_hash"],
            "sdm_dim": spec["sdm_dim"],
            "sparsity_target": spec["sparsity_target"],
            "swr_frequency": spec["swr_frequency_hz"],
        },
    )
    spec["_ingest_receipt"] = receipt

    return spec


def generate_projection_matrix(
    n_channels: int, dim: int, seed: int
) -> list[list[float]]:
    """Generate fixed random projection matrix per implant.

    Uses deterministic PRNG from seed for reproducibility.
    Matrix projects n_channels firing rates to dim-dimensional SDM space.

    Args:
        n_channels: Number of Neuralink N1 channels (1024-4096)
        dim: SDM dimension (typically 16384)
        seed: Random seed for deterministic generation

    Returns:
        Projection matrix as nested list [dim][n_channels]
    """
    # Deterministic random number generation
    # Using linear congruential generator for simplicity
    # Constants from Numerical Recipes
    a, c, m = 1664525, 1013904223, 2**32

    state = seed
    matrix = []

    # Generate orthogonal-ish random projections
    # Scale by 1/sqrt(n_channels) for unit variance output
    scale = 1.0 / math.sqrt(n_channels)

    for _ in range(dim):
        row = []
        for _ in range(n_channels):
            state = (a * state + c) % m
            # Convert to float in [-1, 1]
            val = (state / m) * 2 - 1
            row.append(val * scale)
        matrix.append(row)

    return matrix


def compute_firing_rates(spike_raster: list[list[int]], window_ms: int) -> list[float]:
    """Compute population firing rates over window.

    Args:
        spike_raster: Binary spike raster [channels][samples]
        window_ms: Integration window in milliseconds

    Returns:
        Firing rates per channel [channels]
    """
    if not spike_raster or not spike_raster[0]:
        return []

    n_channels = len(spike_raster)
    n_samples = len(spike_raster[0])

    # Sum spikes per channel and normalize by window
    rates = []
    for ch in range(n_channels):
        spike_count = sum(spike_raster[ch])
        # Rate in Hz (assuming 1kHz sampling)
        rate = (spike_count / n_samples) * 1000.0
        rates.append(rate)

    return rates


def enforce_sparsity(vector: list[float], target: float) -> list[float]:
    """Threshold vector to target sparsity.

    Keeps only the top target fraction of entries by absolute value.
    All other entries are zeroed.

    Args:
        vector: Input vector
        target: Target sparsity (fraction of non-zero entries)

    Returns:
        Sparse vector with approximately target fraction non-zero
    """
    if not vector or target <= 0:
        return [0.0] * len(vector)

    if target >= 1.0:
        return list(vector)

    n = len(vector)
    k = max(1, int(n * target))  # Number of entries to keep

    # Get indices sorted by absolute value (descending)
    indexed = [(i, abs(v)) for i, v in enumerate(vector)]
    indexed.sort(key=lambda x: x[1], reverse=True)

    # Keep top-k indices
    keep_indices = set(idx for idx, _ in indexed[:k])

    # Build sparse vector
    sparse = []
    for i, v in enumerate(vector):
        if i in keep_indices:
            sparse.append(v)
        else:
            sparse.append(0.0)

    return sparse


def normalize_to_sphere(vector: list[float]) -> list[float]:
    """L2 normalize to unit sphere.

    Args:
        vector: Input vector

    Returns:
        Unit-normalized vector (or zero vector if input is zero)
    """
    if not vector:
        return []

    norm = math.sqrt(sum(v * v for v in vector))

    if norm == 0:
        return [0.0] * len(vector)

    return [v / norm for v in vector]


def bind_context(bio_vector: list[float], context_vector: list[float]) -> list[float]:
    """Element-wise multiplication for task binding.

    Re-normalizes after binding to maintain unit sphere.

    Args:
        bio_vector: Biological SDM vector
        context_vector: Task context vector

    Returns:
        Bound vector (bio * context), re-normalized
    """
    if len(bio_vector) != len(context_vector):
        raise ResonanceStopRule(
            "context_dimension_mismatch",
            f"bio_vector dim {len(bio_vector)} != context_vector dim {len(context_vector)}",
            {"bio_dim": len(bio_vector), "context_dim": len(context_vector)},
        )

    # Element-wise multiplication
    bound = [b * c for b, c in zip(bio_vector, context_vector)]

    # Re-normalize to unit sphere
    return normalize_to_sphere(bound)


def biological_to_digital_sdm(
    spike_raster: list[list[int]],
    config: dict,
    context: Optional[list[float]] = None,
    seed: int = 42,
) -> dict:
    """Full bio-to-digital SDM pipeline.

    Pipeline:
        rates → project → sparsify → normalize → bind

    Args:
        spike_raster: Binary spike raster [channels][samples]
        config: Resonance spec configuration
        context: Optional context vector for task binding
        seed: Seed for projection matrix generation

    Returns:
        Dict with vector, sparsity, vector_hash, and bio_ingest_receipt

    Raises:
        ResonanceStopRule: If sparsity > sparsity_max (biological constraint violated)
    """
    if not spike_raster or not spike_raster[0]:
        # Handle empty input gracefully
        dim = config.get("sdm_dim", SDM_DIM)
        zero_vector = [0.0] * dim
        vector_hash = dual_hash(json.dumps(zero_vector))

        receipt = emit_receipt(
            "bio_ingest",
            {
                "spike_hash": dual_hash("empty"),
                "window_ms": config.get("window_ms", WINDOW_MS),
                "sparsity_actual": 0.0,
                "vector_hash": vector_hash,
                "context_bound": False,
            },
        )

        return {
            "vector": zero_vector,
            "sparsity": 0.0,
            "vector_hash": vector_hash,
            "_receipt": receipt,
        }

    n_channels = len(spike_raster)
    dim = config.get("sdm_dim", SDM_DIM)
    window_ms = config.get("window_ms", WINDOW_MS)
    sparsity_target = config.get("sparsity_target", SPARSITY_TARGET)
    sparsity_max = config.get("sparsity_max", SPARSITY_MAX)

    # Step 1: Compute firing rates
    rates = compute_firing_rates(spike_raster, window_ms)

    # Step 2: Generate projection matrix (deterministic from seed)
    projection = generate_projection_matrix(n_channels, dim, seed)

    # Step 3: Project to SDM space
    projected = []
    for row in projection:
        val = sum(r * p for r, p in zip(rates, row))
        projected.append(val)

    # Step 4: Enforce sparsity
    sparse = enforce_sparsity(projected, sparsity_target)

    # Step 5: Normalize to unit sphere
    normalized = normalize_to_sphere(sparse)

    # Step 6: Bind context if provided
    context_bound = False
    if context is not None:
        normalized = bind_context(normalized, context)
        context_bound = True

    # Calculate actual sparsity
    non_zero = sum(1 for v in normalized if v != 0)
    actual_sparsity = non_zero / len(normalized) if normalized else 0

    # StopRule if sparsity violated
    if actual_sparsity > sparsity_max:
        raise ResonanceStopRule(
            "sparsity_violation",
            f"Actual sparsity {actual_sparsity:.4f} > max {sparsity_max}",
            {"actual_sparsity": actual_sparsity, "sparsity_max": sparsity_max},
        )

    # Compute vector hash
    vector_hash = dual_hash(json.dumps([round(v, 8) for v in normalized]))

    # Compute spike hash
    spike_hash = dual_hash(json.dumps(spike_raster[:10]))  # Hash first 10 channels

    # Emit receipt
    receipt = emit_receipt(
        "bio_ingest",
        {
            "spike_hash": spike_hash,
            "window_ms": window_ms,
            "sparsity_actual": round(actual_sparsity, 4),
            "vector_hash": vector_hash,
            "context_bound": context_bound,
        },
    )

    return {
        "vector": normalized,
        "sparsity": actual_sparsity,
        "vector_hash": vector_hash,
        "_receipt": receipt,
    }


def generate_simulated_spikes(
    n_channels: int, n_samples: int, firing_rate: float = 0.01, seed: int = 42
) -> list[list[int]]:
    """Generate simulated spike raster for testing.

    Args:
        n_channels: Number of channels
        n_samples: Number of time samples
        firing_rate: Probability of spike per sample
        seed: Random seed

    Returns:
        Binary spike raster [channels][samples]
    """
    a, c, m = 1664525, 1013904223, 2**32
    state = seed

    raster = []
    for _ in range(n_channels):
        channel = []
        for _ in range(n_samples):
            state = (a * state + c) % m
            spike = 1 if (state / m) < firing_rate else 0
            channel.append(spike)
        raster.append(channel)

    return raster


def generate_context_vector(dim: int, task_id: str) -> list[float]:
    """Generate context vector from task identifier.

    Uses task_id hash to seed deterministic generation.

    Args:
        dim: Vector dimension
        task_id: Task identifier string

    Returns:
        Unit-normalized context vector
    """
    # Use task_id hash as seed
    seed = hash(task_id) & 0xFFFFFFFF

    a, c, m = 1664525, 1013904223, 2**32
    state = seed

    vector = []
    for _ in range(dim):
        state = (a * state + c) % m
        val = (state / m) * 2 - 1
        vector.append(val)

    return normalize_to_sphere(vector)


if __name__ == "__main__":
    print("NEURON-RESONANCE v5.0 - Bio-Digital SDM Bridge")
    print(f"SDM Dimension: {SDM_DIM}")
    print(f"Sparsity Target: {SPARSITY_TARGET}")
    print(f"Sparsity Max: {SPARSITY_MAX}")
    print()
    print("This is not translation. This is resonance.")
    print("Hippocampus and NEURON are the same mathematics.")
