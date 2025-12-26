"""
NEURON-RESONANCE v5.0 Resonance Bridge Tests
Test bio-digital translation layer - spike rasters to SDM vectors.
"""

import os
import tempfile
import math

# Set up isolated test environment BEFORE importing
_test_dir = tempfile.mkdtemp()
os.environ["NEURON_LEDGER"] = os.path.join(_test_dir, "test_receipts.jsonl")
os.environ["NEURON_ARCHIVE"] = os.path.join(_test_dir, "test_archive.jsonl")
os.environ["NEURON_RECEIPTS"] = os.path.join(_test_dir, "test_receipts.jsonl")
os.environ["NEURON_BASE"] = _test_dir

# Create data directory and spec file for tests
_data_dir = os.path.join(_test_dir, "data")
os.makedirs(_data_dir, exist_ok=True)
_spec_path = os.path.join(_data_dir, "resonance_spec.json")
with open(_spec_path, "w") as f:
    f.write("""{
  "sdm_dim": 16384,
  "n1_channels_range": [1024, 4096],
  "window_ms": 100,
  "sparsity_target": 0.01,
  "sparsity_max": 0.05,
  "swr_frequency_hz": [150, 250],
  "swr_confidence_threshold": 0.8,
  "urgency_threshold": 0.8,
  "stim_duration_ms": 200,
  "stim_intensity_max": 1.0,
  "stim_regions": ["insula", "amygdala", "somatosensory_cortex"],
  "prune_threshold": 0.3,
  "top_k_retrieval": 16,
  "simulation_mode": true
}""")

import pytest
from pathlib import Path

from resonance_bridge import (
    load_resonance_spec,
    generate_projection_matrix,
    enforce_sparsity,
    normalize_to_sphere,
    bind_context,
    biological_to_digital_sdm,
    generate_simulated_spikes,
    generate_context_vector,
    ResonanceStopRule,
    SDM_DIM,
    SPARSITY_TARGET,
    SPARSITY_MAX,
)


@pytest.fixture(autouse=True)
def clean_files():
    """Clean test files before and after each test."""
    for f in Path(_test_dir).glob("*.jsonl"):
        f.unlink(missing_ok=True)
    yield
    for f in Path(_test_dir).glob("*.jsonl"):
        f.unlink(missing_ok=True)


# ============================================
# GATE 1: SPEC LOADING TESTS
# ============================================


class TestLoadResonanceSpec:
    """Test load_resonance_spec validates correctly."""

    def test_load_valid_spec(self):
        """Valid spec loads and returns dict."""
        spec = load_resonance_spec(_spec_path)

        assert spec["sdm_dim"] == 16384
        assert spec["sparsity_target"] == 0.01
        assert spec["sparsity_max"] == 0.05
        assert "_spec_hash" in spec
        assert "_ingest_receipt" in spec

    def test_spec_hash_computed(self):
        """Spec has dual-hash computed."""
        spec = load_resonance_spec(_spec_path)

        # Dual hash format: SHA256:BLAKE3
        assert ":" in spec["_spec_hash"]
        parts = spec["_spec_hash"].split(":")
        assert len(parts) == 2

    def test_stoprule_on_missing_spec(self):
        """StopRule if spec file not found."""
        with pytest.raises(ResonanceStopRule) as exc_info:
            load_resonance_spec("/nonexistent/path.json")

        assert "spec_not_found" in str(exc_info.value)

    def test_stoprule_on_constraint_violation(self):
        """StopRule if sparsity_target > sparsity_max."""
        bad_spec_path = os.path.join(_test_dir, "bad_spec.json")
        with open(bad_spec_path, "w") as f:
            f.write("""{
                "sdm_dim": 16384,
                "sparsity_target": 0.10,
                "sparsity_max": 0.05,
                "swr_frequency_hz": [150, 250],
                "stim_intensity_max": 1.0
            }""")

        with pytest.raises(ResonanceStopRule) as exc_info:
            load_resonance_spec(bad_spec_path)

        assert "constraint_violation" in str(exc_info.value)


# ============================================
# GATE 2: PROJECTION MATRIX TESTS
# ============================================


class TestProjectionMatrix:
    """Test projection matrix generation."""

    def test_projection_deterministic(self):
        """Same seed → same projection matrix."""
        p1 = generate_projection_matrix(1024, 16384, 42)
        p2 = generate_projection_matrix(1024, 16384, 42)

        assert p1 == p2

    def test_projection_different_seeds(self):
        """Different seeds → different matrices."""
        p1 = generate_projection_matrix(1024, 16384, 42)
        p2 = generate_projection_matrix(1024, 16384, 43)

        assert p1 != p2

    def test_projection_dimensions(self):
        """Output has correct dimensions."""
        p = generate_projection_matrix(1024, 16384, 42)

        assert len(p) == 16384
        assert len(p[0]) == 1024


# ============================================
# GATE 2: SPARSITY TESTS
# ============================================


class TestEnforceSparsity:
    """Test sparsity enforcement."""

    def test_sparsity_enforced(self):
        """Output sparsity <= sparsity_max for any input."""
        # Generate random vector
        vector = [((i * 1234567) % 100) / 100 - 0.5 for i in range(1000)]

        sparse = enforce_sparsity(vector, 0.01)

        non_zero = sum(1 for v in sparse if v != 0)
        actual_sparsity = non_zero / len(sparse)

        assert actual_sparsity <= 0.05  # Within max

    def test_sparsity_approximately_target(self):
        """Output sparsity approximately matches target."""
        vector = [((i * 1234567) % 100) / 100 - 0.5 for i in range(10000)]

        sparse = enforce_sparsity(vector, 0.01)

        non_zero = sum(1 for v in sparse if v != 0)
        actual_sparsity = non_zero / len(sparse)

        # Should be close to target (within 1%)
        assert abs(actual_sparsity - 0.01) < 0.01


class TestNormalizeToSphere:
    """Test unit sphere normalization."""

    def test_unit_sphere_normalization(self):
        """Output norm == 1.0 (within epsilon)."""
        vector = [1.0, 2.0, 3.0, 4.0, 5.0]

        normalized = normalize_to_sphere(vector)

        norm = math.sqrt(sum(v * v for v in normalized))
        assert abs(norm - 1.0) < 1e-10

    def test_zero_vector_handled(self):
        """Zero-spike window → zero vector (valid, no crash)."""
        vector = [0.0, 0.0, 0.0, 0.0]

        normalized = normalize_to_sphere(vector)

        assert all(v == 0.0 for v in normalized)


# ============================================
# GATE 2: CONTEXT BINDING TESTS
# ============================================


class TestContextBinding:
    """Test context binding for task isolation."""

    def test_context_binding_prevents_collision(self):
        """Same spikes + different context → different vectors."""
        dim = 1000
        bio_vector = [0.01] * dim
        bio_vector = normalize_to_sphere(bio_vector)

        context1 = generate_context_vector(dim, "task_1")
        context2 = generate_context_vector(dim, "task_2")

        bound1 = bind_context(bio_vector, context1)
        bound2 = bind_context(bio_vector, context2)

        # Vectors should be different
        assert bound1 != bound2

    def test_context_dimension_mismatch_raises(self):
        """Dimension mismatch raises StopRule."""
        bio_vector = [0.1] * 100
        context_vector = [0.1] * 200  # Different dimension

        with pytest.raises(ResonanceStopRule) as exc_info:
            bind_context(bio_vector, context_vector)

        assert "dimension_mismatch" in str(exc_info.value)


# ============================================
# GATE 2: FULL PIPELINE TESTS
# ============================================


class TestBiologicalToDigitalSDM:
    """Test full bio-to-digital pipeline."""

    def test_full_pipeline_works(self):
        """Full pipeline produces valid output."""
        spec = load_resonance_spec(_spec_path)
        spikes = generate_simulated_spikes(1024, 100, 0.01, 42)

        result = biological_to_digital_sdm(spikes, spec)

        assert "vector" in result
        assert "sparsity" in result
        assert "vector_hash" in result
        assert "_receipt" in result

    def test_bio_ingest_receipt_emitted(self):
        """All fields present, dual-hash valid."""
        spec = load_resonance_spec(_spec_path)
        spikes = generate_simulated_spikes(1024, 100, 0.01, 42)

        result = biological_to_digital_sdm(spikes, spec)

        receipt = result["_receipt"]
        assert receipt["type"] == "bio_ingest"
        assert "spike_hash" in receipt
        assert "window_ms" in receipt
        assert "sparsity_actual" in receipt
        assert "vector_hash" in receipt
        assert "context_bound" in receipt
        assert ":" in receipt["hash"]  # Dual hash

    def test_empty_spikes_returns_zero_vector(self):
        """Empty spike raster returns zero vector."""
        spec = load_resonance_spec(_spec_path)
        empty_spikes = []

        result = biological_to_digital_sdm(empty_spikes, spec)

        assert result["sparsity"] == 0.0
        assert all(v == 0.0 for v in result["vector"])

    def test_sparsity_enforced_to_target(self):
        """Output sparsity should be approximately target regardless of input density."""
        spec = load_resonance_spec(_spec_path)
        # Dense spikes - all neurons firing
        dense_spikes = [[1] * 1000 for _ in range(1024)]

        result = biological_to_digital_sdm(dense_spikes, spec)

        # Sparsity should be enforced to target, not exceed max
        assert result["sparsity"] <= spec["sparsity_max"]


# ============================================
# CONSTANTS VERIFICATION TESTS
# ============================================


class TestConstants:
    """Test v5.0 resonance constants are correct."""

    def test_sdm_dim(self):
        """SDM_DIM is 16384."""
        assert SDM_DIM == 16384

    def test_sparsity_target(self):
        """SPARSITY_TARGET is 0.01."""
        assert SPARSITY_TARGET == 0.01

    def test_sparsity_max(self):
        """SPARSITY_MAX is 0.05."""
        assert SPARSITY_MAX == 0.05
