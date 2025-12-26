"""
RESONANCE PROTOCOL - HDC Encoder Tests

Tests for Hyperdimensional Computing encoder with dropout robustness.

GATE 1: HDC_FOUNDATION
    - Test: orthogonality check passes (mean similarity < 0.1)
    - Test: dropout robustness (30% loss) > 85% accuracy
"""

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from hdc import (
    create_random_projection,
    encode_spike_window,
    bundle_temporal,
    cosine_similarity,
    classify_intent,
    test_orthogonality,
    test_dropout,
    HDC_DIMENSION,
    N1_CHANNELS,
    CHANNEL_DROPOUT_TOLERANCE,
    MIN_CLASSIFICATION_ACCURACY,
    ORTHOGONALITY_THRESHOLD,
)


class TestRandomProjection:
    """Tests for random projection matrix generation."""

    def test_projection_dimensions(self):
        """Verify projection matrix has correct dimensions."""
        n_channels = 100
        hdc_dim = 1000
        proj = create_random_projection(n_channels, hdc_dim, seed=42)

        assert len(proj) == n_channels
        assert all(len(row) == hdc_dim for row in proj)

    def test_projection_deterministic(self):
        """Verify same seed produces same projection."""
        proj1 = create_random_projection(100, 1000, seed=42)
        proj2 = create_random_projection(100, 1000, seed=42)

        assert proj1 == proj2

    def test_projection_different_seeds(self):
        """Verify different seeds produce different projections."""
        proj1 = create_random_projection(100, 1000, seed=42)
        proj2 = create_random_projection(100, 1000, seed=43)

        assert proj1 != proj2


class TestEncoding:
    """Tests for spike window encoding."""

    def test_encode_returns_correct_dimension(self, sample_projection_matrix):
        """Verify encoded vector has correct dimension."""
        spike_counts = [1.0] * 100
        hv = encode_spike_window(spike_counts, sample_projection_matrix)

        assert len(hv) == 1000

    def test_encode_normalized(self, sample_projection_matrix):
        """Verify encoded vector is unit-normalized."""
        spike_counts = [1.0, 2.0, 0.5] + [0.0] * 97
        hv = encode_spike_window(spike_counts, sample_projection_matrix)

        norm = math.sqrt(sum(v * v for v in hv))
        assert abs(norm - 1.0) < 1e-6

    def test_encode_with_dropout(self, sample_projection_matrix):
        """Verify dropout mask is applied correctly."""
        spike_counts = [1.0] * 100
        dropout_mask = [1] * 50 + [0] * 50  # 50% dropout

        hv = encode_spike_window(spike_counts, sample_projection_matrix, dropout_mask)

        assert len(hv) == 1000
        # Vector should still be normalized
        norm = math.sqrt(sum(v * v for v in hv))
        assert abs(norm - 1.0) < 1e-6

    def test_encode_zero_input(self, sample_projection_matrix):
        """Verify zero input produces zero vector."""
        spike_counts = [0.0] * 100
        hv = encode_spike_window(spike_counts, sample_projection_matrix)

        assert all(v == 0.0 for v in hv)


class TestBundling:
    """Tests for temporal bundling."""

    def test_bundle_single_vector(self):
        """Bundling single vector returns normalized copy."""
        hv = [0.5, 0.5, 0.5, 0.5]
        bundled = bundle_temporal([hv])

        norm = math.sqrt(sum(v * v for v in bundled))
        assert abs(norm - 1.0) < 1e-6

    def test_bundle_multiple_vectors(self):
        """Bundling multiple vectors produces valid result."""
        hv1 = [1.0, 0.0, 0.0, 0.0]
        hv2 = [0.0, 1.0, 0.0, 0.0]

        bundled = bundle_temporal([hv1, hv2])

        assert len(bundled) == 4
        norm = math.sqrt(sum(v * v for v in bundled))
        assert abs(norm - 1.0) < 1e-6

    def test_bundle_with_weights(self):
        """Bundling with weights applies correctly."""
        hv1 = [1.0, 0.0, 0.0, 0.0]
        hv2 = [0.0, 1.0, 0.0, 0.0]

        bundled = bundle_temporal([hv1, hv2], weights=[0.9, 0.1])

        # First component should dominate
        assert bundled[0] > bundled[1]


class TestSimilarity:
    """Tests for cosine similarity."""

    def test_identical_vectors(self):
        """Identical vectors have similarity 1.0."""
        hv = [0.5, 0.5, 0.5, 0.5]
        sim = cosine_similarity(hv, hv)

        assert abs(sim - 1.0) < 1e-6

    def test_orthogonal_vectors(self):
        """Orthogonal vectors have similarity 0.0."""
        hv1 = [1.0, 0.0, 0.0, 0.0]
        hv2 = [0.0, 1.0, 0.0, 0.0]
        sim = cosine_similarity(hv1, hv2)

        assert abs(sim) < 1e-6

    def test_opposite_vectors(self):
        """Opposite vectors have similarity -1.0."""
        hv1 = [1.0, 0.0, 0.0, 0.0]
        hv2 = [-1.0, 0.0, 0.0, 0.0]
        sim = cosine_similarity(hv1, hv2)

        assert abs(sim + 1.0) < 1e-6


class TestClassification:
    """Tests for intent classification."""

    def test_classify_exact_match(self):
        """Classification returns correct class for exact match."""
        prototypes = {
            "class_a": [1.0, 0.0, 0.0, 0.0],
            "class_b": [0.0, 1.0, 0.0, 0.0],
        }

        result = classify_intent([1.0, 0.0, 0.0, 0.0], prototypes)
        assert result == "class_a"

        result = classify_intent([0.0, 1.0, 0.0, 0.0], prototypes)
        assert result == "class_b"

    def test_classify_noisy_input(self):
        """Classification handles noisy input correctly."""
        prototypes = {
            "class_a": [1.0, 0.0, 0.0, 0.0],
            "class_b": [0.0, 1.0, 0.0, 0.0],
        }

        # Noisy version of class_a
        noisy = [0.9, 0.1, 0.05, -0.05]
        result = classify_intent(noisy, prototypes)
        assert result == "class_a"


class TestOrthogonality:
    """Tests for orthogonality verification."""

    def test_orthogonality_passes(self, temp_receipts_dir):
        """GATE 1: Orthogonality check passes with mean similarity < 0.1."""
        mean_sim = test_orthogonality(n_vectors=100)

        assert mean_sim < ORTHOGONALITY_THRESHOLD
        assert mean_sim >= 0.0


class TestDropoutRobustness:
    """Tests for dropout robustness."""

    def test_dropout_robustness_passes(self, temp_receipts_dir):
        """GATE 1: Dropout robustness > 85% with 30% channel loss."""
        accuracy = test_dropout(dropout_rate=CHANNEL_DROPOUT_TOLERANCE)

        assert accuracy > MIN_CLASSIFICATION_ACCURACY
        assert accuracy <= 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
