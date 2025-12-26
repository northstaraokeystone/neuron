"""
RESONANCE PROTOCOL - Federated Learning Tests

Tests for federated learning with privacy guarantees.

GATE 3: FEDERATED_INFRASTRUCTURE
    - Test: model convergence with synthetic updates
    - Test: zero raw data transmissions logged
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from federated_coordinator import (
    compute_local_update,
    aggregate_updates,
    apply_update,
    verify_no_raw_data,
    estimate_privacy_budget,
    emit_fl_update_receipt,
    FederatedCoordinator,
    FederatedViolation,
    FL_MIN_PARTICIPANTS,
    PRIVACY_BUDGET_EPSILON,
)


class TestLocalUpdate:
    """Tests for local model update computation."""

    def test_local_update_dimensions(self, sample_model):
        """Local update has same dimensions as model."""
        local_data = [[0.1, 0.2, 0.3, 0.4, 0.5] for _ in range(10)]
        update = compute_local_update(local_data, sample_model)

        assert set(update.keys()) == set(sample_model.keys())
        for layer in sample_model:
            assert len(update[layer]) == len(sample_model[layer])

    def test_local_update_no_raw_data(self, sample_model):
        """Local update contains no raw spike data."""
        local_data = [[0.1, 0.2, 0.3, 0.4, 0.5] for _ in range(10)]
        update = compute_local_update(local_data, sample_model)

        assert verify_no_raw_data(update)

    def test_empty_data_returns_zeros(self, sample_model):
        """Empty data returns zero update."""
        update = compute_local_update([], sample_model)

        for layer in update:
            assert all(v == 0.0 for v in update[layer])


class TestAggregation:
    """Tests for federated averaging."""

    def test_aggregation_averages_updates(self):
        """Aggregation computes mean of updates."""
        updates = [
            {"layer1": [1.0, 2.0, 3.0]},
            {"layer1": [3.0, 4.0, 5.0]},
            {"layer1": [2.0, 3.0, 4.0]},
        ]

        aggregated = aggregate_updates(updates)

        # Mean should be [2.0, 3.0, 4.0]
        expected = [2.0, 3.0, 4.0]
        for a, e in zip(aggregated["layer1"], expected):
            assert abs(a - e) < 1e-6

    def test_aggregation_requires_min_participants(self):
        """Aggregation fails with insufficient participants."""
        updates = [{"layer1": [1.0, 2.0]}]  # Only 1 participant

        with pytest.raises(FederatedViolation):
            aggregate_updates(updates)

    def test_aggregation_handles_multiple_layers(self):
        """Aggregation works with multiple layers."""
        updates = [
            {"layer1": [1.0], "layer2": [2.0]},
            {"layer1": [3.0], "layer2": [4.0]},
            {"layer1": [2.0], "layer2": [3.0]},
        ]

        aggregated = aggregate_updates(updates)

        assert "layer1" in aggregated
        assert "layer2" in aggregated


class TestModelUpdate:
    """Tests for model update application."""

    def test_apply_update_modifies_weights(self, sample_model):
        """Apply update modifies model weights correctly."""
        delta = {
            "layer1": [0.1] * len(sample_model["layer1"]),
            "layer2": [0.01] * len(sample_model["layer2"]),
        }

        updated = apply_update(sample_model, delta)

        # Check weights were modified
        for i, (old, new) in enumerate(zip(sample_model["layer1"], updated["layer1"])):
            assert abs(new - old - 0.1) < 1e-6


class TestPrivacyVerification:
    """Tests for privacy guarantee verification."""

    def test_verify_no_raw_data_self_test(self, temp_receipts_dir):
        """GATE 3: Self-test for raw data detection passes."""
        result = verify_no_raw_data()

        assert result == True

    def test_verify_rejects_spike_train(self, temp_receipts_dir):
        """Raw spike train is detected and rejected."""
        raw_spikes = [0, 0, 0, 1, 0, 0, 0, 0, 1, 0] * 20

        with pytest.raises(FederatedViolation):
            verify_no_raw_data(raw_spikes)

    def test_verify_accepts_model_delta(self, temp_receipts_dir):
        """Valid model delta is accepted."""
        delta = {"layer1": [0.01, -0.02, 0.015], "layer2": [0.001] * 50}

        assert verify_no_raw_data(delta) == True


class TestPrivacyBudget:
    """Tests for differential privacy accounting."""

    def test_privacy_budget_increases(self):
        """Privacy budget increases with more updates."""
        budget_1 = estimate_privacy_budget(n_updates=1)
        budget_10 = estimate_privacy_budget(n_updates=10)

        assert budget_10 > budget_1

    def test_privacy_budget_with_noise(self):
        """Higher noise multiplier reduces epsilon."""
        budget_low_noise = estimate_privacy_budget(n_updates=10, noise_multiplier=0.5)
        budget_high_noise = estimate_privacy_budget(n_updates=10, noise_multiplier=2.0)

        assert budget_high_noise < budget_low_noise


class TestFederatedCoordinator:
    """Tests for coordinator class."""

    def test_coordinator_receives_updates(self, sample_model, temp_receipts_dir):
        """Coordinator accepts valid updates."""
        coordinator = FederatedCoordinator(sample_model)

        update = {"layer1": [0.01] * 5, "layer2": [0.001] * 10}
        result = coordinator.receive_update(update, n_samples=100)

        assert result == True
        assert len(coordinator.pending_updates) == 1

    def test_coordinator_rejects_raw_data(self, sample_model, temp_receipts_dir):
        """Coordinator rejects updates with raw data."""
        coordinator = FederatedCoordinator(sample_model)

        raw_data_update = [0, 1, 0, 0, 1] * 50  # Looks like spike train

        with pytest.raises(FederatedViolation):
            coordinator.receive_update(raw_data_update, n_samples=100)

    def test_coordinator_aggregates_when_ready(self, sample_model, temp_receipts_dir):
        """Coordinator aggregates when minimum participants reached."""
        coordinator = FederatedCoordinator(sample_model)

        # Add minimum participants
        for i in range(FL_MIN_PARTICIPANTS):
            update = {"layer1": [0.01 * i] * 5, "layer2": [0.001 * i] * 10}
            coordinator.receive_update(update, n_samples=100)

        result = coordinator.aggregate_if_ready()

        assert result == True
        assert len(coordinator.pending_updates) == 0
        assert coordinator.update_count == 1


class TestReceipt:
    """Tests for FL receipt emission."""

    def test_fl_update_receipt(self, temp_receipts_dir):
        """FL update receipt contains required fields."""
        receipt = emit_fl_update_receipt(
            model_delta_hash="abc123",
            n_local_samples=1000,
            gradient_norm=0.5,
            privacy_budget=0.1
        )

        assert receipt["type"] == "fl_update"
        assert "model_delta_hash" in receipt
        assert "n_local_samples" in receipt
        assert "gradient_norm" in receipt
        assert "privacy_budget" in receipt


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
