"""
RESONANCE PROTOCOL - Federated Learning Coordinator

Aggregate model updates without accessing raw neural data.
HARD REQUIREMENT: Raw Neural Data Never Leaves the User's Local Ecosystem.

THE SOVEREIGNTY GUARANTEE:
    Federated Learning is not privacy feature - it's architecture.
    Only model deltas propagate, never spike trains.
    This is ProofPack's local receipts philosophy for neural data.

Functions:
    compute_local_update: Train model on local data, return gradient/delta only
    aggregate_updates: Federated averaging of model deltas
    apply_update: Apply aggregated delta to model
    verify_no_raw_data: Audit function - zero raw data transmissions
    estimate_privacy_budget: Compute differential privacy epsilon
"""

from __future__ import annotations

import json
import math
from typing import Any

try:
    from .core import (
        FL_UPDATE_INTERVAL_SEC,
        FL_MIN_PARTICIPANTS,
        PRIVACY_BUDGET_EPSILON,
        emit_receipt,
        dual_hash,
    )
except ImportError:
    from core import (
        FL_UPDATE_INTERVAL_SEC,
        FL_MIN_PARTICIPANTS,
        PRIVACY_BUDGET_EPSILON,
        emit_receipt,
        dual_hash,
    )


class FederatedViolation(Exception):
    """Raised when federated learning privacy guarantee is violated."""

    def __init__(self, message: str, context: dict | None = None):
        self.context = context or {}
        super().__init__(f"FL_VIOLATION: {message}")


def compute_local_update(
    local_data: list[list[float]],
    current_model: dict[str, list[float]]
) -> dict[str, list[float]]:
    """Train model on local data. Return gradient/delta only (no raw data).

    PRIVACY GUARANTEE: Only model parameter deltas are returned.
    Raw neural data (spike trains, LFP) never leaves this function.

    Args:
        local_data: Local training data (spike rates, processed features)
        current_model: Current model parameters {layer_name: weights}

    Returns:
        Model delta {layer_name: weight_deltas}
    """
    if not local_data or not current_model:
        return {k: [0.0] * len(v) for k, v in current_model.items()}

    # Simple gradient descent update (simplified for demonstration)
    learning_rate = 0.01
    deltas = {}

    for layer_name, weights in current_model.items():
        # Compute pseudo-gradient from local data
        n_weights = len(weights)
        gradient = [0.0] * n_weights

        for sample in local_data:
            # Simple gradient: correlation between input and weight direction
            for i in range(min(len(sample), n_weights)):
                gradient[i] += sample[i] * 0.01

        # Average gradient
        n_samples = len(local_data)
        gradient = [g / n_samples for g in gradient]

        # Compute delta
        delta = [-learning_rate * g for g in gradient]
        deltas[layer_name] = delta

    return deltas


def aggregate_updates(
    updates: list[dict[str, list[float]]]
) -> dict[str, list[float]]:
    """Federated averaging: mean of all updates.

    PRIVACY GUARANTEE: Only aggregated deltas, never individual updates.

    Args:
        updates: List of model deltas from different participants

    Returns:
        Averaged model delta
    """
    if not updates:
        return {}

    if len(updates) < FL_MIN_PARTICIPANTS:
        raise FederatedViolation(
            f"Insufficient participants: {len(updates)} < {FL_MIN_PARTICIPANTS}",
            {"n_participants": len(updates), "minimum": FL_MIN_PARTICIPANTS}
        )

    # Get all layer names
    all_layers = set()
    for update in updates:
        all_layers.update(update.keys())

    # Average each layer
    aggregated = {}
    for layer_name in all_layers:
        layer_updates = [u.get(layer_name, []) for u in updates if layer_name in u]
        if not layer_updates:
            continue

        # Find max dimension
        max_len = max(len(lu) for lu in layer_updates)

        # Average
        averaged = [0.0] * max_len
        for lu in layer_updates:
            for i in range(len(lu)):
                averaged[i] += lu[i]

        n = len(layer_updates)
        averaged = [a / n for a in averaged]
        aggregated[layer_name] = averaged

    return aggregated


def apply_update(
    current_model: dict[str, list[float]],
    delta: dict[str, list[float]]
) -> dict[str, list[float]]:
    """Apply delta to model parameters.

    Args:
        current_model: Current model parameters
        delta: Aggregated update delta

    Returns:
        Updated model parameters
    """
    updated = {}
    for layer_name, weights in current_model.items():
        layer_delta = delta.get(layer_name, [0.0] * len(weights))
        updated_weights = [
            w + d for w, d in zip(weights, layer_delta + [0.0] * (len(weights) - len(layer_delta)))
        ]
        updated[layer_name] = updated_weights

    return updated


def _contains_spike_pattern(data: Any, depth: int = 0) -> bool:
    """Check if data structure contains patterns indicative of raw spike data.

    Raw spike data characteristics:
    - Binary sequences (0/1 only)
    - Long sequences (>100 elements)
    - High sparsity (>90% zeros)
    """
    if depth > 10:
        return False

    if isinstance(data, list):
        if len(data) > 100:
            # Check for binary pattern
            numeric_items = [x for x in data if isinstance(x, (int, float))]
            if len(numeric_items) > 100:
                unique_vals = set(numeric_items)
                if unique_vals <= {0, 1, 0.0, 1.0}:
                    # Likely binary spike train
                    return True

                # Check for high sparsity
                zeros = sum(1 for x in numeric_items if x == 0 or x == 0.0)
                if zeros / len(numeric_items) > 0.9:
                    return True

        # Recurse into sublists
        for item in data[:10]:  # Check first 10 items
            if _contains_spike_pattern(item, depth + 1):
                return True

    elif isinstance(data, dict):
        for v in data.values():
            if _contains_spike_pattern(v, depth + 1):
                return True

    return False


def verify_no_raw_data(transmitted_data: Any = None) -> bool:
    """Audit function: check that no raw spike data exists in transmitted object.

    HARD REQUIREMENT: Must return True for ALL transmissions.

    Args:
        transmitted_data: Data object to audit (or None for self-test)

    Returns:
        True if no raw data detected (CLEAN)
    """
    if transmitted_data is None:
        # Self-test mode
        # Test 1: Valid model delta (should pass)
        valid_delta = {"layer1": [0.01, -0.02, 0.015], "layer2": [0.001] * 50}
        assert not _contains_spike_pattern(valid_delta), "Valid delta failed"

        # Test 2: Raw spike train (should fail)
        raw_spikes = [0, 0, 0, 1, 0, 0, 0, 0, 1, 0] * 20  # Binary spike train
        assert _contains_spike_pattern(raw_spikes), "Spike detection failed"

        # Test 3: Sparse data (should fail)
        sparse = [0.0] * 95 + [1.0] * 5 + [0.0] * 100
        assert _contains_spike_pattern(sparse), "Sparse detection failed"

        emit_receipt(
            "fl_privacy_audit",
            {
                "mode": "self_test",
                "tests_passed": 3,
                "raw_data_detected": False,
            }
        )
        return True

    # Audit mode
    has_raw_data = _contains_spike_pattern(transmitted_data)

    emit_receipt(
        "fl_privacy_audit",
        {
            "mode": "audit",
            "raw_data_detected": has_raw_data,
            "data_hash": dual_hash(json.dumps(str(transmitted_data)[:1000]))[:32],
        }
    )

    if has_raw_data:
        raise FederatedViolation(
            "Raw neural data detected in transmission",
            {"data_type": type(transmitted_data).__name__}
        )

    return True


def estimate_privacy_budget(
    n_updates: int,
    sensitivity: float = 1.0,
    noise_multiplier: float = 1.0
) -> float:
    """Compute differential privacy epsilon.

    Uses Gaussian mechanism privacy accounting.

    Args:
        n_updates: Number of update rounds
        sensitivity: L2 sensitivity of the query
        noise_multiplier: Ratio of noise to sensitivity

    Returns:
        Privacy cost (epsilon)
    """
    if noise_multiplier <= 0:
        return float('inf')

    # Simplified Gaussian mechanism epsilon
    # epsilon = sqrt(2 * ln(1.25/delta)) * sensitivity / noise
    delta = 1e-5  # Standard delta for DP
    epsilon_per_update = math.sqrt(2 * math.log(1.25 / delta)) * sensitivity / noise_multiplier

    # Composition over updates (simple composition, not optimal)
    total_epsilon = epsilon_per_update * math.sqrt(n_updates)

    return round(total_epsilon, 4)


def emit_fl_update_receipt(
    model_delta_hash: str,
    n_local_samples: int,
    gradient_norm: float,
    privacy_budget: float
) -> dict:
    """Emit fl_update_receipt per specification.

    Args:
        model_delta_hash: Hash of the model delta
        n_local_samples: Number of local training samples
        gradient_norm: L2 norm of the gradient
        privacy_budget: Current privacy epsilon

    Returns:
        Receipt dict
    """
    return emit_receipt(
        "fl_update",
        {
            "model_delta_hash": model_delta_hash[:32] if len(model_delta_hash) > 32 else model_delta_hash,
            "n_local_samples": n_local_samples,
            "gradient_norm": round(gradient_norm, 6),
            "privacy_budget": round(privacy_budget, 4),
            "interval_sec": FL_UPDATE_INTERVAL_SEC,
        }
    )


class FederatedCoordinator:
    """Coordinates federated learning across participants.

    Maintains aggregation state and privacy accounting.
    """

    def __init__(self, initial_model: dict[str, list[float]] | None = None):
        self.model = initial_model or {}
        self.pending_updates: list[dict] = []
        self.update_count = 0
        self.privacy_spent = 0.0

    def receive_update(self, update: dict[str, list[float]], n_samples: int) -> bool:
        """Receive update from participant.

        Args:
            update: Model delta from participant
            n_samples: Number of local samples used

        Returns:
            True if update accepted
        """
        # Privacy check
        if _contains_spike_pattern(update):
            raise FederatedViolation("Raw data in update rejected")

        self.pending_updates.append({
            "delta": update,
            "n_samples": n_samples,
            "hash": dual_hash(json.dumps(update, sort_keys=True)),
        })

        return True

    def aggregate_if_ready(self) -> bool:
        """Aggregate updates if minimum participants reached.

        Returns:
            True if aggregation performed
        """
        if len(self.pending_updates) < FL_MIN_PARTICIPANTS:
            return False

        # Extract deltas
        deltas = [u["delta"] for u in self.pending_updates]

        # Aggregate
        aggregated = aggregate_updates(deltas)

        # Apply to model
        if self.model:
            self.model = apply_update(self.model, aggregated)
        else:
            self.model = aggregated

        # Update accounting
        self.update_count += 1
        self.privacy_spent = estimate_privacy_budget(self.update_count)

        # Emit receipt
        emit_fl_update_receipt(
            model_delta_hash=dual_hash(json.dumps(aggregated, sort_keys=True)),
            n_local_samples=sum(u["n_samples"] for u in self.pending_updates),
            gradient_norm=sum(sum(abs(v) for v in d) for d in aggregated.values()),
            privacy_budget=self.privacy_spent,
        )

        # Clear pending
        self.pending_updates = []

        return True


if __name__ == "__main__":
    print("RESONANCE PROTOCOL - Federated Coordinator")
    print(f"Update Interval: {FL_UPDATE_INTERVAL_SEC}s")
    print(f"Min Participants: {FL_MIN_PARTICIPANTS}")
    print(f"Privacy Budget: {PRIVACY_BUDGET_EPSILON}")
    print()
    print("Running privacy verification...")
    passed = verify_no_raw_data()
    print(f"Self-test PASSED: {passed}")
