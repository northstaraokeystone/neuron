"""
RESONANCE PROTOCOL - Entropy Measurement

Extend entropy measurement to HDC vector distributions.
Entropy as fundamental metric for system health.

THE INSIGHT:
    Shannon entropy in HDC vector spaces parallels QED's entropy-based
    compression. Both use information theory as fundamental metric.
    Well-formed random hypervectors have near-maximum entropy (~0.999).

Functions:
    shannon_entropy: Classic Shannon entropy H = -sum(p(x) log2 p(x))
    hdc_vector_entropy: Entropy of bit distribution in hypervector
    measure_class_separation: Mean pairwise distance between prototypes
"""

from __future__ import annotations

import math
from typing import Callable

try:
    from .core import (
        HDC_DIMENSION,
        emit_receipt,
    )
except ImportError:
    from core import (
        HDC_DIMENSION,
        emit_receipt,
    )


def shannon_entropy(distribution: list[float]) -> float:
    """Classic Shannon entropy H = -sum(p(x) log2 p(x)).

    Args:
        distribution: Probability distribution (should sum to 1)

    Returns:
        Entropy in bits
    """
    if not distribution:
        return 0.0

    # Normalize if needed
    total = sum(distribution)
    if total <= 0:
        return 0.0

    probs = [p / total for p in distribution]

    entropy = 0.0
    for p in probs:
        if p > 0:
            entropy -= p * math.log2(p)

    return entropy


def hdc_vector_entropy(hypervector: list[float]) -> float:
    """Compute entropy of bit distribution in hypervector.

    For continuous-valued vectors, discretizes to bins.
    Ideal well-formed random hypervector: entropy ~ 0.999 (near-maximum).

    Args:
        hypervector: Input hypervector (any dimension)

    Returns:
        Normalized entropy (0 to 1)
    """
    if not hypervector:
        return 0.0

    n = len(hypervector)

    # Discretize into bins
    n_bins = min(100, n // 10)
    if n_bins < 2:
        n_bins = 2

    # Find range
    min_val = min(hypervector)
    max_val = max(hypervector)

    if max_val == min_val:
        return 0.0  # No variance = no entropy

    bin_width = (max_val - min_val) / n_bins

    # Count per bin
    bin_counts = [0] * n_bins
    for v in hypervector:
        bin_idx = int((v - min_val) / bin_width)
        bin_idx = min(bin_idx, n_bins - 1)  # Handle edge case
        bin_counts[bin_idx] += 1

    # Compute entropy
    probs = [c / n for c in bin_counts]
    entropy = shannon_entropy(probs)

    # Normalize by maximum possible entropy
    max_entropy = math.log2(n_bins)
    normalized = entropy / max_entropy if max_entropy > 0 else 0.0

    return normalized


def measure_class_separation(
    class_prototypes: dict[str, list[float]]
) -> float:
    """Compute mean pairwise cosine distance between class prototypes.

    Higher separation = better class discrimination.

    Args:
        class_prototypes: Dict {class_name: prototype_hypervector}

    Returns:
        Mean pairwise distance (0 to 2, where 2 = opposite vectors)
    """
    if len(class_prototypes) < 2:
        return 0.0

    prototypes = list(class_prototypes.values())
    n = len(prototypes)

    # Compute pairwise distances
    distances = []
    for i in range(n):
        for j in range(i + 1, n):
            # Cosine distance = 1 - cosine_similarity
            dot = sum(a * b for a, b in zip(prototypes[i], prototypes[j]))
            norm_i = math.sqrt(sum(v * v for v in prototypes[i]))
            norm_j = math.sqrt(sum(v * v for v in prototypes[j]))

            if norm_i > 0 and norm_j > 0:
                cos_sim = dot / (norm_i * norm_j)
                distance = 1 - cos_sim
                distances.append(distance)

    if not distances:
        return 0.0

    return sum(distances) / len(distances)


def vector_sparsity(hypervector: list[float], threshold: float = 0.01) -> float:
    """Measure sparsity of hypervector.

    Args:
        hypervector: Input vector
        threshold: Minimum absolute value to be considered non-zero

    Returns:
        Fraction of near-zero elements (0 to 1)
    """
    if not hypervector:
        return 0.0

    near_zero = sum(1 for v in hypervector if abs(v) < threshold)
    return near_zero / len(hypervector)


def compute_kl_divergence(
    p: list[float],
    q: list[float]
) -> float:
    """Compute KL divergence D_KL(P || Q).

    Args:
        p: True distribution
        q: Approximate distribution

    Returns:
        KL divergence in bits (non-negative)
    """
    if len(p) != len(q):
        raise ValueError("Distributions must have same length")

    # Normalize
    sum_p = sum(p)
    sum_q = sum(q)

    if sum_p <= 0 or sum_q <= 0:
        return float('inf')

    p_norm = [v / sum_p for v in p]
    q_norm = [v / sum_q for v in q]

    kl = 0.0
    for pi, qi in zip(p_norm, q_norm):
        if pi > 0:
            if qi <= 0:
                return float('inf')
            kl += pi * math.log2(pi / qi)

    return kl


def test_hdc_entropy(n_vectors: int = 100) -> dict:
    """Test that random hypervectors have near-maximum entropy.

    PASS CRITERIA: Mean entropy > 0.95 for random vectors

    Args:
        n_vectors: Number of vectors to test

    Returns:
        Test result dict
    """
    import random
    random.seed(42)

    entropies = []

    for _ in range(n_vectors):
        # Generate random unit hypervector
        hv = [random.gauss(0, 1) for _ in range(HDC_DIMENSION)]
        norm = math.sqrt(sum(v * v for v in hv))
        if norm > 0:
            hv = [v / norm for v in hv]

        entropy = hdc_vector_entropy(hv)
        entropies.append(entropy)

    mean_entropy = sum(entropies) / len(entropies)
    min_entropy = min(entropies)
    max_entropy = max(entropies)

    passed = mean_entropy > 0.95

    result = {
        "n_vectors": n_vectors,
        "mean_entropy": round(mean_entropy, 4),
        "min_entropy": round(min_entropy, 4),
        "max_entropy": round(max_entropy, 4),
        "target": 0.95,
        "passed": passed,
    }

    emit_receipt("hdc_entropy_test", result)

    return result


def emit_entropy_receipt(
    entropy_value: float,
    entropy_type: str,
    context: dict | None = None
) -> dict:
    """Emit entropy measurement receipt.

    Args:
        entropy_value: Measured entropy
        entropy_type: Type of entropy (shannon, hdc_vector, class_separation)
        context: Optional context dict

    Returns:
        Receipt dict
    """
    return emit_receipt(
        "entropy_measurement",
        {
            "entropy_value": round(entropy_value, 6),
            "entropy_type": entropy_type,
            "context": context or {},
        }
    )


class EntropyMonitor:
    """Monitor entropy across system components.

    Tracks HDC vector entropy, class separation, and system health.
    """

    def __init__(self):
        self.measurements: list[dict] = []
        self.class_prototypes: dict[str, list[float]] = {}

    def record_vector_entropy(
        self,
        hypervector: list[float],
        label: str = ""
    ) -> float:
        """Record entropy of a hypervector.

        Args:
            hypervector: Vector to measure
            label: Optional label for tracking

        Returns:
            Measured entropy
        """
        entropy = hdc_vector_entropy(hypervector)

        measurement = {
            "type": "vector",
            "label": label,
            "entropy": entropy,
            "dimension": len(hypervector),
        }
        self.measurements.append(measurement)

        return entropy

    def record_class_prototype(
        self,
        class_name: str,
        prototype: list[float]
    ) -> None:
        """Record class prototype for separation tracking.

        Args:
            class_name: Class identifier
            prototype: Prototype hypervector
        """
        self.class_prototypes[class_name] = prototype

    def get_class_separation(self) -> float:
        """Get current class separation metric."""
        return measure_class_separation(self.class_prototypes)

    def get_health_status(self) -> dict:
        """Get overall entropy health status.

        Returns:
            Health status dict with metrics and recommendations
        """
        if not self.measurements:
            return {"status": "no_data", "recommendations": []}

        recent = self.measurements[-100:]  # Last 100 measurements
        mean_entropy = sum(m["entropy"] for m in recent) / len(recent)

        class_sep = self.get_class_separation()

        recommendations = []

        if mean_entropy < 0.8:
            recommendations.append("Vector entropy low - check projection matrix")

        if class_sep < 0.5 and len(self.class_prototypes) > 1:
            recommendations.append("Class separation low - consider prototype refinement")

        status = "healthy" if mean_entropy > 0.9 and class_sep > 0.7 else "degraded"

        return {
            "status": status,
            "mean_vector_entropy": round(mean_entropy, 4),
            "class_separation": round(class_sep, 4),
            "n_measurements": len(self.measurements),
            "n_classes": len(self.class_prototypes),
            "recommendations": recommendations,
        }


if __name__ == "__main__":
    print("RESONANCE PROTOCOL - Entropy Measurement")
    print(f"HDC Dimension: {HDC_DIMENSION}")
    print()
    print("Running entropy test...")
    result = test_hdc_entropy(n_vectors=50)
    print(f"Mean entropy: {result['mean_entropy']:.4f} (target > 0.95)")
    print(f"Min entropy: {result['min_entropy']:.4f}")
    print(f"PASSED: {result['passed']}")
