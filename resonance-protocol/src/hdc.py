"""
RESONANCE PROTOCOL - Hyperdimensional Computing Encoder

Implements HDC encoder with dropout robustness for neural decoding.
10,000-bit hypervectors maintain classification accuracy with 30% channel loss.

THE INSIGHT:
    HDC's holographic robustness is not optimization - it's physical necessity.
    When threads retract, when 30% of channels are lost, the system degrades
    gracefully instead of catastrophically.

Functions:
    create_random_projection: Generate fixed random projection matrix
    encode_spike_window: Map spike counts to hypervector
    bundle_temporal: Weighted sum of hypervectors over time
    cosine_similarity: Similarity between two hypervectors
    classify_intent: Return class with highest similarity
    test_orthogonality: Verify random vector orthogonality
    test_dropout: Verify dropout robustness
"""

from __future__ import annotations

import math
import random
from typing import Callable

try:
    from .core import (
        HDC_DIMENSION,
        N1_CHANNELS,
        CHANNEL_DROPOUT_TOLERANCE,
        MIN_CLASSIFICATION_ACCURACY,
        ORTHOGONALITY_THRESHOLD,
        emit_receipt,
        dual_hash,
    )
except ImportError:
    from core import (
        HDC_DIMENSION,
        N1_CHANNELS,
        CHANNEL_DROPOUT_TOLERANCE,
        MIN_CLASSIFICATION_ACCURACY,
        ORTHOGONALITY_THRESHOLD,
        emit_receipt,
        dual_hash,
    )

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


def create_random_projection(
    n_channels: int = N1_CHANNELS,
    hdc_dim: int = HDC_DIMENSION,
    seed: int = 42
) -> list[list[float]]:
    """Generate fixed random projection matrix (n_channels x hdc_dim).

    Matrix is fixed post-initialization (no retraining required).
    Seed ensures reproducibility across sessions.

    Args:
        n_channels: Number of input channels (default N1_CHANNELS=1024)
        hdc_dim: Output hypervector dimension (default HDC_DIMENSION=10000)
        seed: Random seed for deterministic generation

    Returns:
        Projection matrix as nested list [n_channels][hdc_dim]
    """
    random.seed(seed)

    # Scale for unit variance output: 1/sqrt(n_channels)
    scale = 1.0 / math.sqrt(n_channels)

    matrix = []
    for _ in range(n_channels):
        row = [random.gauss(0, scale) for _ in range(hdc_dim)]
        matrix.append(row)

    return matrix


def encode_spike_window(
    spike_counts: list[float] | list[int],
    projection: list[list[float]],
    dropout_mask: list[int] | None = None
) -> list[float]:
    """Map spike counts to hypervector with optional dropout.

    Args:
        spike_counts: Spike counts per channel [n_channels]
        projection: Random projection matrix [n_channels][hdc_dim]
        dropout_mask: Binary mask (0/1 per channel), None = no dropout

    Returns:
        Normalized hypervector [hdc_dim]
    """
    n_channels = len(spike_counts)
    hdc_dim = len(projection[0]) if projection else HDC_DIMENSION

    if len(projection) != n_channels:
        raise ValueError(f"Projection rows {len(projection)} != channels {n_channels}")

    # Apply dropout mask if provided
    if dropout_mask is not None:
        spike_counts = [s * m for s, m in zip(spike_counts, dropout_mask)]

    # Project: hypervector = sum(spike_counts[i] * projection[i])
    hypervector = [0.0] * hdc_dim
    for ch in range(n_channels):
        if spike_counts[ch] != 0:
            for d in range(hdc_dim):
                hypervector[d] += spike_counts[ch] * projection[ch][d]

    # Normalize to unit hypersphere
    norm = math.sqrt(sum(v * v for v in hypervector))
    if norm > 0:
        hypervector = [v / norm for v in hypervector]

    return hypervector


def bundle_temporal(
    hypervectors: list[list[float]],
    weights: list[float] | None = None
) -> list[float]:
    """Weighted sum of hypervectors over time windows.

    Args:
        hypervectors: List of hypervectors to bundle
        weights: Optional weights per hypervector (default: uniform)

    Returns:
        Bundled and normalized hypervector
    """
    if not hypervectors:
        return []

    hdc_dim = len(hypervectors[0])
    n = len(hypervectors)

    if weights is None:
        weights = [1.0 / n] * n
    else:
        # Normalize weights
        total = sum(weights)
        if total > 0:
            weights = [w / total for w in weights]

    # Weighted sum
    bundled = [0.0] * hdc_dim
    for hv, w in zip(hypervectors, weights):
        for d in range(hdc_dim):
            bundled[d] += w * hv[d]

    # Normalize to unit hypersphere
    norm = math.sqrt(sum(v * v for v in bundled))
    if norm > 0:
        bundled = [v / norm for v in bundled]

    return bundled


def cosine_similarity(hv1: list[float], hv2: list[float]) -> float:
    """Return cosine similarity between two hypervectors.

    Args:
        hv1: First hypervector
        hv2: Second hypervector

    Returns:
        Cosine similarity [-1, 1]
    """
    if len(hv1) != len(hv2):
        raise ValueError(f"Dimension mismatch: {len(hv1)} vs {len(hv2)}")

    dot = sum(a * b for a, b in zip(hv1, hv2))
    norm1 = math.sqrt(sum(v * v for v in hv1))
    norm2 = math.sqrt(sum(v * v for v in hv2))

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot / (norm1 * norm2)


def classify_intent(
    query_hv: list[float],
    prototypes: dict[str, list[float]]
) -> str:
    """Return class with highest similarity to query hypervector.

    Args:
        query_hv: Query hypervector
        prototypes: Dict {class_name: prototype_hypervector}

    Returns:
        Class name with highest similarity
    """
    if not prototypes:
        return ""

    best_class = ""
    best_sim = float('-inf')

    for class_name, proto_hv in prototypes.items():
        sim = cosine_similarity(query_hv, proto_hv)
        if sim > best_sim:
            best_sim = sim
            best_class = class_name

    return best_class


def test_orthogonality(n_vectors: int = 1000) -> float:
    """Generate n random hypervectors, compute pairwise similarities.

    PASS CRITERIA: Mean similarity < 0.1

    Args:
        n_vectors: Number of random vectors to generate

    Returns:
        Mean cosine similarity (should be < 0.1)
    """
    random.seed(42)

    # Generate random unit hypervectors
    vectors = []
    for _ in range(n_vectors):
        hv = [random.gauss(0, 1) for _ in range(HDC_DIMENSION)]
        norm = math.sqrt(sum(v * v for v in hv))
        if norm > 0:
            hv = [v / norm for v in hv]
        vectors.append(hv)

    # Compute pairwise similarities (sample for efficiency)
    n_pairs = min(1000, n_vectors * (n_vectors - 1) // 2)
    similarities = []

    for _ in range(n_pairs):
        i = random.randint(0, n_vectors - 1)
        j = random.randint(0, n_vectors - 1)
        if i != j:
            sim = abs(cosine_similarity(vectors[i], vectors[j]))
            similarities.append(sim)

    mean_sim = sum(similarities) / len(similarities) if similarities else 0.0

    # Emit test receipt
    emit_receipt(
        "hdc_orthogonality_test",
        {
            "n_vectors": n_vectors,
            "n_pairs_tested": len(similarities),
            "mean_similarity": round(mean_sim, 6),
            "threshold": ORTHOGONALITY_THRESHOLD,
            "passed": mean_sim < ORTHOGONALITY_THRESHOLD,
        }
    )

    return mean_sim


def test_dropout(dropout_rate: float = CHANNEL_DROPOUT_TOLERANCE) -> float:
    """Encode test data with dropout, measure classification accuracy.

    PASS CRITERIA: Classification accuracy > 0.85 with 30% dropout

    Args:
        dropout_rate: Fraction of channels to drop (default 0.30)

    Returns:
        Classification accuracy (target > 0.85)
    """
    random.seed(42)
    n_classes = 5
    n_samples_per_class = 50
    n_channels = N1_CHANNELS

    # Create projection matrix
    projection = create_random_projection(n_channels, HDC_DIMENSION, seed=42)

    # Generate synthetic prototypes (class templates)
    prototypes = {}
    for c in range(n_classes):
        class_name = f"intent_{c}"
        # Each class has characteristic firing pattern
        spike_pattern = [0.0] * n_channels
        start_ch = c * (n_channels // n_classes)
        end_ch = start_ch + (n_channels // n_classes)
        for ch in range(start_ch, end_ch):
            spike_pattern[ch] = random.random() * 10

        proto_hv = encode_spike_window(spike_pattern, projection)
        prototypes[class_name] = proto_hv

    # Test classification with dropout
    correct = 0
    total = 0

    for c in range(n_classes):
        class_name = f"intent_{c}"
        start_ch = c * (n_channels // n_classes)
        end_ch = start_ch + (n_channels // n_classes)

        for _ in range(n_samples_per_class):
            # Generate sample with noise
            spike_counts = [0.0] * n_channels
            for ch in range(start_ch, end_ch):
                spike_counts[ch] = random.random() * 10 + random.gauss(0, 1)

            # Generate dropout mask
            dropout_mask = [1 if random.random() > dropout_rate else 0 for _ in range(n_channels)]

            # Encode with dropout
            query_hv = encode_spike_window(spike_counts, projection, dropout_mask)

            # Classify
            predicted = classify_intent(query_hv, prototypes)

            if predicted == class_name:
                correct += 1
            total += 1

    accuracy = correct / total if total > 0 else 0.0

    # Emit test receipt
    emit_receipt(
        "hdc_dropout_test",
        {
            "dropout_rate": dropout_rate,
            "n_classes": n_classes,
            "n_samples": total,
            "correct": correct,
            "accuracy": round(accuracy, 4),
            "threshold": MIN_CLASSIFICATION_ACCURACY,
            "passed": accuracy > MIN_CLASSIFICATION_ACCURACY,
        }
    )

    return accuracy


def emit_encoding_receipt(
    input_channels: int,
    hypervector_dim: int,
    encoding_method: str,
    dropout_mask: list[int] | None,
    vector_hash: str
) -> dict:
    """Emit hdc_encoding_receipt per CLAUDEME.

    Args:
        input_channels: Number of input channels
        hypervector_dim: Output hypervector dimension
        encoding_method: Encoding method name
        dropout_mask: Binary dropout mask (or None)
        vector_hash: Hash of output hypervector

    Returns:
        Receipt dict
    """
    dropout_rate = 0.0
    if dropout_mask:
        dropout_rate = 1.0 - (sum(dropout_mask) / len(dropout_mask))

    return emit_receipt(
        "hdc_encoding",
        {
            "input_channels": input_channels,
            "hypervector_dim": hypervector_dim,
            "encoding_method": encoding_method,
            "dropout_rate": round(dropout_rate, 4),
            "vector_hash": vector_hash[:32] if len(vector_hash) > 32 else vector_hash,
        }
    )


if __name__ == "__main__":
    print("RESONANCE PROTOCOL - HDC Encoder")
    print(f"HDC Dimension: {HDC_DIMENSION}")
    print(f"N1 Channels: {N1_CHANNELS}")
    print(f"Dropout Tolerance: {CHANNEL_DROPOUT_TOLERANCE}")
    print()
    print("Testing orthogonality...")
    mean_sim = test_orthogonality(n_vectors=100)
    print(f"Mean similarity: {mean_sim:.4f} (target < {ORTHOGONALITY_THRESHOLD})")
    print()
    print("Testing dropout robustness...")
    accuracy = test_dropout(dropout_rate=0.30)
    print(f"Accuracy: {accuracy:.2%} (target > {MIN_CLASSIFICATION_ACCURACY:.0%})")
