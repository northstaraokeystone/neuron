"""
RESONANCE PROTOCOL - Test Configuration

Pytest fixtures and configuration for the test suite.
"""

import os
import sys
import tempfile
from pathlib import Path

import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Set up test environment
os.environ["RESONANCE_BASE"] = tempfile.mkdtemp()


@pytest.fixture
def temp_receipts_dir(tmp_path):
    """Create temporary directory for receipts."""
    os.environ["RESONANCE_BASE"] = str(tmp_path)
    return tmp_path


@pytest.fixture
def sample_spike_train():
    """Generate sample spike train for testing."""
    import random
    random.seed(42)
    return [random.random() * 0.1 for _ in range(100)]


@pytest.fixture
def sample_lfp_signal():
    """Generate sample LFP signal for testing."""
    import math
    fs = 20000
    duration_ms = 100
    n_samples = int(fs * duration_ms / 1000)

    signal = []
    for i in range(n_samples):
        t = i / fs
        # 200 Hz ripple + noise
        ripple = 0.5 * math.sin(2 * math.pi * 200 * t)
        noise = 0.1 * math.sin(2 * math.pi * 50 * t + 0.5)
        signal.append(ripple + noise)

    return signal


@pytest.fixture
def sample_projection_matrix():
    """Generate sample projection matrix for HDC testing."""
    from src.hdc import create_random_projection
    return create_random_projection(n_channels=100, hdc_dim=1000, seed=42)


@pytest.fixture
def sample_model():
    """Generate sample model for federated learning testing."""
    return {
        "layer1": [0.1, -0.2, 0.15, 0.05, -0.1],
        "layer2": [0.01] * 10,
    }
