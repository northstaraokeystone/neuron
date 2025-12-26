"""
NEURON-RESONANCE v5.0 Haptic Feedback Tests
Test pre-linguistic cortical stimulation for high-salience retrieval.
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

import pytest
from pathlib import Path

from haptic_feedback import (
    compute_salience,
    validate_stim_intensity,
    deliver_stimulation,
    vector_to_text,
    haptic_feedback_loop,
    simulate_retrieval_for_test,
    HapticStopRule,
    URGENCY_THRESHOLD,
    STIM_DURATION_MS,
    STIM_INTENSITY_MAX,
    STIM_REGIONS,
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
# SALIENCE COMPUTATION TESTS
# ============================================


class TestComputeSalience:
    """Test salience computation from vector and retrieval score."""

    def test_high_retrieval_high_salience(self):
        """High retrieval score → high salience."""
        vector = [0.1] * 100
        vector = [v / math.sqrt(sum(x**2 for x in vector)) for v in vector]

        salience = compute_salience(vector, 0.9)

        assert salience >= 0.5

    def test_zero_retrieval_low_salience(self):
        """Zero retrieval score → low salience."""
        vector = [0.1] * 100

        salience = compute_salience(vector, 0.0)

        assert salience < 0.5

    def test_empty_vector_zero_salience(self):
        """Empty vector → zero salience."""
        salience = compute_salience([], 0.9)

        assert salience == 0.0


# ============================================
# STIMULATION INTENSITY TESTS
# ============================================


class TestValidateStimIntensity:
    """Test stimulation intensity validation."""

    def test_intensity_clamped(self):
        """Input intensity > max → clamped to max."""
        result = validate_stim_intensity(1.5, 1.0)

        assert result == 1.0

    def test_intensity_passed_through(self):
        """Valid intensity passes through unchanged."""
        result = validate_stim_intensity(0.5, 1.0)

        assert result == 0.5

    def test_stoprule_on_dangerous_intensity(self):
        """intensity > 2x max → StopRule."""
        with pytest.raises(HapticStopRule) as exc_info:
            validate_stim_intensity(2.5, 1.0)

        assert "dangerous_intensity" in str(exc_info.value)

    def test_negative_intensity_clamped(self):
        """Negative intensity clamped to 0."""
        result = validate_stim_intensity(-0.5, 1.0)

        assert result == 0.0


# ============================================
# STIMULATION DELIVERY TESTS
# ============================================


class TestDeliverStimulation:
    """Test stimulation delivery."""

    def test_simulation_mode_not_delivered(self):
        """Simulation mode → not actually delivered."""
        pattern = [0.5] * 100

        result = deliver_stimulation(
            pattern,
            STIM_REGIONS,
            0.5,
            STIM_DURATION_MS,
            simulation=True,
        )

        assert result["delivered"] is False
        assert result["simulation_mode"] is True

    def test_stimulate_receipt_emitted(self):
        """All fields present, dual-hash valid."""
        pattern = [0.5] * 100

        result = deliver_stimulation(
            pattern,
            STIM_REGIONS,
            0.5,
            STIM_DURATION_MS,
            simulation=True,
        )

        assert "_receipt" in result
        receipt = result["_receipt"]
        assert receipt["type"] == "stimulate"
        assert "pattern_hash" in receipt
        assert "regions" in receipt
        assert "intensity" in receipt
        assert "duration_ms" in receipt
        assert ":" in receipt["hash"]  # Dual hash


# ============================================
# HAPTIC FEEDBACK LOOP TESTS
# ============================================


class TestHapticFeedbackLoop:
    """Test full haptic feedback pipeline."""

    def test_high_salience_triggers_haptic(self):
        """salience > threshold → haptic_delivered=True."""
        config = {
            "urgency_threshold": 0.8,
            "stim_duration_ms": 200,
            "stim_intensity_max": 1.0,
            "stim_regions": STIM_REGIONS,
            "simulation_mode": True,
        }

        vector, salience = simulate_retrieval_for_test(1000, high_salience=True)
        ledger = [{"task": "test", "next": "next", "salience": 0.9}]

        result = haptic_feedback_loop(vector, 0.9, config, ledger)

        assert result["haptic_delivered"] is True
        assert result["text_delivered"] is True

    def test_haptic_before_text(self):
        """When haptic delivered, haptic_ts < text_ts."""
        config = {
            "urgency_threshold": 0.8,
            "stim_duration_ms": 200,
            "stim_intensity_max": 1.0,
            "stim_regions": STIM_REGIONS,
            "simulation_mode": True,
        }

        vector, _ = simulate_retrieval_for_test(1000, high_salience=True)
        ledger = [{"task": "test", "next": "next", "salience": 0.9}]

        result = haptic_feedback_loop(vector, 0.9, config, ledger)

        if result["haptic_delivered"]:
            assert result["haptic_before_text"] is True
            assert result["haptic_ts"] < result["text_ts"]

    def test_low_salience_text_only(self):
        """salience < threshold → haptic_delivered=False, text_delivered=True."""
        config = {
            "urgency_threshold": 0.8,
            "stim_duration_ms": 200,
            "stim_intensity_max": 1.0,
            "stim_regions": STIM_REGIONS,
            "simulation_mode": True,
        }

        vector, _ = simulate_retrieval_for_test(1000, high_salience=False)
        ledger = [{"task": "test", "next": "next"}]

        result = haptic_feedback_loop(vector, 0.5, config, ledger)

        assert result["haptic_delivered"] is False
        assert result["text_delivered"] is True

    def test_haptic_feedback_receipt_emitted(self):
        """Haptic feedback emits receipt."""
        config = {
            "urgency_threshold": 0.8,
            "stim_duration_ms": 200,
            "stim_intensity_max": 1.0,
            "stim_regions": STIM_REGIONS,
            "simulation_mode": True,
        }

        vector, _ = simulate_retrieval_for_test(1000, high_salience=True)
        ledger = []

        result = haptic_feedback_loop(vector, 0.9, config, ledger)

        assert "_receipt" in result
        receipt = result["_receipt"]
        assert receipt["type"] == "haptic_feedback"
        assert "salience" in receipt
        assert "urgency_threshold" in receipt
        assert "haptic_delivered" in receipt
        assert "text_delivered" in receipt
        assert "haptic_before_text" in receipt


# ============================================
# VECTOR TO TEXT TESTS
# ============================================


class TestVectorToText:
    """Test SDM vector to text conversion."""

    def test_vector_to_text_returns_string(self):
        """Vector to text returns concatenated task strings."""
        vector = [0.1] * 100
        ledger = [
            {"task": "first task", "next": "do something"},
            {"task": "second task", "next": "do more"},
        ]

        text = vector_to_text(vector, ledger, top_k=5)

        assert isinstance(text, str)
        assert "first task" in text or "second task" in text

    def test_empty_ledger_returns_empty(self):
        """Empty ledger returns empty string."""
        vector = [0.1] * 100

        text = vector_to_text(vector, [])

        assert text == ""


# ============================================
# CONSTANTS VERIFICATION TESTS
# ============================================


class TestConstants:
    """Test v5.0 haptic constants are correct."""

    def test_urgency_threshold(self):
        """URGENCY_THRESHOLD is 0.8."""
        assert URGENCY_THRESHOLD == 0.8

    def test_stim_duration_ms(self):
        """STIM_DURATION_MS is 200."""
        assert STIM_DURATION_MS == 200

    def test_stim_intensity_max(self):
        """STIM_INTENSITY_MAX is 1.0."""
        assert STIM_INTENSITY_MAX == 1.0

    def test_stim_regions(self):
        """STIM_REGIONS contains expected regions."""
        assert "insula" in STIM_REGIONS
        assert "amygdala" in STIM_REGIONS
        assert "somatosensory_cortex" in STIM_REGIONS
