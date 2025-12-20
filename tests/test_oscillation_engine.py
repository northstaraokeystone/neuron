"""
NEURON v4.5 Gate 1: Oscillation Engine Tests
Test inject_low_alpha, surge_high_alpha, oscillation_cycle, and amplitude measurement.
"""

import os
import tempfile

# Set up isolated test environment BEFORE importing
_test_dir = tempfile.mkdtemp()
os.environ["NEURON_LEDGER"] = os.path.join(_test_dir, "test_receipts.jsonl")
os.environ["NEURON_ARCHIVE"] = os.path.join(_test_dir, "test_archive.jsonl")
os.environ["NEURON_RECEIPTS"] = os.path.join(_test_dir, "test_receipts.jsonl")
os.environ["NEURON_BASE"] = _test_dir

import pytest

from resonance import (
    inject_low_alpha,
    surge_high_alpha,
    oscillation_cycle,
    ResonanceStopRule,
    LOW_ALPHA_INJECTION_RANGE,
    SURGE_ALPHA_CAP,
    INJECTION_COUNT_MAX,
)


@pytest.fixture(autouse=True)
def clean_ledger():
    """Clean test files before and after each test."""
    from pathlib import Path

    for f in Path(_test_dir).glob("*.jsonl"):
        f.unlink(missing_ok=True)

    yield

    for f in Path(_test_dir).glob("*.jsonl"):
        f.unlink(missing_ok=True)


class TestInjectLowAlphaCreatesEntries:
    """Test that inject_low_alpha creates entries in ledger."""

    def test_inject_low_alpha_creates_entries(self):
        """len(ledger) increases by injection_count."""
        ledger = []
        injection_count = 5

        receipt = inject_low_alpha(ledger, injection_count)

        assert len(ledger) == injection_count
        assert receipt["injection_count"] == injection_count
        assert receipt["ledger_size_after"] == injection_count


class TestInjectAlphaRange:
    """Test injected entries have α in specified range."""

    def test_inject_alpha_range(self):
        """Injected entries have α in (0.1, 0.3)."""
        ledger = []
        inject_low_alpha(ledger, 10)

        for entry in ledger:
            alpha = entry.get("alpha", entry.get("salience", 0.5))
            assert LOW_ALPHA_INJECTION_RANGE[0] <= alpha <= LOW_ALPHA_INJECTION_RANGE[1]


class TestSurgeAmplifies:
    """Test surge_high_alpha amplifies high-α entries."""

    def test_surge_amplifies_high_alpha(self):
        """Entries with α > 0.6 increase salience."""
        ledger = [
            {"alpha": 0.8, "salience": 0.8, "task": "high"},
            {"alpha": 0.7, "salience": 0.7, "task": "medium-high"},
            {"alpha": 0.3, "salience": 0.3, "task": "low"},
        ]

        receipt = surge_high_alpha(ledger, surge_multiplier=1.5)

        # First two should be surged (α >= 0.6)
        assert receipt["entries_surged"] == 2
        assert ledger[0]["salience"] > 0.8  # Amplified
        assert ledger[1]["salience"] > 0.7  # Amplified
        assert ledger[2]["salience"] == 0.3  # Unchanged


class TestSurgeCapped:
    """Test surge_high_alpha caps α at 1.0."""

    def test_surge_capped_at_1(self):
        """No α exceeds 1.0 after surge."""
        ledger = [
            {"alpha": 0.9, "salience": 0.9, "task": "very_high"},
            {"alpha": 0.8, "salience": 0.8, "task": "high"},
        ]

        surge_high_alpha(ledger, surge_multiplier=2.0)

        for entry in ledger:
            assert entry["alpha"] <= SURGE_ALPHA_CAP
            assert entry["salience"] <= SURGE_ALPHA_CAP


class TestOscillationCycleCompletes:
    """Test oscillation_cycle returns complete receipt."""

    def test_oscillation_cycle_completes(self):
        """Returns oscillation_receipt with all fields."""
        ledger = [{"alpha": 0.5, "salience": 0.5, "task": str(i)} for i in range(20)]

        receipt = oscillation_cycle(ledger, frequency_hz=0.001, amplitude=0.5)

        assert receipt["receipt_type"] == "oscillation"
        assert "cycle_id" in receipt
        assert "frequency_hz" in receipt
        assert "amplitude" in receipt
        assert "phase" in receipt
        assert "duration_ms" in receipt
        assert "entries_affected" in receipt
        assert "α_swing" in receipt
        assert "payload_hash" in receipt


class TestOscillationProducesSwing:
    """Test oscillation produces measurable α swing."""

    def test_oscillation_produces_swing(self):
        """α_swing > 0.2 after cycle."""
        ledger = [{"alpha": 0.5, "salience": 0.5, "task": str(i)} for i in range(20)]

        receipt = oscillation_cycle(ledger, frequency_hz=0.001, amplitude=0.5)

        # Swing should be > 0.2 due to injected low-α and surged high-α
        assert receipt["α_swing"] > 0.2


class TestAmplitudeStoprule:
    """Test amplitude stoprule triggers when exceeded."""

    def test_amplitude_stoprule(self):
        """amplitude > 1.5 triggers stoprule."""
        from resonance import stoprule_oscillation_divergent

        # Should not raise for valid amplitude
        stoprule_oscillation_divergent(1.0)

        # Should raise for divergent amplitude
        with pytest.raises(ResonanceStopRule) as exc_info:
            stoprule_oscillation_divergent(2.0)

        assert "oscillation_divergent" in str(exc_info.value)


class TestInjectionFloodStoprule:
    """Test injection flood stoprule triggers."""

    def test_injection_flood_stoprule(self):
        """injection_count > 100 triggers stoprule."""
        ledger = []

        with pytest.raises(ResonanceStopRule) as exc_info:
            inject_low_alpha(ledger, INJECTION_COUNT_MAX + 1)

        assert "injection_flood" in str(exc_info.value)
