"""
NEURON v4.5 Gate 3: Gap Resonance Driver Tests
Test gap_resonance_trigger and amplitude boosting.
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

from neuron import (
    gap_resonance_trigger,
    reset_oscillation_state,
    GAP_AMPLITUDE_BOOST,
)


@pytest.fixture(autouse=True)
def clean_state():
    """Reset oscillation state before and after each test."""
    from neuron import _get_ledger_path, _get_receipts_path

    reset_oscillation_state()

    for path_fn in [_get_ledger_path, _get_receipts_path]:
        path = path_fn()
        if path.exists():
            path.unlink()

    yield

    reset_oscillation_state()

    for path_fn in [_get_ledger_path, _get_receipts_path]:
        path = path_fn()
        if path.exists():
            path.unlink()


class TestGapTriggersOscillation:
    """Test gap triggers oscillation cycle."""

    def test_gap_triggers_oscillation(self):
        """gap → resonance_driver_receipt."""
        gap = {"source": "human", "duration_minutes": 30}

        result = gap_resonance_trigger(gap)

        assert result["receipt_type"] == "resonance_driver"
        assert "triggered_cycle_id" in result
        assert result["triggered_cycle_id"].startswith("gap_osc_")


class TestGapAmplitudeBoost:
    """Test gap boosts amplitude."""

    def test_gap_amplitude_boost(self):
        """amplitude increases by GAP_AMPLITUDE_BOOST."""
        gap = {"source": "grok", "duration_minutes": 60}

        result = gap_resonance_trigger(gap)

        assert result["amplitude_boost"] == GAP_AMPLITUDE_BOOST


class TestHumanGapRecognized:
    """Test human gap source is logged."""

    def test_human_gap_recognized(self):
        """source='human' logged."""
        gap = {"source": "human", "duration_minutes": 45}

        result = gap_resonance_trigger(gap)

        assert result["gap_source"] == "human"


class TestGrokGapRecognized:
    """Test grok gap source is logged."""

    def test_grok_gap_recognized(self):
        """source='grok' logged."""
        gap = {"source": "grok", "duration_minutes": 120}

        result = gap_resonance_trigger(gap)

        assert result["gap_source"] == "grok"


class TestAgentproofGapRecognized:
    """Test agentproof gap source is logged."""

    def test_agentproof_gap_recognized(self):
        """source='agentproof' logged."""
        gap = {"source": "agentproof", "duration_minutes": 90}

        result = gap_resonance_trigger(gap)

        assert result["gap_source"] == "agentproof"


class TestAxiomGapRecognized:
    """Test axiom gap source is logged."""

    def test_axiom_gap_recognized(self):
        """source='axiom' logged."""
        gap = {"source": "axiom", "duration_minutes": 180}

        result = gap_resonance_trigger(gap)

        assert result["gap_source"] == "axiom"


class TestGapLogsDuration:
    """Test gap logs duration."""

    def test_gap_logs_duration(self):
        """duration_minutes in receipt."""
        gap = {"source": "human", "duration_minutes": 75.5}

        result = gap_resonance_trigger(gap)

        assert result["gap_duration_minutes"] == 75.5
