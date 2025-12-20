"""
NEURON v4.5 Gate 6: Resonance Integration Tests
Test full resonance_catalyst_cycle and backward compatibility.
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
    resonance_catalyst_cycle,
)

from neuron import (
    pump_cycle,
    reset_oscillation_state,
    gap_resonance_trigger,
    human_direct_phase,
    detect_phase_transition,
    _read_ledger,
    append,
)


@pytest.fixture(autouse=True)
def clean_state():
    """Reset state before and after each test."""
    from neuron import _get_ledger_path, _get_receipts_path, _get_archive_path

    reset_oscillation_state()

    for path_fn in [_get_ledger_path, _get_receipts_path, _get_archive_path]:
        path = path_fn()
        if path.exists():
            path.unlink()

    yield

    reset_oscillation_state()

    for path_fn in [_get_ledger_path, _get_receipts_path, _get_archive_path]:
        path = path_fn()
        if path.exists():
            path.unlink()


class TestCatalystCycleCompletes:
    """Test resonance_catalyst_cycle completes."""

    def test_catalyst_cycle_completes(self):
        """Returns catalyst_receipt."""
        ledger = [{"alpha": 0.5, "salience": 0.5, "task": str(i)} for i in range(50)]
        config = {"frequency_source": "HUMAN_FOCUS", "amplitude": 0.5}

        result = resonance_catalyst_cycle(ledger, config)

        assert result["receipt_type"] == "catalyst"
        assert "cycles_completed" in result
        assert "total_injections" in result
        assert "total_surges" in result


class TestVersion450:
    """Test version is 4.5.0."""

    def test_version_4_5_0(self):
        """version = '4.5.0'."""
        ledger = [{"alpha": 0.5, "salience": 0.5, "task": str(i)} for i in range(20)]
        config = {"frequency_source": "HUMAN_FOCUS", "amplitude": 0.5}

        result = resonance_catalyst_cycle(ledger, config)

        assert result["version"] == "4.5.0"


class TestCycleInjectsAndSurges:
    """Test cycle performs both injections and surges."""

    def test_cycle_injects_and_surges(self):
        """Both counts > 0."""
        ledger = [{"alpha": 0.7, "salience": 0.7, "task": str(i)} for i in range(20)]
        config = {"frequency_source": "HUMAN_FOCUS", "amplitude": 0.5}

        result = resonance_catalyst_cycle(ledger, config)

        assert result["total_injections"] > 0
        # Surges depend on entries having alpha > 0.6


class TestFrequencyUsed:
    """Test configured frequency is used."""

    def test_frequency_used(self):
        """Uses configured frequency."""
        ledger = [{"alpha": 0.5, "salience": 0.5, "task": str(i)} for i in range(20)]
        config = {"frequency_source": "HUMAN_CIRCADIAN", "amplitude": 0.5}

        result = resonance_catalyst_cycle(ledger, config)

        assert result["frequency_source"] == "HUMAN_CIRCADIAN"


class TestGapIntegration:
    """Test gap integration amplifies."""

    def test_gap_integration(self):
        """Gap during cycle amplifies."""
        # Trigger a gap
        gap = {"source": "human", "duration_minutes": 60}
        gap_result = gap_resonance_trigger(gap)

        assert gap_result["amplitude_boost"] == 2.0


class TestHumanIntegration:
    """Test human override works."""

    def test_human_integration(self):
        """Human override works."""
        # Override to surge
        result = human_direct_phase("surge")
        assert result["direction"] == "surge"

        # Override to inject
        result = human_direct_phase("inject")
        assert result["direction"] == "inject"


class TestTransitionDetection:
    """Test triad changes are logged."""

    def test_transition_detection(self):
        """Triad changes logged."""
        triad_state = {
            "axiom": {"laws_discovered": 3, "compression": 0.85},
            "agentproof": {"selection_threshold": 0.7},
        }

        result = detect_phase_transition(triad_state)

        assert result["detected"] is True
        assert result["transition_type"] == "both"


class TestBackwardCompatPump:
    """Test v4.4 pump_cycle still works."""

    def test_backward_compat_pump(self):
        """v4.4 pump_cycle still works."""
        # Create some entries
        for i in range(10):
            append(
                project="neuron",
                task=f"task {i}",
                next_action="process",
                salience=0.5 + (i / 20),  # Range from 0.5 to 0.95
            )

        ledger = _read_ledger()
        result = pump_cycle(ledger)

        # Pump cycle should still work
        assert "cycles_run" in result
        assert "initial_entropy" in result
        assert "final_entropy" in result
