"""
NEURON v4.5 Gate 5: Phase Transition Detection Tests
Test detect_phase_transition and triad state monitoring.
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
    detect_phase_transition,
    reset_oscillation_state,
)


@pytest.fixture(autouse=True)
def clean_state():
    """Reset state before and after each test."""
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


class TestAxiomLawTransition:
    """Test AXIOM law discovery detection."""

    def test_axiom_law_transition(self):
        """Detects law_discovery event."""
        triad_state = {
            "axiom": {"laws_discovered": 5, "compression": 0.92},
            "agentproof": {"selection_threshold": 0.3},  # Below threshold
        }

        result = detect_phase_transition(triad_state)

        assert result["transition_type"] == "axiom_law"
        assert result["detected"] is True


class TestAgentproofSelectionTransition:
    """Test AgentProof selection detection."""

    def test_agentproof_selection_transition(self):
        """Detects selection_threshold change."""
        triad_state = {
            "axiom": {"laws_discovered": 0, "compression": 0},
            "agentproof": {"selection_threshold": 0.85},  # Above 0.5 threshold
        }

        result = detect_phase_transition(triad_state)

        assert result["transition_type"] == "agentproof_selection"
        assert result["detected"] is True


class TestBothTransition:
    """Test both transitions detected."""

    def test_both_transition(self):
        """Detects simultaneous changes."""
        triad_state = {
            "axiom": {"laws_discovered": 3, "compression": 0.9},
            "agentproof": {"selection_threshold": 0.75},
        }

        result = detect_phase_transition(triad_state)

        assert result["transition_type"] == "both"
        assert result["detected"] is True


class TestNoTransition:
    """Test no transition when no changes."""

    def test_no_transition(self):
        """Returns None/empty when no change."""
        triad_state = {
            "axiom": {"laws_discovered": 0, "compression": 0},
            "agentproof": {"selection_threshold": 0.3},
        }

        result = detect_phase_transition(triad_state)

        assert result["transition_type"] is None
        assert result["detected"] is False


class TestCorrelationThreshold:
    """Test correlation threshold is calculated."""

    def test_correlation_threshold(self):
        """Only counts if correlation > 0.7."""
        triad_state = {
            "axiom": {"laws_discovered": 5, "compression": 0.92},
            "agentproof": {"selection_threshold": 0.85},
        }

        result = detect_phase_transition(triad_state)

        # Correlation is calculated based on oscillation amplitude
        assert "oscillation_correlation" in result
        assert isinstance(result["oscillation_correlation"], float)


class TestTransitionReceipt:
    """Test phase_transition_receipt is returned."""

    def test_transition_receipt(self):
        """Returns phase_transition_receipt."""
        triad_state = {
            "axiom": {"laws_discovered": 2, "compression": 0.8},
            "agentproof": {"selection_threshold": 0.6},
        }

        result = detect_phase_transition(triad_state)

        assert result["receipt_type"] == "phase_transition"
        assert "ts" in result
        assert "hash" in result


class TestBeforeAfterState:
    """Test before and after state are logged."""

    def test_before_after_state(self):
        """Logs before_state and after_state."""
        triad_state = {
            "axiom": {"laws_discovered": 5, "compression": 0.92},
            "agentproof": {"selection_threshold": 0.85},
        }

        result = detect_phase_transition(triad_state)

        assert "before_state" in result
        assert "after_state" in result
        assert result["after_state"]["axiom_laws"] == 5
        assert result["after_state"]["agentproof_threshold"] == 0.85
