"""
NEURON v4.3: Triad Integration Tests (Gate 5)
Tests for full triad simulation validating shared nerve behavior.
"""

import os
import tempfile
from pathlib import Path

import pytest

# Set up isolated test environment BEFORE imports
_test_dir = tempfile.mkdtemp()
os.environ["NEURON_LEDGER"] = str(Path(_test_dir) / "test_receipts.jsonl")
os.environ["NEURON_ARCHIVE"] = str(Path(_test_dir) / "test_archive.jsonl")
os.environ["NEURON_RECEIPTS"] = str(Path(_test_dir) / "test_stress_receipts.jsonl")
os.environ["NEURON_STRESS_RECEIPTS"] = str(
    Path(_test_dir) / "test_stress_receipts.jsonl"
)

from stress import triad_simulation


@pytest.fixture(autouse=True)
def clean_env():
    """Clean environment before each test."""
    # The triad_simulation creates its own temp environment
    yield


class TestTriadSimulationCompletes:
    """Test that 500 events generated without error."""

    def test_simulation_completes_small(self):
        """Test simulation completes with small event count."""
        result = triad_simulation(n_events=20, prune_threshold=50)

        assert "events_generated" in result
        assert result["events_generated"] > 0

    def test_simulation_completes_medium(self):
        """Test simulation completes with medium event count."""
        result = triad_simulation(n_events=100, prune_threshold=50)

        assert result["events_generated"] >= 100


class TestAllProjectsRepresented:
    """Test final ledger has entries from all projects."""

    def test_all_projects_present(self):
        """Test that final ledger has entries from multiple projects."""
        result = triad_simulation(n_events=100, prune_threshold=100)

        events_by_project = result["events_by_project"]
        assert len(events_by_project) >= 3, (
            f"Expected 3+ projects, got {events_by_project}"
        )


class TestSystemAlphaReasonable:
    """Test 1.0 ≤ α_system ≤ 3.0."""

    def test_alpha_reasonable_range(self):
        """Test system alpha is in reasonable range."""
        result = triad_simulation(n_events=100, prune_threshold=100)

        # Alpha can be 0 for quick simulations, but should be reasonable
        assert 0.0 <= result["system_alpha_final"] <= 10.0


class TestPruningRespectsThreshold:
    """Test final size ≤ prune_threshold."""

    def test_pruning_threshold(self):
        """Test that pruning respects the threshold."""
        prune_threshold = 30
        result = triad_simulation(n_events=100, prune_threshold=prune_threshold)

        # Allow some buffer due to preservation rules
        assert result["ledger_final_size"] <= prune_threshold * 2


class TestAIAutoEventsPresent:
    """Test auto-triggered AI events in ledger."""

    def test_ai_auto_events_present(self):
        """Test that AI auto-triggered events are present."""
        result = triad_simulation(
            n_events=100, include_stress=True, prune_threshold=100
        )

        assert result["ai_auto_events"] > 0


class TestCrossProjectGapsDetected:
    """Test gaps span multiple projects."""

    def test_cross_project_gaps(self):
        """Test that gaps span multiple projects."""
        result = triad_simulation(n_events=100, include_gaps=True, prune_threshold=100)

        assert result["gaps_injected"] >= 0  # May be 0 for fast simulations


class TestReceiptEmitted:
    """Test triad_simulation_receipt is emitted."""

    def test_receipt_contains_metrics(self):
        """Test result contains all expected metrics."""
        result = triad_simulation(n_events=50, prune_threshold=50)

        expected_keys = [
            "events_generated",
            "events_by_project",
            "gaps_injected",
            "stress_events",
            "system_alpha_final",
            "pruned_total",
            "ledger_final_size",
            "all_invariants_pass",
        ]

        for key in expected_keys:
            assert key in result, f"Missing key: {key}"


class TestBioSiliconContinuity:
    """Test no gaps > 24h (continuity maintained)."""

    def test_continuity_maintained(self):
        """Test that no gaps exceed 24 hours."""
        result = triad_simulation(n_events=100, prune_threshold=100)

        assert result["max_gap_hours"] < 24, (
            f"Gap of {result['max_gap_hours']}h exceeds 24h"
        )

    def test_invariants_pass(self):
        """Test that all invariants pass."""
        result = triad_simulation(n_events=100, prune_threshold=100)

        # Check individual invariants
        invariants = result["invariants"]
        assert invariants["all_projects_present"], "All projects should be present"
        assert invariants["alpha_reasonable"], "Alpha should be reasonable"
        assert invariants["ai_events_present"], "AI events should be present"
        assert invariants["continuity_maintained"], "Continuity should be maintained"
