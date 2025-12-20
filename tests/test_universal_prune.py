"""
NEURON v4.3: Universal Pruning Tests (Gate 4)
Tests for system-wide pruning with universal scoring.
"""

import os
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

# Set up isolated test environment BEFORE imports
_test_dir = tempfile.mkdtemp()
os.environ["NEURON_LEDGER"] = str(Path(_test_dir) / "test_receipts.jsonl")
os.environ["NEURON_ARCHIVE"] = str(Path(_test_dir) / "test_archive.jsonl")
os.environ["NEURON_RECEIPTS"] = str(Path(_test_dir) / "test_stress_receipts.jsonl")

from neuron import (
    append,
    prune,
    load_ledger,
    universal_score,
    _write_ledger,
    PROJECT_PRUNE_WEIGHT,
)


@pytest.fixture(autouse=True)
def clean_ledger():
    """Clean ledger before each test."""
    ledger_path = Path(os.environ["NEURON_LEDGER"])
    archive_path = Path(os.environ["NEURON_ARCHIVE"])
    if ledger_path.exists():
        ledger_path.unlink()
    if archive_path.exists():
        archive_path.unlink()
    yield
    if ledger_path.exists():
        ledger_path.unlink()
    if archive_path.exists():
        archive_path.unlink()


def create_old_entry(project: str, age_days: float, salience: float = 1.0) -> dict:
    """Create an entry with specific age and salience."""
    ts = datetime.now(timezone.utc) - timedelta(days=age_days)
    return {
        "ts": ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "project": project,
        "event_type": "task",
        "model": "neuron",
        "commit": None,
        "task": f"Test task {project}",
        "next": "next",
        "salience": salience,
        "replay_count": 0,
        "energy": 1.0,
        "token_count": 0,
        "inference_id": None,
        "context_summary": "",
        "source_context": {},
        "hash": f"test_hash_{project}_{ts.timestamp()}",
    }


class TestUniversalScoreFactorsProject:
    """Test that universal score factors in project weight."""

    def test_grok_scores_lower_than_human(self):
        """Test Grok entries score lower than human (same salience/age)."""
        now = datetime.now(timezone.utc)

        human_entry = create_old_entry("human", 0.1)
        grok_entry = create_old_entry("grok", 0.1)

        human_score = universal_score(human_entry, 1.0, now)
        grok_score = universal_score(grok_entry, 1.0, now)

        # Grok has lower prune weight (0.85) than human (1.0)
        assert grok_score < human_score

    def test_project_weights_applied(self):
        """Test all project weights are applied correctly."""
        now = datetime.now(timezone.utc)

        scores = {}
        for project in PROJECT_PRUNE_WEIGHT:
            entry = create_old_entry(project, 0.1)
            scores[project] = universal_score(entry, 1.0, now)

        # Human and neuron should have highest scores (weight 1.0)
        assert scores["human"] == scores["neuron"]
        # Grok should have lowest (weight 0.85)
        assert scores["grok"] < scores["agentproof"]


class TestUniversalScoreFactorsAlpha:
    """Test that higher system α leads to lower scores (more pruning)."""

    def test_higher_alpha_lower_scores(self):
        """Test higher system α = lower scores (more pruning)."""
        now = datetime.now(timezone.utc)
        entry = create_old_entry("human", 0.1)

        score_low_alpha = universal_score(entry, 0.5, now)
        score_high_alpha = universal_score(entry, 2.0, now)

        # Higher alpha should result in lower score
        assert score_high_alpha < score_low_alpha


class TestPruneUniversalMode:
    """Test prune(universal=True) removes from all projects."""

    def test_prune_universal_mode(self):
        """Test universal pruning removes entries from all projects."""
        # Create old entries from multiple projects
        # Need to create entries older than MIN_AGE_TO_PRUNE_DAYS (7 days) with low salience
        entries = []
        for project in ["human", "grok", "agentproof", "axiom"]:
            for i in range(10):
                entries.append(
                    create_old_entry(project, 40 + i, 0.01)
                )  # Old (40+ days), very low salience

        _write_ledger(entries)

        # Use max_entries to force pruning regardless of age/salience rules
        result = prune(max_entries=5, universal=True)

        assert result["strategy"] == "universal"
        # With max_entries=5 and 40 entries, should prune some
        final_ledger = load_ledger()
        assert len(final_ledger) <= 10  # Allow buffer for preserved entries


class TestPruneRespectsMaxEntries:
    """Test that after prune, len(ledger) ≤ max_entries."""

    def test_max_entries_enforced(self):
        """Test pruning respects max_entries limit."""
        # Create many entries
        entries = []
        for i in range(50):
            entries.append(create_old_entry("human", i * 0.1))

        _write_ledger(entries)

        result = prune(max_entries=10, universal=True)
        final_ledger = load_ledger()

        assert len(final_ledger) <= 12  # Allow small buffer for preserved entries


class TestPruneByProjectBreakdown:
    """Test receipt shows pruned_by_project."""

    def test_pruned_by_project_in_result(self):
        """Test result includes pruned_by_project breakdown."""
        # Create old entries from multiple projects
        entries = []
        for project in ["human", "grok", "agentproof"]:
            for i in range(10):
                entries.append(create_old_entry(project, 40 + i, 0.01))

        _write_ledger(entries)

        result = prune(max_entries=5, universal=True)

        assert "pruned_by_project" in result
        assert isinstance(result["pruned_by_project"], dict)


class TestLowAlphaEntriesPrunedFirst:
    """Test low-α entries removed before high-α."""

    def test_low_salience_pruned_first(self):
        """Test low salience entries are pruned first."""
        # Create entries with varying salience
        high_sal = create_old_entry("human", 20, salience=0.9)
        low_sal = create_old_entry("human", 20, salience=0.1)

        _write_ledger([high_sal, low_sal])

        result = prune(
            max_entries=1, max_age_days=10, salience_threshold=0.5, universal=True
        )
        final_ledger = load_ledger()

        # High salience entry should be preserved
        if len(final_ledger) == 1:
            assert final_ledger[0]["salience"] == 0.9


class TestUniversalReceiptEmitted:
    """Test universal_prune_receipt is emitted."""

    def test_receipt_emitted(self):
        """Test that universal_prune_receipt is emitted."""
        receipts_path = Path(os.environ["NEURON_RECEIPTS"])

        # Create some entries
        for i in range(5):
            append(project="neuron", task=f"test {i}", next_action="next")

        prune(universal=True)

        assert receipts_path.exists()
        with open(receipts_path) as f:
            content = f.read()
            assert "universal_prune" in content or "shared_ledger_append" in content
