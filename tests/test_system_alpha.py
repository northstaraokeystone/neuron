"""
NEURON v4.3: System-Wide Alpha Tests (Gate 2)
Tests for unified gap detection and system-wide α calculation.
"""

import os
import tempfile
import time
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
    detect_system_gaps,
    weighted_gap,
    alpha_system_wide,
    alpha_by_project,
    emit_system_alpha_receipt,
    load_ledger,
    _write_ledger,
    SYSTEM_GAP_WEIGHT,
)


@pytest.fixture(autouse=True)
def clean_ledger():
    """Clean ledger before each test."""
    ledger_path = Path(os.environ["NEURON_LEDGER"])
    if ledger_path.exists():
        ledger_path.unlink()
    yield
    if ledger_path.exists():
        ledger_path.unlink()


def create_entry_with_ts(project: str, ts: datetime) -> dict:
    """Create an entry with a specific timestamp."""
    return {
        "ts": ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "project": project,
        "event_type": "task",
        "model": "neuron",
        "commit": None,
        "task": f"Test task for {project}",
        "next": "next",
        "salience": 1.0,
        "replay_count": 0,
        "energy": 1.0,
        "token_count": 0,
        "inference_id": None,
        "context_summary": "",
        "source_context": {},
        "hash": f"test_hash_{project}_{ts.timestamp()}"
    }


class TestDetectGapsSingleProject:
    """Test gap detection within same project."""

    def test_detect_gaps_single_project(self):
        """Test gaps detected within same project."""
        now = datetime.now(timezone.utc)
        entries = [
            create_entry_with_ts("human", now - timedelta(minutes=10)),
            create_entry_with_ts("human", now - timedelta(minutes=5)),
            create_entry_with_ts("human", now),
        ]
        _write_ledger(entries)

        gaps = detect_system_gaps(threshold_minutes=1)
        assert len(gaps) == 2
        for gap in gaps:
            assert gap["from_project"] == "human"
            assert gap["to_project"] == "human"


class TestDetectGapsCrossProject:
    """Test gap detection across projects."""

    def test_detect_gaps_cross_project(self):
        """Test gaps detected across different projects."""
        now = datetime.now(timezone.utc)
        entries = [
            create_entry_with_ts("human", now - timedelta(minutes=10)),
            create_entry_with_ts("agentproof", now - timedelta(minutes=5)),
            create_entry_with_ts("axiom", now),
        ]
        _write_ledger(entries)

        gaps = detect_system_gaps(threshold_minutes=1)
        assert len(gaps) == 2

        # Check cross-project gaps
        projects_in_gaps = set()
        for gap in gaps:
            projects_in_gaps.add(gap["from_project"])
            projects_in_gaps.add(gap["to_project"])

        assert "human" in projects_in_gaps
        assert "agentproof" in projects_in_gaps
        assert "axiom" in projects_in_gaps


class TestWeightedGapHuman:
    """Test human gap uses weight 1.0."""

    def test_weighted_gap_human(self):
        """Test human gaps use baseline weight 1.0."""
        gap = weighted_gap(60.0, "human", "human")
        # human weight = 1.0, so gap * 1.0 * 1.0 = 60.0
        assert gap == 60.0

    def test_weighted_gap_human_to_ai(self):
        """Test human to AI gap weighting."""
        gap = weighted_gap(60.0, "human", "grok")
        # human weight = 1.0, grok weight = 0.8, so 60 * 1.0 * 0.8 = 48.0
        expected = 60.0 * SYSTEM_GAP_WEIGHT["human"] * SYSTEM_GAP_WEIGHT["grok"]
        assert gap == expected


class TestWeightedGapAI:
    """Test AI gaps use lower weights."""

    def test_weighted_gap_grok(self):
        """Test Grok gaps use weight 0.8."""
        gap = weighted_gap(60.0, "grok", "grok")
        expected = 60.0 * SYSTEM_GAP_WEIGHT["grok"] * SYSTEM_GAP_WEIGHT["grok"]
        assert gap == expected
        assert gap < 60.0  # Should be less than human baseline

    def test_weighted_gap_agentproof(self):
        """Test AgentProof gaps use weight 0.6."""
        gap = weighted_gap(60.0, "agentproof", "agentproof")
        expected = 60.0 * SYSTEM_GAP_WEIGHT["agentproof"] * SYSTEM_GAP_WEIGHT["agentproof"]
        assert gap == expected
        assert gap < 60.0 * 0.8 * 0.8  # Should be less than grok


class TestAlphaSystemWide:
    """Test system-wide alpha calculation."""

    def test_alpha_system_wide_empty(self):
        """Test alpha returns 0 for empty ledger."""
        alpha = alpha_system_wide()
        assert alpha == 0.0

    def test_alpha_system_wide_with_gaps(self):
        """Test alpha calculates correctly with gaps."""
        now = datetime.now(timezone.utc)
        entries = [
            create_entry_with_ts("human", now - timedelta(minutes=10)),
            create_entry_with_ts("grok", now - timedelta(minutes=5)),
            create_entry_with_ts("axiom", now),
        ]
        _write_ledger(entries)

        alpha = alpha_system_wide(threshold_minutes=1)
        assert alpha > 0.0
        assert isinstance(alpha, float)


class TestAlphaByProject:
    """Test per-project alpha breakdown."""

    def test_alpha_by_project_returns_dict(self):
        """Test alpha_by_project returns dict with all active projects."""
        now = datetime.now(timezone.utc)
        entries = [
            create_entry_with_ts("human", now - timedelta(minutes=15)),
            create_entry_with_ts("human", now - timedelta(minutes=10)),
            create_entry_with_ts("grok", now - timedelta(minutes=8)),
            create_entry_with_ts("grok", now - timedelta(minutes=5)),
            create_entry_with_ts("axiom", now - timedelta(minutes=3)),
            create_entry_with_ts("axiom", now),
        ]
        _write_ledger(entries)

        by_project = alpha_by_project()

        assert isinstance(by_project, dict)
        assert "human" in by_project
        assert "grok" in by_project
        assert "axiom" in by_project


class TestAlphaReceiptEmitted:
    """Test system_alpha_receipt is emitted."""

    def test_receipt_emitted(self):
        """Test that system_alpha_receipt is emitted."""
        now = datetime.now(timezone.utc)
        entries = [
            create_entry_with_ts("human", now - timedelta(minutes=10)),
            create_entry_with_ts("grok", now),
        ]
        _write_ledger(entries)

        receipt = emit_system_alpha_receipt()

        assert receipt["type"] == "system_alpha"
        assert "alpha_system" in receipt
        assert "alpha_by_project" in receipt
        assert "total_gaps" in receipt
        assert "cross_project_gaps" in receipt


class TestCrossProjectGapLower:
    """Test cross-AI gaps weighted lower than human gaps."""

    def test_cross_ai_gaps_lower(self):
        """Test that cross-AI gaps are weighted lower than human gaps."""
        # Human to human gap
        human_gap = weighted_gap(60.0, "human", "human")

        # AI to AI gap (should be lower)
        ai_gap = weighted_gap(60.0, "grok", "agentproof")

        assert ai_gap < human_gap
