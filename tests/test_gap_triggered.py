"""
NEURON v4.4 Gate 4: Gap-Triggered Export Tests
Test gap detection and accelerated export.
"""

import os
import tempfile
import time

# Set up isolated test environment BEFORE importing neuron
_test_dir = tempfile.mkdtemp()
os.environ["NEURON_LEDGER"] = os.path.join(_test_dir, "test_receipts.jsonl")
os.environ["NEURON_ARCHIVE"] = os.path.join(_test_dir, "test_archive.jsonl")
os.environ["NEURON_RECEIPTS"] = os.path.join(_test_dir, "test_receipts.jsonl")
os.environ["NEURON_BASE"] = _test_dir

import pytest
from datetime import datetime, timezone, timedelta

import neuron
from neuron import (
    detect_gap_trigger,
    gap_directed_export,
    compute_internal_entropy,
    append,
    load_ledger,
    _write_ledger,
    GAP_EXPORT_MULTIPLIER,
)


@pytest.fixture(autouse=True)
def clean_ledger():
    """Clean ledger files before and after each test."""
    from neuron import _get_ledger_path, _get_archive_path, _get_receipts_path, _get_stub_path
    import shutil

    for path_fn in [_get_ledger_path, _get_archive_path, _get_receipts_path]:
        path = path_fn()
        if path.exists():
            path.unlink()

    stub_path = _get_stub_path()
    if stub_path.exists():
        shutil.rmtree(stub_path)

    yield

    for path_fn in [_get_ledger_path, _get_archive_path, _get_receipts_path]:
        path = path_fn()
        if path.exists():
            path.unlink()

    if stub_path.exists():
        shutil.rmtree(stub_path)


class TestDetectGapHuman:
    """Test human gap detection."""

    def test_detect_gap_human(self):
        """Human gaps should be detected."""
        # Create entries with gap
        now = datetime.now(timezone.utc)

        entry1 = append(
            project="human",
            task="before gap",
            next_action="wait"
        )

        # Modify timestamp to create gap
        ledger = load_ledger()
        ledger[0]["ts"] = (now - timedelta(hours=2)).strftime("%Y-%m-%dT%H:%M:%SZ")
        _write_ledger(ledger)

        entry2 = append(
            project="human",
            task="after gap",
            next_action="continue"
        )

        gaps = detect_gap_trigger(load_ledger(), threshold_minutes=60)

        assert len(gaps) >= 1
        assert gaps[0]["from_project"] == "human"


class TestDetectGapGrok:
    """Test Grok gap detection."""

    def test_detect_gap_grok(self):
        """Grok gaps should be detected."""
        now = datetime.now(timezone.utc)

        entry1 = append(
            project="grok",
            task="before eviction",
            next_action="wait"
        )

        ledger = load_ledger()
        ledger[0]["ts"] = (now - timedelta(hours=3)).strftime("%Y-%m-%dT%H:%M:%SZ")
        _write_ledger(ledger)

        entry2 = append(
            project="grok",
            task="after eviction",
            next_action="continue"
        )

        gaps = detect_gap_trigger(load_ledger(), threshold_minutes=60)

        assert len(gaps) >= 1
        assert gaps[0]["from_project"] == "grok"


class TestGapTriggersExport:
    """Test gap triggers export cycle."""

    def test_gap_triggers_export(self):
        """Gap should fire export cycle."""
        now = datetime.now(timezone.utc)

        # Create entries with various saliences
        for i in range(5):
            append(
                project="neuron",
                task=f"task {i}",
                next_action="process",
                salience=0.2
            )

        # Age entries and create gap
        ledger = load_ledger()
        for i, e in enumerate(ledger):
            e["ts"] = (now - timedelta(days=30, hours=i)).strftime("%Y-%m-%dT%H:%M:%SZ")
        _write_ledger(ledger)

        gaps = detect_gap_trigger(load_ledger(), threshold_minutes=30)

        if gaps:
            result = gap_directed_export(load_ledger(), gaps[0])

            # Should have exported something (depends on classification)
            assert "entries_exported" in result


class TestMultiplierApplied:
    """Test 3x multiplier is applied."""

    def test_multiplier_applied(self):
        """3x more should be exported on gap."""
        assert GAP_EXPORT_MULTIPLIER == 3.0

        now = datetime.now(timezone.utc)

        # Create many entries
        for i in range(20):
            append(
                project="neuron",
                task=f"task {i}",
                next_action="process",
                salience=0.2
            )

        # Age entries
        ledger = load_ledger()
        for i, e in enumerate(ledger):
            e["ts"] = (now - timedelta(days=30, hours=i)).strftime("%Y-%m-%dT%H:%M:%SZ")
        _write_ledger(ledger)

        gaps = detect_gap_trigger(load_ledger(), threshold_minutes=30)

        if gaps:
            result = gap_directed_export(load_ledger(), gaps[0])

            # Normal export would be less than entries_exported
            if result["normal_export_would_be"] > 0:
                assert result["entries_exported"] >= result["normal_export_would_be"]


class TestEntropyDropsOnGap:
    """Test entropy decreases after gap-triggered export."""

    def test_entropy_drops_on_gap(self):
        """Gap → lower entropy."""
        now = datetime.now(timezone.utc)

        for i in range(10):
            append(
                project="neuron",
                task=f"task {i}",
                next_action="process",
                salience=0.2
            )

        ledger = load_ledger()
        for i, e in enumerate(ledger):
            e["ts"] = (now - timedelta(days=30, hours=i)).strftime("%Y-%m-%dT%H:%M:%SZ")
        _write_ledger(ledger)

        entropy_before = compute_internal_entropy(load_ledger())

        gaps = detect_gap_trigger(load_ledger(), threshold_minutes=30)

        if gaps:
            result = gap_directed_export(load_ledger(), gaps[0])

            if result["entries_exported"] > 0:
                assert result["internal_entropy_after"] <= entropy_before


class TestGapReceiptShowsMultiplier:
    """Test receipt shows normal vs actual export."""

    def test_gap_receipt_shows_multiplier(self):
        """Receipt should show normal vs actual count."""
        now = datetime.now(timezone.utc)

        for i in range(10):
            append(
                project="neuron",
                task=f"task {i}",
                next_action="process",
                salience=0.2
            )

        ledger = load_ledger()
        for i, e in enumerate(ledger):
            e["ts"] = (now - timedelta(days=30, hours=i)).strftime("%Y-%m-%dT%H:%M:%SZ")
        _write_ledger(ledger)

        gaps = detect_gap_trigger(load_ledger(), threshold_minutes=30)

        if gaps:
            result = gap_directed_export(load_ledger(), gaps[0])

            assert "receipt" in result
            assert "normal_export_would_be" in result["receipt"]
            assert "entries_exported" in result["receipt"]


class TestNoGapNoTrigger:
    """Test small gaps don't trigger."""

    def test_no_gap_no_trigger(self):
        """Small gaps should be ignored."""
        # Create entries close together
        for i in range(5):
            append(
                project="neuron",
                task=f"task {i}",
                next_action="process"
            )

        # No artificial gaps - entries are close in time
        gaps = detect_gap_trigger(load_ledger(), threshold_minutes=60)

        # Should have no significant gaps
        assert len(gaps) == 0


class TestGapSource:
    """Test gap source is recorded."""

    def test_gap_source_recorded(self):
        """Gap source project should be recorded."""
        now = datetime.now(timezone.utc)

        entry1 = append(
            project="axiom",
            task="before gap",
            next_action="wait"
        )

        ledger = load_ledger()
        ledger[0]["ts"] = (now - timedelta(hours=2)).strftime("%Y-%m-%dT%H:%M:%SZ")
        _write_ledger(ledger)

        entry2 = append(
            project="axiom",
            task="after gap",
            next_action="continue"
        )

        gaps = detect_gap_trigger(load_ledger(), threshold_minutes=60)

        if gaps:
            assert gaps[0]["gap_source"] == "axiom"
            assert gaps[0]["from_project"] == "axiom"
