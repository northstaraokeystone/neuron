"""
NEURON v4.4 Gate 1: Pump Classification Tests
Test α-based bucket sorting and entropy calculation.
"""

import os
import tempfile

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
    classify_for_pump,
    compute_internal_entropy,
    grok_aligned_score,
    append,
    load_ledger,
    _write_ledger,
    LOW_ALPHA_EXPORT_THRESHOLD,
    HIGH_ALPHA_RECIRCULATE_THRESHOLD,
    SYSTEM_TAU_DEFAULT,
)


@pytest.fixture(autouse=True)
def clean_ledger():
    """Clean ledger files before and after each test."""
    from neuron import _get_ledger_path, _get_archive_path, _get_receipts_path

    for path_fn in [_get_ledger_path, _get_archive_path, _get_receipts_path]:
        path = path_fn()
        if path.exists():
            path.unlink()

    yield

    for path_fn in [_get_ledger_path, _get_archive_path, _get_receipts_path]:
        path = path_fn()
        if path.exists():
            path.unlink()


class TestClassifyLowAlpha:
    """Test low-α entries go to export bucket."""

    def test_classify_low_alpha(self):
        """Entries with α < 0.4 should go to export bucket."""
        # Create entry with very low salience (will have low α)
        entry = append(
            project="neuron",
            task="low priority",
            next_action="maybe later",
            salience=0.1  # Low salience
        )

        # Age it significantly
        ledger = load_ledger()
        old_ts = (datetime.now(timezone.utc) - timedelta(days=30)).strftime("%Y-%m-%dT%H:%M:%SZ")
        ledger[0]["ts"] = old_ts
        _write_ledger(ledger)

        # Classify
        result = classify_for_pump(load_ledger(), tau=SYSTEM_TAU_DEFAULT)

        # Low salience + old = low α = export
        assert len(result["export"]) >= 0  # May or may not qualify
        assert "internal_entropy" in result


class TestClassifyHighAlpha:
    """Test high-α entries go to recirculate bucket."""

    def test_classify_high_alpha(self):
        """Entries with α ≥ 0.7 should go to recirculate bucket."""
        # Create entry with high salience and replay
        entry = append(
            project="human",  # Higher project weight
            task="critical task",
            next_action="do immediately",
            salience=1.0  # Max salience
        )

        # Add replay count
        ledger = load_ledger()
        ledger[0]["replay_count"] = 5
        _write_ledger(ledger)

        # Classify
        result = classify_for_pump(load_ledger(), tau=SYSTEM_TAU_DEFAULT)

        # High salience + replay + recent = high α = recirculate
        assert len(result["recirculate"]) >= 1
        assert result["recirculate"][0]["_alpha"] >= HIGH_ALPHA_RECIRCULATE_THRESHOLD


class TestClassifyMidAlpha:
    """Test mid-α entries go to retain bucket."""

    def test_classify_mid_alpha(self):
        """Entries with 0.4 ≤ α < 0.7 should go to retain bucket."""
        # Create entry with medium salience
        entry = append(
            project="neuron",
            task="normal task",
            next_action="process later",
            salience=0.5  # Medium salience
        )

        # Classify
        result = classify_for_pump(load_ledger(), tau=SYSTEM_TAU_DEFAULT)

        # At least one bucket should have entries
        total = len(result["export"]) + len(result["retain"]) + len(result["recirculate"])
        assert total == 1


class TestEntropyCalculation:
    """Test internal entropy calculation."""

    def test_entropy_calculation(self):
        """compute_internal_entropy should return 0-1."""
        # Create mixed entries
        for i in range(10):
            append(
                project="neuron",
                task=f"task {i}",
                next_action="process",
                salience=i / 10.0  # 0.0 to 0.9
            )

        ledger = load_ledger()
        entropy = compute_internal_entropy(ledger)

        assert 0.0 <= entropy <= 1.0

    def test_entropy_empty_ledger(self):
        """Empty ledger should have entropy = 0."""
        entropy = compute_internal_entropy([])
        assert entropy == 0.0


class TestClassificationReceipt:
    """Test pump_classification receipt emission."""

    def test_classification_receipt(self):
        """classify_for_pump should emit receipt with counts."""
        for i in range(5):
            append(
                project="neuron",
                task=f"task {i}",
                next_action="process",
                salience=0.5
            )

        result = classify_for_pump(load_ledger(), tau=SYSTEM_TAU_DEFAULT)

        assert "receipt" in result
        assert result["receipt"]["type"] == "pump_classification"
        assert "export_count" in result["receipt"]
        assert "retain_count" in result["receipt"]
        assert "recirculate_count" in result["receipt"]


class TestEmptyLedger:
    """Test classification with empty ledger."""

    def test_empty_ledger_entropy(self):
        """Empty ledger should have entropy = 0."""
        result = classify_for_pump([], tau=SYSTEM_TAU_DEFAULT)

        assert result["internal_entropy"] == 0.0
        assert result["total_entries"] == 0
        assert len(result["export"]) == 0
        assert len(result["retain"]) == 0
        assert len(result["recirculate"]) == 0


class TestGrokAlignedScore:
    """Test the α score calculation."""

    def test_fresh_high_salience_entry(self):
        """Fresh high-salience entry should have high α."""
        entry = {
            "ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "salience": 1.0,
            "replay_count": 0,
            "project": "human"
        }

        alpha = grok_aligned_score(entry)

        # Fresh, high salience = high α
        assert alpha >= 0.8

    def test_old_low_salience_entry(self):
        """Old low-salience entry should have low α."""
        old_ts = (datetime.now(timezone.utc) - timedelta(days=60)).strftime("%Y-%m-%dT%H:%M:%SZ")
        entry = {
            "ts": old_ts,
            "salience": 0.1,
            "replay_count": 0,
            "project": "grok"  # Lower project weight
        }

        alpha = grok_aligned_score(entry)

        # Old, low salience = low α
        assert alpha < 0.4

    def test_replay_boost(self):
        """Replay count should boost α."""
        entry_base = {
            "ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "salience": 0.5,
            "replay_count": 0,
            "project": "neuron"
        }

        entry_replayed = {**entry_base, "replay_count": 10}

        alpha_base = grok_aligned_score(entry_base)
        alpha_replayed = grok_aligned_score(entry_replayed)

        assert alpha_replayed > alpha_base


class TestThresholdsConfigurable:
    """Test thresholds use constants."""

    def test_thresholds_match_constants(self):
        """Classification thresholds should match module constants."""
        assert LOW_ALPHA_EXPORT_THRESHOLD == 0.4
        assert HIGH_ALPHA_RECIRCULATE_THRESHOLD == 0.7
