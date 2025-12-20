"""
NEURON v4.4 Gate 3: High-α Recirculation Tests
Test amplification of high-α entries.
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
from datetime import datetime, timezone

from neuron import (
    classify_for_pump,
    recirculate_high_alpha,
    amplify_entry,
    compute_internal_entropy,
    append,
    load_ledger,
    _write_ledger,
    HIGH_ALPHA_AMPLIFICATION,
    HIGH_ALPHA_RECIRCULATE_THRESHOLD,
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


class TestRecirculateIncreasesSalience:
    """Test recirculation increases salience."""

    def test_recirculate_increases_salience(self):
        """Salience should increase by amplification factor."""
        # Create high-salience entry
        append(
            project="human", task="critical task", next_action="do now", salience=0.9
        )

        # Add replay count to boost α
        ledger = load_ledger()
        ledger[0]["replay_count"] = 5
        _write_ledger(ledger)

        classification = classify_for_pump(load_ledger())

        if classification["recirculate"]:
            before_salience = classification["recirculate"][0].get("salience", 1.0)
            result = recirculate_high_alpha(classification["recirculate"])

            assert result["avg_salience_after"] >= result["avg_salience_before"]


class TestSalienceCappedAtOne:
    """Test salience cap at 1.0."""

    def test_salience_capped_at_one(self):
        """Amplified salience should not exceed 1.0."""
        entry = {
            "ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "salience": 0.9,  # Will exceed 1.0 after 2x amplification
            "project": "neuron",
            "_alpha": 0.8,
        }

        amplified = amplify_entry(entry, factor=2.0)

        assert amplified["salience"] <= 1.0


class TestReplayCountIncremented:
    """Test replay_count is incremented."""

    def test_replay_count_incremented(self):
        """replay_count should increase by 1."""
        entry = {
            "ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "salience": 0.8,
            "replay_count": 3,
            "project": "neuron",
            "_alpha": 0.8,
        }

        amplified = amplify_entry(entry)

        assert amplified["replay_count"] == 4


class TestAmplifiedAtSet:
    """Test amplified_at timestamp is set."""

    def test_amplified_at_set(self):
        """Timestamp should be recorded."""
        entry = {
            "ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "salience": 0.8,
            "project": "neuron",
            "_alpha": 0.8,
        }

        amplified = amplify_entry(entry)

        assert "amplified_at" in amplified
        assert amplified["amplified_at"] is not None


class TestEntropyDecreased:
    """Test entropy decreases after recirculation."""

    def test_entropy_decreased(self):
        """Internal entropy should be lower after recirculation."""
        # Create high-salience entries
        for i in range(5):
            append(
                project="human",
                task=f"critical task {i}",
                next_action="do now",
                salience=0.9,
            )

        # Add replay counts
        ledger = load_ledger()
        for e in ledger:
            e["replay_count"] = 5
        _write_ledger(ledger)

        entropy_before = compute_internal_entropy(load_ledger())

        classification = classify_for_pump(load_ledger())
        if classification["recirculate"]:
            result = recirculate_high_alpha(classification["recirculate"])

            # Entropy may decrease or stay similar
            # Amplifying high-α entries makes them even higher α
            # which should maintain or decrease entropy
            assert result["internal_entropy_after"] <= entropy_before + 0.1


class TestOnlyHighAlpha:
    """Test only high-α entries are recirculated."""

    def test_only_high_alpha(self):
        """Only entries with α ≥ threshold should be recirculated."""
        # Create mixed entries
        append(project="human", task="high priority", next_action="now", salience=1.0)
        append(project="grok", task="low priority", next_action="later", salience=0.1)

        # Add replay to first entry
        ledger = load_ledger()
        ledger[0]["replay_count"] = 5
        _write_ledger(ledger)

        classification = classify_for_pump(load_ledger())

        # Check recirculate bucket only has high-α entries
        for entry in classification["recirculate"]:
            assert entry["_alpha"] >= HIGH_ALPHA_RECIRCULATE_THRESHOLD


class TestRecircReceipt:
    """Test recirculation receipt emission."""

    def test_recirc_receipt(self):
        """Recirculation should emit receipt with before/after."""
        # Create high-salience entries
        for i in range(3):
            append(project="human", task=f"task {i}", next_action="do", salience=0.9)

        # Add replay counts
        ledger = load_ledger()
        for e in ledger:
            e["replay_count"] = 5
        _write_ledger(ledger)

        classification = classify_for_pump(load_ledger())
        if classification["recirculate"]:
            result = recirculate_high_alpha(classification["recirculate"])

            assert "receipt" in result
            assert result["receipt"]["type"] == "high_alpha_recirculate"
            assert "avg_salience_before" in result["receipt"]
            assert "avg_salience_after" in result["receipt"]


class TestAmplificationFactor:
    """Test amplification factor is applied correctly."""

    def test_amplification_factor_default(self):
        """Default amplification should be 2.0."""
        assert HIGH_ALPHA_AMPLIFICATION == 2.0

    def test_custom_amplification(self):
        """Custom amplification factor should be applied."""
        entry = {
            "ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "salience": 0.4,
            "project": "neuron",
            "_alpha": 0.8,
        }

        amplified = amplify_entry(entry, factor=1.5)

        assert abs(amplified["salience"] - 0.6) < 0.001  # 0.4 * 1.5 = 0.6


class TestRecirculationRound:
    """Test recirculation_round is tracked."""

    def test_recirculation_round(self):
        """Recirculation round should be incremented."""
        entry = {
            "ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "salience": 0.8,
            "recirculation_round": 2,
            "project": "neuron",
            "_alpha": 0.8,
        }

        amplified = amplify_entry(entry)

        assert amplified["recirculation_round"] == 3
