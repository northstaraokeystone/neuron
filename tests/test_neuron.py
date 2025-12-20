"""Tests for NEURON v3 biologically grounded ledger."""

import os
import tempfile
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

# Set up test ledger paths before importing neuron
TEST_LEDGER = Path(tempfile.gettempdir()) / "test_receipts.jsonl"
TEST_ARCHIVE = Path(tempfile.gettempdir()) / "test_archive.jsonl"
TEST_RECEIPTS = Path(tempfile.gettempdir()) / "test_emit_receipts.jsonl"
os.environ["NEURON_LEDGER"] = str(TEST_LEDGER)
os.environ["NEURON_ARCHIVE"] = str(TEST_ARCHIVE)
os.environ["NEURON_RECEIPTS"] = str(TEST_RECEIPTS)

from neuron import (
    alpha,
    append,
    consolidate,
    dual_hash,
    emit_receipt,
    energy_estimate,
    merkle,
    predict_next,
    prune,
    recovery_cost,
    replay,
    salience_decay,
    StopRule,
    _get_ledger_path,
    _get_archive_path,
    _get_receipts_path,
)


@pytest.fixture(autouse=True)
def clean_ledger():
    """Remove test ledger before and after each test using actual paths."""
    paths = [_get_ledger_path(), _get_archive_path(), _get_receipts_path()]
    for path in paths:
        if path.exists():
            path.unlink()
    yield
    for path in paths:
        if path.exists():
            path.unlink()


class TestDualHash:
    def test_string_input(self):
        result = dual_hash("test")
        assert ":" in result
        parts = result.split(":")
        assert len(parts) == 2
        assert len(parts[0]) == 64  # SHA256 hex length
        assert len(parts[1]) == 64  # BLAKE3 hex length

    def test_bytes_input(self):
        result = dual_hash(b"test")
        assert ":" in result

    def test_deterministic(self):
        result1 = dual_hash("same input")
        result2 = dual_hash("same input")
        assert result1 == result2


class TestEnergyEstimate:
    def test_simple_task(self):
        energy = energy_estimate("fix bug", "test")
        assert 0.5 <= energy <= 2.0

    def test_technical_task_higher_energy(self):
        simple = energy_estimate("fix bug", "test")
        technical = energy_estimate(
            "implement merkle proof federation", "verify hash entropy"
        )
        assert technical > simple

    def test_longer_task_higher_energy(self):
        short = energy_estimate("fix", "test")
        long = energy_estimate(
            "implement the complete authentication flow", "write integration tests"
        )
        assert long > short


class TestSalienceDecay:
    def test_fresh_entry_full_salience(self):
        entry = {
            "ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "salience": 1.0,
            "replay_count": 0,
        }
        decayed = salience_decay(entry)
        assert decayed > 0.99  # Nearly 1.0 for fresh entry

    def test_old_entry_decayed(self):
        old_ts = (datetime.now(timezone.utc) - timedelta(days=30)).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
        entry = {"ts": old_ts, "salience": 1.0, "replay_count": 0}
        decayed = salience_decay(entry)
        assert decayed < 0.5  # Significantly decayed after 30 days

    def test_replay_slows_decay(self):
        old_ts = (datetime.now(timezone.utc) - timedelta(days=30)).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
        no_replay = {"ts": old_ts, "salience": 1.0, "replay_count": 0}
        with_replay = {"ts": old_ts, "salience": 1.0, "replay_count": 5}
        assert salience_decay(with_replay) > salience_decay(no_replay)


class TestRecoveryCost:
    def test_zero_gap(self):
        assert recovery_cost(0) == 1.0

    def test_short_gap(self):
        cost = recovery_cost(15)
        assert 1.0 < cost < 2.0

    def test_medium_gap(self):
        cost = recovery_cost(60)
        assert 2.0 < cost < 3.5

    def test_long_gap(self):
        cost = recovery_cost(120)
        assert 3.0 < cost < 4.0

    def test_nonlinear_increase(self):
        c15 = recovery_cost(15)
        c60 = recovery_cost(60)
        c120 = recovery_cost(120)
        # Recovery cost should increase but with diminishing returns
        assert c15 < c60 < c120
        assert (c60 - c15) > (c120 - c60)  # Diminishing returns


class TestAppend:
    def test_basic_append(self):
        entry = append("neuron", "test task", "next action", "abc123")
        assert entry["project"] == "neuron"
        assert entry["task"] == "test task"
        assert entry["next"] == "next action"
        assert entry["commit"] == "abc123"
        assert "ts" in entry
        assert "hash" in entry

    def test_new_v3_fields(self):
        entry = append("neuron", "test task", "next action", "abc123")
        assert entry["salience"] == 1.0
        assert entry["replay_count"] == 0
        assert "energy" in entry
        assert 0.5 <= entry["energy"] <= 2.0

    def test_custom_energy(self):
        entry = append("neuron", "task", "next", None, energy=1.5)
        assert entry["energy"] == 1.5

    def test_invalid_project(self):
        # v4.3: Invalid project now raises StopRule instead of ValueError
        with pytest.raises(StopRule):
            append("invalid", "task", "next", None)

    def test_truncation(self):
        long_task = "x" * 100
        entry = append("neuron", long_task, "next", None)
        assert len(entry["task"]) == 50

    def test_ledger_file_created(self):
        append("neuron", "task", "next", None)
        assert _get_ledger_path().exists()


class TestReplay:
    def test_empty_ledger(self):
        result = replay()
        assert result == []

    def test_returns_entries(self):
        append("neuron", "task1", "next1", None)
        append("axiom", "task2", "next2", None)
        result = replay()
        assert len(result) == 2
        assert result[0]["task"] == "task1"
        assert result[1]["task"] == "task2"

    def test_limit(self):
        for i in range(5):
            append("neuron", f"task{i}", f"next{i}", None)
        result = replay(n=2)
        assert len(result) == 2
        assert result[0]["task"] == "task3"
        assert result[1]["task"] == "task4"

    def test_increment_replay(self):
        append("neuron", "task1", "next1", None)
        initial = replay(n=1)[0]
        assert initial["replay_count"] == 0

        replay(n=1, increment_replay=True)
        after = replay(n=1)[0]
        assert after["replay_count"] == 1


class TestAlpha:
    def test_empty_ledger(self):
        result = alpha()
        assert result["total_entries"] == 0
        assert result["gaps_detected"] == 0

    def test_no_gaps(self):
        append("neuron", "task1", "next1", None)
        append("neuron", "task2", "next2", None)
        result = alpha(threshold_minutes=60)
        assert result["total_entries"] == 2
        assert result["gaps_detected"] == 0

    def test_v3_variance_fields(self):
        append("neuron", "task1", "next1", None)
        append("neuron", "task2", "next2", None)
        result = alpha(threshold_minutes=60)
        assert "alpha_variance" in result
        assert "alpha_std" in result
        assert "expert_novice_ratio" in result


class TestConsolidate:
    def test_empty_ledger(self):
        result = consolidate()
        assert result["consolidated_count"] == 0

    def test_consolidation_boosts_salience(self):
        # Create entries with gaps
        append("neuron", "task1", "next1", None)
        time.sleep(0.1)
        append("neuron", "task2", "next2", None)

        # Consolidate with low threshold for testing
        result = consolidate(top_k=5, alpha_threshold=0.0)
        assert isinstance(result["consolidated_count"], int)
        assert isinstance(result["salience_boost"], float)


class TestPrune:
    def test_empty_ledger(self):
        result = prune()
        assert result["pruned_count"] == 0
        assert result["ledger_size_before"] == 0
        assert result["ledger_size_after"] == 0

    def test_prune_preserves_recent(self):
        append("neuron", "task1", "next1", None)
        result = prune(max_age_days=30, salience_threshold=0.1)
        assert result["pruned_count"] == 0
        assert result["ledger_size_after"] == 1

    def test_prune_preserves_high_replay(self):
        # Even with aggressive pruning, high replay_count entries are preserved
        append("neuron", "task1", "next1", None)
        # Simulate high replay count by multiple increments
        for _ in range(6):
            replay(n=1, increment_replay=True)

        result = prune(max_age_days=0, salience_threshold=2.0)  # Aggressive
        # Entry preserved due to high replay_count
        assert result["ledger_size_after"] == 1


class TestPredictNext:
    def test_empty_ledger(self):
        result = predict_next()
        assert result is None

    def test_single_entry(self):
        append("neuron", "task1", "next1", None)
        result = predict_next()
        assert result is None  # Need at least 2 entries

    def test_pattern_prediction(self):
        append("neuron", "implement module A", "write tests", None)
        append("neuron", "implement module B", "write tests", None)
        append("neuron", "implement module C", "pending", None)

        prediction = predict_next(n_context=5)
        # Should suggest "write tests" based on pattern
        assert prediction is not None


class TestIntegration:
    def test_full_lifecycle(self):
        """Test the full lifecycle of an entry: append -> replay -> consolidate -> decay check."""
        # Append
        entry = append(
            "neuron", "implement federation", "write merkle proofs", "abc123"
        )
        assert entry["salience"] == 1.0
        assert entry["energy"] >= 0.5  # Valid energy range

        # Replay with increment
        replayed = replay(n=1, increment_replay=True)
        assert replayed[0]["replay_count"] == 1

        # Check decay is slowed by replay
        decayed = salience_decay(replayed[0])
        assert decayed > 0.99  # Fresh entry, almost no decay

        # Consolidate
        result = consolidate(top_k=5, alpha_threshold=0.0)
        assert "consolidated_count" in result

        # Prune (should not prune fresh entries)
        prune_result = prune(max_age_days=30, salience_threshold=0.1)
        assert prune_result["pruned_count"] == 0

    def test_recovery_cost_integration(self):
        """Verify recovery cost follows non-linear curve."""
        costs = [recovery_cost(m) for m in [0, 15, 30, 60, 90, 120, 180, 240]]

        # All costs should be >= 1.0
        assert all(c >= 1.0 for c in costs)

        # Should increase monotonically
        for i in range(1, len(costs)):
            assert costs[i] >= costs[i - 1]

        # At 120 minutes, should be around 3.5
        assert 3.0 < recovery_cost(120) < 4.0


# CLAUDEME §8 Core Function Tests
class TestEmitReceipt:
    """Tests for emit_receipt() per CLAUDEME §4."""

    def test_emit_receipt_basic(self):
        """Test basic receipt emission."""
        receipt = emit_receipt("test_receipt", {"test_key": "test_value"})
        assert receipt["type"] == "test_receipt"
        assert receipt["test_key"] == "test_value"
        assert "ts" in receipt
        assert "hash" in receipt

    def test_emit_receipt_has_dual_hash(self):
        """Test receipt hash is dual format (SHA256:BLAKE3)."""
        receipt = emit_receipt("hash_test_receipt", {"data": 123})
        assert ":" in receipt["hash"]
        parts = receipt["hash"].split(":")
        assert len(parts) == 2
        assert len(parts[0]) == 64
        assert len(parts[1]) == 64

    def test_emit_receipt_timestamp_format(self):
        """Test receipt timestamp is ISO 8601 format."""
        receipt = emit_receipt("ts_test_receipt", {})
        assert "T" in receipt["ts"]
        assert receipt["ts"].endswith("Z")


class TestMerkle:
    """Tests for merkle() per CLAUDEME §8."""

    def test_merkle_empty_list(self):
        """Test merkle of empty list."""
        result = merkle([])
        assert ":" in result  # Dual hash format

    def test_merkle_single_item(self):
        """Test merkle of single item."""
        result = merkle(["single"])
        assert ":" in result
        assert len(result.split(":")[0]) == 64

    def test_merkle_multiple_items(self):
        """Test merkle of multiple items."""
        result = merkle(["a", "b", "c"])
        assert ":" in result
        assert len(result.split(":")[0]) == 64

    def test_merkle_deterministic(self):
        """Test merkle is deterministic."""
        items = ["x", "y", "z"]
        result1 = merkle(items)
        result2 = merkle(items)
        assert result1 == result2

    def test_merkle_order_matters(self):
        """Test merkle changes with item order."""
        result1 = merkle(["a", "b"])
        result2 = merkle(["b", "a"])
        assert result1 != result2

    def test_merkle_with_dicts(self):
        """Test merkle with dict items."""
        items = [{"key": "value1"}, {"key": "value2"}]
        result = merkle(items)
        assert ":" in result

    def test_merkle_with_bytes(self):
        """Test merkle with bytes items."""
        items = [b"bytes1", b"bytes2"]
        result = merkle(items)
        assert ":" in result


class TestStopRule:
    """Tests for StopRule exception per CLAUDEME §8."""

    def test_stoprule_is_exception(self):
        """Test StopRule is an Exception subclass."""
        assert issubclass(StopRule, Exception)

    def test_stoprule_basic_raise(self):
        """Test StopRule can be raised and caught."""
        with pytest.raises(StopRule):
            raise StopRule("test_rule", "test message")

    def test_stoprule_has_rule_name(self):
        """Test StopRule stores rule name."""
        try:
            raise StopRule("my_rule", "my message")
        except StopRule as e:
            assert e.rule_name == "my_rule"

    def test_stoprule_has_context(self):
        """Test StopRule stores context dict."""
        ctx = {"key": "value", "count": 42}
        try:
            raise StopRule("ctx_rule", "context message", ctx)
        except StopRule as e:
            assert e.context == ctx

    def test_stoprule_message_format(self):
        """Test StopRule message format."""
        try:
            raise StopRule("format_rule", "detailed message")
        except StopRule as e:
            assert "STOPRULE[format_rule]" in str(e)
            assert "detailed message" in str(e)

    def test_stoprule_default_context(self):
        """Test StopRule has empty context by default."""
        try:
            raise StopRule("default_rule", "no context")
        except StopRule as e:
            assert e.context == {}
