"""
NEURON v4.4 Gate 5: Burst Sync Tests
Test burst mode and accumulated sync.
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
    enable_burst_mode,
    is_burst_mode_active,
    accumulate_for_burst,
    burst_sync,
    reset_burst_state,
    compute_internal_entropy,
    classify_for_pump,
    append,
    load_ledger,
    _write_ledger,
    BURST_MODE_LATENCY_THRESHOLD,
    BURST_SYNC_MAX_BATCH,
)


@pytest.fixture(autouse=True)
def clean_ledger():
    """Clean ledger files before and after each test."""
    from neuron import _get_ledger_path, _get_archive_path, _get_receipts_path, _get_stub_path
    import shutil

    # Reset burst state before each test
    reset_burst_state()

    for path_fn in [_get_ledger_path, _get_archive_path, _get_receipts_path]:
        path = path_fn()
        if path.exists():
            path.unlink()

    stub_path = _get_stub_path()
    if stub_path.exists():
        shutil.rmtree(stub_path)

    yield

    reset_burst_state()

    for path_fn in [_get_ledger_path, _get_archive_path, _get_receipts_path]:
        path = path_fn()
        if path.exists():
            path.unlink()

    if stub_path.exists():
        shutil.rmtree(stub_path)


class TestEnableBurstHighLatency:
    """Test burst mode activation on high latency."""

    def test_enable_burst_high_latency(self):
        """Latency > threshold should enable burst mode."""
        high_latency = BURST_MODE_LATENCY_THRESHOLD + 1000

        result = enable_burst_mode(latency_ms=high_latency)

        assert result is True
        assert is_burst_mode_active() is True


class TestDisableBurstLowLatency:
    """Test burst mode disabled on low latency."""

    def test_disable_burst_low_latency(self):
        """Latency < threshold should disable burst mode."""
        low_latency = 100  # 100ms

        result = enable_burst_mode(latency_ms=low_latency)

        assert result is False
        assert is_burst_mode_active() is False


class TestAccumulateEntries:
    """Test entries are queued without sync."""

    def test_accumulate_entries(self):
        """Entries should be queued, not synced."""
        entries = [
            {"ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
             "project": "neuron", "salience": 0.5}
            for _ in range(10)
        ]

        result = accumulate_for_burst(entries)

        assert result["entries_accumulated"] == 10
        assert result["queue_size"] == 10
        assert result["max_batch"] == BURST_SYNC_MAX_BATCH


class TestBurstSyncAll:
    """Test all accumulated entries are synced."""

    def test_burst_sync_all(self):
        """All accumulated entries should be synced."""
        # Create and add entries to ledger
        for i in range(5):
            append(
                project="neuron",
                task=f"task {i}",
                next_action="process",
                salience=0.2
            )

        # Age entries
        ledger = load_ledger()
        old_ts = (datetime.now(timezone.utc) - timedelta(days=30)).strftime("%Y-%m-%dT%H:%M:%SZ")
        for e in ledger:
            e["ts"] = old_ts
        _write_ledger(ledger)

        # Classify and get export candidates
        classification = classify_for_pump(load_ledger())

        if classification["export"]:
            # Enable burst mode
            enable_burst_mode(latency_ms=50000)

            # Accumulate
            accumulate_for_burst(classification["export"])

            # Sync
            result = burst_sync()

            assert result["entries_synced"] == len(classification["export"])


class TestEfficiencyRatio:
    """Test efficiency ratio calculation."""

    def test_efficiency_ratio(self):
        """Burst should be more efficient than continuous."""
        # Create entries
        for i in range(5):
            append(
                project="neuron",
                task=f"task {i}",
                next_action="process",
                salience=0.2
            )

        # Age entries
        ledger = load_ledger()
        old_ts = (datetime.now(timezone.utc) - timedelta(days=30)).strftime("%Y-%m-%dT%H:%M:%SZ")
        for e in ledger:
            e["ts"] = old_ts
        _write_ledger(ledger)

        classification = classify_for_pump(load_ledger())

        if classification["export"]:
            enable_burst_mode(latency_ms=50000)
            accumulate_for_burst(classification["export"])
            result = burst_sync()

            # Efficiency should be > 0 for non-empty batch
            if result["entries_synced"] > 0:
                assert result["efficiency_ratio"] > 0


class TestEntropyDropsOnBurst:
    """Test internal entropy decreases after burst."""

    def test_entropy_drops_on_burst(self):
        """Internal entropy should be near-zero after burst."""
        # Create entries
        for i in range(10):
            append(
                project="neuron",
                task=f"task {i}",
                next_action="process",
                salience=0.2
            )

        # Age entries
        ledger = load_ledger()
        old_ts = (datetime.now(timezone.utc) - timedelta(days=30)).strftime("%Y-%m-%dT%H:%M:%SZ")
        for e in ledger:
            e["ts"] = old_ts
        _write_ledger(ledger)

        classification = classify_for_pump(load_ledger())

        if classification["export"]:
            enable_burst_mode(latency_ms=50000)
            accumulate_for_burst(classification["export"])

            result = burst_sync()

            # Entropy after should be lower or equal
            assert result["internal_entropy_after"] <= result["internal_entropy_before"]


class TestBurstReceipt:
    """Test burst sync receipt emission."""

    def test_burst_receipt(self):
        """Burst sync should emit receipt with efficiency metrics."""
        for i in range(5):
            append(
                project="neuron",
                task=f"task {i}",
                next_action="process",
                salience=0.2
            )

        ledger = load_ledger()
        old_ts = (datetime.now(timezone.utc) - timedelta(days=30)).strftime("%Y-%m-%dT%H:%M:%SZ")
        for e in ledger:
            e["ts"] = old_ts
        _write_ledger(ledger)

        classification = classify_for_pump(load_ledger())

        if classification["export"]:
            enable_burst_mode(latency_ms=50000)
            accumulate_for_burst(classification["export"])
            result = burst_sync()

            assert "receipt" in result
            assert result["receipt"]["type"] == "burst_sync"
            assert "efficiency_ratio" in result["receipt"]
            assert "sync_duration_ms" in result["receipt"]


class TestEmptyBurstQueue:
    """Test burst sync with empty queue."""

    def test_empty_burst_queue(self):
        """Empty queue should return zeros."""
        reset_burst_state()  # Ensure empty queue

        result = burst_sync()

        assert result["entries_synced"] == 0
        assert result["efficiency_ratio"] == 0.0


class TestMaxBatchEnforced:
    """Test max batch size is enforced."""

    def test_max_batch_enforced(self):
        """Queue should not exceed max batch size."""
        # Try to accumulate more than max
        entries = [
            {"ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
             "project": "neuron", "salience": 0.5}
            for _ in range(BURST_SYNC_MAX_BATCH + 100)
        ]

        result = accumulate_for_burst(entries)

        assert result["queue_size"] <= BURST_SYNC_MAX_BATCH
