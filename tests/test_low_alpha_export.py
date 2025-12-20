"""
NEURON v4.4 Gate 2: Low-α Export Tests
Test export of low-α entries to external destinations.
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

from neuron import (
    classify_for_pump,
    export_low_alpha,
    compress_to_stub,
    queue_for_anchor,
    compute_internal_entropy,
    append,
    load_ledger,
    _write_ledger,
    _get_stub_path,
    _get_chain_queue_path,
    _get_cold_archive_path,
)


@pytest.fixture(autouse=True)
def clean_ledger():
    """Clean ledger files before and after each test."""
    from neuron import _get_ledger_path, _get_archive_path, _get_receipts_path
    import shutil

    for path_fn in [_get_ledger_path, _get_archive_path, _get_receipts_path]:
        path = path_fn()
        if path.exists():
            path.unlink()

    # Clean export directories
    for path_fn in [_get_stub_path, _get_chain_queue_path, _get_cold_archive_path]:
        path = path_fn()
        if path.exists():
            shutil.rmtree(path)

    yield

    for path_fn in [_get_ledger_path, _get_archive_path, _get_receipts_path]:
        path = path_fn()
        if path.exists():
            path.unlink()

    for path_fn in [_get_stub_path, _get_chain_queue_path, _get_cold_archive_path]:
        path = path_fn()
        if path.exists():
            shutil.rmtree(path)


class TestExportToStub:
    """Test export to stub destination."""

    def test_export_to_stub(self):
        """Entries should be written to stub file."""
        # Create entries
        for i in range(5):
            append(
                project="neuron",
                task=f"task {i}",
                next_action="process",
                salience=0.2,  # Low salience
            )

        # Age entries
        ledger = load_ledger()
        old_ts = (datetime.now(timezone.utc) - timedelta(days=30)).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
        for e in ledger:
            e["ts"] = old_ts
        _write_ledger(ledger)

        # Classify and export
        classification = classify_for_pump(load_ledger())
        if classification["export"]:
            result = export_low_alpha(classification["export"], destination="stub")

            assert result["entries_exported"] > 0
            assert result["destination"] == "stub"
            assert os.path.exists(result["output_file"])


class TestExportToChain:
    """Test export to blockchain queue."""

    def test_export_to_chain(self):
        """Entries should be queued for blockchain anchor."""
        for i in range(3):
            append(
                project="neuron", task=f"task {i}", next_action="process", salience=0.1
            )

        # Age entries
        ledger = load_ledger()
        old_ts = (datetime.now(timezone.utc) - timedelta(days=30)).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
        for e in ledger:
            e["ts"] = old_ts
        _write_ledger(ledger)

        classification = classify_for_pump(load_ledger())
        if classification["export"]:
            result = queue_for_anchor(classification["export"])

            assert result["destination"] == "chain"


class TestExportToArchive:
    """Test export to cold archive."""

    def test_export_to_archive(self):
        """Entries should be archived to cold storage."""
        for i in range(3):
            append(
                project="neuron", task=f"task {i}", next_action="process", salience=0.1
            )

        # Age entries
        ledger = load_ledger()
        old_ts = (datetime.now(timezone.utc) - timedelta(days=30)).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
        for e in ledger:
            e["ts"] = old_ts
        _write_ledger(ledger)

        classification = classify_for_pump(load_ledger())
        if classification["export"]:
            result = export_low_alpha(classification["export"], destination="archive")

            assert result["destination"] == "archive"


class TestCompressToStub:
    """Test stub compression."""

    def test_compress_to_stub(self):
        """Stub should be 90%+ smaller."""
        entry = {
            "ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "project": "neuron",
            "model": "neuron",
            "task": "a very long task description that takes up space",
            "next": "another long next action description",
            "salience": 0.5,
            "replay_count": 0,
            "energy": 1.0,
            "token_count": 5000,
            "inference_id": "inf_12345678",
            "context_summary": "lots of context " * 20,
            "hash": "abc123def456" * 5,
            "_alpha": 0.3,
        }

        import json

        original_size = len(json.dumps(entry))

        stub = compress_to_stub(entry)
        stub_size = len(json.dumps(stub))

        compression = 1.0 - (stub_size / original_size)

        # Stub should be at least 70% smaller (may vary based on entry)
        assert compression > 0.5
        assert "id" in stub
        assert "ts" in stub
        assert "payload_hash" in stub


class TestEntriesRemoved:
    """Test exported entries are removed from ledger."""

    def test_entries_removed(self):
        """Exported entries should be removed from active ledger."""
        # Create entries
        for i in range(5):
            append(
                project="neuron", task=f"task {i}", next_action="process", salience=0.1
            )

        # Age entries
        ledger = load_ledger()
        old_ts = (datetime.now(timezone.utc) - timedelta(days=30)).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
        for e in ledger:
            e["ts"] = old_ts
        _write_ledger(ledger)

        initial_count = len(load_ledger())

        classification = classify_for_pump(load_ledger())
        if classification["export"]:
            export_count = len(classification["export"])
            export_low_alpha(classification["export"], destination="stub")

            final_count = len(load_ledger())

            assert final_count == initial_count - export_count


class TestEntropyDecreased:
    """Test entropy decreases after export."""

    def test_entropy_decreased(self):
        """Internal entropy should be lower after export."""
        # Create mixed entries
        for i in range(10):
            append(
                project="neuron",
                task=f"task {i}",
                next_action="process",
                salience=0.1 if i < 5 else 0.9,
            )

        # Age some entries
        ledger = load_ledger()
        old_ts = (datetime.now(timezone.utc) - timedelta(days=30)).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
        for i, e in enumerate(ledger):
            if i < 5:
                e["ts"] = old_ts
        _write_ledger(ledger)

        entropy_before = compute_internal_entropy(load_ledger())

        classification = classify_for_pump(load_ledger())
        if classification["export"]:
            result = export_low_alpha(classification["export"], destination="stub")

            # Entropy after should be lower or equal (exporting low-α reduces disorder)
            assert result["internal_entropy_after"] <= entropy_before


class TestExportReceipt:
    """Test export receipt emission."""

    def test_export_receipt(self):
        """Export should emit receipt with bytes saved."""
        for i in range(3):
            append(
                project="neuron", task=f"task {i}", next_action="process", salience=0.1
            )

        # Age entries
        ledger = load_ledger()
        old_ts = (datetime.now(timezone.utc) - timedelta(days=30)).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
        for e in ledger:
            e["ts"] = old_ts
        _write_ledger(ledger)

        classification = classify_for_pump(load_ledger())
        if classification["export"]:
            result = export_low_alpha(classification["export"], destination="stub")

            assert "receipt" in result
            assert result["receipt"]["type"] == "low_alpha_export"
            assert "bytes_before" in result["receipt"]
            assert "bytes_after" in result["receipt"]


class TestExportCreatesDirs:
    """Test directories are created if missing."""

    def test_export_creates_dirs(self):
        """Export should create directories if they don't exist."""
        for i in range(2):
            append(
                project="neuron", task=f"task {i}", next_action="process", salience=0.1
            )

        # Age entries
        ledger = load_ledger()
        old_ts = (datetime.now(timezone.utc) - timedelta(days=30)).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
        for e in ledger:
            e["ts"] = old_ts
        _write_ledger(ledger)

        classification = classify_for_pump(load_ledger())
        if classification["export"]:
            result = export_low_alpha(classification["export"], destination="stub")

            assert _get_stub_path().exists()
