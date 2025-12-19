"""Tests for NEURON v2 shared ledger."""

import json
import os
import tempfile
from pathlib import Path

import pytest

# Set up test ledger path before importing neuron
TEST_LEDGER = Path(tempfile.gettempdir()) / "test_receipts.jsonl"
os.environ["NEURON_LEDGER"] = str(TEST_LEDGER)

from neuron import dual_hash, append, replay, alpha, ALLOWED_PROJECTS


@pytest.fixture(autouse=True)
def clean_ledger():
    """Remove test ledger before and after each test."""
    if TEST_LEDGER.exists():
        TEST_LEDGER.unlink()
    yield
    if TEST_LEDGER.exists():
        TEST_LEDGER.unlink()


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


class TestAppend:
    def test_basic_append(self):
        entry = append("neuron", "test task", "next action", "abc123")
        assert entry["project"] == "neuron"
        assert entry["task"] == "test task"
        assert entry["next"] == "next action"
        assert entry["commit"] == "abc123"
        assert "ts" in entry
        assert "hash" in entry

    def test_invalid_project(self):
        with pytest.raises(ValueError):
            append("invalid", "task", "next", None)

    def test_truncation(self):
        long_task = "x" * 100
        entry = append("neuron", long_task, "next", None)
        assert len(entry["task"]) == 50

    def test_ledger_file_created(self):
        append("neuron", "task", "next", None)
        assert TEST_LEDGER.exists()


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
