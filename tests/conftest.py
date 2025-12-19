"""
NEURON Test Configuration - CLAUDEME Compliant
Shared fixtures and configuration for all tests.
"""

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pytest


# Create temp paths for test isolation
@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Configure test environment with isolated ledger paths."""
    temp_dir = tempfile.mkdtemp(prefix="neuron_test_")

    os.environ["NEURON_LEDGER"] = str(Path(temp_dir) / "test_receipts.jsonl")
    os.environ["NEURON_ARCHIVE"] = str(Path(temp_dir) / "test_archive.jsonl")
    os.environ["NEURON_RECEIPTS"] = str(Path(temp_dir) / "test_emit_receipts.jsonl")
    os.environ["NEURON_STRESS_RECEIPTS"] = str(Path(temp_dir) / "test_stress_receipts.jsonl")

    yield temp_dir

    # Cleanup
    for f in Path(temp_dir).glob("*.jsonl"):
        try:
            f.unlink()
        except Exception:
            pass


@pytest.fixture
def clean_ledger(setup_test_environment):
    """Clean ledger files before and after each test."""
    temp_dir = setup_test_environment
    ledger_files = [
        Path(temp_dir) / "test_receipts.jsonl",
        Path(temp_dir) / "test_archive.jsonl",
        Path(temp_dir) / "test_emit_receipts.jsonl",
        Path(temp_dir) / "test_stress_receipts.jsonl",
    ]

    for path in ledger_files:
        if path.exists():
            path.unlink()

    yield

    for path in ledger_files:
        if path.exists():
            path.unlink()


@pytest.fixture
def sample_entry():
    """Provide a sample ledger entry for testing."""
    return {
        "ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "project": "neuron",
        "model": "neuron",
        "commit": "test123",
        "task": "test task",
        "next": "test next",
        "salience": 1.0,
        "replay_count": 0,
        "energy": 1.0,
        "token_count": 1000,
        "inference_id": "inf_test",
        "context_summary": "Test context"
    }


@pytest.fixture
def temp_ledger_path(tmp_path):
    """Provide a temporary ledger path."""
    return tmp_path / "temp_receipts.jsonl"


# Markers for test categorization
def pytest_configure(config):
    """Configure custom markers."""
    config.addinivalue_line("markers", "slow: mark test as slow running")
    config.addinivalue_line("markers", "stress: mark test as stress/load test")
    config.addinivalue_line("markers", "integration: mark test as integration test")
    config.addinivalue_line("markers", "claudeme: mark test as CLAUDEME compliance test")
