"""
NEURON Test Configuration - CLAUDEME Compliant
Shared fixtures and configuration for all tests.

NOTE: Environment variables are set by individual test files before importing modules.
This conftest only provides helpers and markers, not global env setup, to avoid
conflicts with test-file-specific paths.
"""

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pytest


@pytest.fixture
def clean_ledger():
    """Clean ledger files before and after each test using current env paths."""
    from neuron import _get_ledger_path, _get_archive_path, _get_receipts_path

    ledger_files = [
        _get_ledger_path(),
        _get_archive_path(),
        _get_receipts_path(),
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
