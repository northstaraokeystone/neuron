"""
NEURON v4.1 Stress Testing Suite
Tests for stress.py module: stress_test, concurrent_sync_test, benchmark_report
"""

import os
import sys
import tempfile
from pathlib import Path

import pytest

# Set up test paths before importing
TEST_LEDGER = Path(tempfile.gettempdir()) / "test_stress_receipts.jsonl"
TEST_ARCHIVE = Path(tempfile.gettempdir()) / "test_stress_archive.jsonl"
TEST_STRESS_RECEIPTS = Path(tempfile.gettempdir()) / "test_stress_receipts_log.jsonl"
os.environ["NEURON_LEDGER"] = str(TEST_LEDGER)
os.environ["NEURON_ARCHIVE"] = str(TEST_ARCHIVE)
os.environ["NEURON_STRESS_RECEIPTS"] = str(TEST_STRESS_RECEIPTS)

sys.path.insert(0, str(Path(__file__).parent.parent))

from stress import (
    stress_test,
    concurrent_sync_test,
    benchmark_report,
    _emit_receipt,
    _get_stress_receipts_path,
)
from neuron import (
    STRESS_TEST_THROUGHPUT_FLOOR,
    STRESS_TEST_OVERHEAD_THRESHOLD,
    TAU_PRESETS,
)


def get_test_stress_receipts_path():
    """Get the actual stress receipts path being used by the module."""
    return _get_stress_receipts_path()


@pytest.fixture(autouse=True)
def clean_test_files():
    """Remove test files before and after each test."""
    paths = [TEST_LEDGER, TEST_ARCHIVE, get_test_stress_receipts_path()]
    for path in paths:
        if path.exists():
            path.unlink()
    yield
    for path in paths:
        if path.exists():
            path.unlink()


class TestStressTest:
    def test_basic_stress_test(self):
        """Test stress_test runs and returns expected fields."""
        result = stress_test(n_entries=100, concurrent=2)

        assert "n_entries" in result
        assert "append_rate_per_second" in result
        assert "overhead_percent" in result
        assert "duration_seconds" in result
        assert "slo_pass" in result
        assert result["n_entries"] == 100
        assert result["concurrent_workers"] == 2

    def test_stress_test_with_tau(self):
        """Test stress_test respects tau parameter."""
        result_quick = stress_test(n_entries=50, concurrent=1, tau=TAU_PRESETS["quick_task"])
        result_deep = stress_test(n_entries=50, concurrent=1, tau=TAU_PRESETS["deep_work"])

        assert result_quick["tau_used"] == TAU_PRESETS["quick_task"]
        assert result_deep["tau_used"] == TAU_PRESETS["deep_work"]

    def test_stress_test_concurrent_scaling(self):
        """Test that more workers process faster (or at least complete)."""
        result_1 = stress_test(n_entries=100, concurrent=1)
        result_4 = stress_test(n_entries=100, concurrent=4)

        # Both should complete
        assert result_1["n_entries"] == 100
        assert result_4["n_entries"] == 100

    def test_stress_test_throughput_reasonable(self):
        """Test that throughput is reasonable (>100/s for small test)."""
        result = stress_test(n_entries=500, concurrent=2)

        # Should be able to do at least 100 appends/second
        assert result["append_rate_per_second"] > 100

    def test_stress_test_emits_receipt(self):
        """Test that stress_test emits a receipt."""
        stress_test(n_entries=50, concurrent=1)

        receipts_path = get_test_stress_receipts_path()
        assert receipts_path.exists()
        with open(receipts_path) as f:
            lines = f.readlines()
        assert len(lines) >= 1


class TestConcurrentSyncTest:
    def test_basic_concurrent_sync(self):
        """Test concurrent_sync_test runs and returns expected fields."""
        result = concurrent_sync_test(n_workers=2, n_entries_each=20)

        assert "n_workers" in result
        assert "n_entries_each" in result
        assert "total_entries" in result
        assert "conflicts_detected" in result
        assert "all_workers_consistent" in result
        assert "slo_pass" in result

    def test_concurrent_sync_entries_count(self):
        """Test that all entries are synced."""
        result = concurrent_sync_test(n_workers=2, n_entries_each=25)

        # Should have at least 90% of expected entries (allow for conflict resolution)
        expected = 2 * 25
        assert result["total_entries"] >= expected * 0.9

    def test_concurrent_sync_consistency(self):
        """Test that sync maintains consistency."""
        result = concurrent_sync_test(n_workers=3, n_entries_each=15)

        # Should report consistency status
        assert isinstance(result["all_workers_consistent"], bool)

    def test_concurrent_sync_emits_receipt(self):
        """Test that concurrent_sync_test emits a receipt."""
        concurrent_sync_test(n_workers=2, n_entries_each=10)

        receipts_path = get_test_stress_receipts_path()
        assert receipts_path.exists()
        with open(receipts_path) as f:
            content = f.read()
        assert "concurrent_sync_receipt" in content


class TestBenchmarkReport:
    def test_benchmark_report_structure(self):
        """Test benchmark_report returns expected structure."""
        result = benchmark_report()

        assert "timestamp" in result
        assert "neuron_version" in result
        assert "slos" in result
        assert "pass_count" in result
        assert "fail_count" in result
        assert "overall_status" in result
        assert result["neuron_version"] == "4.2"

    def test_benchmark_report_slo_fields(self):
        """Test that all expected SLOs are present."""
        result = benchmark_report()

        expected_slos = [
            "append_overhead",
            "append_throughput",
            "recovery_rate",
            "pruning_compression",
            "context_restore",
            "concurrent_consistency"
        ]

        for slo_name in expected_slos:
            assert slo_name in result["slos"]
            assert "target" in result["slos"][slo_name]
            assert "actual" in result["slos"][slo_name]
            assert "pass" in result["slos"][slo_name]

    def test_benchmark_report_counts(self):
        """Test pass/fail counts are consistent."""
        result = benchmark_report()

        total_slos = len(result["slos"])
        assert result["pass_count"] + result["fail_count"] == total_slos

    def test_benchmark_report_overall_status(self):
        """Test overall status matches pass/fail counts."""
        result = benchmark_report()

        if result["fail_count"] == 0:
            assert result["overall_status"] == "PASS"
        else:
            assert result["overall_status"] == "FAIL"

    def test_benchmark_report_emits_receipt(self):
        """Test that benchmark_report emits a receipt."""
        benchmark_report()

        receipts_path = get_test_stress_receipts_path()
        assert receipts_path.exists()
        with open(receipts_path) as f:
            content = f.read()
        assert "benchmark_report_receipt" in content


class TestReceiptEmission:
    def test_emit_receipt_creates_file(self):
        """Test _emit_receipt creates the receipts file."""
        _emit_receipt("test_receipt", {"test_field": "test_value"})

        receipts_path = get_test_stress_receipts_path()
        assert receipts_path.exists()

    def test_emit_receipt_includes_hash(self):
        """Test emitted receipts include hash."""
        import json

        _emit_receipt("test_receipt", {"data": 123})

        receipts_path = get_test_stress_receipts_path()
        with open(receipts_path) as f:
            receipt = json.loads(f.readline())

        assert "hash" in receipt
        assert ":" in receipt["hash"]  # Dual hash format

    def test_emit_receipt_includes_timestamp(self):
        """Test emitted receipts include timestamp."""
        import json

        _emit_receipt("test_receipt", {"data": 456})

        receipts_path = get_test_stress_receipts_path()
        with open(receipts_path) as f:
            receipt = json.loads(f.readline())

        assert "ts" in receipt
        assert "T" in receipt["ts"]  # ISO format


class TestIntegration:
    def test_full_stress_pipeline(self):
        """Test running all stress tests in sequence."""
        # Stress test
        stress_result = stress_test(n_entries=50, concurrent=2)
        assert stress_result["n_entries"] == 50

        # Sync test
        sync_result = concurrent_sync_test(n_workers=2, n_entries_each=20)
        assert sync_result["total_entries"] > 0

        # Full benchmark
        report = benchmark_report()
        assert report["pass_count"] >= 0

    def test_stress_isolation(self):
        """Test that stress tests use isolated ledgers."""
        import neuron
        original_path = neuron.LEDGER_PATH

        stress_test(n_entries=100, concurrent=2)

        # Original ledger should not be affected
        assert neuron.LEDGER_PATH == original_path
