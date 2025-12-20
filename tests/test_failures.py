"""
NEURON v4.1 Fault Injection Test Suite
Tests for stress.py inject_failure() function
"""

import os
import sys
import tempfile
from pathlib import Path

import pytest

# Set up test paths before importing
TEST_LEDGER = Path(tempfile.gettempdir()) / "test_failures_receipts.jsonl"
TEST_ARCHIVE = Path(tempfile.gettempdir()) / "test_failures_archive.jsonl"
TEST_STRESS_RECEIPTS = Path(tempfile.gettempdir()) / "test_failures_stress_log.jsonl"
os.environ["NEURON_LEDGER"] = str(TEST_LEDGER)
os.environ["NEURON_ARCHIVE"] = str(TEST_ARCHIVE)
os.environ["NEURON_STRESS_RECEIPTS"] = str(TEST_STRESS_RECEIPTS)

sys.path.insert(0, str(Path(__file__).parent.parent))

from stress import inject_failure, _get_stress_receipts_path
from neuron import FAILURE_TYPES, RECOVERY_SUCCESS_THRESHOLD


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


class TestInjectFailureBasic:
    def test_inject_failure_returns_expected_fields(self):
        """Test inject_failure returns all expected fields."""
        result = inject_failure(failure_type="timeout", rate=0.1, operations=50)

        assert "failure_type" in result
        assert "injection_rate" in result
        assert "total_operations" in result
        assert "failures_injected" in result
        assert "successful_recoveries" in result
        assert "recovery_rate" in result
        assert "slo_pass" in result

    def test_inject_failure_timeout(self):
        """Test timeout failure injection."""
        result = inject_failure(failure_type="timeout", rate=0.2, operations=50)

        assert result["failure_type"] == "timeout"
        assert result["total_operations"] == 50
        # With 20% rate, expect some failures
        assert result["failures_injected"] > 0

    def test_inject_failure_disconnect(self):
        """Test disconnect failure injection."""
        result = inject_failure(failure_type="disconnect", rate=0.2, operations=50)

        assert result["failure_type"] == "disconnect"
        assert result["recovery_rate"] >= 0

    def test_inject_failure_corrupt(self):
        """Test corrupt data failure injection."""
        result = inject_failure(failure_type="corrupt", rate=0.2, operations=50)

        assert result["failure_type"] == "corrupt"
        assert result["recovery_rate"] >= 0

    def test_inject_failure_slow(self):
        """Test slow response failure injection."""
        result = inject_failure(failure_type="slow", rate=0.2, operations=50)

        assert result["failure_type"] == "slow"
        assert result["recovery_rate"] >= 0


class TestInjectFailureValidation:
    def test_invalid_failure_type_raises(self):
        """Test that invalid failure type raises ValueError."""
        with pytest.raises(ValueError) as excinfo:
            inject_failure(failure_type="invalid_type", rate=0.1, operations=10)

        assert "failure_type must be one of" in str(excinfo.value)

    def test_all_failure_types_supported(self):
        """Test all FAILURE_TYPES are supported."""
        for ftype in FAILURE_TYPES:
            result = inject_failure(failure_type=ftype, rate=0.1, operations=20)
            assert result["failure_type"] == ftype


class TestRecoveryRates:
    def test_high_recovery_rate_with_low_failure_rate(self):
        """Test that low failure rate leads to high recovery."""
        result = inject_failure(failure_type="timeout", rate=0.05, operations=100)

        # With low failure rate, should have high recovery
        assert result["recovery_rate"] >= 0.8

    def test_recovery_times_recorded(self):
        """Test that recovery times are recorded."""
        result = inject_failure(failure_type="slow", rate=0.3, operations=50)

        if result["failures_injected"] > 0:
            assert result["average_recovery_time_ms"] >= 0
            assert result["max_recovery_time_ms"] >= 0

    def test_zero_failure_rate(self):
        """Test with zero failure rate (no failures injected)."""
        result = inject_failure(failure_type="timeout", rate=0.0, operations=50)

        assert result["failures_injected"] == 0
        assert result["recovery_rate"] == 1.0  # Perfect when no failures


class TestSLOValidation:
    def test_slo_pass_with_high_recovery(self):
        """Test SLO passes with high recovery rate."""
        result = inject_failure(failure_type="timeout", rate=0.01, operations=100)

        # Very low failure rate should pass SLO
        if result["recovery_rate"] >= RECOVERY_SUCCESS_THRESHOLD:
            assert result["slo_pass"] is True

    def test_slo_logic_consistency(self):
        """Test SLO pass/fail logic is consistent."""
        result = inject_failure(failure_type="disconnect", rate=0.1, operations=50)

        if result["recovery_rate"] >= RECOVERY_SUCCESS_THRESHOLD:
            assert result["slo_pass"] is True
        else:
            assert result["slo_pass"] is False


class TestReceiptEmission:
    def test_failure_injection_emits_receipt(self):
        """Test that inject_failure emits a receipt."""
        inject_failure(failure_type="timeout", rate=0.1, operations=30)

        receipts_path = get_test_stress_receipts_path()
        assert receipts_path.exists()
        with open(receipts_path) as f:
            content = f.read()
        assert "failure_injection_receipt" in content

    def test_receipt_contains_failure_data(self):
        """Test receipt contains failure injection data."""
        import json

        inject_failure(failure_type="corrupt", rate=0.15, operations=40)

        receipts_path = get_test_stress_receipts_path()
        with open(receipts_path) as f:
            receipt = json.loads(f.readline())

        assert receipt["type"] == "failure_injection_receipt"
        assert receipt["failure_type"] == "corrupt"
        assert receipt["injection_rate"] == 0.15


class TestDurationLimits:
    def test_respects_duration_limit(self):
        """Test that injection respects duration limit."""
        import time

        start = time.time()
        result = inject_failure(
            failure_type="timeout", rate=0.5, duration_s=2, operations=1000
        )
        elapsed = time.time() - start

        # Should complete within reasonable time of duration limit
        assert elapsed < 10  # Allow some overhead

    def test_respects_operations_limit(self):
        """Test that injection respects operations limit."""
        result = inject_failure(failure_type="slow", rate=0.1, operations=25)

        assert result["total_operations"] == 25


class TestIsolation:
    def test_uses_isolated_ledger(self):
        """Test that inject_failure uses isolated ledger."""
        import neuron

        original_path = neuron.LEDGER_PATH

        inject_failure(failure_type="timeout", rate=0.1, operations=50)

        # Original path should be restored
        assert neuron.LEDGER_PATH == original_path


class TestAllFailureTypes:
    @pytest.mark.parametrize("failure_type", FAILURE_TYPES)
    def test_failure_type_completes(self, failure_type):
        """Test each failure type completes successfully."""
        result = inject_failure(failure_type=failure_type, rate=0.1, operations=30)

        assert result["total_operations"] == 30
        assert "recovery_rate" in result
        assert 0 <= result["recovery_rate"] <= 1

    @pytest.mark.parametrize("failure_type", FAILURE_TYPES)
    def test_failure_type_recovers(self, failure_type):
        """Test each failure type has reasonable recovery."""
        result = inject_failure(failure_type=failure_type, rate=0.1, operations=50)

        # Should have some recovery capability
        if result["failures_injected"] > 0:
            assert result["successful_recoveries"] >= 0
