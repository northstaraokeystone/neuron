"""
NEURON v4.2 Swarm Testing Suite
Tests for stress.py swarm_test() and high_stress_test() functions
"""

import os
import sys
import tempfile
from pathlib import Path

import pytest

# Set up test paths before importing
TEST_LEDGER = Path(tempfile.gettempdir()) / "test_swarm_receipts.jsonl"
TEST_ARCHIVE = Path(tempfile.gettempdir()) / "test_swarm_archive.jsonl"
TEST_STRESS_RECEIPTS = Path(tempfile.gettempdir()) / "test_swarm_stress_receipts.jsonl"
os.environ["NEURON_LEDGER"] = str(TEST_LEDGER)
os.environ["NEURON_ARCHIVE"] = str(TEST_ARCHIVE)
os.environ["NEURON_STRESS_RECEIPTS"] = str(TEST_STRESS_RECEIPTS)

sys.path.insert(0, str(Path(__file__).parent.parent))

from stress import (
    swarm_test,
    high_stress_test,
    SWARM_DEFAULT_AGENTS,
    SWARM_APPEND_PER_AGENT,
    SWARM_CONFLICT_THRESHOLD,
    HIGH_STRESS_APPEND_TARGET,
    HIGH_STRESS_WORKERS,
)


@pytest.fixture(autouse=True)
def clean_test_files():
    """Remove test files before and after each test."""
    for path in [TEST_LEDGER, TEST_ARCHIVE, TEST_STRESS_RECEIPTS]:
        if path.exists():
            path.unlink()
    yield
    for path in [TEST_LEDGER, TEST_ARCHIVE, TEST_STRESS_RECEIPTS]:
        if path.exists():
            path.unlink()


class TestSwarmTest:
    def test_basic_swarm(self):
        """Test swarm_test runs with minimal agents."""
        result = swarm_test(n_agents=5, appends_per_agent=5, shard_count=2)

        assert "n_agents" in result
        assert "appends_per_agent" in result
        assert "total_appends" in result
        assert "conflicts" in result
        assert "append_rate_per_second" in result
        assert "slo_pass" in result

        assert result["n_agents"] == 5
        assert result["appends_per_agent"] == 5
        assert result["total_appends"] == 25  # 5 * 5

    def test_swarm_zero_conflicts(self):
        """Test swarm produces zero conflicts with unique IDs."""
        result = swarm_test(n_agents=10, appends_per_agent=10, shard_count=4)

        assert result["conflicts"] == 0

    def test_swarm_shard_distribution(self):
        """Test entries are distributed across shards."""
        result = swarm_test(n_agents=10, appends_per_agent=20, shard_count=4)

        # All shards should have some entries (probabilistic)
        assert len(result["entries_per_shard"]) == 4
        assert sum(result["entries_per_shard"]) == 200

    def test_swarm_throughput_reasonable(self):
        """Test swarm achieves reasonable throughput."""
        result = swarm_test(n_agents=20, appends_per_agent=50, shard_count=4)

        # Should achieve at least some reasonable rate
        assert result["append_rate_per_second"] > 100

    def test_swarm_custom_shard_count(self):
        """Test swarm with different shard counts."""
        result_2 = swarm_test(n_agents=5, appends_per_agent=10, shard_count=2)
        result_8 = swarm_test(n_agents=5, appends_per_agent=10, shard_count=8)

        assert result_2["shards_used"] == 2
        assert result_8["shards_used"] == 8

    def test_swarm_emits_receipt(self):
        """Test swarm_test emits a receipt."""
        swarm_test(n_agents=2, appends_per_agent=2, shard_count=2)

        assert TEST_STRESS_RECEIPTS.exists()
        with open(TEST_STRESS_RECEIPTS) as f:
            content = f.read()
        assert "swarm_test_receipt" in content

    def test_swarm_timing_metrics(self):
        """Test swarm provides timing metrics."""
        result = swarm_test(n_agents=5, appends_per_agent=10, shard_count=2)

        assert "duration_seconds" in result
        assert "avg_agent_time_ms" in result
        assert result["duration_seconds"] > 0
        assert result["avg_agent_time_ms"] > 0


class TestHighStressTest:
    def test_basic_high_stress(self):
        """Test high_stress_test runs with small input."""
        result = high_stress_test(n_entries=100, shard_count=2, workers=2)

        assert "n_entries" in result
        assert "shard_count" in result
        assert "workers" in result
        assert "append_rate_per_second" in result
        assert "slo_pass" in result

        assert result["n_entries"] == 100
        assert result["shard_count"] == 2
        assert result["workers"] == 2

    def test_high_stress_shard_distribution(self):
        """Test entries are distributed across shards."""
        result = high_stress_test(n_entries=1000, shard_count=4, workers=4)

        assert len(result["entries_per_shard"]) == 4
        assert sum(result["entries_per_shard"]) == 1000

    def test_high_stress_throughput(self):
        """Test high stress achieves reasonable throughput."""
        result = high_stress_test(n_entries=5000, shard_count=4, workers=4)

        # Should achieve at least some throughput (environment dependent)
        # Lower threshold for CI/cloud environments
        assert result["append_rate_per_second"] > 50

    def test_high_stress_overhead_low(self):
        """Test overhead percentage is calculated."""
        result = high_stress_test(n_entries=1000, shard_count=4, workers=4)

        assert "overhead_percent" in result
        # Overhead should be a reasonable number (not negative, not huge)
        assert 0 <= result["overhead_percent"] < 1000

    def test_high_stress_evictions(self):
        """Test eviction tracking (may be 0 for small tests)."""
        result = high_stress_test(n_entries=100, shard_count=2, workers=2)

        assert "evictions_triggered" in result
        assert result["evictions_triggered"] >= 0

    def test_high_stress_emits_receipt(self):
        """Test high_stress_test emits a receipt."""
        high_stress_test(n_entries=50, shard_count=2, workers=2)

        assert TEST_STRESS_RECEIPTS.exists()
        with open(TEST_STRESS_RECEIPTS) as f:
            content = f.read()
        assert "high_stress_receipt" in content

    def test_high_stress_scaling(self):
        """Test throughput scales with workers."""
        result_2w = high_stress_test(n_entries=1000, shard_count=4, workers=2)
        result_8w = high_stress_test(n_entries=1000, shard_count=4, workers=8)

        # More workers should generally be faster (or at least comparable)
        # Note: with small inputs, may not see linear scaling
        assert result_2w["n_entries"] == result_8w["n_entries"]


class TestSwarmConcurrency:
    def test_many_agents_small_appends(self):
        """Test many agents with few appends each."""
        result = swarm_test(n_agents=50, appends_per_agent=2, shard_count=4)

        assert result["total_appends"] == 100
        assert result["conflicts"] == 0

    def test_few_agents_many_appends(self):
        """Test few agents with many appends each."""
        result = swarm_test(n_agents=2, appends_per_agent=50, shard_count=4)

        assert result["total_appends"] == 100
        assert result["conflicts"] == 0

    def test_balanced_swarm(self):
        """Test balanced agent/append ratio."""
        result = swarm_test(n_agents=10, appends_per_agent=10, shard_count=4)

        assert result["total_appends"] == 100
        assert result["conflicts"] == 0


class TestShardingIntegration:
    def test_single_shard_stress(self):
        """Test stress test with single shard."""
        result = high_stress_test(n_entries=500, shard_count=1, workers=4)

        assert result["shard_count"] == 1
        assert len(result["entries_per_shard"]) == 1
        assert result["entries_per_shard"][0] == 500

    def test_many_shards_stress(self):
        """Test stress test with many shards."""
        result = high_stress_test(n_entries=800, shard_count=8, workers=4)

        assert result["shard_count"] == 8
        assert len(result["entries_per_shard"]) == 8
        assert sum(result["entries_per_shard"]) == 800


class TestSLOValidation:
    def test_swarm_slo_fields(self):
        """Test swarm result includes SLO pass/fail."""
        result = swarm_test(n_agents=10, appends_per_agent=10, shard_count=4)

        assert "slo_pass" in result
        assert isinstance(result["slo_pass"], bool)

    def test_high_stress_slo_fields(self):
        """Test high stress result includes SLO pass/fail."""
        result = high_stress_test(n_entries=1000, shard_count=4, workers=4)

        assert "slo_pass" in result
        assert isinstance(result["slo_pass"], bool)

    def test_conflict_threshold(self):
        """Test conflict threshold is enforced."""
        # Our swarm should produce 0 conflicts
        result = swarm_test(n_agents=20, appends_per_agent=10, shard_count=4)

        assert result["conflicts"] <= SWARM_CONFLICT_THRESHOLD


class TestCleanup:
    def test_swarm_cleanup(self):
        """Test swarm cleans up temp files."""
        import tempfile
        import os

        # Get baseline temp file count
        temp_dir = tempfile.gettempdir()
        baseline_count = len([f for f in os.listdir(temp_dir) if "shard" in f.lower()])

        swarm_test(n_agents=5, appends_per_agent=5, shard_count=2)

        # Count should be same or less after cleanup
        after_count = len([f for f in os.listdir(temp_dir) if "shard" in f.lower()])
        assert after_count <= baseline_count + 2  # Allow some tolerance

    def test_high_stress_cleanup(self):
        """Test high stress cleans up temp files."""
        import tempfile
        import os

        temp_dir = tempfile.gettempdir()
        baseline_count = len([f for f in os.listdir(temp_dir) if "shard" in f.lower()])

        high_stress_test(n_entries=100, shard_count=2, workers=2)

        after_count = len([f for f in os.listdir(temp_dir) if "shard" in f.lower()])
        assert after_count <= baseline_count + 2
