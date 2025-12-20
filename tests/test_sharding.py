"""
NEURON v4.2 Sharding Test Suite
Tests for sharding.py module: ShardedLedger, routing, eviction, sync
"""

import json
import shutil
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from sharding import (
    ShardedLedger,
    DEFAULT_SHARD_COUNT,
    MAX_SHARD_COUNT,
    SHARD_STRATEGIES,
    DEFAULT_SHARD_STRATEGY,
    SHARD_MAX_ENTRIES,
)


@pytest.fixture
def temp_shard_dir():
    """Create a temporary directory for shards."""
    temp_dir = tempfile.mkdtemp()
    shard_dir = Path(temp_dir) / "shards"
    yield shard_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def sample_entry():
    """Create a sample entry for testing."""
    return {
        "ts": "2025-01-15T10:30:00Z",
        "project": "neuron",
        "model": "neuron",
        "commit": None,
        "task": "test_task",
        "next": "next_action",
        "salience": 1.0,
        "replay_count": 0,
        "energy": 1.0,
        "token_count": 1000,
        "inference_id": "test_inf_001",
        "context_summary": "",
        "hash": "abc123:def456",
    }


class TestShardedLedgerInit:
    def test_default_init(self, temp_shard_dir):
        """Test ShardedLedger initializes with defaults."""
        ledger = ShardedLedger(shard_dir=temp_shard_dir)

        assert ledger.shard_count == DEFAULT_SHARD_COUNT
        assert ledger.strategy == DEFAULT_SHARD_STRATEGY
        assert ledger.max_entries_per_shard == SHARD_MAX_ENTRIES
        assert temp_shard_dir.exists()

    def test_custom_shard_count(self, temp_shard_dir):
        """Test custom shard count."""
        ledger = ShardedLedger(shard_count=8, shard_dir=temp_shard_dir)

        assert ledger.shard_count == 8
        # Check all shard files exist
        for i in range(8):
            assert (temp_shard_dir / f"shard_{i}.jsonl").exists()

    def test_shard_count_limits(self, temp_shard_dir):
        """Test shard count is clamped to valid range."""
        ledger_min = ShardedLedger(shard_count=0, shard_dir=temp_shard_dir)
        assert ledger_min.shard_count == 1

        ledger_max = ShardedLedger(shard_count=100, shard_dir=temp_shard_dir)
        assert ledger_max.shard_count == MAX_SHARD_COUNT

    def test_all_strategies_valid(self, temp_shard_dir):
        """Test all routing strategies are accepted."""
        for strategy in SHARD_STRATEGIES:
            ledger = ShardedLedger(strategy=strategy, shard_dir=temp_shard_dir)
            assert ledger.strategy == strategy


class TestShardRouting:
    def test_hash_routing_deterministic(self, temp_shard_dir, sample_entry):
        """Test hash routing is deterministic."""
        ledger = ShardedLedger(shard_count=4, strategy="hash", shard_dir=temp_shard_dir)

        shard1 = ledger.route(sample_entry)
        shard2 = ledger.route(sample_entry)

        assert shard1 == shard2
        assert 0 <= shard1 < 4

    def test_hash_routing_distribution(self, temp_shard_dir):
        """Test hash routing distributes entries across shards."""
        ledger = ShardedLedger(shard_count=4, strategy="hash", shard_dir=temp_shard_dir)

        shard_counts = {0: 0, 1: 0, 2: 0, 3: 0}
        for i in range(100):
            entry = {"hash": f"test_hash_{i}:suffix_{i}"}
            shard = ledger.route(entry)
            shard_counts[shard] += 1

        # All shards should have some entries (probabilistic)
        assert all(count > 0 for count in shard_counts.values())

    def test_time_routing(self, temp_shard_dir):
        """Test time-based routing."""
        ledger = ShardedLedger(
            shard_count=24, strategy="time", shard_dir=temp_shard_dir
        )

        # Different hours should route to different shards
        entry_10am = {"ts": "2025-01-15T10:30:00Z"}
        entry_14pm = {"ts": "2025-01-15T14:30:00Z"}

        assert ledger.route(entry_10am) == 10
        assert ledger.route(entry_14pm) == 14

    def test_project_routing(self, temp_shard_dir):
        """Test project-based routing."""
        ledger = ShardedLedger(
            shard_count=4, strategy="project", shard_dir=temp_shard_dir
        )

        entry_neuron = {"project": "neuron"}
        entry_axiom = {"project": "axiom"}
        entry_agentproof = {"project": "agentproof"}

        assert ledger.route(entry_neuron) == 2  # index of "neuron"
        assert ledger.route(entry_axiom) == 1  # index of "axiom"
        assert ledger.route(entry_agentproof) == 0  # index of "agentproof"

    def test_model_routing(self, temp_shard_dir):
        """Test model-based routing."""
        ledger = ShardedLedger(
            shard_count=4, strategy="model", shard_dir=temp_shard_dir
        )

        entry_grok = {"model": "grok"}
        entry_claude = {"model": "claude"}

        assert ledger.route(entry_grok) == 0  # index of "grok"
        assert ledger.route(entry_claude) == 1  # index of "claude"


class TestShardAppend:
    def test_append_creates_entry(self, temp_shard_dir, sample_entry):
        """Test append writes entry to shard."""
        ledger = ShardedLedger(shard_count=4, shard_dir=temp_shard_dir)

        result = ledger.append(sample_entry)

        assert "shard_id" in result
        assert 0 <= result["shard_id"] < 4

    def test_append_writes_to_correct_shard(self, temp_shard_dir, sample_entry):
        """Test entry is written to the correct shard file."""
        ledger = ShardedLedger(shard_count=4, shard_dir=temp_shard_dir)

        result = ledger.append(sample_entry)
        shard_id = result["shard_id"]

        shard_path = temp_shard_dir / f"shard_{shard_id}.jsonl"
        with open(shard_path) as f:
            content = f.read()

        assert sample_entry["hash"] in content

    def test_multiple_appends(self, temp_shard_dir):
        """Test multiple appends work correctly."""
        ledger = ShardedLedger(shard_count=4, shard_dir=temp_shard_dir)

        for i in range(100):
            entry = {
                "ts": f"2025-01-15T{i % 24:02d}:00:00Z",
                "hash": f"hash_{i}:suffix",
                "project": "neuron",
            }
            ledger.append(entry)

        stats = ledger.stats()
        assert stats["total_entries"] == 100


class TestShardReplay:
    def test_replay_empty(self, temp_shard_dir):
        """Test replay on empty ledger."""
        ledger = ShardedLedger(shard_count=4, shard_dir=temp_shard_dir)

        result = ledger.replay()
        assert result == []

    def test_replay_aggregates_all_shards(self, temp_shard_dir):
        """Test replay aggregates entries from all shards."""
        ledger = ShardedLedger(shard_count=4, shard_dir=temp_shard_dir)

        # Add entries that will land in different shards
        for i in range(20):
            entry = {"ts": f"2025-01-15T{i:02d}:00:00Z", "hash": f"hash_{i}:suffix"}
            ledger.append(entry)

        result = ledger.replay()
        assert len(result) == 20

    def test_replay_with_limit(self, temp_shard_dir):
        """Test replay respects n limit."""
        ledger = ShardedLedger(shard_count=4, shard_dir=temp_shard_dir)

        for i in range(50):
            entry = {
                "ts": f"2025-01-15T{i % 24:02d}:{i % 60:02d}:00Z",
                "hash": f"h{i}:s",
            }
            ledger.append(entry)

        result = ledger.replay(n=10)
        assert len(result) == 10

    def test_replay_sorted_by_timestamp(self, temp_shard_dir):
        """Test replay returns entries sorted by timestamp."""
        ledger = ShardedLedger(shard_count=4, shard_dir=temp_shard_dir)

        # Add entries in random order
        for i in [5, 2, 8, 1, 9, 3]:
            entry = {"ts": f"2025-01-15T0{i}:00:00Z", "hash": f"h{i}:s"}
            ledger.append(entry)

        result = ledger.replay()
        timestamps = [e["ts"] for e in result]
        assert timestamps == sorted(timestamps)

    def test_replay_since(self, temp_shard_dir):
        """Test replay with since filter."""
        ledger = ShardedLedger(shard_count=4, shard_dir=temp_shard_dir)

        for i in range(10):
            entry = {"ts": f"2025-01-1{i}T00:00:00Z", "hash": f"h{i}:s"}
            ledger.append(entry)

        result = ledger.replay(since="2025-01-15T00:00:00Z")
        assert all(e["ts"] >= "2025-01-15T00:00:00Z" for e in result)


class TestShardEviction:
    def test_eviction_under_limit(self, temp_shard_dir):
        """Test eviction does nothing when under limit."""
        ledger = ShardedLedger(
            shard_count=2, max_entries_per_shard=100, shard_dir=temp_shard_dir
        )

        for i in range(50):
            entry = {"ts": f"2025-01-{i % 28 + 1:02d}T00:00:00Z", "hash": f"h{i}:s"}
            ledger.append(entry)

        result = ledger.evict(0)
        assert result["reason"] == "under_limit"
        assert result["evicted"] == 0

    def test_eviction_over_limit(self, temp_shard_dir):
        """Test eviction when over limit."""
        ledger = ShardedLedger(
            shard_count=1, max_entries_per_shard=10, shard_dir=temp_shard_dir
        )

        for i in range(20):
            entry = {"ts": f"2025-01-{i % 28 + 1:02d}T00:00:00Z", "hash": f"h{i}:s"}
            # Write directly to bypass auto-eviction
            shard_path = temp_shard_dir / "shard_0.jsonl"
            with open(shard_path, "a") as f:
                f.write(json.dumps(entry) + "\n")

        result = ledger.evict(0)
        assert result["reason"] == "over_limit"
        assert result["evicted"] > 0
        assert result["remaining"] <= ledger.max_entries_per_shard

    def test_eviction_archives_entries(self, temp_shard_dir):
        """Test evicted entries are archived."""
        import sharding

        original_archive = sharding.ARCHIVE_PATH
        test_archive = temp_shard_dir.parent / "archive.jsonl"
        sharding.ARCHIVE_PATH = test_archive

        try:
            ledger = ShardedLedger(
                shard_count=1, max_entries_per_shard=5, shard_dir=temp_shard_dir
            )

            for i in range(15):
                entry = {"ts": f"2025-01-{i % 28 + 1:02d}T00:00:00Z", "hash": f"h{i}:s"}
                shard_path = temp_shard_dir / "shard_0.jsonl"
                with open(shard_path, "a") as f:
                    f.write(json.dumps(entry) + "\n")

            ledger.evict(0)

            assert test_archive.exists()
            with open(test_archive) as f:
                archived = [json.loads(line) for line in f if line.strip()]
            assert len(archived) > 0
            assert all("archived_ts" in e for e in archived)

        finally:
            sharding.ARCHIVE_PATH = original_archive


class TestShardSync:
    def test_sync_empty_ledger(self, temp_shard_dir):
        """Test sync on empty ledger."""
        ledger = ShardedLedger(shard_count=4, shard_dir=temp_shard_dir)

        result = ledger.sync()

        assert result["shards_synced"] == 4
        assert result["total_entries"] == 0
        assert result["conflicts"] == 0
        assert "shard_hashes" in result

    def test_sync_with_entries(self, temp_shard_dir):
        """Test sync with entries."""
        ledger = ShardedLedger(shard_count=4, shard_dir=temp_shard_dir)

        for i in range(50):
            entry = {"ts": f"2025-01-15T{i % 24:02d}:00:00Z", "hash": f"h{i}:s"}
            ledger.append(entry)

        result = ledger.sync()

        assert result["total_entries"] == 50
        assert len(result["shard_hashes"]) == 4


class TestShardRebalance:
    def test_rebalance_empty(self, temp_shard_dir):
        """Test rebalance on empty ledger."""
        ledger = ShardedLedger(shard_count=4, shard_dir=temp_shard_dir)

        result = ledger.rebalance()
        assert result["rebalanced"] is False
        assert result["reason"] == "no_entries"

    def test_rebalance_redistributes(self, temp_shard_dir):
        """Test rebalance redistributes entries."""
        ledger = ShardedLedger(shard_count=4, strategy="hash", shard_dir=temp_shard_dir)

        for i in range(100):
            entry = {"ts": "2025-01-15T00:00:00Z", "hash": f"h{i}:s"}
            ledger.append(entry)

        result = ledger.rebalance()

        assert result["rebalanced"] is True
        assert result["total_entries"] == 100
        assert sum(result["entries_per_shard"]) == 100


class TestShardStats:
    def test_stats_empty(self, temp_shard_dir):
        """Test stats on empty ledger."""
        ledger = ShardedLedger(shard_count=4, shard_dir=temp_shard_dir)

        stats = ledger.stats()

        assert stats["shard_count"] == 4
        assert stats["total_entries"] == 0
        assert len(stats["shards"]) == 4

    def test_stats_with_entries(self, temp_shard_dir):
        """Test stats with entries."""
        ledger = ShardedLedger(shard_count=4, shard_dir=temp_shard_dir)

        for i in range(100):
            entry = {"ts": "2025-01-15T00:00:00Z", "hash": f"h{i}:s"}
            ledger.append(entry)

        stats = ledger.stats()

        assert stats["total_entries"] == 100
        assert sum(s["count"] for s in stats["shards"]) == 100
        assert all(s["size_bytes"] > 0 for s in stats["shards"] if s["count"] > 0)
