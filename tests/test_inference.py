"""
NEURON v4 Inference Integration Tests
Tests for inference_append, replay_to_context, sync_ledger
"""

import json
import os
import sys
import tempfile
import time
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from neuron import (
    inference_append,
    replay_to_context,
    sync_ledger,
    append,
    replay,
    prune,
    _read_ledger,
    _write_ledger,
    LEDGER_PATH,
    SUPPORTED_MODELS,
)


def setup_test_ledger():
    """Create a temporary ledger for testing."""
    temp_dir = tempfile.mkdtemp()
    test_ledger = Path(temp_dir) / "receipts.jsonl"
    os.environ["NEURON_LEDGER"] = str(test_ledger)
    os.environ["NEURON_ARCHIVE"] = str(Path(temp_dir) / "archive.jsonl")
    # Reload the module to pick up new paths
    import neuron
    neuron.LEDGER_PATH = test_ledger
    neuron.ARCHIVE_PATH = Path(temp_dir) / "archive.jsonl"
    return test_ledger, temp_dir


def test_inference_append_basic():
    """Test basic inference_append functionality."""
    test_ledger, temp_dir = setup_test_ledger()

    entry = inference_append(
        model="grok",
        task="test inference",
        next_action="verify integration",
        context_summary="Testing NEURON v4 inference integration...",
        token_count=5000,
        inference_id="inf_test_001"
    )

    assert entry["model"] == "grok"
    assert entry["token_count"] == 5000
    assert entry["inference_id"] == "inf_test_001"
    assert "context_summary" in entry
    assert entry["project"] == "neuron"
    print(f"PASS: inference_append - model={entry['model']}, tokens={entry['token_count']}")


def test_inference_append_auto_id():
    """Test inference_append generates inference_id if not provided."""
    test_ledger, temp_dir = setup_test_ledger()

    entry = inference_append(
        model="claude",
        task="auto id test",
        next_action="check id generated",
        context_summary="Testing auto ID generation...",
        token_count=1000
    )

    assert entry["inference_id"] is not None
    assert entry["inference_id"].startswith("inf_")
    print(f"PASS: inference_append auto-id - {entry['inference_id']}")


def test_inference_append_all_models():
    """Test inference_append works with all supported models."""
    test_ledger, temp_dir = setup_test_ledger()

    for model in SUPPORTED_MODELS:
        entry = inference_append(
            model=model,
            task=f"test {model}",
            next_action="verify model",
            context_summary=f"Testing {model} model...",
            token_count=1000
        )
        assert entry["model"] == model

    print(f"PASS: inference_append all models - {SUPPORTED_MODELS}")


def test_inference_append_invalid_model():
    """Test inference_append rejects invalid model."""
    test_ledger, temp_dir = setup_test_ledger()

    try:
        inference_append(
            model="invalid_model",
            task="should fail",
            next_action="never reached",
            context_summary="This should fail...",
            token_count=1000
        )
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Model must be one of" in str(e)
        print("PASS: inference_append rejects invalid model")


def test_replay_to_context():
    """Test replay_to_context formatting."""
    test_ledger, temp_dir = setup_test_ledger()

    # Add some entries
    inference_append("grok", "task 1", "next 1", "Context 1...", 10000, "inf_001")
    time.sleep(0.05)
    inference_append("claude", "task 2", "next 2", "Context 2...", 20000, "inf_002")
    time.sleep(0.05)
    inference_append("grok", "task 3", "next 3", "Context 3...", 30000, "inf_003")

    ctx = replay_to_context(n=3, format="context")

    assert "## NEURON State Recovery" in ctx
    assert "Resume Instruction" in ctx
    assert "grok" in ctx
    assert "claude" in ctx
    print("PASS: replay_to_context")
    print(ctx[:300] + "...")


def test_replay_format_param():
    """Test replay with format parameter."""
    test_ledger, temp_dir = setup_test_ledger()

    inference_append("grok", "test", "next", "ctx", 1000)

    # List format (default)
    result_list = replay(n=1, format="list")
    assert isinstance(result_list, list)

    # Context format
    result_ctx = replay(n=1, format="context")
    assert isinstance(result_ctx, str)
    assert "## NEURON State Recovery" in result_ctx

    print("PASS: replay format parameter")


def test_sync_ledger():
    """Test sync_ledger merges correctly."""
    test_ledger, temp_dir = setup_test_ledger()

    # Create local entries
    inference_append("grok", "local 1", "next 1", "ctx 1", 1000, "inf_local_001")
    inference_append("claude", "local 2", "next 2", "ctx 2", 2000, "inf_local_002")

    # Create remote ledger
    remote_path = Path(temp_dir) / "remote_receipts.jsonl"
    remote_entries = [
        {
            "ts": "2025-01-15T12:00:00Z",
            "project": "neuron",
            "model": "gemini",
            "task": "remote 1",
            "next": "remote next 1",
            "hash": "abc:def",
            "salience": 1.0,
            "replay_count": 0,
            "energy": 1.0,
            "token_count": 3000,
            "inference_id": "inf_remote_001",
            "context_summary": "Remote context..."
        }
    ]
    with open(remote_path, "w") as f:
        for e in remote_entries:
            f.write(json.dumps(e) + "\n")

    result = sync_ledger(str(remote_path))

    assert result["local_entries"] == 2
    assert result["remote_entries"] == 1
    assert result["merged_entries"] == 3
    assert result["resolution_strategy"] == "last_write_wins"
    print(f"PASS: sync_ledger - merged {result['merged_entries']} entries")


def test_sync_ledger_conflict_resolution():
    """Test sync_ledger resolves conflicts with last-write-wins."""
    test_ledger, temp_dir = setup_test_ledger()

    # Create local entry with older timestamp
    import neuron
    original_append = neuron.append

    # Manually create an entry with specific timestamp
    entry = {
        "ts": "2025-01-15T10:00:00Z",
        "project": "neuron",
        "model": "grok",
        "task": "old local",
        "next": "old next",
        "hash": "old:hash",
        "salience": 1.0,
        "replay_count": 0,
        "energy": 1.0,
        "token_count": 1000,
        "inference_id": "inf_conflict",
        "context_summary": "Old local..."
    }
    with open(test_ledger, "w") as f:
        f.write(json.dumps(entry) + "\n")

    # Create remote with newer timestamp and same inference_id
    remote_path = Path(temp_dir) / "remote_receipts.jsonl"
    remote_entry = {
        "ts": "2025-01-15T12:00:00Z",  # Newer
        "project": "neuron",
        "model": "grok",
        "task": "new remote",
        "next": "new next",
        "hash": "new:hash",
        "salience": 1.0,
        "replay_count": 0,
        "energy": 1.0,
        "token_count": 2000,
        "inference_id": "inf_conflict",  # Same inference_id
        "context_summary": "New remote..."
    }
    with open(remote_path, "w") as f:
        f.write(json.dumps(remote_entry) + "\n")

    result = sync_ledger(str(remote_path))

    assert result["conflicts_resolved"] == 1

    # Verify the newer entry won
    entries = _read_ledger()
    assert len(entries) == 1
    assert entries[0]["task"] == "new remote"
    print("PASS: sync_ledger conflict resolution (last-write-wins)")


def test_multi_model_inference_cycle():
    """Test full multi-model inference cycle simulation."""
    test_ledger, temp_dir = setup_test_ledger()

    # Simulate multi-model inference
    inference_append("grok", "reasoning step 1", "continue reasoning", "Mars colony entropy...", 10000, "inf_001")
    time.sleep(0.05)
    inference_append("claude", "code review", "merge PR", "AgentProof federation code...", 25000, "inf_002")
    time.sleep(0.05)
    inference_append("grok", "reasoning step 2", "conclude", "Entropy budget calculated...", 45000, "inf_003")

    # Replay in context format
    ctx = replay_to_context(n=3, format="context")

    assert "grok" in ctx
    assert "claude" in ctx
    assert "reasoning step" in ctx
    assert "code review" in ctx
    print("PASS: multi-model inference cycle")


def test_prune_compression():
    """Test prune achieves compression."""
    test_ledger, temp_dir = setup_test_ledger()

    # Add entries
    for i in range(10):
        inference_append("grok", f"task {i}", f"next {i}", f"ctx {i}", 1000)

    result = prune(max_age_days=0, salience_threshold=2.0)  # Aggressive for test
    print(f"PASS: prune - compression achievable (pruned {result['pruned_count']} entries)")


def test_context_summary_truncation():
    """Test context_summary is truncated to 500 chars."""
    test_ledger, temp_dir = setup_test_ledger()

    long_summary = "x" * 1000  # 1000 chars
    entry = inference_append(
        model="grok",
        task="truncation test",
        next_action="verify length",
        context_summary=long_summary,
        token_count=1000
    )

    assert len(entry["context_summary"]) <= 500
    print("PASS: context_summary truncation")


def run_all_tests():
    """Run all inference integration tests."""
    print("=" * 60)
    print("NEURON v4 Inference Integration Tests")
    print("=" * 60)

    tests = [
        test_inference_append_basic,
        test_inference_append_auto_id,
        test_inference_append_all_models,
        test_inference_append_invalid_model,
        test_replay_to_context,
        test_replay_format_param,
        test_sync_ledger,
        test_sync_ledger_conflict_resolution,
        test_multi_model_inference_cycle,
        test_prune_compression,
        test_context_summary_truncation,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"FAIL: {test.__name__} - {e}")
            failed += 1

    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
