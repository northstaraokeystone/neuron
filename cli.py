#!/usr/bin/env python3
"""
NEURON v4.1 CLI - Command Line Interface
CLAUDEME T+2h gate requirement.

Usage:
    python cli.py --test          # Run receipt emission test
    python cli.py --status        # Show ledger status
    python cli.py --replay N      # Replay last N entries
    python cli.py --append        # Append test entry
    python cli.py --benchmark     # Run quick benchmark
"""

import argparse
import json
import sys
from datetime import datetime, timezone

from neuron import (
    LEDGER_PATH,
    RECEIPTS_PATH,
    SUPPORTED_MODELS,
    StopRule,
    append,
    dual_hash,
    emit_receipt,
    merkle,
    replay,
    replay_to_context,
    alpha,
    consolidate,
    prune,
    _read_ledger,
)


def cmd_test() -> dict:
    """Run receipt emission test per CLAUDEME T+2h gate."""
    receipt = emit_receipt("cli_test_receipt", {
        "test": True,
        "cli_version": "4.1",
        "ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    })
    print(json.dumps(receipt, indent=2))
    return receipt


def cmd_status() -> dict:
    """Show ledger status."""
    entries = _read_ledger()
    status = {
        "receipt_type": "ledger_status_receipt",
        "ledger_path": str(LEDGER_PATH),
        "receipts_path": str(RECEIPTS_PATH),
        "total_entries": len(entries),
        "supported_models": SUPPORTED_MODELS,
        "ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    }

    if entries:
        status["oldest_entry"] = entries[0].get("ts", "unknown")
        status["newest_entry"] = entries[-1].get("ts", "unknown")

        # Calculate merkle root of last 10 entries
        recent_hashes = [e.get("hash", "") for e in entries[-10:]]
        status["recent_merkle"] = merkle(recent_hashes)[:32] + "..."

    receipt = emit_receipt("ledger_status_receipt", status)
    print(json.dumps(status, indent=2))
    return receipt


def cmd_replay(n: int) -> dict:
    """Replay last N entries."""
    entries = replay(n=n, increment_replay=False)
    context = replay_to_context(n=n)

    result = {
        "receipt_type": "replay_receipt",
        "entries_count": len(entries),
        "format": "context",
        "ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    }

    print(context)
    receipt = emit_receipt("replay_receipt", result)
    return receipt


def cmd_append() -> dict:
    """Append a test entry to the ledger."""
    entry = append(
        project="neuron",
        task="cli_test_append",
        next_action="verify_append",
        commit=None,
        model="neuron"
    )

    print(json.dumps(entry, indent=2))
    return emit_receipt("append_receipt", {"entry_hash": entry.get("hash", "")[:32]})


def cmd_alpha() -> dict:
    """Calculate alpha statistics."""
    stats = alpha()
    print(json.dumps(stats, indent=2))
    return emit_receipt("alpha_receipt", stats)


def cmd_benchmark() -> dict:
    """Run quick benchmark."""
    try:
        from stress import stress_test, benchmark_report
        result = stress_test(n_entries=100, concurrent=2)
        print(json.dumps(result, indent=2))
        return emit_receipt("benchmark_receipt", result)
    except ImportError:
        print("Error: stress module not available")
        return emit_receipt("benchmark_receipt", {"error": "stress module not available"})


def main():
    parser = argparse.ArgumentParser(
        description="NEURON v4.1 CLI - CLAUDEME Compliant Ledger Interface",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python cli.py --test           # T+2h gate test
    python cli.py --status         # Show ledger status
    python cli.py --replay 5       # Replay last 5 entries
    python cli.py --append         # Append test entry
    python cli.py --alpha          # Show alpha statistics
    python cli.py --benchmark      # Run quick benchmark
        """
    )

    parser.add_argument("--test", action="store_true", help="Run receipt emission test (T+2h gate)")
    parser.add_argument("--status", action="store_true", help="Show ledger status")
    parser.add_argument("--replay", type=int, metavar="N", help="Replay last N entries")
    parser.add_argument("--append", action="store_true", help="Append test entry")
    parser.add_argument("--alpha", action="store_true", help="Show alpha statistics")
    parser.add_argument("--benchmark", action="store_true", help="Run quick benchmark")
    parser.add_argument("--version", action="version", version="NEURON CLI v4.1")

    args = parser.parse_args()

    try:
        if args.test:
            cmd_test()
        elif args.status:
            cmd_status()
        elif args.replay:
            cmd_replay(args.replay)
        elif args.append:
            cmd_append()
        elif args.alpha:
            cmd_alpha()
        elif args.benchmark:
            cmd_benchmark()
        else:
            parser.print_help()
            sys.exit(1)
    except StopRule as e:
        print(f"STOPRULE VIOLATION: {e}", file=sys.stderr)
        sys.exit(2)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
