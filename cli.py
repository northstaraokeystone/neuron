#!/usr/bin/env python3
"""
NEURON v4.5 CLI - Command Line Interface
CLAUDEME T+2h gate requirement.

Usage:
    python cli.py --test              # Run receipt emission test
    python cli.py --status            # Show ledger status
    python cli.py --replay N          # Replay last N entries
    python cli.py --append            # Append test entry
    python cli.py --benchmark         # Run quick benchmark
    python cli.py --resonance_mode    # Enable oscillation mode
    python cli.py --frequency SOURCE  # Set frequency source
    python cli.py --inject            # Manual injection trigger
    python cli.py --surge             # Manual surge trigger
    python cli.py --oscillation_status # Show current oscillation state
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
    emit_receipt,
    merkle,
    replay,
    replay_to_context,
    alpha,
    _read_ledger,
    # v4.5 resonance imports
    RESONANCE_MODE,
    OSCILLATION_AMPLITUDE_DEFAULT,
    FREQUENCY_SOURCES,
    DEFAULT_FREQUENCY,
    GAP_AMPLITUDE_BOOST,
    get_current_phase,
    human_direct_phase,
    detect_phase_transition,
)


def cmd_test() -> dict:
    """Run receipt emission test per CLAUDEME T+2h gate."""
    receipt = emit_receipt(
        "cli_test_receipt",
        {
            "test": True,
            "cli_version": "4.1",
            "ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        },
    )
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
        "ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
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
        "ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
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
        model="neuron",
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
        return emit_receipt(
            "benchmark_receipt", {"error": "stress module not available"}
        )


# ============================================
# v4.5 RESONANCE COMMANDS
# ============================================


def cmd_resonance_mode() -> dict:
    """Enable resonance oscillation mode."""
    from frequency import tune_frequency

    # Get current frequency
    freq_receipt = tune_frequency(DEFAULT_FREQUENCY)

    status = {
        "resonance_mode": RESONANCE_MODE,
        "current_phase": get_current_phase(),
        "amplitude": OSCILLATION_AMPLITUDE_DEFAULT,
        "frequency_source": DEFAULT_FREQUENCY,
        "frequency_hz": freq_receipt["frequency_hz"],
        "gap_boost": GAP_AMPLITUDE_BOOST,
        "ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }

    print(json.dumps(status, indent=2))
    return emit_receipt("resonance_mode_receipt", status)


def cmd_frequency(source: str) -> dict:
    """Set oscillation frequency source."""
    from frequency import tune_frequency

    if source not in FREQUENCY_SOURCES:
        print(f"Error: Invalid frequency source. Valid: {FREQUENCY_SOURCES}")
        return emit_receipt("frequency_error", {"error": f"Invalid source: {source}"})

    freq_receipt = tune_frequency(source)

    print(json.dumps(freq_receipt, indent=2))
    return emit_receipt("frequency_receipt", freq_receipt)


def cmd_inject() -> dict:
    """Manual injection trigger (human override)."""
    result = human_direct_phase(
        "inject", human_id="cli_user", override_reason="manual_inject"
    )

    print(json.dumps(result, indent=2))
    return emit_receipt("inject_receipt", result)


def cmd_surge() -> dict:
    """Manual surge trigger (human override)."""
    result = human_direct_phase(
        "surge", human_id="cli_user", override_reason="manual_surge"
    )

    print(json.dumps(result, indent=2))
    return emit_receipt("surge_receipt", result)


def cmd_oscillation_status() -> dict:
    """Show current oscillation state."""
    from neuron import (
        _oscillation_phase,
        _oscillation_amplitude,
        _oscillation_frequency,
    )

    status = {
        "receipt_type": "oscillation_status",
        "current_phase": _oscillation_phase,
        "amplitude": _oscillation_amplitude,
        "frequency_hz": _oscillation_frequency,
        "resonance_mode": RESONANCE_MODE,
        "available_sources": FREQUENCY_SOURCES,
        "default_source": DEFAULT_FREQUENCY,
        "gap_boost": GAP_AMPLITUDE_BOOST,
        "ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }

    print(json.dumps(status, indent=2))
    return emit_receipt("oscillation_status_receipt", status)


def cmd_simulate_transition() -> dict:
    """Test phase transition detection."""
    # Simulate triad state with transitions
    triad_state = {
        "axiom": {"laws_discovered": 5, "compression": 0.92},
        "agentproof": {"selection_threshold": 0.85},
    }

    result = detect_phase_transition(triad_state)

    print(json.dumps(result, indent=2))
    return emit_receipt("simulate_transition_receipt", result)


def main():
    parser = argparse.ArgumentParser(
        description="NEURON v4.5 CLI - CLAUDEME Compliant Ledger Interface",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python cli.py --test              # T+2h gate test
    python cli.py --status            # Show ledger status
    python cli.py --replay 5          # Replay last 5 entries
    python cli.py --append            # Append test entry
    python cli.py --alpha             # Show alpha statistics
    python cli.py --benchmark         # Run quick benchmark

v4.5 Resonance Commands:
    python cli.py --resonance_mode    # Enable oscillation mode
    python cli.py --frequency HUMAN_FOCUS  # Set frequency source
    python cli.py --inject            # Manual injection trigger
    python cli.py --surge             # Manual surge trigger
    python cli.py --oscillation_status # Show oscillation state
    python cli.py --simulate_transition # Test phase transition
        """,
    )

    parser.add_argument(
        "--test", action="store_true", help="Run receipt emission test (T+2h gate)"
    )
    parser.add_argument("--status", action="store_true", help="Show ledger status")
    parser.add_argument("--replay", type=int, metavar="N", help="Replay last N entries")
    parser.add_argument("--append", action="store_true", help="Append test entry")
    parser.add_argument("--alpha", action="store_true", help="Show alpha statistics")
    parser.add_argument("--benchmark", action="store_true", help="Run quick benchmark")

    # v4.5 Resonance Commands
    parser.add_argument(
        "--resonance_mode", action="store_true", help="Enable oscillation mode"
    )
    parser.add_argument(
        "--frequency", type=str, metavar="SOURCE", help="Set frequency source"
    )
    parser.add_argument(
        "--inject", action="store_true", help="Manual injection trigger"
    )
    parser.add_argument("--surge", action="store_true", help="Manual surge trigger")
    parser.add_argument(
        "--oscillation_status", action="store_true", help="Show oscillation state"
    )
    parser.add_argument(
        "--simulate_transition", action="store_true", help="Test phase transition"
    )

    parser.add_argument("--version", action="version", version="NEURON CLI v4.5")

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
        # v4.5 Resonance Commands
        elif args.resonance_mode:
            cmd_resonance_mode()
        elif args.frequency:
            cmd_frequency(args.frequency)
        elif args.inject:
            cmd_inject()
        elif args.surge:
            cmd_surge()
        elif args.oscillation_status:
            cmd_oscillation_status()
        elif args.simulate_transition:
            cmd_simulate_transition()
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
