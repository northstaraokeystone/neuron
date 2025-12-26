#!/usr/bin/env python3
"""
NEURON v5.0 CLI - Command Line Interface
Chain Rhythm Conductor + Bio-Digital Resonance Bridge.
CLAUDEME T+2h gate requirement.

Usage:
    python cli.py --test              # Run receipt emission test
    python cli.py --status            # Show ledger status
    python cli.py --replay N          # Replay last N entries
    python cli.py --append            # Append test entry
    python cli.py --benchmark         # Run quick benchmark
    python cli.py --self_conduct_mode # Enable chain conductor mode
    python cli.py --human_meta_test   # Simulate human meta-append
    python cli.py --simulate          # Run one conductor cycle

v5.0 Resonance Mode:
    python cli.py --resonance_mode    # Enable bio-digital SDM bridge
    python cli.py --resonance_mode --simulate_n1  # Use simulated N1 data
    python cli.py --resonance_mode --test_swr     # Test SWR detection
    python cli.py --resonance_mode --test_haptic  # Test haptic feedback
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
    # v4.6 chain conductor constants
    RHYTHM_SOURCE,
    ALPHA_MODE,
    HUMAN_ROLE,
    SELF_CONDUCT_ENABLED,
    PERSISTENCE_WINDOW_ENTRIES,
    MIN_GAP_MS_FOR_RHYTHM,
)

# v4.6 chain conductor imports
from chain_conductor import (
    load_self_conduct_spec,
    human_meta_append,
    detect_induced_oscillation,
    conductor_cycle,
    ConductorStopRule,
)

# v5.0 resonance bridge imports
from resonance_bridge import (
    load_resonance_spec,
    biological_to_digital_sdm,
    generate_simulated_spikes,
    ResonanceStopRule,
)
from swr_detector import (
    detect_biological_swr,
    generate_simulated_lfp,
    SWRStopRule,
)
from haptic_feedback import (
    haptic_feedback_loop,
    simulate_retrieval_for_test,
    HapticStopRule,
)
from neuron import consolidate_swr_sync


def cmd_test() -> dict:
    """Run receipt emission test per CLAUDEME T+2h gate."""
    receipt = emit_receipt(
        "cli_test_receipt",
        {
            "test": True,
            "cli_version": "4.6",
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
        from stress import stress_test

        result = stress_test(n_entries=100, concurrent=2)
        print(json.dumps(result, indent=2))
        return emit_receipt("benchmark_receipt", result)
    except ImportError:
        print("Error: stress module not available")
        return emit_receipt(
            "benchmark_receipt", {"error": "stress module not available"}
        )


# ============================================
# v4.6 CHAIN CONDUCTOR COMMANDS
# ============================================


def cmd_self_conduct_mode() -> dict:
    """Enable chain conductor mode (load spec, show status)."""
    try:
        spec = load_self_conduct_spec()

        status = {
            "receipt_type": "self_conduct_mode",
            "rhythm_source": RHYTHM_SOURCE,
            "alpha_mode": ALPHA_MODE,
            "human_role": HUMAN_ROLE,
            "self_conduct_enabled": SELF_CONDUCT_ENABLED,
            "persistence_window": PERSISTENCE_WINDOW_ENTRIES,
            "min_gap_ms": MIN_GAP_MS_FOR_RHYTHM,
            "spec_hash": spec.get("_spec_hash", "")[:32] + "...",
            "ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }

        print(json.dumps(status, indent=2))
        return emit_receipt("self_conduct_mode_receipt", status)
    except ConductorStopRule as e:
        print(f"Error: {e}", file=sys.stderr)
        return emit_receipt("self_conduct_error", {"error": str(e)})


def cmd_human_meta_test() -> dict:
    """Simulate human meta-append for validation."""
    meta_entry = {
        "steering_type": "observation",
        "note": "Human adds harmony, not direction",
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }

    result = human_meta_append(meta_entry)

    if result:
        print(json.dumps(result, indent=2))
        return emit_receipt(
            "human_meta_test_receipt", {"meta_hash": result.get("_meta_hash", "")[:32]}
        )
    else:
        print("No meta entry appended (None input)")
        return emit_receipt("human_meta_test_receipt", {"meta_hash": None})


def cmd_simulate() -> dict:
    """Run one conductor cycle and print receipts."""
    try:
        # Load ledger
        ledger = _read_ledger()

        # Run conductor cycle
        result = conductor_cycle(ledger=ledger)

        # Print key receipts
        print("=== NEURON v4.6 Conductor Cycle ===")
        print()
        print(f"Version: {result['version']}")
        print(f"Rhythm Source: {result['spec']['rhythm_source']}")
        print(f"Alpha Mode: {result['spec']['alpha_mode']}")
        print(f"Human Role: {result['spec']['human_role']}")
        print()
        print("--- Gap Rhythm ---")
        print(f"Pattern: {result['rhythm']['rhythm_pattern']}")
        print(f"Tempo: {result['rhythm']['tempo_ms']}ms")
        print(f"Gap Count: {result['rhythm']['gap_count']}")
        print()
        print("--- Persistence Alpha ---")
        print(f"Entry Count: {result['alpha']['entry_count']}")
        print(f"Max Gap Survived: {result['alpha']['max_gap_survived_ms']}ms")
        print()
        print("--- Self-Conduct ---")
        print(f"Self-Conducting: {result['self_conducting']}")
        print(f"Rhythm Metrics: {json.dumps(result['rhythm_metrics'], indent=2)}")

        # Emit receipts
        print()
        print("--- Receipts Emitted ---")
        print("gap_rhythm_receipt: OK")
        print("persistence_alpha_receipt: OK")
        print("self_conduct_receipt: OK")

        return emit_receipt(
            "conductor_cycle_receipt",
            {
                "version": result["version"],
                "self_conducting": result["self_conducting"],
                "rhythm_pattern": result["rhythm"]["rhythm_pattern"],
                "gap_count": result["rhythm"]["gap_count"],
            },
        )
    except ConductorStopRule as e:
        print(f"STOPRULE: {e}", file=sys.stderr)
        return emit_receipt("conductor_error", {"error": str(e)})


def cmd_conductor_status() -> dict:
    """Show current conductor state."""
    ledger = _read_ledger()

    # Check for induced oscillation
    has_induced = detect_induced_oscillation(ledger)

    status = {
        "receipt_type": "conductor_status",
        "rhythm_source": RHYTHM_SOURCE,
        "alpha_mode": ALPHA_MODE,
        "human_role": HUMAN_ROLE,
        "self_conduct_enabled": SELF_CONDUCT_ENABLED,
        "ledger_entries": len(ledger),
        "induced_oscillation_detected": has_induced,
        "ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }

    print(json.dumps(status, indent=2))
    return emit_receipt("conductor_status_receipt", status)


# ============================================
# v5.0 RESONANCE MODE COMMANDS
# ============================================


def cmd_resonance_mode(
    simulate_n1: bool = False, test_swr: bool = False, test_haptic: bool = False
) -> dict:
    """Enable bio-digital SDM bridge mode."""
    try:
        spec = load_resonance_spec()

        print("=== NEURON v5.0 Bio-Digital Resonance Bridge ===")
        print()
        print(f"SDM Dimension: {spec['sdm_dim']}")
        print(f"Sparsity Target: {spec['sparsity_target']}")
        print(f"SWR Frequency: {spec['swr_frequency_hz']} Hz")
        print(f"Urgency Threshold: {spec['urgency_threshold']}")
        print(f"Simulation Mode: {spec['simulation_mode']}")
        print()

        results = {
            "spec_loaded": True,
            "spec_hash": spec["_spec_hash"][:32] + "...",
        }

        # Generate simulated spikes if requested
        if simulate_n1:
            print("--- Simulated N1 Spikes ---")
            spikes = generate_simulated_spikes(1024, 100, 0.01, seed=42)
            bio_result = biological_to_digital_sdm(spikes, spec)

            print("Channels: 1024")
            print("Samples: 100")
            print(f"Sparsity: {bio_result['sparsity']:.4f}")
            print(f"Vector Hash: {bio_result['vector_hash'][:32]}...")
            print()

            # Emit bio_ingest_receipt
            print("bio_ingest_receipt: EMITTED")
            results["bio_ingest"] = True
            results["sparsity"] = bio_result["sparsity"]

            # Test SWR detection if requested
            if test_swr:
                print()
                print("--- SWR Detection Test ---")
                lfp = generate_simulated_lfp(500, 1000, swr_present=True)
                swr_result = detect_biological_swr(lfp, spec)

                if swr_result:
                    print(f"SWR Detected: {swr_result['detected']}")
                    print(f"Frequency: {swr_result['frequency_hz']} Hz")
                    print(f"Burst Count: {swr_result['burst_count']}")
                    print(f"Confidence: {swr_result['confidence']:.4f}")
                    print()
                    print("swr_detect_receipt: EMITTED")
                    results["swr_detected"] = swr_result["detected"]

                    # Test consolidation sync
                    print()
                    print("--- SWR-Synchronized Consolidation ---")
                    sync_result = consolidate_swr_sync(
                        user_id="test_user",
                        bio_vector=bio_result["vector"],
                        config=spec,
                        swr_detected=True,
                        max_cycles=1,
                    )
                    print(f"Cycles: {sync_result['cycles']}")
                    print(f"Entries Added: {sync_result['entries_added']}")
                    print(f"Entries Pruned: {sync_result['entries_pruned']}")
                    print()
                    print("consolidate_sync_receipt: EMITTED")
                    results["consolidate_sync"] = True
                else:
                    print("SWR not detected (confidence below threshold)")
                    results["swr_detected"] = False

            # Test haptic feedback if requested
            if test_haptic:
                print()
                print("--- Haptic Feedback Test ---")
                vector, salience = simulate_retrieval_for_test(
                    dim=1000, high_salience=True
                )
                ledger = _read_ledger()

                haptic_result = haptic_feedback_loop(vector, salience, spec, ledger)

                print(f"Salience: {haptic_result['salience']:.4f}")
                print(f"Urgency Threshold: {haptic_result['urgency_threshold']}")
                print(f"Haptic Delivered: {haptic_result['haptic_delivered']}")
                print(f"Text Delivered: {haptic_result['text_delivered']}")
                print(f"Haptic Before Text: {haptic_result['haptic_before_text']}")
                print()
                print("haptic_feedback_receipt: EMITTED")
                results["haptic_delivered"] = haptic_result["haptic_delivered"]

        print()
        print("This is not translation. This is resonance.")

        return emit_receipt("resonance_mode", results)

    except ResonanceStopRule as e:
        print(f"RESONANCE STOPRULE: {e}", file=sys.stderr)
        return emit_receipt("resonance_error", {"error": str(e)})
    except SWRStopRule as e:
        print(f"SWR STOPRULE: {e}", file=sys.stderr)
        return emit_receipt("swr_error", {"error": str(e)})
    except HapticStopRule as e:
        print(f"HAPTIC STOPRULE: {e}", file=sys.stderr)
        return emit_receipt("haptic_error", {"error": str(e)})


def main():
    parser = argparse.ArgumentParser(
        description="NEURON v5.0 CLI - Chain Rhythm Conductor + Bio-Digital Resonance",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python cli.py --test              # T+2h gate test
    python cli.py --status            # Show ledger status
    python cli.py --replay 5          # Replay last 5 entries
    python cli.py --append            # Append test entry
    python cli.py --alpha             # Show alpha statistics
    python cli.py --benchmark         # Run quick benchmark

v4.6 Chain Conductor Commands:
    python cli.py --self_conduct_mode # Enable chain conductor
    python cli.py --human_meta_test   # Test human meta-append
    python cli.py --simulate          # Run conductor cycle
    python cli.py --conductor_status  # Show conductor state

v5.0 Bio-Digital Resonance Commands:
    python cli.py --resonance_mode              # Enable bio-digital SDM bridge
    python cli.py --resonance_mode --simulate_n1  # Use simulated N1 data
    python cli.py --resonance_mode --simulate_n1 --test_swr  # Test SWR detection
    python cli.py --resonance_mode --simulate_n1 --test_haptic  # Test haptic feedback

KILLED (v4.5 induced oscillation):
    --frequency         # KILLED: rhythm from gaps only
    --inject            # KILLED: no injection
    --surge             # KILLED: no surge
    --oscillation_status # KILLED: no oscillation state
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

    # v4.6 Chain Conductor Commands
    parser.add_argument(
        "--self_conduct_mode", action="store_true", help="Enable chain conductor mode"
    )
    parser.add_argument(
        "--human_meta_test", action="store_true", help="Test human meta-append"
    )
    parser.add_argument(
        "--simulate", action="store_true", help="Run one conductor cycle"
    )
    parser.add_argument(
        "--conductor_status", action="store_true", help="Show conductor state"
    )

    # v5.0 Bio-Digital Resonance Commands
    parser.add_argument(
        "--resonance_mode", action="store_true", help="Enable bio-digital SDM bridge"
    )
    parser.add_argument(
        "--simulate_n1", action="store_true", help="Use simulated N1 data (no hardware)"
    )
    parser.add_argument(
        "--test_swr", action="store_true", help="Inject synthetic SWR for testing"
    )
    parser.add_argument(
        "--test_haptic", action="store_true", help="Simulate haptic feedback loop"
    )

    parser.add_argument("--version", action="version", version="NEURON CLI v5.0")

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
        # v4.6 Chain Conductor Commands
        elif args.self_conduct_mode:
            cmd_self_conduct_mode()
        elif args.human_meta_test:
            cmd_human_meta_test()
        elif args.simulate:
            cmd_simulate()
        elif args.conductor_status:
            cmd_conductor_status()
        # v5.0 Bio-Digital Resonance Commands
        elif args.resonance_mode:
            cmd_resonance_mode(
                simulate_n1=args.simulate_n1,
                test_swr=args.test_swr,
                test_haptic=args.test_haptic,
            )
        else:
            parser.print_help()
            sys.exit(1)
    except StopRule as e:
        print(f"STOPRULE VIOLATION: {e}", file=sys.stderr)
        sys.exit(2)
    except ConductorStopRule as e:
        print(f"CONDUCTOR STOPRULE: {e}", file=sys.stderr)
        sys.exit(2)
    except ResonanceStopRule as e:
        print(f"RESONANCE STOPRULE: {e}", file=sys.stderr)
        sys.exit(2)
    except SWRStopRule as e:
        print(f"SWR STOPRULE: {e}", file=sys.stderr)
        sys.exit(2)
    except HapticStopRule as e:
        print(f"HAPTIC STOPRULE: {e}", file=sys.stderr)
        sys.exit(2)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
