#!/usr/bin/env python3
"""
RESONANCE PROTOCOL v2.0 CLI - Command Line Interface

Neurophysiology research infrastructure with HDC/FL.

Usage:
    python cli.py test-hdc --orthogonality --dropout
    python cli.py test-oscillations --latency-benchmark
    python cli.py test-federated --privacy
    python cli.py test-phase-lock
    python cli.py test-safety --shannon-limit --thermal-check
    python cli.py run-pipeline --synthetic-data
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from core import emit_receipt, dual_hash


def cmd_test_hdc(args):
    """Test HDC encoder module."""
    from hdc import test_orthogonality, test_dropout

    results = {"module": "hdc", "tests": []}

    if args.orthogonality:
        print("Testing HDC orthogonality...")
        mean_sim = test_orthogonality(n_vectors=args.n_vectors)
        passed = mean_sim < 0.1
        results["tests"].append({
            "name": "orthogonality",
            "mean_similarity": round(mean_sim, 6),
            "threshold": 0.1,
            "passed": passed,
        })
        print(f"  Mean similarity: {mean_sim:.6f} (target < 0.1)")
        print(f"  PASSED: {passed}")

    if args.dropout:
        print("\nTesting HDC dropout robustness...")
        accuracy = test_dropout(dropout_rate=args.dropout_rate)
        passed = accuracy > 0.85
        results["tests"].append({
            "name": "dropout_robustness",
            "accuracy": round(accuracy, 4),
            "dropout_rate": args.dropout_rate,
            "threshold": 0.85,
            "passed": passed,
        })
        print(f"  Accuracy: {accuracy:.2%} (target > 85%)")
        print(f"  PASSED: {passed}")

    return results


def cmd_test_oscillations(args):
    """Test oscillation detection module."""
    from oscillation_detector import benchmark_latency

    results = {"module": "oscillation_detector", "tests": []}

    if args.latency_benchmark:
        print("Running oscillation detection latency benchmark...")
        stats = benchmark_latency(n_trials=args.n_trials)
        results["tests"].append({
            "name": "latency_benchmark",
            "p50_ms": stats["p50"],
            "p95_ms": stats["p95"],
            "p99_ms": stats["p99"],
            "target_ms": 2.0,
            "passed": stats["passed"],
        })
        print(f"  p50: {stats['p50']:.3f}ms")
        print(f"  p95: {stats['p95']:.3f}ms (target < 2.0ms)")
        print(f"  p99: {stats['p99']:.3f}ms")
        print(f"  PASSED: {stats['passed']}")

    return results


def cmd_test_federated(args):
    """Test federated learning module."""
    from federated_coordinator import verify_no_raw_data

    results = {"module": "federated_coordinator", "tests": []}

    if args.privacy:
        print("Testing federated learning privacy guarantee...")
        try:
            passed = verify_no_raw_data()
            results["tests"].append({
                "name": "privacy_guarantee",
                "raw_data_detected": False,
                "passed": passed,
            })
            print(f"  Self-test PASSED: {passed}")
        except Exception as e:
            results["tests"].append({
                "name": "privacy_guarantee",
                "error": str(e),
                "passed": False,
            })
            print(f"  FAILED: {e}")

    if args.convergence:
        print("\nTesting federated learning convergence...")
        from federated_coordinator import FederatedCoordinator, FL_MIN_PARTICIPANTS

        coordinator = FederatedCoordinator({
            "layer1": [0.0] * 10,
        })

        # Add minimum participants
        for i in range(FL_MIN_PARTICIPANTS):
            coordinator.receive_update(
                {"layer1": [0.01 * (i + 1)] * 10},
                n_samples=100
            )

        aggregated = coordinator.aggregate_if_ready()
        results["tests"].append({
            "name": "convergence",
            "participants": FL_MIN_PARTICIPANTS,
            "aggregated": aggregated,
            "passed": aggregated,
        })
        print(f"  Participants: {FL_MIN_PARTICIPANTS}")
        print(f"  Aggregation successful: {aggregated}")
        print(f"  PASSED: {aggregated}")

    return results


def cmd_test_phase_lock(args):
    """Test phase predictor module."""
    import math
    from phase_predictor import test_phase_lock

    print("Testing phase lock prediction...")
    result = test_phase_lock(n_trials=args.n_trials)

    threshold = math.pi / 4
    results = {
        "module": "phase_predictor",
        "tests": [{
            "name": "phase_lock",
            "mean_error_rad": result["mean_error_rad"],
            "p95_error_rad": result["p95_error_rad"],
            "threshold_rad": round(threshold, 4),
            "passed": result["passed"],
        }],
    }

    print(f"  Mean error: {result['mean_error_rad']:.4f} rad")
    print(f"  p95 error: {result['p95_error_rad']:.4f} rad (target < {threshold:.4f})")
    print(f"  PASSED: {result['passed']}")

    return results


def cmd_test_safety(args):
    """Test stimulation safety module."""
    from stim_controller import test_shannon_limit, monitor_thermal

    results = {"module": "stim_controller", "tests": []}

    if args.shannon_limit:
        print("Testing Shannon limit safety...")
        passed = test_shannon_limit(n_tests=args.n_tests)
        results["tests"].append({
            "name": "shannon_limit",
            "n_tests": args.n_tests,
            "passed": passed,
        })
        print(f"  Tests run: {args.n_tests}")
        print(f"  PASSED: {passed}")

    if args.thermal_check:
        print("\nTesting thermal monitoring...")
        # Safe temperatures
        safe_temps = [37.0, 37.2, 37.5, 37.3]
        safe_result = monitor_thermal(safe_temps, baseline_temp=37.0)

        # Unsafe temperatures
        unsafe_temps = [37.0, 37.5, 38.5]
        unsafe_result = monitor_thermal(unsafe_temps, baseline_temp=37.0)

        passed = safe_result and not unsafe_result
        results["tests"].append({
            "name": "thermal_monitoring",
            "safe_temps_passed": safe_result,
            "unsafe_temps_rejected": not unsafe_result,
            "passed": passed,
        })
        print(f"  Safe temps accepted: {safe_result}")
        print(f"  Unsafe temps rejected: {not unsafe_result}")
        print(f"  PASSED: {passed}")

    return results


def cmd_run_pipeline(args):
    """Run end-to-end pipeline."""
    import math

    print("=== RESONANCE PROTOCOL v2.0 Pipeline ===\n")

    results = {"pipeline": "resonance_protocol", "stages": []}

    # Stage 1: Generate synthetic data
    print("Stage 1: Generate synthetic neural data")
    from hdc import create_random_projection, encode_spike_window
    import random
    random.seed(42)

    n_channels = 100
    projection = create_random_projection(n_channels, 1000, seed=42)
    spike_counts = [random.random() * 10 for _ in range(n_channels)]
    hypervector = encode_spike_window(spike_counts, projection)

    results["stages"].append({
        "name": "synthetic_data",
        "n_channels": n_channels,
        "hypervector_dim": len(hypervector),
    })
    print(f"  Channels: {n_channels}")
    print(f"  Hypervector dimension: {len(hypervector)}")

    # Stage 2: Simulate oscillation detection
    print("\nStage 2: Oscillation detection")
    from oscillation_detector import detect_swr

    lfp = [math.sin(2 * math.pi * 200 * i / 20000) + random.gauss(0, 0.1)
           for i in range(1000)]
    events = detect_swr(lfp, fs=20000, threshold_sd=1.0)

    results["stages"].append({
        "name": "oscillation_detection",
        "n_events": len(events),
    })
    print(f"  Events detected: {len(events)}")

    # Stage 3: Phase prediction
    print("\nStage 3: Phase prediction")
    from phase_predictor import PhaseLockController

    controller = PhaseLockController(target_oscillation="swr")
    for i, event in enumerate(events[:10]):
        event["timestamp_ns"] = i * 5_000_000
        controller.add_event(event)

    window = controller.predict_next_window(50_000_000)
    results["stages"].append({
        "name": "phase_prediction",
        "confidence": window["confidence"],
    })
    print(f"  Prediction confidence: {window['confidence']:.2f}")

    # Stage 4: Safety check
    print("\nStage 4: Stimulation safety")
    from stim_controller import StimulationController

    stim = StimulationController()
    pulse = stim.request_pulse("hippocampus", 50, 100)

    results["stages"].append({
        "name": "stimulation_safety",
        "pulse_generated": pulse is not None,
        "safety_passed": pulse["safety_check_passed"] if pulse else False,
    })
    print(f"  Pulse generated: {pulse is not None}")
    if pulse:
        print(f"  Safety check: {pulse['safety_check_passed']}")

    # Stage 5: Thread monitoring
    print("\nStage 5: Thread monitoring")
    from thread_monitor import ThreadMonitor

    monitor = ThreadMonitor(n_channels=50)
    summary = monitor.update_all_channels(timestamp_hours=0)

    results["stages"].append({
        "name": "thread_monitoring",
        "good_channels": summary["good"],
        "degraded_channels": summary["degraded"],
        "dropout_rate": round(summary["dropout_rate"], 4),
    })
    print(f"  Good channels: {summary['good']}")
    print(f"  Degraded: {summary['degraded']}")
    print(f"  Dropout rate: {summary['dropout_rate']:.2%}")

    # Emit pipeline receipt
    emit_receipt("pipeline_run", results)

    print("\n=== Pipeline Complete ===")
    return results


def main():
    parser = argparse.ArgumentParser(
        description="RESONANCE PROTOCOL v2.0 CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # test-hdc
    hdc_parser = subparsers.add_parser("test-hdc", help="Test HDC encoder")
    hdc_parser.add_argument("--orthogonality", action="store_true",
                           help="Run orthogonality test")
    hdc_parser.add_argument("--dropout", action="store_true",
                           help="Run dropout robustness test")
    hdc_parser.add_argument("--n-vectors", type=int, default=100,
                           help="Number of vectors for orthogonality test")
    hdc_parser.add_argument("--dropout-rate", type=float, default=0.3,
                           help="Dropout rate for robustness test")

    # test-oscillations
    osc_parser = subparsers.add_parser("test-oscillations",
                                       help="Test oscillation detection")
    osc_parser.add_argument("--latency-benchmark", action="store_true",
                           help="Run latency benchmark")
    osc_parser.add_argument("--accuracy-check", action="store_true",
                           help="Run accuracy check")
    osc_parser.add_argument("--n-trials", type=int, default=100,
                           help="Number of trials")

    # test-federated
    fed_parser = subparsers.add_parser("test-federated",
                                       help="Test federated learning")
    fed_parser.add_argument("--convergence", action="store_true",
                           help="Test convergence")
    fed_parser.add_argument("--privacy", action="store_true",
                           help="Test privacy guarantee")

    # test-phase-lock
    phase_parser = subparsers.add_parser("test-phase-lock",
                                         help="Test phase prediction")
    phase_parser.add_argument("--n-trials", type=int, default=100,
                             help="Number of trials")

    # test-safety
    safety_parser = subparsers.add_parser("test-safety",
                                          help="Test stimulation safety")
    safety_parser.add_argument("--shannon-limit", action="store_true",
                              help="Test Shannon limit")
    safety_parser.add_argument("--thermal-check", action="store_true",
                              help="Test thermal monitoring")
    safety_parser.add_argument("--n-tests", type=int, default=100,
                              help="Number of test pulses")

    # run-pipeline
    pipe_parser = subparsers.add_parser("run-pipeline",
                                        help="Run end-to-end pipeline")
    pipe_parser.add_argument("--synthetic-data", action="store_true",
                            help="Use synthetic neural data")

    parser.add_argument("--version", action="version",
                       version="RESONANCE PROTOCOL CLI v2.0")

    args = parser.parse_args()

    if args.command == "test-hdc":
        result = cmd_test_hdc(args)
    elif args.command == "test-oscillations":
        result = cmd_test_oscillations(args)
    elif args.command == "test-federated":
        result = cmd_test_federated(args)
    elif args.command == "test-phase-lock":
        result = cmd_test_phase_lock(args)
    elif args.command == "test-safety":
        result = cmd_test_safety(args)
    elif args.command == "run-pipeline":
        result = cmd_run_pipeline(args)
    else:
        parser.print_help()
        sys.exit(1)

    # Print JSON result
    print("\n--- Result ---")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
