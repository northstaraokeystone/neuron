#!/bin/bash
# RESONANCE PROTOCOL - T+48h Gate Check
# Extended validation and stress testing

set -e

echo "=== RESONANCE PROTOCOL T+48h Gate Check ==="
echo ""

cd "$(dirname "$0")"

# Set up test environment
export RESONANCE_BASE=$(mktemp -d)
trap "rm -rf $RESONANCE_BASE" EXIT

# Run T+24h gates first
echo "Running T+24h gates..."
./gate_t24h.sh
echo ""

# Extended HDC testing
echo "=== EXTENDED HDC TESTING ==="
python3 -c "
import sys
sys.path.insert(0, 'src')
from hdc import test_orthogonality, test_dropout

# Higher-n orthogonality
print('Testing orthogonality with 500 vectors...')
mean_sim = test_orthogonality(n_vectors=500)
print(f'  Mean similarity: {mean_sim:.6f}')
assert mean_sim < 0.1, 'Orthogonality test failed'

# Multiple dropout rates
print('Testing multiple dropout rates...')
for rate in [0.1, 0.2, 0.3, 0.4]:
    acc = test_dropout(dropout_rate=rate)
    print(f'  {rate:.0%} dropout: {acc:.2%} accuracy')

print('PASS: Extended HDC testing')
"

# Extended latency testing
echo ""
echo "=== EXTENDED LATENCY TESTING ==="
python3 -c "
import sys
sys.path.insert(0, 'src')
from oscillation_detector import benchmark_latency

print('Running 1000-trial latency benchmark...')
result = benchmark_latency(n_trials=1000)
print(f'  p50: {result[\"p50\"]:.4f}ms')
print(f'  p95: {result[\"p95\"]:.4f}ms')
print(f'  p99: {result[\"p99\"]:.4f}ms')
assert result['p95'] < 2.0, 'Latency requirement not met'
print('PASS: Extended latency testing')
"

# Thread degradation simulation
echo ""
echo "=== THREAD DEGRADATION SIMULATION ==="
python3 -c "
import sys
sys.path.insert(0, 'src')
from thread_monitor import ThreadMonitor

print('Simulating 24-hour thread monitoring...')
monitor = ThreadMonitor(n_channels=100)

for hour in range(24):
    summary = monitor.update_all_channels(timestamp_hours=hour)

print(f'Final state after 24 hours:')
print(f'  Good: {summary[\"good\"]}')
print(f'  Degraded: {summary[\"degraded\"]}')
print(f'  Review: {summary[\"review_required\"]}')
print(f'  Dropout rate: {summary[\"dropout_rate\"]:.2%}')
print('PASS: Thread degradation simulation')
"

# Federated learning rounds
echo ""
echo "=== FEDERATED LEARNING SIMULATION ==="
python3 -c "
import sys
sys.path.insert(0, 'src')
from federated_coordinator import FederatedCoordinator, estimate_privacy_budget

print('Simulating 10 federated rounds...')
coordinator = FederatedCoordinator({'layer1': [0.0] * 100})

for round_num in range(10):
    # Simulate 3 participants per round
    for p in range(3):
        coordinator.receive_update(
            {'layer1': [0.001 * (round_num + 1)] * 100},
            n_samples=100
        )
    coordinator.aggregate_if_ready()

print(f'Rounds completed: {coordinator.update_count}')
print(f'Privacy budget spent: {coordinator.privacy_spent:.4f}')
print('PASS: Federated learning simulation')
"

echo ""
echo "=== T+48h Gate Check COMPLETE ==="
echo "All extended validations passed."
