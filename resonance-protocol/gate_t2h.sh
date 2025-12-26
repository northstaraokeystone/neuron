#!/bin/bash
# RESONANCE PROTOCOL - T+2h Gate Check
# Quick validation of core primitives

set -e

echo "=== RESONANCE PROTOCOL T+2h Gate Check ==="
echo ""

cd "$(dirname "$0")"

# Check Python availability
if ! command -v python3 &> /dev/null; then
    echo "ERROR: python3 not found"
    exit 1
fi

# Core neural primitives
echo "Test 1: Core neural primitives"
python3 -c "
import sys
sys.path.insert(0, 'src')
from core import neural_hash, validate_channel_quality

# Test neural_hash
spikes = [0.001, 0.005, 0.012]
h = neural_hash(spikes, window_ms=50)
assert ':' in h, 'Dual-hash format required'
print('  neural_hash: OK')

# Test validate_channel_quality
assert validate_channel_quality(impedance_kohm=50, snr_db=10), 'Good channel should validate'
assert not validate_channel_quality(impedance_kohm=150, snr_db=4), 'Bad channel should fail'
print('  validate_channel_quality: OK')

print('PASS: core neural functions')
"

echo ""
echo "Test 2: HDC orthogonality (quick)"
python3 -c "
import sys
sys.path.insert(0, 'src')
from hdc import test_orthogonality

mean_sim = test_orthogonality(n_vectors=50)
assert mean_sim < 0.15, f'Orthogonality violated: {mean_sim}'
print(f'PASS: HDC orthogonality {mean_sim:.4f}')
"

echo ""
echo "Test 3: Receipt emission"
python3 -c "
import sys
import os
import tempfile
sys.path.insert(0, 'src')

# Use temp directory
os.environ['RESONANCE_BASE'] = tempfile.mkdtemp()

from core import emit_receipt

receipt = emit_receipt('test_receipt', {'test': True})
assert 'hash' in receipt
assert ':' in receipt['hash']
print('PASS: Receipt emission with dual-hash')
"

echo ""
echo "=== T+2h Gate Check PASSED ==="
