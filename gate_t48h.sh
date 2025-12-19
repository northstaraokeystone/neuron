#!/bin/bash
# NEURON T+48h Gate - CLAUDEME Compliance Check (Ship-Ready)
# Required: stoprule usage, dual_hash validation, full compliance
#
# Usage: ./gate_t48h.sh
# Exit 0 = PASS (SHIP IT), Exit 1 = FAIL

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== NEURON T+48h Gate ==="
echo "Checking CLAUDEME T+48h requirements (Ship-Ready)..."
echo ""

# Run T+24h gate first
if ./gate_t24h.sh > /dev/null 2>&1; then
    echo "PASS: T+24h gate passed"
else
    echo "FAIL: T+24h gate failed (prerequisite)"
    exit 1
fi

# Check stoprule/StopRule usage in Python files
if grep -riq "stoprule\|StopRule" *.py 2>/dev/null; then
    echo "PASS: stoprule found in code"
else
    echo "FAIL: no stoprule in code"
    exit 1
fi

# Validate dual_hash produces correct format
if python -c "from neuron import dual_hash; assert ':' in dual_hash('test')" 2>&1; then
    echo "PASS: dual_hash format validated"
else
    echo "FAIL: dual_hash format invalid"
    exit 1
fi

# Check MANIFEST.anchor exists
if [ -f MANIFEST.anchor ]; then
    echo "PASS: MANIFEST.anchor exists"
else
    echo "FAIL: no MANIFEST.anchor"
    exit 1
fi

# Validate core functions exist
if python -c "
from neuron import dual_hash, emit_receipt, merkle, StopRule
assert callable(dual_hash)
assert callable(emit_receipt)
assert callable(merkle)
assert issubclass(StopRule, Exception)
" 2>&1; then
    echo "PASS: all core functions present"
else
    echo "FAIL: missing core functions"
    exit 1
fi

# Check no bare 'except: pass' patterns (anti-pattern)
# Exclude comments and docstrings
if grep -rn "^[^#\"']*except:.*pass" *.py 2>/dev/null | grep -v '"""' | grep -v "'''"; then
    echo "WARN: bare 'except: pass' found (anti-pattern)"
    # Not a hard fail, just a warning
fi

# Final validation: run benchmark if available
echo ""
echo "Running quick validation..."
if python -c "
from neuron import dual_hash, emit_receipt, merkle
# Test dual_hash
h = dual_hash('test')
assert ':' in h
assert len(h.split(':')[0]) == 64
assert len(h.split(':')[1]) == 64

# Test merkle
m = merkle(['a', 'b', 'c'])
assert ':' in m

# Test emit_receipt
r = emit_receipt('gate_validation_receipt', {'gate': 't48h', 'status': 'pass'})
assert 'hash' in r
assert 'ts' in r

print('All validations passed')
" 2>&1; then
    echo "PASS: all validations passed"
else
    echo "FAIL: validation failed"
    exit 1
fi

echo ""
echo "==================================="
echo "=== PASS: T+48h gate — SHIP IT ==="
echo "==================================="
exit 0
