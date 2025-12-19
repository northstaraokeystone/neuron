# NEURON CLAUDEME Compliance Report

**Generated:** 2025-12-19
**Version:** NEURON v4.1
**Status:** ✅ FULLY COMPLIANT

---

## §0 LAWS

| Law | Requirement | Status | Evidence |
|-----|-------------|--------|----------|
| LAW_1 | "No receipt → not real" - Every function emits receipt | ✅ PASS | `emit_receipt()` in neuron.py:157, used throughout |
| LAW_2 | "No test → not shipped" - Every function has test with assertions | ✅ PASS | 53 tests with assertions in tests/test_neuron.py |
| LAW_3 | "No gate → not alive" - Gate scripts exist and pass | ✅ PASS | gate_t2h.sh, gate_t24h.sh, gate_t48h.sh all pass |

---

## §2 THE STACK

| Standard | Requirement | Status | Evidence |
|----------|-------------|--------|----------|
| HASH | SHA256 + BLAKE3 (dual_hash) | ✅ PASS | `dual_hash()` at neuron.py:110 |
| Storage | PostgreSQL/JSONL for audit | ✅ PASS | receipts.jsonl append-only format |
| Crypto | ALWAYS dual-hash, never single | ✅ PASS | All hashes use `dual_hash()` |

---

## §4 RECEIPT BLOCKS

| Receipt Type | SCHEMA | EMIT | TEST | STOPRULE |
|--------------|--------|------|------|----------|
| emit_receipt | ✅ {type, ts, hash, **data} | ✅ neuron.py:157 | ✅ test_neuron.py:320 | ✅ StopRule on failure |
| append | ✅ ledger_schema.json | ✅ neuron.py:228 | ✅ test_neuron.py:129 | ✅ ValueError on invalid |
| stress_test | ✅ Documented in stress.py | ✅ _emit_receipt | ✅ test_stress.py | ✅ Isolated failure handling |

---

## §7 ANTI-PATTERNS

| Anti-Pattern | Replacement | Status | Evidence |
|--------------|-------------|--------|----------|
| Function without emit_receipt() | Add receipt to return | ✅ FIXED | `emit_receipt()` added |
| hashlib.sha256() alone | dual_hash() | ✅ PASS | All hashing uses dual_hash |
| except: pass | except: stoprule_X(e) | ✅ PASS | StopRule class added |
| Global variable | Ledger entry | ✅ PASS | Config as constants only |
| print(result) | emit_receipt("...", result) | ✅ PASS | CLI uses emit_receipt |
| Test without assert | Add SLO assert | ✅ PASS | 53 tests with assertions |

---

## §8 CORE FUNCTIONS

| Function | Requirement | Status | Location |
|----------|-------------|--------|----------|
| `dual_hash(data: bytes \| str) -> str` | SHA256:BLAKE3 | ✅ PRESENT | neuron.py:110 |
| `emit_receipt(receipt_type: str, data: dict) -> dict` | Receipt emission | ✅ PRESENT | neuron.py:157 |
| `merkle(items: list) -> str` | Merkle root hash | ✅ PRESENT | neuron.py:119 |
| `class StopRule(Exception)` | Exception class | ✅ PRESENT | neuron.py:101 |

---

## §9 FILE STRUCTURE

| File | Requirement | Status |
|------|-------------|--------|
| `spec.md` | T+2h requirement | ✅ EXISTS |
| `ledger_schema.json` | T+2h requirement | ✅ EXISTS |
| `cli.py` | T+2h requirement | ✅ EXISTS |
| `receipts.jsonl` | Append-only ledger | ✅ CONFIGURED |
| `neuron.py` | Core implementation | ✅ EXISTS |
| `stress.py` | Stress testing | ✅ EXISTS |
| `tests/test_*.py` | With assertions | ✅ EXISTS (4 files) |
| `tests/conftest.py` | Test config | ✅ EXISTS |
| `gate_t2h.sh` | T+2h gate | ✅ EXISTS |
| `gate_t24h.sh` | T+24h gate | ✅ EXISTS |
| `gate_t48h.sh` | T+48h gate | ✅ EXISTS |
| `MANIFEST.anchor` | Deploy artifact | ✅ EXISTS |

---

## §10 COMMIT FORMAT

Commits follow CLAUDEME format:
```
<type>(<scope>): <description ≤50 chars>

Receipt: <receipt_type>
SLO: <threshold affected | none>
Gate: <t2h | t24h | t48h | post>
```

---

## VIOLATIONS FOUND

| # | File:Line | Issue | Fix Applied |
|---|-----------|-------|-------------|
| 1 | neuron.py | Missing StopRule class | Added at line 101 |
| 2 | neuron.py | Missing emit_receipt() | Added at line 157 |
| 3 | neuron.py | Missing merkle() | Added at line 119 |
| 4 | (missing) | No cli.py | Created cli.py |
| 5 | (missing) | No gate scripts | Created gate_t2h.sh, gate_t24h.sh, gate_t48h.sh |
| 6 | (missing) | No MANIFEST.anchor | Created MANIFEST.anchor |
| 7 | tests/ | No conftest.py | Created tests/conftest.py |
| 8 | tests/test_neuron.py | Missing tests for new functions | Added TestEmitReceipt, TestMerkle, TestStopRule |

---

## FIXES APPLIED

### 1. Core Functions Added (neuron.py)
- **StopRule class** (line 101): Exception for stoprule failures
- **merkle()** (line 119): Merkle root hash computation
- **emit_receipt()** (line 157): Receipt emission with dual-hash

### 2. Files Created
- **cli.py**: Command-line interface for T+2h gate compliance
- **gate_t2h.sh**: T+2h gate script (spec, schema, cli verification)
- **gate_t24h.sh**: T+24h gate script (tests, receipts, assertions)
- **gate_t48h.sh**: T+48h gate script (stoprules, full validation)
- **MANIFEST.anchor**: Deploy artifact with version info
- **tests/conftest.py**: Shared pytest configuration

### 3. Tests Added (tests/test_neuron.py)
- **TestEmitReceipt**: 3 tests for emit_receipt()
- **TestMerkle**: 7 tests for merkle()
- **TestStopRule**: 6 tests for StopRule class

---

## GATE STATUS

```
$ ./gate_t2h.sh && ./gate_t24h.sh && ./gate_t48h.sh

=== PASS: T+2h gate ===
=== PASS: T+24h gate ===
=== PASS: T+48h gate — SHIP IT ===
```

---

## SUMMARY

| Metric | Value |
|--------|-------|
| Total violations found | 8 |
| Total fixes applied | 8 |
| Core tests passing | 53/53 |
| Gate scripts passing | 3/3 |
| CLAUDEME compliance | 100% |

---

**The document IS the standard. The standard IS the code. The code IS the receipt.**

```python
assert understand(CLAUDEME) == True, "Re-read from §0"
```
