# RESONANCE PROTOCOL v2.0 - Specification

## Overview

The Resonance Protocol is a comprehensive research and development roadmap for
Neuralink neural interface analysis, implementing receipts-native data pipeline
for neurophysiology findings and HDC simulation framework.

## Paradigm Inversion

**OLD:** Build separate neural processing system disconnected from QED/ProofPack architecture

**NEW:** Extend QED/ProofPack patterns to neurophysiology - receipts ARE the neural state,
HDC IS the compression, FL IS the sovereignty guarantee

## Core Principles

### 1. Privacy by Architecture
Raw Neural Data Never Leaves the User's Local Ecosystem. This is not a feature -
it's the fundamental design constraint. Federated Learning aggregates model deltas,
never spike trains.

### 2. Physics-Bounded Latency
Phase-locked stimulation requires <2ms loop latency. At 200 Hz ripple frequency
(5ms period), the system must react within a specific phase window. This makes
cloud processing physically impossible.

### 3. Holographic Robustness
Thread retraction is mechanically inevitable due to brain pulsation. HDC's
holographic data representation maintains high-fidelity decoding even with
30% channel loss. This is not optimization - it's physical necessity.

### 4. Receipts-Native Architecture
Per CLAUDEME: "No receipt -> not real." All operations emit receipts to the
append-only ledger for audit and verification.

## Module Specifications

### Core (src/core.py)
- `neural_hash()`: Compute dual-hash of spike timing within temporal window
- `validate_channel_quality()`: Check impedance < 100 kOhm, SNR > 6 dB
- `enforce_thermal_limit()`: Raise StopRule if delta T > 1.0 C

### HDC Encoder (src/hdc.py)
- Dimension: 10,000 bits
- Input: 1,024 channels (N1) to 3,072 (N2)
- Dropout tolerance: 30% channel loss with >85% accuracy
- Orthogonality: Mean cosine similarity < 0.1 for random vectors

### Oscillation Detector (src/oscillation_detector.py)
- SWR: 150-250 Hz (Sharp-Wave Ripples)
- Spindle: 10-16 Hz (Thalamic)
- SO: <1 Hz (Slow Oscillation)
- Latency: p95 < 2.0 ms

### Phase Predictor (src/phase_predictor.py)
- Phase error target: < pi/4 radians for 95% of events
- Prediction latency: < 1 ms (leaving 1 ms for stimulation trigger)

### Federated Coordinator (src/federated_coordinator.py)
- HARD REQUIREMENT: Zero raw data transmissions
- Minimum participants: 3 for aggregation
- Update interval: 24 hours (86400 seconds)
- Privacy budget tracking: Differential privacy epsilon

### Stimulation Controller (src/stim_controller.py)
- Charge density limit: 30 uC/cm^2 (Shannon limit)
- Thermal limit: 1.0 C temperature rise
- Sub-threshold only: <1 mA typical

### Thread Monitor (src/thread_monitor.py)
- Monitor: 1,024 channels at 1 Hz
- Impedance threshold: 100 kOhm
- SNR threshold: 6 dB
- Retraction warning: >0.5 mm

### Entropy (src/entropy.py)
- Shannon entropy for probability distributions
- HDC vector entropy (target > 0.95 for random vectors)
- Class separation metric

## Receipt Types

| Receipt Type | Emitted By | Key Fields |
|--------------|------------|------------|
| hdc_encoding | hdc.py | input_channels, hypervector_dim, dropout_rate, vector_hash |
| swr_detection | oscillation_detector.py | event_type, frequency_hz, amplitude, timestamp_ns |
| fl_update | federated_coordinator.py | model_delta_hash, n_local_samples, gradient_norm, privacy_budget |
| phase_lock | phase_predictor.py | target_oscillation, predicted_phase, actual_phase, lock_error_ms |
| channel_quality | thread_monitor.py | channel_id, snr_db, impedance_kohm, retraction_estimate_mm, status |
| stimulation | stim_controller.py | target_region, charge_density, pulse_width_us, safety_check_passed |

## Constants

```python
# Neural Signal Parameters
HDC_DIMENSION = 10000
N1_CHANNELS = 1024
N2_CHANNELS = 3072

# Oscillation Frequency Bands
SWR_FREQ_MIN = 150  # Hz
SWR_FREQ_MAX = 250  # Hz
SPINDLE_FREQ_MIN = 10  # Hz
SPINDLE_FREQ_MAX = 16  # Hz
SO_FREQ_MAX = 1.0  # Hz

# Timing Constraints
PHASE_LOCK_LATENCY_MS = 2.0
STIMULATION_WINDOW_MS = 1.0

# Safety Limits
STIM_CHARGE_DENSITY_LIMIT = 30.0  # uC/cm^2
THERMAL_LIMIT_DELTA_C = 1.0  # C
IMPEDANCE_THRESHOLD_KOHM = 100.0
SNR_THRESHOLD_DB = 6.0

# HDC Parameters
CHANNEL_DROPOUT_TOLERANCE = 0.30
MIN_CLASSIFICATION_ACCURACY = 0.85
ORTHOGONALITY_THRESHOLD = 0.1

# Federated Learning
FL_UPDATE_INTERVAL_SEC = 86400  # 24 hours
FL_MIN_PARTICIPANTS = 3
PRIVACY_BUDGET_EPSILON = 1.0

# Sampling
SAMPLING_RATE_HZ = 20000
```

## Gate Sequence

```
GATE 1: HDC_FOUNDATION
    - core.py extended with neural_hash()
    - hdc.py implements 10,000-bit encoding
    - Test: orthogonality check passes
    - Test: dropout robustness (30% loss) > 85% accuracy

GATE 2: OSCILLATION_DETECTION (parallel with GATE 3)
    - oscillation_detector.py implements SWR/Spindle/SO detection
    - Test: latency benchmark p95 < 2ms
    - Test: frequency accuracy +/-5% on synthetic data

GATE 3: FEDERATED_INFRASTRUCTURE (parallel with GATE 2)
    - federated_coordinator.py implements FL aggregation
    - Test: model convergence with synthetic updates
    - Test: zero raw data transmissions logged

GATE 4: PHASE_LOCKING
    - phase_predictor.py estimates oscillation phase
    - Test: phase prediction error < pi/4 radians
    - Test: stimulation trigger within <2ms window

GATE 5: SAFETY_COMPLIANCE
    - stim_controller.py enforces Shannon limit
    - Test: all simulated pulses < 30 uC/cm^2
    - Test: thermal limit check halts on 1C rise

GATE 6: INTEGRATION
    - All modules emit receipts per CLAUDEME
    - Thread monitor tracks channel quality
    - Test: end-to-end pipeline from spikes to stimulation
    - Test: receipts ledger validates full chain
```

## KILLED Items

| Item | Reason | Source |
|------|--------|--------|
| Cloud-based neural processing | Latency physics impossible | "Cloud processing is impossible... must reside on-chip" |
| Global neural data aggregation | Privacy violation | "Raw Neural Data Never Leaves the User's Local Ecosystem" |
| Fixed CNN architecture | Fails catastrophically on channel loss | "a standard CNN... may fail catastrophically" |
| Suprathreshold stimulation | Too crude for memory modulation | "Sub-threshold currents... do not trigger action potentials directly" |
| Single-hash verification | Insufficient for neural data | CLAUDEME section 8: dual-hash required |

## Verification Commands

```bash
# HDC orthogonality
python -c "from src.hdc import test_orthogonality; print(test_orthogonality(n_vectors=1000))"

# HDC dropout robustness
python -c "from src.hdc import test_dropout; print(test_dropout(dropout_rate=0.3))"

# SWR detection latency
python -c "from src.oscillation_detector import benchmark_latency; print(benchmark_latency())"

# Phase lock precision
python -c "from src.phase_predictor import test_phase_lock; print(test_phase_lock())"

# FL privacy guarantee
python -c "from src.federated_coordinator import verify_no_raw_data; print(verify_no_raw_data())"

# Stimulation safety
python -c "from src.stim_controller import test_shannon_limit; print(test_shannon_limit())"
```

## References

- Grok: Resonance Protocol v2.0+ research document
- CLAUDEME: Sections 4, 8 for receipt and hash requirements
- Shannon (1959): Electrical stimulation safety limits
- McCreery et al.: Shannon k=1.85 constant
- Monsell (2003): Task-switching recovery curves
