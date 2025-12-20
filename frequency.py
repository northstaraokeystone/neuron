"""
NEURON v4.5: Frequency Tuning Module
Natural frequency alignment: circadian, token windows, light-speed delays.
Oscillations tuned to biological, AI, and physical rhythms.
~200 lines. Frequency determines resonance.
"""

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

try:
    import blake3

    HAS_BLAKE3 = True
except ImportError:
    HAS_BLAKE3 = False


# ============================================
# v4.5 FREQUENCY CONSTANTS
# ============================================
FREQUENCY_SOURCES = [
    "HUMAN_CIRCADIAN",  # 24h
    "HUMAN_FOCUS",  # 90min
    "GROK_TOKEN_REFRESH",  # Dynamic
    "MARS_LIGHT_DELAY",  # 3-22min
    "INTERSTELLAR_BURST",  # 4yr
]
DEFAULT_FREQUENCY = "HUMAN_FOCUS"

# Hardcoded fallback values (if config not loaded)
FALLBACK_FREQUENCIES = {
    "HUMAN_CIRCADIAN": {
        "period_seconds": 86400,
        "description": "24-hour human attention cycle",
    },
    "HUMAN_FOCUS": {
        "period_seconds": 5400,
        "description": "90-minute ultradian focus rhythm",
    },
    "GROK_TOKEN_REFRESH": {
        "period_seconds": 82,
        "description": "~8192 tokens at 100 tok/s",
    },
    "MARS_LIGHT_DELAY": {
        "period_seconds": 750,
        "description": "~12.5 min average light delay",
    },
    "INTERSTELLAR_BURST": {
        "period_seconds": 126000000,
        "description": "4-year Proxima Centauri cycle",
    },
}


def dual_hash(data: bytes | str) -> str:
    """Compute SHA256:BLAKE3 hash per CLAUDEME §8."""
    if isinstance(data, str):
        data = data.encode("utf-8")
    sha256_hex = hashlib.sha256(data).hexdigest()
    blake3_hex = (
        blake3.blake3(data).hexdigest()
        if HAS_BLAKE3
        else hashlib.sha256(b"blake3:" + data).hexdigest()
    )
    return f"{sha256_hex}:{blake3_hex}"


def _get_config_path() -> Path:
    """Get path to frequencies.json config file."""
    base = Path(os.environ.get("NEURON_BASE", Path(__file__).parent))
    return base / "config" / "frequencies.json"


def load_frequencies(config_path: str | None = None) -> dict:
    """Load frequencies from config file.

    Args:
        config_path: Path to frequencies.json (default: config/frequencies.json)

    Returns:
        Dict with frequencies configuration
    """
    if config_path is None:
        config_path = str(_get_config_path())

    config_file = Path(config_path)

    if config_file.exists():
        try:
            with open(config_file) as f:
                config = json.load(f)
            return config
        except (json.JSONDecodeError, IOError):
            pass

    # Return fallback structure
    return {
        "frequencies": FALLBACK_FREQUENCIES,
        "default": DEFAULT_FREQUENCY,
        "adaptive_enabled": True,
    }


def tune_frequency(source: str, config_path: str | None = None) -> dict:
    """Select frequency by source name.

    v4.5: "oscillations tuned to natural frequencies"

    Args:
        source: Frequency source name (e.g., "HUMAN_CIRCADIAN")
        config_path: Optional path to config file

    Returns:
        frequency_receipt with source, period_seconds, frequency_hz

    Raises:
        ValueError: If source is invalid
    """
    config = load_frequencies(config_path)
    frequencies = config.get("frequencies", FALLBACK_FREQUENCIES)

    if source not in frequencies and source not in FALLBACK_FREQUENCIES:
        raise ValueError(
            f"Invalid frequency source: {source}. Valid: {list(frequencies.keys())}"
        )

    freq_config = frequencies.get(source, FALLBACK_FREQUENCIES.get(source))

    # Handle dynamic computation
    if freq_config.get("compute") == "dynamic":
        period_seconds = compute_period(source, {})
    elif "period_range" in freq_config:
        # For ranges like Mars delay, use midpoint
        low, high = freq_config["period_range"]
        period_seconds = (low + high) / 2
    else:
        period_seconds = freq_config.get("period_seconds", 5400)

    frequency_hz = 1.0 / period_seconds if period_seconds > 0 else 0.0

    # Build receipt
    payload = json.dumps(
        {
            "source": source,
            "period_seconds": period_seconds,
            "frequency_hz": frequency_hz,
        },
        sort_keys=True,
    )

    receipt = {
        "source": source,
        "period_seconds": period_seconds,
        "frequency_hz": frequency_hz,
        "adaptive": config.get("adaptive_enabled", True),
        "reason": freq_config.get("description", ""),
        "ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "payload_hash": dual_hash(payload),
    }

    return receipt


def compute_period(source: str, context: dict) -> float:
    """Calculate period based on source and context.

    Args:
        source: Frequency source name
        context: Additional context (token counts, latency, etc.)

    Returns:
        Period in seconds
    """
    if source == "HUMAN_CIRCADIAN":
        return 86400.0  # 24 hours

    elif source == "HUMAN_FOCUS":
        return 5400.0  # 90 minutes (ultradian rhythm)

    elif source == "GROK_TOKEN_REFRESH":
        # Dynamic based on token context
        base_tokens = context.get("base_tokens", 8192)
        tokens_per_second = context.get("tokens_per_second", 100)
        return base_tokens / tokens_per_second

    elif source == "MARS_LIGHT_DELAY":
        # Variable 3-22 minutes based on orbital position
        # Return average
        min_delay = 180  # 3 minutes
        max_delay = 1320  # 22 minutes
        position = context.get("orbital_position", 0.5)  # 0-1
        return min_delay + (max_delay - min_delay) * position

    elif source == "INTERSTELLAR_BURST":
        return 126_000_000.0  # 4 years in seconds

    else:
        # Default to HUMAN_FOCUS
        return 5400.0


def adaptive_frequency(ledger: list, target_transitions: int = 1) -> float:
    """Auto-tune frequency based on observed phase transitions.

    v4.5: "More transitions = keep frequency. Fewer = adjust."

    Args:
        ledger: Active ledger with transition history
        target_transitions: Target number of transitions to achieve

    Returns:
        Adjusted frequency in Hz
    """
    # Count recent phase transitions (entries with transition markers)
    recent_transitions = sum(
        1
        for e in ledger[-100:]
        if e.get("event_type") in ["phase_transition", "axiom_law_discovery"]
    )

    # Base frequency (HUMAN_FOCUS)
    base_hz = 1.0 / 5400.0

    if recent_transitions >= target_transitions:
        # Keep frequency - resonance is working
        return base_hz
    else:
        # Adjust frequency - try to find resonance
        # Increase frequency to probe more quickly
        adjustment_factor = 1.0 + (target_transitions - recent_transitions) * 0.1
        return base_hz * adjustment_factor


def get_frequency_for_gap_duration(gap_minutes: float) -> str:
    """Select appropriate frequency source based on gap duration.

    Matches oscillation frequency to gap characteristics.

    Args:
        gap_minutes: Duration of detected gap

    Returns:
        Recommended frequency source name
    """
    if gap_minutes < 30:
        return "HUMAN_FOCUS"  # Short gaps - fast oscillation
    elif gap_minutes < 1440:  # < 24 hours
        return "HUMAN_CIRCADIAN"
    elif gap_minutes < 30000:  # < 20 days
        return "MARS_LIGHT_DELAY"
    else:
        return "INTERSTELLAR_BURST"  # Long isolation


def validate_frequency_config(config: dict) -> list[str]:
    """Validate frequencies.json configuration.

    Args:
        config: Loaded config dict

    Returns:
        List of validation errors (empty if valid)
    """
    errors = []

    if "frequencies" not in config:
        errors.append("Missing 'frequencies' key")
        return errors

    frequencies = config["frequencies"]

    for source in FREQUENCY_SOURCES:
        if source not in frequencies:
            errors.append(f"Missing frequency source: {source}")
            continue

        freq_config = frequencies[source]

        # Check for period definition
        has_period = (
            "period_seconds" in freq_config
            or "period_range" in freq_config
            or freq_config.get("compute") == "dynamic"
        )

        if not has_period:
            errors.append(f"{source}: No period definition found")

        # Validate period values
        if "period_seconds" in freq_config:
            if freq_config["period_seconds"] <= 0:
                errors.append(f"{source}: period_seconds must be > 0")

        if "period_range" in freq_config:
            low, high = freq_config["period_range"]
            if low <= 0 or high <= 0:
                errors.append(f"{source}: period_range values must be > 0")
            if low >= high:
                errors.append(f"{source}: period_range low must be < high")

    return errors


if __name__ == "__main__":
    print("NEURON v4.5 - Frequency Tuning Module")
    print(f"Available sources: {FREQUENCY_SOURCES}")
    print(f"Default frequency: {DEFAULT_FREQUENCY}")
    print()

    for source in FREQUENCY_SOURCES:
        receipt = tune_frequency(source)
        print(f"{source}:")
        print(f"  Period: {receipt['period_seconds']:.2f}s")
        print(f"  Frequency: {receipt['frequency_hz']:.6f} Hz")
