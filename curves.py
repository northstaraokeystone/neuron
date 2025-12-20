"""
NEURON v4.2: Recovery Curve Module
Pluggable recovery cost models for alpha calculation.
Default: exponential decay (Monsell 2003).
Alternative: power law (Altmann & Trafton 2002).
~60 lines. Model-agnostic. Fit-capable.
"""

import math
from typing import Literal

# Recovery curve constants
RECOVERY_CURVE_MODELS = ["exponential_decay", "power_law", "linear"]
DEFAULT_RECOVERY_CURVE = "exponential_decay"

# Exponential decay parameters (Monsell 2003 task-set inertia)
EXP_DECAY_K = 4.0  # Maximum additional cost
EXP_DECAY_TAU = 120.0  # Time constant (default, configurable)

# Power law parameters (Altmann & Trafton 2002 memory-for-goals)
POWER_LAW_ALPHA = 0.5  # Decay exponent
POWER_LAW_SCALE = 2.0  # Scale factor


def exponential_decay(
    gap_minutes: float, K: float = EXP_DECAY_K, tau: float = EXP_DECAY_TAU
) -> float:
    """
    Monsell 2003 task-set inertia model.

    Cost approaches K+1 asymptotically.
    tau controls how fast cost rises.

    Args:
        gap_minutes: Time gap since last activity in minutes
        K: Maximum additional cost (default 4.0)
        tau: Time constant (default 120.0 minutes)

    Returns:
        Recovery cost multiplier (1.0 to ~5.0)
    """
    if gap_minutes <= 0:
        return 1.0
    return 1.0 + K * (1.0 - math.exp(-gap_minutes / tau))


def power_law(
    gap_minutes: float, alpha: float = POWER_LAW_ALPHA, scale: float = POWER_LAW_SCALE
) -> float:
    """
    Altmann & Trafton 2002 memory-for-goals model.

    Slower rise than exponential for short gaps.
    Faster rise for very long gaps.

    Args:
        gap_minutes: Time gap since last activity in minutes
        alpha: Decay exponent (default 0.5)
        scale: Scale factor (default 2.0)

    Returns:
        Recovery cost multiplier (1.0+)
    """
    if gap_minutes <= 0:
        return 1.0
    return 1.0 + scale * (gap_minutes**alpha)


def linear(gap_minutes: float, tau: float = EXP_DECAY_TAU) -> float:
    """
    Simple linear recovery model (baseline).

    Args:
        gap_minutes: Time gap since last activity in minutes
        tau: Time constant for slope (default 120.0 minutes)

    Returns:
        Recovery cost multiplier (1.0+)
    """
    if gap_minutes <= 0:
        return 1.0
    return 1.0 + gap_minutes / tau


class RecoveryCurve:
    """
    Pluggable recovery cost models.
    Default: exponential decay (Monsell 2003).
    Alternative: power law (Altmann & Trafton 2002).
    """

    def __init__(
        self,
        model: Literal[
            "exponential_decay", "power_law", "linear"
        ] = DEFAULT_RECOVERY_CURVE,
        **params,
    ):
        """
        Initialize recovery curve with specified model.

        Args:
            model: Model type - "exponential_decay", "power_law", or "linear"
            **params: Model-specific parameters:
                - exponential_decay: K (float), tau (float)
                - power_law: alpha (float), scale (float)
                - linear: tau (float)
        """
        self.model = model if model in RECOVERY_CURVE_MODELS else DEFAULT_RECOVERY_CURVE
        self.params = params

        # Set defaults based on model
        if self.model == "exponential_decay":
            self.K = params.get("K", EXP_DECAY_K)
            self.tau = params.get("tau", EXP_DECAY_TAU)
        elif self.model == "power_law":
            self.alpha = params.get("alpha", POWER_LAW_ALPHA)
            self.scale = params.get("scale", POWER_LAW_SCALE)
        elif self.model == "linear":
            self.tau = params.get("tau", EXP_DECAY_TAU)

    def cost(self, gap_minutes: float) -> float:
        """
        Compute recovery cost for a given gap.

        Args:
            gap_minutes: Time gap since last activity in minutes

        Returns:
            Recovery cost multiplier
        """
        if self.model == "exponential_decay":
            return exponential_decay(gap_minutes, K=self.K, tau=self.tau)
        elif self.model == "power_law":
            return power_law(gap_minutes, alpha=self.alpha, scale=self.scale)
        elif self.model == "linear":
            return linear(gap_minutes, tau=self.tau)
        return 1.0

    def fit(self, gaps: list[float], recoveries: list[float]) -> dict:
        """
        Fit model to observed data and return goodness-of-fit metrics.

        Args:
            gaps: List of gap durations in minutes
            recoveries: List of observed recovery costs

        Returns:
            Dict with model, parameters, and fit score (R-squared)
        """
        if not gaps or not recoveries or len(gaps) != len(recoveries):
            return {
                "model": self.model,
                "parameters": self._get_params(),
                "fit_score": 0.0,
            }

        # Compute predicted values
        predicted = [self.cost(g) for g in gaps]

        # Compute R-squared
        mean_observed = sum(recoveries) / len(recoveries)
        ss_tot = sum((obs - mean_observed) ** 2 for obs in recoveries)
        ss_res = sum((obs - pred) ** 2 for obs, pred in zip(recoveries, predicted))

        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

        return {
            "model": self.model,
            "parameters": self._get_params(),
            "fit_score": round(max(0.0, r_squared), 4),
            "n_samples": len(gaps),
        }

    def compare(self, other: "RecoveryCurve", gaps: list[float]) -> dict:
        """
        Compare this model's predictions against another model.

        Args:
            other: Another RecoveryCurve instance
            gaps: List of gap durations to compare at

        Returns:
            Dict with comparison metrics
        """
        self_costs = [self.cost(g) for g in gaps]
        other_costs = [other.cost(g) for g in gaps]

        differences = [abs(s - o) for s, o in zip(self_costs, other_costs)]
        avg_diff = sum(differences) / len(differences) if differences else 0.0

        return {
            "model_a": self.model,
            "model_b": other.model,
            "gaps_compared": len(gaps),
            "avg_absolute_difference": round(avg_diff, 4),
            "self_costs": [round(c, 2) for c in self_costs],
            "other_costs": [round(c, 2) for c in other_costs],
        }

    def _get_params(self) -> dict:
        """Get current model parameters."""
        if self.model == "exponential_decay":
            return {"K": self.K, "tau": self.tau}
        elif self.model == "power_law":
            return {"alpha": self.alpha, "scale": self.scale}
        elif self.model == "linear":
            return {"tau": self.tau}
        return {}

    def __repr__(self) -> str:
        return f"RecoveryCurve(model='{self.model}', params={self._get_params()})"
