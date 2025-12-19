"""
NEURON v4.2 Recovery Curves Test Suite
Tests for curves.py module: RecoveryCurve, exponential_decay, power_law, linear
"""

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from curves import (
    RecoveryCurve,
    exponential_decay,
    power_law,
    linear,
    RECOVERY_CURVE_MODELS,
    DEFAULT_RECOVERY_CURVE,
    EXP_DECAY_K,
    EXP_DECAY_TAU,
    POWER_LAW_ALPHA,
    POWER_LAW_SCALE,
)


class TestExponentialDecay:
    def test_zero_gap(self):
        """Test exponential decay at zero gap."""
        cost = exponential_decay(0)
        assert cost == 1.0

    def test_negative_gap(self):
        """Test exponential decay with negative gap."""
        cost = exponential_decay(-10)
        assert cost == 1.0

    def test_short_gap(self):
        """Test exponential decay for short gap."""
        cost = exponential_decay(15)
        assert 1.0 < cost < 2.0

    def test_medium_gap(self):
        """Test exponential decay for medium gap."""
        cost = exponential_decay(60, tau=120)
        assert 2.0 < cost < 3.5

    def test_long_gap(self):
        """Test exponential decay for long gap."""
        cost = exponential_decay(120, tau=120)
        assert 3.0 < cost < 4.0

    def test_asymptote(self):
        """Test that cost approaches K+1 asymptotically."""
        cost_1h = exponential_decay(60)
        cost_2h = exponential_decay(120)
        cost_4h = exponential_decay(240)
        cost_8h = exponential_decay(480)

        # Cost should increase but with diminishing returns
        assert cost_1h < cost_2h < cost_4h < cost_8h
        assert cost_8h < EXP_DECAY_K + 1 + 0.1  # Should approach but not exceed K+1

    def test_tau_effect(self):
        """Test that lower tau increases cost faster."""
        cost_fast = exponential_decay(60, tau=60)   # Fast recovery expected
        cost_slow = exponential_decay(60, tau=240)  # Slow recovery expected

        # Same gap, but fast tau should give higher cost
        assert cost_fast > cost_slow

    def test_k_effect(self):
        """Test that K controls maximum cost."""
        cost_k2 = exponential_decay(1000, K=2)
        cost_k6 = exponential_decay(1000, K=6)

        # At very long gaps, should approach K+1
        assert cost_k2 < 3.1  # ~K+1 = 3
        assert cost_k6 < 7.1  # ~K+1 = 7


class TestPowerLaw:
    def test_zero_gap(self):
        """Test power law at zero gap."""
        cost = power_law(0)
        assert cost == 1.0

    def test_negative_gap(self):
        """Test power law with negative gap."""
        cost = power_law(-10)
        assert cost == 1.0

    def test_short_gap(self):
        """Test power law for short gap."""
        cost = power_law(15)
        assert 1.0 < cost

    def test_unbounded_growth(self):
        """Test that power law grows without bound (unlike exponential)."""
        cost_8h = power_law(480)
        cost_24h = power_law(1440)

        # Power law should continue growing
        assert cost_24h > cost_8h
        # At very long gaps, power law exceeds exponential's asymptote
        assert cost_24h > EXP_DECAY_K + 1

    def test_alpha_effect(self):
        """Test that alpha controls growth rate."""
        cost_a025 = power_law(100, alpha=0.25)
        cost_a05 = power_law(100, alpha=0.5)
        cost_a075 = power_law(100, alpha=0.75)

        # Higher alpha = faster growth
        assert cost_a025 < cost_a05 < cost_a075

    def test_scale_effect(self):
        """Test that scale multiplies the power term."""
        cost_s1 = power_law(100, scale=1)
        cost_s2 = power_law(100, scale=2)
        cost_s4 = power_law(100, scale=4)

        # Scale should multiply the additional cost
        assert cost_s2 > cost_s1
        assert cost_s4 > cost_s2


class TestLinear:
    def test_zero_gap(self):
        """Test linear at zero gap."""
        cost = linear(0)
        assert cost == 1.0

    def test_negative_gap(self):
        """Test linear with negative gap."""
        cost = linear(-10)
        assert cost == 1.0

    def test_at_tau(self):
        """Test linear at gap = tau."""
        cost = linear(120, tau=120)
        assert cost == 2.0  # 1 + 120/120

    def test_at_double_tau(self):
        """Test linear at gap = 2*tau."""
        cost = linear(240, tau=120)
        assert cost == 3.0  # 1 + 240/120

    def test_linear_growth(self):
        """Test that cost grows linearly."""
        cost_1h = linear(60, tau=120)
        cost_2h = linear(120, tau=120)
        cost_3h = linear(180, tau=120)

        # Differences should be constant
        diff1 = cost_2h - cost_1h
        diff2 = cost_3h - cost_2h

        assert abs(diff1 - diff2) < 0.001


class TestRecoveryCurveClass:
    def test_default_init(self):
        """Test RecoveryCurve initializes with defaults."""
        curve = RecoveryCurve()

        assert curve.model == DEFAULT_RECOVERY_CURVE
        assert hasattr(curve, "K")
        assert hasattr(curve, "tau")

    def test_all_models(self):
        """Test all models can be instantiated."""
        for model in RECOVERY_CURVE_MODELS:
            curve = RecoveryCurve(model=model)
            assert curve.model == model

    def test_invalid_model_defaults(self):
        """Test invalid model falls back to default."""
        curve = RecoveryCurve(model="invalid_model")
        assert curve.model == DEFAULT_RECOVERY_CURVE

    def test_custom_params_exponential(self):
        """Test custom parameters for exponential decay."""
        curve = RecoveryCurve(model="exponential_decay", K=2.0, tau=60.0)

        assert curve.K == 2.0
        assert curve.tau == 60.0

    def test_custom_params_power_law(self):
        """Test custom parameters for power law."""
        curve = RecoveryCurve(model="power_law", alpha=0.7, scale=3.0)

        assert curve.alpha == 0.7
        assert curve.scale == 3.0

    def test_custom_params_linear(self):
        """Test custom parameters for linear."""
        curve = RecoveryCurve(model="linear", tau=60.0)

        assert curve.tau == 60.0

    def test_cost_method(self):
        """Test cost method works for all models."""
        for model in RECOVERY_CURVE_MODELS:
            curve = RecoveryCurve(model=model)
            cost = curve.cost(60)
            assert cost >= 1.0

    def test_cost_matches_function(self):
        """Test cost method matches standalone functions."""
        curve_exp = RecoveryCurve(model="exponential_decay", K=EXP_DECAY_K, tau=EXP_DECAY_TAU)
        curve_pow = RecoveryCurve(model="power_law", alpha=POWER_LAW_ALPHA, scale=POWER_LAW_SCALE)
        curve_lin = RecoveryCurve(model="linear", tau=EXP_DECAY_TAU)

        gap = 90

        assert abs(curve_exp.cost(gap) - exponential_decay(gap)) < 0.001
        assert abs(curve_pow.cost(gap) - power_law(gap)) < 0.001
        assert abs(curve_lin.cost(gap) - linear(gap)) < 0.001


class TestRecoveryCurveFit:
    def test_fit_empty_data(self):
        """Test fit with empty data."""
        curve = RecoveryCurve()

        result = curve.fit([], [])

        assert result["fit_score"] == 0.0
        assert result["model"] == DEFAULT_RECOVERY_CURVE

    def test_fit_mismatched_lengths(self):
        """Test fit with mismatched data lengths."""
        curve = RecoveryCurve()

        result = curve.fit([10, 20, 30], [1.5, 2.0])  # Mismatched

        assert result["fit_score"] == 0.0

    def test_fit_perfect_data(self):
        """Test fit with perfect data."""
        curve = RecoveryCurve(model="exponential_decay")

        # Generate perfect data from the model
        gaps = [15, 30, 60, 90, 120, 180, 240]
        recoveries = [curve.cost(g) for g in gaps]

        result = curve.fit(gaps, recoveries)

        assert result["fit_score"] >= 0.99
        assert result["n_samples"] == len(gaps)

    def test_fit_returns_parameters(self):
        """Test fit returns model parameters."""
        curve = RecoveryCurve(model="exponential_decay", K=3.0, tau=90.0)

        result = curve.fit([60], [2.5])

        assert "parameters" in result
        assert result["parameters"]["K"] == 3.0
        assert result["parameters"]["tau"] == 90.0


class TestRecoveryCurveCompare:
    def test_compare_same_model(self):
        """Test comparing identical models."""
        curve1 = RecoveryCurve(model="exponential_decay")
        curve2 = RecoveryCurve(model="exponential_decay")

        result = curve1.compare(curve2, [30, 60, 90])

        assert result["avg_absolute_difference"] == 0.0

    def test_compare_different_models(self):
        """Test comparing different models."""
        curve_exp = RecoveryCurve(model="exponential_decay")
        curve_pow = RecoveryCurve(model="power_law")

        result = curve_exp.compare(curve_pow, [30, 60, 90, 120])

        assert result["model_a"] == "exponential_decay"
        assert result["model_b"] == "power_law"
        assert result["gaps_compared"] == 4
        assert result["avg_absolute_difference"] > 0

    def test_compare_returns_costs(self):
        """Test compare returns individual costs."""
        curve1 = RecoveryCurve(model="exponential_decay")
        curve2 = RecoveryCurve(model="linear")

        result = curve1.compare(curve2, [60, 120])

        assert len(result["self_costs"]) == 2
        assert len(result["other_costs"]) == 2


class TestRecoveryCurveRepr:
    def test_repr(self):
        """Test string representation."""
        curve = RecoveryCurve(model="exponential_decay", K=4.0, tau=120.0)

        repr_str = repr(curve)

        assert "exponential_decay" in repr_str
        assert "K" in repr_str
        assert "tau" in repr_str


class TestModelComparison:
    def test_exponential_vs_power_law_short_gap(self):
        """Test that models differ for short gaps."""
        cost_exp = exponential_decay(30)
        cost_pow = power_law(30)

        # Both should be > 1
        assert cost_exp > 1.0
        assert cost_pow > 1.0
        # They should differ
        assert cost_exp != cost_pow

    def test_exponential_vs_linear(self):
        """Test exponential is bounded, linear is not."""
        exp_8h = exponential_decay(480)
        lin_8h = linear(480, tau=120)

        # Exponential approaches asymptote, linear keeps growing
        assert exp_8h < 5.0  # Bounded by K+1
        assert lin_8h == 5.0  # 1 + 480/120 = 5.0

        # At very long gaps, linear exceeds exponential
        exp_24h = exponential_decay(1440)
        lin_24h = linear(1440, tau=120)

        assert lin_24h > exp_24h

    def test_model_selection_rationale(self):
        """Test that exponential fits short gaps better (as claimed in spec)."""
        # Exponential decay: steep rise for short gaps, levels off
        # Power law: slower start, continuous rise
        # At 60 min gap:
        exp_60 = exponential_decay(60, tau=120)
        pow_60 = power_law(60)

        # At 120 min gap:
        exp_120 = exponential_decay(120, tau=120)
        pow_120 = power_law(120)

        # Exponential should show more initial cost
        # but then diminishing returns
        exp_ratio = exp_120 / exp_60
        pow_ratio = pow_120 / pow_60

        # Power law has more consistent growth
        # (ratio closer to sqrt(2) for alpha=0.5)
        assert exp_ratio < pow_ratio
