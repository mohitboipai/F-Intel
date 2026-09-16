import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from calculations.AdvancedVolEngine import (
    RealizedSemiVariance,
    CorsiHARModel,
    BipowerVariation,
    HigherMoments,
    AdvancedVolEngine
)
from calculations.StranglePositionSizer import StranglePositionSizer


class TestAdvancedVolEngine(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        # Create synthetic daily returns
        n_days = 120
        dates = [datetime(2026, 1, 1) + timedelta(days=i) for i in range(n_days)]
        normal_returns = np.random.normal(0.0005, 0.01, n_days)
        prices = [22000.0]
        for r in normal_returns:
            prices.append(prices[-1] * (1.0 + r))

        self.df_daily = pd.DataFrame({
            'date': dates,
            'close': prices[1:],
            'open': prices[:-1],
            'high': [p * 1.005 for p in prices[1:]],
            'low': [p * 0.995 for p in prices[1:]],
            'return': normal_returns
        })

    def test_semi_variance_bullish_grind(self):
        """Test semi-variance on predominantly positive returns (Bullish Grind)."""
        pos_returns = np.array([0.008, 0.006, 0.005, 0.007, -0.002, 0.006, -0.001, 0.009])
        semi = RealizedSemiVariance.calculate(pos_returns)

        self.assertGreater(semi["rv_plus"], semi["rv_minus"])
        self.assertLess(semi["vai"], 0)  # Negative VAI indicates upside momentum
        self.assertIn("BULLISH", semi["bias"])
        self.assertGreater(semi["rv_plus_pct"], 50.0)

    def test_semi_variance_downside_panic(self):
        """Test semi-variance on sharp crash returns (Downside Panic)."""
        crash_returns = np.array([-0.025, -0.018, 0.002, -0.030, 0.004, -0.015, -0.005])
        semi = RealizedSemiVariance.calculate(crash_returns)

        self.assertGreater(semi["rv_minus"], semi["rv_plus"])
        self.assertGreater(semi["vai"], 0)  # Positive VAI indicates downside crash hazard
        self.assertTrue("BEARISH" in semi["bias"] or "DOWNSIDE" in semi["bias"])
        self.assertGreater(semi["rv_minus_pct"], 50.0)

    def test_semi_variance_symmetric(self):
        """Test semi-variance on zero-mean balanced returns."""
        balanced_returns = np.array([0.01, -0.01, 0.01, -0.01, 0.005, -0.005])
        semi = RealizedSemiVariance.calculate(balanced_returns)
        self.assertAlmostEqual(semi["rv_plus_pct"], 50.0, delta=1.0)
        self.assertAlmostEqual(semi["vai"], 0.0, delta=0.05)
        self.assertEqual(semi["bias"], "BALANCED")

    def test_corsi_har_model_fitting_and_forecast(self):
        """Test Corsi (2009) HAR-RV model estimation and out-of-sample forecasting."""
        har_engine = CorsiHARModel(min_obs=30)
        # Generate realistic annualized RV series (e.g. 10% to 15%)
        rv_series = np.random.uniform(8.0, 16.0, 90)
        fitted = har_engine.fit(rv_series)
        self.assertTrue(fitted)
        self.assertTrue(har_engine.is_fitted)

        res = har_engine.forecast(current_rv_d=12.0, current_rv_w=11.5, current_rv_m=11.0)
        self.assertGreater(res["forecast_1d"], 0.0)
        self.assertGreater(res["forecast_5d"], 0.0)
        # Verify confidence intervals
        self.assertLess(res["forecast_1d_low"], res["forecast_1d"])
        self.assertGreater(res["forecast_1d_high"], res["forecast_1d"])
        self.assertLess(res["forecast_5d_low"], res["forecast_5d"])
        self.assertGreater(res["forecast_5d_high"], res["forecast_5d"])
        # Weights
        self.assertIn("beta_d", res["weights"])
        self.assertIn("beta_w", res["weights"])
        self.assertIn("beta_m", res["weights"])

    def test_bipower_variation_continuous_vs_jump(self):
        """Test Barndorff-Nielsen & Shephard jump decomposition."""
        # 1. Continuous smooth returns
        smooth_returns = np.random.normal(0, 0.002, 50)
        smooth_bv = BipowerVariation.decompose(smooth_returns)
        self.assertLessEqual(smooth_bv["jump_ratio"], 0.25)
        self.assertEqual(smooth_bv["jump_regime"], "CONTINUOUS_FLOW")

        # 2. Return series with an extreme discontinuous jump
        jump_returns = np.array(list(smooth_returns) + [-0.06, 0.04])
        jump_bv = BipowerVariation.decompose(jump_returns)
        self.assertGreater(jump_bv["jump_ratio"], smooth_bv["jump_ratio"])
        self.assertGreater(jump_bv["jump_rv"], 0.0)

    def test_higher_moments(self):
        """Test realized skewness and kurtosis calculations."""
        ret = np.random.normal(0, 0.01, 100)
        moments = HigherMoments.calculate(ret)
        self.assertIn("realized_skew", moments)
        self.assertIn("realized_kurtosis", moments)
        self.assertIn("tail_risk", moments)

    def test_advanced_vol_engine_pipeline(self):
        """Test full AdvancedVolEngine end-to-end integration."""
        engine = AdvancedVolEngine()
        result = engine.analyze(
            df_daily=self.df_daily,
            atm_iv=14.0
        )
        self.assertIn("semi_variance", result)
        self.assertIn("har_forecast", result)
        self.assertIn("jump_decomposition", result)
        self.assertIn("higher_moments", result)
        self.assertIn("forward_vrp", result)

        fvrp = result["forward_vrp"]
        self.assertIn("vrp_5d", fvrp)
        self.assertIn("verdict", fvrp)
        self.assertIn("action", fvrp)


class TestStranglePositionSizer(unittest.TestCase):
    def setUp(self):
        self.sizer = StranglePositionSizer()

    def test_base_capital_capacity(self):
        """Test base margin capacity equation: floor(capital * 70% / 1.2L)."""
        res_5l = self.sizer.calculate_sizing(capital=500000)
        self.assertEqual(res_5l["base_lots"], 2)  # 5L * 0.70 = 3.5L / 1.2L = 2 lots

        res_10l = self.sizer.calculate_sizing(capital=1000000)
        self.assertEqual(res_10l["base_lots"], 5)  # 10L * 0.70 = 7.0L / 1.2L = 5 lots

        res_20l = self.sizer.calculate_sizing(capital=2000000)
        self.assertEqual(res_20l["base_lots"], 11)  # 20L * 0.70 = 14.0L / 1.2L = 11 lots

    def test_vrp_edge_multipliers(self):
        """Test that higher forward VRP increases size and negative VRP halts selling."""
        # Rich premium (Forward VRP = +6.0%)
        rich_sizing = self.sizer.calculate_sizing(
            capital=1000000,
            forward_vrp=6.0,
            jump_ratio=0.05
        )
        self.assertEqual(rich_sizing["edge"]["multiplier"], 1.4)
        self.assertGreater(rich_sizing["optimal_lots"], rich_sizing["base_lots"])
        self.assertEqual(rich_sizing["verdict"], "INCREASE_SIZE")

        # Negative VRP (Underpriced premium = -3.5%)
        cheap_sizing = self.sizer.calculate_sizing(
            capital=1000000,
            forward_vrp=-3.5,
            jump_ratio=0.05
        )
        self.assertEqual(cheap_sizing["edge"]["multiplier"], 0.0)
        self.assertEqual(cheap_sizing["optimal_lots"], 0)
        self.assertEqual(cheap_sizing["verdict"], "AVOID_STRANGLES")

    def test_hazard_haircut(self):
        """Test that high jump ratio cuts position sizing down."""
        normal_sizing = self.sizer.calculate_sizing(
            capital=2000000,
            forward_vrp=3.0,
            jump_ratio=0.05
        )
        high_jump_sizing = self.sizer.calculate_sizing(
            capital=2000000,
            forward_vrp=3.0,
            jump_ratio=0.45  # Extreme jump hazard
        )
        self.assertLess(high_jump_sizing["hazard"]["multiplier"], normal_sizing["hazard"]["multiplier"])
        self.assertLess(high_jump_sizing["optimal_lots"], normal_sizing["optimal_lots"])

    def test_asymmetric_leg_skew(self):
        """Test asymmetric leg distribution based on Volatility Asymmetry Index (VAI)."""
        # Upside grind (VAI = -0.30): Less crash risk on downside -> Put heavy skew
        grind_sizing = self.sizer.calculate_sizing(
            capital=2000000,
            forward_vrp=5.0,
            vai=-0.30
        )
        self.assertGreater(grind_sizing["pe_lots"], grind_sizing["ce_lots"])
        self.assertEqual(grind_sizing["skew"]["bias"], "PUT_HEAVY_SKEW")

        # Downside crash risk (VAI = +0.30): High crash hazard -> Call heavy skew
        crash_sizing = self.sizer.calculate_sizing(
            capital=2000000,
            forward_vrp=5.0,
            vai=0.30
        )
        self.assertGreater(crash_sizing["ce_lots"], crash_sizing["pe_lots"])
        self.assertEqual(crash_sizing["skew"]["bias"], "CALL_HEAVY_SKEW")

    def test_stress_test_calculation(self):
        """Test 2-sigma overnight gap risk stress test."""
        sizing = self.sizer.calculate_sizing(
            capital=1000000,
            spot=23000.0,
            atm_iv=12.0
        )
        self.assertIn("stress_test", sizing)
        st = sizing["stress_test"]
        self.assertGreater(st["gap_2sigma_pts"], 0.0)
        self.assertGreater(st["estimated_loss_inr"], 0.0)
        self.assertGreater(st["risk_pct_of_capital"], 0.0)


if __name__ == "__main__":
    unittest.main()
