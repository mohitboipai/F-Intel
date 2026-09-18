"""
tests/test_gex_rebalance_engine.py
==================================
Unit tests for calculations/GexRebalanceEngine.py.
Validates:
1. Wall inversion detection (Wall 2 closer than Wall 1).
2. Tri-factor ignition (price break + negative OI velocity).
3. Rebalance target & fuel apex calculation.
4. Dual-strike selector (Primary ATM vs 0DTE/1DTE OTM Gamma Rocket).
5. Direct premium ₹ conversion logic.
"""

import unittest
import numpy as np
import pandas as pd
from calculations.GexRebalanceEngine import GexRebalanceEngine


class TestGexRebalanceEngine(unittest.TestCase):

    def setUp(self):
        self.engine = GexRebalanceEngine(lot_size=65)
        self.spot = 24720.0

        # Construct synthetic chain:
        # Spot = 24720
        # Call Wall 2 = 24750 (8.5M OI, closer to spot, 30 pts away)
        # Call Wall 1 = 24900 (15.0M OI, farther from spot, 180 pts away)
        # Put Wall 1 = 24600 (12.0M OI)
        # Put Wall 2 = 24500 (6.0M OI)
        rows = []
        strikes = np.arange(24500, 25050, 50)
        for K in strikes:
            # Calls
            oi_ce = 1_000_000
            price_ce = max(1.0, 24720.0 - K + 80.0) if K <= 24750 else max(1.0, 95.0 - (K - 24750) * 0.5)
            if K == 24900:
                oi_ce = 15_000_000 # Wall 1
            elif K == 24750:
                oi_ce = 8_500_000  # Wall 2 (closer)
            elif K == 24800:
                price_ce = 16.0    # OTM candidate

            rows.append({
                'strike': float(K), 'type': 'CE', 'oi': float(oi_ce),
                'price': float(price_ce), 'iv': 0.14, 'dte': 1.0,
                'delta': 0.50 if K == 24750 else (0.20 if K == 24800 else 0.08),
                'gamma': 0.003
            })

            # Puts
            oi_pe = 1_000_000
            price_pe = max(1.0, K - 24720.0 + 80.0) if K >= 24700 else max(1.0, 95.0 - (24700 - K) * 0.5)
            if K == 24600:
                oi_pe = 12_000_000 # Put Wall 1
            elif K == 24500:
                oi_pe = 6_000_000  # Put Wall 2

            rows.append({
                'strike': float(K), 'type': 'PE', 'oi': float(oi_pe),
                'price': float(price_pe), 'iv': 0.15, 'dte': 1.0,
                'delta': -0.50 if K == 24700 else -0.20,
                'gamma': 0.003
            })

        self.df_chain = pd.DataFrame(rows)

    def test_wall_inversion_detected(self):
        walls = self.engine.detect_walls(self.df_chain, self.spot)
        self.assertEqual(walls['call_wall_1'], 24900.0)
        self.assertEqual(walls['call_wall_2'], 24750.0)
        self.assertTrue(walls['ce_inverted'])
        self.assertEqual(walls['ce_runway'], 150.0)

    def test_coiling_status_before_break(self):
        res = self.engine.evaluate(self.df_chain, self.spot, dte=1.0)
        self.assertTrue(res['ok'])
        self.assertEqual(res['direction'], "BULLISH_CE")
        self.assertEqual(res['status'], "COILING")
        self.assertEqual(res['trigger_strike'], 24750.0)
        self.assertEqual(res['terminal_fortress'], 24900.0)

    def test_ignition_when_wall_2_breaks_with_unwind(self):
        # Spot breaks 24750 to 24753, with negative OI velocity on 24750 CE
        vel_data = {
            'vel_by_strike': {
                (24750.0, 'CE'): -125_000,
                (24700.0, 'PE'): +60_000  # Put adding support
            }
        }
        res = self.engine.evaluate(self.df_chain, 24753.0, oi_velocity_data=vel_data, dte=1.0)
        self.assertEqual(res['status'], "IGNITED")
        self.assertTrue(res['dual_wall_confirmed'])
        self.assertEqual(res['velocity_speed'], "VIOLENT_CAPITULATION")
        self.assertGreater(res['rebalance_target'], 24753.0)

    def test_dual_strikes_generated(self):
        res = self.engine.evaluate(self.df_chain, 24753.0, dte=1.0)
        primary = res['primary_option']
        otm_rocket = res['otm_gamma_rocket']

        # Primary ATM strike
        self.assertIsNotNone(primary)
        self.assertEqual(primary['type'], 'CE')
        self.assertGreater(primary['target_1'], primary['current_price'])
        self.assertLess(primary['stop_loss'], primary['current_price'])

        # OTM Gamma Rocket active on 1DTE
        self.assertIsNotNone(otm_rocket)
        self.assertTrue(otm_rocket['is_active'])
        self.assertGreater(otm_rocket['target_1'], otm_rocket['current_price'])

    def test_target_reached_state(self):
        # 1. Ignite squeeze at 24753 with unwinding OI
        vel_data = {'vel_by_strike': {(24750.0, 'CE'): -125_000}}
        res_initial = self.engine.evaluate(self.df_chain, 24753.0, oi_velocity_data=vel_data, dte=1.0)
        self.assertEqual(res_initial['status'], "IGNITED")
        target = res_initial['rebalance_target']

        # 2. Spot reaches target level
        res_hit = self.engine.evaluate(self.df_chain, target + 2.0, dte=1.0)
        self.assertEqual(res_hit['status'], "TARGET_REACHED")
        self.assertIn("PROFIT", res_hit['action_summary'].upper())

    def test_normal_market_regime_fully_populated(self):
        """Validates that even in a normal, non-inverted market, all radar levels and options are populated."""
        # Construct non-inverted chain: Wall 1 is closer than Wall 2
        rows = []
        for K in [23000, 23100, 23200, 23300, 23400, 23500]:
            rows.append({
                'strike': float(K), 'type': 'CE',
                'oi': 15_000_000.0 if K == 23300 else (10_000_000.0 if K == 23500 else 2_000_000.0),
                'price': max(1.0, 23200.0 - K + 80.0), 'iv': 0.14, 'dte': 1.0
            })
            rows.append({
                'strike': float(K), 'type': 'PE',
                'oi': 14_000_000.0 if K == 23200 else (8_000_000.0 if K == 23000 else 2_000_000.0),
                'price': max(1.0, K - 23200.0 + 80.0), 'iv': 0.14, 'dte': 1.0
            })
        normal_df = pd.DataFrame(rows)
        spot = 23210.0

        res = self.engine.evaluate(normal_df, spot, dte=1.0)
        self.assertTrue(res['ok'])
        self.assertIn(res['status'], ["MONITORING", "COILING"])
        self.assertGreater(res['trigger_strike'], 0)
        self.assertGreater(res['rebalance_target'], 0)
        self.assertGreater(res['terminal_fortress'], 0)
        self.assertGreater(res['runway_pts'], 0)
        self.assertIsNotNone(res['primary_option'])
        self.assertIsNotNone(res['otm_gamma_rocket'])
        self.assertGreater(res['primary_option']['target_1'], res['primary_option']['current_price'])

    def test_gex_res_integration(self):
        """Validates that gex_res anchors zero-gamma level and dealer fuel lots."""
        gex_res = {
            'call_wall': 23300.0,
            'put_wall': 23200.0,
            'zero_gamma_level': 23250.0,
            'net_gex_lots_50pt': 42000.0,
            'net_gex': 1.5e11
        }
        res = self.engine.evaluate(self.df_chain, self.spot, dte=1.0, gex_res=gex_res)
        self.assertTrue(res['ok'])
        self.assertGreater(res['dealer_fuel_lots'], 0)
        self.assertIsNotNone(res['primary_option'])

    def test_chop_filter_stands_aside(self):
        """Validates that CHOPPY market swing quality suppresses trade ideas to protect capital."""
        intraday_data = {
            'swing_quality': {'quality': 'CHOPPY', 'adr_pct': 22.0, 'session_phase': 'MID_TREND'},
            'absorption': {'setup_quality': 'NONE'}
        }
        res = self.engine.evaluate(self.df_chain, self.spot, intraday_signal_data=intraday_data)
        self.assertTrue(res['ok'])
        self.assertFalse(res['trade_ready'])
        self.assertEqual(res['status'], "STAND_ASIDE")
        self.assertIn("STAND ASIDE", res['action_summary'])
        self.assertTrue(any("Chop" in r for r in res['rejection_reasons']))

    def test_opposing_writer_trap_veto(self):
        """Validates that when opposing writers are actively fortifying (+40k/m addition), buy trades are vetoed."""
        vel_data = {
            'vel_by_strike': {(24750.0, 'CE'): +45_000}  # Call writers adding contracts
        }
        res = self.engine.evaluate(self.df_chain, 24753.0, oi_velocity_data=vel_data, dte=1.0)
        self.assertFalse(res['trade_ready'])
        self.assertEqual(res['status'], "STAND_ASIDE")
        self.assertIn("defend", res['action_summary'].lower())

    def test_tier_1_roi_guarantee(self):
        """Validates that Tier 1 momentum setups deliver at least 20% ROI on Target 1 with tight SL."""
        vel_data = {
            'vel_by_strike': {(24750.0, 'CE'): -95_000}
        }
        intraday_data = {
            'swing_quality': {'quality': 'TRENDING', 'adr_pct': 68.0, 'session_phase': 'MID_TREND'},
            'absorption': {'setup_quality': 'STRONG', 'wick_bars': 3, 'vol_accel': 1.8}
        }
        gex_res = {'net_gex': -2.5e10, 'zero_gamma_level': 24700.0}
        res = self.engine.evaluate(
            self.df_chain, 24755.0,
            oi_velocity_data=vel_data,
            dte=2.0,
            gex_res=gex_res,
            intraday_signal_data=intraday_data
        )
        self.assertTrue(res['trade_ready'])
        self.assertGreaterEqual(res['confluence_score'], 65.0)
        primary = res['primary_option']
        self.assertGreaterEqual(primary['target_gain_pct'], 20.0, "Target 1 ROI must be >= 20%")
        self.assertGreaterEqual(primary['runner_gain_pct'], 35.0, "Runner Target ROI must be >= 35%")
        self.assertLessEqual(abs(primary['stop_loss_pct']), 16.0, "Stop loss must be tight (<= 16%)")
        self.assertGreaterEqual(primary['rr_ratio'], 1.3, "R:R ratio must be >= 1.3")

    def test_tier_3_0dte_expiry_mega_move(self):
        """Validates that 0DTE/1DTE expiry with high confluence activates Tier 3 Expiry Mega Move (100%+ ROI)."""
        vel_data = {
            'vel_by_strike': {(24750.0, 'CE'): -120_000}
        }
        intraday_data = {
            'swing_quality': {'quality': 'TRENDING', 'adr_pct': 85.0, 'session_phase': 'EXPIRY_HEAT'},
            'absorption': {'setup_quality': 'STRONG', 'wick_bars': 4, 'vol_accel': 2.4}
        }
        gamma_exp_data = {
            'active_pins': [{'strike': 24750.0, 'duration_secs': 7200, 'unpinning_risk': 'IMMINENT'}]
        }
        gex_res = {'net_gex': -5e10, 'zero_gamma_level': 24720.0}
        res = self.engine.evaluate(
            self.df_chain, 24754.0,
            oi_velocity_data=vel_data,
            dte=0.5,  # 0DTE
            gex_res=gex_res,
            intraday_signal_data=intraday_data,
            gamma_explosion_data=gamma_exp_data
        )
        self.assertTrue(res['trade_ready'])
        self.assertEqual(res['active_tier'], "TIER_3_EXPIRY_MEGA_MOVE")
        self.assertIn("0DTE EXPIRY MEGA MOVE", res['tier_name'])
        otm_rocket = res['otm_gamma_rocket']
        self.assertTrue(otm_rocket['is_active'])
        self.assertGreaterEqual(otm_rocket['target_gain_pct'], 100.0, "0DTE Rocket Target 1 must be >= 100% (2x)")
        self.assertGreaterEqual(otm_rocket['runner_gain_pct'], 200.0, "0DTE Rocket Runner must be >= 200%")


if __name__ == '__main__':
    unittest.main()

