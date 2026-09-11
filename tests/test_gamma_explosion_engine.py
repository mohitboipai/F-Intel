"""
tests/test_gamma_explosion_engine.py
=============================================================================
Unit tests for calculations/GammaExplosionEngine.py
Tests:
  1. GEX Switch-On & Active Pin Detection
  2. Pin duration and unpinning risk transitions
  3. GEX level retest & Order Flow Absorption detection
  4. Gamma Explosion & Cascade target projection (RR, levels, hedge acceleration)
=============================================================================
"""

import unittest
import time
import pandas as pd
import numpy as np

from calculations.GammaExplosionEngine import GammaExplosionEngine
from calculations.GexEngine import GexEngine


class TestGammaExplosionEngine(unittest.TestCase):

    def setUp(self):
        self.engine = GammaExplosionEngine(lot_size=65)
        self.spot = 23500.0

        # Create a synthetic chain around 23500
        strikes = np.arange(23300, 23750, 50)
        data = []
        for k in strikes:
            # Add CE
            data.append({
                'strike': float(k),
                'type': 'CE',
                'oi': 50000.0 if k == 23600 else 10000.0,
                'volume': 25000.0 if k == 23600 else 5000.0,
                'iv': 0.14,
                'dte': 1.0
            })
            # Add PE
            data.append({
                'strike': float(k),
                'type': 'PE',
                'oi': 60000.0 if k == 23400 else 10000.0,
                'volume': 30000.0 if k == 23400 else 5000.0,
                'iv': 0.15,
                'dte': 1.0
            })
        self.chain_df = pd.DataFrame(data)

    def test_pin_detection_and_switch_on(self):
        # 1. First evaluation - strikes with huge OI should register as active pins
        t0 = 1700000000.0
        pins = self.engine.update_and_detect_pins(self.chain_df, self.spot, now_ts=t0)
        self.assertIsInstance(pins, list)

        # Check if any pin has been identified
        # Either 23600 (Call Wall) or 23400 (Put Wall)
        pin_strikes = [p['strike'] for p in pins]
        self.assertTrue(len(pins) >= 1)
        self.assertTrue(23600.0 in pin_strikes or 23400.0 in pin_strikes)

        # 2. Advance time by 30 minutes
        t1 = t0 + 1800.0  # 30 mins
        pins_t1 = self.engine.update_and_detect_pins(self.chain_df, self.spot, now_ts=t1)
        p30 = next((p for p in pins_t1 if p['strike'] in (23600.0, 23400.0)), None)
        self.assertIsNotNone(p30)
        self.assertAlmostEqual(p30['duration_min'], 30.0, places=1)
        self.assertEqual(p30['unpinning_risk'], 'LOW')

    def test_absorption_at_put_wall(self):
        # Place spot right at Put Wall 23400
        spot_at_wall = 23405.0
        # Candle low dipped to 23395 (breached slightly), close at 23415 (closed back above)
        # Heavy negative delta (aggressive sellers absorbed)
        events = self.engine.detect_gex_absorption(
            chain_df=self.chain_df,
            spot_price=spot_at_wall,
            candle_high=23420.0,
            candle_low=23395.0,
            candle_close=23415.0,
            buyer_vol=1000.0,
            seller_vol=5000.0  # Heavy seller volume absorbed
        )
        self.assertIsInstance(events, list)
        self.assertTrue(len(events) > 0)
        put_event = next((e for e in events if 'PUT' in e['level_name']), None)
        self.assertIsNotNone(put_event)
        self.assertEqual(put_event['absorption_type'], 'SELLERS_ABSORBED')
        self.assertTrue(put_event['absorption_score'] >= 70.0)
        self.assertEqual(put_event['status'], 'CONFIRMED')

    def test_explosion_targets_projection(self):
        # Bullish absorption scenario at 23400
        confirmed_event = {
            'level_name': 'PUT WALL',
            'level_strike': 23400.0,
            'absorption_type': 'SELLERS_ABSORBED',
            'status': 'CONFIRMED',
            'barrier_type': 'SUPPORT'
        }
        proj = self.engine.project_explosion_targets(self.chain_df, 23410.0, confirmed_event=confirmed_event)
        self.assertEqual(proj['direction'], 'BULLISH')
        self.assertGreater(proj['target_1'], 23410.0)
        self.assertGreater(proj['target_2'], proj['target_1'])
        self.assertGreater(proj['risk_reward_ratio'], 1.0)
        self.assertIn('hedge_acceleration', proj)

    def test_full_status_payload(self):
        payload = self.engine.get_full_status_payload(self.chain_df, self.spot)
        self.assertTrue(payload['ok'])
        self.assertIn('active_pins', payload)
        self.assertIn('retest_absorptions', payload)
        self.assertIn('explosion_targets', payload)


if __name__ == '__main__':
    unittest.main()
