"""
tests/test_ignition_scanner.py
==============================
Unit tests for calculations/IgnitionScannerEngine.py and PortfolioManager
integration for 0DTE Gamma Ignition Scanner and position tracking.
"""

import unittest
import time
import numpy as np
import pandas as pd
from calculations.IgnitionScannerEngine import (
    IgnitionScannerEngine, _StrikeState, _bsm_greeks
)
from PortfolioManager import PortfolioManager


class TestIgnitionScannerEngine(unittest.TestCase):

    def setUp(self):
        self.engine = IgnitionScannerEngine(lot_size=65)
        self.spot = 24700.0

    def _build_synthetic_chain(self, spot=24700.0, base_premium=12.0):
        """Build a synthetic option chain DataFrame around spot."""
        strikes = np.arange(spot - 300, spot + 350, 50)
        rows = []
        for K in strikes:
            dist = abs(K - spot)
            # Call
            c_prem = max(2.0, base_premium + (spot - K) * 0.4) if K < spot else max(2.0, base_premium - dist * 0.1)
            rows.append({
                'strike': float(K), 'type': 'CE', 'price': float(c_prem),
                'oi': 50000 + int(dist * 10), 'iv': 0.14, 'volume': 15000.0
            })
            # Put
            p_prem = max(2.0, base_premium + (K - spot) * 0.4) if K > spot else max(2.0, base_premium - dist * 0.1)
            rows.append({
                'strike': float(K), 'type': 'PE', 'price': float(p_prem),
                'oi': 45000 + int(dist * 10), 'iv': 0.145, 'volume': 14000.0
            })
        return pd.DataFrame(rows)

    def test_bsm_greeks(self):
        """Verify Black-Scholes-Merton Greeks helper produces sensible values."""
        greeks = _bsm_greeks(S=24700, K=24700, T=1/365, r=0.05, q=0.01, iv=0.14, opt_type='CE')
        self.assertIn('delta', greeks)
        self.assertIn('gamma', greeks)
        self.assertIn('theta', greeks)
        self.assertIn('vega', greeks)
        self.assertIn('iv', greeks)
        self.assertGreater(greeks['delta'], 0.4)
        self.assertLess(greeks['delta'], 0.6)
        self.assertGreater(greeks['gamma'], 0.0)

    def test_strike_state_push_and_range_pct(self):
        """Verify _StrikeState accumulates price history and calculates range_pct correctly."""
        state = _StrikeState(24750.0, 'CE')
        ts = time.time()
        for i in range(15):
            # Flat prices: 10.0 to 10.5 (compressed base)
            p = 10.0 + (i % 3) * 0.2
            state.push(price=p, oi=10000 + i * 100, volume=5000.0 + i * 50, iv=0.14, ts=ts + i * 10)

        self.assertEqual(len(state.prices), 15)
        self.assertEqual(len(state.range_pcts), 13) # Starts calculating after prices >= 3 (15 - 2 = 13)
        # Range should be very small (~ (10.4 - 10.0) / 10.4 ~ 0.038)
        latest_range = state.range_pcts[-1]
        self.assertLess(latest_range, 0.10)

    def test_compression_detection(self):
        """Verify compression check detects tight consolidation below ceiling."""
        engine = IgnitionScannerEngine(lot_size=65)
        state = _StrikeState(24750.0, 'PE')
        ts = time.time()
        # Feed 15 cycles of tight flat pricing (e.g. ₹8.0 - ₹8.4)
        for i in range(15):
            state.push(price=8.0 + (i % 2) * 0.3, oi=20000, volume=4000.0, iv=0.13, ts=ts + i * 10)

        is_comp, range_pct, rank = engine._is_compressed(state)
        self.assertTrue(is_comp)
        self.assertGreater(rank, 0.0)
        self.assertLessEqual(rank, 1.0)

    def test_compression_rejected_if_above_ceiling(self):
        """Verify options above premium ceiling (e.g. ₹40) are rejected by compression filter."""
        engine = IgnitionScannerEngine(lot_size=65)
        state = _StrikeState(24500.0, 'CE')
        ts = time.time()
        # Flat pricing but expensive (₹80.0)
        for i in range(15):
            state.push(price=80.0 + (i % 2) * 0.2, oi=20000, volume=4000.0, iv=0.13, ts=ts + i * 10)

        is_comp, range_pct, rank = engine._is_compressed(state)
        self.assertFalse(is_comp)
        self.assertEqual(rank, 0.0)

    def test_ignition_trigger(self):
        """Verify ignition trigger fires on directional spot velocity + volume surge."""
        engine = IgnitionScannerEngine(lot_size=65)
        state = _StrikeState(24800.0, 'CE')
        ts = time.time()

        # Seed baseline spot history and option history (10 quiet cycles)
        for i in range(10):
            spot = 24700.0 + (i % 2) * 2.0
            engine._spot_history.append((ts + i * 10, spot))
            state.push(price=10.0, oi=10000, volume=1000.0, iv=0.13, ts=ts + i * 10)

        # Now simulate sudden fast spot rally and volume surge
        fast_spot = 24780.0 # +80 pts surge (~0.32%)
        engine._spot_history.append((ts + 110, fast_spot))
        state.push(price=18.0, oi=15000, volume=8000.0, iv=0.15, ts=ts + 110) # 8x volume surge

        is_igniting, ign_rank, details = engine._check_ignition(state, fast_spot)
        self.assertTrue(is_igniting)
        self.assertGreater(ign_rank, 0.5)
        self.assertTrue(details.get('vol_surge'))
        self.assertTrue(details.get('oi_rising'))

    def test_full_scan_pipeline(self):
        """Simulate multi-cycle scan on synthetic chain and verify ranked output."""
        engine = IgnitionScannerEngine(lot_size=65)
        spot = 24700.0

        # Cycle 1-9: normal compression building
        for c in range(9):
            df = self._build_synthetic_chain(spot=spot, base_premium=12.0)
            res = engine.scan(df, spot=spot, dte=0.5)
            self.assertTrue(res['ok'])
            self.assertGreater(res['scan_count'], 0)

        # Cycle 10: spot drops sharply to 24650 (-50 pts), triggering PE explosion
        fast_spot = 24650.0
        df_ign = self._build_synthetic_chain(spot=fast_spot, base_premium=14.0)
        # Pump volume on 24650 PE
        df_ign.loc[(df_ign['strike'] == 24650.0) & (df_ign['type'] == 'PE'), 'volume'] = 50000.0
        df_ign.loc[(df_ign['strike'] == 24650.0) & (df_ign['type'] == 'PE'), 'oi'] = 80000.0

        res = engine.scan(df_ign, spot=fast_spot, dte=0.5)
        self.assertTrue(res['ok'])
        candidates = res.get('candidates', [])
        # If candidates returned, verify structure
        for cand in candidates:
            self.assertIn('strike', cand)
            self.assertIn('type', cand)
            self.assertIn('entry_premium', cand)
            self.assertIn('sl_premium', cand)
            self.assertIn('target_1', cand)
            self.assertIn('target_2', cand)
            self.assertIn('greeks_at_entry', cand)
            self.assertIn('confluence_score', cand)
            self.assertGreaterEqual(cand['confluence_score'], 0)
            self.assertLessEqual(cand['confluence_score'], 100)
            # Stop loss must be below entry for buy
            self.assertLess(cand['sl_premium'], cand['entry_premium'])
            # Target 1 must be above entry
            self.assertGreater(cand['target_1'], cand['entry_premium'])


class TestPortfolioManagerEnhancements(unittest.TestCase):

    def setUp(self):
        import tempfile
        import os
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            self.tmp_file = f.name
        self.mgr = PortfolioManager(filepath=self.tmp_file)

    def tearDown(self):
        import os
        if os.path.exists(self.tmp_file):
            try:
                os.remove(self.tmp_file)
            except Exception:
                pass

    def test_deploy_with_sl_targets_and_greeks(self):
        """Verify deploy records source, SL, targets, and greeks at entry."""
        legs = [{
            'action': 'BUY', 'type': 'PE', 'strike': 25400.0,
            'price': 8.5, 'lots': 2, 'iv': 14.5
        }]
        greeks = {'delta': -0.12, 'gamma': 0.0045, 'theta': -1.2, 'vega': 3.8, 'iv': 14.5}
        pos = self.mgr.deploy(
            strategy_name="Ignition: 25400 PE",
            legs=legs,
            spot=25450.0,
            source="IGNITION_SCANNER",
            sl_premium=5.5,
            target_premiums=[17.0, 22.0],
            greeks_at_entry=greeks
        )

        self.assertEqual(pos['name'], "Ignition: 25400 PE")
        self.assertEqual(pos['source'], "IGNITION_SCANNER")
        self.assertEqual(pos['sl_premium'], 5.5)
        self.assertEqual(pos['target_premiums'], [17.0, 22.0])
        self.assertEqual(pos['greeks_at_entry']['delta'], -0.12)
        self.assertFalse(pos['sl_hit'])
        self.assertFalse(pos['target_1_hit'])

    def test_get_status_sl_hit_and_target_hit(self):
        """Verify get_status flags SL hit when price drops, and target hit when price expands."""
        legs = [{
            'action': 'BUY', 'type': 'PE', 'strike': 25400.0,
            'price': 8.5, 'lots': 1, 'iv': 14.5
        }]
        self.mgr.deploy(
            strategy_name="Ignition: 25400 PE",
            legs=legs,
            spot=25450.0,
            source="IGNITION_SCANNER",
            sl_premium=5.5,
            target_premiums=[17.0, 22.0]
        )

        # Case 1: Normal movement (price = ₹10.0) -> No SL, No T1
        live_chain = {'CE': {}, 'PE': {25400.0: 10.0}}
        st = self.mgr.get_status(live_chain, spot=25420.0, T_now=1/365, atm_iv=0.15)
        p0 = st['active'][0]
        self.assertFalse(p0['sl_hit'])
        self.assertFalse(p0['target_1_hit'])
        self.assertGreater(p0['live_pnl'], 0) # +1.5 * 65 = +97.5

        # Case 2: Price explodes to ₹18.0 -> Target 1 HIT
        live_chain = {'CE': {}, 'PE': {25400.0: 18.0}}
        st = self.mgr.get_status(live_chain, spot=25380.0, T_now=1/365, atm_iv=0.15)
        p0 = st['active'][0]
        self.assertFalse(p0['sl_hit'])
        self.assertTrue(p0['target_1_hit'])
        self.assertFalse(p0['target_2_hit'])

        # Case 3: Price drops to ₹4.5 (below SL ₹5.5) -> SL HIT
        live_chain = {'CE': {}, 'PE': {25400.0: 4.5}}
        st = self.mgr.get_status(live_chain, spot=25500.0, T_now=1/365, atm_iv=0.15)
        p0 = st['active'][0]
        self.assertTrue(p0['sl_hit'])
        self.assertLess(p0['live_pnl'], 0)


if __name__ == '__main__':
    unittest.main()
