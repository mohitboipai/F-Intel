import unittest
import numpy as np
import pandas as pd

from KeyLevelsEngine import KeyLevelsEngine
from NiftyPredictor import NiftyRangePredictor
from NiftyHestonMC import NiftyHestonMC
from InteractiveOptionsAnalyzer import InteractiveOptionsAnalyzer
from calculations.GreeksEngine import GreeksEngine
from calculations.GexEngine import GexEngine
from calculations.DealerPositionEngine import DealerPositionEngine
from calculations.DealerHedgingSimulator import DealerHedgingSimulator


class TestAnalysisEnginesAudit(unittest.TestCase):
    """
    Comprehensive regression tests for analysis engines audit fixes:
    1. Option type parsing ('CE'/'PE' vs 'CALL'/'PUT')
    2. PCR, Call Wall, and Max Pain calculations
    3. 0DTE Charm numerical stability
    4. OI unit scaling without 100k threshold hazard
    5. Dealer hedging simulator IV scaling
    """

    def setUp(self):
        # Mock Fyers-style option chain data
        self.mock_chain_data = {
            'optionsChain': [
                {'strike_price': 24400, 'option_type': 'PE', 'ltp': 80.0, 'oi': 45000, 'iv': 14.0, 'dte': 1.0},
                {'strike_price': 24500, 'option_type': 'PE', 'ltp': 120.0, 'oi': 85000, 'iv': 14.5, 'dte': 1.0},
                {'strike_price': 24500, 'option_type': 'CE', 'ltp': 130.0, 'oi': 70000, 'iv': 14.2, 'dte': 1.0},
                {'strike_price': 24600, 'option_type': 'CE', 'ltp': 75.0, 'oi': 90000, 'iv': 13.8, 'dte': 1.0},
                {'strike_price': 24700, 'option_type': 'CE', 'ltp': 40.0, 'oi': 60000, 'iv': 13.5, 'dte': 1.0},
            ]
        }
        self.spot = 24500.0

    def test_key_levels_engine_option_type_and_pcr(self):
        engine = KeyLevelsEngine()
        df = engine.parse_chain(self.mock_chain_data)
        
        self.assertFalse(df.empty, "DataFrame should not be empty")
        ce_count = len(df[df['type'] == 'CE'])
        pe_count = len(df[df['type'] == 'PE'])
        self.assertEqual(ce_count, 3, "There should be 3 CE rows")
        self.assertEqual(pe_count, 2, "There should be 2 PE rows")

        # Test PCR: total put OI (45k + 85k = 130k) / total call OI (70k + 90k + 60k = 220k)
        pcr = engine.calculate_pcr(df)
        expected_pcr = (45000 + 85000) / (70000 + 90000 + 60000)
        self.assertAlmostEqual(pcr, expected_pcr, places=3)
        self.assertGreater(pcr, 0.0, "PCR should not be 0.0")

        # Test Call Wall above spot: should be 24600 (OI 90,000)
        # Test Call Wall above spot (24600) and Put Wall below spot (24400)
        walls = engine.calculate_oi_walls(df, self.spot)
        self.assertEqual(walls['call_wall'], 24600)
        self.assertEqual(walls['put_wall'], 24400)

        # Test Max Pain
        max_pain = engine.calculate_max_pain(df)
        self.assertIn(max_pain, [24400, 24500, 24600, 24700])

    def test_nifty_predictor_option_type_and_gex(self):
        predictor = NiftyRangePredictor()
        # Mock get_option_chain_data to return mock data
        mock_data_with_expiry = {
            'optionsChain': self.mock_chain_data['optionsChain'],
            'expiryData': [{'expiry': 1766500000, 'date': '24-Dec-2026'}]
        }
        predictor.get_option_chain_data = lambda: mock_data_with_expiry
        res = predictor.fetch_gex_profile(self.spot)
        
        self.assertIsNotNone(res)
        self.assertIn('call_wall', res)
        self.assertIn('put_wall', res)
        # Call wall should be 24600 (OI 90,000)
        self.assertEqual(res['call_wall'], 24600)
        self.assertEqual(res['put_wall'], 24500)
        # Net gamma should be finite number
        self.assertTrue(np.isfinite(res['net_gamma']))

    def test_nifty_heston_option_type_parsing(self):
        heston = NiftyHestonMC()
        df = heston.parse_chain(self.mock_chain_data)
        self.assertFalse(df.empty)
        ce_rows = df[df['type'] == 'CE']
        self.assertEqual(len(ce_rows), 3)

    def test_interactive_options_analyzer_type_parsing(self):
        analyzer = InteractiveOptionsAnalyzer()
        chain_with_symbols = {
            'optionsChain': [
                {'symbol': 'NIFTY24DEC24500CE', 'strike_price': 24500, 'option_type': 'CE', 'ltp': 120.0, 'expiry_date': '24-Dec-2026'},
                {'symbol': 'NIFTY24DEC24500PE', 'strike_price': 24500, 'option_type': 'PE', 'ltp': 110.0, 'expiry_date': '24-Dec-2026'},
                {'symbol': 'NIFTY24DEC24600CE', 'strike_price': 24600, 'option_type': 'CALL', 'ltp': 70.0, 'expiry_date': '24-Dec-2026'},
                {'symbol': 'NIFTY24DEC24400PE', 'strike_price': 24400, 'option_type': 'PUT', 'ltp': 65.0, 'expiry_date': '24-Dec-2026'}
            ]
        }
        df = analyzer.parse_and_filter(chain_with_symbols)
        self.assertEqual(len(df), 4, "All 4 rows (CE, PE, CALL, PUT) should be parsed")
        self.assertEqual(len(df[df['type'] == 'CE']), 2)
        self.assertEqual(len(df[df['type'] == 'PE']), 2)

    def test_greeks_engine_0dte_charm_stability(self):
        engine = GreeksEngine()
        S = 24500.0
        K = np.array([24450.0, 24500.0, 24550.0])
        # Very short DTE: 10 minutes before expiry (~0.007 days)
        T_days = np.array([0.007, 0.007, 0.007])
        iv = np.array([0.15, 0.15, 0.15])
        option_types = np.array(['CE', 'CE', 'PE'])
        
        greeks = engine.calculate_all_greeks(S, K, T_days, iv, option_types)
        charm_vals = greeks['charm'].values
        
        # Verify Charm does not explode to infinity or > 1.0
        self.assertTrue(np.all(np.isfinite(charm_vals)), "Charm should be finite")
        self.assertTrue(np.all(np.abs(charm_vals) <= 1.0), f"Charm should be bounded within [-1.0, 1.0], got {charm_vals}")

    def test_gex_engine_oi_scaling_integrity(self):
        # Chain where max OI is under 100,000 (e.g. 85,000)
        # Should NOT multiply by lot_size if oi_is_shares=True
        chain_small_oi = pd.DataFrame({
            'strike': [24400, 24500, 24600],
            'type': ['PE', 'CE', 'CE'],
            'oi': [45000, 85000, 60000],
            'iv': [0.14, 0.14, 0.14],
            'dte': [1.0, 1.0, 1.0],
            'volume': [1000, 2000, 1500]
        })
        
        engine_std = GexEngine(lot_size=65, positioning_model='standard', oi_is_shares=True)
        res = engine_std.calculate_gex(chain_small_oi, self.spot)
        raw_df = res['raw_df']
        
        # Verify GEX calculation completed normally
        self.assertTrue(np.isfinite(res['net_gex']))
        self.assertEqual(res['call_wall'], 24500)
        self.assertEqual(res['put_wall'], 24400)

    def test_dealer_hedging_simulator_percentage_iv(self):
        pos_engine = DealerPositionEngine(lot_size=65, positioning_model='standard')
        sim = DealerHedgingSimulator(pos_engine)
        
        # Test with percentage IV (e.g. 15.0)
        chain_pct_iv = pd.DataFrame({
            'strike': [24500],
            'type': ['CE'],
            'oi': [100000],
            'iv': [15.0],
            'dte': [5.0]
        })
        
        # Shift IV by 2% (0.02 decimal)
        res = sim.simulate_scenario(chain_pct_iv, self.spot, iv_shift_abs=0.02)
        self.assertIn('required_hedge_flow', res)
        self.assertIn('inventory_changes', res)


if __name__ == '__main__':
    unittest.main()
