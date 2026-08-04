import unittest
import numpy as np
from calculations.GreeksEngine import GreeksEngine

class TestGreeksEngine(unittest.TestCase):
    
    def setUp(self):
        self.engine = GreeksEngine(risk_free_rate=0.07, days_in_year=365.0)

    def test_first_order_greeks(self):
        S = 24500.0
        K = np.array([24500.0, 25000.0]) # ATM, OTM Call
        T_days = np.array([30.0, 30.0])
        iv = np.array([0.15, 0.15])
        option_types = np.array(['CE', 'CE'])
        
        df = self.engine.calculate_all_greeks(S, K, T_days, iv, option_types)
        
        # ATM Delta should be near 0.5 (higher due to forward premium from r=0.07)
        self.assertTrue(0.50 < df.iloc[0]['delta'] < 0.60)
        
        # OTM Delta should be lower than ATM
        self.assertTrue(df.iloc[1]['delta'] < df.iloc[0]['delta'])
        
        # Theta should be negative for buyers
        self.assertTrue(df.iloc[0]['theta'] < 0)
        self.assertTrue(df.iloc[1]['theta'] < 0)
        
        # Vega should be positive
        self.assertTrue(df.iloc[0]['vega'] > 0)
        
    def test_second_order_greeks(self):
        S = 24500.0
        K = np.array([24500.0])
        T_days = np.array([30.0])
        iv = np.array([0.15])
        option_types = np.array(['CE'])
        
        df = self.engine.calculate_all_greeks(S, K, T_days, iv, option_types)
        
        # Gamma should be positive
        self.assertTrue(df.iloc[0]['gamma'] > 0)
        
        # Charm is delta decay. Call Delta decays toward 0 (OTM) or 1 (ITM).
        # Since ATM call is slightly ITM if forward price > K, but let's just check it's populated.
        self.assertTrue(not np.isnan(df.iloc[0]['charm']))
        self.assertTrue(not np.isnan(df.iloc[0]['vanna']))
        self.assertTrue(not np.isnan(df.iloc[0]['vomma']))
        
    def test_put_call_parity_greeks(self):
        S = 24500.0
        K = np.array([24500.0])
        T_days = np.array([30.0])
        iv = np.array([0.15])
        
        ce_df = self.engine.calculate_all_greeks(S, K, T_days, iv, np.array(['CE']))
        pe_df = self.engine.calculate_all_greeks(S, K, T_days, iv, np.array(['PE']))
        
        # Gamma and Vega are identical for calls and puts
        self.assertAlmostEqual(ce_df.iloc[0]['gamma'], pe_df.iloc[0]['gamma'], places=6)
        self.assertAlmostEqual(ce_df.iloc[0]['vega'], pe_df.iloc[0]['vega'], places=6)
        
        # Delta difference should be 1
        self.assertAlmostEqual(ce_df.iloc[0]['delta'] - pe_df.iloc[0]['delta'], 1.0, places=6)

if __name__ == '__main__':
    unittest.main()
