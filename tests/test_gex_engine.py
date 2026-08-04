import unittest
import numpy as np
import pandas as pd
from calculations.GexEngine import GexEngine

class TestGexEngine(unittest.TestCase):
    
    def setUp(self):
        # Create a standard GexEngine (Lot size 75 for NIFTY)
        self.engine_std = GexEngine(lot_size=75, positioning_model='standard')
        self.engine_inv = GexEngine(lot_size=75, positioning_model='inverted')
        self.engine_flw = GexEngine(lot_size=75, positioning_model='flow')
        
        # Sample Chain
        self.spot = 24500.0
        self.chain = pd.DataFrame({
            'strike': [24400, 24500, 24600],
            'type': ['PE', 'CE', 'CE'],
            'oi': [1000000, 500000, 1500000],
            'iv': [0.12, 0.14, 0.15],
            'dte': [1.0, 1.0, 1.0],
            'volume': [50000, 20000, 80000],
            'initiator': ['buyer', 'seller', 'buyer']
        })

    def test_bsm_gamma_precision(self):
        # Test basic BSM math to ensure vectorization matches scalar
        S = 24500.0
        K = np.array([24500.0])
        T = np.array([1/365.0])
        iv = np.array([0.15])
        
        gamma = self.engine_std.compute_gamma_vectorized(S, K, T, iv)
        self.assertTrue(len(gamma) == 1)
        self.assertTrue(gamma[0] > 0)
        
        # Gamma should be highest ATM
        K_otm = np.array([24600.0])
        gamma_otm = self.engine_std.compute_gamma_vectorized(S, K_otm, T, iv)
        self.assertTrue(gamma[0] > gamma_otm[0])

    def test_standard_positioning(self):
        res = self.engine_std.calculate_gex(self.chain, self.spot)
        df = res['raw_df']
        
        # PE should have negative dealer gamma multiplier
        pe_row = df[df['type'] == 'PE'].iloc[0]
        self.assertTrue(pe_row['gex_oi'] < 0, "Standard Put GEX should be negative")
        
        # CE should have positive dealer gamma multiplier
        ce_row = df[df['type'] == 'CE'].iloc[0]
        self.assertTrue(ce_row['gex_oi'] > 0, "Standard Call GEX should be positive")

    def test_inverted_positioning(self):
        res = self.engine_inv.calculate_gex(self.chain, self.spot)
        df = res['raw_df']
        
        # PE should have positive dealer gamma multiplier
        pe_row = df[df['type'] == 'PE'].iloc[0]
        self.assertTrue(pe_row['gex_oi'] > 0, "Inverted Put GEX should be positive")
        
        # CE should have negative dealer gamma multiplier
        ce_row = df[df['type'] == 'CE'].iloc[0]
        self.assertTrue(ce_row['gex_oi'] < 0, "Inverted Call GEX should be negative")

    def test_flow_positioning(self):
        res = self.engine_flw.calculate_gex(self.chain, self.spot)
        df = res['raw_df']
        
        # 1st Row: Buyer Initiated (Customer Buys -> Dealer Sells -> Short Gamma)
        self.assertTrue(df.iloc[0]['gex_oi'] < 0, "Buyer flow should create Negative GEX")
        
        # 2nd Row: Seller Initiated (Customer Sells -> Dealer Buys -> Long Gamma)
        self.assertTrue(df.iloc[1]['gex_oi'] > 0, "Seller flow should create Positive GEX")

    def test_edge_cases(self):
        # Missing IV and DTE
        df_edge = pd.DataFrame({
            'strike': [24500],
            'type': ['CE'],
            'oi': [1000]
        })
        res = self.engine_std.calculate_gex(df_edge, self.spot)
        self.assertTrue(not res['raw_df']['gamma'].isna().any())
        self.assertTrue(not res['raw_df']['gex_oi'].isna().any())
        
        # Zero DTE
        df_0dte = pd.DataFrame({
            'strike': [24500],
            'type': ['CE'],
            'oi': [1000],
            'dte': [0.0]
        })
        res = self.engine_std.calculate_gex(df_0dte, self.spot)
        self.assertTrue(not res['raw_df']['gamma'].isna().any(), "Zero DTE should not produce NaN")
        self.assertTrue(res['0dte_gex'] > 0, "0DTE GEX should be captured")

if __name__ == '__main__':
    unittest.main()
