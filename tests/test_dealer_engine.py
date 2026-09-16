import unittest
import pandas as pd
from calculations.DealerPositionEngine import DealerPositionEngine

class TestDealerPositionEngine(unittest.TestCase):
    
    def setUp(self):
        self.engine_std = DealerPositionEngine(lot_size=75, positioning_model='standard')
        self.spot = 24500.0
        self.chain = pd.DataFrame({
            'strike': [24400, 24500, 24600],
            'type': ['PE', 'CE', 'CE'],
            'oi': [1000000, 500000, 1500000],
            'iv': [0.12, 0.14, 0.15],
            'dte': [1.0, 1.0, 1.0]
        })

    def test_dealer_inventory_calculation(self):
        res = self.engine_std.calculate_dealer_inventory(self.chain, self.spot)
        
        # Test required top-level keys exist
        expected_keys = [
            'net_delta_exposure', 'net_vega_exposure', 'net_gamma_shares',
            'net_vanna_exposure', 'net_charm_exposure', 'projected_hedging'
        ]
        for key in expected_keys:
            self.assertIn(key, res)
            
        # Standard assumption: CE -> Dealer Short Call -> Negative Gamma -> Destabilizing?
        # Wait, Standard assumption: CE -> Dealer Long Call (+1). 
        # Let's check: PE is -1 (Dealer Short Put)
        # So PE has -1 multiplier.
        # Call has +1 multiplier.
        
        # Let's check GEX mapping matches
        df = res['raw_df']
        pe_row = df[df['type'] == 'PE'].iloc[0]
        ce_row = df[df['type'] == 'CE'].iloc[0]
        
        # Dealer Sign for PE is -1 in standard
        # PE Delta is negative. So PE DEX = (-Delta) * (-1) = Positive!
        # Wait, PE Delta is negative (e.g. -0.4). Dealer sells Put.
        # Dealer position is Short Put. Short Put Delta is Positive (+0.4).
        # So PE DEX should be positive!
        self.assertTrue(pe_row['dex'] > 0, "Short Put Delta should be positive")
        
        # Dealer Sign for CE is +1 in standard (Dealer Long Call)
        # Call Delta is positive (+0.5). Dealer buys Call.
        # Dealer position is Long Call. Long Call Delta is Positive (+0.5).
        self.assertTrue(ce_row['dex'] > 0, "Long Call Delta should be positive")
        
    def test_projected_hedging(self):
        res = self.engine_std.calculate_dealer_inventory(self.chain, self.spot)
        proj = res['projected_hedging']
        
        # If GEX is positive, dealer delta increases when spot goes up.
        # To flatten, dealer must sell shares. 
        # So if net_gamma_shares > 0, buy_shares_if_spot_up_1pct should be < 0
        self.assertEqual(proj['buy_shares_if_spot_up_1pct'], -res['net_gamma_shares'])
        self.assertEqual(proj['buy_shares_if_iv_up_1pct'], -res['net_vanna_exposure'])
        self.assertEqual(proj['buy_shares_if_1_day_passes'], -res['net_charm_exposure'])

    def test_lots_based_metrics(self):
        res = self.engine_std.calculate_dealer_inventory(self.chain, self.spot)
        self.assertIn('net_delta_lots', res)
        self.assertIn('net_gamma_lots_50pt', res)
        self.assertIn('buy_lots_if_spot_up_50pt', res['projected_hedging'])
        self.assertIsInstance(res['net_delta_lots'], float)
        self.assertIsInstance(res['net_gamma_lots_50pt'], float)
        # Hedge demand in lots should exactly oppose net gamma in lots
        self.assertAlmostEqual(
            res['projected_hedging']['buy_lots_if_spot_up_50pt'],
            -res['net_gamma_lots_50pt'],
            places=4
        )
        self.assertIn('dex_lots', res['strike_profile'])
        self.assertIn('gex_lots_50pt', res['strike_profile'])

if __name__ == '__main__':
    unittest.main()
