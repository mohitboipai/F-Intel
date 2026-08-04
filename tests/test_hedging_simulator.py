import unittest
import pandas as pd
from calculations.DealerPositionEngine import DealerPositionEngine
from calculations.DealerHedgingSimulator import DealerHedgingSimulator

class TestDealerHedgingSimulator(unittest.TestCase):
    
    def setUp(self):
        self.pos_engine = DealerPositionEngine(lot_size=75, positioning_model='standard')
        self.sim = DealerHedgingSimulator(self.pos_engine)
        
        self.base_spot = 24500.0
        self.chain = pd.DataFrame({
            'strike': [24500],
            'type': ['CE'],
            'oi': [100000],
            'iv': [0.15],
            'dte': [5.0]  # 5 days to expiry
        })

    def test_zero_shift_scenario(self):
        res = self.sim.simulate_scenario(self.chain, self.base_spot, spot_shift_pct=0.0, iv_shift_abs=0.0)
        
        self.assertEqual(res['inventory_changes']['delta_change'], 0.0)
        self.assertEqual(res['required_hedge_flow']['shares_to_trade'], 0.0)
        self.assertEqual(res['required_hedge_flow']['direction'], "NEUTRAL")

    def test_spot_up_hedging_flow(self):
        # Base state: Dealer is Long Call (Standard model). GEX is positive.
        # If Spot goes UP by 1%, Delta goes UP.
        # Dealer must SELL to hedge.
        res = self.sim.simulate_scenario(self.chain, self.base_spot, spot_shift_pct=1.0)
        
        dex_change = res['inventory_changes']['delta_change']
        self.assertTrue(dex_change > 0, "Dealer Delta should increase when spot goes up on a Long Call")
        
        req_flow = res['required_hedge_flow']['shares_to_trade']
        self.assertTrue(req_flow < 0, "Dealer must sell shares to flatten delta")
        self.assertEqual(res['required_hedge_flow']['direction'], "SELL")
        
    def test_iv_up_hedging_flow(self):
        # Base state: Dealer is Long Call (ATM). 
        # Vanna is typically max OTM/ITM, but ATM vega is max.
        # Let's shift IV up by +5 points (+0.05).
        res = self.sim.simulate_scenario(self.chain, self.base_spot, iv_shift_abs=0.05)
        
        # We just assert it calculates without error and returns structured data
        self.assertIn('shares_to_trade', res['required_hedge_flow'])
        
    def test_charm_time_decay(self):
        # Simulate 1 day passing
        res = self.sim.simulate_scenario(self.chain, self.base_spot, days_passed=1.0)
        
        # Base state: Dealer Long Call ATM.
        # As time passes, ATM delta stays near 0.5 but might shift slightly due to forward drift.
        # OTM delta decays to 0. Let's test with OTM call.
        chain_otm = pd.DataFrame({
            'strike': [25000],
            'type': ['CE'],
            'oi': [100000],
            'iv': [0.15],
            'dte': [5.0]
        })
        res_otm = self.sim.simulate_scenario(chain_otm, self.base_spot, days_passed=1.0)
        dex_change = res_otm['inventory_changes']['delta_change']
        
        # OTM Call delta decreases as time passes. Dealer is Long Call, so Dealer Delta decreases.
        # Dealer must BUY to replace the lost delta.
        self.assertTrue(dex_change < 0, "OTM Dealer Delta should decay (decrease)")
        self.assertTrue(res_otm['required_hedge_flow']['shares_to_trade'] > 0, "Dealer must buy to replace decayed delta")
        
    def test_simulate_matrix(self):
        matrix = self.sim.simulate_matrix(
            self.chain, 
            self.base_spot, 
            spot_shifts=[-1.0, 0.0, 1.0], 
            iv_shifts=[0.0, 0.02]
        )
        
        self.assertEqual(matrix.shape, (3, 2))
        self.assertTrue(0.0 in matrix.index)
        self.assertTrue("IV_+0.020" in matrix.columns)

if __name__ == '__main__':
    unittest.main()
