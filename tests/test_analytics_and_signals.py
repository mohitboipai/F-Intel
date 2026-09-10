import unittest
import numpy as np
import pandas as pd
import config
from OptionAnalytics import OptionAnalytics
from AdvancedVolatilityScanner import AdvancedVolatilityScanner
from RealizedVolEngine import RealizedVolEngine
from VolatilityAnalyzer import VolatilityAnalyzer

class TestAnalyticsAndSignals(unittest.TestCase):
    def setUp(self):
        config.load_profile("default")
        self.analytics = OptionAnalytics()
        self.spot = 24000.0
        self.strike = 24000.0
        self.T = 30.0 / 365.0
        self.r = config.get("risk_free_rate")
        self.q = config.get("dividend_yield")
        self.sigma = 0.15

    def test_merton_dividend_pricing(self):
        # Call price with dividend should be lower than without dividend: F = S*e^{(r-q)T} < S*e^{rT}
        price_no_div = self.analytics.black_scholes(self.spot, self.strike, self.T, self.r, self.sigma, 'CE', q=0.0)
        price_with_div = self.analytics.black_scholes(self.spot, self.strike, self.T, self.r, self.sigma, 'CE', q=self.q)
        self.assertGreater(price_no_div, price_with_div)

        # Put price with dividend should be higher than without dividend
        put_no_div = self.analytics.black_scholes(self.spot, self.strike, self.T, self.r, self.sigma, 'PE', q=0.0)
        put_with_div = self.analytics.black_scholes(self.spot, self.strike, self.T, self.r, self.sigma, 'PE', q=self.q)
        self.assertLess(put_no_div, put_with_div)

        # Merton put-call parity: C - P = S*e^{-qT} - K*e^{-rT}
        lhs = price_with_div - put_with_div
        rhs = self.spot * np.exp(-self.q * self.T) - self.strike * np.exp(-self.r * self.T)
        self.assertAlmostEqual(lhs, rhs, places=4)

    def test_iv_solver_with_dividend(self):
        # Generate market price with known vol
        known_vol = 18.5 # 18.5%
        price = self.analytics.black_scholes(self.spot, self.strike, self.T, self.r, known_vol / 100.0, 'CE', q=self.q)
        solved_iv = self.analytics.implied_volatility(price, self.spot, self.strike, self.T, self.r, 'CE', q=self.q)
        self.assertAlmostEqual(solved_iv, known_vol, places=2)

    def test_ivp_and_ivr(self):
        scanner = AdvancedVolatilityScanner()
        # Synthetic IV series over 252 days from 10 to 30
        iv_series = pd.Series(np.linspace(10, 30, 252))
        
        # IV = 20 is exactly the median / midpoint (126 of 252 points are < 20)
        ivp = scanner.compute_ivp(20.0, iv_series)
        self.assertEqual(ivp, 50.0)
        
        ivr = scanner.compute_ivr(20.0, iv_series)
        # (20 - 10) / (30 - 10) * 100 = 50.0%
        self.assertEqual(ivr, 50.0)

        # Boundary checks
        ivp_low = scanner.compute_ivp(5.0, iv_series)
        self.assertEqual(ivp_low, 0.0)
        ivp_high = scanner.compute_ivp(35.0, iv_series)
        self.assertEqual(ivp_high, 100.0)

    def test_realized_vol_yz_weighting(self):
        engine = RealizedVolEngine()
        # Verify consensus weights: Yang-Zhang at 40%
        w_yz = config.get("rv_weight_yz")
        self.assertEqual(w_yz, 0.40)
        self.assertEqual(config.get("rv_weight_c2c"), 0.20)
        self.assertEqual(config.get("rv_weight_park"), 0.20)
        self.assertEqual(config.get("rv_weight_gk"), 0.20)

    def test_volatility_analyzer_4tier_iv(self):
        va = VolatilityAnalyzer()
        va.spot_price = 24000.0

        # Tier 1: Valid API IV
        iv1 = va._ensure_iv(14.5, 250.0, 24000.0, 30/365, 'CE')
        self.assertEqual(iv1, 14.5)

        # Tier 2: Missing/zero API IV, but valid price -> solves IV
        known_price = va.analytics.black_scholes(24000.0, 24000.0, 30/365, 0.051274, 0.16, 'CE', q=0.0122)
        iv2 = va._ensure_iv(0.0, known_price, 24000.0, 30/365, 'CE')
        self.assertAlmostEqual(iv2, 16.0, places=1)

        # Tier 3: Zero IV and zero price -> smile quadratic fallback
        # ATM (k=0) -> 15% fallback
        iv3_atm = va._ensure_iv(0.0, 0.0, 24000.0, 30/365, 'CE')
        self.assertAlmostEqual(iv3_atm, 15.0, places=1)

        # OTM Put (strike 23000, k = ln(23000/24000) < 0) -> smile skew gives higher IV
        iv3_otm = va._ensure_iv(0.0, 0.0, 23000.0, 30/365, 'PE')
        self.assertGreater(iv3_otm, 15.0)

    def test_dataserver_config_endpoints(self):
        from DataServer import app
        client = app.test_client()

        # GET /api/config
        res = client.get('/api/config')
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertTrue(data.get('ok'))
        self.assertIn('config', data)
        self.assertIn('profiles', data)

        # POST /api/config/profile/0dte
        res_prof = client.post('/api/config/profile/0dte')
        self.assertEqual(res_prof.status_code, 200)
        self.assertEqual(config.active_profile(), '0dte')
        self.assertEqual(config.get('dividend_yield'), 0.0)

        # Revert back to default
        res_def = client.post('/api/config/profile/default')
        self.assertEqual(res_def.status_code, 200)
        self.assertEqual(config.active_profile(), 'default')

if __name__ == "__main__":
    unittest.main()
