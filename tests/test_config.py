import unittest
import json
from pathlib import Path
import config

class TestConfigSystem(unittest.TestCase):
    def setUp(self):
        # Reset to default profile before each test
        config.load_profile("default")

    def test_default_values(self):
        self.assertEqual(config.get("nifty_lot_size"), 65)
        self.assertAlmostEqual(config.get("risk_free_rate"), 0.051274, places=5)
        self.assertAlmostEqual(config.get("dividend_yield"), 0.0122, places=4)
        self.assertEqual(config.get("days_in_year"), 365)
        self.assertEqual(config.get("trading_days_year"), 252)
        self.assertEqual(config.get("dividend_dte_threshold"), 7)
        self.assertEqual(config.get("rv_weight_yz"), 0.40)

    def test_profile_switching(self):
        # Switch to 0dte
        success = config.load_profile("0dte")
        self.assertTrue(success)
        self.assertEqual(config.active_profile(), "0dte")
        self.assertEqual(config.get("dividend_yield"), 0.0)
        self.assertEqual(config.get("dividend_dte_threshold"), 0)
        self.assertEqual(config.get("strike_filter_range"), 0.02)

        # Switch to high_vol
        success = config.load_profile("high_vol")
        self.assertTrue(success)
        self.assertEqual(config.active_profile(), "high_vol")
        self.assertEqual(config.get("iv_fallback_flat"), 0.22)
        self.assertEqual(config.get("iv_hv_premium"), 1.25)
        self.assertEqual(config.get("ver_high_threshold"), 1.25)

        # Switch back to default
        success = config.load_profile("default")
        self.assertTrue(success)
        self.assertEqual(config.active_profile(), "default")
        self.assertAlmostEqual(config.get("dividend_yield"), 0.0122, places=4)

    def test_invalid_profile(self):
        success = config.load_profile("non_existent_profile_xyz")
        self.assertFalse(success)

    def test_list_profiles(self):
        profiles = config.list_profiles()
        self.assertIn("default", profiles)
        self.assertIn("0dte", profiles)
        self.assertIn("high_vol", profiles)
        self.assertIn("conservative", profiles)

    def test_as_dict(self):
        d = config.as_dict()
        self.assertIsInstance(d, dict)
        self.assertIn("nifty_lot_size", d)
        self.assertIn("risk_free_rate", d)
        self.assertNotIn("_active_profile", d)

if __name__ == "__main__":
    unittest.main()
