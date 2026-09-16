import unittest
import json
from DataServer import app


class TestStrangleSizingAPI(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def test_econometric_vol_endpoint(self):
        """Test GET /api/volatility/econometric returns complete mathematical payload."""
        resp = self.client.get("/api/volatility/econometric")
        self.assertEqual(resp.status_code, 200)
        data = json.loads(resp.data.decode("utf-8"))

        self.assertTrue(data.get("ok"))
        self.assertIn("data", data)
        payload = data["data"]

        # Check Semi-Variance
        self.assertIn("semi_variance", payload)
        sv = payload["semi_variance"]
        self.assertIn("rv_plus", sv)
        self.assertIn("rv_minus", sv)
        self.assertIn("vai", sv)
        self.assertIn("bias", sv)

        # Check HAR Forecast
        self.assertIn("har_forecast", payload)
        har = payload["har_forecast"]
        self.assertIn("forecast_1d", har)
        self.assertIn("forecast_5d", har)
        self.assertIn("weights", har)

        # Check Jump Decomposition
        self.assertIn("jump_decomposition", payload)
        jump = payload["jump_decomposition"]
        self.assertIn("jump_ratio_pct", jump)
        self.assertIn("action_badge", jump)

        # Check Forward VRP
        self.assertIn("forward_vrp", payload)
        fvrp = payload["forward_vrp"]
        self.assertIn("vrp_5d", fvrp)
        self.assertIn("verdict", fvrp)

    def test_strangle_sizing_endpoint(self):
        """Test GET /api/strangle/sizing with various capital levels."""
        # 1. Standard 10L capital
        resp = self.client.get("/api/strangle/sizing?capital=1000000")
        self.assertEqual(resp.status_code, 200)
        res = json.loads(resp.data.decode("utf-8"))
        self.assertTrue(res.get("ok"))
        self.assertIn("sizing", res)

        sizing = res["sizing"]
        self.assertEqual(sizing["capital"], 1000000.0)
        self.assertIn("optimal_lots", sizing)
        self.assertIn("base_lots", sizing)
        self.assertIn("ce_lots", sizing)
        self.assertIn("pe_lots", sizing)
        self.assertIn("verdict", sizing)
        self.assertIn("verdict_badge", sizing)
        self.assertIn("edge", sizing)
        self.assertIn("hazard", sizing)
        self.assertIn("skew", sizing)
        self.assertIn("stress_test", sizing)

        # Total lots should equal ce_lots + pe_lots
        self.assertEqual(sizing["optimal_lots"], sizing["ce_lots"] + sizing["pe_lots"])

    def test_strangle_sizing_capital_scaling(self):
        """Test capital parameter scaling."""
        resp_5l = self.client.get("/api/strangle/sizing?capital=500000")
        resp_20l = self.client.get("/api/strangle/sizing?capital=2000000")

        data_5l = json.loads(resp_5l.data.decode("utf-8"))["sizing"]
        data_20l = json.loads(resp_20l.data.decode("utf-8"))["sizing"]

        self.assertLess(data_5l["base_lots"], data_20l["base_lots"])
        self.assertLess(data_5l["optimal_lots"], data_20l["optimal_lots"])


if __name__ == "__main__":
    unittest.main()
