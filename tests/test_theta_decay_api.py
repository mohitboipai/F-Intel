import unittest
import json
import urllib.parse
from DataServer import app

class TestThetaDecayAPI(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def _fetch(self, params):
        url = f"/api/theta_decay?{urllib.parse.urlencode(params)}"
        resp = self.client.get(url)
        self.assertEqual(resp.status_code, 200)
        return json.loads(resp.data.decode('utf-8'))

    def test_call_bsm(self):
        data = self._fetch({"opt_type": "CE", "model": "bsm", "range_pct": 5})
        self.assertTrue(data.get("ok"))
        self.assertGreater(data.get("spot", 0), 0)
        self.assertIn("bsm", data["series"])
        self.assertIn("theta", data["series"]["bsm"])
        self.assertGreater(len(data["strikes"]), 0)

    def test_put_bsm(self):
        data = self._fetch({"opt_type": "PE", "model": "bsm", "range_pct": 5})
        self.assertTrue(data.get("ok"))
        self.assertIn("bsm", data["series"])
        self.assertIn("theta", data["series"]["bsm"])

    def test_model_both(self):
        data = self._fetch({"opt_type": "CE", "model": "both", "range_pct": 5})
        self.assertTrue(data.get("ok"))
        self.assertIn("bsm", data["series"])
        self.assertIn("heston", data["series"])
        self.assertEqual(len(data["series"]["bsm"]["theta"]), len(data["strikes"]))
        self.assertEqual(len(data["series"]["heston"]["theta"]), len(data["strikes"]))

    def test_straddle(self):
        data = self._fetch({"opt_type": "STRADDLE", "model": "both", "range_pct": 5})
        self.assertTrue(data.get("ok"))
        self.assertIn("bsm", data["series"])
        self.assertIn("theta", data["series"]["bsm"])
        # Straddle theta should be negative and roughly sum of call & put
        atm_idx = data["strikes"].index(data["atm_strike"])
        atm_theta = data["series"]["bsm"]["theta"][atm_idx][0]
        self.assertLess(atm_theta, 0)

    def test_theta_asymmetry(self):
        data = self._fetch({"opt_type": "STRADDLE", "model": "bsm", "range_pct": 5})
        self.assertTrue(data.get("ok"))
        self.assertIn("theta_asymmetry", data)
        asym = data["theta_asymmetry"]
        self.assertIn("leader", asym)
        self.assertIn(asym["leader"], ["CALLS", "PUTS", "BALANCED"])
        self.assertIn("ce_pct", asym)
        self.assertIn("pe_pct", asym)
        self.assertAlmostEqual(asym["ce_pct"] + asym["pe_pct"], 100.0, places=0)
        self.assertIn("chain_totals", asym)
        self.assertGreater(asym["chain_totals"]["ce_total_inr"], 0)
        self.assertGreater(asym["chain_totals"]["pe_total_inr"], 0)

if __name__ == "__main__":
    unittest.main()
