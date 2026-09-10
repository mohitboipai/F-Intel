import unittest
import json
import urllib.request
import urllib.parse

class TestThetaDecayAPI(unittest.TestCase):
    BASE_URL = "http://localhost:8082/api/theta_decay"

    def _fetch(self, params):
        url = f"{self.BASE_URL}?{urllib.parse.urlencode(params)}"
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=5) as resp:
            self.assertEqual(resp.status, 200)
            return json.loads(resp.read().decode('utf-8'))

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

if __name__ == "__main__":
    unittest.main()
