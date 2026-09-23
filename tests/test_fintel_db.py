"""
test_fintel_db.py — Comprehensive Integration Test for F-Intel DB & Playback
=============================================================================
Verifies:
1. FIntelDB connection and schema setup
2. HistoricalDataWriter asynchronous push and batch flushing
3. HistoricalDataReader point-in-time and time-series queries
4. PlaybackEngine state reconstruction and timeline queries
5. Flask Playback Blueprint API endpoints
"""

import os
import sys
import unittest
from datetime import datetime, timedelta

# Add parent path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from FIntelDB import get_db
from HistoricalDataWriter import get_writer
from HistoricalDataReader import get_reader
from PlaybackEngine import get_playback_engine, playback_bp
from flask import Flask


class TestFIntelDatabaseAndPlayback(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.db = get_db()
        cls.writer = get_writer()
        cls.reader = get_reader()
        cls.playback = get_playback_engine()

        # Set up test Flask app with blueprint
        cls.app = Flask(__name__)
        cls.app.register_blueprint(playback_bp)
        cls.client = cls.app.test_client()

    def test_01_db_schema_and_version(self):
        """Verify schema tables exist and schema version is recorded."""
        version_row = self.db.fetchone("SELECT version FROM _schema_version")
        self.assertIsNotNone(version_row)
        self.assertGreaterEqual(version_row[0], 1)

        tables = [
            "spot_ticks", "chain_snapshots", "gex_snapshots",
            "vol_snapshots", "oi_velocity", "signals", "alerts", "heston_params"
        ]
        for tbl in tables:
            cnt = self.db.fetchone(f"SELECT COUNT(*) FROM {tbl}")
            self.assertIsNotNone(cnt)

    def test_02_writer_and_reader_round_trip(self):
        """Verify writes are queued and flushed to DB, then queried via reader."""
        test_time = datetime(2026, 9, 22, 10, 30, 0)

        # 1. Push spot
        self.writer.push_spot_tick(23450.0, 1, ts=test_time)

        # 2. Push chain
        chain_sample = [
            {"strike": 23400.0, "opt_type": "CE", "ltp": 120.0, "iv": 14.5, "oi": 50000, "volume": 10000, "delta": 0.55, "gamma": 0.002, "theta": -12.0, "vega": 8.5},
            {"strike": 23400.0, "opt_type": "PE", "ltp": 70.0, "iv": 14.8, "oi": 60000, "volume": 8000, "delta": -0.45, "gamma": 0.002, "theta": -10.0, "vega": 8.2},
            {"strike": 23500.0, "opt_type": "CE", "ltp": 65.0, "iv": 14.2, "oi": 80000, "volume": 15000, "delta": 0.42, "gamma": 0.0025, "theta": -11.0, "vega": 8.0},
        ]
        self.writer.push_chain_snapshot(chain_sample, "2026-09-25", ts=test_time)

        # 3. Push GEX
        gex_sample = {
            "net_gex": 1500000.0,
            "gex_flip_point": 23400.0,
            "regime": "POSITIVE_GAMMA",
            "score": 82.0,
            "direction": "BULLISH",
            "atm_concentration": 0.35,
            "oi_surge_bias": 0.12,
        }
        self.writer.push_gex_snapshot(gex_sample, ts=test_time)

        # 4. Push Volatility
        vol_sample = {
            "spot": 23450.0,
            "atm_iv": 14.2,
            "rv_5d": 12.0,
            "rv_20d": 12.5,
            "rv_60d": 13.0,
            "consensus_rv": 12.5,
            "vrp": 1.7,
            "ivp": 45.0,
            "ivr": 50.0,
            "ver": 1.1,
            "vov": 0.45,
            "regime": "FAIR",
            "har_forecast_1d": 12.4,
            "har_forecast_5d": 12.8,
            "jump_ratio": 0.05,
            "forward_vrp": 1.5,
            "skew_ratio": 1.05,
        }
        self.writer.push_vol_snapshot(vol_sample, ts=test_time)

        # 5. Push Signal
        sig_sample = {
            "signal_id": "TEST-SIG-01",
            "source": "OptionBuyerEngine",
            "direction": "BULLISH",
            "action": "BUY 23500 CE",
            "strike": 23500.0,
            "entry_price": 65.0,
            "sl_spot": 23380.0,
            "t1_spot": 23550.0,
            "t2_spot": 23620.0,
            "score": 85.0,
            "status": "ACTIVE",
            "outcome_pnl": 0.0,
            "context": {"setup": "Gamma Squeeze Trigger"}
        }
        self.writer.push_signal(sig_sample, ts=test_time)

        # 6. Push Alert
        self.writer.push_alert("GEXEngine", "WARNING", "Call Wall Broken", "Spot breached 23500", ts=test_time)

        # Flush queues synchronously
        self.writer.flush()

        # Query back via Reader
        spot = self.reader.get_spot_at(test_time)
        self.assertIsNotNone(spot)
        self.assertAlmostEqual(spot, 23450.0, places=1)

        chain_res = self.reader.get_chain_at(test_time)
        self.assertIn("strikes", chain_res)
        self.assertIn(23400.0, chain_res["strikes"])
        self.assertEqual(chain_res["strikes"][23400.0]["CE"]["ltp"], 120.0)

        gex_res = self.reader.get_gex_at(test_time)
        self.assertIsNotNone(gex_res)
        self.assertEqual(gex_res["regime"], "POSITIVE_GAMMA")
        self.assertAlmostEqual(gex_res["score"], 82.0, places=1)

        vol_res = self.reader.get_vol_at(test_time)
        self.assertIsNotNone(vol_res)
        self.assertEqual(vol_res["regime"], "FAIR")
        self.assertAlmostEqual(vol_res["atm_iv"], 14.2, places=1)

    def test_03_playback_engine_reconstruction(self):
        """Verify PlaybackEngine reconstructs complete dashboard state."""
        test_time = datetime(2026, 9, 22, 10, 30, 0)
        state = self.playback.get_state(test_time)

        self.assertIn("spot", state)
        self.assertIn("chain", state)
        self.assertIn("gex", state)
        self.assertIn("volatility", state)
        self.assertIn("signals", state)
        self.assertIn("alerts", state)
        self.assertGreaterEqual(len(state["signals"]), 1)
        self.assertGreaterEqual(len(state["alerts"]), 1)

    def test_04_playback_api_endpoints(self):
        """Verify Flask blueprint /api/playback routes return HTTP 200 with valid JSON."""
        # 1. Dates
        resp = self.client.get("/api/playback/dates")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertEqual(data["status"], "success")
        self.assertIsInstance(data["dates"], list)

        # 2. State
        resp = self.client.get("/api/playback/state?ts=2026-09-22T10:30:00")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertEqual(data["status"], "success")
        self.assertIn("spot", data["data"])
        self.assertIn("gex", data["data"])

        # 3. Range
        resp = self.client.get("/api/playback/range?start=2026-09-22T10:00:00&end=2026-09-22T11:00:00")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertEqual(data["status"], "success")
        self.assertIn("spot_ticks", data["data"])

        # 4. Timeline
        resp = self.client.get("/api/playback/timeline?date=2026-09-22")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertEqual(data["status"], "success")
        self.assertIn("timeline", data)


if __name__ == "__main__":
    unittest.main()
