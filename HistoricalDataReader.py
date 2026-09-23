"""
HistoricalDataReader.py — Historical Query & Retrieval Layer for F-Intel
========================================================================
High-performance query interface to retrieve point-in-time snapshots and
time-series ranges of market data, Greeks, GEX, volatility, and signals.

Used by:
- PlaybackEngine: for multi-day historical rewinds
- Strategy engines: for historical regime comparison
- Backtesters: for high-fidelity intraday historical analysis
"""

from __future__ import annotations

import io
import os
import sys
import json
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')  # type: ignore
    except (AttributeError, Exception):
        pass

from FIntelDB import get_db

_log = logging.getLogger("fintel.reader")


class HistoricalDataReader:
    """
    Unified query interface over FIntelDB for historical data and playback.
    """

    def __init__(self):
        self._db = get_db()

    def get_available_dates(self) -> List[str]:
        """Return unique calendar dates (YYYY-MM-DD) that have stored spot or chain data."""
        try:
            query = "SELECT DISTINCT strftime('%Y-%m-%d', ts) as d FROM spot_ticks ORDER BY d DESC"
            rows = self._db.fetchall(query)
            return [str(r[0]) for r in rows if r[0]]
        except Exception as e:
            _log.error(f"Error fetching available dates: {e}")
            return []

    def get_spot_at(self, target_ts: datetime, window_minutes: int = 15) -> Optional[float]:
        """Get the closest spot price at or before target_ts."""
        min_ts = target_ts - timedelta(minutes=window_minutes)
        max_ts = target_ts + timedelta(minutes=window_minutes)
        query = """
            SELECT spot FROM spot_ticks
            WHERE ts BETWEEN ? AND ?
            ORDER BY ABS(strftime('%s', ts) - strftime('%s', ?)) ASC
            LIMIT 1
        """
        # Portable fallback if strftime %s isn't supported in all engines:
        try:
            row = self._db.fetchone(query, (min_ts, max_ts, target_ts))
            if row:
                return float(row[0])
        except Exception:
            # Standard <= fallback
            query_fallback = "SELECT spot FROM spot_ticks WHERE ts <= ? ORDER BY ts DESC LIMIT 1"
            row = self._db.fetchone(query_fallback, (target_ts,))
            if row:
                return float(row[0])
        return None

    def get_spot_series(self, start_ts: datetime, end_ts: datetime) -> List[Dict[str, Any]]:
        """Return spot tick series between start_ts and end_ts."""
        query = "SELECT ts, spot, tick_count FROM spot_ticks WHERE ts BETWEEN ? AND ? ORDER BY ts ASC"
        rows = self._db.fetchall(query, (start_ts, end_ts))
        return [
            {"ts": r[0].isoformat() if hasattr(r[0], "isoformat") else str(r[0]), "spot": float(r[1]), "tick_count": int(r[2] or 1)}
            for r in rows
        ]

    def get_chain_at(self, target_ts: datetime, expiry: Optional[str] = None, window_minutes: int = 15) -> Dict[str, Any]:
        """
        Reconstruct the full option chain at the closest snapshot to target_ts.
        Returns a dict: {'timestamp': ts_str, 'expiry': expiry, 'strikes': {strike: {'CE': {...}, 'PE': {...}}}}
        """
        # 1. Find closest snapshot timestamp
        min_ts = target_ts - timedelta(minutes=window_minutes)
        max_ts = target_ts + timedelta(minutes=window_minutes)
        
        # Try finding snapshot timestamp
        ts_row = self._db.fetchone(
            "SELECT ts FROM chain_snapshots WHERE ts <= ? ORDER BY ts DESC LIMIT 1",
            (target_ts,)
        )
        if not ts_row:
            ts_row = self._db.fetchone(
                "SELECT ts FROM chain_snapshots ORDER BY ts ASC LIMIT 1"
            )
        if not ts_row:
            return {"timestamp": None, "strikes": {}}

        snapshot_ts = ts_row[0]

        sql = """
            SELECT strike, opt_type, expiry, ltp, iv, oi, volume, delta, gamma, theta, vega
            FROM chain_snapshots
            WHERE ts = ?
        """
        params = [snapshot_ts]
        if expiry:
            sql += " AND expiry = ?"
            params.append(expiry)

        rows = self._db.fetchall(sql, tuple(params))
        strikes_dict: Dict[float, Dict[str, Any]] = {}
        found_expiry = expiry

        for r in rows:
            strike = float(r[0])
            opt_type = str(r[1]).upper()
            found_expiry = str(r[2])
            if strike not in strikes_dict:
                strikes_dict[strike] = {}
            strikes_dict[strike][opt_type] = {
                "strike": strike,
                "opt_type": opt_type,
                "expiry": found_expiry,
                "ltp": float(r[3] or 0.0),
                "iv": float(r[4] or 0.0),
                "oi": int(r[5] or 0),
                "volume": int(r[6] or 0),
                "delta": float(r[7] or 0.0),
                "gamma": float(r[8] or 0.0),
                "theta": float(r[9] or 0.0),
                "vega": float(r[10] or 0.0),
            }

        return {
            "timestamp": snapshot_ts.isoformat() if hasattr(snapshot_ts, "isoformat") else str(snapshot_ts),
            "expiry": found_expiry,
            "strikes": strikes_dict,
        }

    def get_gex_at(self, target_ts: datetime) -> Optional[Dict[str, Any]]:
        """Get GEX snapshot at or before target_ts."""
        query = """
            SELECT ts, net_gex, gex_flip_point, regime, atm_concentration, score, direction, oi_surge_bias
            FROM gex_snapshots
            WHERE ts <= ?
            ORDER BY ts DESC
            LIMIT 1
        """
        row = self._db.fetchone(query, (target_ts,))
        if not row:
            return None
        return {
            "ts": row[0].isoformat() if hasattr(row[0], "isoformat") else str(row[0]),
            "net_gex": float(row[1] or 0.0),
            "gex_flip_point": float(row[2] or 0.0),
            "regime": str(row[3] or ""),
            "atm_concentration": float(row[4] or 0.0),
            "score": float(row[5] or 0.0),
            "direction": str(row[6] or ""),
            "oi_surge_bias": float(row[7] or 0.0),
        }

    def get_vol_at(self, target_ts: datetime) -> Optional[Dict[str, Any]]:
        """Get Volatility metrics snapshot at or before target_ts."""
        query = """
            SELECT ts, spot, atm_iv, rv_5d, rv_20d, rv_60d, consensus_rv,
                   vrp, ivp, ivr, ver, vov, regime, har_forecast_1d,
                   har_forecast_5d, jump_ratio, forward_vrp, skew_ratio
            FROM vol_snapshots
            WHERE ts <= ?
            ORDER BY ts DESC
            LIMIT 1
        """
        row = self._db.fetchone(query, (target_ts,))
        if not row:
            return None
        return {
            "ts": row[0].isoformat() if hasattr(row[0], "isoformat") else str(row[0]),
            "spot": float(row[1] or 0.0),
            "atm_iv": float(row[2] or 0.0),
            "rv_5d": float(row[3] or 0.0),
            "rv_20d": float(row[4] or 0.0),
            "rv_60d": float(row[5] or 0.0),
            "consensus_rv": float(row[6] or 0.0),
            "vrp": float(row[7] or 0.0),
            "ivp": float(row[8] or 0.0),
            "ivr": float(row[9] or 0.0),
            "ver": float(row[10] or 0.0),
            "vov": float(row[11] or 0.0),
            "regime": str(row[12] or ""),
            "har_forecast_1d": float(row[13] or 0.0),
            "har_forecast_5d": float(row[14] or 0.0),
            "jump_ratio": float(row[15] or 0.0),
            "forward_vrp": float(row[16] or 0.0),
            "skew_ratio": float(row[17] or 0.0),
        }

    def get_oi_velocity_at(self, target_ts: datetime) -> List[Dict[str, Any]]:
        """Get the latest OI velocity snapshot rows at or before target_ts."""
        ts_row = self._db.fetchone(
            "SELECT ts FROM oi_velocity WHERE ts <= ? ORDER BY ts DESC LIMIT 1",
            (target_ts,)
        )
        if not ts_row:
            return []
        snapshot_ts = ts_row[0]
        query = """
            SELECT strike, opt_type, oi_abs, vel_15m, vel_5m, accel, pct_change, classification
            FROM oi_velocity
            WHERE ts = ?
        """
        rows = self._db.fetchall(query, (snapshot_ts,))
        return [
            {
                "strike": float(r[0]),
                "opt_type": str(r[1]),
                "oi_abs": int(r[2] or 0),
                "vel_15m": float(r[3] or 0.0),
                "vel_5m": float(r[4] or 0.0),
                "accel": float(r[5] or 0.0),
                "pct_change": float(r[6] or 0.0),
                "classification": str(r[7] or ""),
            }
            for r in rows
        ]

    def get_signals_between(self, start_ts: datetime, end_ts: datetime) -> List[Dict[str, Any]]:
        """Get signals generated in the specified window."""
        query = """
            SELECT ts, signal_id, source, direction, action, strike, entry_price,
                   sl_spot, t1_spot, t2_spot, score, status, outcome_pnl, context
            FROM signals
            WHERE ts BETWEEN ? AND ?
            ORDER BY ts ASC
        """
        rows = self._db.fetchall(query, (start_ts, end_ts))
        results = []
        for r in rows:
            ctx = r[13]
            try:
                ctx_parsed = json.loads(ctx) if ctx else {}
            except Exception:
                ctx_parsed = ctx
            results.append({
                "ts": r[0].isoformat() if hasattr(r[0], "isoformat") else str(r[0]),
                "signal_id": str(r[1] or ""),
                "source": str(r[2] or ""),
                "direction": str(r[3] or ""),
                "action": str(r[4] or ""),
                "strike": float(r[5] or 0.0),
                "entry_price": float(r[6] or 0.0),
                "sl_spot": float(r[7] or 0.0),
                "t1_spot": float(r[8] or 0.0),
                "t2_spot": float(r[9] or 0.0),
                "score": float(r[10] or 0.0),
                "status": str(r[11] or "ACTIVE"),
                "outcome_pnl": float(r[12] or 0.0),
                "context": ctx_parsed,
            })
        return results

    def get_alerts_between(self, start_ts: datetime, end_ts: datetime, level: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get system alerts in the specified window."""
        sql = "SELECT ts, source, level, title, body, data FROM alerts WHERE ts BETWEEN ? AND ?"
        params: List[Any] = [start_ts, end_ts]
        if level:
            sql += " AND level = ?"
            params.append(level.upper())
        sql += " ORDER BY ts ASC"

        rows = self._db.fetchall(sql, tuple(params))
        results = []
        for r in rows:
            data = r[5]
            try:
                data_parsed = json.loads(data) if data else {}
            except Exception:
                data_parsed = data
            results.append({
                "ts": r[0].isoformat() if hasattr(r[0], "isoformat") else str(r[0]),
                "source": str(r[1] or ""),
                "level": str(r[2] or ""),
                "title": str(r[3] or ""),
                "body": str(r[4] or ""),
                "data": data_parsed,
            })
        return results

    def get_dashboard_state_at(self, target_ts: datetime) -> Dict[str, Any]:
        """
        Reconstruct the entire Unified Dashboard state as it appeared at target_ts.
        Returns all metrics, chain, GEX, volatility, and recent signals.
        """
        spot = self.get_spot_at(target_ts)
        chain_info = self.get_chain_at(target_ts)
        gex_info = self.get_gex_at(target_ts)
        vol_info = self.get_vol_at(target_ts)
        oi_info = self.get_oi_velocity_at(target_ts)
        
        # Recent signals in the preceding 2 hours
        signals = self.get_signals_between(target_ts - timedelta(hours=2), target_ts)
        alerts = self.get_alerts_between(target_ts - timedelta(hours=1), target_ts)

        return {
            "query_time": target_ts.isoformat(),
            "spot": spot,
            "chain": chain_info,
            "gex": gex_info,
            "volatility": vol_info,
            "oi_velocity": oi_info,
            "signals": signals,
            "alerts": alerts,
        }


# ── Global Singleton Access ──────────────────────────────────────────────────

_reader_instance: Optional[HistoricalDataReader] = None


def get_reader() -> HistoricalDataReader:
    global _reader_instance
    if _reader_instance is None:
        _reader_instance = HistoricalDataReader()
    return _reader_instance


if __name__ == "__main__":
    reader = get_reader()
    print("[OK] HistoricalDataReader initialized")
    now = datetime.now()
    state = reader.get_dashboard_state_at(now)
    print(f"[OK] Reconstructed state at {now.isoformat()}:")
    print(f"  Spot: {state.get('spot')}")
    print(f"  GEX: {state.get('gex')}")
    print(f"  Vol: {state.get('volatility')}")
    print(f"  Alerts count: {len(state.get('alerts', []))}")
    print("[OK] Reader test completed successfully")
