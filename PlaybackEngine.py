"""
PlaybackEngine.py — Multi-Day Dashboard Playback Engine for F-Intel
===================================================================
Enables full historical rewind of the F-Intel dashboard across any recorded day.
Allows traders to scrub through past market sessions second-by-second or minute-by-minute,
observing exactly how GEX profiles, IV surfaces, volatility regimes, and signals evolved.

Features:
- Point-in-time state reconstruction
- Fast range queries for multi-hour charts
- Flask Blueprint (`playback_bp`) for seamless plug-in to DataServer.py
- Timeline index generation for scrub sliders
"""

from __future__ import annotations

import io
import os
import sys
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')  # type: ignore
    except (AttributeError, Exception):
        pass

from flask import Blueprint, jsonify, request
from HistoricalDataReader import get_reader

_log = logging.getLogger("fintel.playback")

playback_bp = Blueprint("playback", __name__, url_prefix="/api/playback")


class PlaybackEngine:
    """
    Core playback coordinator that interfaces with HistoricalDataReader.
    """

    def __init__(self):
        self._reader = get_reader()

    def parse_iso_or_fallback(self, dt_str: Optional[str]) -> datetime:
        """Parse ISO datetime string, or fallback to current time."""
        if not dt_str:
            return datetime.now()
        dt_clean = dt_str.replace("Z", "+00:00")
        try:
            return datetime.fromisoformat(dt_clean)
        except Exception:
            for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
                try:
                    return datetime.strptime(dt_str, fmt)
                except ValueError:
                    continue
        return datetime.now()

    def get_dates(self) -> List[str]:
        return self._reader.get_available_dates()

    def get_state(self, target_time: datetime) -> Dict[str, Any]:
        return self._reader.get_dashboard_state_at(target_time)

    def get_series(self, start_ts: datetime, end_ts: datetime) -> Dict[str, Any]:
        spots = self._reader.get_spot_series(start_ts, end_ts)
        signals = self._reader.get_signals_between(start_ts, end_ts)
        alerts = self._reader.get_alerts_between(start_ts, end_ts)
        return {
            "start": start_ts.isoformat(),
            "end": end_ts.isoformat(),
            "spot_ticks": spots,
            "signals": signals,
            "alerts": alerts,
        }

    def get_timeline_index(self, date_str: str) -> List[str]:
        """
        Return all timestamps (minute-level) that have data on the given date (YYYY-MM-DD).
        Useful for UI scrubbers.
        """
        try:
            start_ts = datetime.strptime(date_str, "%Y-%m-%d")
            end_ts = start_ts + timedelta(days=1)
            spots = self._reader.get_spot_series(start_ts, end_ts)
            # Sample every ~minute
            timestamps = []
            last_minute = None
            for s in spots:
                ts = s["ts"]
                min_key = ts[:16]  # YYYY-MM-DDTHH:MM
                if min_key != last_minute:
                    timestamps.append(ts)
                    last_minute = min_key
            return timestamps
        except Exception as e:
            _log.error(f"Error getting timeline index: {e}")
            return []


# ── Global Singleton & Flask Blueprint Routes ────────────────────────────────

_engine: Optional[PlaybackEngine] = None


def get_playback_engine() -> PlaybackEngine:
    global _engine
    if _engine is None:
        _engine = PlaybackEngine()
    return _engine


@playback_bp.route("/dates", methods=["GET"])
def api_playback_dates():
    """Return list of dates with recorded market data."""
    engine = get_playback_engine()
    dates = engine.get_dates()
    return jsonify({"status": "success", "dates": dates})


@playback_bp.route("/state", methods=["GET"])
def api_playback_state():
    """
    Get full reconstructed dashboard state at a specific timestamp.
    Query param: `ts` (ISO format, e.g. 2026-09-22T10:30:00)
    """
    ts_param = request.args.get("ts")
    engine = get_playback_engine()
    target_dt = engine.parse_iso_or_fallback(ts_param)
    state = engine.get_state(target_dt)
    return jsonify({"status": "success", "data": state})


@playback_bp.route("/range", methods=["GET"])
def api_playback_range():
    """
    Get time-series spot and signals across a time range.
    Query params: `start`, `end`
    """
    start_param = request.args.get("start")
    end_param = request.args.get("end")
    engine = get_playback_engine()
    start_dt = engine.parse_iso_or_fallback(start_param)
    end_dt = engine.parse_iso_or_fallback(end_param) if end_param else start_dt + timedelta(hours=6)
    data = engine.get_series(start_dt, end_dt)
    return jsonify({"status": "success", "data": data})


@playback_bp.route("/timeline", methods=["GET"])
def api_playback_timeline():
    """
    Get all scrubbable timestamps for a given date.
    Query param: `date` (YYYY-MM-DD)
    """
    date_param = request.args.get("date", datetime.now().strftime("%Y-%m-%d"))
    engine = get_playback_engine()
    timeline = engine.get_timeline_index(date_param)
    return jsonify({"status": "success", "date": date_param, "timeline": timeline})


if __name__ == "__main__":
    eng = get_playback_engine()
    print("[OK] PlaybackEngine initialized")
    dates = eng.get_dates()
    print(f"[OK] Available dates: {dates}")
    now_state = eng.get_state(datetime.now())
    print(f"[OK] Retrieved state query_time={now_state.get('query_time')}")
    print("[OK] PlaybackEngine test passed")
