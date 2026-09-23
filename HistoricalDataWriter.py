"""
HistoricalDataWriter.py — Buffered Asynchronous Writer for F-Intel
===================================================================
High-throughput, non-blocking writer that buffers market data, analysis results,
and system signals in memory queues and flushes them to FIntelDB in batches via
a background worker thread.

Key Features:
- Non-blocking: Producers (DataServer loops, WebSocket handlers) simply push to queues
- Micro-batching: Consolidates writes into bulk executemany queries every 5s or at batch threshold
- Rate throttling: Spot ticks throttled to 1/second to prevent database bloat
- Resilient: Catches exceptions on flush so writer threads never crash the main application
- Clean shutdown: Flushes remaining items before exit
"""

from __future__ import annotations

import io
import os
import sys
import time
import json
import logging
import threading
import queue
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')  # type: ignore
    except (AttributeError, Exception):
        pass

from FIntelDB import get_db

_log = logging.getLogger("fintel.writer")

FLUSH_INTERVAL_SEC = 5.0
BATCH_SIZE_THRESHOLD = 500


class HistoricalDataWriter:
    """
    Thread-safe buffered async writer singleton for market and analytics data.
    """

    def __init__(self, flush_interval: float = FLUSH_INTERVAL_SEC):
        self.flush_interval = flush_interval
        self._db = get_db()

        # Dedicated queues for each hypertable
        self._spot_queue: queue.Queue[Tuple] = queue.Queue()
        self._chain_queue: queue.Queue[Tuple] = queue.Queue()
        self._gex_queue: queue.Queue[Tuple] = queue.Queue()
        self._vol_queue: queue.Queue[Tuple] = queue.Queue()
        self._oi_queue: queue.Queue[Tuple] = queue.Queue()
        self._signal_queue: queue.Queue[Tuple] = queue.Queue()
        self._alert_queue: queue.Queue[Tuple] = queue.Queue()
        self._heston_queue: queue.Queue[Tuple] = queue.Queue()

        self._last_spot_write = 0.0
        self._running = True
        self._thread = threading.Thread(target=self._worker_loop, name="FIntelDataWriter", daemon=True)
        self._thread.start()
        _log.info("HistoricalDataWriter background worker started")

    # ── Producer Methods ─────────────────────────────────────────────────────

    def push_spot_tick(self, spot: float, tick_count: int = 1, ts: Optional[datetime] = None) -> None:
        """Throttled to max 1 write per second to prevent database bloat."""
        now = time.time()
        if now - self._last_spot_write < 1.0:
            return
        self._last_spot_write = now
        record_ts = ts or datetime.now()
        self._spot_queue.put((record_ts, float(spot), tick_count))

    def push_chain_snapshot(self, strikes_data: List[Dict[str, Any]], expiry: str, ts: Optional[datetime] = None) -> None:
        """
        strikes_data list with dicts containing:
        strike, opt_type ('CE'/'PE'), ltp, iv, oi, volume, delta, gamma, theta, vega
        """
        record_ts = ts or datetime.now()
        for item in strikes_data:
            row = (
                record_ts,
                expiry,
                float(item.get("strike", 0.0)),
                str(item.get("opt_type", "CE")).upper()[:2],
                float(item.get("ltp") or 0.0),
                float(item.get("iv") or 0.0),
                int(item.get("oi") or 0),
                int(item.get("volume") or 0),
                float(item.get("delta") or 0.0),
                float(item.get("gamma") or 0.0),
                float(item.get("theta") or 0.0),
                float(item.get("vega") or 0.0),
            )
            self._chain_queue.put(row)

    def push_gex_snapshot(self, gex_data: Dict[str, Any], ts: Optional[datetime] = None) -> None:
        record_ts = ts or datetime.now()
        row = (
            record_ts,
            float(gex_data.get("net_gex") or 0.0),
            float(gex_data.get("gex_flip_point") or 0.0),
            str(gex_data.get("regime") or ""),
            float(gex_data.get("atm_concentration") or 0.0),
            float(gex_data.get("score") or 0.0),
            str(gex_data.get("direction") or ""),
            float(gex_data.get("oi_surge_bias") or 0.0),
        )
        self._gex_queue.put(row)

    def push_vol_snapshot(self, vol_data: Dict[str, Any], ts: Optional[datetime] = None) -> None:
        record_ts = ts or datetime.now()
        row = (
            record_ts,
            float(vol_data.get("spot") or 0.0),
            float(vol_data.get("atm_iv") or 0.0),
            float(vol_data.get("rv_5d") or 0.0),
            float(vol_data.get("rv_20d") or 0.0),
            float(vol_data.get("rv_60d") or 0.0),
            float(vol_data.get("consensus_rv") or 0.0),
            float(vol_data.get("vrp") or 0.0),
            float(vol_data.get("ivp") or 0.0),
            float(vol_data.get("ivr") or 0.0),
            float(vol_data.get("ver") or 0.0),
            float(vol_data.get("vov") or 0.0),
            str(vol_data.get("regime") or ""),
            float(vol_data.get("har_forecast_1d") or 0.0),
            float(vol_data.get("har_forecast_5d") or 0.0),
            float(vol_data.get("jump_ratio") or 0.0),
            float(vol_data.get("forward_vrp") or 0.0),
            float(vol_data.get("skew_ratio") or 0.0),
        )
        self._vol_queue.put(row)

    def push_oi_velocity(self, velocity_rows: List[Dict[str, Any]], ts: Optional[datetime] = None) -> None:
        record_ts = ts or datetime.now()
        for item in velocity_rows:
            row = (
                record_ts,
                float(item.get("strike", 0.0)),
                str(item.get("opt_type", "CE")).upper()[:2],
                int(item.get("oi_abs") or 0),
                float(item.get("vel_15m") or 0.0),
                float(item.get("vel_5m") or 0.0),
                float(item.get("accel") or 0.0),
                float(item.get("pct_change") or 0.0),
                str(item.get("classification") or ""),
            )
            self._oi_queue.put(row)

    def push_signal(self, signal_data: Dict[str, Any], ts: Optional[datetime] = None) -> None:
        record_ts = ts or datetime.now()
        context = signal_data.get("context", {})
        context_str = json.dumps(context) if isinstance(context, dict) else str(context or "")
        row = (
            record_ts,
            str(signal_data.get("signal_id") or ""),
            str(signal_data.get("source") or "StrategyEngine"),
            str(signal_data.get("direction") or ""),
            str(signal_data.get("action") or ""),
            float(signal_data.get("strike") or 0.0),
            float(signal_data.get("entry_price") or 0.0),
            float(signal_data.get("sl_spot") or 0.0),
            float(signal_data.get("t1_spot") or 0.0),
            float(signal_data.get("t2_spot") or 0.0),
            float(signal_data.get("score") or 0.0),
            str(signal_data.get("status") or "ACTIVE"),
            float(signal_data.get("outcome_pnl") or 0.0),
            context_str,
        )
        self._signal_queue.put(row)

    def push_alert(self, source: str, level: str, title: str, body: str, data: Any = None, ts: Optional[datetime] = None) -> None:
        record_ts = ts or datetime.now()
        data_str = json.dumps(data) if isinstance(data, (dict, list)) else (str(data) if data is not None else "")
        row = (
            record_ts,
            source,
            level.upper(),
            title,
            body,
            data_str,
        )
        self._alert_queue.put(row)

    def push_heston_params(self, heston_data: Dict[str, Any], ts: Optional[datetime] = None) -> None:
        record_ts = ts or datetime.now()
        row = (
            record_ts,
            float(heston_data.get("v0") or 0.0),
            float(heston_data.get("kappa") or 0.0),
            float(heston_data.get("theta") or 0.0),
            float(heston_data.get("sigma") or 0.0),
            float(heston_data.get("rho") or 0.0),
            float(heston_data.get("fit_error") or 0.0),
        )
        self._heston_queue.put(row)

    # ── Background Worker & Flush ────────────────────────────────────────────

    def _drain_queue(self, q: queue.Queue, max_items: int = 2000) -> List[Tuple]:
        items = []
        while not q.empty() and len(items) < max_items:
            try:
                items.append(q.get_nowait())
            except queue.Empty:
                break
        return items

    def flush(self) -> None:
        """Synchronously drain all queues and write batches to the database."""
        # 1. Spot ticks
        spot_rows = self._drain_queue(self._spot_queue)
        if spot_rows:
            try:
                self._db.executemany(
                    "INSERT INTO spot_ticks (ts, spot, tick_count) VALUES (?, ?, ?)",
                    spot_rows
                )
            except Exception as e:
                _log.error(f"Error flushing spot_ticks ({len(spot_rows)} rows): {e}")

        # 2. Chain snapshots
        chain_rows = self._drain_queue(self._chain_queue)
        if chain_rows:
            try:
                self._db.executemany(
                    """INSERT INTO chain_snapshots (
                        ts, expiry, strike, opt_type, ltp, iv, oi, volume, delta, gamma, theta, vega
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    chain_rows
                )
            except Exception as e:
                _log.error(f"Error flushing chain_snapshots ({len(chain_rows)} rows): {e}")

        # 3. GEX snapshots
        gex_rows = self._drain_queue(self._gex_queue)
        if gex_rows:
            try:
                self._db.executemany(
                    """INSERT INTO gex_snapshots (
                        ts, net_gex, gex_flip_point, regime, atm_concentration, score, direction, oi_surge_bias
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    gex_rows
                )
            except Exception as e:
                _log.error(f"Error flushing gex_snapshots ({len(gex_rows)} rows): {e}")

        # 4. Vol snapshots
        vol_rows = self._drain_queue(self._vol_queue)
        if vol_rows:
            try:
                self._db.executemany(
                    """INSERT INTO vol_snapshots (
                        ts, spot, atm_iv, rv_5d, rv_20d, rv_60d, consensus_rv,
                        vrp, ivp, ivr, ver, vov, regime, har_forecast_1d,
                        har_forecast_5d, jump_ratio, forward_vrp, skew_ratio
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    vol_rows
                )
            except Exception as e:
                _log.error(f"Error flushing vol_snapshots ({len(vol_rows)} rows): {e}")

        # 5. OI Velocity
        oi_rows = self._drain_queue(self._oi_queue)
        if oi_rows:
            try:
                self._db.executemany(
                    """INSERT INTO oi_velocity (
                        ts, strike, opt_type, oi_abs, vel_15m, vel_5m, accel, pct_change, classification
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    oi_rows
                )
            except Exception as e:
                _log.error(f"Error flushing oi_velocity ({len(oi_rows)} rows): {e}")

        # 6. Signals
        sig_rows = self._drain_queue(self._signal_queue)
        if sig_rows:
            try:
                self._db.executemany(
                    """INSERT INTO signals (
                        ts, signal_id, source, direction, action, strike, entry_price,
                        sl_spot, t1_spot, t2_spot, score, status, outcome_pnl, context
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    sig_rows
                )
            except Exception as e:
                _log.error(f"Error flushing signals ({len(sig_rows)} rows): {e}")

        # 7. Alerts
        alert_rows = self._drain_queue(self._alert_queue)
        if alert_rows:
            try:
                self._db.executemany(
                    """INSERT INTO alerts (
                        ts, source, level, title, body, data
                    ) VALUES (?, ?, ?, ?, ?, ?)""",
                    alert_rows
                )
            except Exception as e:
                _log.error(f"Error flushing alerts ({len(alert_rows)} rows): {e}")

        # 8. Heston params
        heston_rows = self._drain_queue(self._heston_queue)
        if heston_rows:
            try:
                self._db.executemany(
                    """INSERT INTO heston_params (
                        ts, v0, kappa, theta, sigma, rho, fit_error
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    heston_rows
                )
            except Exception as e:
                _log.error(f"Error flushing heston_params ({len(heston_rows)} rows): {e}")

    def _worker_loop(self) -> None:
        while self._running:
            try:
                time.sleep(self.flush_interval)
                self.flush()
            except Exception as e:
                _log.error(f"Error in writer worker loop: {e}", exc_info=True)

    def close(self) -> None:
        """Stop worker thread and do final flush."""
        if not self._running:
            return
        self._running = False
        if self._thread.is_alive():
            self._thread.join(timeout=3.0)
        self.flush()
        _log.info("HistoricalDataWriter shut down successfully")


# ── Global Singleton Access ──────────────────────────────────────────────────

_writer_instance: Optional[HistoricalDataWriter] = None
_writer_lock = threading.Lock()


def get_writer() -> HistoricalDataWriter:
    """Return singleton instance of HistoricalDataWriter."""
    global _writer_instance
    with _writer_lock:
        if _writer_instance is None or not _writer_instance._running:
            _writer_instance = HistoricalDataWriter()
        return _writer_instance


if __name__ == "__main__":
    writer = get_writer()
    print("[OK] HistoricalDataWriter initialized")

    # Smoke test writing various structures
    writer.push_spot_tick(23500.5, 1)
    writer.push_gex_snapshot({"net_gex": 1250000.0, "regime": "POSITIVE_GAMMA", "score": 78.5})
    writer.push_vol_snapshot({"spot": 23500.5, "atm_iv": 13.8, "rv_20d": 12.1, "vrp": 1.7, "regime": "FAIR"})
    writer.push_alert("TestEngine", "INFO", "System Initialized", "Historical writer test")

    print("[...] Flushing writes to DB...")
    writer.flush()

    db = get_db()
    for tbl in ["spot_ticks", "gex_snapshots", "vol_snapshots", "alerts"]:
        row = db.fetchone(f"SELECT COUNT(*) FROM {tbl}")
        cnt = row[0] if row else 0
        print(f"  {tbl}: {cnt} rows")

    writer.close()
    print("[OK] Writer test completed successfully")
