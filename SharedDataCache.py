"""
SharedDataCache.py — Shared API Data Cache for All Modules
===========================================================
Provides a single authenticated Fyers session and TTL-based
caching of frequently fetched data so all modules share the
same data fetch (no repeated identical API calls).

Usage:
    from SharedDataCache import SharedDataCache
    cache = SharedDataCache(fyers_instance)

    spot  = cache.get_spot()           # cached 15s
    df    = cache.get_chain(expiry)    # cached 30s
    rv    = cache.get_rv_data()        # cached session-long
"""

import time
import collections
import numpy as np
import pandas as pd
from datetime import datetime


class SharedDataCache:
    """
    TTL-caching layer around Fyers API calls.
    All modules pass their fyers instance on construction;
    spot/chain are re-fetched only when the TTL expires.
    """

    SPOT_TTL  = 15    # seconds
    CHAIN_TTL = 30    # seconds
    LOT_SIZE  = 75    # NIFTY

    def __init__(self, fyers, symbol="NSE:NIFTY50-INDEX"):
        self.fyers  = fyers
        self.symbol = symbol

        # ── Spot cache ──────────────────────────────────────────────
        self._spot      = 0.0
        self._spot_ts   = 0.0

        # ── Chain cache {expiry: (df, fetched_ts)} ──────────────────
        self._chains    = {}

        # ── OHLC + RV cache (fetched once per session) ──────────────
        self._ohlc_df   = None
        self._rv_data   = None

        # ── Heston params cache (5-min TTL) ─────────────────────────
        self._heston_params: dict | None = None
        self._heston_ts: float = 0.0
        self.HESTON_TTL = 300   # 5 minutes

        # ── Raw chain blob from DataServer (dict, not parsed) ────────
        self._raw_chain: dict | None = None
        self._raw_chain_ts: float = 0.0

        # ── Near-expiry T (DTE in years) ─────────────────────────────
        self._T: float = 7 / 365   # sensible default

        # ── Listeners: list of callables notified on new spot ────────
        self._spot_listeners = []

        # ── OI Snapshot Ring (for per-strike OI velocity & historical rewind) ─
        # Each entry: {'ts': float, 'time_str': str, 'spot': float, 'oi_map': {(strike, type): int}}
        # 1500 snapshots with ~10s throttling ≈ 4-6+ hours of rolling OI history (full trading session)
        self._oi_ring: collections.deque = collections.deque(maxlen=1500)

        # ── 1-Min Candle Ring (for multi-candle absorption scoring) ──
        # Each entry: [ts, open, high, low, close, volume]  (Fyers format)
        # 120 bars = 2 trading hours
        self._candle_ring: collections.deque = collections.deque(maxlen=120)

    # ─────────────────────────────────────────────────────────────────
    # OI SNAPSHOT RING
    # ─────────────────────────────────────────────────────────────────

    def push_oi_snapshot(self, chain_df: pd.DataFrame, spot: float = 0.0):
        """
        Push a fresh OI snapshot from the live chain DataFrame.
        Called by DataServer's chain refresh loop.
        Stores {(strike, type): oi} keyed map with timestamp, time_str, and spot.
        """
        if chain_df is None or chain_df.empty:
            return
        try:
            now_ts = time.time()
            spot_val = float(spot or self._spot or 0.0)

            # Throttling: If last snapshot was recorded < 8s ago and spot hasn't moved noticeably, avoid duplicate
            if self._oi_ring:
                last_snap = self._oi_ring[-1]
                if (now_ts - last_snap['ts'] < 8.0) and abs(spot_val - last_snap.get('spot', 0.0)) < 3.0:
                    return

            oi_map = {}
            for _, row in chain_df[['strike', 'type', 'oi']].iterrows():
                key = (float(row['strike']), str(row['type']))
                oi_map[key] = int(row.get('oi', 0) or 0)
            time_str = datetime.now().strftime('%H:%M:%S')
            self._oi_ring.append({
                'ts': now_ts,
                'time_str': time_str,
                'spot': spot_val,
                'oi_map': oi_map
            })
        except Exception:
            pass

    def get_oi_velocity_data(self, window_secs: int = 900, fast_window_secs: int = 300) -> dict:
        """
        Compute per-strike OI change rate (15-min velocity) and acceleration (5-min fast velocity).
        Returns:
            {
                'vel_by_strike': {(strike, type): oi_delta_per_min},      # 15m rate
                'fast_vel_by_strike': {(strike, type): oi_delta_per_min}, # 5m rate
                'accel_by_strike': {(strike, type): delta_accel_per_min2},# 2nd derivative
                'pct_vel_by_strike': {(strike, type): pct_delta},         # % change vs baseline
                'violently_unwinding': [(strike, type), ...],             # delta < -100k/min
                'accelerating_unwind': [(strike, type), ...],             # V < -40k & A < -5k
                'exhausting_unwind': [(strike, type), ...],               # V < -40k & A > +5k
                'fortress_building': [(strike, type), ...],               # V > +50k & A >= 0
                'window_secs': int,
                'elapsed_min': float,
                'is_warmed_up': bool,
                'snapshot_count': int
            }
        """
        snaps = list(self._oi_ring)
        if len(snaps) < 2:
            return {}

        latest = snaps[-1]
        now = latest['ts']

        target_15m = now - window_secs
        target_5m = now - fast_window_secs

        # Find snapshot closest to target windows (guaranteed prior to latest)
        candidates = snaps[:-1]
        baseline_15m = min(candidates, key=lambda s: abs(s['ts'] - target_15m))
        baseline_5m = min(candidates, key=lambda s: abs(s['ts'] - target_5m))

        latest = snaps[-1]
        elapsed_min_15m = max((latest['ts'] - baseline_15m['ts']) / 60.0, 0.01)
        elapsed_min_5m = max((latest['ts'] - baseline_5m['ts']) / 60.0, 0.01)

        vel_by_strike = {}
        fast_vel_by_strike = {}
        accel_by_strike = {}
        pct_vel_by_strike = {}

        for key, oi_now in latest['oi_map'].items():
            # 15m velocity
            oi_base_15m = baseline_15m['oi_map'].get(key, oi_now)
            delta_15m = oi_now - oi_base_15m
            v_15m = round(delta_15m / elapsed_min_15m, 0)
            vel_by_strike[key] = v_15m

            # 5m fast velocity
            oi_base_5m = baseline_5m['oi_map'].get(key, oi_now)
            delta_5m = oi_now - oi_base_5m
            v_5m = round(delta_5m / elapsed_min_5m, 0)
            fast_vel_by_strike[key] = v_5m

            # Acceleration (2nd derivative): rate of velocity change
            accel = round((v_5m - v_15m) / max(elapsed_min_15m - elapsed_min_5m, 1.0), 1)
            accel_by_strike[key] = accel

            # Relative percentage change
            base_ref = max(oi_base_15m, 1000)
            pct_vel_by_strike[key] = round((delta_15m / base_ref) * 100.0, 2)

        violently_unwinding = [
            k for k, v in vel_by_strike.items() if v < -100_000
        ]
        accelerating_unwind = [
            k for k, v in vel_by_strike.items() if v < -40_000 and accel_by_strike.get(k, 0) < -5_000
        ]
        exhausting_unwind = [
            k for k, v in vel_by_strike.items() if v < -40_000 and accel_by_strike.get(k, 0) > 5_000
        ]
        fortress_building = [
            k for k, v in vel_by_strike.items() if (v > 30_000 or fast_vel_by_strike.get(k, 0) > 40_000) and accel_by_strike.get(k, 0) >= 0
        ]

        is_warmed_up = elapsed_min_15m >= 8.0

        return {
            'vel_by_strike': vel_by_strike,
            'fast_vel_by_strike': fast_vel_by_strike,
            'accel_by_strike': accel_by_strike,
            'pct_vel_by_strike': pct_vel_by_strike,
            'violently_unwinding': violently_unwinding,
            'accelerating_unwind': accelerating_unwind,
            'exhausting_unwind': exhausting_unwind,
            'fortress_building': fortress_building,
            'window_secs': window_secs,
            'elapsed_min': round(elapsed_min_15m, 2),
            'is_warmed_up': is_warmed_up,
            'snapshot_count': len(snaps)
        }

    # ─────────────────────────────────────────────────────────────────
    # 1-MIN CANDLE RING
    # ─────────────────────────────────────────────────────────────────

    def push_candle(self, candle: list):
        """
        Push a 1-min OHLCV candle [ts, open, high, low, close, volume].
        Called by DataServer's intraday background thread every minute.
        """
        if candle and len(candle) >= 6:
            self._candle_ring.append(candle)

    def get_recent_candles(self, n: int = 60) -> list:
        """
        Return the last `n` 1-min candles as a list of lists.
        Each candle: [ts, open, high, low, close, volume].
        """
        return list(self._candle_ring)[-n:]

    # ─────────────────────────────────────────────────────────────────
    # SPOT
    # ─────────────────────────────────────────────────────────────────

    def get_spot(self, force=False) -> float:
        """Return cached spot; re-fetch if older than SPOT_TTL."""
        age = time.time() - self._spot_ts
        if force or age > self.SPOT_TTL or self._spot == 0:
            self._fetch_spot()
        return self._spot

    def _fetch_spot(self):
        try:
            r = self.fyers.quotes(data={"symbols": self.symbol})
            if r.get('s') == 'ok':
                self._spot    = float(r['d'][0]['v'].get('lp', 0))
                self._spot_ts = time.time()
                # Notify listeners
                for fn in self._spot_listeners:
                    try:
                        fn(self._spot)
                    except Exception:
                        pass
        except Exception as e:
            pass  # keep previous value

    def add_spot_listener(self, fn):
        """Register a callback(spot) called whenever spot is refreshed."""
        self._spot_listeners.append(fn)

    # ─────────────────────────────────────────────────────────────────
    # OPTION CHAIN
    # ─────────────────────────────────────────────────────────────────

    def get_chain(self, expiry: str, force=False) -> pd.DataFrame:
        """
        Return cached option chain DataFrame for given expiry (YYYY-MM-DD).
        Re-fetches if older than CHAIN_TTL.
        """
        if expiry in self._chains:
            df, ts = self._chains[expiry]
            if not force and (time.time() - ts) < self.CHAIN_TTL:
                return df

        df = self._fetch_chain(expiry)
        self._chains[expiry] = (df, time.time())
        return df

    def _fetch_chain(self, expiry: str) -> pd.DataFrame:
        try:
            dt = datetime.strptime(expiry, "%Y-%m-%d")
            ts = int(dt.timestamp())
        except Exception:
            ts = ""

        try:
            r = self.fyers.optionchain(data={
                "symbol": self.symbol,
                "strikecount": 500,
                "timestamp": ts
            })

            # Handle expiry mismatch
            if r.get('s') == 'error' and isinstance(r.get('data'), dict):
                for item in r['data'].get('expiryData', []):
                    try:
                        a_date = datetime.strptime(item['date'], "%d-%m-%Y").date()
                        if a_date == datetime.strptime(expiry, "%Y-%m-%d").date():
                            r = self.fyers.optionchain(data={
                                "symbol": self.symbol,
                                "strikecount": 500,
                                "timestamp": item['expiry']
                            })
                            break
                    except Exception:
                        continue

            if r.get('s') == 'ok':
                records = []
                for item in r['data'].get('optionsChain', []):
                    records.append({
                        'strike': float(item.get('strike_price', 0)),
                        'type':   'CE' if item.get('option_type', '') in ('CE', 'CALL') else 'PE',
                        'price':  float(item.get('ltp', 0) or 0),
                        'iv':     float(item.get('iv', 0) or 0),
                        'oi':     int(item.get('oi', 0) or 0),
                        'delta':  float(item.get('delta', 0) or 0),
                        'gamma':  float(item.get('gamma', 0) or 0),
                        'theta':  float(item.get('theta', 0) or 0),
                        'vega':   float(item.get('vega', 0) or 0),
                    })
                return pd.DataFrame(records)
        except Exception:
            pass
        return pd.DataFrame()

    def list_expiries(self):
        """Fetch and return list of available expiry dates."""
        try:
            r = self.fyers.optionchain(data={
                "symbol": self.symbol, "strikecount": 1, "timestamp": ""
            })
            expiry_data = []
            if isinstance(r.get('data'), dict):
                expiry_data = r['data'].get('expiryData', [])
            return expiry_data
        except Exception:
            return []

    # ─────────────────────────────────────────────────────────────────
    # OHLC + RV  (fetched once per session)
    # ─────────────────────────────────────────────────────────────────

    def get_rv_data(self, force=False) -> dict:
        """
        Return cached RV/HV computation.
        Fetches 365d OHLC once per session and computes:
          rv_5d, rv_20d, hv_20d, consensus_rv, hv_percentile,
          closes (Series), ohlc_df (DataFrame)
        """
        if self._rv_data is not None and not force:
            return self._rv_data

        df = self._fetch_ohlc(365)
        if df.empty or len(df) < 30:
            return {}

        closes = df['close']
        log_rets = np.log(closes / closes.shift(1)).dropna()

        rv_5d  = float(log_rets.tail(5).std()  * np.sqrt(252) * 100)
        rv_10d = float(log_rets.tail(10).std() * np.sqrt(252) * 100)
        rv_20d = float(log_rets.tail(20).std() * np.sqrt(252) * 100)

        # Garman-Klass (more efficient)
        o, h, l, c = df['open'], df['high'], df['low'], df['close']
        log_hl = np.log(h / l) ** 2
        log_co = np.log(c / o) ** 2
        gk = (0.5 * log_hl - (2 * np.log(2) - 1) * log_co).rolling(20).mean()
        gk_rv = float(np.sqrt(gk.dropna().iloc[-1] * 252) * 100) if len(gk.dropna()) > 0 else rv_20d

        consensus_rv = float(np.mean([rv_20d, gk_rv]))

        # HV series for percentile
        hv_series = log_rets.rolling(20).std() * np.sqrt(252) * 100
        hv_series = hv_series.dropna()
        cur_hv    = float(hv_series.iloc[-1]) if len(hv_series) > 0 else rv_20d
        hv_pctile = float((hv_series < cur_hv).mean() * 100)

        rv_acceleration = rv_5d / rv_20d if rv_20d > 0 else 1.0

        self._rv_data = {
            'rv_5d':           round(rv_5d, 2),
            'rv_10d':          round(rv_10d, 2),
            'rv_20d':          round(rv_20d, 2),
            'gk_rv':           round(gk_rv, 2),
            'consensus_rv':    round(consensus_rv, 2),
            'hv_20d':          round(cur_hv, 2),
            'hv_percentile':   round(hv_pctile, 1),
            'rv_acceleration': round(rv_acceleration, 2),
            'closes':          closes,
            'hv_series':       hv_series,
            'ohlc_df':         df,
            'fetched_at':      datetime.now().strftime('%H:%M:%S')
        }
        return self._rv_data

    def _fetch_ohlc(self, days=365) -> pd.DataFrame:
        today = datetime.now()
        start = today - pd.Timedelta(days=days)
        try:
            r = self.fyers.history(data={
                "symbol": self.symbol, "resolution": "D", "date_format": "1",
                "range_from": start.strftime("%Y-%m-%d"),
                "range_to":   today.strftime("%Y-%m-%d"),
                "cont_flag":  "1"
            })
            if r.get('s') == 'ok':
                df = pd.DataFrame(r['candles'],
                                  columns=['ts', 'open', 'high', 'low', 'close', 'volume'])
                df['date'] = pd.to_datetime(df['ts'], unit='s')
                return df
        except Exception:
            pass
        return pd.DataFrame()

    # ─────────────────────────────────────────────────────────────────
    # HESTON PARAMS  (calibrated, TTL 300s)
    # ─────────────────────────────────────────────────────────────────

    def get_heston_params(self) -> dict | None:
        """Return cached Heston params, or None if stale / not yet calibrated."""
        if self._heston_params is None:
            return None
        if time.time() - self._heston_ts > self.HESTON_TTL:
            return None   # TTL expired — caller should trigger recalibration
        return self._heston_params

    def set_heston_params(self, params: dict):
        """Store freshly calibrated Heston params with a timestamp."""
        self._heston_params = params
        self._heston_ts = time.time()

    # ─────────────────────────────────────────────────────────────────
    # RAW CHAIN  (DataServer pushes its chain dict here so calibrator
    #             and PricingRouter can read without an extra Fyers call)
    # ─────────────────────────────────────────────────────────────────

    def set_raw_chain(self, chain_dict: dict):
        """Store the raw chain payload from DataServer."""
        self._raw_chain    = chain_dict
        self._raw_chain_ts = time.time()

    def get_raw_chain(self) -> dict | None:
        """Return the last pushed raw chain dict, or None if never set."""
        return self._raw_chain

    # ─────────────────────────────────────────────────────────────────
    # TIME-TO-EXPIRY  (DTE in years for the near expiry)
    # ─────────────────────────────────────────────────────────────────

    def set_T(self, T: float):
        """Store computed DTE in years for the current near expiry."""
        self._T = max(0.0, float(T))

    def get_T(self) -> float:
        """Return cached DTE in years (default 7/365 if never set)."""
        return self._T

