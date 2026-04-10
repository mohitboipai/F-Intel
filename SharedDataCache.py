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

