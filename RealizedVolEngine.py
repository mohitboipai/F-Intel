"""
RealizedVolEngine.py — Standalone Realized Volatility & Price Prediction Engine
================================================================================
Calculates Realized Volatility (RV) using multiple estimators, compares with
Historical Volatility (HV) and Implied Volatility (IV), and produces
directional price predictions based on volatility regime analysis.

Standalone module — does NOT modify or interrupt any existing program.
"""

import sys
import os
import time
import json
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Local imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from FyersAuth import FyersAuthenticator
from OptionAnalytics import OptionAnalytics


# ══════════════════════════════════════════════════════════════════════════════
# REALIZED VOLATILITY ESTIMATORS
# ══════════════════════════════════════════════════════════════════════════════

class RVEstimators:
    """Collection of Realized Volatility estimators."""

    @staticmethod
    def close_to_close(closes, window=5):
        """
        Standard Close-to-Close Realized Volatility.
        RV = std(log returns) * sqrt(252) over a rolling window.
        """
        s = pd.Series(closes) if not isinstance(closes, pd.Series) else closes
        log_rets = np.log(s / s.shift(1))
        rv = log_rets.rolling(window=window).std() * np.sqrt(252) * 100
        return rv.dropna()

    @staticmethod
    def parkinson(highs, lows, window=5):
        """
        Parkinson estimator — uses High/Low range.
        More efficient than close-to-close (uses 2 prices vs 1).
        """
        h = pd.Series(highs) if not isinstance(highs, pd.Series) else highs
        l = pd.Series(lows) if not isinstance(lows, pd.Series) else lows
        log_hl_sq = np.log(h / l) ** 2
        factor = 1.0 / (4.0 * window * np.log(2.0))
        rv = np.sqrt(factor * log_hl_sq.rolling(window=window).sum()) * np.sqrt(252) * 100
        return rv.dropna()

    @staticmethod
    def garman_klass(opens, highs, lows, closes, window=5):
        """
        Garman-Klass estimator — uses OHLC data.
        Most efficient estimator using all 4 prices.
        """
        o = pd.Series(opens) if not isinstance(opens, pd.Series) else opens
        h = pd.Series(highs) if not isinstance(highs, pd.Series) else highs
        l = pd.Series(lows) if not isinstance(lows, pd.Series) else lows
        c = pd.Series(closes) if not isinstance(closes, pd.Series) else closes

        log_hl = np.log(h / l) ** 2
        log_co = np.log(c / o) ** 2

        gk = 0.5 * log_hl - (2 * np.log(2) - 1) * log_co
        rv = np.sqrt(gk.rolling(window=window).mean() * 252) * 100
        return rv.dropna()

    @staticmethod
    def yang_zhang(opens, highs, lows, closes, window=5):
        """
        Yang-Zhang estimator — best when opening jumps are significant.
        Combines overnight and intraday components.
        """
        o = pd.Series(opens) if not isinstance(opens, pd.Series) else opens
        h = pd.Series(highs) if not isinstance(highs, pd.Series) else highs
        l = pd.Series(lows) if not isinstance(lows, pd.Series) else lows
        c = pd.Series(closes) if not isinstance(closes, pd.Series) else closes

        # Overnight returns
        log_oc = np.log(o / c.shift(1))
        # Open-to-close returns
        log_co = np.log(c / o)
        # Rogers-Satchell component
        rs = np.log(h / c) * np.log(h / o) + np.log(l / c) * np.log(l / o)

        k = 0.34 / (1.34 + (window + 1) / (window - 1))

        overnight_var = log_oc.rolling(window).var()
        close_var = log_co.rolling(window).var()
        rs_var = rs.rolling(window).mean()

        yz_var = overnight_var + k * close_var + (1 - k) * rs_var
        rv = np.sqrt(yz_var.clip(lower=0) * 252) * 100
        return rv.dropna()


# ══════════════════════════════════════════════════════════════════════════════
# MAIN ENGINE
# ══════════════════════════════════════════════════════════════════════════════

class RealizedVolEngine:
    """
    Standalone engine that:
      1. Fetches historical OHLC data via Fyers API
      2. Calculates multi-window Realized Volatility (RV)
      3. Compares RV with HV (rolling 20d) and ATM IV
      4. Produces a volatility regime classification
      5. Generates a price prediction (direction + magnitude)
    """

    def __init__(self):
        self.fyers = self._authenticate()
        self.analytics = OptionAnalytics()
        self.symbol = "NSE:NIFTY50-INDEX"
        self.spot_price = 0
        self.rv_estimators = RVEstimators()
        # Optional shared signal memory (inject from outside if desired)
        self.memory = None

    # ── Auth ──────────────────────────────────────────────────────────────
    def _authenticate(self):
        print("Authenticating with Fyers...")
        APP_ID = "QUTT4YYMIG-100"
        SECRET_ID = "ZG0WN2NL1B"
        REDIRECT_URI = "http://127.0.0.1:3000/callback"
        auth = FyersAuthenticator(APP_ID, SECRET_ID, REDIRECT_URI)
        fyers = auth.get_fyers_instance()
        if not fyers:
            print("Authentication Failed!")
            sys.exit(1)
        print("Authentication Successful.")
        return fyers

    # ── Spot ──────────────────────────────────────────────────────────────
    def _get_spot(self):
        data = {"symbols": self.symbol}
        try:
            r = self.fyers.quotes(data=data)
            if r.get('code') == -15 or "token" in r.get('message', '').lower():
                self.fyers = self._authenticate()
                r = self.fyers.quotes(data=data)
            if r.get('s') == 'ok':
                self.spot_price = r['d'][0]['v'].get('lp', 0)
        except Exception as e:
            print(f"Spot fetch error: {e}")
        return self.spot_price

    # ── History ───────────────────────────────────────────────────────────
    def _fetch_daily_history(self, days=365):
        """Fetch daily OHLC candles for the last N days."""
        today = datetime.now()
        start = today - pd.Timedelta(days=days)
        data = {
            "symbol": self.symbol, "resolution": "D", "date_format": "1",
            "range_from": start.strftime("%Y-%m-%d"),
            "range_to": today.strftime("%Y-%m-%d"),
            "cont_flag": "1"
        }
        r = self.fyers.history(data=data)
        if r.get('s') == 'ok':
            candles = r['candles']
            df = pd.DataFrame(candles, columns=['ts', 'open', 'high', 'low', 'close', 'volume'])
            df['date'] = pd.to_datetime(df['ts'], unit='s')
            return df
        else:
            print(f"History fetch failed: {r}")
            return pd.DataFrame()

    def _fetch_intraday_history(self):
        """Fetch today's 5-minute intraday candles for real-time RV/VWAP context."""
        today = datetime.now()
        data = {
            "symbol": self.symbol, "resolution": "5", "date_format": "1",
            "range_from": today.strftime("%Y-%m-%d"),
            "range_to": today.strftime("%Y-%m-%d"),
            "cont_flag": "1"
        }
        try:
            r = self.fyers.history(data=data)
            if r.get('s') == 'ok' and r.get('candles'):
                df_intra = pd.DataFrame(r['candles'], columns=['ts', 'open', 'high', 'low', 'close', 'volume'])
                return df_intra
        except Exception as e:
            print(f"Intraday fetch error: {e}")
        return pd.DataFrame()

    # ── ATM IV ────────────────────────────────────────────────────────────
    def _fetch_atm_iv(self, expiry):
        """Fetch current ATM IV from the option chain for a given expiry."""
        spot = self.spot_price
        if spot <= 0:
            return 0

        try:
            atm_strike = round(spot / 50) * 50
            strike_count = 5
            data_req = {
                "symbol": self.symbol,
                "strikecount": strike_count,
                "timestamp": expiry
            }
            r = self.fyers.optionchain(data=data_req)

            if r.get('s') != 'ok' or 'data' not in r:
                return 0

            options = r['data'].get('optionsChain', [])
            best_iv = 0
            best_dist = float('inf')

            for opt in options:
                strike = opt.get('strike_price', opt.get('strikePrice', 0))
                o_type = opt.get('option_type', opt.get('optionType', ''))
                iv_val = opt.get('iv', 0)

                # Normalize type
                if o_type in ('CE', 'call', 'Call'): o_type = 'CE'
                elif o_type in ('PE', 'put', 'Put'): o_type = 'PE'

                dist = abs(strike - spot)
                if dist < best_dist and iv_val > 0:
                    best_dist = dist
                    best_iv = iv_val

            # If IV looks like decimal (e.g., 0.15), convert to percentage
            if 0 < best_iv < 1:
                best_iv *= 100

            return best_iv

        except Exception as e:
            print(f"Option chain fetch error: {e}")
            return 0

    # ══════════════════════════════════════════════════════════════════════
    # PROGRAMMATIC API (NON-BLOCKING)
    # ══════════════════════════════════════════════════════════════════════

    def get_regime_snapshot(self, spot, df_daily, atm_iv, intra_vwap=0, intra_rv=0):
        """
        Computes regime stats programmatically returning a dictionary.
        Does not block or prompt for user input.
        """
        if df_daily.empty or len(df_daily) < 20:
            return None

        closes = df_daily['closes'] if 'closes' in df_daily else df_daily['close']
        highs = df_daily['highs'] if 'highs' in df_daily else df_daily['high']
        lows = df_daily['lows'] if 'lows' in df_daily else df_daily['low']
        opens = df_daily['opens'] if 'opens' in df_daily else df_daily['open']

        rv_5d  = self.rv_estimators.close_to_close(closes, window=5)
        rv_10d = self.rv_estimators.close_to_close(closes, window=10)
        rv_20d = self.rv_estimators.close_to_close(closes, window=20)
        rv_60d = self.rv_estimators.close_to_close(closes, window=60) if len(closes) >= 60 else rv_20d

        rv_park = self.rv_estimators.parkinson(highs, lows, window=20)
        rv_gk   = self.rv_estimators.garman_klass(opens, highs, lows, closes, window=20)
        rv_yz   = self.rv_estimators.yang_zhang(opens, highs, lows, closes, window=20)

        cur_rv_5  = float(rv_5d.iloc[-1]) if len(rv_5d) > 0 else 0
        cur_rv_10 = float(rv_10d.iloc[-1]) if len(rv_10d) > 0 else 0
        cur_rv_20 = float(rv_20d.iloc[-1]) if len(rv_20d) > 0 else 0
        cur_rv_60 = float(rv_60d.iloc[-1]) if len(rv_60d) > 0 else 0
        cur_park  = float(rv_park.iloc[-1]) if len(rv_park) > 0 else 0
        cur_gk    = float(rv_gk.iloc[-1]) if len(rv_gk) > 0 else 0
        cur_yz    = float(rv_yz.iloc[-1]) if len(rv_yz) > 0 else 0

        # Consensus RV
        vals = [v for v in [cur_rv_20, cur_park, cur_gk, cur_yz] if v > 0]
        consensus_rv = np.mean(vals) if vals else cur_rv_20

        # Historical Vol
        hv_series = self.analytics.calculate_rolling_historical_volatility(closes, window=20)
        cur_hv = float(hv_series.iloc[-1]) if len(hv_series) > 0 else 0
        hv_pctile = float((hv_series < cur_hv).mean() * 100) if len(hv_series) > 0 else 50
        
        hv_mean = float(hv_series.mean()) if len(hv_series) > 0 else cur_hv

        # Trend and spreads
        rv_trend = "STABLE"
        if cur_rv_5 > 0 and cur_rv_20 > 0:
            if cur_rv_5 / cur_rv_20 > 1.3: rv_trend = "ACCELERATING"
            elif cur_rv_5 / cur_rv_20 < 0.7: rv_trend = "DECELERATING"

        vrp_iv_hv = atm_iv - cur_hv
        vrp_iv_rv = atm_iv - consensus_rv
        rv_term_slope = cur_rv_5 - cur_rv_60 # short term vs long term spread

        # Classify regime
        regime = self._classify_regime(atm_iv, cur_hv, consensus_rv, rv_trend, hv_pctile)

        return {
            'spot': spot,
            'atm_iv': atm_iv,
            'rv': {
                '5d': cur_rv_5, '10d': cur_rv_10, '20d': cur_rv_20, '60d': cur_rv_60,
                'parkinson_20d': cur_park, 'garman_klass_20d': cur_gk, 'yang_zhang_20d': cur_yz,
                'consensus': consensus_rv, 'intraday': intra_rv,
                'trend': rv_trend, 'term_slope': rv_term_slope
            },
            'hv': {'20d': cur_hv, 'percentile': hv_pctile, 'mean': hv_mean},
            'vrp': {'iv_hv': vrp_iv_hv, 'iv_rv': vrp_iv_rv},
            'regime': regime
        }

    # ══════════════════════════════════════════════════════════════════════
    # CORE: Volatility Comparison & Prediction
    # ══════════════════════════════════════════════════════════════════════


    def run_analysis(self):
        """Main analysis pipeline — called from menu."""
        print("\n" + "═" * 70)
        print("  REALIZED VOLATILITY & PRICE PREDICTION ENGINE")
        print("═" * 70)

        # ── 1. Fetch Data ─────────────────────────────────────────────────
        print("\n[1/6] Fetching 1 Year of Daily OHLC Data...")
        df = self._fetch_daily_history(365)
        if df.empty or len(df) < 40:
            print("Not enough data. Aborting.")
            return
        print(f"      Loaded {len(df)} trading days.")

        opens  = df['open']
        highs  = df['high']
        lows   = df['low']
        closes = df['close']

        # ── 2. Multi-Window RV ────────────────────────────────────────────
        print("\n[2/6] Computing Multi-Window Realized Volatility...")

        rv_5d  = self.rv_estimators.close_to_close(closes, window=5)
        rv_10d = self.rv_estimators.close_to_close(closes, window=10)
        rv_20d = self.rv_estimators.close_to_close(closes, window=20)

        # Advanced estimators (20d window)
        rv_park = self.rv_estimators.parkinson(highs, lows, window=20)
        rv_gk   = self.rv_estimators.garman_klass(opens, highs, lows, closes, window=20)
        rv_yz   = self.rv_estimators.yang_zhang(opens, highs, lows, closes, window=20)

        # Current values (last reading)
        cur_rv_5  = rv_5d.iloc[-1]  if len(rv_5d)  > 0 else 0
        cur_rv_10 = rv_10d.iloc[-1] if len(rv_10d) > 0 else 0
        cur_rv_20 = rv_20d.iloc[-1] if len(rv_20d) > 0 else 0
        cur_park  = rv_park.iloc[-1] if len(rv_park) > 0 else 0
        cur_gk    = rv_gk.iloc[-1]   if len(rv_gk)   > 0 else 0
        cur_yz    = rv_yz.iloc[-1]   if len(rv_yz)   > 0 else 0

        # Consensus RV = average of all 20d estimators (robust)
        estimator_vals = [v for v in [cur_rv_20, cur_park, cur_gk, cur_yz] if v > 0]
        consensus_rv = np.mean(estimator_vals) if estimator_vals else cur_rv_20

        # ── 2.5 Intraday Responsiveness (VWAP & Intraday Parkinson) ───────
        df_intra = self._fetch_intraday_history()
        intraday_parkinson = 0
        vwap = 0
        spot_vs_vwap = 0

        if not df_intra.empty and len(df_intra) > 2:
            # Intraday Parkinson
            highs_i = df_intra['high']
            lows_i  = df_intra['low']
            log_hl_sq_i = np.log(highs_i / lows_i) ** 2
            N = len(df_intra)
            factor = 1.0 / (4.0 * N * np.log(2.0))
            intraday_variance = factor * log_hl_sq_i.sum()
            # Annualize assuming 75 5-min periods per day
            intraday_parkinson = np.sqrt(intraday_variance) * np.sqrt(75 * 252) * 100
            
            # Intraday VWAP
            df_intra['typ'] = (df_intra['high'] + df_intra['low'] + df_intra['close']) / 3
            df_intra['vol_price'] = df_intra['typ'] * df_intra['volume']
            vwap = df_intra['vol_price'].sum() / (df_intra['volume'].sum() + 1e-5)
            spot_vs_vwap = self.spot_price - vwap

        print(f"\n  ┌─────────────────────────────────────────────┐")
        print(f"  │  REALIZED VOLATILITY (Annualized)           │")
        print(f"  ├─────────────────────────────────────────────┤")
        print(f"  │  Close-to-Close  5d:  {cur_rv_5:>7.2f}%              │")
        print(f"  │  Close-to-Close 10d:  {cur_rv_10:>7.2f}%              │")
        print(f"  │  Close-to-Close 20d:  {cur_rv_20:>7.2f}%              │")
        print(f"  │  Parkinson       20d:  {cur_park:>7.2f}%             │")
        print(f"  │  Garman-Klass    20d:  {cur_gk:>7.2f}%             │")
        print(f"  │  Yang-Zhang      20d:  {cur_yz:>7.2f}%             │")
        print(f"  ├─────────────────────────────────────────────┤")
        print(f"  │  ★ Consensus RV:       {consensus_rv:>7.2f}%             │")
        print(f"  │  ⚡ Intraday Park:      {intraday_parkinson:>7.2f}%             │")
        print(f"  └─────────────────────────────────────────────┘")
        if vwap > 0:
            vwap_txt = "BULLISH" if spot_vs_vwap > 0 else "BEARISH"
            print(f"      VWAP Signal: {vwap_txt} (Spot {spot_vs_vwap:+.1f} pts from {vwap:.0f})")

        # ── 3. Historical Volatility (HV) ────────────────────────────────
        print("\n[3/6] Computing Historical Volatility (HV)...")
        hv_series = self.analytics.calculate_rolling_historical_volatility(closes, window=20)
        cur_hv = hv_series.iloc[-1] if len(hv_series) > 0 else 0
        hv_mean = hv_series.mean()
        hv_percentile = (hv_series < cur_hv).mean() * 100

        print(f"      20d HV: {cur_hv:.2f}% | Mean: {hv_mean:.2f}% | Percentile: {hv_percentile:.0f}%")

        # ── 4. Implied Volatility (IV) ───────────────────────────────────
        print("\n[4/6] Fetching Current ATM Implied Volatility...")
        spot = self._get_spot()
        print(f"      Spot: {spot:,.2f}")

        expiry = input("      Enter Expiry (YYYY-MM-DD) or press Enter for manual IV: ").strip()

        atm_iv = 0
        if expiry:
            atm_iv = self._fetch_atm_iv(expiry)

        if atm_iv <= 0:
            try:
                atm_iv = float(input("      Enter Manual ATM IV (%): ") or "15")
            except ValueError:
                atm_iv = 15.0

        print(f"      ATM IV: {atm_iv:.2f}%")

        # ── 5. Volatility Comparison ─────────────────────────────────────
        print("\n[5/6] Volatility Comparison & Regime Detection...")

        vrp_iv_hv  = atm_iv - cur_hv          # Traditional VRP
        vrp_iv_rv  = atm_iv - consensus_rv     # True VRP (IV vs actual RV)
        rv_hv_diff = consensus_rv - cur_hv     # RV vs HV spread

        # RV Trend (short-term vs long-term)
        rv_trend = "STABLE"
        if cur_rv_5 > 0 and cur_rv_20 > 0:
            ratio = cur_rv_5 / cur_rv_20
            if ratio > 1.3:
                rv_trend = "ACCELERATING"
            elif ratio < 0.7:
                rv_trend = "DECELERATING"

        # ── REGIME CLASSIFICATION ─────────────────────────────────────────
        regime = self._classify_regime(atm_iv, cur_hv, consensus_rv, rv_trend, hv_percentile)

        print(f"\n  ┌─────────────────────────────────────────────────────────┐")
        print(f"  │  VOLATILITY TRIANGLE COMPARISON                        │")
        print(f"  ├─────────────────────────────────────────────────────────┤")
        print(f"  │  ATM IV:       {atm_iv:>7.2f}%                                │")
        print(f"  │  20d HV:       {cur_hv:>7.2f}%                                │")
        print(f"  │  Consensus RV: {consensus_rv:>7.2f}%                                │")
        print(f"  ├─────────────────────────────────────────────────────────┤")
        print(f"  │  VRP (IV - HV):    {vrp_iv_hv:>+7.2f}%                          │")
        print(f"  │  VRP (IV - RV):    {vrp_iv_rv:>+7.2f}%  ← True Mispricing      │")
        print(f"  │  RV - HV Spread:   {rv_hv_diff:>+7.2f}%                          │")
        print(f"  │  RV Trend:          {rv_trend:<15s}                   │")
        print(f"  ├─────────────────────────────────────────────────────────┤")
        print(f"  │  ★ REGIME: {regime['name']:<20s}                      │")
        print(f"  │    {regime['description']:<55s}│")
        print(f"  └─────────────────────────────────────────────────────────┘")

        # ── 6. Price Prediction ──────────────────────────────────────────
        print("\n[6/7] Computing Price Prediction...")
        prediction = self._predict_price(
            spot, atm_iv, cur_hv, consensus_rv, rv_trend, regime,
            hv_percentile, closes, hv_series, rv_5d, rv_20d
        )

        self._print_prediction(prediction, spot)

        # ── 7. Gap Analysis & Prediction ─────────────────────────────────
        print("\n[7/7] Analyzing Overnight Gap Patterns...")
        gap_data = self._analyze_gaps(df)
        if gap_data is not None:
            gap_pred = self._predict_next_gap(
                gap_data, atm_iv, cur_hv, consensus_rv, rv_trend, regime, spot
            )
            self._print_gap_prediction(gap_pred, gap_data, spot)
        else:
            gap_pred = None
            print("      Insufficient data for gap analysis.")

        # ── Save & Plot ──────────────────────────────────────────────────
        self._save_snapshot(spot, atm_iv, cur_hv, consensus_rv, regime, prediction)

        # ── Persist to SignalMemory if available ─────────────────────────
        if self.memory is not None:
            try:
                vrp = round(atm_iv - consensus_rv, 2) if consensus_rv > 0 else 0
                self.memory.update_context({
                    'regime':       regime.get('name', 'EQUILIBRIUM'),
                    'vrp':          vrp,
                    'atm_iv':       round(float(atm_iv), 2),
                    'consensus_rv': round(float(consensus_rv), 2),
                    'intraday_rv':  round(float(intraday_parkinson), 2),
                    'rv_5d':        round(float(rv_5d.iloc[-1] if len(rv_5d)>0 else 0), 2),
                    'rv_20d':       round(float(cur_rv_20), 2),
                    'vwap_dist':    round(float(spot_vs_vwap), 2)
                }, spot=spot)
            except Exception:
                pass

        self._plot_dashboard(
            df, rv_5d, rv_10d, rv_20d, hv_series, atm_iv,
            consensus_rv, cur_hv, regime, prediction, spot
        )

        input("\nPress Enter to continue...")

    # ══════════════════════════════════════════════════════════════════════
    # REGIME CLASSIFIER
    # ══════════════════════════════════════════════════════════════════════

    def _classify_regime(self, iv, hv, rv, rv_trend, hv_pctile):
        """
        Classify the current volatility regime based on IV/HV/RV relationships.

        Regimes:
          SQUEEZE      → Low IV, Low RV, Low HV — coiled spring, expect big move
          OVERPRICED   → IV >> RV — options expensive, sell vol
          UNDERPRICED  → IV << RV — options cheap, buy vol
          MOMENTUM     → High RV > HV, rising RV — trending market
          MEAN REVERT  → RV falling after spike, IV still elevated
          EQUILIBRIUM  → IV ≈ HV ≈ RV — fairly priced, no edge
        """
        iv_rv_ratio = iv / rv if rv > 0 else 1.0
        iv_hv_ratio = iv / hv if hv > 0 else 1.0
        rv_hv_ratio = rv / hv if hv > 0 else 1.0

        # SQUEEZE: everything compressed
        if hv_pctile < 20 and iv < hv * 1.1 and rv < hv * 1.1:
            return {
                'name': 'SQUEEZE',
                'description': 'Vol coiled tight — explosive move likely coming',
                'bias': 'NEUTRAL',  # direction unknown
                'vol_action': 'BUY VOL',
                'confidence': 0.75
            }

        # OVERPRICED: IV well above actual realized moves
        if iv_rv_ratio > 1.25 and iv_hv_ratio > 1.15:
            return {
                'name': 'OVERPRICED',
                'description': 'Options expensive vs actual moves — sell premium',
                'bias': 'NEUTRAL',
                'vol_action': 'SELL VOL',
                'confidence': min(0.9, (iv_rv_ratio - 1.0) * 1.5)
            }

        # UNDERPRICED: IV well below actual realized moves
        if iv_rv_ratio < 0.80 and iv_hv_ratio < 0.90:
            return {
                'name': 'UNDERPRICED',
                'description': 'Options cheap vs actual moves — buy protection',
                'bias': 'BEARISH',  # cheap options in fear markets
                'vol_action': 'BUY VOL',
                'confidence': min(0.9, (1.0 - iv_rv_ratio) * 2.0)
            }

        # MOMENTUM: RV accelerating, market trending
        if rv_trend == 'ACCELERATING' and rv_hv_ratio > 1.15:
            return {
                'name': 'MOMENTUM',
                'description': 'RV accelerating — strong trend in play',
                'bias': 'TREND',
                'vol_action': 'BUY VOL',
                'confidence': 0.65
            }

        # MEAN REVERSION: RV declining from spike
        if rv_trend == 'DECELERATING' and hv_pctile > 70:
            return {
                'name': 'MEAN REVERSION',
                'description': 'Post-spike vol collapse — sell premium into decay',
                'bias': 'NEUTRAL',
                'vol_action': 'SELL VOL',
                'confidence': 0.70
            }

        # EQUILIBRIUM: everything balanced
        return {
            'name': 'EQUILIBRIUM',
            'description': 'IV ≈ HV ≈ RV — fairly priced, wait for dislocation',
            'bias': 'NEUTRAL',
            'vol_action': 'WAIT',
            'confidence': 0.30
        }

    # ══════════════════════════════════════════════════════════════════════
    # GAP ANALYSIS ENGINE
    # ══════════════════════════════════════════════════════════════════════

    def _analyze_gaps(self, df):
        """
        Analyze historical overnight gaps from OHLC data.

        Gap = (Today's Open - Yesterday's Close) / Yesterday's Close * 100

        Returns a dict with gap statistics, distributions, and regime buckets.
        """
        opens  = df['open'].values
        closes = df['close'].values
        highs  = df['high'].values
        lows   = df['low'].values

        # Overnight gap series: (Open_t - Close_{t-1}) / Close_{t-1} * 100
        gap_pct = pd.Series(
            [(opens[i] - closes[i-1]) / closes[i-1] * 100 for i in range(1, len(opens))],
            index=df.index[1:]
        )
        gap_pts = pd.Series(
            [opens[i] - closes[i-1] for i in range(1, len(opens))],
            index=df.index[1:]
        )

        if len(gap_pct) < 20:
            return None

        # ── Basic Statistics ──────────────────────────────────────────────
        abs_gaps = gap_pct.abs()
        stats = {
            'total_days': len(gap_pct),
            'mean_gap': gap_pct.mean(),
            'median_gap': gap_pct.median(),
            'std_gap': gap_pct.std(),
            'mean_abs_gap': abs_gaps.mean(),
            'max_gap_up': gap_pct.max(),
            'max_gap_down': gap_pct.min(),
            'max_gap_up_pts': gap_pts.max(),
            'max_gap_down_pts': gap_pts.min(),
            'pct_gap_up': (gap_pct > 0).mean() * 100,
            'pct_gap_down': (gap_pct < 0).mean() * 100,
            'pct_large_gap': (abs_gaps > 0.5).mean() * 100,  # > 0.5% is "large"
            'pct_very_large_gap': (abs_gaps > 1.0).mean() * 100,  # > 1% is "very large"
        }

        # ── Recent Gap Pattern (last 10 days) ─────────────────────────────
        recent_10 = gap_pct.tail(10)
        stats['recent_mean_gap'] = recent_10.mean()
        stats['recent_std_gap'] = recent_10.std()
        stats['recent_up_pct'] = (recent_10 > 0).mean() * 100
        stats['consecutive_direction'] = 0
        if len(recent_10) >= 2:
            last_dir = 1 if recent_10.iloc[-1] > 0 else -1
            count = 0
            for g in reversed(recent_10.values):
                if (g > 0 and last_dir == 1) or (g <= 0 and last_dir == -1):
                    count += 1
                else:
                    break
            stats['consecutive_direction'] = count * last_dir  # positive = consecutive up

        # ── Gap Size by Volatility Regime ─────────────────────────────────
        # Compute 5d RV at each point and bucket gaps by vol environment
        log_rets = np.log(pd.Series(closes) / pd.Series(closes).shift(1)).dropna()
        rolling_rv = (log_rets.rolling(5).std() * np.sqrt(252) * 100).dropna()

        # Align: rv is shorter, gap is shifted by 1
        min_len = min(len(rolling_rv), len(gap_pct))
        aligned = pd.DataFrame({
            'rv': rolling_rv.values[-min_len:],
            'gap': gap_pct.values[-min_len:]
        })

        regime_gaps = {}
        if len(aligned) > 30:
            try:
                aligned['vol_bucket'] = pd.qcut(aligned['rv'], 4,
                    labels=['Low Vol', 'Mid-Low Vol', 'Mid-High Vol', 'High Vol'],
                    duplicates='drop')
                for bucket in ['Low Vol', 'Mid-Low Vol', 'Mid-High Vol', 'High Vol']:
                    b_data = aligned[aligned['vol_bucket'] == bucket]['gap']
                    if len(b_data) > 3:
                        regime_gaps[bucket] = {
                            'mean': b_data.mean(),
                            'std': b_data.std(),
                            'mean_abs': b_data.abs().mean(),
                            'pct_up': (b_data > 0).mean() * 100,
                            'pct_large': (b_data.abs() > 0.5).mean() * 100,
                            'count': len(b_data)
                        }
            except Exception:
                pass

        # ── Day-of-Week Seasonality ───────────────────────────────────────
        dow_gaps = {}
        if 'date' in df.columns:
            gap_df = pd.DataFrame({'gap': gap_pct, 'date': df['date'].iloc[1:].values})
            gap_df['dow'] = pd.to_datetime(gap_df['date']).dt.day_name()
            for day in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']:
                day_data = gap_df[gap_df['dow'] == day]['gap']
                if len(day_data) > 3:
                    dow_gaps[day] = {
                        'mean': day_data.mean(),
                        'mean_abs': day_data.abs().mean(),
                        'pct_up': (day_data > 0).mean() * 100,
                        'count': len(day_data)
                    }

        return {
            'stats': stats,
            'gap_series': gap_pct,
            'gap_pts_series': gap_pts,
            'regime_gaps': regime_gaps,
            'dow_gaps': dow_gaps,
            'aligned': aligned
        }

    def _predict_next_gap(self, gap_data, iv, hv, rv, rv_trend, regime, spot):
        """
        Predict tomorrow's opening gap direction and magnitude.

        Uses:
          - Recent gap pattern (momentum/mean-reversion)
          - Vol regime (high vol = bigger gaps)
          - VRP (overpriced IV = likely gap UP as fear unwinds)
          - Day-of-week seasonality
          - Consecutive gap direction (mean reversion after 3+ same-direction)
        """
        stats = gap_data['stats']
        gap_series = gap_data['gap_series']
        regime_gaps = gap_data['regime_gaps']
        dow_gaps = gap_data['dow_gaps']

        scores = {}  # -100 (gap down) to +100 (gap up)

        # 1. Recent gap momentum
        recent_mean = stats['recent_mean_gap']
        if recent_mean > 0.15:
            scores['recent_pattern'] = 25   # Recent gaps are up
        elif recent_mean < -0.15:
            scores['recent_pattern'] = -25  # Recent gaps are down
        else:
            scores['recent_pattern'] = 0

        # 2. Consecutive direction → mean reversion
        consec = stats['consecutive_direction']
        if abs(consec) >= 3:
            # After 3+ consecutive same-direction gaps, expect reversal
            scores['mean_reversion'] = -30 if consec > 0 else 30
        elif abs(consec) >= 2:
            scores['mean_reversion'] = -15 if consec > 0 else 15
        else:
            scores['mean_reversion'] = 0

        # 3. VRP signal (IV vs RV)
        vrp = iv - rv
        if vrp > 5:
            scores['vrp'] = 20    # Market overpricing fear → fear likely fades → gap up
        elif vrp > 2:
            scores['vrp'] = 10
        elif vrp < -3:
            scores['vrp'] = -25   # Options too cheap → real risk → gap down
        elif vrp < -1:
            scores['vrp'] = -10
        else:
            scores['vrp'] = 0

        # 4. RV Trend
        if rv_trend == 'ACCELERATING':
            scores['rv_trend'] = -20  # Rising vol → bigger/negative gaps
        elif rv_trend == 'DECELERATING':
            scores['rv_trend'] = 15   # Calming → positive gaps
        else:
            scores['rv_trend'] = 0

        # 5. Regime-based expected gap
        if regime['name'] == 'SQUEEZE':
            scores['regime'] = 0      # Direction unknown but size will be big
        elif regime['name'] == 'OVERPRICED':
            scores['regime'] = 15     # Fear overstated → gap up
        elif regime['name'] == 'UNDERPRICED':
            scores['regime'] = -20    # Real risk → gap down
        elif regime['name'] == 'MOMENTUM':
            # Follow recent direction
            scores['regime'] = scores.get('recent_pattern', 0) * 0.5
        else:
            scores['regime'] = 0

        # 6. Day-of-week
        tomorrow = datetime.now() + pd.Timedelta(days=1)
        # Skip weekends
        while tomorrow.weekday() >= 5:
            tomorrow += pd.Timedelta(days=1)
        tomorrow_name = tomorrow.strftime('%A')
        if tomorrow_name in dow_gaps:
            dow_bias = dow_gaps[tomorrow_name]['mean']
            scores['day_of_week'] = max(-20, min(20, dow_bias * 40))
        else:
            scores['day_of_week'] = 0

        # ── Composite ─────────────────────────────────────────────────────
        weights = {
            'recent_pattern': 0.20,
            'mean_reversion': 0.20,
            'vrp': 0.25,
            'rv_trend': 0.15,
            'regime': 0.10,
            'day_of_week': 0.10
        }

        composite = sum(scores.get(k, 0) * w for k, w in weights.items())

        # Direction
        if composite > 8:
            direction = 'GAP UP'
        elif composite < -8:
            direction = 'GAP DOWN'
        else:
            direction = 'FLAT OPEN'

        # Expected magnitude — from vol regime bucket
        # Find which vol bucket current RV falls into
        expected_gap_pct = stats['mean_abs_gap']  # default
        current_rv_bucket = None
        if regime_gaps:
            # Determine which bucket we're in
            if rv < stats.get('recent_std_gap', 999) * 5:  # rough check
                for bucket_name in ['Low Vol', 'Mid-Low Vol', 'Mid-High Vol', 'High Vol']:
                    if bucket_name in regime_gaps:
                        current_rv_bucket = bucket_name
                        expected_gap_pct = regime_gaps[bucket_name]['mean_abs']

            # Better: use the aligned data to find actual bucket
            aligned = gap_data['aligned']
            if 'vol_bucket' in aligned.columns and len(aligned) > 0:
                try:
                    rv_breaks = aligned['rv'].quantile([0, 0.25, 0.5, 0.75, 1.0]).values
                    if rv <= rv_breaks[1]:   current_rv_bucket = 'Low Vol'
                    elif rv <= rv_breaks[2]: current_rv_bucket = 'Mid-Low Vol'
                    elif rv <= rv_breaks[3]: current_rv_bucket = 'Mid-High Vol'
                    else:                   current_rv_bucket = 'High Vol'

                    if current_rv_bucket in regime_gaps:
                        expected_gap_pct = regime_gaps[current_rv_bucket]['mean_abs']
                except Exception:
                    pass

        expected_gap_pts = spot * expected_gap_pct / 100
        confidence = min(0.85, abs(composite) / 50 + 0.15)

        return {
            'direction': direction,
            'composite_score': round(composite, 1),
            'confidence': round(confidence, 2),
            'expected_gap_pct': round(expected_gap_pct, 3),
            'expected_gap_pts': round(expected_gap_pts, 0),
            'current_vol_bucket': current_rv_bucket or 'Unknown',
            'tomorrow': tomorrow_name,
            'sub_scores': scores,
            'stats': stats
        }

    def _print_gap_prediction(self, gap_pred, gap_data, spot):
        """Rich console output for gap analysis and prediction."""
        stats = gap_pred['stats']

        print(f"\n{'=' * 70}")
        print(f"  OVERNIGHT GAP ANALYSIS & PREDICTION ENGINE")
        print(f"{'=' * 70}")

        # ── Historical Gap Statistics ─────────────────────────────────────
        print(f"\n  Historical Gap Statistics ({stats['total_days']} trading days):")
        print(f"  {'─' * 55}")
        print(f"  Mean Gap:            {stats['mean_gap']:+.3f}%  ({spot * stats['mean_gap']/100:+.1f} pts)")
        print(f"  Mean |Gap|:          {stats['mean_abs_gap']:.3f}%   ({spot * stats['mean_abs_gap']/100:.1f} pts)")
        print(f"  Std Dev:             {stats['std_gap']:.3f}%")
        print(f"  Max Gap Up:          {stats['max_gap_up']:+.3f}%  ({stats['max_gap_up_pts']:+.0f} pts)")
        print(f"  Max Gap Down:        {stats['max_gap_down']:+.3f}%  ({stats['max_gap_down_pts']:+.0f} pts)")
        print(f"  % Days Gap Up:       {stats['pct_gap_up']:.1f}%")
        print(f"  % Days Gap Down:     {stats['pct_gap_down']:.1f}%")
        print(f"  % Large Gaps (>0.5%): {stats['pct_large_gap']:.1f}%")
        print(f"  % Very Large (>1%):  {stats['pct_very_large_gap']:.1f}%")

        # ── Recent Pattern ────────────────────────────────────────────────
        print(f"\n  Recent 10-Day Gap Pattern:")
        print(f"  {'─' * 55}")
        print(f"  Recent Mean Gap:     {stats['recent_mean_gap']:+.3f}%")
        print(f"  Recent Gap Up %:     {stats['recent_up_pct']:.0f}%")
        consec = stats['consecutive_direction']
        c_label = f"{abs(consec)} consecutive {'UP' if consec > 0 else 'DOWN'}" if consec != 0 else "Mixed"
        print(f"  Consecutive Streak:  {c_label}")

        # ── Gap by Volatility Regime ──────────────────────────────────────
        regime_gaps = gap_data['regime_gaps']
        if regime_gaps:
            print(f"\n  Gap Size by Volatility Regime:")
            print(f"  {'─' * 55}")
            print(f"  {'Regime':<16} | {'Mean':<8} | {'|Gap|':<8} | {'%Up':<6} | {'%Large':<8} | N")
            print(f"  {'─'*16}─┼─{'─'*8}─┼─{'─'*8}─┼─{'─'*6}─┼─{'─'*8}─┼─{'─'*4}")
            for bucket, data in regime_gaps.items():
                marker = " <<" if bucket == gap_pred['current_vol_bucket'] else ""
                print(f"  {bucket:<16} | {data['mean']:+.3f}% | {data['mean_abs']:.3f}% | {data['pct_up']:>4.0f}% | {data['pct_large']:>6.0f}% | {data['count']}{marker}")

        # ── Day-of-Week Seasonality ───────────────────────────────────────
        dow_gaps = gap_data['dow_gaps']
        if dow_gaps:
            print(f"\n  Day-of-Week Gap Seasonality:")
            print(f"  {'─' * 55}")
            print(f"  {'Day':<12} | {'Mean Gap':<10} | {'|Gap|':<8} | {'%Up':<6} | N")
            print(f"  {'─'*12}─┼─{'─'*10}─┼─{'─'*8}─┼─{'─'*6}─┼─{'─'*4}")
            for day in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']:
                if day in dow_gaps:
                    d = dow_gaps[day]
                    marker = " <<" if day == gap_pred['tomorrow'] else ""
                    print(f"  {day:<12} | {d['mean']:+.4f}% | {d['mean_abs']:.4f}% | {d['pct_up']:>4.0f}% | {d['count']}{marker}")

        # ── PREDICTION ────────────────────────────────────────────────────
        d = gap_pred['direction']
        arrow = "▲" if 'UP' in d else "▼" if 'DOWN' in d else "━"

        print(f"\n  {'═' * 60}")
        print(f"  {arrow}  NEXT-DAY GAP PREDICTION ({gap_pred['tomorrow']})  {arrow}")
        print(f"  {'═' * 60}")
        print(f"  Direction:      {d} (score: {gap_pred['composite_score']:+.1f})")
        print(f"  Confidence:     {gap_pred['confidence']:.0%}")
        print(f"  Vol Bucket:     {gap_pred['current_vol_bucket']}")
        print(f"  Expected |Gap|: {gap_pred['expected_gap_pct']:.3f}%  ({gap_pred['expected_gap_pts']:.0f} pts)")
        print()

        # Predicted open range
        up_open = spot + gap_pred['expected_gap_pts']
        dn_open = spot - gap_pred['expected_gap_pts']
        print(f"  Likely Open Range:  [{dn_open:,.0f}] ─── [{spot:,.0f}] ─── [{up_open:,.0f}]")
        print()

        # Signal breakdown
        print(f"  Signal Breakdown:")
        for k, v in gap_pred['sub_scores'].items():
            bar_pos = int(max(0, min(20, (v + 50) / 5)))
            bar = '░' * bar_pos + '█' + '░' * (20 - bar_pos)
            label = k.replace('_', ' ').title()
            print(f"    {label:<18s} {bar} {v:>+6.1f}")
        print(f"  {'═' * 60}")

    def gap_analysis(self):
        """
        Standalone gap analysis menu option.
        Fetches data and runs gap analysis + prediction without full RV analysis.
        """
        print("\n" + "=" * 70)
        print("  OVERNIGHT GAP ANALYZER")
        print("=" * 70)

        # Fetch data
        print("\nFetching 1 Year of Daily OHLC Data...")
        df = self._fetch_daily_history(365)
        if df.empty or len(df) < 40:
            print("Not enough data. Aborting.")
            return
        print(f"Loaded {len(df)} trading days.")

        spot = self._get_spot()
        print(f"Spot: {spot:,.2f}")

        # Quick RV/HV for regime context
        closes = df['close']
        rv_5d = self.rv_estimators.close_to_close(closes, 5)
        rv_20d = self.rv_estimators.close_to_close(closes, 20)
        hv_series = self.analytics.calculate_rolling_historical_volatility(closes, 20)
        cur_rv = rv_20d.iloc[-1] if len(rv_20d) > 0 else 0
        cur_hv = hv_series.iloc[-1] if len(hv_series) > 0 else 0
        hv_pctile = (hv_series < cur_hv).mean() * 100

        # RV Trend
        cur_rv_5 = rv_5d.iloc[-1] if len(rv_5d) > 0 else 0
        rv_trend = 'STABLE'
        if cur_rv_5 > 0 and cur_rv > 0:
            ratio = cur_rv_5 / cur_rv
            if ratio > 1.3: rv_trend = 'ACCELERATING'
            elif ratio < 0.7: rv_trend = 'DECELERATING'

        # IV (quick — prompt or manual)
        expiry = input("Enter Expiry for IV (YYYY-MM-DD) or Enter for manual: ").strip()
        iv = 0
        if expiry:
            iv = self._fetch_atm_iv(expiry)
        if iv <= 0:
            try: iv = float(input("Enter Manual ATM IV (%): ") or "15")
            except: iv = 15.0

        regime = self._classify_regime(iv, cur_hv, cur_rv, rv_trend, hv_pctile)

        # Gap analysis
        gap_data = self._analyze_gaps(df)
        if gap_data is None:
            print("Not enough gap data.")
            return

        gap_pred = self._predict_next_gap(gap_data, iv, cur_hv, cur_rv, rv_trend, regime, spot)
        self._print_gap_prediction(gap_pred, gap_data, spot)

        # ── Gap Distribution Plot ─────────────────────────────────────────
        try:
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            fig.suptitle(f'Overnight Gap Analysis — {self.symbol}', fontsize=13, fontweight='bold')

            # Panel 1: Gap distribution histogram
            gap_vals = gap_data['gap_series'].values
            axes[0].hist(gap_vals, bins=40, color='steelblue', alpha=0.7, edgecolor='white', density=True)
            axes[0].axvline(0, color='black', linewidth=1)
            axes[0].axvline(gap_data['gap_series'].mean(), color='red', linestyle='--',
                           label=f'Mean: {gap_data["gap_series"].mean():.3f}%')
            # Mark +-1 std
            std = gap_data['gap_series'].std()
            axes[0].axvline(-std, color='orange', linestyle=':', alpha=0.7)
            axes[0].axvline(std, color='orange', linestyle=':', alpha=0.7, label=f'+/-1 Std: {std:.3f}%')
            axes[0].set_title('Gap Distribution (1 Year)', fontweight='bold')
            axes[0].set_xlabel('Gap %')
            axes[0].set_ylabel('Density')
            axes[0].legend(fontsize=8)
            axes[0].grid(True, alpha=0.3)

            # Panel 2: Rolling gap magnitude
            rolling_abs_gap = gap_data['gap_series'].abs().rolling(10).mean()
            axes[1].plot(rolling_abs_gap.values, color='#e74c3c', linewidth=1.5)
            axes[1].set_title('Rolling 10d Avg |Gap| %', fontweight='bold')
            axes[1].set_xlabel('Trading Days')
            axes[1].set_ylabel('Avg Absolute Gap %')
            axes[1].grid(True, alpha=0.3)

            # Panel 3: Recent gaps bar chart (last 20)
            recent_20 = gap_data['gap_series'].tail(20)
            colors = ['#27ae60' if g > 0 else '#e74c3c' for g in recent_20.values]
            axes[2].bar(range(len(recent_20)), recent_20.values, color=colors, edgecolor='white')
            axes[2].axhline(0, color='black', linewidth=0.5)
            axes[2].set_title('Last 20 Overnight Gaps', fontweight='bold')
            axes[2].set_xlabel('Recent Days')
            axes[2].set_ylabel('Gap %')
            axes[2].grid(True, alpha=0.3)

            plt.tight_layout()
            print("\nOpening Gap Analysis Dashboard...")
            plt.show()
        except Exception as e:
            print(f"Plot error: {e}")

        input("\nPress Enter to continue...")


    # ══════════════════════════════════════════════════════════════════════
    # PRICE PREDICTOR
    # ══════════════════════════════════════════════════════════════════════

    def _predict_price(self, spot, iv, hv, rv, rv_trend, regime,
                       hv_pctile, closes, hv_series, rv_5d, rv_20d):
        """
        Synthesize IV/HV/RV into a price direction and expected move.

        Logic:
          1. Expected Daily Move = RV / sqrt(252) → scale to 5-day horizon
          2. Direction from: regime bias + RV structure + mean-reversion signals
          3. Probability from: historical bucket analysis
        """
        s = pd.Series(closes) if not isinstance(closes, pd.Series) else closes

        # ── Expected Move ─────────────────────────────────────────────────
        daily_rv_move = rv / np.sqrt(252)            # Daily % move from RV
        expected_5d_move = daily_rv_move * np.sqrt(5)  # 5-day expected move
        expected_10d_move = daily_rv_move * np.sqrt(10)

        # ── Directional Bias ──────────────────────────────────────────────
        # Score multiple signals on a -100 (bearish) to +100 (bullish) scale

        scores = {}

        # 1. Price momentum (5d vs 20d returns)
        if len(s) >= 21:
            ret_5d  = (s.iloc[-1] / s.iloc[-6] - 1) * 100
            ret_20d = (s.iloc[-1] / s.iloc[-21] - 1) * 100
            # Positive returns → bullish
            scores['momentum'] = max(-100, min(100, ret_5d * 20))

        # 2. VRP Signal: High IV vs RV → market overpricing fear → bullish reversion
        vrp = iv - rv
        if vrp > 5:
            scores['vrp'] = 40     # Overpriced fear → lean bullish
        elif vrp > 2:
            scores['vrp'] = 20
        elif vrp < -3:
            scores['vrp'] = -40    # Underpriced → lean bearish (real risk)
        elif vrp < -1:
            scores['vrp'] = -20
        else:
            scores['vrp'] = 0

        # 3. RV Trend
        if rv_trend == 'ACCELERATING':
            scores['rv_trend'] = -30  # Rising vol → uncertainty
        elif rv_trend == 'DECELERATING':
            scores['rv_trend'] = 20   # Calming vol → stabilizing
        else:
            scores['rv_trend'] = 0

        # 4. HV Mean Reversion
        hv_z = (hv - hv_series.mean()) / hv_series.std() if hv_series.std() > 0 else 0
        if hv_z > 1.5:
            scores['hv_mean_rev'] = 20   # High HV → expect compression → calmer → bullish
        elif hv_z < -1.5:
            scores['hv_mean_rev'] = -20  # Low HV → expect expansion → storms → bearish
        else:
            scores['hv_mean_rev'] = 0

        # 5. Regime override
        if regime['bias'] == 'BEARISH':
            scores['regime'] = -35
        elif regime['bias'] == 'TREND':
            # Use momentum direction
            scores['regime'] = scores.get('momentum', 0) * 0.3
        else:
            scores['regime'] = 0

        # Composite
        weights = {
            'momentum': 0.30,
            'vrp': 0.25,
            'rv_trend': 0.20,
            'hv_mean_rev': 0.10,
            'regime': 0.15
        }

        total_score = 0
        total_weight = 0
        for k, w in weights.items():
            if k in scores:
                total_score += scores[k] * w
                total_weight += w

        if total_weight > 0:
            composite = total_score / total_weight
        else:
            composite = 0

        # Direction
        if composite > 15:
            direction = "BULLISH"
        elif composite < -15:
            direction = "BEARISH"
        else:
            direction = "NEUTRAL"

        # Confidence
        confidence = min(0.95, abs(composite) / 100 + regime['confidence'] * 0.3)

        # ── Historical Probability (bucket analysis) ──────────────────────
        prob_up = 50.0
        fwd_5d_move = (s.shift(-5) / s - 1) * 100
        log_rets = np.log(s / s.shift(1)).dropna()
        rolling_rv = log_rets.rolling(5).std() * np.sqrt(252) * 100

        aligned = pd.DataFrame({
            'rv': rolling_rv,
            'fwd': fwd_5d_move
        }).dropna()

        if len(aligned) > 30:
            # Find bucket where current RV falls
            try:
                aligned['bucket'] = pd.qcut(aligned['rv'], 5,
                    labels=['VLow', 'Low', 'Mid', 'High', 'VHigh'], duplicates='drop')
                rv_breaks = aligned['rv'].quantile([0, 0.2, 0.4, 0.6, 0.8, 1.0]).values

                cur_rv_5_val = rv_5d.iloc[-1] if len(rv_5d) > 0 else rv
                if cur_rv_5_val <= rv_breaks[1]:   bucket = 'VLow'
                elif cur_rv_5_val <= rv_breaks[2]: bucket = 'Low'
                elif cur_rv_5_val <= rv_breaks[3]: bucket = 'Mid'
                elif cur_rv_5_val <= rv_breaks[4]: bucket = 'High'
                else:                              bucket = 'VHigh'

                bucket_moves = aligned[aligned['bucket'] == bucket]['fwd']
                if len(bucket_moves) > 5:
                    prob_up = (bucket_moves > 0).mean() * 100
            except Exception:
                pass

        return {
            'direction': direction,
            'composite_score': round(composite, 1),
            'confidence': round(confidence, 2),
            'expected_5d_move_pct': round(expected_5d_move, 2),
            'expected_10d_move_pct': round(expected_10d_move, 2),
            'expected_5d_pts': round(spot * expected_5d_move / 100, 0),
            'expected_10d_pts': round(spot * expected_10d_move / 100, 0),
            'prob_up_5d': round(prob_up, 1),
            'regime': regime['name'],
            'vol_action': regime['vol_action'],
            'sub_scores': scores
        }

    # ══════════════════════════════════════════════════════════════════════
    # OUTPUT
    # ══════════════════════════════════════════════════════════════════════

    def _print_prediction(self, pred, spot):
        """Rich console output for the price prediction."""
        d = pred['direction']
        arrow = "▲" if d == "BULLISH" else "▼" if d == "BEARISH" else "◆"
        color_label = d

        print(f"\n{'═' * 65}")
        print(f"  {arrow}  PRICE PREDICTION  {arrow}")
        print(f"{'═' * 65}")
        print(f"  Direction:    {d} (score: {pred['composite_score']:+.1f})")
        print(f"  Confidence:   {pred['confidence']:.0%}")
        print(f"  Regime:       {pred['regime']} → {pred['vol_action']}")
        print()
        print(f"  Expected Move (from RV):")
        print(f"    5-Day:  ±{pred['expected_5d_move_pct']:.2f}%  (±{pred['expected_5d_pts']:.0f} pts)")
        print(f"   10-Day:  ±{pred['expected_10d_move_pct']:.2f}%  (±{pred['expected_10d_pts']:.0f} pts)")
        print()

        # Predicted ranges
        up_5   = spot + pred['expected_5d_pts']
        down_5 = spot - pred['expected_5d_pts']
        up_10   = spot + pred['expected_10d_pts']
        down_10 = spot - pred['expected_10d_pts']

        print(f"  5-Day Range:   [{down_5:,.0f}] ─── [{spot:,.0f}] ─── [{up_5:,.0f}]")
        print(f"  10-Day Range:  [{down_10:,.0f}] ─── [{spot:,.0f}] ─── [{up_10:,.0f}]")
        print()
        print(f"  Historical P(Up in 5d):  {pred['prob_up_5d']:.1f}%")
        print()

        # Sub-score breakdown
        print(f"  Signal Breakdown:")
        for k, v in pred['sub_scores'].items():
            bar_pos = int(max(0, min(20, (v + 100) / 10)))
            bar = "░" * bar_pos + "█" + "░" * (20 - bar_pos)
            label = k.replace('_', ' ').title()
            print(f"    {label:<15s} {bar} {v:>+6.1f}")
        print(f"{'═' * 65}")

    # ══════════════════════════════════════════════════════════════════════
    # PERSISTENCE
    # ══════════════════════════════════════════════════════════════════════

    def _save_snapshot(self, spot, iv, hv, rv, regime, prediction):
        """Save today's analysis to a JSON memory file."""
        mem_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'rv_engine_memory.json')
        today_str = datetime.now().strftime("%Y-%m-%d")

        memory = {}
        try:
            if os.path.exists(mem_file):
                with open(mem_file, 'r') as f:
                    memory = json.load(f)
        except Exception:
            memory = {}

        memory[today_str] = {
            'spot': round(spot, 2),
            'iv': round(iv, 2),
            'hv': round(hv, 2),
            'rv': round(rv, 2),
            'vrp_iv_rv': round(iv - rv, 2),
            'vrp_iv_hv': round(iv - hv, 2),
            'regime': regime['name'],
            'prediction': prediction['direction'],
            'confidence': prediction['confidence'],
            'expected_5d_move': prediction['expected_5d_move_pct'],
            'prob_up': prediction['prob_up_5d']
        }

        # Keep last 90 entries
        if len(memory) > 90:
            for old_key in sorted(memory.keys())[:-90]:
                del memory[old_key]

        try:
            with open(mem_file, 'w') as f:
                json.dump(memory, f, indent=2)
            print(f"\n[Memory] Saved to {mem_file}")
        except Exception as e:
            print(f"[Memory] Save failed: {e}")

        # Show recent history
        sorted_dates = sorted(memory.keys(), reverse=True)
        if len(sorted_dates) > 1:
            print(f"\n  Recent History:")
            print(f"  {'Date':<12} │ {'IV':<6} │ {'HV':<6} │ {'RV':<6} │ {'VRP':<6} │ {'Regime':<14} │ {'Prediction'}")
            print(f"  {'─'*12}─┼─{'─'*6}─┼─{'─'*6}─┼─{'─'*6}─┼─{'─'*6}─┼─{'─'*14}─┼─{'─'*10}")
            for d in sorted_dates[:7]:
                m = memory[d]
                print(f"  {d:<12} │ {m['iv']:<6.1f} │ {m['hv']:<6.1f} │ {m['rv']:<6.1f} │ {m['vrp_iv_rv']:<+5.1f} │ {m['regime']:<14} │ {m['prediction']}")

    # ══════════════════════════════════════════════════════════════════════
    # DASHBOARD PLOT
    # ══════════════════════════════════════════════════════════════════════

    def _plot_dashboard(self, df, rv_5d, rv_10d, rv_20d, hv_series,
                        atm_iv, consensus_rv, cur_hv, regime, prediction, spot):
        """4-panel dashboard: RV multi-window, IV/HV/RV triangle, prediction gauge, RV vs price."""
        try:
            fig = plt.figure(figsize=(18, 10))
            fig.suptitle(
                f'Realized Volatility Engine — {self.symbol} | Regime: {regime["name"]}',
                fontsize=14, fontweight='bold', y=0.98
            )
            gs = gridspec.GridSpec(2, 3, hspace=0.35, wspace=0.30)

            # ── Panel 1: Multi-Window RV (top-left, spans 2 cols) ─────────
            ax1 = fig.add_subplot(gs[0, :2])
            ax1.plot(rv_5d.values, label='RV 5d', color='#e74c3c', linewidth=1.5, alpha=0.8)
            ax1.plot(rv_10d.values, label='RV 10d', color='#e67e22', linewidth=1.5, alpha=0.8)
            ax1.plot(rv_20d.values, label='RV 20d', color='#2980b9', linewidth=2)
            ax1.plot(hv_series.values, label='HV 20d', color='#27ae60', linewidth=1.5, linestyle='--')
            ax1.axhline(atm_iv, color='#8e44ad', linewidth=2, linestyle='-', label=f'ATM IV ({atm_iv:.1f}%)')
            ax1.fill_between(range(len(rv_20d)), rv_20d.values, atm_iv,
                             where=rv_20d.values < atm_iv, alpha=0.1, color='green', label='IV > RV (overpriced)')
            ax1.fill_between(range(len(rv_20d)), rv_20d.values, atm_iv,
                             where=rv_20d.values > atm_iv, alpha=0.1, color='red', label='RV > IV (underpriced)')
            ax1.set_title('Multi-Window RV vs HV vs IV', fontweight='bold')
            ax1.set_ylabel('Annualized Vol (%)')
            ax1.set_xlabel('Trading Days')
            ax1.legend(fontsize=7, loc='upper left', ncol=2)
            ax1.grid(True, alpha=0.3)

            # ── Panel 2: Volatility Triangle (top-right) ─────────────────
            ax2 = fig.add_subplot(gs[0, 2])
            categories = ['IV', 'HV', 'RV']
            values = [atm_iv, cur_hv, consensus_rv]
            colors = ['#8e44ad', '#27ae60', '#2980b9']
            bars = ax2.barh(categories, values, color=colors, height=0.5, edgecolor='white')
            for bar, val in zip(bars, values):
                ax2.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                         f'{val:.1f}%', va='center', fontweight='bold', fontsize=11)
            ax2.set_title('Volatility Triangle', fontweight='bold')
            ax2.set_xlabel('Annualized Vol (%)')
            ax2.set_xlim(0, max(values) * 1.3)

            # Add VRP annotations
            vrp_text = f"VRP (IV-RV): {atm_iv - consensus_rv:+.1f}%\nVRP (IV-HV): {atm_iv - cur_hv:+.1f}%"
            ax2.text(0.95, 0.05, vrp_text, transform=ax2.transAxes, fontsize=9,
                     ha='right', va='bottom', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

            # ── Panel 3: Price Prediction Gauge (bottom-left) ────────────
            ax3 = fig.add_subplot(gs[1, 0])
            ax3.set_xlim(-100, 100)
            ax3.set_ylim(0, 1)
            ax3.barh(0.5, prediction['composite_score'], height=0.3,
                     color='#27ae60' if prediction['composite_score'] > 0 else '#e74c3c',
                     edgecolor='white')
            ax3.axvline(0, color='black', linewidth=1)
            ax3.set_title(f'Direction Gauge: {prediction["direction"]}', fontweight='bold')
            ax3.set_xlabel('← BEARISH        Score        BULLISH →')
            ax3.set_yticks([])

            # Add text
            ax3.text(0, 0.15,
                     f'5d Range: ±{prediction["expected_5d_move_pct"]:.1f}% (±{prediction["expected_5d_pts"]:.0f}pts)\n'
                     f'P(Up 5d): {prediction["prob_up_5d"]:.0f}% | Confidence: {prediction["confidence"]:.0%}',
                     transform=ax3.transAxes, fontsize=9, va='bottom')

            # ── Panel 4: RV overlaid on Price (bottom-center+right) ──────
            ax4 = fig.add_subplot(gs[1, 1:])
            price_vals = df['close'].values
            ax4_price = ax4
            ax4_price.plot(price_vals, color='#2c3e50', linewidth=1.5, label='Close Price')
            ax4_price.set_ylabel('Price', color='#2c3e50')
            ax4_price.set_xlabel('Trading Days')

            ax4_rv = ax4_price.twinx()
            # Align RV series to last N values of price
            rv_plot = rv_20d.values
            offset = len(price_vals) - len(rv_plot)
            ax4_rv.plot(range(offset, offset + len(rv_plot)), rv_plot,
                        color='#e74c3c', linewidth=1.5, alpha=0.7, label='RV 20d')
            ax4_rv.axhline(atm_iv, color='#8e44ad', linestyle='--', alpha=0.5, label=f'IV {atm_iv:.1f}%')
            ax4_rv.set_ylabel('Volatility (%)', color='#e74c3c')

            ax4_price.set_title('Price Action with RV Overlay', fontweight='bold')
            lines1, labels1 = ax4_price.get_legend_handles_labels()
            lines2, labels2 = ax4_rv.get_legend_handles_labels()
            ax4_rv.legend(lines1 + lines2, labels1 + labels2, fontsize=7, loc='upper left')
            ax4_price.grid(True, alpha=0.2)

            plt.tight_layout(rect=[0, 0, 1, 0.96])
            print("\nOpening RV Engine Dashboard...")
            plt.show()

        except Exception as e:
            print(f"Plot error: {e}")

    # ══════════════════════════════════════════════════════════════════════
    # MENU
    # ══════════════════════════════════════════════════════════════════════

    def run(self):
        """Interactive menu loop."""
        while True:
            print("\n" + "─" * 50)
            print("  REALIZED VOLATILITY ENGINE")
            print("─" * 50)
            print(f"  Symbol: {self.symbol}")
            print("  1. Run Full Analysis (RV + IV/HV Comparison + Prediction + Gaps)")
            print("  2. Gap Analysis Only (Overnight Gap Prediction)")
            print("  3. Change Symbol")
            print("  4. Exit")
            print("─" * 50)

            c = input("  Select: ").strip()

            if c == '1':
                self.run_analysis()
            elif c == '2':
                self.gap_analysis()
            elif c == '3':
                self.symbol = input("  Enter Symbol (e.g., NSE:NIFTY50-INDEX): ").strip() or self.symbol
            elif c == '4':
                print("  Exiting RV Engine.")
                break


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    engine = RealizedVolEngine()
    engine.run()
