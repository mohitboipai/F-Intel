"""
OptionPriceReconstructor.py
===========================
Reconstructs theoretical intraday option prices using Black-Scholes
when only EOD bhavcopy data is available.

IV priority:
  1. ATM IV from BhavCopyEngine.get_atm_iv(date)
  2. 20-day historical volatility from minute closes
  3. Hard default of 15.0%

Usage:
    recon = OptionPriceReconstructor(bhav_engine)
    price = recon.price(spot, strike, dte_years, iv_pct, 'CE')

    net = recon.reconstruct_strategy_prices(
        legs   = [{'strike': 24000, 'option_type': 'CE', 'action': 'SELL'},
                  {'strike': 24000, 'option_type': 'PE', 'action': 'SELL'}],
        spot   = 24050.0,
        dte    = 5 / 365.0,
        iv_pct = 14.5,
    )
    # net is ₹/lot (lot=65)
"""

import os
import sys
import numpy as np
import pandas as pd
from datetime import date as date_type

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from OptionAnalytics import OptionAnalytics

# ── constants ─────────────────────────────────────────────────────────────────
NIFTY_LOT    = 65      # current lot size (was 75)
RISK_FREE    = 0.07
DEFAULT_IV   = 15.0    # % fallback

_oa = OptionAnalytics()


def _round50(x: float) -> float:
    return round(x / 50) * 50


# ── OptionPriceReconstructor ──────────────────────────────────────────────────
class OptionPriceReconstructor:
    """
    Reconstructs theoretical option prices at arbitrary intraday timestamps
    using Black-Scholes with a cached IV estimate per date.
    """

    def __init__(self, bhav_engine=None):
        """
        bhav_engine: BhavCopyEngine instance (optional, used for ATM IV lookup).
        """
        self._bhav   = bhav_engine
        self._iv_cache: dict[str, float] = {}    # date_str → IV %

    # ── IV resolution ────────────────────────────────────────────────────────

    def get_iv_for_date(self, date, daily_closes: pd.Series | None = None) -> float:
        """
        Return IV % for `date` using the 3-priority fallback chain.
        Caches the result per date.

        Priority:
          1. ATM IV from BhavCopyEngine (if available)
          2. 20-day HV from `daily_closes` (passed in from MinuteDataFetcher)
          3. DEFAULT_IV (15.0%)
        """
        if isinstance(date, date_type):
            date_str = date.strftime('%Y-%m-%d')
        else:
            date_str = str(date)[:10]

        if date_str in self._iv_cache:
            return self._iv_cache[date_str]

        iv = None

        # 1. ATM IV from bhavcopy
        if self._bhav is not None:
            try:
                iv = self._bhav.get_atm_iv(date_str)
            except Exception:
                iv = None

        # 2. 20-day HV from daily closes
        if (iv is None or iv <= 0) and daily_closes is not None and not daily_closes.empty:
            try:
                # Only use closes up to and including this date
                closes_to_date = daily_closes[daily_closes.index <= pd.Timestamp(date_str)]
                if len(closes_to_date) >= 21:
                    log_ret = np.log(closes_to_date / closes_to_date.shift(1)).dropna()
                    hv = log_ret.rolling(20).std().iloc[-1] * np.sqrt(252) * 100
                    if not np.isnan(hv) and hv > 0:
                        iv = float(hv)
            except Exception:
                iv = None

        # 3. Hard default
        if iv is None or iv <= 0:
            iv = DEFAULT_IV

        self._iv_cache[date_str] = iv
        return iv

    def warm_iv_cache(self, dates: list, daily_closes: pd.Series | None = None):
        """Pre-populate IV cache for a list of dates."""
        for d in dates:
            self.get_iv_for_date(d, daily_closes)

    # ── pricing ──────────────────────────────────────────────────────────────

    def price(self, spot: float, strike: float, dte_years: float,
              iv_pct: float, option_type: str) -> float:
        """
        Compute a single theoretical option price via Black-Scholes.

        Parameters
        ----------
        spot        : current underlying price
        strike      : option strike
        dte_years   : time to expiry in years (e.g. 5/365)
        iv_pct      : implied / historical volatility in % (e.g. 14.5)
        option_type : 'CE' or 'PE'

        Returns
        -------
        Theoretical price (float), minimum 0.05
        """
        sigma = max(iv_pct / 100.0, 1e-4)
        T     = max(dte_years, 1e-5)
        price = _oa.black_scholes(spot, strike, T, RISK_FREE, sigma, option_type.upper())
        return max(float(price), 0.05)

    def reconstruct_strategy_prices(
        self,
        legs:      list[dict],
        spot:      float,
        dte:       float,
        iv_pct:    float,
    ) -> float:
        """
        Compute net premium for a multi-leg strategy in ₹ per lot.

        Each leg dict:
            strike      : float
            option_type : 'CE' or 'PE'
            action      : 'SELL' (positive contribution) or 'BUY' (negative)

        Returns net premium in ₹ (lot=65).
        Positive  = net credit received (sell strategy).
        Negative  = net debit paid (buy strategy).
        """
        net_per_unit = 0.0
        for leg in legs:
            p = self.price(
                spot        = spot,
                strike      = float(leg['strike']),
                dte_years   = dte,
                iv_pct      = iv_pct,
                option_type = leg['option_type'],
            )
            if leg['action'].upper() == 'SELL':
                net_per_unit += p      # receive premium
            else:
                net_per_unit -= p      # pay premium

        return net_per_unit * NIFTY_LOT

    def mark_to_market(
        self,
        legs:        list[dict],
        entry_prices: list[float],
        spot:         float,
        dte:          float,
        iv_pct:       float,
    ) -> float:
        """
        Compute current mark-to-market P&L relative to entry prices.
        Returns P&L in ₹ per lot.

        legs         : same list of dicts as reconstruct_strategy_prices
        entry_prices : list of entry prices (parallel to legs)
        spot         : current spot
        dte          : remaining DTE in years
        iv_pct       : current IV estimate %

        For a SELL leg: P&L = (entry_price - current_price) × lot
        For a BUY  leg: P&L = (current_price - entry_price) × lot
        """
        pnl_per_unit = 0.0
        for leg, entry in zip(legs, entry_prices):
            current = self.price(
                spot        = spot,
                strike      = float(leg['strike']),
                dte_years   = dte,
                iv_pct      = iv_pct,
                option_type = leg['option_type'],
            )
            if leg['action'].upper() == 'SELL':
                pnl_per_unit += (entry - current)   # theta decay benefits seller
            else:
                pnl_per_unit += (current - entry)   # buyer profits from appreciation

        return pnl_per_unit * NIFTY_LOT

    def expiry_pnl(
        self,
        legs:         list[dict],
        entry_prices: list[float],
        spot_at_expiry: float,
    ) -> float:
        """
        P&L at expiry using intrinsic value only. Returns ₹ per lot.
        """
        pnl_per_unit = 0.0
        for leg, entry in zip(legs, entry_prices):
            strike = float(leg['strike'])
            if leg['option_type'].upper() == 'CE':
                intrinsic = max(0.0, spot_at_expiry - strike)
            else:
                intrinsic = max(0.0, strike - spot_at_expiry)

            if leg['action'].upper() == 'SELL':
                pnl_per_unit += (entry - intrinsic)
            else:
                pnl_per_unit += (intrinsic - entry)

        return pnl_per_unit * NIFTY_LOT
