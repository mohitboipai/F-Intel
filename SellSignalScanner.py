"""
SellSignalScanner.py
====================
Scans historical NIFTY minute data for four sell-signal types.

Signal definitions (all evaluated at the 09:30 bar, 15 min after open):

  VRP_SELL       – 5d realised vol < ATM IV − 2pp (IV is rich vs realised)
  IV_PERCENTILE  – ATM IV > 70th percentile of last 60 trading days
  GEX_COMPRESSION– Avg of last 5 daily ranges < 0.8% of spot
  COMBINED       – at least 2 of the 3 above are simultaneously true

Public API:
    scanner = SellSignalScanner(bhav_engine)
    signals_df = scanner.scan_year(minute_df, start_date, end_date)

Returns DataFrame columns:
    date, signal_type, spot_at_signal, atm_iv, hv_20d, signal_time
"""

import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, date as date_type

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── constants ─────────────────────────────────────────────────────────────────
SIGNAL_BAR_TIME  = '09:30'      # fire signal at this bar (15 min after open)
VRP_THRESHOLD    = 2.0          # IV must exceed 5d HV by this many pp
IV_PERCENTILE    = 70           # IV above this percentile → expensive
GEX_RANGE_PCT    = 0.008        # daily range < 0.8% of spot → compression
LOOKBACK_HV      = 5            # days for realised vol
LOOKBACK_IV_HIST = 60           # trading days for IV percentile
LOOKBACK_GEX     = 5            # days for range average


# ── SellSignalScanner ─────────────────────────────────────────────────────────
class SellSignalScanner:

    def __init__(self, bhav_engine):
        """
        bhav_engine: BhavCopyEngine instance (must already be loaded).
        """
        self._bhav = bhav_engine

    # ── main entry point ─────────────────────────────────────────────────────

    def scan_year(
        self,
        minute_df: pd.DataFrame,
        start_date=None,
        end_date=None,
    ) -> pd.DataFrame:
        """
        Scan all trading days in minute_df for sell signals.

        Parameters
        ----------
        minute_df   : DatetimeIndex IST, columns open/high/low/close/volume
        start_date  : optional filter (string or date)
        end_date    : optional filter

        Returns
        -------
        DataFrame with columns:
            date, signal_type, spot_at_signal, atm_iv, hv_20d, signal_time
        """
        if minute_df.empty:
            return pd.DataFrame()

        # Build per-day OHLC from minute data
        daily = self._build_daily(minute_df)

        # Optionally filter range
        if start_date is not None:
            daily = daily[daily.index >= pd.Timestamp(str(start_date)[:10])]
        if end_date is not None:
            daily = daily[daily.index <= pd.Timestamp(str(end_date)[:10])]

        if daily.empty:
            return pd.DataFrame()

        # Pre-compute rolling features
        daily = self._add_features(daily)

        # Scan each day
        events = []
        dates = sorted(daily.index)

        for i, day_ts in enumerate(dates):
            date_str = day_ts.strftime('%Y-%m-%d')

            # Get the 09:30 bar for this date
            signal_bar = self._get_signal_bar(minute_df, day_ts)
            if signal_bar is None:
                continue

            spot = float(signal_bar['close'])
            if spot <= 0:
                continue

            # Read pre-computed feature row
            row = daily.loc[day_ts]

            atm_iv = self._bhav.get_atm_iv(date_str)
            hv_20d = self._bhav.get_hv_20d(date_str)
            rv5    = float(row.get('rv5', np.nan))  if not pd.isna(row.get('rv5', np.nan))  else None
            iv_p60 = float(row.get('iv_p60', np.nan)) if not pd.isna(row.get('iv_p60', np.nan)) else None
            avg_range_pct = float(row.get('avg_range_pct', np.nan)) if not pd.isna(row.get('avg_range_pct', np.nan)) else None

            # ── Signal 1: VRP sell ──────────────────────────────────────────
            vrp_fire = False
            if atm_iv is not None and atm_iv > 0 and rv5 is not None:
                vrp_fire = (rv5 < atm_iv - VRP_THRESHOLD)

            # ── Signal 2: IV percentile ─────────────────────────────────────
            ivp_fire = False
            if atm_iv is not None and iv_p60 is not None:
                ivp_fire = (atm_iv > iv_p60)

            # ── Signal 3: GEX compression ───────────────────────────────────
            gex_fire = False
            if avg_range_pct is not None:
                gex_fire = (avg_range_pct < GEX_RANGE_PCT)

            signals_today: list[str] = []
            if vrp_fire:
                signals_today.append('VRP_SELL')
            if ivp_fire:
                signals_today.append('IV_PERCENTILE')
            if gex_fire:
                signals_today.append('GEX_COMPRESSION')

            n_true = len(signals_today)

            # ── Signal 4: Combined ──────────────────────────────────────────
            if n_true >= 2:
                signals_today.append('COMBINED')

            for sig_type in signals_today:
                events.append({
                    'date':            date_str,
                    'signal_type':     sig_type,
                    'spot_at_signal':  round(spot, 2),
                    'atm_iv':          round(atm_iv, 4) if atm_iv is not None else None,
                    'hv_20d':          round(hv_20d, 4) if hv_20d is not None else None,
                    'rv5':             round(rv5, 4)    if rv5    is not None else None,
                    'signal_time':     f'{date_str} {SIGNAL_BAR_TIME}',
                })

        if not events:
            return pd.DataFrame()

        df = pd.DataFrame(events)
        df = df.sort_values(['date', 'signal_type']).reset_index(drop=True)
        return df

    # ── helpers ──────────────────────────────────────────────────────────────

    def _build_daily(self, minute_df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate 1-min bars to daily OHLC."""
        daily = minute_df.groupby(minute_df.index.date).agg(
            open  =('open',  'first'),
            high  =('high',  'max'),
            low   =('low',   'min'),
            close =('close', 'last'),
        )
        daily.index = pd.to_datetime([str(d) for d in daily.index])
        daily['range'] = daily['high'] - daily['low']
        return daily

    def _add_features(self, daily: pd.DataFrame) -> pd.DataFrame:
        """Add rolling features needed for signal evaluation."""
        closes = daily['close'].copy()

        # 5-day realised vol (annualised %)
        log_ret = np.log(closes / closes.shift(1))
        daily['rv5'] = log_ret.rolling(LOOKBACK_HV).std() * np.sqrt(252) * 100

        # 5-day average daily range as % of spot
        daily['avg_range_pct'] = (
            daily['range'].rolling(LOOKBACK_GEX).mean() / closes
        )

        # ATM IV per date (pull from bhavcopy engine)
        iv_series = {}
        for day_ts in daily.index:
            date_str = day_ts.strftime('%Y-%m-%d')
            iv = self._bhav.get_atm_iv(date_str)
            iv_series[day_ts] = iv if iv is not None else np.nan

        daily['atm_iv_bhav'] = pd.Series(iv_series)

        # 60-day rolling 70th percentile of ATM IV
        daily['iv_p60'] = (
            daily['atm_iv_bhav']
            .rolling(LOOKBACK_IV_HIST, min_periods=20)
            .quantile(IV_PERCENTILE / 100.0)
        )

        return daily

    def _get_signal_bar(self, minute_df: pd.DataFrame, day_ts: pd.Timestamp):
        """Return the 09:30 bar (or nearest bar after 09:30) for `day_ts`."""
        date_str = day_ts.strftime('%Y-%m-%d')
        target   = pd.Timestamp(f'{date_str} {SIGNAL_BAR_TIME}')

        day_bars = minute_df[minute_df.index.date == day_ts.date()]
        if day_bars.empty:
            return None

        # Find bar at or after 09:30
        after = day_bars[day_bars.index >= target]
        if not after.empty:
            return after.iloc[0]

        # Fallback: last bar of the day
        return day_bars.iloc[-1]
