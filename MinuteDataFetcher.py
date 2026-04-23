"""
MinuteDataFetcher.py
====================
Fetches 1-minute OHLCV data for NSE:NIFTY50-INDEX from Fyers API.

Features:
  - Splits date range into 90-day chunks (Fyers limit)
  - Converts UTC epoch → IST (Asia/Kolkata = UTC+5:30, no pytz needed)
  - Filters to market hours 09:15 – 15:30 IST
  - Saves/loads one Parquet file per calendar year:
      data/minute/NIFTY_1min_YYYY.parquet

Usage:
    from MinuteDataFetcher import MinuteDataFetcher
    fetcher = MinuteDataFetcher()
    df = fetcher.get(start_date, end_date)
    # df.index is DatetimeIndex IST, columns: open high low close volume
"""

import os
import sys
import time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from FyersAuth import FyersAuthenticator

# ── constants ─────────────────────────────────────────────────────────────────
SAVE_DIR    = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'minute')
SYMBOL      = 'NSE:NIFTY50-INDEX'
RESOLUTION  = '1'              # 1-minute bars
CHUNK_DAYS  = 90               # Fyers max per request
IST_OFFSET  = timedelta(hours=5, minutes=30)   # UTC → IST without pytz
MARKET_OPEN = '09:15'
MARKET_CLOSE = '15:30'

# Fyers credentials — same pattern as OIBacktester
_APP_ID   = 'QUTT4YYMIG-100'
_SECRET   = 'ZG0WN2NL1B'
_REDIR    = 'http://127.0.0.1:3000/callback'


# ── MinuteDataFetcher ─────────────────────────────────────────────────────────
class MinuteDataFetcher:
    """
    Fetches and caches 1-minute NIFTY data from Fyers API.
    One parquet file per calendar year under data/minute/.
    """

    def __init__(self):
        os.makedirs(SAVE_DIR, exist_ok=True)
        self._fyers = None      # lazy-initialised on first network call

    # ── auth ────────────────────────────────────────────────────────────────

    def _get_fyers(self):
        if self._fyers is None:
            auth = FyersAuthenticator(_APP_ID, _SECRET, _REDIR)
            self._fyers = auth.get_fyers_instance()
        return self._fyers

    # ── parquet helpers ─────────────────────────────────────────────────────

    def _parquet_path(self, year: int) -> str:
        return os.path.join(SAVE_DIR, f'NIFTY_1min_{year}.parquet')

    def _load_parquet(self, year: int) -> pd.DataFrame | None:
        path = self._parquet_path(year)
        if os.path.exists(path):
            try:
                df = pd.read_parquet(path)
                return df
            except Exception as exc:
                print(f'[MinuteDataFetcher] Parquet load error ({year}): {exc}')
        return None

    def _save_parquet(self, df: pd.DataFrame, year: int):
        path = self._parquet_path(year)
        try:
            df.to_parquet(path)
        except Exception as exc:
            print(f'[MinuteDataFetcher] Parquet save error ({year}): {exc}')

    # ── core fetch ──────────────────────────────────────────────────────────

    def _fetch_chunk(self, from_dt: datetime, to_dt: datetime) -> pd.DataFrame:
        """Fetch a single chunk (≤90 days) from Fyers."""
        fyers = self._get_fyers()
        if fyers is None:
            return pd.DataFrame()

        data = {
            'symbol':       SYMBOL,
            'resolution':   RESOLUTION,
            'date_format':  '1',
            'range_from':   from_dt.strftime('%Y-%m-%d'),
            'range_to':     to_dt.strftime('%Y-%m-%d'),
            'cont_flag':    '1',
        }

        try:
            resp = fyers.history(data=data)
            if resp.get('s') != 'ok' or not resp.get('candles'):
                return pd.DataFrame()

            candles = resp['candles']
            df = pd.DataFrame(candles, columns=['ts', 'open', 'high', 'low', 'close', 'volume'])

            # Convert epoch (UTC seconds) → IST DatetimeIndex
            df['datetime'] = pd.to_datetime(df['ts'], unit='s') + IST_OFFSET
            df = df.set_index('datetime').drop(columns=['ts'])
            df.index.name = 'datetime'

            # Filter to market hours
            df = df.between_time(MARKET_OPEN, MARKET_CLOSE)

            return df[['open', 'high', 'low', 'close', 'volume']]

        except Exception as exc:
            print(f'[MinuteDataFetcher] Chunk fetch error: {exc}')
            return pd.DataFrame()

    def _fetch_range(self, start_dt: datetime, end_dt: datetime) -> pd.DataFrame:
        """Fetch full date range by splitting into 90-day chunks."""
        frames = []
        cur = start_dt
        while cur < end_dt:
            chunk_end = min(cur + timedelta(days=CHUNK_DAYS - 1), end_dt)
            print(f'  [Minute] Fetching {cur.date()} → {chunk_end.date()} …')
            chunk = self._fetch_chunk(cur, chunk_end)
            if not chunk.empty:
                frames.append(chunk)
            cur = chunk_end + timedelta(days=1)
            time.sleep(0.3)   # polite rate-limit

        if not frames:
            return pd.DataFrame()

        df = pd.concat(frames).sort_index()
        df = df[~df.index.duplicated(keep='first')]
        return df

    # ── public API ──────────────────────────────────────────────────────────

    def get(self, start_date, end_date) -> pd.DataFrame:
        """
        Return 1-min OHLCV DataFrame for [start_date, end_date].
        Loads from disk if parquet exists for the year; fetches otherwise.
        Handles multi-year ranges transparently.

        Returns DataFrame with IST DatetimeIndex and columns:
            open, high, low, close, volume
        """
        if isinstance(start_date, str):
            start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        else:
            start_dt = datetime(start_date.year, start_date.month, start_date.day)

        if isinstance(end_date, str):
            end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        else:
            end_dt = datetime(end_date.year, end_date.month, end_date.day)

        # Group by year so we can load cached files per year
        years = list(range(start_dt.year, end_dt.year + 1))
        frames = []

        for year in years:
            yr_start = max(start_dt, datetime(year, 1, 1))
            yr_end   = min(end_dt,   datetime(year, 12, 31))
            parquet_path = self._parquet_path(year)

            if os.path.exists(parquet_path):
                print(f'  [Minute] Loading from cache: {parquet_path}')
                cached = self._load_parquet(year)
                if cached is not None and not cached.empty:
                    # Slice to requested range
                    mask = (cached.index >= pd.Timestamp(yr_start)) & \
                           (cached.index <= pd.Timestamp(yr_end) + timedelta(hours=16))
                    slice_df = cached[mask]
                    if not slice_df.empty:
                        frames.append(slice_df)
                        continue

            # Need to fetch
            year_df = self._fetch_range(yr_start, yr_end)
            if not year_df.empty:
                self._save_parquet(year_df, year)
                mask = (year_df.index >= pd.Timestamp(yr_start)) & \
                       (year_df.index <= pd.Timestamp(yr_end) + timedelta(hours=16))
                frames.append(year_df[mask])

        if not frames:
            return pd.DataFrame()

        result = pd.concat(frames).sort_index()
        result = result[~result.index.duplicated(keep='first')]
        return result

    def get_daily_closes(self, start_date, end_date) -> pd.Series:
        """
        Return daily close prices from the 1-min data (last bar of each day).
        Useful for HV calculation.
        """
        df = self.get(start_date, end_date)
        if df.empty:
            return pd.Series()
        # Last bar of each calendar date
        daily = df.groupby(df.index.date)['close'].last()
        daily.index = pd.to_datetime([str(d) for d in daily.index])
        return daily
