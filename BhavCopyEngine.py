"""
BhavCopyEngine.py
=================
Downloads, parses and indexes NSE F&O bhavcopy data for NIFTY options.

Wraps BhavcopyFetcher + OIDataProcessor from OIBacktester.py (no duplicate
download logic) and exposes a clean interface for the sell-signal pipeline.

NSE URL patterns:
  Old (pre 8-Jul-2024):
    https://nsearchives.nseindia.com/content/historical/DERIVATIVES/
    YYYY/MMM/fo_DDMMMYYYY_bhav.csv.zip
  New (post 8-Jul-2024):
    https://nsearchives.nseindia.com/content/fo/
    BhavCopy_NSE_FO_0_0_0_YYYYMMDD_F_0000.csv.zip

Public API:
    engine = BhavCopyEngine()
    engine.load_range(start_date, end_date)          # download + index
    price  = engine.get_price(date, expiry, strike, option_type)
    expiries = engine.get_expiries(start_date, end_date)
    iv     = engine.get_atm_iv(date)
"""

import os
import io
import sys
import time
import zipfile
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, date as date_type

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from OptionAnalytics import OptionAnalytics

# ── constants ─────────────────────────────────────────────────────────────────
NIFTY_LOT   = 65      # current lot size (was 75 before)
RISK_FREE   = 0.07
SAVE_DIR    = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'bhavcopy')
CUTOVER     = datetime(2024, 7, 8)   # old → new URL format

NSE_HEADERS = {
    'User-Agent': ('Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                   'AppleWebKit/537.36 (KHTML, like Gecko) '
                   'Chrome/120.0.0.0 Safari/537.36'),
    'Accept':          'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.5',
    'Referer':         'https://www.nseindia.com/',
}

_oa = OptionAnalytics()


# ── helpers ───────────────────────────────────────────────────────────────────
def _round50(x: float) -> float:
    return round(x / 50) * 50


def _to_date(d) -> date_type:
    """Coerce str / datetime / date to date."""
    if isinstance(d, date_type):
        return d
    if isinstance(d, datetime):
        return d.date()
    return datetime.strptime(str(d), '%Y-%m-%d').date()


def _build_url_old(dt: datetime) -> str:
    m  = dt.strftime('%b').upper()      # JAN, FEB …
    dd = dt.strftime('%d')              # 01-31
    return (
        f"https://nsearchives.nseindia.com/content/historical/DERIVATIVES/"
        f"{dt.year}/{m}/fo_{dd}{m}{dt.year}bhav.csv.zip"
    )


def _build_url_new(dt: datetime) -> str:
    return (
        f"https://nsearchives.nseindia.com/content/fo/"
        f"BhavCopy_NSE_FO_0_0_0_{dt.strftime('%Y%m%d')}_F_0000.csv.zip"
    )


# ── BhavCopyEngine ────────────────────────────────────────────────────────────
class BhavCopyEngine:
    """
    Downloads NSE F&O bhavcopy files, caches them locally, and provides
    structured access to NIFTY option prices, expiries and ATM IV.
    """

    def __init__(self):
        os.makedirs(SAVE_DIR, exist_ok=True)
        self._session = requests.Session()
        self._session.headers.update(NSE_HEADERS)
        try:
            self._session.get('https://www.nseindia.com/', timeout=10)
        except Exception:
            pass

        # Master index: date_str → DataFrame (expiry_date, strike, option_type,
        #               close, volume, oi, underlying_close)
        self._data: dict[str, pd.DataFrame] = {}
        # IV cache: date_str → float (%) or None
        self._iv_cache: dict[str, float | None] = {}
        # HV cache: date_str → list[float] (daily closes up to that date)
        self._spot_history: list[tuple[str, float]] = []   # [(date_str, close)]

    # ── download ────────────────────────────────────────────────────────────

    def _csv_path(self, dt: datetime) -> str:
        return os.path.join(SAVE_DIR, f"fo_bhav_{dt.strftime('%Y%m%d')}.csv")

    def _fetch_day(self, dt: datetime, verbose: bool = False) -> pd.DataFrame | None:
        """Return raw CSV DataFrame for a single trading day, or None."""
        csv_path = self._csv_path(dt)
        if os.path.exists(csv_path):
            try:
                return pd.read_csv(csv_path, low_memory=False)
            except Exception:
                pass

        url = _build_url_old(dt) if dt < CUTOVER else _build_url_new(dt)
        try:
            r = self._session.get(url, timeout=20)
            if r.status_code == 200 and len(r.content) > 100:
                z = zipfile.ZipFile(io.BytesIO(r.content))
                csv_name = z.namelist()[0]
                with z.open(csv_name) as f:
                    df = pd.read_csv(f, low_memory=False)
                df.to_csv(csv_path, index=False)
                if verbose:
                    print('.', end='', flush=True)
                return df
        except Exception as exc:
            if verbose:
                pass   # silently skip
        return None

    def load_range(self, start_date, end_date, verbose: bool = False) -> int:
        """
        Download & parse bhavcopy for every trading day in [start, end].
        Returns count of successfully loaded days.
        Skips weekends and non-existent (holiday) dates silently.
        """
        s = _to_date(start_date)
        e = _to_date(end_date)
        cur = s
        loaded = 0

        while cur <= e:
            if cur.weekday() >= 5:      # skip weekends
                cur += timedelta(days=1)
                continue

            dt = datetime(cur.year, cur.month, cur.day)
            date_str = cur.strftime('%Y-%m-%d')

            if date_str not in self._data:
                raw = self._fetch_day(dt, verbose=verbose)
                if raw is not None:
                    parsed = self._parse_raw(raw, date_str)
                    if not parsed.empty:
                        self._data[date_str] = parsed
                        self._update_spot_history(date_str, parsed)
                        loaded += 1
                else:
                    if not os.path.exists(self._csv_path(dt)):
                        time.sleep(0.3)   # polite rate limit only when hitting server
            else:
                loaded += 1

            cur += timedelta(days=1)

        return loaded

    # ── parsing ─────────────────────────────────────────────────────────────

    def _parse_raw(self, df: pd.DataFrame, date_str: str) -> pd.DataFrame:
        """Parse raw bhavcopy CSV → standardised NIFTY-only DataFrame."""
        df.columns = [c.strip() for c in df.columns]
        cols = set(df.columns)

        if 'INSTRUMENT' in cols:
            return self._parse_old(df, date_str)
        elif 'TckrSymb' in cols:
            return self._parse_new(df, date_str)
        return pd.DataFrame()

    def _parse_old(self, df: pd.DataFrame, date_str: str) -> pd.DataFrame:
        """Pre-July 2024 format."""
        mask = (
            (df['INSTRUMENT'].astype(str).str.strip() == 'OPTIDX') &
            (df['SYMBOL'].astype(str).str.strip() == 'NIFTY')
        )
        df = df[mask].copy()
        if df.empty:
            return pd.DataFrame()

        df['_ot'] = df['OPTION_TYP'].astype(str).str.strip().str.upper()
        df = df[df['_ot'].isin(['CE', 'PE'])].copy()
        if df.empty:
            return pd.DataFrame()

        # Parse expiry (DDMMMYYYY e.g. 27APR2023)
        expiry_series = pd.to_datetime(
            df['EXPIRY_DT'].astype(str).str.strip(), format='%d-%b-%Y', errors='coerce'
        )
        if expiry_series.isna().all():
            expiry_series = pd.to_datetime(
                df['EXPIRY_DT'].astype(str).str.strip(), format='%d%b%Y', errors='coerce'
            )

        # Underlying close: try UNDERLYING_VALUE, then SETTLE_PR, then 0
        if 'UNDERLYING_VALUE' in df.columns:
            uc = pd.to_numeric(df['UNDERLYING_VALUE'], errors='coerce').fillna(0)
        elif 'SETTLE_PR' in df.columns:
            uc = pd.to_numeric(df['SETTLE_PR'], errors='coerce').fillna(0)
        else:
            uc = pd.Series(0.0, index=df.index)

        return pd.DataFrame({
            'date':              date_str,
            'expiry_date':       expiry_series.dt.date.values,
            'strike':            pd.to_numeric(df['STRIKE_PR'], errors='coerce').fillna(0).astype(float),
            'option_type':       df['_ot'].values,
            'close':             pd.to_numeric(df['CLOSE'], errors='coerce').fillna(0).astype(float),
            'volume':            pd.to_numeric(df.get('CONTRACTS', df.get('TRDQTY', 0)), errors='coerce').fillna(0).astype(int),
            'oi':                pd.to_numeric(df['OPEN_INT'], errors='coerce').fillna(0).astype(int),
            'underlying_close':  uc.values,
        }).query('strike > 0').reset_index(drop=True)

    def _parse_new(self, df: pd.DataFrame, date_str: str) -> pd.DataFrame:
        """Post-July 2024 UDiFF format."""
        ticker = df['TckrSymb'].astype(str).str.strip()
        mask = (
            (df['FinInstrmTp'].astype(str).str.strip() == 'IDO') &
            ticker.str.startswith('NIFTY') &
            ~ticker.str.startswith('NIFTYBANK') &
            ~ticker.str.startswith('FINNIFTY')
        )
        df = df[mask].copy()
        if df.empty:
            return pd.DataFrame()

        df['_ot'] = df['OptnTp'].astype(str).str.strip().str.upper()
        df = df[df['_ot'].isin(['CE', 'PE'])].copy()
        if df.empty:
            return pd.DataFrame()

        # Underlying close
        if 'UndrlygPric' in df.columns:
            uc_vals = pd.to_numeric(df['UndrlygPric'], errors='coerce').fillna(0)
        else:
            uc_vals = pd.Series(0.0, index=df.index)

        expiry_series = pd.to_datetime(
            df['XpryDt'].astype(str).str.strip(), format='%d-%b-%Y', errors='coerce'
        )
        if expiry_series.isna().all():
            expiry_series = pd.to_datetime(
                df['XpryDt'].astype(str).str.strip(), errors='coerce'
            )

        return pd.DataFrame({
            'date':              date_str,
            'expiry_date':       expiry_series.dt.date.values,
            'strike':            pd.to_numeric(df['StrkPric'], errors='coerce').fillna(0).astype(float),
            'option_type':       df['_ot'].values,
            'close':             pd.to_numeric(df['ClsPric'], errors='coerce').fillna(0).astype(float),
            'volume':            pd.to_numeric(df.get('TtlTradgVol', df.get('TradgVol', 0)), errors='coerce').fillna(0).astype(int),
            'oi':                pd.to_numeric(df['OpnIntrst'], errors='coerce').fillna(0).astype(int),
            'underlying_close':  uc_vals.values,
        }).query('strike > 0').reset_index(drop=True)

    # ── spot history helper ──────────────────────────────────────────────────

    def _update_spot_history(self, date_str: str, parsed: pd.DataFrame):
        """Track the underlying close per date for HV calculation."""
        uc = parsed['underlying_close']
        non_zero = uc[uc > 0]
        if not non_zero.empty:
            close = float(non_zero.median())
            self._spot_history.append((date_str, close))
            # keep sorted
            self._spot_history.sort(key=lambda x: x[0])

    # ── public query API ─────────────────────────────────────────────────────

    def get_price(self, date, expiry, strike: float, option_type: str) -> float | None:
        """
        Return EOD closing price for a specific contract, or None if not found.
        date / expiry: str ('YYYY-MM-DD'), date, or datetime.
        """
        date_str   = _to_date(date).strftime('%Y-%m-%d')
        expiry_dt  = _to_date(expiry)
        df = self._data.get(date_str)
        if df is None or df.empty:
            return None

        row = df[
            (df['expiry_date'] == expiry_dt) &
            (df['strike']      == float(strike)) &
            (df['option_type'] == option_type.upper())
        ]
        if row.empty:
            return None
        price = float(row.iloc[0]['close'])
        return price if price > 0 else None

    def get_expiries(self, start_date, end_date) -> list[date_type]:
        """Return sorted unique expiry dates present in the loaded data range."""
        s = _to_date(start_date).strftime('%Y-%m-%d')
        e = _to_date(end_date).strftime('%Y-%m-%d')
        expiries = set()
        for date_str, df in self._data.items():
            if s <= date_str <= e:
                for exp in df['expiry_date'].dropna().unique():
                    expiries.add(exp)
        return sorted(expiries)

    def get_underlying_close(self, date) -> float | None:
        """Return the NIFTY underlying close from bhavcopy for a given date."""
        date_str = _to_date(date).strftime('%Y-%m-%d')
        df = self._data.get(date_str)
        if df is None or df.empty:
            return None
        uc = df['underlying_close']
        non_zero = uc[uc > 0]
        return float(non_zero.median()) if not non_zero.empty else None

    def get_atm_iv(self, date) -> float | None:
        """
        Compute ATM implied volatility for `date`:
          1. Find nearest weekly expiry from the available expiry list.
          2. Find ATM strike (nearest 50 to underlying close).
          3. Take midpoint of ATM CE and PE closing prices.
          4. Back-solve IV using Black-Scholes (OptionAnalytics).
        Returns IV in % (e.g. 15.3), or None if insufficient data.
        """
        date_str = _to_date(date).strftime('%Y-%m-%d')
        if date_str in self._iv_cache:
            return self._iv_cache[date_str]

        df = self._data.get(date_str)
        if df is None or df.empty:
            self._iv_cache[date_str] = None
            return None

        # Spot
        uc = df['underlying_close']
        non_zero = uc[uc > 0]
        if non_zero.empty:
            self._iv_cache[date_str] = None
            return None
        spot = float(non_zero.median())

        # Nearest weekly expiry (prefer expiries within 7 days)
        date_d = _to_date(date)
        expiries = sorted(df['expiry_date'].dropna().unique())
        future_exp = [e for e in expiries if isinstance(e, date_type) and e >= date_d]
        if not future_exp:
            self._iv_cache[date_str] = None
            return None
        # Prefer shortest DTE (weekly)
        nearest_exp = min(future_exp)

        # DTE in years
        dte_days = (nearest_exp - date_d).days
        if dte_days <= 0:
            dte_days = 1
        T = dte_days / 365.0

        # ATM strike
        atm = _round50(spot)

        # CE and PE closing prices at ATM
        subset = df[df['expiry_date'] == nearest_exp]
        ce_row = subset[(subset['strike'] == atm) & (subset['option_type'] == 'CE')]
        pe_row = subset[(subset['strike'] == atm) & (subset['option_type'] == 'PE')]

        if ce_row.empty or pe_row.empty:
            self._iv_cache[date_str] = None
            return None

        ce_price = float(ce_row.iloc[0]['close'])
        pe_price = float(pe_row.iloc[0]['close'])
        if ce_price <= 0 or pe_price <= 0:
            self._iv_cache[date_str] = None
            return None

        mid_price = (ce_price + pe_price) / 2.0

        # Solve for IV using midpoint price as a synthetic ATM straddle leg
        try:
            iv = _oa.implied_volatility(mid_price, spot, atm, T, RISK_FREE, 'CE')
            if iv > 0:
                self._iv_cache[date_str] = iv
                return iv
        except Exception:
            pass

        self._iv_cache[date_str] = None
        return None

    def get_hv_20d(self, date) -> float | None:
        """
        20-day historical volatility (annualised, %) of NIFTY daily closes
        up to and including `date`. Uses data already loaded.
        """
        date_str = _to_date(date).strftime('%Y-%m-%d')
        hist = [(ds, c) for ds, c in self._spot_history if ds <= date_str]
        if len(hist) < 21:
            return None
        closes = pd.Series([c for _, c in hist[-21:]])
        log_ret = np.log(closes / closes.shift(1)).dropna()
        hv = log_ret.std() * np.sqrt(252) * 100
        return round(float(hv), 4)

    def get_all_dates(self) -> list[str]:
        """Return sorted list of all loaded trading dates."""
        return sorted(self._data.keys())

    def loaded_dates_count(self) -> int:
        return len(self._data)
