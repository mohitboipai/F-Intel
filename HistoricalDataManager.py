"""
HistoricalDataManager.py
========================
High-performance historical data store for NIFTY options & futures.
Parses 310+ daily NSE Bhavcopy files into an indexed SQLite database for
sub-millisecond backtesting queries across 1+ year of data.
"""

import os
import glob
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
BHAVCOPY_DIR = os.path.join(PROJECT_ROOT, "data", "bhavcopy")
DB_PATH = os.path.join(PROJECT_ROOT, "data", "nifty_fo_historical.db")


class HistoricalDataManager:
    def __init__(self, db_path: str = DB_PATH, bhav_dir: str = BHAVCOPY_DIR):
        self.db_path = db_path
        self.bhav_dir = bhav_dir
        self._ensure_db_initialized()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_db_initialized(self):
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute("""
                CREATE TABLE IF NOT EXISTS meta_ingested_files (
                    filename TEXT PRIMARY KEY,
                    ingested_at TEXT,
                    record_count INTEGER
                )
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS daily_spot (
                    date TEXT PRIMARY KEY,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL
                )
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS option_chain (
                    date TEXT,
                    expiry TEXT,
                    strike REAL,
                    type TEXT,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    ltp REAL,
                    oi REAL,
                    chg_oi REAL,
                    volume REAL,
                    dte REAL,
                    spot REAL
                )
            """)
            cur.execute("CREATE INDEX IF NOT EXISTS idx_opt_date ON option_chain(date)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_opt_date_exp ON option_chain(date, expiry)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_opt_lookup ON option_chain(date, expiry, strike, type)")
            conn.commit()

    def sync_bhavcopy_data(self, force_reload: bool = False) -> int:
        """
        Scans BHAVCOPY_DIR and ingests any unindexed fo_bhav_*.csv files.
        Returns the number of newly indexed files.
        """
        files = sorted(glob.glob(os.path.join(self.bhav_dir, "fo_bhav_*.csv")))
        if not files:
            return 0

        with self._get_connection() as conn:
            cur = conn.cursor()
            if force_reload:
                cur.execute("DELETE FROM option_chain")
                cur.execute("DELETE FROM daily_spot")
                cur.execute("DELETE FROM meta_ingested_files")
                conn.commit()

            cur.execute("SELECT filename FROM meta_ingested_files")
            already_ingested = {row['filename'] for row in cur.fetchall()}

        to_ingest = [f for f in files if os.path.basename(f) not in already_ingested]
        if not to_ingest:
            return 0

        total_files = len(to_ingest)
        print(f"[HistoricalDataManager] Ingesting {total_files} new Bhavcopy files...")

        ingested_count = 0
        with self._get_connection() as conn:
            for i, fpath in enumerate(to_ingest):
                fname = os.path.basename(fpath)
                try:
                    df = pd.read_csv(fpath)
                    cols = {c.lower(): c for c in df.columns}
                    sym_col = cols.get('tckrsymb', 'TckrSymb')
                    type_col = cols.get('optntp', 'OptnTp')
                    inst_col = cols.get('fininstrmtp', 'FinInstrmTp')

                    # Filter for NIFTY options and futures
                    nifty_mask = (df[sym_col] == 'NIFTY')
                    if not nifty_mask.any():
                        continue

                    nifty_df = df[nifty_mask].copy()

                    # Extract Trade Date
                    trade_dt_col = cols.get('traddt', 'TradDt')
                    trade_date = str(nifty_df[trade_dt_col].dropna().iloc[0])[:10]

                    # Spot price from UndrlygPric or futures
                    spot_col = cols.get('undrlygpric', 'UndrlygPric')
                    spot_val = 0.0
                    if spot_col in nifty_df.columns:
                        sp_series = nifty_df[spot_col].dropna()
                        if not sp_series.empty:
                            spot_val = float(sp_series.iloc[0])

                    # Futures for high/low/open estimation if spot is 0
                    fut_df = nifty_df[nifty_df[type_col].isna() | (nifty_df[type_col] == 'XX') | (nifty_df[inst_col].isin(['FUTIDX', 'IDF', 'IFF']))]
                    spot_open = spot_val
                    spot_high = spot_val
                    spot_low = spot_val
                    spot_close = spot_val

                    if not fut_df.empty:
                        # Nearest future
                        cls_col = cols.get('clspric', 'ClsPric')
                        opn_col = cols.get('opnpric', 'OpnPric')
                        hgh_col = cols.get('hghpric', 'HghPric')
                        low_col = cols.get('lwpric', 'LwPric')
                        
                        row0 = fut_df.iloc[0]
                        if spot_val <= 0:
                            spot_val = float(row0.get(cls_col, 0) or 0)
                        spot_open = float(row0.get(opn_col, spot_val) or spot_val)
                        spot_high = float(row0.get(hgh_col, spot_val) or spot_val)
                        spot_low = float(row0.get(low_col, spot_val) or spot_val)
                        spot_close = spot_val

                    # Insert spot
                    cur = conn.cursor()
                    cur.execute("""
                        INSERT OR REPLACE INTO daily_spot (date, open, high, low, close)
                        VALUES (?, ?, ?, ?, ?)
                    """, (trade_date, spot_open, spot_high, spot_low, spot_close))

                    # Filter for options
                    opts_df = nifty_df[nifty_df[type_col].isin(['CE', 'PE'])].copy()
                    if opts_df.empty:
                        continue

                    xpry_col = cols.get('xprydt', 'XpryDt')
                    strk_col = cols.get('strkpric', 'StrkPric')
                    cls_col  = cols.get('clspric', 'ClsPric')
                    opn_col  = cols.get('opnpric', 'OpnPric')
                    hgh_col  = cols.get('hghpric', 'HghPric')
                    low_col  = cols.get('lwpric', 'LwPric')
                    ltp_col  = cols.get('lastpric', 'LastPric')
                    oi_col   = cols.get('opnintrst', 'OpnIntrst')
                    choi_col = cols.get('chnginopnintrst', 'ChngInOpnIntrst')
                    vol_col  = cols.get('ttltradgvol', 'TtlTradgVol')

                    trade_d = datetime.strptime(trade_date, "%Y-%m-%d")

                    records = []
                    for _, r in opts_df.iterrows():
                        exp_str = str(r[xpry_col])[:10]
                        try:
                            exp_d = datetime.strptime(exp_str, "%Y-%m-%d")
                            dte = max(0.0, (exp_d - trade_d).days)
                        except Exception:
                            dte = 1.0

                        records.append((
                            trade_date,
                            exp_str,
                            float(r[strk_col]),
                            str(r[type_col]).upper(),
                            float(r.get(opn_col, 0) or 0),
                            float(r.get(hgh_col, 0) or 0),
                            float(r.get(low_col, 0) or 0),
                            float(r.get(cls_col, 0) or 0),
                            float(r.get(ltp_col, 0) or 0),
                            float(r.get(oi_col, 0) or 0),
                            float(r.get(choi_col, 0) or 0),
                            float(r.get(vol_col, 0) or 0),
                            dte,
                            spot_val
                        ))

                    cur.executemany("""
                        INSERT INTO option_chain (
                            date, expiry, strike, type, open, high, low, close, ltp, oi, chg_oi, volume, dte, spot
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, records)

                    cur.execute("""
                        INSERT INTO meta_ingested_files (filename, ingested_at, record_count)
                        VALUES (?, ?, ?)
                    """, (fname, datetime.now().isoformat(), len(records)))

                    ingested_count += 1
                    if (i + 1) % 50 == 0 or (i + 1) == total_files:
                        conn.commit()
                        print(f"  Processed {i + 1}/{total_files} files...")

                except Exception as ex:
                    print(f"  Warning: Failed to process {fname}: {ex}")

            conn.commit()

        print(f"[HistoricalDataManager] Ingestion complete. {ingested_count} files stored.")
        return ingested_count

    def get_available_dates(self, start_date: str = "", end_date: str = "") -> List[str]:
        """Returns sorted list of distinct trading dates in YYYY-MM-DD format."""
        with self._get_connection() as conn:
            cur = conn.cursor()
            query = "SELECT DISTINCT date FROM daily_spot"
            params = []
            if start_date and end_date:
                query += " WHERE date >= ? AND date <= ?"
                params = [start_date, end_date]
            elif start_date:
                query += " WHERE date >= ?"
                params = [start_date]
            elif end_date:
                query += " WHERE date <= ?"
                params = [end_date]
            query += " ORDER BY date ASC"
            cur.execute(query, params)
            return [row['date'] for row in cur.fetchall()]

    def get_spot_series(self, start_date: str = "", end_date: str = "") -> Dict[str, Dict[str, float]]:
        """Returns map: date -> {open, high, low, close}."""
        dates_map = {}
        with self._get_connection() as conn:
            cur = conn.cursor()
            query = "SELECT date, open, high, low, close FROM daily_spot"
            params = []
            if start_date and end_date:
                query += " WHERE date >= ? AND date <= ?"
                params = [start_date, end_date]
            query += " ORDER BY date ASC"
            cur.execute(query, params)
            for r in cur.fetchall():
                dates_map[r['date']] = {
                    'open': float(r['open']),
                    'high': float(r['high']),
                    'low': float(r['low']),
                    'close': float(r['close']),
                }
        return dates_map

    def get_daily_chain(self, date: str, expiry: Optional[str] = None) -> pd.DataFrame:
        """
        Returns DataFrame of option chain for a given date.
        If expiry is None, returns all options for that date.
        """
        with self._get_connection() as conn:
            if expiry:
                query = """
                    SELECT date, expiry, strike, type, open, high, low, close, ltp, oi, chg_oi, volume, dte, spot
                    FROM option_chain
                    WHERE date = ? AND expiry = ?
                    ORDER BY strike ASC, type ASC
                """
                df = pd.read_sql_query(query, conn, params=(date, expiry))
            else:
                query = """
                    SELECT date, expiry, strike, type, open, high, low, close, ltp, oi, chg_oi, volume, dte, spot
                    FROM option_chain
                    WHERE date = ?
                    ORDER BY expiry ASC, strike ASC, type ASC
                """
                df = pd.read_sql_query(query, conn, params=(date,))
        return df

    def get_nearest_expiry(self, date: str) -> Optional[str]:
        """Returns the nearest future/current expiry date for the given trade date."""
        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute("""
                SELECT DISTINCT expiry FROM option_chain
                WHERE date = ? AND expiry >= ?
                ORDER BY expiry ASC LIMIT 1
            """, (date, date))
            row = cur.fetchone()
            return row['expiry'] if row else None

    def get_all_expiries(self, date: str) -> List[str]:
        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute("""
                SELECT DISTINCT expiry FROM option_chain
                WHERE date = ?
                ORDER BY expiry ASC
            """, (date,))
            return [r['expiry'] for r in cur.fetchall()]

    def get_next_trading_day(self, date: str) -> Optional[str]:
        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute("""
                SELECT date FROM daily_spot
                WHERE date > ?
                ORDER BY date ASC LIMIT 1
            """, (date,))
            row = cur.fetchone()
            return row['date'] if row else None

    def get_option_bar(self, date: str, expiry: str, strike: float, opt_type: str) -> Optional[Dict[str, float]]:
        """Returns OHLC bar for specific strike option on date."""
        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute("""
                SELECT open, high, low, close, ltp, oi, volume
                FROM option_chain
                WHERE date = ? AND expiry = ? AND strike = ? AND type = ?
                LIMIT 1
            """, (date, expiry, strike, opt_type.upper()))
            row = cur.fetchone()
            if row:
                return {
                    'open': float(row['open']),
                    'high': float(row['high']),
                    'low': float(row['low']),
                    'close': float(row['close']),
                    'ltp': float(row['ltp']),
                    'oi': float(row['oi']),
                    'volume': float(row['volume']),
                }
            return None
