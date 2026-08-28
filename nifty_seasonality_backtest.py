"""
nifty_seasonality_backtest.py
==============================
NIFTY Index Options — Sept–Jan Seasonality Backtest (Fyers API)

Hypothesis: Short ATM straddle gives muted/poor returns during the
Sept–Jan window vs the rest-of-year baseline.

Strategy: Short ATM straddle at monthly expiry open, held to expiry
          or stopped at 2x credit received.
Data:     5-year span (2021-2026) — actual option history via Fyers
          history() API where available; Black-Scholes synthetic fill
          for expired contracts not served by the API.

Outputs:
  1. Year-by-year table (seasonal PnL vs annual avg)
  2. Correlation: VIX level / NIFTY return vs window PnL
  3. Mean / median + t-test: in-window vs out-of-window
  4. Win rate comparison
  5. CSV export + monthly bar chart (avg PnL by month)

Usage:
  python nifty_seasonality_backtest.py [--no-cache] [--output-dir PATH]
"""

# ── stdlib ────────────────────────────────────────────────────────────────────
import os
import sys
import json
import math
import time
import logging
import argparse
import calendar
import warnings
from datetime import date, datetime, timedelta
from typing import Optional, Dict, List, Tuple

warnings.filterwarnings("ignore", category=FutureWarning)

# Force UTF-8 on Windows so Rs / arrow chars don't crash the logger
import io as _io
if isinstance(sys.stdout, _io.TextIOWrapper) and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass
if isinstance(sys.stderr, _io.TextIOWrapper) and sys.stderr.encoding.lower() != "utf-8":
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass

# ── third-party ───────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy import stats as scipy_stats

# ── Fyers SDK ─────────────────────────────────────────────────────────────────
# Fyers removed; using BhavCopyEngine for EOD data.

# ─────────────────────────────────────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
# Backtest window
BACKTEST_START = date(2022, 1, 1)
BACKTEST_END   = date(2026, 1, 31)   # from 2022 to now

# Seasonality window definition
SEASON_MONTHS = {9, 10, 11, 12, 1}   # Sept, Oct, Nov, Dec, Jan

# Strategy params
NIFTY_LOT_SIZE  = 75     # units per lot (current: 75; adjust if needed)
STOP_MULTIPLIER = 2.0    # stop at 2x credit received
RISK_FREE_RATE  = 0.07   # 7% proxy for Indian risk-free rate

# API retry config
MAX_RETRIES    = 3
RETRY_DELAY_S  = 2.0     # base delay; exponential backoff applied

# Cache
CACHE_DIR = os.path.join(os.path.dirname(__file__), "historical_data", "seasonality_cache")

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(
            os.path.join(os.path.dirname(__file__), "seasonality_backtest.log"),
            mode="w", encoding="utf-8",
        ),
    ],
)
log = logging.getLogger("SeasonalityBT")

# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 1 — BHAVCOPY ENGINE WRAPPER
# ─────────────────────────────────────────────────────────────────────────────
from BhavCopyEngine import BhavCopyEngine



# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 2 — EXPIRY CALENDAR (MONTHLY)
# ─────────────────────────────────────────────────────────────────────────────

def _last_weekday(y: int, m: int, weekday: int) -> date:
    last = calendar.monthrange(y, m)[1]
    d = date(y, m, last)
    while d.weekday() != weekday:
        d -= timedelta(days=1)
    return d

def monthly_expiry(y: int, m: int) -> date:
    """NIFTY monthly expiry: last Thursday before 2024, last Wednesday from 2024."""
    return _last_weekday(y, m, 3 if y < 2024 else 2)

def get_monthly_cycles(start: date, end: date) -> List[Dict]:
    cycles = []
    y, m = start.year, start.month
    while True:
        exp = monthly_expiry(y, m)
        prev_y, prev_m = (y - 1, 12) if m == 1 else (y, m - 1)
        prev_exp  = monthly_expiry(prev_y, prev_m)
        open_date = prev_exp + timedelta(days=1)
        while open_date.weekday() >= 5:
            open_date += timedelta(days=1)
        if open_date > end:
            break
        if exp >= start:
            cycles.append({
                "open_date":   open_date,
                "expiry_date": exp,
                "year":        y,
                "month":       m,
                "month_name":  date(y, m, 1).strftime("%b"),
            })
        m += 1
        if m > 12:
            m, y = 1, y + 1
        if date(y, m, 1) > end + timedelta(days=31):
            break
    return cycles


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 3 — OPTION SYMBOL BUILDER
# ─────────────────────────────────────────────────────────────────────────────

_MONTH_CODE = {
    1: "JAN", 2: "FEB", 3: "MAR", 4: "APR",
    5: "MAY", 6: "JUN", 7: "JUL", 8: "AUG",
    9: "SEP", 10: "OCT", 11: "NOV", 12: "DEC",
}

def atm_strike(spot: float) -> float:
    return round(spot / 50) * 50

def build_option_symbol(year: int, month: int, strike: float, opt_type: str) -> str:
    """NSE:NIFTY{YY}{MMM}{STRIKE}{CE/PE}"""
    return f"NSE:NIFTY{str(year)[-2:]}{_MONTH_CODE[month]}{int(atm_strike(strike))}{opt_type.upper()}"


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 4 — BLACK-SCHOLES SYNTHETIC PRICING
# ─────────────────────────────────────────────────────────────────────────────

def _ncdf(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))

def black_scholes(S: float, K: float, T: float, r: float, sigma: float, opt_type: str) -> float:
    if T <= 0:
        return max(S - K, 0.0) if opt_type == "CE" else max(K - S, 0.0)
    T, sigma = max(T, 1e-6), max(sigma, 1e-4)
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    if opt_type == "CE":
        return max(S * _ncdf(d1) - K * math.exp(-r * T) * _ncdf(d2), 0.05)
    return max(K * math.exp(-r * T) * _ncdf(-d2) - S * _ncdf(-d1), 0.05)

def synthetic_premium(spot: float, dte_days: float, iv_pct: float, K_ce: float, K_pe: float) -> float:
    T = max(dte_days / 365.0, 1e-5)
    s = iv_pct / 100.0
    return black_scholes(spot, K_ce, T, RISK_FREE_RATE, s, "CE") + \
           black_scholes(spot, K_pe, T, RISK_FREE_RATE, s, "PE")

def simulate_short_strategy(
    spot_open: float, spot_close: float,
    entry_premium: float, dte_days: float,
    spot_series: List[float], vix_series: List[float],
    K_ce: float, K_pe: float
) -> Tuple[float, bool, str]:
    """
    Simulate short combination (straddle/strangle) with path-based 2x stop.
    Returns (pnl_per_unit, was_stopped, note).
    """
    stop_loss = entry_premium * STOP_MULTIPLIER

    n = min(len(spot_series), len(vix_series))
    for i in range(n):
        remaining = max(dte_days - i, 0)
        current = synthetic_premium(spot_series[i], remaining, vix_series[i], K_ce, K_pe)
        if (current - entry_premium) >= stop_loss:
            return -stop_loss, True, "stopped"

    # Expiry intrinsic
    intrinsic = max(0.0, spot_close - K_ce) + max(0.0, K_pe - spot_close)
    return entry_premium - intrinsic, False, "expired"


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 5 — CACHE LAYER
# ─────────────────────────────────────────────────────────────────────────────

os.makedirs(CACHE_DIR, exist_ok=True)

def _cache_path(key: str) -> str:
    return os.path.join(CACHE_DIR, key.replace(":", "_").replace("/", "_") + ".csv")

def _load_cache(key: str) -> Optional[pd.DataFrame]:
    p = _cache_path(key)
    if os.path.exists(p):
        try:
            df = pd.read_csv(p, parse_dates=["date"], index_col="date")
            # Convert DatetimeIndex → plain date objects for consistent indexing
            df.index = [ts.date() for ts in pd.DatetimeIndex(df.index)]
            return df
        except Exception:
            pass
    return None

def _save_cache(key: str, df: pd.DataFrame):
    df.to_csv(_cache_path(key))

def fetch_index_bhav(bhav: BhavCopyEngine, start: date, end: date) -> pd.DataFrame:
    """Build NIFTY 50 and VIX proxy dataframes from BhavCopyEngine."""
    dates = bhav.get_all_dates()
    df_data = []
    
    for d_str in dates:
        d = date.fromisoformat(d_str)
        if start <= d <= end:
            spot = bhav.get_underlying_close(d)
            iv = bhav.get_atm_iv(d)
            if spot is not None:
                df_data.append({
                    "date": d,
                    "spot_close": spot,
                    "iv": iv if iv else 15.0
                })
                
    if not df_data:
        return pd.DataFrame()
        
    df = pd.DataFrame(df_data).set_index("date")
    # Add dummy columns for compatibility
    df["close"] = df["spot_close"] 
    return df

def fetch_option_chain_bhav(
    bhav: BhavCopyEngine, year: int, month: int,
    open_date: date, expiry_date: date, K_ce: int, K_pe: int
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], str]:
    """Returns (ce_df, pe_df, source='actual'|'unavailable') directly from Bhavcopy."""
    
    dates = []
    curr = open_date
    while curr <= expiry_date:
        if curr.weekday() < 5:
            dates.append(curr)
        curr += timedelta(days=1)
        
    ce_rows, pe_rows = [], []
    for d in dates:
        ce_price = bhav.get_price(d, expiry_date, K_ce, 'CE')
        if ce_price is not None:
            ce_rows.append({"date": d, "close": ce_price})
            
        pe_price = bhav.get_price(d, expiry_date, K_pe, 'PE')
        if pe_price is not None:
            pe_rows.append({"date": d, "close": pe_price})
            
    ce_df = pd.DataFrame(ce_rows).set_index("date") if ce_rows else None
    pe_df = pd.DataFrame(pe_rows).set_index("date") if pe_rows else None

    if ce_df is not None and pe_df is not None and not ce_df.empty and not pe_df.empty:
        log.info(f"[bhav] Option chain {K_ce}CE / {K_pe}PE for {open_date} -> {expiry_date}")
        return ce_df, pe_df, "actual"
        
    log.warning(f"[SKIP] Bhavcopy missing options for K={K_ce}CE/{K_pe}PE exp={expiry_date} — synthetic fallback")
    return None, None, "unavailable"


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 6 — BACKTEST ENGINE
# ─────────────────────────────────────────────────────────────────────────────

def _price_at(df: pd.DataFrame, d: date, col: str = "close", fwd: int = 5) -> Optional[float]:
    if df is None or df.empty:
        return None
    for delta in range(fwd + 1):
        day = d + timedelta(days=delta)
        if day in df.index:
            # Use .at[] with string key to satisfy Pyright's loc overload checks
            return float(df.at[day, col])  # type: ignore[arg-type]
    return None

def _series_between(df: pd.DataFrame, start: date, end: date, col: str = "close") -> List[float]:
    if df is None or df.empty:
        return []
    return [float(df.at[d, col]) for d in sorted(df.index) if start <= d <= end]  # type: ignore[arg-type]

def run_backtest(
    nifty_df: pd.DataFrame,
    cycles: List[Dict],
    bhav: BhavCopyEngine,
) -> Tuple[List[dict], int, int]:
    records     = []
    synth_count = 0

    for cyc in cycles:
        od    = cyc["open_date"]
        exp   = cyc["expiry_date"]
        yr    = cyc["year"]
        mo    = cyc["month"]
        mname = cyc["month_name"]
        label = f"{yr}-{mname}"

        spot_open  = _price_at(nifty_df, od,  "close")
        spot_close = _price_at(nifty_df, exp, "close")
        if spot_open is None or spot_close is None:
            log.warning(f"[{label}] Missing spot data — skipping.")
            continue

        vix_open = _price_at(nifty_df, od, "iv") or 15.0
        dte      = max((exp - od).days, 1)
        spot_series = _series_between(nifty_df, od, exp, "close")
        vix_series  = _series_between(nifty_df, od, exp, "iv")
        if not vix_series:
            vix_series = [vix_open] * len(spot_series)
        n = min(len(spot_series), len(vix_series))
        spot_series, vix_series = spot_series[:n], vix_series[:n]

        otm_percentages = [0.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]

        for otm_pct in otm_percentages:
            K_ce = int(atm_strike(spot_open * (1 + otm_pct/100.0)))
            K_pe = int(atm_strike(spot_open * (1 - otm_pct/100.0)))
            strat_type = "straddle" if otm_pct == 0.0 else "strangle"

            # Attempt actual option data
            ce_df, pe_df, src = fetch_option_chain_bhav(bhav, yr, mo, od, exp, K_ce, K_pe)
            data_source   = "actual"
            entry_premium = None

            if ce_df is not None and pe_df is not None:
                ce_p = _price_at(ce_df, od, "close")
                pe_p = _price_at(pe_df, od, "close")
                if ce_p and pe_p:
                    entry_premium = ce_p + pe_p

            if entry_premium is None:
                entry_premium = synthetic_premium(spot_open, dte, vix_open, K_ce, K_pe)
                data_source   = "synthetic"
                synth_count  += 1
                log.info(f"[{label}] {strat_type.upper()} {otm_pct}% SYNTHETIC prem={entry_premium:.2f} "
                         f"(S={spot_open:.0f}, IV={vix_open:.1f}%, DTE={dte}d)")

            pnl_unit, stopped, note = simulate_short_strategy(
                spot_open, spot_close, entry_premium, dte, spot_series, vix_series, K_ce, K_pe
            )
            pnl_rs    = pnl_unit * NIFTY_LOT_SIZE
            in_window = mo in SEASON_MONTHS

            records.append({
                "open_date":     od,
                "expiry_date":   exp,
                "year":          yr,
                "month":         mo,
                "month_name":    mname,
                "spot_open":     round(spot_open, 2),
                "spot_close":    round(spot_close, 2),
                "iv_at_open":    round(vix_open, 2),
                "dte_days":      dte,
                "strategy":      strat_type,
                "otm_pct":       otm_pct,
                "entry_premium": round(entry_premium, 2),
                "pnl_per_unit":  round(pnl_unit, 2),
                "pnl_rs":        round(pnl_rs, 2),
                "was_stopped":   stopped,
                "note":          f"{note} [{data_source}]",
                "data_source":   data_source,
                "in_window":     in_window,
            })

            log.info(
                f"[{label:>8}] {'IN ' if in_window else 'OUT'} | {strat_type[:4].upper()} {otm_pct}% | "
                f"S={spot_open:.0f}->{spot_close:.0f} | "
                f"IV={vix_open:.1f}% DTE={dte}d prem={entry_premium:.1f} "
                f"PnL=Rs{pnl_rs:>9,.0f} | {'STOP' if stopped else 'EXP'} [{data_source[0].upper()}]"
            )

    total = len(records)
    log.info(f"\nData quality: {synth_count}/{total} synthetic "
             f"({synth_count/total*100:.1f}%)" if total else "No cycles.")
    return records, synth_count, total


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 7 — STATISTICS
# ─────────────────────────────────────────────────────────────────────────────

def compute_stats(records: List[dict]) -> Dict:
    df   = pd.DataFrame(records)
    inw  = df[df["in_window"]].copy()
    outw = df[~df["in_window"]].copy()

    def _s(sub: pd.DataFrame) -> dict:
        if sub.empty:
            return {}
        p = sub["pnl_rs"]
        return {
            "n": len(sub),
            "mean_pnl":   p.mean(),
            "median_pnl": p.median(),
            "std_pnl":    p.std(),
            "win_rate":   (p > 0).mean() * 100,
            "total_pnl":  p.sum(),
            "min_pnl":    p.min(),
            "max_pnl":    p.max(),
            "stop_rate":  sub["was_stopped"].mean() * 100,
        }

    stats: Dict = {"in_window": _s(inw), "out_window": _s(outw)}

    # t-test
    if len(inw) >= 2 and len(outw) >= 2:
        t, p = scipy_stats.ttest_ind(inw["pnl_rs"], outw["pnl_rs"], equal_var=False)
        stats["t_stat"] = t; stats["p_value"] = p
    else:
        stats["t_stat"] = float("nan"); stats["p_value"] = float("nan")

    # Correlation: avg VIX vs year window PnL
    yr_vix = inw.groupby("year")["iv_at_open"].mean()
    yr_pnl = inw.groupby("year")["pnl_rs"].sum()
    c = yr_vix.index.intersection(yr_pnl.index)
    if len(c) >= 3:
        rv, pv = scipy_stats.pearsonr(yr_vix[c], yr_pnl[c])
        stats["corr_vix_r"] = rv; stats["corr_vix_p"] = pv
    else:
        stats["corr_vix_r"] = float("nan"); stats["corr_vix_p"] = float("nan")

    # Correlation: NIFTY spot return vs year window PnL
    spot_rets = {}
    for yr, g in inw.groupby("year"):
        spot_rets[yr] = (g["spot_close"].iloc[-1] - g["spot_open"].iloc[0]) \
                        / g["spot_open"].iloc[0] * 100
    sr = pd.Series(spot_rets)
    c2 = sr.index.intersection(yr_pnl.index)
    if len(c2) >= 3:
        rs, ps = scipy_stats.pearsonr(sr[c2], yr_pnl[c2])
        stats["corr_spot_r"] = rs; stats["corr_spot_p"] = ps
    else:
        stats["corr_spot_r"] = float("nan"); stats["corr_spot_p"] = float("nan")

    # Year-by-year table
    table = []
    for yr in sorted(df["year"].unique()):
        yr_df   = df[df["year"] == yr]
        win_df  = yr_df[yr_df["in_window"]]
        table.append({
            "Year":                    yr,
            "Window PnL (Rs)":         round(win_df["pnl_rs"].sum(), 2),
            "Window Cycles":           len(win_df),
            "Window Avg/Cycle (Rs)":   round(win_df["pnl_rs"].mean(), 2) if len(win_df) else float("nan"),
            "Annual Avg/Cycle (Rs)":   round(yr_df["pnl_rs"].mean(), 2),
            "Window Win Rate %":       round((win_df["pnl_rs"] > 0).mean() * 100, 1) if len(win_df) else float("nan"),
            "Avg VIX (Window)":        round(win_df["iv_at_open"].mean(), 2) if len(win_df) else float("nan"),
        })
    stats["year_table"]   = table
    stats["full_df"]      = df
    stats["monthly_avg"]  = df.groupby("month")["pnl_rs"].mean().reindex(range(1, 13))
    return stats


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 8 — REPORTING
# ─────────────────────────────────────────────────────────────────────────────

def print_report(stats: Dict, synth_count: int, total: int):
    line = "─" * 72
    pct  = synth_count / total * 100 if total else 0

    print(f"\n{'='*72}")
    print("  NIFTY OPTIONS — SEPT-JAN SEASONALITY BACKTEST RESULTS")
    print(f"{'='*72}")
    print(f"  Total cycles : {total}")
    print(f"  Actual data  : {total - synth_count} ({100-pct:.1f}%)")
    print(f"  Synthetic BS : {synth_count} ({pct:.1f}%)")

    # Year-by-year
    yt = stats["year_table"]
    print(f"\n{line}")
    print("  YEAR-BY-YEAR SEASONALITY TABLE  (1 lot = NIFTY_LOT_SIZE units)")
    print(line)
    print(f"{'Year':>6} {'Win PnL(Rs)':>14} {'Win Cyc':>8} "
          f"{'Win Avg':>10} {'Ann Avg':>10} {'WinRate':>8} {'AvgVIX':>8}")
    print(line)
    for r in yt:
        wa = r["Window Avg/Cycle (Rs)"]
        aa = r["Annual Avg/Cycle (Rs)"]
        wr = r["Window Win Rate %"]
        av = r["Avg VIX (Window)"]
        print(f"{r['Year']:>6} "
              f"{r['Window PnL (Rs)']:>14,.0f} "
              f"{r['Window Cycles']:>8} "
              f"{wa:>10,.0f} "
              f"{aa:>10,.0f} "
              f"{wr:>7.1f}% "
              f"{av:>7.1f}%")

    # Summary stats
    iw = stats["in_window"]
    ow = stats["out_window"]
    print(f"\n{line}")
    print("  AVG PnL STATISTICS (per cycle, 1 lot)")
    print(line)
    print(f"{'Metric':<28} {'IN WINDOW (Sep-Jan)':>22} {'OUT OF WINDOW':>18}")
    print(line)
    for label, key in [
        ("Cycles",        "n"),
        ("Mean PnL (Rs)", "mean_pnl"),
        ("Median PnL",    "median_pnl"),
        ("Std Dev",       "std_pnl"),
        ("Win Rate (%)",  "win_rate"),
        ("Stop Rate (%)", "stop_rate"),
        ("Total PnL",     "total_pnl"),
    ]:
        vi = iw.get(key, float("nan"))
        vo = ow.get(key, float("nan"))
        fmt = "d" if key == "n" else ",.0f" if "pnl" in key or key == "total_pnl" else ".1f"
        try:
            si = format(int(vi), fmt) if fmt == "d" else format(vi, fmt)
            so = format(int(vo), fmt) if fmt == "d" else format(vo, fmt)
        except Exception:
            si, so = str(vi), str(vo)
        print(f"  {label:<26} {si:>22} {so:>18}")

    # t-test
    t = stats["t_stat"]; p = stats["p_value"]
    sig = "*** SIGNIFICANT (p<0.05)" if not math.isnan(p) and p < 0.05 else "not significant"
    print(f"\n{line}")
    print("  WELCH t-TEST (in-window vs out-of-window PnL)")
    print(line)
    print(f"  t = {t:.4f}   p = {p:.4f}   → {sig}")

    # Correlations
    print(f"\n{line}")
    print("  CORRELATIONS (year-level, in-window)")
    print(line)
    print(f"  Avg VIX vs Window PnL   : r = {stats['corr_vix_r']:>7.4f}   p = {stats['corr_vix_p']:.4f}")
    print(f"  NIFTY Return vs Win PnL : r = {stats['corr_spot_r']:>7.4f}   p = {stats['corr_spot_p']:.4f}")
    print()


def export_csv(records: List[Dict], out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    t_path = os.path.join(out_dir, "seasonality_trades.csv")
    pd.DataFrame(records).to_csv(t_path, index=False)
    log.info(f"Trade log   → {t_path}")
    return t_path


def plot_chart(stats: Dict, out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    monthly  = stats["monthly_avg"]
    full_df  = stats["full_df"].copy()
    full_df["open_date"] = pd.to_datetime(full_df["open_date"])
    full_df.sort_values("open_date", inplace=True)
    full_df["cum_pnl"] = full_df["pnl_rs"].cumsum()
    in_df = full_df[full_df["in_window"]]

    month_names = ["Jan","Feb","Mar","Apr","May","Jun",
                   "Jul","Aug","Sep","Oct","Nov","Dec"]
    vals   = [float(monthly.get(m, 0) or 0) for m in range(1, 13)]
    colors = ["#ff4b4b" if m in SEASON_MONTHS else "#00b894" for m in range(1, 13)]

    BG = "#0e1117"
    fig, axes = plt.subplots(2, 1, figsize=(14, 10),
                              gridspec_kw={"height_ratios": [2, 1]})
    fig.patch.set_facecolor(BG)

    # ── Bar chart ─────────────────────────────────────────────────────────────
    ax = axes[0]
    ax.set_facecolor(BG)
    bars = ax.bar(month_names, vals, color=colors, alpha=0.88,
                  edgecolor="white", linewidth=0.4, width=0.65)

    for bar, val in zip(bars, vals):
        if not math.isnan(val) and val != 0:
            ypos = bar.get_height() + 300 if val >= 0 else bar.get_height() - 800
            ax.text(bar.get_x() + bar.get_width() / 2, ypos,
                    f"₹{val:,.0f}", ha="center", va="bottom" if val >= 0 else "top",
                    color="white", fontsize=8, fontweight="bold")

    for i, m in enumerate(range(1, 13)):
        if m in SEASON_MONTHS:
            ax.axvspan(i - 0.5, i + 0.5, color="#ff4b4b", alpha=0.07)

    ax.axhline(0, color="white", lw=0.8, ls="--", alpha=0.5)
    ax.set_title("NIFTY Short ATM Straddle — Avg PnL per Cycle by Month (5-Year Backtest)",
                 color="white", fontsize=14, fontweight="bold", pad=12)
    ax.set_ylabel("Avg PnL per Cycle (Rs)", color="white")
    ax.tick_params(colors="white")
    ax.spines[:].set_color("#333")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"₹{x:,.0f}"))

    from matplotlib.patches import Patch
    ax.legend(
        handles=[
            Patch(facecolor="#ff4b4b", alpha=0.88, label="Sept–Jan (Seasonal Window)"),
            Patch(facecolor="#00b894", alpha=0.88, label="Rest of Year (Baseline)"),
        ],
        loc="upper right", facecolor="#1e2130", edgecolor="#555", labelcolor="white", fontsize=9
    )

    # ── Cumulative PnL ────────────────────────────────────────────────────────
    ax2 = axes[1]
    ax2.set_facecolor(BG)
    ax2.plot(full_df["open_date"], full_df["cum_pnl"],
             color="#f8f8f2", lw=1.6, label="All cycles", alpha=0.9)
    ax2.fill_between(full_df["open_date"], full_df["cum_pnl"], alpha=0.08, color="#f8f8f2")
    ax2.scatter(in_df["open_date"], in_df["cum_pnl"],
                color="#ff4b4b", s=35, zorder=5, label="In-window ends")
    ax2.set_title("Cumulative PnL Over Time", color="white", fontsize=11, pad=8)
    ax2.set_ylabel("Cumulative PnL (Rs)", color="white")
    ax2.tick_params(colors="white")
    ax2.spines[:].set_color("#333")
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"₹{x:,.0f}"))
    ax2.legend(facecolor="#1e2130", edgecolor="#555", labelcolor="white", fontsize=8)

    plt.tight_layout(pad=2.0)
    chart_path = os.path.join(out_dir, "seasonality_chart.png")
    plt.savefig(chart_path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    log.info(f"Chart       → {chart_path}")
    return chart_path


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 9 — SYNTHETIC FALLBACK DATA GENERATORS
# ─────────────────────────────────────────────────────────────────────────────

def _trading_dates(start: date, end: date) -> List[date]:
    return [start + timedelta(days=i) for i in range((end - start).days + 1)
            if (start + timedelta(days=i)).weekday() < 5]

def _synthetic_spot_df(start: date, end: date) -> pd.DataFrame:
    """GBM calibrated to NIFTY 50 historical stats (mu=12%, sigma=18%, S0=15000)."""
    np.random.seed(42)
    dates = _trading_dates(start, end)
    n     = len(dates)
    drift = 0.12 / 252
    vol   = 0.18 / math.sqrt(252)
    spot  = 15000.0 * np.exp(np.cumsum(
        np.random.normal(drift - 0.5 * vol**2, vol, n)
    ))
    df = pd.DataFrame({"open": spot * 0.999, "high": spot * 1.003,
                       "low": spot * 0.997, "close": spot,
                       "volume": np.zeros(n)}, index=dates)
    df.index.name = "date"
    log.info(f"[synthetic] NIFTY spot {spot[0]:.0f}->{spot[-1]:.0f} ({n} days)")
    return df

def _synthetic_vix_df(start: date, end: date) -> pd.DataFrame:
    """Ornstein-Uhlenbeck VIX with seasonal bump in Sept-Jan."""
    np.random.seed(99)
    dates = _trading_dates(start, end)
    n     = len(dates)
    vix   = []
    v     = 14.0
    for d in dates:
        mu = 16.0 if d.month in SEASON_MONTHS else 13.0
        v  = max(8.0, min(55.0, v + 0.08 * (mu - v) + 1.5 * np.random.randn()))
        vix.append(v)
    vix_arr = np.array(vix)
    df = pd.DataFrame({"open": vix_arr, "high": vix_arr + 0.5,
                       "low": vix_arr - 0.5, "close": vix_arr,
                       "volume": np.zeros(n)}, index=dates)
    df.index.name = "date"
    log.info(f"[synthetic] India VIX mean={np.mean(vix):.1f}%")
    return df


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 10 — MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="NIFTY Sept-Jan Seasonality Backtest")
    parser.add_argument("--no-cache",   action="store_true", help="Bypass disk cache")
    parser.add_argument("--output-dir", default=os.path.join(
        os.path.dirname(__file__), "historical_data", "seasonality_results"))
    parser.add_argument("--start", default=BACKTEST_START.isoformat())
    parser.add_argument("--end",   default=BACKTEST_END.isoformat())
    args = parser.parse_args()

    use_cache = not args.no_cache
    out_dir   = args.output_dir
    bt_start  = date.fromisoformat(args.start)
    bt_end    = date.fromisoformat(args.end)

    log.info("=" * 72)
    log.info("  NIFTY OPTIONS SEPT-JAN SEASONALITY BACKTEST")
    log.info(f"  Range  : {bt_start} to {bt_end}")
    log.info(f"  Window : September to January")
    log.info(f"  Lot    : {NIFTY_LOT_SIZE}  Stop: {STOP_MULTIPLIER}x credit")
    log.info(f"  Cache  : {'OFF' if not use_cache else CACHE_DIR}")
    log.info("=" * 72)

    # 1. Download BhavCopy Data
    log.info("\n[INIT] Loading Bhavcopy data... (This might take a while if not cached)")
    bhav = BhavCopyEngine()
    bhav.load_range(bt_start - timedelta(days=40), bt_end, verbose=True)

    # 2. Monthly cycle list
    cycles = get_monthly_cycles(bt_start, bt_end)
    log.info(f"Generated {len(cycles)} monthly expiry cycles")

    # 3. Fetch index data
    log.info("\n[FETCH] Processing NIFTY 50 and ATM IV from Bhavcopy...")
    nifty_df = fetch_index_bhav(bhav, bt_start, bt_end)

    if nifty_df.empty:
        log.warning("NIFTY index unavailable from Bhavcopy — using GBM synthetic spot path.")
        nifty_df = _synthetic_spot_df(bt_start, bt_end)
        nifty_df["iv"] = _synthetic_vix_df(bt_start, bt_end)["close"]

    # 4. Backtest
    log.info("\n[BACKTEST] Simulating monthly cycles...")
    records, synth_count, total = run_backtest(
        nifty_df, cycles, bhav
    )
    if not records:
        log.error("No cycles completed — cannot continue.")
        return

    # 5. Export
    log.info(f"\n[EXPORT] Saving to {out_dir}")
    t_path = export_csv(records, out_dir)

    print(f"{'='*72}")
    print("  OUTPUT FILES")
    print(f"{'─'*72}")
    print(f"  Trade CSV  : {t_path}")
    print(f"{'='*72}\n")


if __name__ == "__main__":
    main()
