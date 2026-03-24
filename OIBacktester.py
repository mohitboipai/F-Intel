"""
OIBacktester.py — OI Pressure Signal Backtesting Engine

Backtests the OI Pressure signal (BULLISH/BEARISH/NEUTRAL) on 2+ years
of historical NSE NIFTY options OI data from NSE F&O Bhavcopy.

Usage:
    python OIBacktester.py
"""

import sys
import os
import time
import io
import zipfile
import requests
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# ═══════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'historical_data', 'oi_bhavcopy')
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'historical_data', 'backtest_results')
UNDERLYING = 'NIFTY'
STRIKE_RANGE = 10
NSE_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.5',
    'Referer': 'https://www.nseindia.com/',
}


# ═══════════════════════════════════════════════════════════════════
# 1. DATA FETCHER — Download NSE Bhavcopy
# ═══════════════════════════════════════════════════════════════════
class BhavcopyFetcher:
    """Downloads and caches daily NSE F&O Bhavcopy files."""

    def __init__(self):
        os.makedirs(CACHE_DIR, exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update(NSE_HEADERS)
        try:
            self.session.get('https://www.nseindia.com/', timeout=10)
        except:
            pass

    def _get_url_old_format(self, date):
        month_str = date.strftime('%b').upper()
        return (
            f"https://nsearchives.nseindia.com/content/historical/DERIVATIVES/"
            f"{date.year}/{month_str}/fo{date.strftime('%d%b%Y').upper()}bhav.csv.zip"
        )

    def _get_url_new_format(self, date):
        return (
            f"https://nsearchives.nseindia.com/content/fo/"
            f"BhavCopy_NSE_FO_0_0_0_{date.strftime('%Y%m%d')}_F_0000.csv.zip"
        )

    def _cached_path(self, date):
        return os.path.join(CACHE_DIR, f"fo_bhav_{date.strftime('%Y%m%d')}.csv")

    def fetch_single_day(self, date):
        cached = self._cached_path(date)
        if os.path.exists(cached):
            try:
                return pd.read_csv(cached)
            except:
                pass

        cutover = datetime(2024, 7, 8)
        url = self._get_url_old_format(date) if date < cutover else self._get_url_new_format(date)

        try:
            r = self.session.get(url, timeout=15)
            if r.status_code == 200 and len(r.content) > 100:
                z = zipfile.ZipFile(io.BytesIO(r.content))
                csv_name = z.namelist()[0]
                with z.open(csv_name) as f:
                    df = pd.read_csv(f)
                df.to_csv(cached, index=False)
                return df
        except:
            pass
        return None

    def fetch_date_range(self, start_date, end_date):
        all_data = {}
        current = start_date
        fetched = 0
        skipped = 0

        print(f"\n{'='*60}")
        print(f"  NSE BHAVCOPY DOWNLOADER")
        print(f"  Range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        print(f"{'='*60}")

        while current <= end_date:
            if current.weekday() >= 5:
                current += timedelta(days=1)
                continue

            date_str = current.strftime('%Y-%m-%d')
            df = self.fetch_single_day(current)

            if df is not None and not df.empty:
                all_data[date_str] = df
                fetched += 1
                if fetched % 50 == 0:
                    print(f"  Fetched {fetched} days... (last: {date_str})")
            else:
                skipped += 1

            current += timedelta(days=1)
            if not os.path.exists(self._cached_path(current - timedelta(days=1))):
                time.sleep(0.5)

        print(f"  Done: {fetched} trading days fetched, {skipped} skipped (holidays/errors)")
        return all_data


# ═══════════════════════════════════════════════════════════════════
# 2. DATA PROCESSOR — Parse & Filter NIFTY Options
# ═══════════════════════════════════════════════════════════════════
class OIDataProcessor:
    """Processes raw bhavcopy DataFrames into clean NIFTY option OI data."""

    def _detect_format(self, df):
        cols = set(c.strip() for c in df.columns)
        if 'INSTRUMENT' in cols:
            return 'old'
        elif 'TckrSymb' in cols:
            return 'new'
        return 'unknown'

    def process_single_day(self, df, date_str):
        if df is None or df.empty:
            return pd.DataFrame()

        df.columns = [c.strip() for c in df.columns]
        fmt = self._detect_format(df)

        if fmt == 'old':
            return self._process_old(df, date_str)
        elif fmt == 'new':
            return self._process_new(df, date_str)
        return pd.DataFrame()

    def _process_old(self, df, date_str):
        """Pre-July 2024: INSTRUMENT, SYMBOL, STRIKE_PR, OPTION_TYP, OPEN_INT, CHG_IN_OI, CLOSE"""
        mask = (
            (df['INSTRUMENT'].astype(str).str.strip() == 'OPTIDX') &
            (df['SYMBOL'].astype(str).str.strip() == UNDERLYING)
        )
        df = df[mask].copy()
        if df.empty:
            return pd.DataFrame()

        df['_type'] = df['OPTION_TYP'].astype(str).str.strip().str.upper()
        df = df[df['_type'].isin(['CE', 'PE'])].copy()
        if df.empty:
            return pd.DataFrame()

        return pd.DataFrame({
            'date': date_str,
            'strike': pd.to_numeric(df['STRIKE_PR'], errors='coerce').fillna(0).astype(int),
            'type': df['_type'].values,
            'oi': pd.to_numeric(df['OPEN_INT'], errors='coerce').fillna(0).astype(int),
            'oi_change': pd.to_numeric(df['CHG_IN_OI'], errors='coerce').fillna(0).astype(int),
            'close': pd.to_numeric(df['CLOSE'], errors='coerce').fillna(0),
            'expiry': df['EXPIRY_DT'].values,
            'spot': 0,
        }).query('strike > 0').reset_index(drop=True)

    def _process_new(self, df, date_str):
        """Post-July 2024 UDiFF: TckrSymb, FinInstrmTp, OptnTp, StrkPric, OpnIntrst, etc."""
        # FinInstrmTp=IDO (Index Options) + TckrSymb starts with NIFTY (not BANKNIFTY/FINNIFTY)
        ticker_col = df['TckrSymb'].astype(str).str.strip()
        mask = (
            (df['FinInstrmTp'].astype(str).str.strip() == 'IDO') &
            (ticker_col.str.startswith('NIFTY')) &
            (~ticker_col.str.startswith('NIFTYBANK')) &
            (~ticker_col.str.startswith('FINNIFTY'))
        )
        df = df[mask].copy()
        if df.empty:
            return pd.DataFrame()

        df['_type'] = df['OptnTp'].astype(str).str.strip().str.upper()
        df = df[df['_type'].isin(['CE', 'PE'])].copy()
        if df.empty:
            return pd.DataFrame()

        # Extract spot from UndrlygPric (underlying price in bhavcopy!)
        spot = 0
        if 'UndrlygPric' in df.columns:
            spot_vals = pd.to_numeric(df['UndrlygPric'], errors='coerce').dropna()
            spot_non_zero = spot_vals[spot_vals > 0]
            if not spot_non_zero.empty:
                spot = float(spot_non_zero.iloc[0])

        return pd.DataFrame({
            'date': date_str,
            'strike': pd.to_numeric(df['StrkPric'], errors='coerce').fillna(0).astype(int),
            'type': df['_type'].values,
            'oi': pd.to_numeric(df['OpnIntrst'], errors='coerce').fillna(0).astype(int),
            'oi_change': pd.to_numeric(df['ChngInOpnIntrst'], errors='coerce').fillna(0).astype(int),
            'close': pd.to_numeric(df['ClsPric'], errors='coerce').fillna(0),
            'expiry': df['XpryDt'].values if 'XpryDt' in df.columns else '',
            'spot': spot,
        }).query('strike > 0').reset_index(drop=True)

    def build_master_dataset(self, raw_data_dict, spot_series):
        """Process all days, extract spot from bhavcopy or spot_series, filter nearby strikes."""
        all_frames = []
        processed = 0
        skipped = 0

        print(f"\n  Processing {len(raw_data_dict)} days of bhavcopy data...")

        for date_str in sorted(raw_data_dict.keys()):
            df = self.process_single_day(raw_data_dict[date_str], date_str)
            if df.empty:
                continue

            # Get spot: prefer bhavcopy UndrlygPric, fallback to spot_series
            spot = 0
            if 'spot' in df.columns:
                spots = df['spot'].unique()
                non_zero = [s for s in spots if s > 0]
                if non_zero:
                    spot = non_zero[0]
            if spot <= 0:
                spot = spot_series.get(date_str, 0)
            if spot <= 0:
                skipped += 1
                continue

            df['spot'] = spot

            # Filter to nearby strikes
            strikes = sorted(df['strike'].unique())
            if not strikes:
                continue
            atm_idx = min(range(len(strikes)), key=lambda i: abs(strikes[i] - spot))
            nearby = strikes[max(0, atm_idx - STRIKE_RANGE):atm_idx + STRIKE_RANGE + 1]
            df = df[df['strike'].isin(nearby)].copy()

            all_frames.append(df)
            processed += 1

        if not all_frames:
            print("  WARNING: No data processed!")
            return pd.DataFrame()

        master = pd.concat(all_frames, ignore_index=True)
        print(f"  Processed {processed} trading days -> {len(master):,} option records")
        if skipped > 0:
            print(f"  Skipped {skipped} days (no spot price)")
        return master


# ═══════════════════════════════════════════════════════════════════
# 3. SIGNAL ENGINE — Replicate OI Pressure Logic
# ═══════════════════════════════════════════════════════════════════
class OISignalEngine:
    """Computes OI Pressure signal from daily OI snapshots."""

    def compute_signal(self, today_df, prev_df, spot):
        if today_df.empty or prev_df.empty or spot <= 0:
            return self._empty_signal()

        today_oi = {(int(r['strike']), r['type']): int(r['oi'])
                    for _, r in today_df.iterrows()}
        prev_oi = {(int(r['strike']), r['type']): int(r['oi'])
                   for _, r in prev_df.iterrows()}

        strikes = sorted(set(s for s, _ in today_oi.keys()))
        if not strikes:
            return self._empty_signal()

        atm_idx = min(range(len(strikes)), key=lambda i: abs(strikes[i] - spot))
        nearby = strikes[max(0, atm_idx - STRIKE_RANGE):atm_idx + STRIKE_RANGE + 1]

        call_oi_added = 0
        call_oi_unwound = 0
        put_oi_added = 0
        put_oi_unwound = 0
        total_ce_chg = 0
        total_pe_chg = 0

        for strike in nearby:
            ce_now = today_oi.get((strike, 'CE'), 0)
            ce_prev = prev_oi.get((strike, 'CE'), 0)
            pe_now = today_oi.get((strike, 'PE'), 0)
            pe_prev = prev_oi.get((strike, 'PE'), 0)

            ce_chg = ce_now - ce_prev
            pe_chg = pe_now - pe_prev
            total_ce_chg += ce_chg
            total_pe_chg += pe_chg

            if strike >= spot:
                if ce_chg > 0: call_oi_added += ce_chg
                elif ce_chg < 0: call_oi_unwound += abs(ce_chg)
            if strike <= spot:
                if pe_chg > 0: put_oi_added += pe_chg
                elif pe_chg < 0: put_oi_unwound += abs(pe_chg)

        bullish_force = put_oi_added + call_oi_unwound
        bearish_force = call_oi_added + put_oi_unwound
        total_force = bullish_force + bearish_force

        score = ((bullish_force - bearish_force) / total_force * 100) if total_force > 0 else 0

        if score > 15: signal = 'BULLISH'
        elif score < -15: signal = 'BEARISH'
        else: signal = 'NEUTRAL'

        chg_pcr = abs(total_pe_chg) / abs(total_ce_chg) if abs(total_ce_chg) > 0 else 0

        return {
            'signal': signal, 'score': round(score, 1),
            'bullish_force': bullish_force, 'bearish_force': bearish_force,
            'call_oi_added': call_oi_added, 'call_oi_unwound': call_oi_unwound,
            'put_oi_added': put_oi_added, 'put_oi_unwound': put_oi_unwound,
            'total_ce_chg': total_ce_chg, 'total_pe_chg': total_pe_chg,
            'net_oi_change': total_ce_chg + total_pe_chg, 'chg_pcr': round(chg_pcr, 3),
        }

    def _empty_signal(self):
        return {
            'signal': 'NEUTRAL', 'score': 0, 'bullish_force': 0, 'bearish_force': 0,
            'call_oi_added': 0, 'call_oi_unwound': 0, 'put_oi_added': 0, 'put_oi_unwound': 0,
            'total_ce_chg': 0, 'total_pe_chg': 0, 'net_oi_change': 0, 'chg_pcr': 0,
        }


# ═══════════════════════════════════════════════════════════════════
# 4. BACKTEST RUNNER
# ═══════════════════════════════════════════════════════════════════
class OIBacktestRunner:
    def __init__(self):
        self.engine = OISignalEngine()

    def run(self, master_df, spot_series):
        dates = sorted(master_df['date'].unique())
        if len(dates) < 2:
            print("  Not enough dates for backtest.")
            return pd.DataFrame()

        spot_dates = sorted(spot_series.keys())
        results = []

        print(f"\n{'='*60}")
        print(f"  RUNNING BACKTEST")
        print(f"  Period: {dates[0]} to {dates[-1]} ({len(dates)} trading days)")
        print(f"{'='*60}")

        for i in range(1, len(dates)):
            today_str = dates[i]
            prev_str = dates[i - 1]

            today_data = master_df[master_df['date'] == today_str]
            prev_data = master_df[master_df['date'] == prev_str]
            spot = spot_series.get(today_str, 0)

            if spot <= 0:
                continue

            sig = self.engine.compute_signal(today_data, prev_data, spot)
            fwd = self._forward_returns(today_str, spot_dates, spot_series)

            results.append({
                'date': today_str, 'spot': round(spot, 2), **sig,
                'fwd_1d_ret': fwd.get(1, np.nan),
                'fwd_3d_ret': fwd.get(3, np.nan),
                'fwd_5d_ret': fwd.get(5, np.nan),
            })

            if i % 100 == 0:
                print(f"  Processed {i}/{len(dates)} days...")

        result_df = pd.DataFrame(results)
        print(f"  Generated {len(result_df)} signals")
        return result_df

    def _forward_returns(self, date_str, sorted_dates, spot_series):
        if date_str not in sorted_dates:
            return {}
        idx = sorted_dates.index(date_str)
        spot_now = spot_series[date_str]
        ret = {}
        for horizon in [1, 3, 5]:
            fwd_idx = idx + horizon
            if fwd_idx < len(sorted_dates):
                ret[horizon] = round((spot_series[sorted_dates[fwd_idx]] / spot_now - 1) * 100, 4)
        return ret


# ═══════════════════════════════════════════════════════════════════
# 5. PERFORMANCE ANALYZER
# ═══════════════════════════════════════════════════════════════════
class PerformanceAnalyzer:
    def analyze(self, results_df):
        if results_df.empty:
            print("No results to analyze.")
            return {}

        print(f"\n{'='*70}")
        print(f"  BACKTEST RESULTS ANALYSIS")
        print(f"{'='*70}")

        sig_counts = results_df['signal'].value_counts()
        total = len(results_df)
        print(f"\n  Signal Distribution ({total} trading days):")
        for sig in ['BULLISH', 'BEARISH', 'NEUTRAL']:
            count = sig_counts.get(sig, 0)
            print(f"    {sig:>8}: {count:>5} ({count/total*100:.1f}%)")

        horizons = {'1d': 'fwd_1d_ret', '3d': 'fwd_3d_ret', '5d': 'fwd_5d_ret'}
        metrics = {}

        print(f"\n  {'Signal':>10} | {'Horizon':>8} | {'Count':>6} | {'Win%':>6} | {'Avg Ret':>9} | {'Med Ret':>9} | {'Sharpe':>7}")
        print(f"  {'-'*10} | {'-'*8} | {'-'*6} | {'-'*6} | {'-'*9} | {'-'*9} | {'-'*7}")

        for sig in ['BULLISH', 'BEARISH']:
            sig_df = results_df[results_df['signal'] == sig]
            if sig_df.empty:
                continue
            for h_name, h_col in horizons.items():
                returns = sig_df[h_col].dropna()
                if len(returns) < 5:
                    continue
                wins = (returns > 0).sum() if sig == 'BULLISH' else (returns < 0).sum()
                win_rate = wins / len(returns) * 100
                avg_ret = returns.mean()
                med_ret = returns.median()
                if sig == 'BEARISH':
                    sharpe = (-returns.mean()) / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
                else:
                    sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0

                print(f"  {sig:>10} | {h_name:>8} | {len(returns):>6} | {win_rate:>5.1f}% | {avg_ret:>+8.4f}% | {med_ret:>+8.4f}% | {sharpe:>+6.2f}")
                metrics[f'{sig}_{h_name}_win_rate'] = win_rate
                metrics[f'{sig}_{h_name}_avg_ret'] = avg_ret
                metrics[f'{sig}_{h_name}_count'] = len(returns)

        # Score vs return correlation
        print(f"\n  Score-Return Correlation:")
        for h_name, h_col in horizons.items():
            valid = results_df[['score', h_col]].dropna()
            if len(valid) > 20:
                corr = valid['score'].corr(valid[h_col])
                print(f"    Score vs {h_name} return: r = {corr:+.3f}")
                metrics[f'score_{h_name}_corr'] = corr

        # Monthly breakdown
        print(f"\n  Monthly Signal Win Rate (5d horizon):")
        results_df = results_df.copy()
        results_df['month'] = pd.to_datetime(results_df['date']).dt.to_period('M')

        print(f"  {'Month':>10} | {'BULL Win%':>10} | {'BEAR Win%':>10} | {'Signals':>8}")
        print(f"  {'-'*10} | {'-'*10} | {'-'*10} | {'-'*8}")

        for month in sorted(results_df['month'].unique()):
            grp = results_df[results_df['month'] == month]
            bull = grp[grp['signal'] == 'BULLISH']
            bear = grp[grp['signal'] == 'BEARISH']
            bull_wr = (bull['fwd_5d_ret'] > 0).mean() * 100 if len(bull) > 0 else float('nan')
            bear_wr = (bear['fwd_5d_ret'] < 0).mean() * 100 if len(bear) > 0 else float('nan')
            n_sigs = len(bull) + len(bear)
            if n_sigs > 0:
                bull_str = f"{bull_wr:.1f}%" if not np.isnan(bull_wr) else "  N/A"
                bear_str = f"{bear_wr:.1f}%" if not np.isnan(bear_wr) else "  N/A"
                print(f"  {str(month):>10} | {bull_str:>10} | {bear_str:>10} | {n_sigs:>8}")

        # Verdict
        print(f"\n{'='*70}")
        bull_5d = metrics.get('BULLISH_5d_win_rate', 0)
        bear_5d = metrics.get('BEARISH_5d_win_rate', 0)
        avg_wr = (bull_5d + bear_5d) / 2 if bull_5d > 0 and bear_5d > 0 else 0

        if avg_wr > 58: verdict = "EXCELLENT -- Signal has strong predictive power"
        elif avg_wr > 52: verdict = "GOOD -- Signal beats random, usable edge"
        elif avg_wr > 48: verdict = "MARGINAL -- Slight edge, combine with other factors"
        else: verdict = "WEAK -- Signal alone is not reliable"

        print(f"  VERDICT: {verdict}")
        print(f"  Average 5d Win Rate: {avg_wr:.1f}%")
        print(f"{'='*70}")

        return metrics


# ═══════════════════════════════════════════════════════════════════
# 6. REPORT GENERATOR
# ═══════════════════════════════════════════════════════════════════
class ReportGenerator:
    def __init__(self):
        os.makedirs(RESULTS_DIR, exist_ok=True)

    def generate(self, results_df, metrics):
        if results_df.empty:
            return

        csv_path = os.path.join(RESULTS_DIR, f"oi_backtest_{datetime.now().strftime('%Y%m%d_%H%M')}.csv")
        results_df.to_csv(csv_path, index=False)
        print(f"\n  Results saved to: {csv_path}")

        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(2, 2, figsize=(16, 10))
            fig.suptitle('OI Pressure Signal -- Backtest Results', fontsize=14, fontweight='bold')

            # Chart 1: 5d return distribution
            ax1 = axes[0, 0]
            bull = results_df[results_df['signal'] == 'BULLISH']['fwd_5d_ret'].dropna()
            bear = results_df[results_df['signal'] == 'BEARISH']['fwd_5d_ret'].dropna()
            if not bull.empty:
                ax1.hist(bull, bins=30, alpha=0.6, color='green', label=f'BULL (n={len(bull)})')
            if not bear.empty:
                ax1.hist(bear, bins=30, alpha=0.6, color='red', label=f'BEAR (n={len(bear)})')
            ax1.axvline(0, color='black', linewidth=1)
            ax1.set_title('5d Forward Return Distribution by Signal')
            ax1.set_xlabel('5-Day Return (%)')
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # Chart 2: Score vs 5d Return
            ax2 = axes[0, 1]
            valid = results_df[['score', 'fwd_5d_ret']].dropna()
            if not valid.empty:
                colors = ['green' if s > 15 else 'red' if s < -15 else 'gray' for s in valid['score']]
                ax2.scatter(valid['score'], valid['fwd_5d_ret'], c=colors, alpha=0.3, s=10)
                z = np.polyfit(valid['score'], valid['fwd_5d_ret'], 1)
                p = np.poly1d(z)
                x_line = np.linspace(valid['score'].min(), valid['score'].max(), 100)
                ax2.plot(x_line, p(x_line), 'b-', linewidth=2, alpha=0.7)
                corr = valid['score'].corr(valid['fwd_5d_ret'])
                ax2.set_title(f'Score vs 5d Return (r={corr:.3f})')
            ax2.set_xlabel('OI Pressure Score')
            ax2.set_ylabel('5d Forward Return (%)')
            ax2.axhline(0, color='black', linewidth=0.5)
            ax2.axvline(0, color='black', linewidth=0.5)
            ax2.grid(True, alpha=0.3)

            # Chart 3: Rolling win rate
            ax3 = axes[1, 0]
            directional = results_df[results_df['signal'] != 'NEUTRAL'].copy()
            if len(directional) > 50:
                directional['correct'] = (
                    ((directional['signal'] == 'BULLISH') & (directional['fwd_5d_ret'] > 0)) |
                    ((directional['signal'] == 'BEARISH') & (directional['fwd_5d_ret'] < 0))
                ).astype(int)
                rolling_wr = directional['correct'].rolling(50, min_periods=20).mean() * 100
                ax3.plot(range(len(rolling_wr)), rolling_wr.values, color='blue', linewidth=1)
                ax3.axhline(50, color='red', linestyle='--', alpha=0.7, label='50% (random)')
                ax3.set_title('Rolling 50-Signal Win Rate')
                ax3.set_ylabel('Win Rate (%)')
                ax3.legend()
                ax3.grid(True, alpha=0.3)

            # Chart 4: Monthly signal counts
            ax4 = axes[1, 1]
            results_df['date_dt'] = pd.to_datetime(results_df['date'])
            try:
                monthly_signals = results_df.groupby([results_df['date_dt'].dt.to_period('M'), 'signal']).size().unstack(fill_value=0)
                if not monthly_signals.empty:
                    color_map = {'BULLISH': 'green', 'BEARISH': 'red', 'NEUTRAL': 'gray'}
                    avail_colors = {k: v for k, v in color_map.items() if k in monthly_signals.columns}
                    monthly_signals.plot(kind='bar', stacked=True, ax=ax4, color=avail_colors)
                    ax4.set_title('Monthly Signal Distribution')
                    ax4.set_xlabel('Month')
                    ax4.set_ylabel('Count')
                    ax4.tick_params(axis='x', rotation=45)
            except Exception:
                pass

            plt.tight_layout()
            chart_path = os.path.join(RESULTS_DIR, f"oi_backtest_charts_{datetime.now().strftime('%Y%m%d_%H%M')}.png")
            plt.savefig(chart_path, dpi=150, bbox_inches='tight')
            print(f"  Charts saved to: {chart_path}")
            plt.show()

        except ImportError:
            print("  Matplotlib not available for charts.")
        except Exception as e:
            print(f"  Chart generation error: {e}")


# ═══════════════════════════════════════════════════════════════════
# MAIN ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════
class OIBacktester:
    def __init__(self):
        self.fetcher = BhavcopyFetcher()
        self.processor = OIDataProcessor()
        self.runner = OIBacktestRunner()
        self.analyzer = PerformanceAnalyzer()
        self.reporter = ReportGenerator()

    def _get_spot_series(self, start_date, end_date):
        """Get spot prices from Fyers API (cached)."""
        print(f"\n  Fetching NIFTY Spot Prices...")

        spot_cache = os.path.join(CACHE_DIR, 'spot_prices.csv')
        if os.path.exists(spot_cache):
            cached = pd.read_csv(spot_cache)
            cached['date'] = pd.to_datetime(cached['date']).dt.strftime('%Y-%m-%d')
            if len(cached) > 100:
                print(f"  Using cached spot prices ({len(cached)} days)")
                return dict(zip(cached['date'], cached['close']))

        try:
            from FyersAuth import FyersAuthenticator
            auth = FyersAuthenticator("QUTT4YYMIG-100", "ZG0WN2NL1B", "http://127.0.0.1:3000/callback")
            fyers = auth.get_fyers_instance()
            if fyers:
                all_spots = {}
                chunk_start = start_date
                while chunk_start < end_date:
                    chunk_end = min(chunk_start + timedelta(days=90), end_date)
                    data = {
                        "symbol": "NSE:NIFTY50-INDEX", "resolution": "D", "date_format": "1",
                        "range_from": chunk_start.strftime("%Y-%m-%d"),
                        "range_to": chunk_end.strftime("%Y-%m-%d"), "cont_flag": "1"
                    }
                    r = fyers.history(data=data)
                    if r.get('s') == 'ok':
                        for candle in r['candles']:
                            dt = datetime.fromtimestamp(candle[0])
                            all_spots[dt.strftime('%Y-%m-%d')] = candle[4]
                    chunk_start = chunk_end + timedelta(days=1)
                    time.sleep(0.3)

                if all_spots:
                    spot_df = pd.DataFrame(list(all_spots.items()), columns=['date', 'close'])
                    spot_df.to_csv(spot_cache, index=False)
                    print(f"  Fetched {len(all_spots)} spot prices via Fyers API")
                    return all_spots
        except Exception as e:
            print(f"  Fyers API unavailable: {e}")

        return {}

    def run(self, years_back=2):
        print("\n" + "="*60)
        print("  OI PRESSURE BACKTESTING ENGINE")
        print("="*60)

        end_date = datetime.now()
        start_date = end_date - timedelta(days=int(years_back * 365))

        print(f"  Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        print(f"  Underlying: {UNDERLYING}")

        # Step 1: Get spot prices
        spot_series = self._get_spot_series(start_date, end_date)
        if not spot_series:
            print("  WARNING: No spot prices from Fyers. Will rely on bhavcopy UndrlygPric.")
            spot_series = {}

        # Step 2: Download bhavcopy
        raw_data = self.fetcher.fetch_date_range(start_date, end_date)
        if not raw_data:
            print("ERROR: No bhavcopy data.")
            return

        # Step 3: Process
        master_df = self.processor.build_master_dataset(raw_data, spot_series)
        if master_df.empty:
            print("ERROR: No valid data after processing.")
            return

        # Enrich spot_series from bhavcopy data (for forward return calculation)
        for date_str in master_df['date'].unique():
            if date_str not in spot_series:
                day_data = master_df[master_df['date'] == date_str]
                if not day_data.empty:
                    s = day_data['spot'].iloc[0]
                    if s > 0:
                        spot_series[date_str] = s

        # Step 4: Run backtest
        results_df = self.runner.run(master_df, spot_series)
        if results_df.empty:
            print("ERROR: No results.")
            return

        # Step 5: Analyze
        metrics = self.analyzer.analyze(results_df)

        # Step 6: Report
        self.reporter.generate(results_df, metrics)

        return results_df, metrics


def main():
    print("\n" + "="*60)
    print("  OI PRESSURE BACKTESTER")
    print("="*60)
    print("\n  1. Full Backtest (2 years)")
    print("  2. Custom Date Range")
    print("  3. Quick Backtest (6 months)")
    print("  4. Exit")

    choice = input("\n  Select: ").strip()
    bt = OIBacktester()

    if choice == '1':
        bt.run(years_back=2)
    elif choice == '2':
        try:
            start = input("  Start date (YYYY-MM-DD): ").strip()
            end = input("  End date (YYYY-MM-DD): ").strip()
            start_dt = datetime.strptime(start, '%Y-%m-%d')
            end_dt = datetime.strptime(end, '%Y-%m-%d')
            years = (end_dt - start_dt).days / 365
            bt.run(years_back=years)
        except Exception as e:
            print(f"  Error: {e}")
    elif choice == '3':
        bt.run(years_back=0.5)
    elif choice == '4':
        return


if __name__ == "__main__":
    main()
