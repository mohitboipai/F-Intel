"""
run_backtest.py
===============
F-Intel Sell Signal Backtester — Command-line Runner

Pipeline:
  1. Ask for date range (default: past 12 months)
  2. Ask which strategy (or ALL)
  3. Ask which signal type (or ALL)
  4. Ask for stop-loss multiplier
  5. Download bhavcopy data (print one dot per day)
  6. Fetch/load 1-min Nifty data
  7. Run signal scanner
  8. Run backtest for each selected combination
  9. Print summary table
  10. Generate + open HTML report

Usage:
    python run_backtest.py
"""

import os
import sys
import sqlite3
import traceback
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Ensure repo root is on path
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from BhavCopyEngine        import BhavCopyEngine
from MinuteDataFetcher     import MinuteDataFetcher
from OptionPriceReconstructor import OptionPriceReconstructor
from SellSignalScanner     import SellSignalScanner
from SellSignalBacktester  import SellSignalBacktester
from BacktestReporter      import BacktestReporter

# ── config ────────────────────────────────────────────────────────────────────
STRATEGIES = ['SHORT_STRADDLE', 'SHORT_STRANGLE', 'IRON_CONDOR', 'BEAR_CALL_SPREAD']
SIGNALS    = ['VRP_SELL', 'IV_PERCENTILE', 'GEX_COMPRESSION', 'COMBINED']


def _prompt(msg: str, default: str = '') -> str:
    val = input(msg).strip()
    return val if val else default


def _print_header():
    print()
    print('=' * 65)
    print('  [*] F-INTEL SELL SIGNAL BACKTESTER')
    print('  NIFTY Options | Lot Size: 65 | Risk-Free: 7%')
    print('=' * 65)
    print()


def _ask_dates() -> tuple[datetime, datetime]:
    default_end   = datetime.now()
    default_start = default_end - timedelta(days=365)

    print('  Date Range (press Enter for last 12 months)')
    start_str = _prompt(f'  Start date [YYYY-MM-DD] ({default_start.strftime("%Y-%m-%d")}): ',
                        default_start.strftime('%Y-%m-%d'))
    end_str   = _prompt(f'  End date   [YYYY-MM-DD] ({default_end.strftime("%Y-%m-%d")}): ',
                        default_end.strftime('%Y-%m-%d'))

    try:
        start_dt = datetime.strptime(start_str, '%Y-%m-%d')
        end_dt   = datetime.strptime(end_str,   '%Y-%m-%d')
    except ValueError:
        print('  Invalid date format. Using defaults.')
        start_dt, end_dt = default_start, default_end

    return start_dt, end_dt


def _ask_strategy() -> list[str]:
    print()
    print('  Strategy:')
    for i, s in enumerate(STRATEGIES, 1):
        print(f'    {i}. {s}')
    print(f'    5. ALL')
    choice = _prompt('  Select [1-5] (default=5): ', '5')
    if choice == '5':
        return list(STRATEGIES)
    try:
        idx = int(choice) - 1
        if 0 <= idx < len(STRATEGIES):
            return [STRATEGIES[idx]]
    except ValueError:
        pass
    return list(STRATEGIES)


def _ask_signal() -> list[str]:
    print()
    print('  Signal Type:')
    for i, s in enumerate(SIGNALS, 1):
        print(f'    {i}. {s}')
    print(f'    5. ALL')
    choice = _prompt('  Select [1-5] (default=5): ', '5')
    if choice == '5':
        return list(SIGNALS)
    try:
        idx = int(choice) - 1
        if 0 <= idx < len(SIGNALS):
            return [SIGNALS[idx]]
    except ValueError:
        pass
    return list(SIGNALS)


def _ask_sl() -> float:
    print()
    choice = _prompt('  Stop-loss multiplier (default=2.0): ', '2.0')
    try:
        return float(choice)
    except ValueError:
        return 2.0


def _download_bhavcopy(engine: BhavCopyEngine, start_dt: datetime, end_dt: datetime):
    """Download with dot-per-day progress indicator."""
    print()
    print('  Downloading bhavcopy data ', end='', flush=True)
    loaded = engine.load_range(start_dt.date(), end_dt.date(), verbose=True)
    print(f'  [OK] {loaded} trading day(s) loaded into BhavCopyEngine')


def _fetch_minute_data(start_dt: datetime, end_dt: datetime) -> pd.DataFrame:
    print()
    print('  Fetching / loading 1-min NIFTY data ...')
    fetcher = MinuteDataFetcher()
    try:
        df = fetcher.get(start_dt.date(), end_dt.date())
        if df.empty:
            print('  [FAIL] No minute data returned (check Fyers auth).')
        else:
            print(f'  [OK] {len(df):,} minute bars loaded '
                  f'({df.index[0].date()} to {df.index[-1].date()})')
        return df
    except Exception as exc:
        print(f'  [FAIL] Minute data fetch error: {exc}')
        return pd.DataFrame()


def _run_scanner(engine: BhavCopyEngine, minute_df: pd.DataFrame,
                 start_dt: datetime, end_dt: datetime) -> pd.DataFrame:
    print()
    print('  Running sell signal scanner ...')
    scanner = SellSignalScanner(engine)
    signals_df = scanner.scan_year(minute_df, start_dt.date(), end_dt.date())

    if signals_df.empty:
        print('  [!] No signals found in the selected date range.')
    else:
        n = len(signals_df)
        counts = signals_df['signal_type'].value_counts().to_dict()
        print(f'  [OK] {n} signal event(s) found:')
        for sname, cnt in sorted(counts.items()):
            print(f'      {sname}: {cnt}')
    return signals_df


def _run_backtest(
    engine,
    reconstructor,
    signals_df,
    minute_df,
    selected_strategies,
    selected_signals,
    sl_mult,
) -> dict:
    """Returns dict: (strategy, signal_type) → stats."""
    print()
    print('  Running backtest ...')

    summary = {}
    bt = SellSignalBacktester(engine, reconstructor)

    for strategy in selected_strategies:
        for sig_type in selected_signals:
            mask = (signals_df['signal_type'] == sig_type) if not signals_df.empty else pd.Series([], dtype=bool)
            sig_subset = signals_df[mask] if not signals_df.empty else pd.DataFrame()

            try:
                result = bt.run(
                    signals_df           = sig_subset,
                    minute_df            = minute_df,
                    strategy             = strategy,
                    stop_loss_multiplier = sl_mult,
                )
                summary[(strategy, sig_type)] = result['stats']
                n = result['stats'].get('trade_count', 0)
                print(f'    {strategy} / {sig_type}: {n} trade(s)')
            except Exception as exc:
                print(f'    FAIL {strategy} / {sig_type}: {exc}')
                summary[(strategy, sig_type)] = {}

    return summary


def _print_summary_table(summary: dict):
    """Print a formatted terminal table of results."""
    print()
    print('=' * 95)
    print(f"  {'STRATEGY':<22} {'SIGNAL':<18} {'TRADES':>6} {'WIN%':>6} "
          f"{'TOTAL P&L':>12} {'AVG P&L':>10} {'MAX DD':>12} {'SHARPE':>7} {'KELLY':>6}")
    print('=' * 95)

    for (strategy, sig_type), s in sorted(summary.items()):
        if not s:
            print(f"  {strategy:<22} {sig_type:<18} {'--':>6}")
            continue
        print(
            f"  {strategy:<22} {sig_type:<18}"
            f"  {s.get('trade_count',0):>5}"
            f"  {s.get('win_rate',0):>5.1f}%"
            f"  Rs.{s.get('total_pnl',0):>9,.0f}"
            f"  Rs.{s.get('avg_pnl',0):>7,.0f}"
            f"  Rs.{s.get('max_drawdown',0):>9,.0f}"
            f"  {s.get('sharpe',0):>6.2f}"
            f"  {s.get('kelly',0):>5.1f}%"
        )
    print('=' * 95)


def main():
    _print_header()

    try:
        start_dt, end_dt = _ask_dates()
        print(f'\n  Period: {start_dt.date()} to {end_dt.date()} '
              f'({(end_dt - start_dt).days} calendar days)')

        selected_strategies = _ask_strategy()
        selected_signals    = _ask_signal()
        sl_mult             = _ask_sl()

        print(f'\n  Strategies : {", ".join(selected_strategies)}')
        print(f'  Signals    : {", ".join(selected_signals)}')
        print(f'  SL mult    : {sl_mult}x')

        # ── Step 5: Bhavcopy ──────────────────────────────────────────────
        engine = BhavCopyEngine()
        _download_bhavcopy(engine, start_dt, end_dt)

        # ── Step 6: Minute data ───────────────────────────────────────────
        minute_df = _fetch_minute_data(start_dt, end_dt)

        # Build daily closes for HV fallback
        if not minute_df.empty:
            daily_closes = minute_df.groupby(minute_df.index.date)['close'].last()
            daily_closes.index = pd.to_datetime([str(d) for d in daily_closes.index])
        else:
            daily_closes = pd.Series()

        # ── Step 7: Reconstructor (warm IV cache) ─────────────────────────
        reconstructor = OptionPriceReconstructor(bhav_engine=engine)
        reconstructor.warm_iv_cache(engine.get_all_dates(), daily_closes)

        # ── Step 8: Signal scanner ────────────────────────────────────────
        if minute_df.empty:
            print('\n  [!] Skipping signal scanner (no minute data).')
            signals_df = pd.DataFrame()
        else:
            signals_df = _run_scanner(engine, minute_df, start_dt, end_dt)

        # ── Step 9: Backtest ──────────────────────────────────────────────
        if signals_df.empty or minute_df.empty:
            print('\n  [!] No signals or minute data -- backtest skipped.')
            summary = {}
        else:
            summary = _run_backtest(
                engine, reconstructor,
                signals_df, minute_df,
                selected_strategies, selected_signals,
                sl_mult,
            )

        # -- Step 10: Summary table
        if summary:
            _print_summary_table(summary)

        # -- Step 11: HTML report
        print()
        print('  Generating HTML report ...')
        reporter = BacktestReporter()
        report_path = reporter.generate()
        print(f'  [OK] Report saved to: {report_path}')

        print()
        print('  Done.')
        print()

    except KeyboardInterrupt:
        print('\n\n  Interrupted by user.')
    except Exception as exc:
        print(f'\n  [FAIL] Fatal error: {exc}')
        traceback.print_exc()


if __name__ == '__main__':
    main()
