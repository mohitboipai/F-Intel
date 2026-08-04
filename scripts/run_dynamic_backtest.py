"""
run_dynamic_backtest.py
=======================
Interactive CLI runner for the F-Intel Weekly Dynamic Backtest.

Prompts for:
  1. Date range (default: last 12 months)
  2. Stop-loss multiplier (default: 2.0)
  3. CE/PE shift threshold % (default: 0.8)
  4. Show shift log during simulation (Y/N, default: Y)

Usage
-----
    python run_dynamic_backtest.py
"""

import os
import sys
import subprocess
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── helpers ───────────────────────────────────────────────────────────────────

def _ask(prompt: str, default: str) -> str:
    try:
        val = input(f'  {prompt} [{default}]: ').strip()
        return val if val else default
    except (EOFError, KeyboardInterrupt):
        return default


def _parse_date(s: str):
    for fmt in ('%Y-%m-%d', '%d-%m-%Y', '%d/%m/%Y'):
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            pass
    raise ValueError(f'Cannot parse date: {s}')


def _banner():
    print()
    print('=' * 65)
    print('  [*] F-INTEL WEEKLY DYNAMIC BACKTEST')
    print('  NIFTY Options | Lot Size: 75 | Risk-Free: 7%')
    print('  Primary: Short Strangle + Dynamic Position Shifting')
    print('=' * 65)
    print()


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    _banner()

    # ── Date range ────────────────────────────────────────────────────────────
    today   = datetime.now().date()
    default_start = (today - timedelta(days=365)).strftime('%Y-%m-%d')
    default_end   = today.strftime('%Y-%m-%d')

    print('  Date Range (press Enter for last 12 months)')
    start_str = _ask(f'Start date [YYYY-MM-DD] ({default_start})', default_start)
    end_str   = _ask(f'End date   [YYYY-MM-DD] ({default_end})',   default_end)

    try:
        start_date = _parse_date(start_str)
        end_date   = _parse_date(end_str)
    except ValueError as e:
        print(f'  [ERROR] {e}')
        sys.exit(1)

    print(f'\n  Period: {start_date} to {end_date} '
          f'({(end_date - start_date).days} calendar days)')

    # ── Parameters ────────────────────────────────────────────────────────────
    sl_str    = _ask('Stop-loss multiplier (default=2.0)', '2.0')
    thr_str   = _ask('Shift threshold % distance (default=0.8)', '0.8')
    log_str   = _ask('Show shift log? [Y/N] (default=Y)', 'Y')

    try:
        sl_mult    = float(sl_str)
        shift_thr  = float(thr_str)
    except ValueError:
        print('  [ERROR] Invalid numeric input.')
        sys.exit(1)

    show_log = log_str.strip().upper() not in ('N', 'NO')

    print()
    print(f'  Stop-loss mult : {sl_mult}x')
    print(f'  Shift threshold: {shift_thr}%')
    print(f'  Show shift log : {"Yes" if show_log else "No"}')
    print()

    # ── Run backtest ──────────────────────────────────────────────────────────
    print('  Starting simulation ...')
    print('-' * 65)

    try:
        from WeeklyDynamicBacktester import WeeklyDynamicBacktester
        bt = WeeklyDynamicBacktester(
            start_date=str(start_date),
            end_date=str(end_date),
            sl_mult=sl_mult,
            shift_threshold_pct=shift_thr,
            show_shift_log=show_log,
        )
        bt.run()
    except Exception as e:
        print(f'\n  [ERROR] Backtest failed: {e}')
        import traceback; traceback.print_exc()
        sys.exit(1)

    # ── Summary table ─────────────────────────────────────────────────────────
    print()
    print('=' * 85)
    print(f'  {"STRATEGY":<22} {"POSITIONS":>10} {"WIN%":>8} {"TOTAL P&L":>13} '
          f'{"AVG P&L":>11} {"SHIFTS":>7} {"SHARPE":>8} {"KELLY":>7}')
    print('=' * 85)

    for strat in ('strangle', 'straddle', 'condor', 'callspread'):
        try:
            s = bt.aggregate_stats(strat)
            if not s or s.get('positions', 0) == 0:
                print(f'  {strat.upper():<22} {"--":>10}')
                continue
            pnl_str = f'Rs.{s["total_pnl_rupees"]:>10,.0f}'
            avg_str = f'Rs.{s["avg_pnl_rupees"]:>8,.0f}'
            print(
                f'  {s["strategy"]:<22} {s["positions"]:>10} '
                f'{s["win_rate_pct"]:>7.1f}% {pnl_str:>13} {avg_str:>11} '
                f'{s["avg_shifts"]:>7.2f} {s["sharpe"]:>8.2f} {s["kelly_pct"]:>6.1f}%'
            )
        except Exception:
            pass

    print('=' * 85)

    # ── Shift diagnostic ──────────────────────────────────────────────────────
    try:
        s = bt.aggregate_stats('strangle')
        if s.get('positions', 0) > 0:
            print()
            print('  Short Strangle Shift Breakdown:')
            print(f'    Never shifted   : {s["no_shift_count"]} positions | Avg P&L: Rs.{s["no_shift_avg_pnl"]:,.0f}')
            print(f'    Shifted once    : {s["once_shift_count"]} positions | Avg P&L: Rs.{s["once_shift_avg_pnl"]:,.0f}')
            print(f'    Shifted 2+      : {s["multi_shift_count"]} positions | Avg P&L: Rs.{s["multi_shift_avg_pnl"]:,.0f}')

            if s['no_shift_count'] + s['once_shift_count'] + s['multi_shift_count'] == s['positions']:
                if s['no_shift_count'] == s['positions']:
                    # Zero shifts -- print diagnostic
                    print()
                    print('  [DIAG] Zero shifts occurred. Reading minimum distances from DB ...')
                    import sqlite3
                    DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                           'data', 'backtest_results.db')
                    conn = sqlite3.connect(DB_PATH)
                    lt = None
                    try:
                        import pandas as pd
                        lt = pd.read_sql_query('SELECT * FROM leg_transactions', conn)
                    except Exception:
                        pass
                    conn.close()
                    if lt is not None and not lt.empty:
                        # Try to estimate min distances from opening transactions
                        opens = lt[lt['time_type'] == 'OPEN'][lt['shift_number'] == 0]
                        if not opens.empty:
                            print('  Check: are CE/PE strikes 2% from spot? Sample entries:')
                            for _, r in opens.head(6).iterrows():
                                print(f'    {r["position_id"]} {r["option_type"]} strike={r["strike"]:.0f}')
                    print(f'  Threshold is currently {shift_thr}%. '
                          f'Try lowering to 1.5% if market stayed rangebound.')
    except Exception:
        pass

    # ── Expiry calendar verification ──────────────────────────────────────────
    try:
        print()
        print('  Expiry calendar weekday counts:')
        wstats = bt.cal.weekday_stats()
        for wday, cnt in sorted(wstats.items()):
            print(f'    {wday}: {cnt}')
    except Exception:
        pass

    # ── Generate report ───────────────────────────────────────────────────────
    print()
    from WeeklyBacktestReporter import WeeklyBacktestReporter
    reporter = WeeklyBacktestReporter()
    report_path = reporter.generate()
    print(f'  [OK] Report saved to: {report_path}')

    # ── Offer to open ─────────────────────────────────────────────────────────
    try:
        open_it = _ask('Open report in browser? [Y/N]', 'Y')
        if open_it.strip().upper() in ('Y', 'YES'):
            import subprocess
            subprocess.Popen(['start', report_path], shell=True)
    except Exception:
        pass

    print()
    print('  Done.')
    print()


if __name__ == '__main__':
    main()
