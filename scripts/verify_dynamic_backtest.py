"""Quick 3-month verification of the dynamic backtest pipeline."""
import sys
import os
import sqlite3
import pandas as pd
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

start = (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d')
end   = datetime.now().strftime('%Y-%m-%d')
print(f'Test window: {start} -> {end}')

from WeeklyDynamicBacktester import WeeklyDynamicBacktester

bt = WeeklyDynamicBacktester(
    start_date=start, end_date=end,
    sl_mult=2.0, shift_threshold_pct=0.8, show_shift_log=True
)
bt.run()

print()
print('=== AGGREGATE STATS ===')
for strat in ('strangle', 'straddle', 'condor', 'callspread'):
    s = bt.aggregate_stats(strat)
    if s and s.get('positions', 0) > 0:
        print(f'{strat.upper():<14} pos={s["positions"]} win%={s["win_rate_pct"]} '
              f'total=Rs.{s["total_pnl_rupees"]:,.0f} shifts={s["avg_shifts"]}')
    else:
        print(f'{strat.upper():<14} no positions')

print()
print('=== EXPIRY WEEKDAY COUNTS ===')
if bt.cal:
    for day, cnt in sorted(bt.cal.weekday_stats().items()):
        print(f'  {day}: {cnt}')

print()
print('=== P&L SPOT CHECK (first 5 strangle positions) ===')
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'backtest_results.db')
conn = sqlite3.connect(DB_PATH)
try:
    ps = pd.read_sql_query('SELECT position_id, final_pnl_rupees FROM position_summaries LIMIT 5', conn)
    lt = pd.read_sql_query('SELECT position_id, cashflow_rupees FROM leg_transactions', conn)
    for _, row in ps.iterrows():
        pid      = row['position_id']
        computed = lt[lt['position_id'] == pid]['cashflow_rupees'].sum()
        match    = 'OK' if abs(computed - row['final_pnl_rupees']) < 1 else 'MISMATCH'
        print(f'  {pid}: summary=Rs.{row["final_pnl_rupees"]:.0f}  '
              f'legs_sum=Rs.{computed:.0f}  [{match}]')
except Exception as e:
    print(f'  DB check error: {e}')
conn.close()

print()
print('Generating HTML report ...')
from WeeklyBacktestReporter import WeeklyBacktestReporter
rpath = WeeklyBacktestReporter().generate()
print(f'Report: {rpath}')
print('Done.')
