"""Diagnose exact unicode characters that fail cp1252."""
import os

files = [
    'BacktestSignalExtractor.py', 'ExpiryCalendar.py', 'ShiftEvaluator.py',
    'PositionLedger.py', 'WeeklyDynamicBacktester.py',
    'WeeklyBacktestReporter.py', 'run_dynamic_backtest.py',
]

for f in files:
    with open(f, encoding='utf-8') as fh:
        src = fh.read()
    for i, c in enumerate(src):
        try:
            c.encode('cp1252')
        except UnicodeEncodeError:
            # Show context
            snippet = src[max(0, i-20):i+30].replace('\n', ' ')
            print(f'{f} pos={i} char={repr(c)} (U+{ord(c):04X}) ctx: ...{snippet}...')
            break  # just show first per file
