"""Fix all non-cp1252 characters in the new backtest files."""
import os

CHAR_MAP = {
    '\u2192': '->',
    '\u2190': '<-',
    '\u2014': '--',
    '\u2013': '-',
    '\u2018': "'",
    '\u2019': "'",
    '\u201c': '"',
    '\u201d': '"',
    '\u00bb': '>>',
    '\u2026': '...',
    '\u2605': '*',
    '\u2264': '<=',
    '\u2265': '>=',
}

files = [
    'BacktestSignalExtractor.py', 'ExpiryCalendar.py', 'ShiftEvaluator.py',
    'PositionLedger.py', 'WeeklyDynamicBacktester.py',
    'WeeklyBacktestReporter.py', 'run_dynamic_backtest.py',
]

for f in files:
    with open(f, encoding='utf-8') as fh:
        src = fh.read()
    orig = src
    for char, repl in CHAR_MAP.items():
        src = src.replace(char, repl)
    with open(f, 'w', encoding='utf-8') as fh:
        fh.write(src)
    try:
        src.encode('cp1252')
        if src != orig:
            print(f'Fixed: {f}')
        else:
            print(f'Clean: {f}')
    except UnicodeEncodeError as e:
        print(f'STILL BROKEN {f}: {e}')

print('Done.')
