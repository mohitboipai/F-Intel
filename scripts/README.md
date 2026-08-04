# scripts/

Standalone utility and analysis scripts. These are **not imported** by the main platform — run them directly from the project root.

| Script | Purpose |
|--------|---------|
| `analyze_logs.py` | Parse and summarize `fyersRequests.log` |
| `analyze_logs_interactive.py` | Interactive log explorer |
| `check_db.py` | Quick SQLite DB inspection tool |
| `generate_icon.py` | Generate `static/icon.png` (run once) |
| `reconstruct_sell_zones.py` | Rebuild sell zone Excel from BhavCopy data |
| `run_backtest.py` | Run SellSignal backtest (standalone) |
| `run_dynamic_backtest.py` | Run WeeklyDynamic backtest (standalone) |
| `verify_dynamic_backtest.py` | Validate dynamic backtest results |
| `visualize_backtest_ohlc.py` | Plot OHLC backtest results |
| `test_lstm.py` | Quick LSTM model smoke test |

## Usage

Run from the **project root** so relative imports work:

```powershell
# From c:\Users\User\Dhan API\
.\.venv\Scripts\python.exe scripts\run_backtest.py
.\.venv\Scripts\python.exe scripts\check_db.py
```
