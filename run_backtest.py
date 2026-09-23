"""
run_backtest.py
===============
Command-line runner and optimizer for Options Backtesting.

Usage Examples:
    # 1. Backtest Option Buyer Radar across 1 year (ATM Strike)
    python run_backtest.py --strategy RADAR_ATM --days 365 --sl 0.40 --target 0.80

    # 2. Backtest Option Buyer Radar across 1 year (OTM Momentum Strike)
    python run_backtest.py --strategy RADAR_OTM --days 365 --sl 0.50 --target 1.20

    # 3. Run GEX Move-Size Backtest (Big vs Small move classification)
    python run_backtest.py --gex-moves --days 365

    # 4. Compare Buyer Radar against Seller Strategies
    python run_backtest.py --compare --days 365

    # 5. Export complete granular trade log to CSV
    python run_backtest.py --strategy RADAR_ATM --days 365 --export trades.csv
"""

import os
import sys
import argparse
import json
import pandas as pd

if sys.stdout and getattr(sys.stdout, 'encoding', '').lower() != 'utf-8':
    try:
        reconfig = getattr(sys.stdout, 'reconfigure', None)
        if callable(reconfig):
            reconfig(encoding='utf-8')
    except Exception:
        pass

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from HistoricalDataManager import HistoricalDataManager
from StrategyBacktestEngine import StrategyBacktestEngine, OptionBuyerRadarStrategy, ShortStraddleStrategy
from GexMoveForecaster import GexBacktestEngine


def print_banner(title: str):
    print("\n" + "═" * 70)
    print(f"  {title}".center(70))
    print("═" * 70)


def format_currency(val: float) -> str:
    sign = "+" if val > 0 else ("-" if val < 0 else "")
    return f"{sign}₹{abs(val):,.2f}"


def print_strategy_summary(res: dict):
    if not res.get('ok'):
        print(f"Error: {res.get('error')}")
        return

    s = res['summary']
    print(f"  Strategy:               {s['strategy']}")
    print(f"  Evaluation Period:      {res['start_date']}  to  {res['end_date']}")
    print(f"  Total Trades:           {s['total_trades']}")
    print(f"  Total Net P&L:          {format_currency(s['total_net_pnl'])}")
    print(f"  Win Rate:               {s['win_rate_pct']:.1f}%")
    print(f"  Profit Factor:          {s['profit_factor']:.2f}")
    print(f"  Max Drawdown:           {format_currency(s['max_drawdown'])}")
    print(f"  Sharpe Ratio:           {s['sharpe_ratio']:.2f}")
    print(f"  Avg Trade P&L:          {format_currency(s['avg_trade_pnl'])}")
    print(f"  Avg Win / Avg Loss:     {format_currency(s['avg_win'])}  /  {format_currency(s['avg_loss'])}")
    print(f"  Best / Worst Trade:     {format_currency(s['max_profit'])}  /  {format_currency(s['max_loss'])}")
    print(f"  Target 1 Hits:          {s['target_1_hits']} ({s['target_1_hits']/s['total_trades']*100:.1f}%)")
    print(f"  Stop Loss Hits:         {s['stop_loss_hits']} ({s['stop_loss_hits']/s['total_trades']*100:.1f}%)")
    print(f"  Avg Holding Duration:   {s['avg_duration_days']:.1f} days")


def print_gex_summary(res: dict):
    if not res.get('ok'):
        print(f"Error: {res.get('error')}")
        return

    s = res['summary']
    print(f"  Historical Evaluation:  {res['start_date']}  to  {res['end_date']}")
    print(f"  Total Days Evaluated:   {s['total_days_evaluated']} days")
    print(f"  Tier Accuracy:          {s['tier_accuracy_pct']:.1f}%")
    print(f"  Range Hit Rate:         {s['range_hit_rate_pct']:.1f}%")
    print(f"  Avg Next-Day Range:     {s['avg_next_day_range_pts']:.1f} pts")
    print(f"  Net GEX Correlation:    {s['net_gex_range_correlation']:.3f}")
    print("─" * 70)
    print(f"  🔴 BIG MOVE Days:       {s['big_move_days_count']} days (Avg Range: {s['avg_range_on_big_move_days']:.1f} pts)")
    print(f"     Realization Rate:    {s['big_move_realization_rate']:.1f}%")
    print(f"  🟢 SMALL MOVE Days:     {s['small_move_days_count']} days (Avg Range: {s['avg_range_on_small_move_days']:.1f} pts)")
    print(f"     Realization Rate:    {s['small_move_realization_rate']:.1f}%")


def main():
    parser = argparse.ArgumentParser(description="Options Backtesting & GEX Move CLI")
    parser.add_argument("--strategy", type=str, default="RADAR_ATM",
                        choices=["RADAR_ATM", "RADAR_OTM", "SHORT_STRADDLE", "ALL"],
                        help="Strategy to backtest")
    parser.add_argument("--days", type=int, default=365, help="Number of lookback days (default: 365)")
    parser.add_argument("--sl", type=float, default=0.40, help="Stop loss %% (e.g. 0.40 = 40%%)")
    parser.add_argument("--target", type=float, default=0.80, help="Target %% (e.g. 0.80 = 80%%)")
    parser.add_argument("--confluence", type=float, default=50.0, help="Min confluence score")
    parser.add_argument("--lots", type=int, default=1, help="Number of lots per trade")
    parser.add_argument("--gex-moves", action="store_true", help="Run GEX move-size backtest")
    parser.add_argument("--compare", action="store_true", help="Compare Buyer Radar vs Seller strategies")
    parser.add_argument("--export", type=str, default="", help="Export trade log to CSV file")
    args = parser.parse_args()

    hdm = HistoricalDataManager()
    # Ensure database has latest data
    hdm.sync_bhavcopy_data()

    if args.gex_moves:
        print_banner("GEX MOVE-SIZE BACKTEST (BIG VS SMALL MOVES)")
        gex_engine = GexBacktestEngine(hdm)
        res = gex_engine.run(days=args.days)
        print_gex_summary(res)
        if args.export and res.get('records'):
            df = pd.DataFrame(res['records'])
            df.to_csv(args.export, index=False)
            print(f"\n  ✓ Exported {len(df)} GEX records to {args.export}")
        return

    bt_engine = StrategyBacktestEngine(hdm)

    if args.compare:
        print_banner("STRATEGY COMPARISON (1-YEAR WALK-FORWARD)")
        strats = [
            OptionBuyerRadarStrategy(mode="ATM", stop_loss_pct=args.sl, target_pct=args.target, min_confluence=args.confluence),
            OptionBuyerRadarStrategy(mode="OTM", stop_loss_pct=0.50, target_pct=1.20, min_confluence=args.confluence),
            ShortStraddleStrategy(stop_loss_mult=1.5),
        ]
        results = []
        for s in strats:
            r = bt_engine.run(s, days=args.days, lots=args.lots)
            if r.get('ok'):
                summ = r['summary']
                results.append({
                    'Strategy': summ['strategy'],
                    'Trades': summ['total_trades'],
                    'Net P&L (₹)': f"{summ['total_net_pnl']:,.2f}",
                    'Win Rate': f"{summ['win_rate_pct']:.1f}%",
                    'Profit Factor': f"{summ['profit_factor']:.2f}",
                    'Max DD (₹)': f"{summ['max_drawdown']:,.2f}",
                    'Sharpe': f"{summ['sharpe_ratio']:.2f}",
                })
        comp_df = pd.DataFrame(results)
        print(comp_df.to_string(index=False))
        return

    # Single strategy execution
    if args.strategy == "RADAR_ATM":
        strat = OptionBuyerRadarStrategy(mode="ATM", stop_loss_pct=args.sl, target_pct=args.target, min_confluence=args.confluence)
    elif args.strategy == "RADAR_OTM":
        strat = OptionBuyerRadarStrategy(mode="OTM", stop_loss_pct=args.sl, target_pct=args.target, min_confluence=args.confluence)
    elif args.strategy == "SHORT_STRADDLE":
        strat = ShortStraddleStrategy(stop_loss_mult=1.5)
    else:
        strat = OptionBuyerRadarStrategy(mode="ATM", stop_loss_pct=args.sl, target_pct=args.target)

    print_banner(f"BACKTEST: {strat.name}")
    res = bt_engine.run(strat, days=args.days, lots=args.lots)
    print_strategy_summary(res)

    # Show first 5 and last 5 trades if present
    trades = res.get('trades', [])
    if trades:
        print("\n" + "─" * 70)
        print("  RECENT TRADE SAMPLE:")
        sample_df = pd.DataFrame(trades[-5:])
        cols_to_show = ['trade_id', 'entry_date', 'contract', 'action', 'entry_price', 'exit_date', 'exit_price', 'exit_reason', 'lot_pnl', 'roi_pct']
        print(sample_df[cols_to_show].to_string(index=False))

    if args.export and trades:
        export_path = args.export
        df = pd.DataFrame(trades)
        df.to_csv(export_path, index=False)
        print(f"\n  ✓ Successfully exported all {len(trades)} trades to: {export_path}")


if __name__ == "__main__":
    main()
