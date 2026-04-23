"""
SellSignalBacktester.py
=======================
Full backtesting engine for NIFTY sell-signal strategies.

For each signal event from SellSignalScanner:
  1. Opens position on the next available bar after the signal.
  2. Checks exit conditions on every subsequent 1-min bar:
       a. Stop loss  : net cost > stop_loss_multiplier × entry_credit
       b. Target     : position decayed to ≤30% of original credit (kept 70%)
       c. Expiry     : exit at intrinsic value on expiry date
       d. Time stop  : 5 trading days without other exit
  3. Records all trades to SQLite: data/backtest_results.db → table 'trades'.

NIFTY lot size: 65 (current)  /  risk-free: 7%

Usage:
    bt = SellSignalBacktester(bhav_engine, reconstructor)
    results = bt.run(signals_df, minute_df,
                     strategy='SHORT_STRADDLE',
                     stop_loss_multiplier=2.0)
    stats = results['stats']
    trades_df = results['trades']
"""

import os
import sys
import sqlite3
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, date as date_type

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── constants ─────────────────────────────────────────────────────────────────
NIFTY_LOT          = 65
RISK_FREE           = 0.07
DB_PATH             = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   'data', 'backtest_results.db')
MAX_HOLDING_DAYS    = 5      # time-stop after this many trading days
TARGET_DECAY        = 0.30   # exit when position value ≤ 30% of entry credit
SUPPORTED_STRATEGIES = ['SHORT_STRADDLE', 'SHORT_STRANGLE',
                         'IRON_CONDOR', 'BEAR_CALL_SPREAD']


def _round50(x: float) -> float:
    return round(x / 50) * 50


def _to_date(d) -> date_type:
    if isinstance(d, date_type):
        return d
    if isinstance(d, datetime):
        return d.date()
    return datetime.strptime(str(d)[:10], '%Y-%m-%d').date()


# ── strategy leg builders ─────────────────────────────────────────────────────
def _build_legs(strategy: str, spot: float) -> list[dict]:
    """Return leg definitions (without prices) for a strategy."""
    atm = _round50(spot)

    if strategy == 'SHORT_STRADDLE':
        return [
            {'strike': atm,                   'option_type': 'CE', 'action': 'SELL'},
            {'strike': atm,                   'option_type': 'PE', 'action': 'SELL'},
        ]

    elif strategy == 'SHORT_STRANGLE':
        ce_k = _round50(spot * 1.02)
        pe_k = _round50(spot * 0.98)
        return [
            {'strike': ce_k, 'option_type': 'CE', 'action': 'SELL'},
            {'strike': pe_k, 'option_type': 'PE', 'action': 'SELL'},
        ]

    elif strategy == 'IRON_CONDOR':
        short_ce = _round50(spot * 1.02)
        long_ce  = _round50(spot * 1.03)
        short_pe = _round50(spot * 0.98)
        long_pe  = _round50(spot * 0.97)
        return [
            {'strike': short_ce, 'option_type': 'CE', 'action': 'SELL'},
            {'strike': long_ce,  'option_type': 'CE', 'action': 'BUY'},
            {'strike': short_pe, 'option_type': 'PE', 'action': 'SELL'},
            {'strike': long_pe,  'option_type': 'PE', 'action': 'BUY'},
        ]

    elif strategy == 'BEAR_CALL_SPREAD':
        sell_ce = atm
        buy_ce  = atm + 100
        return [
            {'strike': sell_ce, 'option_type': 'CE', 'action': 'SELL'},
            {'strike': buy_ce,  'option_type': 'CE', 'action': 'BUY'},
        ]

    raise ValueError(f'Unknown strategy: {strategy}')


# ── SellSignalBacktester ──────────────────────────────────────────────────────
class SellSignalBacktester:

    def __init__(self, bhav_engine, reconstructor):
        """
        bhav_engine   : BhavCopyEngine (loaded)
        reconstructor : OptionPriceReconstructor
        """
        self._bhav  = bhav_engine
        self._recon = reconstructor
        os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
        self._init_db()

    # ── DB ───────────────────────────────────────────────────────────────────

    def _init_db(self):
        conn = sqlite3.connect(DB_PATH)
        conn.execute('''
            CREATE TABLE IF NOT EXISTS trades (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                entry_date      TEXT,
                exit_date       TEXT,
                signal_type     TEXT,
                strategy_type   TEXT,
                entry_credit    REAL,
                exit_debit      REAL,
                pnl_per_lot     REAL,
                exit_reason     TEXT,
                dte_at_entry    REAL,
                entry_spot      REAL,
                exit_spot       REAL,
                atm_iv          REAL,
                hv_20d          REAL,
                run_id          TEXT
            )
        ''')
        conn.commit()
        conn.close()

    def _save_trades(self, trades: list[dict], run_id: str):
        conn = sqlite3.connect(DB_PATH)
        rows = []
        for t in trades:
            rows.append((
                t.get('entry_date'),   t.get('exit_date'),
                t.get('signal_type'),  t.get('strategy_type'),
                t.get('entry_credit'), t.get('exit_debit'),
                t.get('pnl_per_lot'),  t.get('exit_reason'),
                t.get('dte_at_entry'), t.get('entry_spot'),
                t.get('exit_spot'),    t.get('atm_iv'),
                t.get('hv_20d'),       run_id,
            ))
        conn.executemany('''
            INSERT INTO trades
                (entry_date, exit_date, signal_type, strategy_type,
                 entry_credit, exit_debit, pnl_per_lot, exit_reason,
                 dte_at_entry, entry_spot, exit_spot, atm_iv, hv_20d, run_id)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        ''', rows)
        conn.commit()
        conn.close()

    # ── run ──────────────────────────────────────────────────────────────────

    def run(
        self,
        signals_df:            pd.DataFrame,
        minute_df:             pd.DataFrame,
        strategy:              str   = 'SHORT_STRADDLE',
        stop_loss_multiplier:  float = 2.0,
        run_id:                str   = None,
    ) -> dict:
        """
        Run the backtest for one strategy type across all signal events.

        Returns
        -------
        dict with keys:
            trades : list[dict]
            stats  : dict (aggregated performance metrics)
        """
        if run_id is None:
            run_id = datetime.now().strftime('%Y%m%d_%H%M%S')

        strategy = strategy.upper()
        if strategy not in SUPPORTED_STRATEGIES:
            raise ValueError(f'Unsupported strategy: {strategy}')

        # Filter signals to this strategy (all signals; strategy is the trade type)
        sig_df = signals_df.copy()
        if sig_df.empty:
            return {'trades': [], 'stats': {}}

        # Get all trading dates from minute_df
        trading_dates = sorted(set(minute_df.index.date))

        trades = []
        for _, sig_row in sig_df.iterrows():
            trade = self._simulate_trade(
                sig_row         = sig_row,
                minute_df       = minute_df,
                trading_dates   = trading_dates,
                strategy        = strategy,
                sl_mult         = stop_loss_multiplier,
            )
            if trade is not None:
                trades.append(trade)

        # Save to DB
        if trades:
            self._save_trades(trades, run_id)

        stats = self._compute_stats(trades)
        return {'trades': trades, 'stats': stats, 'run_id': run_id}

    # ── trade simulation ─────────────────────────────────────────────────────

    def _simulate_trade(
        self,
        sig_row:       pd.Series,
        minute_df:     pd.DataFrame,
        trading_dates: list,
        strategy:      str,
        sl_mult:       float,
    ) -> dict | None:
        """Simulate a single trade from entry to exit. Returns trade dict or None."""

        signal_date_str = str(sig_row['date'])[:10]
        signal_dt       = pd.Timestamp(sig_row.get('signal_time', f'{signal_date_str} 09:30'))
        atm_iv          = sig_row.get('atm_iv')
        hv_20d          = sig_row.get('hv_20d')

        # Use IV from reconstructor (handles fallback automatically)
        iv_pct = self._recon.get_iv_for_date(signal_date_str)

        # Find entry bar: first bar strictly after the signal time
        bars_after = minute_df[minute_df.index > signal_dt]
        if bars_after.empty:
            return None

        entry_bar = bars_after.iloc[0]
        entry_spot = float(entry_bar['close'])
        entry_datetime = bars_after.index[0]
        entry_date_str = entry_datetime.strftime('%Y-%m-%d')
        entry_date_d   = entry_datetime.date()

        # Find nearest weekly expiry (next Thursday or next-to-next)
        expiry_d = self._find_expiry(entry_date_d)
        if expiry_d is None:
            return None

        dte_at_entry = max((expiry_d - entry_date_d).days, 1) / 365.0

        # Build strategy legs
        legs = _build_legs(strategy, entry_spot)

        # Entry prices via reconstructor
        entry_prices = [
            self._recon.price(entry_spot, leg['strike'], dte_at_entry, iv_pct, leg['option_type'])
            for leg in legs
        ]

        # Net credit received (₹/lot, positive = credit)
        net_credit_per_unit = sum(
            p if leg['action'] == 'SELL' else -p
            for leg, p in zip(legs, entry_prices)
        )
        entry_credit_rs = net_credit_per_unit * NIFTY_LOT

        if entry_credit_rs <= 0:
            return None   # debit strategy not supported as sell signal

        # Exit thresholds (per unit)
        sl_threshold    = -sl_mult * net_credit_per_unit      # negative (loss)
        target_threshold = net_credit_per_unit * TARGET_DECAY  # residual value

        # Scan subsequent bars for exit
        holding_trading_days = 0
        prev_bar_date = entry_date_d
        exit_info = None

        bars_from_entry = minute_df[minute_df.index > entry_datetime]

        for bar_dt, bar_row in bars_from_entry.iterrows():
            bar_date_d = bar_dt.date()

            # Count trading days
            if bar_date_d != prev_bar_date:
                holding_trading_days += 1
                prev_bar_date = bar_date_d

            spot    = float(bar_row['close'])
            dte_now = max((expiry_d - bar_date_d).days, 0) / 365.0

            # ── Exit 3: Expiry ───────────────────────────────────────────
            if bar_date_d >= expiry_d:
                pnl_rs = self._recon.expiry_pnl(legs, entry_prices, spot)
                exit_info = {
                    'exit_date':   bar_date_d.strftime('%Y-%m-%d'),
                    'exit_spot':   spot,
                    'exit_debit':  max(0.0, -pnl_rs / NIFTY_LOT + net_credit_per_unit),  # residual cost
                    'pnl_per_lot': pnl_rs,
                    'exit_reason': 'EXPIRY',
                }
                break

            # Mid-trade MtM for stop loss / target check
            mtm_pnl_rs = self._recon.mark_to_market(
                legs, entry_prices, spot, dte_now, iv_pct
            )
            mtm_per_unit = mtm_pnl_rs / NIFTY_LOT

            # Remaining position value per unit
            remaining = net_credit_per_unit + mtm_per_unit   # credit received + change

            # ── Exit 1: Stop loss ─────────────────────────────────────────
            if mtm_per_unit <= sl_threshold:
                exit_debit = max(0.0, net_credit_per_unit - remaining)
                exit_info = {
                    'exit_date':   bar_date_d.strftime('%Y-%m-%d'),
                    'exit_spot':   spot,
                    'exit_debit':  exit_debit,
                    'pnl_per_lot': mtm_pnl_rs,
                    'exit_reason': 'STOP_LOSS',
                }
                break

            # ── Exit 2: Target profit ─────────────────────────────────────
            if remaining <= target_threshold:
                exit_info = {
                    'exit_date':   bar_date_d.strftime('%Y-%m-%d'),
                    'exit_spot':   spot,
                    'exit_debit':  remaining,
                    'pnl_per_lot': mtm_pnl_rs,
                    'exit_reason': 'TARGET',
                }
                break

            # ── Exit 4: Time stop ─────────────────────────────────────────
            if holding_trading_days >= MAX_HOLDING_DAYS:
                exit_info = {
                    'exit_date':   bar_date_d.strftime('%Y-%m-%d'),
                    'exit_spot':   spot,
                    'exit_debit':  max(0.0, net_credit_per_unit - remaining),
                    'pnl_per_lot': mtm_pnl_rs,
                    'exit_reason': 'TIME_STOP',
                }
                break

        if exit_info is None:
            # Never exited (no data past entry) — treat as still open, skip
            return None

        return {
            'entry_date':    entry_date_str,
            'exit_date':     exit_info['exit_date'],
            'signal_type':   str(sig_row['signal_type']),
            'strategy_type': strategy,
            'entry_credit':  round(entry_credit_rs, 2),
            'exit_debit':    round(exit_info['exit_debit'] * NIFTY_LOT, 2),
            'pnl_per_lot':   round(exit_info['pnl_per_lot'], 2),
            'exit_reason':   exit_info['exit_reason'],
            'dte_at_entry':  round(dte_at_entry * 365, 1),   # store in days
            'entry_spot':    round(entry_spot, 2),
            'exit_spot':     round(exit_info['exit_spot'], 2),
            'atm_iv':        round(float(atm_iv), 4) if atm_iv is not None else None,
            'hv_20d':        round(float(hv_20d), 4) if hv_20d is not None else None,
        }

    # ── expiry helper ─────────────────────────────────────────────────────────

    def _find_expiry(self, from_date: date_type) -> date_type | None:
        """
        Return the nearest weekly expiry (Thursday) on or after from_date,
        but also check known expiry dates from bhavcopy when available.
        """
        bhav_expiries = self._bhav.get_expiries(
            from_date, from_date + timedelta(days=30)
        )
        future = [e for e in bhav_expiries if e >= from_date]
        if future:
            return min(future)

        # Fallback: next Thursday
        d = from_date
        for _ in range(10):
            if d.weekday() == 3:   # Thursday
                return d
            d += timedelta(days=1)
        return None

    # ── stats ─────────────────────────────────────────────────────────────────

    def _compute_stats(self, trades: list[dict]) -> dict:
        """Compute aggregate backtest statistics."""
        if not trades:
            return {
                'total_pnl': 0, 'trade_count': 0, 'win_rate': 0,
                'avg_pnl': 0, 'max_loss': 0, 'max_drawdown': 0,
                'sharpe': 0, 'kelly': 0,
            }

        pnls = np.array([t['pnl_per_lot'] for t in trades], dtype=float)
        wins  = pnls[pnls > 0]
        losses = pnls[pnls <= 0]

        total_pnl   = float(pnls.sum())
        trade_count = len(pnls)
        win_rate    = len(wins) / trade_count
        avg_pnl     = float(pnls.mean())
        max_loss    = float(pnls.min())

        # Max drawdown (peak-to-trough cumulative P&L)
        cum_pnl  = np.cumsum(pnls)
        peak     = np.maximum.accumulate(cum_pnl)
        drawdown = cum_pnl - peak
        max_dd   = float(drawdown.min())

        # Annualised Sharpe
        if len(pnls) > 1 and pnls.std() > 0:
            sharpe = (pnls.mean() / pnls.std()) * np.sqrt(52)   # weekly-approx
        else:
            sharpe = 0.0

        # Kelly fraction: f = (p*b - q) / b  capped at 0.25
        avg_win  = float(wins.mean())  if len(wins)   > 0 else 0.0
        avg_loss = float(abs(losses.mean())) if len(losses) > 0 else 1.0
        if avg_loss > 0:
            b = avg_win / avg_loss
            q = 1 - win_rate
            kelly = (win_rate * b - q) / b if b > 0 else 0
            kelly = min(max(kelly, 0.0), 0.25)
        else:
            kelly = 0.0

        return {
            'total_pnl':    round(total_pnl, 2),
            'trade_count':  trade_count,
            'win_rate':     round(win_rate * 100, 2),
            'avg_pnl':      round(avg_pnl, 2),
            'max_loss':     round(max_loss, 2),
            'max_drawdown': round(max_dd, 2),
            'sharpe':       round(sharpe, 4),
            'kelly':        round(kelly * 100, 2),   # as %
            'avg_win':      round(avg_win, 2),
            'avg_loss':     round(avg_loss, 2),
            'cumulative':   cum_pnl.tolist(),
        }
