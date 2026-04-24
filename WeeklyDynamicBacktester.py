"""
WeeklyDynamicBacktester.py
==========================
Main simulation engine. For each week where the Combined signal fires,
it runs a full position lifecycle including daily shift reassessment.

Strategies simulated
--------------------
  1. SHORT_STRANGLE   (primary)  -- 2% OTM CE + 2% OTM PE, delta-breach shifting
  2. SHORT_STRADDLE   (compare)  -- ATM CE + ATM PE, 1.5% drift triggers re-centre
  3. IRON_CONDOR      (compare)  -- short at 2%, long wing at 3%
  4. BEAR_CALL_SPREAD (compare)  -- sell ATM CE, buy CE 100 pts higher; signal-only exit

SQLite tables (data/backtest_results.db)
----------------------------------------
  position_summaries, leg_transactions
  straddle_summaries, straddle_transactions
  condor_summaries,   condor_transactions
  callspread_summaries, callspread_transactions

Usage
-----
    bt = WeeklyDynamicBacktester(
            start_date='2025-01-01', end_date='2025-12-31',
            sl_mult=2.0, shift_threshold_pct=0.8, show_shift_log=True)
    bt.run()
    stats = bt.aggregate_stats('strangle')
"""

import os
import sys
import sqlite3
import numpy as np
import pandas as pd
from datetime import date as date_type, datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from BhavCopyEngine           import BhavCopyEngine
from MinuteDataFetcher        import MinuteDataFetcher
from ExpiryCalendar           import ExpiryCalendar
from BacktestSignalExtractor  import BacktestSignalExtractor
from OptionPriceReconstructor import OptionPriceReconstructor
from ShiftEvaluator           import ShiftEvaluator
from PositionLedger           import PositionLedger

# ── constants ─────────────────────────────────────────────────────────────────
LOT_SIZE  = 75
RISK_FREE = 0.07
DB_PATH   = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         'data', 'backtest_results.db')
STRANGLE_WIDTH_PCT = 2.0   # 2% from spot for short strikes
WING_PCT           = 1.0   # additional 1% for condor wings (3% from spot)
TARGET_DECAY       = 0.30  # exit when cost-to-close <= 30% of entry credit


def _to_date(d) -> date_type:
    if isinstance(d, date_type) and not isinstance(d, datetime):
        return d
    if isinstance(d, datetime):
        return d.date()
    return datetime.strptime(str(d)[:10], '%Y-%m-%d').date()


def _round50(x: float) -> float:
    return round(x / 50) * 50


def _week_label(entry_date: date_type) -> str:
    y, w, _ = entry_date.isocalendar()
    return f'{y}-W{w:02d}'


# ── database helpers ──────────────────────────────────────────────────────────

def _init_db(conn: sqlite3.Connection):
    cur = conn.cursor()
    strategies = ['', 'straddle_', 'condor_', 'callspread_']
    for prefix in strategies:
        cur.execute(f'''
            CREATE TABLE IF NOT EXISTS {prefix}position_summaries (
                position_id TEXT, entry_date TEXT, expiry_date TEXT,
                entry_spot REAL, ce_strike_initial REAL, pe_strike_initial REAL,
                entry_credit_per_unit REAL, shift_count INTEGER,
                exit_date TEXT, exit_reason TEXT,
                final_pnl_per_unit REAL, final_pnl_rupees REAL,
                atm_iv_at_entry REAL, hv_at_entry REAL,
                vrp_ratio REAL, signal_vrp TEXT, signal_regime TEXT, signal_vov TEXT,
                strategy TEXT
            )''')
        cur.execute(f'''
            CREATE TABLE IF NOT EXISTS {prefix}leg_transactions (
                position_id TEXT, date TEXT, time_type TEXT, option_type TEXT,
                strike REAL, action TEXT, price_per_unit REAL,
                cashflow_per_unit REAL, cashflow_rupees REAL,
                shift_number INTEGER, reason TEXT, strategy TEXT
            )''')
    conn.commit()


def _save_summary(conn, prefix: str, ledger: PositionLedger,
                  entry_meta: dict, strategy: str):
    s   = ledger.summary()
    em  = entry_meta
    cur = conn.cursor()
    cur.execute(
        f'INSERT INTO {prefix}position_summaries VALUES '
        f'(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)',
        (s['position_id'], s['entry_date'], s['expiry_date'],
         s['entry_spot'], em.get('ce_strike'), em.get('pe_strike'),
         ledger.entry_credit_per_unit, s['shift_count'],
         s['exit_date'], s['exit_reason'],
         s['final_pnl_per_unit'], s['final_pnl_rupees'],
         em.get('atm_iv'), em.get('hv'), em.get('vrp_ratio'),
         em.get('signal_vrp'), em.get('signal_regime'), em.get('signal_vov'),
         strategy)
    )
    for tx in ledger.get_transactions():
        cur.execute(
            f'INSERT INTO {prefix}leg_transactions VALUES '
            f'(?,?,?,?,?,?,?,?,?,?,?,?)',
            (tx['position_id'], tx['date'], tx['time_type'], tx['option_type'],
             tx['strike'], tx['action'], tx['price_per_unit'],
             tx['cashflow_per_unit'], tx['cashflow_rupees'],
             tx['shift_number'], tx['reason'], strategy)
        )
    conn.commit()


# ── price helpers ─────────────────────────────────────────────────────────────

def _price(reconstructor: OptionPriceReconstructor, bhav_engine: BhavCopyEngine,
           spot: float, strike: float, dte_days: int, expiry_str: str,
           opt_type: str, date_str: str) -> float:
    """Price a single option leg via the reconstructor."""
    try:
        T = max(dte_days / 365.0, 1e-4)
        iv_raw = bhav_engine.get_atm_iv(date_str)
        if iv_raw and iv_raw > 0:
            sigma = iv_raw / 100
        else:
            hv = bhav_engine.get_hv_20d(date_str)
            sigma = (hv / 100) if hv and hv > 0 else 0.15
        from OptionAnalytics import OptionAnalytics
        oa = OptionAnalytics()
        return max(0.05, oa.black_scholes(spot, strike, T, RISK_FREE, sigma, opt_type))
    except Exception:
        return 0.05


def _trading_days_between(min_fetcher: MinuteDataFetcher,
                           start: date_type, end: date_type) -> list[date_type]:
    """Return list of trading days in minute data between start and end inclusive."""
    try:
        df = min_fetcher.get_data(start.year)
        if df is None or df.empty:
            return []
        mask = (df.index.date >= start) & (df.index.date <= end)
        sub  = df[mask]
        return sorted(set(sub.index.date))
    except Exception:
        return []


# ── main engine ───────────────────────────────────────────────────────────────

class WeeklyDynamicBacktester:

    def __init__(
        self,
        start_date:         str,
        end_date:           str,
        sl_mult:            float = 2.0,
        shift_threshold_pct: float = 0.8,
        show_shift_log:     bool  = True,
    ):
        self.start_date         = _to_date(start_date)
        self.end_date           = _to_date(end_date)
        self.sl_mult            = sl_mult
        self.shift_threshold    = shift_threshold_pct
        self.show_shift_log     = show_shift_log

        # Components
        self.bhav        = BhavCopyEngine()
        self.minute      = MinuteDataFetcher()
        self.pricer      = OptionPriceReconstructor()
        self.shift_eval  = ShiftEvaluator(threshold_pct=shift_threshold_pct)
        self._minute_df  = pd.DataFrame()   # populated in run()
        self.cal         = None             # populated in run()
        self.signals_df  = pd.DataFrame()   # populated in run()

        self._results: list[dict] = []

    # ── run ──────────────────────────────────────────────────────────────────

    def run(self):
        print(f'\n[WeeklyDynamicBacktester] Loading bhavcopy {self.start_date} -> {self.end_date}')
        self.bhav.load_range(self.start_date, self.end_date)

        print('[WeeklyDynamicBacktester] Building expiry calendar ...')
        self.cal = ExpiryCalendar(self.bhav)
        self.cal.build(self.start_date, self.end_date)

        stats = self.cal.weekday_stats()
        print(f'  Expiry weekday counts: {stats}')

        print('[WeeklyDynamicBacktester] Computing daily signals ...')
        extractor   = BacktestSignalExtractor(self.bhav, self.minute)
        self.signals_df = extractor.compute(self.start_date, self.end_date)

        print('[WeeklyDynamicBacktester] Fetching minute data ...')
        self._minute_df = self.minute.get(str(self.start_date), str(self.end_date))
        if self._minute_df is None or self._minute_df.empty:
            print('  [WARN] No minute data loaded -- spot prices will fall back to bhavcopy.')
            self._minute_df = pd.DataFrame()

        # Get entry weeks
        entry_weeks = self.cal.get_weekly_entry_dates(self.start_date, self.end_date)
        print(f'[WeeklyDynamicBacktester] {len(entry_weeks)} entry weeks found')

        os.makedirs('data', exist_ok=True)
        conn = sqlite3.connect(DB_PATH)
        _init_db(conn)

        for entry_date, expiry_date in entry_weeks:
            entry_str  = str(entry_date)
            expiry_str = str(expiry_date)
            sig = self._get_signal(entry_str)

            if not sig.get('entry_signal', False):
                continue   # Combined signal not firing -- skip this week

            # Simulate all four strategies
            self._simulate_strangle(entry_date, expiry_date, sig, conn)
            self._simulate_straddle(entry_date, expiry_date, sig, conn)
            self._simulate_condor(entry_date, expiry_date, sig, conn)
            self._simulate_callspread(entry_date, expiry_date, sig, conn)

        conn.close()
        print(f'\n[WeeklyDynamicBacktester] Done. DB: {DB_PATH}')

    # ── signal helper ─────────────────────────────────────────────────────────

    def _get_signal(self, date_str: str) -> dict:
        try:
            row = self.signals_df.loc[pd.Timestamp(date_str)]
            return row.to_dict()
        except (KeyError, TypeError, AttributeError):
            return {'entry_signal': False, 'signal_vrp': 'NEUTRAL',
                    'signal_regime': 'NORMAL', 'signal_vov': 'STABLE',
                    'vrp_ratio': None, 'atm_iv': None, 'hv_20d': None}

    def _get_spot(self, trade_date: date_type) -> float:
        """Get 15:29 bar close (daily) from minute data, fallback to bhavcopy."""
        try:
            if self._minute_df is not None and not self._minute_df.empty:
                day_bars = self._minute_df[self._minute_df.index.date == trade_date]
                if not day_bars.empty:
                    return float(day_bars['close'].iloc[-1])
            uc = self.bhav.get_underlying_close(str(trade_date))
            return uc or 0.0
        except Exception:
            return self.bhav.get_underlying_close(str(trade_date)) or 0.0

    def _get_entry_spot(self, trade_date: date_type) -> float:
        """09:30 bar price for entry, fallback to daily close."""
        try:
            if self._minute_df is not None and not self._minute_df.empty:
                day_bars = self._minute_df[self._minute_df.index.date == trade_date]
                target   = day_bars.between_time('09:30', '09:31')
                if not target.empty:
                    return float(target['open'].iloc[0])
                if not day_bars.empty:
                    return float(day_bars['close'].iloc[-1])
            uc = self.bhav.get_underlying_close(str(trade_date))
            return uc or 0.0
        except Exception:
            return self.bhav.get_underlying_close(str(trade_date)) or 0.0

    def _daily_dates(self, entry: date_type, expiry: date_type) -> list[date_type]:
        """Return trading days from entry+1 through expiry."""
        all_days = []
        d = entry + timedelta(days=1)
        while d <= expiry:
            if d.weekday() < 5:  # Mon-Fri
                all_days.append(d)
            d += timedelta(days=1)
        return all_days

    def _log(self, msg: str):
        if self.show_shift_log:
            print(msg)

    # ── SHORT STRANGLE (primary) ──────────────────────────────────────────────

    def _simulate_strangle(self, entry_date: date_type, expiry_date: date_type,
                           entry_sig: dict, conn: sqlite3.Connection):
        entry_str  = str(entry_date)
        expiry_str = str(expiry_date)
        wlabel     = _week_label(entry_date)
        dte_entry  = self.cal.get_dte(entry_date, expiry_date)

        spot = self._get_entry_spot(entry_date)
        if spot <= 0:
            return

        ce_strike = _round50(spot * (1 + STRANGLE_WIDTH_PCT / 100))
        pe_strike = _round50(spot * (1 - STRANGLE_WIDTH_PCT / 100))

        ce_price  = _price(self.pricer, self.bhav, spot, ce_strike, dte_entry, expiry_str, 'CE', entry_str)
        pe_price  = _price(self.pricer, self.bhav, spot, pe_strike, dte_entry, expiry_str, 'PE', entry_str)

        ledger = PositionLedger(wlabel, entry_str, expiry_str, entry_spot=spot)
        ledger.add_transaction(entry_str, 'OPEN', 'CE', ce_strike, 'SELL',
                               ce_price, 0, 'original entry')
        ledger.add_transaction(entry_str, 'OPEN', 'PE', pe_strike, 'SELL',
                               pe_price, 0, 'original entry')

        entry_meta = {
            'ce_strike': ce_strike, 'pe_strike': pe_strike,
            'atm_iv':    entry_sig.get('atm_iv'),
            'hv':        entry_sig.get('hv_20d'),
            'vrp_ratio': entry_sig.get('vrp_ratio'),
            'signal_vrp': entry_sig.get('signal_vrp'),
            'signal_regime': entry_sig.get('signal_regime'),
            'signal_vov': entry_sig.get('signal_vov'),
        }
        credit = ledger.entry_credit_per_unit

        self._log(f'\n[Week {wlabel}] Entry: {entry_str} | Expiry: {expiry_str} | '
                  f'Spot: {spot:.0f}')
        self._log(f'  Strikes: CE {ce_strike:.0f} | PE {pe_strike:.0f} | '
                  f'Credit: Rs.{credit * LOT_SIZE:.0f}')

        shifted_today = False
        current_shift = 0
        exit_reason   = None

        for day in self._daily_dates(entry_date, expiry_date):
            day_str  = str(day)
            cur_spot = self._get_spot(day)
            if cur_spot <= 0:
                continue
            dte_now  = self.cal.get_dte(day, expiry_date)

            is_expiry = (day == expiry_date)
            near_exp  = self.cal.is_too_close_to_expiry(day, expiry_date)

            # Current leg strikes
            cur_ce = ledger.current_ce_strike or ce_strike
            cur_pe = ledger.current_pe_strike or pe_strike

            # STOP LOSS check (using the current legs, against original entry credit)
            close_ce = _price(self.pricer, self.bhav, cur_spot, cur_ce, dte_now,
                              expiry_str, 'CE', day_str)
            close_pe = _price(self.pricer, self.bhav, cur_spot, cur_pe, dte_now,
                              expiry_str, 'PE', day_str)
            cost_to_close = close_ce + close_pe

            if credit > 0 and cost_to_close >= credit * self.sl_mult:
                ledger.add_transaction(day_str, 'CLOSE', 'CE', cur_ce, 'BUY',
                                       close_ce, current_shift, 'SL_HIT')
                ledger.add_transaction(day_str, 'CLOSE', 'PE', cur_pe, 'BUY',
                                       close_pe, current_shift, 'SL_HIT')
                exit_reason = 'SL_HIT'
                ledger.close(day_str, exit_reason)
                self._log(f'  Day {(day - entry_date).days} ({day_str}): SL HIT | '
                          f'Cost {cost_to_close:.2f} vs credit {credit:.2f} | '
                          f'Final P&L: Rs.{ledger.running_net_rupees:.0f}')
                break

            # TARGET check
            if credit > 0 and cost_to_close <= credit * TARGET_DECAY:
                ledger.add_transaction(day_str, 'CLOSE', 'CE', cur_ce, 'BUY',
                                       close_ce, current_shift, 'TARGET_HIT')
                ledger.add_transaction(day_str, 'CLOSE', 'PE', cur_pe, 'BUY',
                                       close_pe, current_shift, 'TARGET_HIT')
                exit_reason = 'TARGET_HIT'
                ledger.close(day_str, exit_reason)
                self._log(f'  Day {(day - entry_date).days} ({day_str}): TARGET HIT | '
                          f'Final P&L: Rs.{ledger.running_net_rupees:.0f}')
                break

            # EXPIRY settlement
            if is_expiry:
                settle_ce = max(0.0, cur_spot - cur_ce)
                settle_pe = max(0.0, cur_pe - cur_spot)
                ledger.add_transaction(day_str, 'SETTLE', 'CE', cur_ce, 'BUY',
                                       settle_ce, current_shift, 'expiry settlement')
                ledger.add_transaction(day_str, 'SETTLE', 'PE', cur_pe, 'BUY',
                                       settle_pe, current_shift, 'expiry settlement')
                exit_reason = 'EXPIRY'
                ledger.close(day_str, exit_reason)
                self._log(f'  Day {(day - entry_date).days} ({day_str}): EXPIRY | '
                          f'Final P&L: Rs.{ledger.running_net_rupees:.0f}')
                break

            # SHIFT evaluation
            if not near_exp:
                sig_today = self._get_signal(day_str)
                decision  = self.shift_eval.evaluate(
                    cur_spot, cur_ce, cur_pe, expiry_date, day,
                    sig_today, ledger.running_net_per_unit,
                    already_shifted_today=shifted_today
                )
                if decision.should_shift:
                    current_shift += 1
                    # Close breached legs
                    if decision.ce_action == 'CLOSE_AND_REOPEN':
                        close_p = _price(self.pricer, self.bhav, cur_spot, cur_ce, dte_now,
                                         expiry_str, 'CE', day_str)
                        ledger.add_transaction(day_str, 'CLOSE', 'CE', cur_ce, 'BUY',
                                               close_p, current_shift - 1,
                                               f'CE shifted -- {decision.shift_type}')
                        new_ce_p = _price(self.pricer, self.bhav, cur_spot,
                                          decision.new_ce_strike, dte_now,
                                          expiry_str, 'CE', day_str)
                        ledger.add_transaction(day_str, 'OPEN', 'CE',
                                               decision.new_ce_strike, 'SELL',
                                               new_ce_p, current_shift,
                                               f'CE new strike -- {decision.reason[:60]}')
                    if decision.pe_action == 'CLOSE_AND_REOPEN':
                        close_p = _price(self.pricer, self.bhav, cur_spot, cur_pe, dte_now,
                                         expiry_str, 'PE', day_str)
                        ledger.add_transaction(day_str, 'CLOSE', 'PE', cur_pe, 'BUY',
                                               close_p, current_shift - 1,
                                               f'PE shifted -- {decision.shift_type}')
                        new_pe_p = _price(self.pricer, self.bhav, cur_spot,
                                          decision.new_pe_strike, dte_now,
                                          expiry_str, 'PE', day_str)
                        ledger.add_transaction(day_str, 'OPEN', 'PE',
                                               decision.new_pe_strike, 'SELL',
                                               new_pe_p, current_shift,
                                               f'PE new strike -- {decision.reason[:60]}')
                    ledger.mark_shift()
                    shifted_today = True

                    ce_after = decision.new_ce_strike or cur_ce
                    pe_after = decision.new_pe_strike or cur_pe
                    self._log(
                        f'  Day {(day - entry_date).days} ({day_str}): SHIFT {decision.shift_type} | '
                        f'CE {cur_ce:.0f}->{ce_after:.0f}, PE {cur_pe:.0f}->{pe_after:.0f} | '
                        f'Running: Rs.{ledger.running_net_rupees:.0f}'
                    )
                else:
                    self._log(
                        f'  Day {(day - entry_date).days} ({day_str}): HOLD | '
                        f'Running: Rs.{ledger.running_net_rupees:.0f}'
                    )
            else:
                self._log(
                    f'  Day {(day - entry_date).days} ({day_str}): HOLD (near expiry) | '
                    f'Running: Rs.{ledger.running_net_rupees:.0f}'
                )
            shifted_today = False  # reset for next day

        if ledger.is_active:
            # Fallback if loop ended without closing
            ledger.close(expiry_str, 'EXPIRY_FALLBACK')

        _save_summary(conn, '', ledger, entry_meta, 'SHORT_STRANGLE')

    # ── SHORT STRADDLE ────────────────────────────────────────────────────────

    def _simulate_straddle(self, entry_date: date_type, expiry_date: date_type,
                           entry_sig: dict, conn: sqlite3.Connection):
        entry_str  = str(entry_date)
        expiry_str = str(expiry_date)
        wlabel     = _week_label(entry_date) + '_STD'
        dte_entry  = self.cal.get_dte(entry_date, expiry_date)

        spot      = self._get_entry_spot(entry_date)
        if spot <= 0:
            return
        atm_strike = _round50(spot)

        ce_price = _price(self.pricer, self.bhav, spot, atm_strike, dte_entry, expiry_str, 'CE', entry_str)
        pe_price = _price(self.pricer, self.bhav, spot, atm_strike, dte_entry, expiry_str, 'PE', entry_str)

        ledger = PositionLedger(wlabel, entry_str, expiry_str, entry_spot=spot)
        ledger.add_transaction(entry_str, 'OPEN', 'CE', atm_strike, 'SELL', ce_price, 0, 'straddle entry')
        ledger.add_transaction(entry_str, 'OPEN', 'PE', atm_strike, 'SELL', pe_price, 0, 'straddle entry')

        entry_meta = {
            'ce_strike': atm_strike, 'pe_strike': atm_strike,
            'atm_iv': entry_sig.get('atm_iv'), 'hv': entry_sig.get('hv_20d'),
            'vrp_ratio': entry_sig.get('vrp_ratio'),
            'signal_vrp': entry_sig.get('signal_vrp'),
            'signal_regime': entry_sig.get('signal_regime'),
            'signal_vov': entry_sig.get('signal_vov'),
        }
        credit        = ledger.entry_credit_per_unit
        shifted_today = False
        current_shift = 0

        for day in self._daily_dates(entry_date, expiry_date):
            day_str  = str(day)
            cur_spot = self._get_spot(day)
            if cur_spot <= 0:
                continue
            dte_now   = self.cal.get_dte(day, expiry_date)
            is_expiry = (day == expiry_date)
            near_exp  = self.cal.is_too_close_to_expiry(day, expiry_date)

            cur_ce = ledger.current_ce_strike or atm_strike
            cur_pe = ledger.current_pe_strike or atm_strike

            close_ce = _price(self.pricer, self.bhav, cur_spot, cur_ce, dte_now, expiry_str, 'CE', day_str)
            close_pe = _price(self.pricer, self.bhav, cur_spot, cur_pe, dte_now, expiry_str, 'PE', day_str)
            cost_to_close = close_ce + close_pe

            if credit > 0 and cost_to_close >= credit * self.sl_mult:
                ledger.add_transaction(day_str, 'CLOSE', 'CE', cur_ce, 'BUY', close_ce, current_shift, 'SL_HIT')
                ledger.add_transaction(day_str, 'CLOSE', 'PE', cur_pe, 'BUY', close_pe, current_shift, 'SL_HIT')
                ledger.close(day_str, 'SL_HIT')
                break
            if credit > 0 and cost_to_close <= credit * TARGET_DECAY:
                ledger.add_transaction(day_str, 'CLOSE', 'CE', cur_ce, 'BUY', close_ce, current_shift, 'TARGET_HIT')
                ledger.add_transaction(day_str, 'CLOSE', 'PE', cur_pe, 'BUY', close_pe, current_shift, 'TARGET_HIT')
                ledger.close(day_str, 'TARGET_HIT')
                break
            if is_expiry:
                ledger.add_transaction(day_str, 'SETTLE', 'CE', cur_ce, 'BUY', max(0, cur_spot - cur_ce), current_shift, 'expiry')
                ledger.add_transaction(day_str, 'SETTLE', 'PE', cur_pe, 'BUY', max(0, cur_pe - cur_spot), current_shift, 'expiry')
                ledger.close(day_str, 'EXPIRY')
                break
            if not near_exp:
                decision = self.shift_eval.evaluate_straddle(
                    cur_spot, cur_ce, expiry_date, day,
                    self._get_signal(day_str), ledger.running_net_per_unit, shifted_today)
                if decision.should_shift:
                    current_shift += 1
                    new_atm = decision.new_ce_strike or _round50(cur_spot)
                    for otype, old_k in [('CE', cur_ce), ('PE', cur_pe)]:
                        old_p = _price(self.pricer, self.bhav, cur_spot, old_k, dte_now, expiry_str, otype, day_str)
                        ledger.add_transaction(day_str, 'CLOSE', otype, old_k, 'BUY', old_p, current_shift - 1, 'straddle shift close')
                        new_p = _price(self.pricer, self.bhav, cur_spot, new_atm, dte_now, expiry_str, otype, day_str)
                        ledger.add_transaction(day_str, 'OPEN', otype, new_atm, 'SELL', new_p, current_shift, 'straddle shift open')
                    ledger.mark_shift()
                    shifted_today = True
            shifted_today = False

        if ledger.is_active:
            ledger.close(expiry_str, 'EXPIRY_FALLBACK')
        _save_summary(conn, 'straddle_', ledger, entry_meta, 'SHORT_STRADDLE')

    # ── IRON CONDOR ───────────────────────────────────────────────────────────

    def _simulate_condor(self, entry_date: date_type, expiry_date: date_type,
                         entry_sig: dict, conn: sqlite3.Connection):
        entry_str  = str(entry_date)
        expiry_str = str(expiry_date)
        wlabel     = _week_label(entry_date) + '_IC'
        dte_entry  = self.cal.get_dte(entry_date, expiry_date)

        spot = self._get_entry_spot(entry_date)
        if spot <= 0:
            return

        short_ce = _round50(spot * (1 + STRANGLE_WIDTH_PCT / 100))
        long_ce  = _round50(spot * (1 + STRANGLE_WIDTH_PCT / 100 + WING_PCT / 100))
        short_pe = _round50(spot * (1 - STRANGLE_WIDTH_PCT / 100))
        long_pe  = _round50(spot * (1 - STRANGLE_WIDTH_PCT / 100 - WING_PCT / 100))

        def p(strike, otype): return _price(self.pricer, self.bhav, spot, strike, dte_entry, expiry_str, otype, entry_str)

        ledger = PositionLedger(wlabel, entry_str, expiry_str, entry_spot=spot)
        ledger.add_transaction(entry_str, 'OPEN', 'CE', short_ce, 'SELL', p(short_ce, 'CE'), 0, 'condor entry short CE')
        ledger.add_transaction(entry_str, 'OPEN', 'CE', long_ce,  'BUY',  p(long_ce,  'CE'), 0, 'condor entry long CE wing')
        ledger.add_transaction(entry_str, 'OPEN', 'PE', short_pe, 'SELL', p(short_pe, 'PE'), 0, 'condor entry short PE')
        ledger.add_transaction(entry_str, 'OPEN', 'PE', long_pe,  'BUY',  p(long_pe,  'PE'), 0, 'condor entry long PE wing')

        entry_meta = {
            'ce_strike': short_ce, 'pe_strike': short_pe,
            'atm_iv': entry_sig.get('atm_iv'), 'hv': entry_sig.get('hv_20d'),
            'vrp_ratio': entry_sig.get('vrp_ratio'),
            'signal_vrp': entry_sig.get('signal_vrp'),
            'signal_regime': entry_sig.get('signal_regime'),
            'signal_vov': entry_sig.get('signal_vov'),
        }
        credit = ledger.running_net_per_unit   # net credit after all 4 legs

        for day in self._daily_dates(entry_date, expiry_date):
            day_str  = str(day)
            cur_spot = self._get_spot(day)
            if cur_spot <= 0:
                continue
            dte_now   = self.cal.get_dte(day, expiry_date)
            is_expiry = (day == expiry_date)
            near_exp  = self.cal.is_too_close_to_expiry(day, expiry_date)

            cur_sce = ledger.current_ce_strike or short_ce
            cur_spe = ledger.current_pe_strike or short_pe
            cur_lce = long_ce
            cur_lpe = long_pe

            def cp(k, ot): return _price(self.pricer, self.bhav, cur_spot, k, dte_now, expiry_str, ot, day_str)

            cost_to_close = cp(cur_sce, 'CE') - cp(cur_lce, 'CE') + cp(cur_spe, 'PE') - cp(cur_lpe, 'PE')

            if credit > 0 and cost_to_close >= credit * self.sl_mult:
                for k, ot, act in [(cur_sce, 'CE', 'BUY'), (cur_lce, 'CE', 'SELL'),
                                   (cur_spe, 'PE', 'BUY'), (cur_lpe, 'PE', 'SELL')]:
                    ledger.add_transaction(day_str, 'CLOSE', ot, k, act, cp(k, ot), 0, 'SL_HIT')
                ledger.close(day_str, 'SL_HIT')
                break
            if credit > 0 and cost_to_close <= credit * TARGET_DECAY:
                for k, ot, act in [(cur_sce, 'CE', 'BUY'), (cur_lce, 'CE', 'SELL'),
                                   (cur_spe, 'PE', 'BUY'), (cur_lpe, 'PE', 'SELL')]:
                    ledger.add_transaction(day_str, 'CLOSE', ot, k, act, cp(k, ot), 0, 'TARGET_HIT')
                ledger.close(day_str, 'TARGET_HIT')
                break
            if is_expiry:
                for k, ot, sign in [(cur_sce, 'CE', 1), (cur_lce, 'CE', -1),
                                    (cur_spe, 'PE', 1), (cur_lpe, 'PE', -1)]:
                    intrinsic = max(0, cur_spot - k) if ot == 'CE' else max(0, k - cur_spot)
                    act = 'BUY' if sign == 1 else 'SELL'
                    ledger.add_transaction(day_str, 'SETTLE', ot, k, act, intrinsic, 0, 'expiry')
                ledger.close(day_str, 'EXPIRY')
                break
            if not near_exp:
                decision = self.shift_eval.evaluate_condor(
                    cur_spot, cur_sce, cur_spe, cur_lce, cur_lpe,
                    expiry_date, day, ledger.running_net_per_unit)
                if decision.should_shift:
                    new_sce = decision.new_ce_strike or cur_sce
                    new_spe = decision.new_pe_strike or cur_spe
                    new_lce = _round50(new_sce + spot * WING_PCT / 100)
                    new_lpe = _round50(new_spe - spot * WING_PCT / 100)
                    for k, ot, act in [(cur_sce, 'CE', 'BUY'), (cur_lce, 'CE', 'SELL'),
                                       (cur_spe, 'PE', 'BUY'), (cur_lpe, 'PE', 'SELL')]:
                        ledger.add_transaction(day_str, 'CLOSE', ot, k, act, cp(k, ot), 1, 'condor shift close')
                    for k, ot, act in [(new_sce, 'CE', 'SELL'), (new_lce, 'CE', 'BUY'),
                                       (new_spe, 'PE', 'SELL'), (new_lpe, 'PE', 'BUY')]:
                        ledger.add_transaction(day_str, 'OPEN', ot, k, act,
                                               cp(k, ot), 1, 'condor shift open')
                    ledger.mark_shift()
                    long_ce  = new_lce
                    long_pe  = new_lpe

        if ledger.is_active:
            ledger.close(expiry_str, 'EXPIRY_FALLBACK')
        _save_summary(conn, 'condor_', ledger, entry_meta, 'IRON_CONDOR')

    # ── BEAR CALL SPREAD ──────────────────────────────────────────────────────

    def _simulate_callspread(self, entry_date: date_type, expiry_date: date_type,
                              entry_sig: dict, conn: sqlite3.Connection):
        entry_str  = str(entry_date)
        expiry_str = str(expiry_date)
        wlabel     = _week_label(entry_date) + '_BCS'
        dte_entry  = self.cal.get_dte(entry_date, expiry_date)

        spot = self._get_entry_spot(entry_date)
        if spot <= 0:
            return

        sell_strike = _round50(spot)
        buy_strike  = _round50(spot + 100)

        def p(k, ot, dte=None): return _price(self.pricer, self.bhav, spot, k, dte or dte_entry, expiry_str, ot, entry_str)

        ledger = PositionLedger(wlabel, entry_str, expiry_str, entry_spot=spot)
        ledger.add_transaction(entry_str, 'OPEN', 'CE', sell_strike, 'SELL', p(sell_strike, 'CE'), 0, 'callspread entry sell')
        ledger.add_transaction(entry_str, 'OPEN', 'CE', buy_strike,  'BUY',  p(buy_strike,  'CE'), 0, 'callspread entry buy')

        entry_meta = {
            'ce_strike': sell_strike, 'pe_strike': buy_strike,
            'atm_iv': entry_sig.get('atm_iv'), 'hv': entry_sig.get('hv_20d'),
            'vrp_ratio': entry_sig.get('vrp_ratio'),
            'signal_vrp': entry_sig.get('signal_vrp'),
            'signal_regime': entry_sig.get('signal_regime'),
            'signal_vov': entry_sig.get('signal_vov'),
        }

        for day in self._daily_dates(entry_date, expiry_date):
            day_str  = str(day)
            cur_spot = self._get_spot(day)
            if cur_spot <= 0:
                continue
            dte_now   = self.cal.get_dte(day, expiry_date)
            is_expiry = (day == expiry_date)

            def cp(k, ot): return _price(self.pricer, self.bhav, cur_spot, k, dte_now, expiry_str, ot, day_str)

            if is_expiry:
                settle_sell = max(0, cur_spot - sell_strike)
                settle_buy  = max(0, cur_spot - buy_strike)
                ledger.add_transaction(day_str, 'SETTLE', 'CE', sell_strike, 'BUY',  settle_sell, 0, 'expiry')
                ledger.add_transaction(day_str, 'SETTLE', 'CE', buy_strike,  'SELL', settle_buy,  0, 'expiry')
                ledger.close(day_str, 'EXPIRY')
                break

            # Signal-only exit: VRP flipped
            sig_today = self._get_signal(day_str)
            vrp_r = sig_today.get('vrp_ratio')
            if vrp_r is not None and vrp_r < 1.0:
                close_sell = cp(sell_strike, 'CE')
                close_buy  = cp(buy_strike,  'CE')
                ledger.add_transaction(day_str, 'CLOSE', 'CE', sell_strike, 'BUY',  close_sell, 0, 'VRP signal exit')
                ledger.add_transaction(day_str, 'CLOSE', 'CE', buy_strike,  'SELL', close_buy,  0, 'VRP signal exit')
                ledger.close(day_str, 'SIGNAL_EXIT')
                break

        if ledger.is_active:
            ledger.close(expiry_str, 'EXPIRY_FALLBACK')
        _save_summary(conn, 'callspread_', ledger, entry_meta, 'BEAR_CALL_SPREAD')

    # ── aggregate stats ───────────────────────────────────────────────────────

    def aggregate_stats(self, strategy: str = 'strangle') -> dict:
        """Read position_summaries from DB and compute aggregate stats."""
        prefix_map = {
            'strangle':   '',
            'straddle':   'straddle_',
            'condor':     'condor_',
            'callspread': 'callspread_',
        }
        prefix = prefix_map.get(strategy, '')
        conn = sqlite3.connect(DB_PATH)
        try:
            df = pd.read_sql_query(f'SELECT * FROM {prefix}position_summaries', conn)
        except Exception:
            conn.close()
            return {}
        conn.close()

        if df.empty:
            return {'strategy': strategy, 'positions': 0}

        pnls = df['final_pnl_rupees'].values
        wins = pnls[pnls > 0]
        cum  = np.cumsum(pnls)
        peak = np.maximum.accumulate(cum)
        dd   = cum - peak

        sharpe = (pnls.mean() / (pnls.std() + 1e-9)) * np.sqrt(52) if len(pnls) > 1 else 0
        wr     = len(wins) / len(pnls)
        avg_w  = float(wins.mean()) if len(wins) > 0 else 0
        avg_l  = float(abs(pnls[pnls <= 0].mean())) if len(pnls[pnls <= 0]) > 0 else 1
        b      = avg_w / avg_l if avg_l > 0 else 0
        kelly  = min(max((wr * b - (1 - wr)) / b if b > 0 else 0, 0), 0.25)

        avg_shifts = df['shift_count'].mean()
        never_shifted = df[df['shift_count'] == 0]
        shifted_once  = df[df['shift_count'] == 1]
        shifted_multi = df[df['shift_count'] >= 2]

        return {
            'strategy':           strategy.upper(),
            'positions':          len(pnls),
            'total_pnl_rupees':   round(float(pnls.sum()), 0),
            'win_rate_pct':       round(wr * 100, 1),
            'avg_pnl_rupees':     round(float(pnls.mean()), 0),
            'max_drawdown':       round(float(dd.min()), 0),
            'sharpe':             round(float(sharpe), 2),
            'kelly_pct':          round(kelly * 100, 1),
            'avg_shifts':         round(float(avg_shifts), 2),
            'no_shift_count':     len(never_shifted),
            'no_shift_avg_pnl':   round(float(never_shifted['final_pnl_rupees'].mean()), 0) if len(never_shifted) > 0 else 0,
            'once_shift_count':   len(shifted_once),
            'once_shift_avg_pnl': round(float(shifted_once['final_pnl_rupees'].mean()), 0) if len(shifted_once) > 0 else 0,
            'multi_shift_count':  len(shifted_multi),
            'multi_shift_avg_pnl': round(float(shifted_multi['final_pnl_rupees'].mean()), 0) if len(shifted_multi) > 0 else 0,
        }
