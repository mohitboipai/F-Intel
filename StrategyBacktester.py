"""
StrategyBacktester.py
=====================
Walk-forward backtester for NIFTY option strategies.
Uses NSE Bhavcopy historical data (reuses BhavcopyFetcher from OIBacktester.py).

Simulates weekly-expiry option strategies over historical data.
Entry: Monday of each expiry week (or first available trading day).
Exit:  Thursday expiry (at intrinsic value) or stop-loss.

Usage:
    bt = OptionStrategyBacktester()
    report = bt.run('SHORT_STRADDLE', days=365)
    html = report.to_html()
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Reuse existing infrastructure
sys.path.insert(0, os.path.dirname(__file__))
from OIBacktester import BhavcopyFetcher, OIDataProcessor
from StrategyEngine import bsm_price, NIFTY_LOT, DARK_BG, CARD_BG, GREEN, RED, ACCENT, YELLOW, WHITE, MUTED, ORANGE

# ──────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────
SUPPORTED_STRATEGIES = [
    'SHORT_STRADDLE',
    'SHORT_STRANGLE',
    'IRON_CONDOR',
    'BULL_PUT_SPREAD',
    'BEAR_CALL_SPREAD',
]

DEFAULT_STOP_LOSS_MULT = 2.0   # Exit when loss > 2x premium received
DEFAULT_LOTS           = 1
RISK_FREE              = 0.07
WING_WIDTH             = 150   # Iron condor wing width (points)
STRANGLE_OTM_PCT       = 0.03  # 3% OTM for strangle

def _round50(x): return round(x / 50) * 50


# ──────────────────────────────────────────────────────────────
# TradeRecord — single backtest trade
# ──────────────────────────────────────────────────────────────
class TradeRecord:
    def __init__(self, entry_date: str, expiry_date: str, strategy: str,
                 spot: float, legs: list, premium: float):
        self.entry_date  = entry_date
        self.expiry_date = expiry_date
        self.strategy    = strategy
        self.entry_spot  = spot
        self.legs        = legs        # [{'type','action','strike','entry_price','iv'}]
        self.premium     = premium     # net premium received (per unit)
        self.exit_pnl    = None        # set after exit
        self.exit_date   = None
        self.exit_reason = ''          # 'EXPIRY' | 'STOP_LOSS'
        self.lot_pnl     = None        # exit_pnl * NIFTY_LOT

    def compute_expiry_pnl(self, exit_spot: float) -> float:
        """Compute P&L at expiry."""
        total = 0.0
        for leg in self.legs:
            if leg['type'] == 'CE':
                intrinsic = max(0.0, exit_spot - leg['strike'])
            else:
                intrinsic = max(0.0, leg['strike'] - exit_spot)
            direction = 1 if leg['action'] == 'BUY' else -1
            total += direction * intrinsic - direction * leg['entry_price']
        return total

    def compute_mid_pnl(self, spot: float, T: float) -> float:
        """Estimate mid-cycle BSM P&L (for stop-loss check)."""
        total = 0.0
        for leg in self.legs:
            iv = leg.get('iv', 0.15)
            current = bsm_price(spot, leg['strike'], T, RISK_FREE, iv, leg['type'])
            direction = 1 if leg['action'] == 'BUY' else -1
            total += direction * (current - leg['entry_price'])
        return total

    def to_dict(self) -> dict:
        return {
            'entry_date': self.entry_date,
            'expiry_date': self.expiry_date,
            'strategy': self.strategy,
            'entry_spot': self.entry_spot,
            'premium': round(self.premium, 2),
            'lot_pnl': round(self.lot_pnl, 2) if self.lot_pnl is not None else None,
            'exit_date': self.exit_date,
            'exit_reason': self.exit_reason,
        }


# ──────────────────────────────────────────────────────────────
# BacktestReport
# ──────────────────────────────────────────────────────────────
class BacktestReport:
    def __init__(self, strategy_name: str, trades: List[TradeRecord],
                 start_date: str, end_date: str, params: dict):
        self.strategy_name = strategy_name
        self.trades        = [t for t in trades if t.lot_pnl is not None]
        self.start_date    = start_date
        self.end_date      = end_date
        self.params        = params
        self._compute_stats()

    def _compute_stats(self):
        pnls = [t.lot_pnl for t in self.trades]
        if not pnls:
            self.stats = {}
            return
        pnl_arr = np.array(pnls)
        wins    = pnl_arr[pnl_arr > 0]
        losses  = pnl_arr[pnl_arr <= 0]

        cumulative = np.cumsum(pnl_arr)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns   = cumulative - running_max
        max_dd      = float(np.min(drawdowns))

        weekly_ret  = pnl_arr / (np.array([t.premium for t in self.trades]) * NIFTY_LOT + 1e-9)
        sharpe      = (weekly_ret.mean() / (weekly_ret.std() + 1e-9)) * np.sqrt(52) if len(weekly_ret) > 1 else 0

        self.stats = {
            'total_trades':   len(self.trades),
            'win_rate':       len(wins) / len(pnl_arr) * 100,
            'avg_pnl':        float(np.mean(pnl_arr)),
            'total_pnl':      float(np.sum(pnl_arr)),
            'max_profit':     float(np.max(pnl_arr)) if len(pnl_arr) else 0,
            'max_loss':       float(np.min(pnl_arr)) if len(pnl_arr) else 0,
            'avg_win':        float(np.mean(wins)) if len(wins) else 0,
            'avg_loss':       float(np.mean(losses)) if len(losses) else 0,
            'max_drawdown':   max_dd,
            'sharpe':         float(sharpe),
            'profit_factor':  (abs(np.sum(wins)) / (abs(np.sum(losses)) + 1e-9)),
            'cumulative':     cumulative.tolist(),
        }

    def build_equity_chart(self) -> go.Figure:
        if not self.trades:
            return go.Figure()

        dates    = [t.exit_date or t.expiry_date for t in self.trades]
        cumulative = self.stats.get('cumulative', [])

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=dates, y=cumulative, name='Cumulative P&L',
            fill='tozeroy',
            fillcolor='rgba(102,187,106,0.10)',
            line=dict(color=GREEN, width=2),
            hovertemplate='%{x}<br>₹%{y:,.0f}<extra></extra>'
        ))

        # Zero line
        fig.add_hline(y=0, line=dict(color=MUTED, width=1, dash='dot'))

        # Color segments
        prev_cum = 0
        for i, (d, t) in enumerate(zip(dates, self.trades)):
            color = GREEN if t.lot_pnl > 0 else RED
            fig.add_trace(go.Bar(
                x=[d], y=[t.lot_pnl], name='',
                marker_color=color, opacity=0.55,
                showlegend=False,
                hovertemplate=f'{d}: ₹{t.lot_pnl:,.0f}<extra></extra>'
            ))

        fig.update_layout(
            title=dict(text=f'{self.strategy_name} — Equity Curve', font=dict(color=WHITE, size=14)),
            paper_bgcolor=DARK_BG, plot_bgcolor='rgba(20,20,35,0.9)',
            font=dict(color=WHITE, family='Inter, sans-serif', size=11),
            height=320, margin=dict(l=60, r=20, t=50, b=40),
            xaxis=dict(title='Date', gridcolor='rgba(100,100,100,0.12)'),
            yaxis=dict(title='Cumulative P&L (₹)', gridcolor='rgba(100,100,100,0.12)',
                       tickformat=',.0f'),
            hovermode='x unified',
            showlegend=False,
        )
        return fig

    def build_monthly_chart(self) -> go.Figure:
        if not self.trades:
            return go.Figure()

        df = pd.DataFrame([{
            'month': (t.exit_date or t.expiry_date)[:7],
            'pnl': t.lot_pnl
        } for t in self.trades])

        monthly = df.groupby('month')['pnl'].sum().reset_index()
        colors  = [GREEN if p > 0 else RED for p in monthly['pnl']]

        fig = go.Figure(go.Bar(
            x=monthly['month'], y=monthly['pnl'],
            marker_color=colors, opacity=0.8,
            hovertemplate='%{x}: ₹%{y:,.0f}<extra></extra>'
        ))
        fig.update_layout(
            title=dict(text='Monthly P&L Breakdown', font=dict(color=WHITE, size=13)),
            paper_bgcolor=DARK_BG, plot_bgcolor='rgba(20,20,35,0.9)',
            font=dict(color=WHITE, family='Inter, sans-serif', size=11),
            height=240, margin=dict(l=60, r=20, t=45, b=40),
            xaxis=dict(gridcolor='rgba(100,100,100,0.12)'),
            yaxis=dict(title='P&L (₹)', gridcolor='rgba(100,100,100,0.12)', tickformat=',.0f'),
        )
        return fig

    def build_distribution_chart(self) -> go.Figure:
        if not self.trades:
            return go.Figure()
        pnls = [t.lot_pnl for t in self.trades]
        fig = go.Figure(go.Histogram(
            x=pnls, nbinsx=25,
            marker=dict(
                color=[GREEN if p > 0 else RED for p in pnls],
                line=dict(color='rgba(0,0,0,0.3)', width=0.5)
            ),
            hovertemplate='P&L: ₹%{x:,.0f}<br>Count: %{y}<extra></extra>'
        ))
        fig.add_vline(x=0, line=dict(color=MUTED, dash='dot', width=1))
        fig.update_layout(
            title=dict(text='Return Distribution', font=dict(color=WHITE, size=13)),
            paper_bgcolor=DARK_BG, plot_bgcolor='rgba(20,20,35,0.9)',
            font=dict(color=WHITE, family='Inter, sans-serif', size=11),
            height=220, margin=dict(l=60, r=20, t=45, b=40),
            xaxis=dict(title='P&L per Trade (₹)', gridcolor='rgba(100,100,100,0.12)', tickformat=',.0f'),
            yaxis=dict(title='Frequency', gridcolor='rgba(100,100,100,0.12)'),
        )
        return fig

    def stats_table_html(self) -> str:
        s = self.stats
        if not s:
            return '<p style="color:#888;">No trades to show.</p>'
        wr_color    = GREEN if s['win_rate'] >= 60 else YELLOW if s['win_rate'] >= 50 else RED
        pnl_color   = GREEN if s['avg_pnl'] > 0 else RED
        sharp_color = GREEN if s['sharpe'] >= 1.0 else YELLOW if s['sharpe'] >= 0.5 else RED

        rows = [
            ('Total Trades',    s['total_trades'],          WHITE,       ''),
            ('Win Rate',        f"{s['win_rate']:.1f}%",    wr_color,    ''),
            ('Avg P&L / Trade', f"₹{s['avg_pnl']:,.0f}",   pnl_color,   ''),
            ('Total P&L',       f"₹{s['total_pnl']:,.0f}",  pnl_color,  ''),
            ('Best Trade',      f"₹{s['max_profit']:,.0f}", GREEN,       ''),
            ('Worst Trade',     f"₹{s['max_loss']:,.0f}",   RED,         ''),
            ('Max Drawdown',    f"₹{s['max_drawdown']:,.0f}", RED,       ''),
            ('Sharpe Ratio',    f"{s['sharpe']:.2f}",       sharp_color, ''),
            ('Profit Factor',   f"{s['profit_factor']:.2f}", GREEN if s['profit_factor'] > 1 else RED, ''),
        ]

        html = '<table style="width:100%;border-collapse:collapse;font-size:12px;">'
        html += f'<thead><tr style="color:{MUTED};border-bottom:1px solid #333;">'
        html += '<th style="text-align:left;padding:5px 10px;">Metric</th>'
        html += '<th style="text-align:right;padding:5px 10px;">Value</th></tr></thead><tbody>'

        for label, val, color, _ in rows:
            html += (f'<tr><td style="padding:5px 10px;color:{MUTED};">{label}</td>'
                     f'<td style="padding:5px 10px;text-align:right;font-weight:700;color:{color};">'
                     f'{val}</td></tr>')
        html += '</tbody></table>'
        return html

    def to_html(self) -> str:
        """Returns full HTML for embedding in the dashboard fragment."""
        equity_html  = self.build_equity_chart().to_html(include_plotlyjs=False, full_html=False)
        monthly_html = self.build_monthly_chart().to_html(include_plotlyjs=False, full_html=False)
        dist_html    = self.build_distribution_chart().to_html(include_plotlyjs=False, full_html=False)
        stats_html   = self.stats_table_html()

        return f'''
        <div style="border:1px solid #2a2a4a;border-radius:10px;padding:16px;background:#12122a;margin-top:12px;">
            <div style="color:{ACCENT};font-size:13px;font-weight:700;letter-spacing:2px;margin-bottom:12px;">
                📊 BACKTEST RESULTS — {self.strategy_name}
                <span style="color:{MUTED};font-size:10px;margin-left:12px;font-weight:400;">
                    {self.start_date} → {self.end_date} | {self.stats.get("total_trades",0)} trades
                </span>
            </div>
            <div style="display:grid;grid-template-columns:1fr 1.8fr;gap:14px;">
                <div>{stats_html}</div>
                <div>{equity_html}</div>
            </div>
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-top:12px;">
                <div>{monthly_html}</div>
                <div>{dist_html}</div>
            </div>
            <div style="margin-top:10px;padding:6px 10px;background:#0d0d1e;border-radius:6px;">
                <span style="color:{MUTED};font-size:10px;">
                    ⚠ EOD Bhavcopy prices used — entry/exit at closing prices. Slippage not modelled.
                    Stop-loss: {self.params.get("stop_loss_mult", 2.0)}x premium received.
                </span>
            </div>
        </div>'''

    def to_json(self) -> str:
        return json.dumps({
            'strategy': self.strategy_name,
            'start': self.start_date, 'end': self.end_date,
            'stats': self.stats,
            'trades': [t.to_dict() for t in self.trades],
            'params': self.params,
        })


# ──────────────────────────────────────────────────────────────
# OptionStrategyBacktester  — main runner
# ──────────────────────────────────────────────────────────────
class OptionStrategyBacktester:
    """
    Walk-forward option strategy backtester using NSE Bhavcopy data.

    For each week ending on Thursday (expiry), simulates the chosen strategy:
      Entry: Monday open (approximated by Monday close from Bhavcopy)
      Exit:  Thursday at intrinsic (expiry) OR stop-loss mid-week
    """

    def __init__(self):
        self.fetcher   = BhavcopyFetcher()
        self.processor = OIDataProcessor()

    def run(self, strategy_type: str = 'SHORT_STRADDLE',
            days: int = 365,
            stop_loss_mult: float = DEFAULT_STOP_LOSS_MULT,
            lots: int = DEFAULT_LOTS,
            atm_iv_pct: float = 14.0,
            wing_width: int = WING_WIDTH) -> BacktestReport:
        """
        Run a walk-forward backtest.

        Args:
            strategy_type: One of SUPPORTED_STRATEGIES
            days: Lookback period in calendar days
            stop_loss_mult: Exit if loss > N x premium
            lots: Number of lots per trade
            atm_iv_pct: Assumed ATM IV % (used when live IV not available from Bhavcopy)
            wing_width: For Iron Condor — spread width in points

        Returns:
            BacktestReport
        """
        end_dt   = datetime.now()
        start_dt = end_dt - timedelta(days=days)
        start_str = start_dt.strftime('%Y-%m-%d')
        end_str   = end_dt.strftime('%Y-%m-%d')

        print(f"[Backtest] {strategy_type} | {start_str} → {end_str}")

        # 1. Fetch bhavcopy data
        try:
            raw_data = self.fetcher.fetch_date_range(start_dt, end_dt)
        except Exception as e:
            print(f"[Backtest] Data fetch error: {e}")
            return BacktestReport(strategy_type, [], start_str, end_str,
                                  {'stop_loss_mult': stop_loss_mult})

        if not raw_data:
            print("[Backtest] No data returned.")
            return BacktestReport(strategy_type, [], start_str, end_str,
                                  {'stop_loss_mult': stop_loss_mult})

        # 2. Build master dataset
        spot_series = self._extract_spot_series(raw_data)
        master_df   = self.processor.build_master_dataset(raw_data, spot_series)

        if master_df.empty:
            print("[Backtest] Empty master dataset.")
            return BacktestReport(strategy_type, [], start_str, end_str,
                                  {'stop_loss_mult': stop_loss_mult})

        # 3. Identify weekly Thursday expiries
        thursdays = self._get_thursdays(start_dt, end_dt)
        print(f"[Backtest] {len(thursdays)} weekly expiries found")

        # 4. Simulate each expiry week
        trades  = []
        all_dates = sorted(master_df['date'].unique())

        for expiry_dt in thursdays:
            expiry_str = expiry_dt.strftime('%Y-%m-%d')
            # Entry: Monday of the same week
            entry_dt  = expiry_dt - timedelta(days=3)
            entry_str = entry_dt.strftime('%Y-%m-%d')

            # Find nearest available dates
            entry_str = self._nearest_date(all_dates, entry_str) or expiry_str
            if entry_str not in all_dates or expiry_str not in all_dates:
                continue

            trade = self._simulate_week(
                master_df, spot_series, entry_str, expiry_str,
                strategy_type, atm_iv_pct / 100.0, stop_loss_mult,
                lots, wing_width, all_dates
            )
            if trade:
                trades.append(trade)

        print(f"[Backtest] Completed: {len(trades)} trades simulated")

        params = {
            'stop_loss_mult': stop_loss_mult,
            'lots': lots,
            'atm_iv_pct': atm_iv_pct,
            'wing_width': wing_width,
        }
        return BacktestReport(strategy_type, trades, start_str, end_str, params)

    def _simulate_week(self, master_df, spot_series, entry_date, expiry_date,
                       strategy_type, iv_dec, stop_loss_mult, lots, wing_width, all_dates):
        """Simulate one strategy from entry to expiry or SL."""
        entry_spot = spot_series.get(entry_date, 0)
        if entry_spot <= 0:
            return None

        expiry_spot = spot_series.get(expiry_date, 0)
        if expiry_spot <= 0:
            return None

        # Entry day OI data
        day_df = master_df[master_df['date'] == entry_date].copy()
        if day_df.empty:
            return None

        T_entry  = self._dte_frac(entry_date, expiry_date)
        atm      = _round50(entry_spot)

        try:
            legs = self._build_legs(strategy_type, entry_spot, atm, T_entry,
                                    iv_dec, day_df, lots, wing_width)
        except Exception as e:
            return None

        if not legs:
            return None

        net_premium = sum(
            (-1 if l['action'] == 'SELL' else 1) * l['entry_price'] *
            (-1)  # SELL = receive premium (+)
            for l in legs
        )
        # Correct sign: SELL premium received is positive for seller
        net_premium = sum(
            (1 if l['action'] == 'SELL' else -1) * l['entry_price']
            for l in legs
        )
        stop_loss_level = -stop_loss_mult * net_premium  # per unit

        trade = TradeRecord(entry_date, expiry_date, strategy_type,
                            entry_spot, legs, net_premium)

        # Mid-week stop-loss check (Wednesday)
        week_dates = [d for d in all_dates if entry_date < d < expiry_date]
        for mid_date in week_dates:
            mid_spot = spot_series.get(mid_date, 0)
            if mid_spot <= 0: continue
            T_mid = self._dte_frac(mid_date, expiry_date)
            mid_pnl = trade.compute_mid_pnl(mid_spot, T_mid)
            if mid_pnl < stop_loss_level:
                trade.exit_pnl    = mid_pnl
                trade.exit_date   = mid_date
                trade.exit_reason = 'STOP_LOSS'
                trade.lot_pnl     = mid_pnl * NIFTY_LOT * lots
                return trade

        # Expiry exit
        exp_pnl          = trade.compute_expiry_pnl(expiry_spot)
        trade.exit_pnl   = exp_pnl
        trade.exit_date  = expiry_date
        trade.exit_reason = 'EXPIRY'
        trade.lot_pnl    = exp_pnl * NIFTY_LOT * lots
        return trade

    def _build_legs(self, strategy_type, spot, atm, T, iv_dec, day_df,
                    lots, wing_width) -> list:
        """Build strategy legs from entry-day chain data."""
        def price(strike, opt_type):
            row = day_df[(day_df['strike'] == strike) & (day_df['type'] == opt_type)]
            if not row.empty:
                p = float(row.iloc[0]['close']) if 'close' in row.columns else float(row.iloc[0].get('price', 0))
                if p > 0: return p
            # Fallback to BSM
            return max(0.5, bsm_price(spot, strike, T, 0.07, iv_dec, opt_type))

        if strategy_type == 'SHORT_STRADDLE':
            return [
                {'type': 'CE', 'action': 'SELL', 'strike': atm, 'entry_price': price(atm, 'CE'), 'iv': iv_dec},
                {'type': 'PE', 'action': 'SELL', 'strike': atm, 'entry_price': price(atm, 'PE'), 'iv': iv_dec},
            ]
        elif strategy_type == 'SHORT_STRANGLE':
            ce_s = _round50(spot * (1 + STRANGLE_OTM_PCT))
            pe_s = _round50(spot * (1 - STRANGLE_OTM_PCT))
            return [
                {'type': 'CE', 'action': 'SELL', 'strike': ce_s, 'entry_price': price(ce_s, 'CE'), 'iv': iv_dec},
                {'type': 'PE', 'action': 'SELL', 'strike': pe_s, 'entry_price': price(pe_s, 'PE'), 'iv': iv_dec},
            ]
        elif strategy_type == 'IRON_CONDOR':
            short_ce = _round50(spot * 1.03)
            long_ce  = _round50(short_ce + wing_width)
            short_pe = _round50(spot * 0.97)
            long_pe  = _round50(short_pe - wing_width)
            return [
                {'type': 'CE', 'action': 'SELL', 'strike': short_ce, 'entry_price': price(short_ce, 'CE'), 'iv': iv_dec},
                {'type': 'CE', 'action': 'BUY',  'strike': long_ce,  'entry_price': price(long_ce,  'CE'), 'iv': iv_dec},
                {'type': 'PE', 'action': 'SELL', 'strike': short_pe, 'entry_price': price(short_pe, 'PE'), 'iv': iv_dec},
                {'type': 'PE', 'action': 'BUY',  'strike': long_pe,  'entry_price': price(long_pe,  'PE'), 'iv': iv_dec},
            ]
        elif strategy_type == 'BULL_PUT_SPREAD':
            sell_pe = _round50(spot * 0.97)
            buy_pe  = _round50(sell_pe - wing_width)
            return [
                {'type': 'PE', 'action': 'SELL', 'strike': sell_pe, 'entry_price': price(sell_pe, 'PE'), 'iv': iv_dec},
                {'type': 'PE', 'action': 'BUY',  'strike': buy_pe,  'entry_price': price(buy_pe,  'PE'), 'iv': iv_dec},
            ]
        elif strategy_type == 'BEAR_CALL_SPREAD':
            sell_ce = _round50(spot * 1.03)
            buy_ce  = _round50(sell_ce + wing_width)
            return [
                {'type': 'CE', 'action': 'SELL', 'strike': sell_ce, 'entry_price': price(sell_ce, 'CE'), 'iv': iv_dec},
                {'type': 'CE', 'action': 'BUY',  'strike': buy_ce,  'entry_price': price(buy_ce,  'CE'), 'iv': iv_dec},
            ]
        return []

    # ── Utilities ─────────────────────────────────────────────
    @staticmethod
    def _extract_spot_series(raw_data: dict) -> dict:
        """Extract spot price per date from NIFTY futures or index row in Bhavcopy."""
        spot_series = {}
        for date_str, df in raw_data.items():
            if df is None or df.empty: continue
            cols = [c.upper() for c in df.columns]
            df.columns = cols
            # Try NIFTY spot from futures
            try:
                nifty = df[df.get('SYMBOL', df.get('TCKRSYMB', pd.Series())).str.contains('NIFTY', na=False)]
                if not nifty.empty:
                    close_col = next((c for c in ['CLOSE', 'CLSPRIC', 'LASTPRIC'] if c in nifty.columns), None)
                    if close_col:
                        spot_series[date_str] = float(nifty.iloc[0][close_col])
                        continue
            except Exception:
                pass
            # Fallback: estimate from ATM strike midpoint of CE/PE
            try:
                ce = df[df.get('OPTION_TYP', df.get('OPTNTP', pd.Series())).str.upper() == 'CE']
                pe = df[df.get('OPTION_TYP', df.get('OPTNTP', pd.Series())).str.upper() == 'PE']
                if not ce.empty and not pe.empty:
                    close_col = next((c for c in ['CLOSE', 'CLSPRIC'] if c in df.columns), None)
                    strike_col = next((c for c in ['STRIKE_PR', 'STRKPRIC'] if c in df.columns), None)
                    if close_col and strike_col:
                        ce_row = ce.nlargest(1, close_col)
                        spot_series[date_str] = float(ce_row.iloc[0][strike_col])
            except Exception:
                pass
        return spot_series

    @staticmethod
    def _get_thursdays(start_dt: datetime, end_dt: datetime) -> List[datetime]:
        """Return all Thursdays between start and end."""
        thursdays = []
        d = start_dt
        while d <= end_dt:
            if d.weekday() == 3:  # Thursday
                thursdays.append(d)
            d += timedelta(days=1)
        return thursdays

    @staticmethod
    def _nearest_date(all_dates: list, target: str) -> Optional[str]:
        if target in all_dates:
            return target
        # Next available date
        for d in sorted(all_dates):
            if d >= target:
                return d
        return None

    @staticmethod
    def _dte_frac(entry_date: str, expiry_date: str) -> float:
        """Days to expiry as fraction of year."""
        try:
            e = datetime.strptime(expiry_date, '%Y-%m-%d')
            n = datetime.strptime(entry_date, '%Y-%m-%d')
            dte = max(0, (e - n).days)
            return dte / 365.0
        except:
            return 7 / 365.0
