"""
BacktestReporter.py
===================
Reads the trades table from data/backtest_results.db and produces a
self-contained, dark-themed HTML report at data/backtest_report.html.

Report sections:
  1. Summary stats (total P&L, trade count, win rate, Sharpe, max drawdown,
     Kelly recommendation)
  2. Breakdown by signal_type × strategy_type
  3. Month-by-month P&L table
  4. Full trade log (colour-coded P&L)
  5. Inline SVG cumulative P&L chart with zero line + drawdown shading

Usage:
    reporter = BacktestReporter()
    html_path = reporter.generate()
    print(f"Report: {html_path}")
"""

import os
import sys
import sqlite3
import numpy as np
import pandas as pd
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

DB_PATH      = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             'data', 'backtest_results.db')
REPORT_PATH  = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             'data', 'backtest_report.html')


# ── BacktestReporter ──────────────────────────────────────────────────────────
class BacktestReporter:

    def __init__(self, db_path: str = DB_PATH, report_path: str = REPORT_PATH):
        self.db_path     = db_path
        self.report_path = report_path

    # ── data loading ─────────────────────────────────────────────────────────

    def _load_trades(self) -> pd.DataFrame:
        if not os.path.exists(self.db_path):
            return pd.DataFrame()
        try:
            conn = sqlite3.connect(self.db_path)
            df = pd.read_sql_query(
                'SELECT * FROM trades ORDER BY entry_date ASC', conn
            )
            conn.close()
            return df
        except Exception as exc:
            print(f'[BacktestReporter] DB read error: {exc}')
            return pd.DataFrame()

    # ── stats helper ─────────────────────────────────────────────────────────

    @staticmethod
    def _stats(pnls):
        pnls = np.array(pnls, dtype=float)
        if len(pnls) == 0:
            return {}
        wins   = pnls[pnls > 0]
        losses = pnls[pnls <= 0]
        cum    = np.cumsum(pnls)
        peak   = np.maximum.accumulate(cum)
        dd     = cum - peak
        sharpe = (pnls.mean() / (pnls.std() + 1e-9)) * np.sqrt(52) if len(pnls) > 1 else 0
        avg_w  = float(wins.mean())  if len(wins)   > 0 else 0
        avg_l  = float(abs(losses.mean())) if len(losses) > 0 else 1
        wr     = len(wins) / len(pnls)
        b      = avg_w / avg_l if avg_l > 0 else 0
        kelly  = min(max((wr * b - (1 - wr)) / b if b > 0 else 0, 0), 0.25)
        return {
            'total_pnl':    round(float(pnls.sum()), 0),
            'trade_count':  len(pnls),
            'win_rate':     round(wr * 100, 1),
            'avg_pnl':      round(float(pnls.mean()), 0),
            'max_loss':     round(float(pnls.min()), 0),
            'max_drawdown': round(float(dd.min()), 0),
            'sharpe':       round(float(sharpe), 2),
            'kelly':        round(kelly * 100, 1),
            'cumulative':   cum.tolist(),
        }

    # ── SVG chart ─────────────────────────────────────────────────────────────

    @staticmethod
    def _svg_chart(cum: list, width: int = 900, height: int = 300) -> str:
        if not cum:
            return '<p style="color:#888">No data to chart.</p>'

        arr   = np.array(cum, dtype=float)
        n     = len(arr)
        lo    = float(arr.min())
        hi    = float(arr.max())
        span  = hi - lo if hi != lo else 1.0
        pad_x = 60
        pad_y = 30
        inner_w = width  - pad_x * 2
        inner_h = height - pad_y * 2

        def xp(i):
            return pad_x + i / max(n - 1, 1) * inner_w

        def yp(v):
            return pad_y + (1 - (v - lo) / span) * inner_h

        zero_y = yp(0)

        # Find max-drawdown period
        peak = np.maximum.accumulate(arr)
        dd   = arr - peak
        dd_end = int(np.argmin(dd))
        dd_start = int(np.argmax(arr[:dd_end + 1]))

        # Drawdown shading rectangle
        x0_dd = xp(dd_start)
        x1_dd = xp(dd_end)
        shading = (
            f'<rect x="{x0_dd:.1f}" y="{pad_y}" '
            f'width="{max(x1_dd - x0_dd, 2):.1f}" height="{inner_h}" '
            f'fill="rgba(255,68,68,0.08)" />'
        )

        # Define color and points
        line_color = "#66bb6a" if arr[-1] >= 0 else "#ff4444"
        pts = " ".join([f"{xp(i):.1f},{yp(v):.1f}" for i, v in enumerate(arr)])

        # Gradient definition for the line
        def_gradient = (
            f'<defs>'
            f'<linearGradient id="lineGrad" x1="0%" y1="0%" x2="0%" y2="100%">'
            f'<stop offset="0%" stop-color="{line_color}" stop-opacity="1"/>'
            f'<stop offset="100%" stop-color="{line_color}" stop-opacity="0.2"/>'
            f'</linearGradient>'
            f'</defs>'
        )

        polyline = f'<polyline points="{pts}" fill="none" stroke="url(#lineGrad)" stroke-width="2.5"/>'

        # Filled area under the curve
        area_pts = f'{xp(0):.1f},{zero_y:.1f} {pts} {xp(n-1):.1f},{zero_y:.1f}'
        area = f'<polygon points="{area_pts}" fill="url(#lineGrad)" opacity="0.4"/>'

        # Zero line
        zero_line = (
            f'<line x1="{pad_x}" y1="{zero_y:.1f}" '
            f'x2="{width - pad_x}" y2="{zero_y:.1f}" '
            f'stroke="#555" stroke-dasharray="4,4" stroke-width="1.5"/>'
        )

        # Y axis labels
        y_labels = []
        for frac in [0, 0.25, 0.5, 0.75, 1.0]:
            v = lo + frac * span
            y = yp(v)
            lbl = f'₹{v:,.0f}'
            y_labels.append(
                f'<text x="{pad_x - 6}" y="{y + 4:.1f}" '
                f'fill="#888" font-size="10" text-anchor="end">{lbl}</text>'
            )

        # X axis labels (trade numbers, equally spaced)
        x_labels = []
        ticks = min(8, n)
        for t in range(ticks + 1):
            i = int(t * (n - 1) / max(ticks, 1))
            x = xp(i)
            x_labels.append(
                f'<text x="{x:.1f}" y="{height - 8}" '
                f'fill="#888" font-size="10" text-anchor="middle">#{i + 1}</text>'
            )

        svg = (
            f'<svg width="100%" viewBox="0 0 {width} {height}" '
            f'xmlns="http://www.w3.org/2000/svg" style="background: linear-gradient(180deg, #12122a 0%, #0d0d1e 100%); border-radius:8px; border: 1px solid #2a2a44;">'
            f'{def_gradient}{shading}{zero_line}{area}{polyline}'
            + ''.join(y_labels) + ''.join(x_labels) +
            f'</svg>'
        )
        return svg

    # ── HTML generation ───────────────────────────────────────────────────────

    def generate(self) -> str:
        df = self._load_trades()

        if df.empty:
            html = self._html_shell('<div class="card"><p style="color:#888">No trades found in database.</p></div>')
        else:
            html = self._build_html(df)

        os.makedirs(os.path.dirname(self.report_path), exist_ok=True)
        with open(self.report_path, 'w', encoding='utf-8') as f:
            f.write(html)

        return self.report_path

    def _build_html(self, df: pd.DataFrame) -> str:
        pnls    = df['pnl_per_lot'].tolist()
        overall = self._stats(pnls)

        # ── 1. Summary ──────────────────────────────────────────────────────
        summary_html = self._summary_section(overall)

        # ── 2. Breakdown by signal × strategy ───────────────────────────────
        breakdown_html = self._breakdown_section(df)

        # ── 3. Monthly P&L ──────────────────────────────────────────────────
        monthly_html = self._monthly_section(df)

        # ── 4. Trade log ─────────────────────────────────────────────────────
        log_html = self._tradelog_section(df)

        # ── 5. SVG chart ─────────────────────────────────────────────────────
        cum = overall.get('cumulative', [])
        chart_html = self._svg_chart(cum)

        body = f'''
        {summary_html}
        <div class="card">
            <div class="section-title">📈 Cumulative P&amp;L Chart</div>
            <div style="margin-top:12px">{chart_html}</div>
            <p style="color:#555;font-size:11px;margin-top:6px">
                Red-shaded region = maximum drawdown period. Dashed line = zero.
            </p>
        </div>
        {breakdown_html}
        {monthly_html}
        {log_html}
        '''
        return self._html_shell(body)

    def _summary_section(self, s: dict) -> str:
        if not s:
            return ''
        pnl_color  = '#66bb6a' if s['total_pnl'] >= 0 else '#ff4444'
        wr_color   = '#66bb6a' if s['win_rate'] >= 55 else '#ffd54f' if s['win_rate'] >= 45 else '#ff4444'
        sh_color   = '#66bb6a' if s['sharpe'] >= 1.0  else '#ffd54f' if s['sharpe'] >= 0.5  else '#ff4444'
        dd_color   = '#ff4444'

        kelly_rec  = f"{s['kelly']:.1f}% of capital per trade"

        metrics = [
            ('Total P&L',      f"₹{s['total_pnl']:,.0f}", pnl_color),
            ('Trade Count',    str(s['trade_count']),      '#e0e0e0'),
            ('Win Rate',       f"{s['win_rate']}%",        wr_color),
            ('Avg P&L/Trade',  f"₹{s['avg_pnl']:,.0f}",   pnl_color),
            ('Max Single Loss',f"₹{s['max_loss']:,.0f}",   dd_color),
            ('Max Drawdown',   f"₹{s['max_drawdown']:,.0f}", dd_color),
            ('Sharpe (ann.)',  str(s['sharpe']),            sh_color),
        ]

        cards = ''.join(
            f'<div class="metric-card">'
            f'<div class="metric-label">{lbl}</div>'
            f'<div class="metric-value" style="color:{col}">{val}</div>'
            f'</div>'
            for lbl, val, col in metrics
        )

        return f'''
        <div class="card">
            <div class="section-title">📊 Summary</div>
            <div class="metrics-row">{cards}</div>
            <div class="kelly-box">
                🎲 <strong>Kelly Recommendation:</strong> Suggested position size:
                <span style="color:#4fc3f7;font-weight:700">{kelly_rec}</span>
            </div>
        </div>
        '''

    def _breakdown_section(self, df: pd.DataFrame) -> str:
        margins = {
            'SHORT_STRADDLE': 150000,
            'SHORT_STRANGLE': 140000,
            'BEAR_CALL_SPREAD': 40000,
            'IRON_CONDOR': 60000
        }

        rows_html = ''
        grouped = df.groupby(['signal_type', 'strategy_type'])
        for (sig, strat), grp in grouped:
            s = self._stats(grp['pnl_per_lot'].tolist())
            margin = margins.get(strat, 100000)
            roc = (s["total_pnl"] / margin) * 100 if margin > 0 else 0
            
            pc = '#66bb6a' if s['total_pnl'] >= 0 else '#ff4444'
            wc = '#66bb6a' if s['win_rate'] >= 55 else '#ffd54f' if s['win_rate'] >= 45 else '#ff4444'
            rc = '#66bb6a' if roc > 0 else '#ff4444'
            
            rows_html += (
                f'<tr>'
                f'<td>{sig}</td><td>{strat}</td>'
                f'<td>{s["trade_count"]}</td>'
                f'<td>₹{margin:,.0f}</td>'
                f'<td style="color:{wc}">{s["win_rate"]}%</td>'
                f'<td style="color:{pc}; font-weight: 600;">₹{s["total_pnl"]:,.0f}</td>'
                f'<td style="color:{rc}; font-weight: 600;">{roc:.1f}%</td>'
                f'<td style="color:{pc}">₹{s["avg_pnl"]:,.0f}</td>'
                f'<td style="color:#ff4444">₹{s["max_drawdown"]:,.0f}</td>'
                f'<td>{s["sharpe"]}</td>'
                f'<td style="color:#4fc3f7">{s["kelly"]}%</td>'
                f'</tr>'
            )
        return f'''
        <div class="card">
            <div class="section-title">🔍 Breakdown by Signal × Strategy</div>
            <table>
                <tr>
                    <th>Signal</th><th>Strategy</th><th>Trades</th>
                    <th>Est. Margin/Lot</th><th>Win%</th><th>Total P&L</th>
                    <th>RoC %</th><th>Avg P&L</th><th>Max DD</th>
                    <th>Sharpe</th><th>Kelly</th>
                </tr>
                {rows_html}
            </table>
        </div>
        '''

    def _monthly_section(self, df: pd.DataFrame) -> str:
        df2 = df.copy()
        df2['month'] = pd.to_datetime(df2['entry_date']).dt.to_period('M').astype(str)
        monthly = df2.groupby('month').agg(
            total_pnl  =('pnl_per_lot', 'sum'),
            trade_count=('pnl_per_lot', 'count'),
        ).reset_index()

        rows_html = ''
        for _, row in monthly.iterrows():
            pc = '#66bb6a' if row['total_pnl'] >= 0 else '#ff4444'
            rows_html += (
                f'<tr>'
                f'<td>{row["month"]}</td>'
                f'<td style="color:{pc}">₹{row["total_pnl"]:,.0f}</td>'
                f'<td>{row["trade_count"]}</td>'
                f'</tr>'
            )
        return f'''
        <div class="card">
            <div class="section-title">📅 Month-by-Month P&amp;L</div>
            <table>
                <tr><th>Month</th><th>Total P&amp;L</th><th>Trades</th></tr>
                {rows_html}
            </table>
        </div>
        '''

    def _tradelog_section(self, df: pd.DataFrame) -> str:
        rows_html = ''
        for _, row in df.iterrows():
            pnl    = row.get('pnl_per_lot', 0)
            pc     = '#66bb6a' if pnl >= 0 else '#ff4444'
            reason = row.get('exit_reason', '')
            reason_color = {
                'TARGET':    '#66bb6a',
                'EXPIRY':    '#4fc3f7',
                'STOP_LOSS': '#ff4444',
                'TIME_STOP': '#ffd54f',
            }.get(reason, '#888')

            bg_class = "table-row-credit" if pnl >= 0 else "table-row-debit"
            rows_html += (
                f'<tr class="{bg_class}">'
                f'<td>{row.get("entry_date","")}</td>'
                f'<td>{row.get("exit_date","")}</td>'
                f'<td>{row.get("signal_type","")}</td>'
                f'<td>{row.get("strategy_type","")}</td>'
                f'<td>₹{row.get("entry_credit",0):,.0f}</td>'
                f'<td style="color:{pc}; font-weight: 600;">₹{pnl:,.0f}</td>'
                f'<td style="color:{reason_color}">{reason}</td>'
                f'<td>{row.get("dte_at_entry","")}</td>'
                f'<td>{row.get("entry_spot","")}</td>'
                f'</tr>'
            )

        return f'''
        <div class="card">
            <div class="section-title">📋 Full Trade Log</div>
            <div style="overflow-x:auto">
            <table>
                <tr>
                    <th>Entry Date</th><th>Exit Date</th><th>Signal</th>
                    <th>Strategy</th><th>Entry Credit</th><th>P&amp;L/Lot</th>
                    <th>Exit Reason</th><th>DTE</th><th>Entry Spot</th>
                </tr>
                {rows_html}
            </table>
            </div>
        </div>
        '''

    @staticmethod
    def _html_shell(body: str) -> str:
        ts = datetime.now().strftime('%Y-%m-%d %H:%M')
        return f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>F-Intel Sell Signal Backtest Report</title>
<style>
  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    font-family: 'Segoe UI', Inter, system-ui, sans-serif;
    background: #0d0d1e;
    color: #e0e0e0;
    padding: 24px;
    font-size: 13px;
  }}
  h1 {{ color: #4fc3f7; font-size: 22px; margin-bottom: 4px; letter-spacing: 1px; }}
  .subtitle {{ color: #888; font-size: 12px; margin-bottom: 20px; }}
  .card {{
    background: #12122a;
    border: 1px solid #2a2a4a;
    border-radius: 10px;
    padding: 20px;
    margin-bottom: 20px;
  }}
  .section-title {{
    color: #4fc3f7;
    font-size: 13px;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 14px;
  }}
  .metrics-row {{
    display: flex; flex-wrap: wrap; gap: 14px;
  }}
  .metric-card {{
    background: linear-gradient(135deg, #1a1a35 0%, #0f0f23 100%);
    border: 1px solid #2a2a44;
    border-radius: 8px;
    padding: 16px 20px;
    min-width: 140px;
    flex: 1;
    box-shadow: 0 4px 6px rgba(0,0,0,0.2);
    transition: transform 0.2s;
  }}
  .metric-card:hover {{
    transform: translateY(-2px);
  }}
  .metric-label {{ color: #888; font-size: 11px; text-transform: uppercase; margin-bottom: 6px; }}
  .metric-value {{ font-size: 20px; font-weight: 700; }}
  .kelly-box {{
    margin-top: 16px;
    padding: 10px 14px;
    background: #0a1a2a;
    border-left: 3px solid #4fc3f7;
    border-radius: 4px;
    color: #ccc;
    font-size: 12px;
  }}
  table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 12px;
    margin-top: 4px;
  }}
  th {{
    color: #888;
    text-align: left;
    padding: 6px 10px;
    border-bottom: 1px solid #2a2a4a;
    white-space: nowrap;
  }}
  td {{
    padding: 6px 10px;
    border-bottom: 1px solid #1a1a35;
    white-space: nowrap;
  }}
  tr:hover td {{ background: #1a1a35; }}
  
  .table-row-credit {{ background-color: rgba(102, 187, 106, 0.05); }}
  .table-row-debit {{ background-color: rgba(255, 68, 68, 0.05); }}
  
</style>
</head>
<body>
<h1>⚡ F-Intel Sell Signal Backtest Report</h1>
<div class="subtitle">Generated: {ts} | NIFTY Options | Lot Size: 65 | Risk-Free: 7%</div>
{body}
</body>
</html>'''
