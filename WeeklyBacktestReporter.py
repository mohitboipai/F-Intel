"""
WeeklyBacktestReporter.py
=========================
Reads all tables from data/backtest_results.db and produces a
self-contained dark-themed HTML report at data/weekly_backtest_report.html.

Sections
--------
  1. Strategy comparison summary table
  2. Short Strangle shift analysis (frequency + trigger breakdown)
  3. Multi-line SVG cumulative P&L chart (4 strategies)
  4. Week-by-week position detail with expandable leg sub-tables
  5. Month-by-month breakdown
  6. Full leg transaction log (strangle only)

Usage
-----
    reporter = WeeklyBacktestReporter()
    path = reporter.generate()
"""

import os
import sys
import sqlite3
import json
import numpy as np
import pandas as pd
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

DB_PATH     = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           'data', 'backtest_results.db')
REPORT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           'data', 'weekly_backtest_report.html')

LOT_SIZE = 75


class WeeklyBacktestReporter:

    def __init__(self, db_path: str = DB_PATH, report_path: str = REPORT_PATH):
        self.db_path     = db_path
        self.report_path = report_path

    # ── load ─────────────────────────────────────────────────────────────────

    def _load(self, table: str) -> pd.DataFrame:
        if not os.path.exists(self.db_path):
            return pd.DataFrame()
        try:
            conn = sqlite3.connect(self.db_path)
            df   = pd.read_sql_query(f'SELECT * FROM {table}', conn)
            conn.close()
            return df
        except Exception:
            return pd.DataFrame()

    # ── generate ─────────────────────────────────────────────────────────────

    def generate(self) -> str:
        ps   = self._load('position_summaries')
        lt   = self._load('leg_transactions')
        std  = self._load('straddle_position_summaries')
        stlt = self._load('straddle_leg_transactions')
        ic   = self._load('condor_position_summaries')
        bcs  = self._load('callspread_position_summaries')

        ts   = datetime.now().strftime('%Y-%m-%d %H:%M')
        body = ''

        # Section 1 -- Strategy comparison
        body += self._section_comparison(ps, std, ic, bcs)
        # Section 2 -- Strangle shift analysis
        body += self._section_shift_analysis(ps, lt)
        # Section 3 -- Multi-line equity chart
        body += self._section_chart(ps, std, ic, bcs)
        # Section 4a -- Strangle week-by-week detail
        body += self._section_week_detail_generic(
            ps, lt,
            title='Short Strangle',
            id_prefix='ss',
            col_label='CE / PE Strike',
        )
        # Section 4b -- Straddle week-by-week detail
        body += self._section_week_detail_generic(
            std, stlt,
            title='Short Straddle',
            id_prefix='std',
            col_label='ATM Strike',
        )
        # Section 5 -- Monthly breakdown (Strangle + Straddle side-by-side)
        body += self._section_monthly_dual(ps, std)
        # Section 6 -- Full leg log (Strangle)
        body += self._section_leg_log(lt, title='Short Strangle')
        # Section 7 -- Full leg log (Straddle)
        body += self._section_leg_log(stlt, title='Short Straddle')

        html = self._shell(body, ts)
        os.makedirs(os.path.dirname(self.report_path), exist_ok=True)
        with open(self.report_path, 'w', encoding='utf-8') as f:
            f.write(html)
        return self.report_path

    # ── helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _stats(df: pd.DataFrame) -> dict:
        if df.empty:
            return {}
        pnls = df['final_pnl_rupees'].values
        wins  = pnls[pnls > 0]
        cum   = np.cumsum(pnls)
        peak  = np.maximum.accumulate(cum)
        dd    = cum - peak
        sharpe = (pnls.mean() / (pnls.std() + 1e-9)) * np.sqrt(52) if len(pnls) > 1 else 0
        wr    = len(wins) / len(pnls)
        avg_w = float(wins.mean()) if len(wins) > 0 else 0
        avg_l = float(abs(pnls[pnls <= 0].mean())) if any(pnls <= 0) else 1
        b     = avg_w / avg_l if avg_l > 0 else 0
        kelly = min(max((wr * b - (1 - wr)) / b if b > 0 else 0, 0), 0.25)
        return {
            'positions':   len(pnls),
            'total_pnl':   round(float(pnls.sum()), 0),
            'win_rate':    round(wr * 100, 1),
            'avg_pnl':     round(float(pnls.mean()), 0),
            'max_dd':      round(float(dd.min()), 0),
            'sharpe':      round(float(sharpe), 2),
            'kelly':       round(kelly * 100, 1),
            'avg_shifts':  round(float(df['shift_count'].mean()), 2) if 'shift_count' in df.columns else 0,
            'cumulative':  cum.tolist(),
        }

    # ── Section 1: Strategy comparison ────────────────────────────────────────

    def _section_comparison(self, ps, std, ic, bcs) -> str:
        rows = ''
        def row(name, df, highlight=False):
            s = self._stats(df)
            if not s:
                return f'<tr><td>{name}</td>' + '<td>--</td>' * 7 + '</tr>'
            pc = '#66bb6a' if s['total_pnl'] >= 0 else '#ef5350'
            wc = '#66bb6a' if s['win_rate'] >= 55 else '#ffd54f' if s['win_rate'] >= 45 else '#ef5350'
            bg = 'background:rgba(79,195,247,0.07);' if highlight else ''
            badge_col = '#66bb6a' if s['total_pnl'] >= 0 else '#ef5350'
            badge = (f'<span style="background:{badge_col};color:#000;'
                     f'font-size:10px;border-radius:4px;padding:2px 6px;font-weight:700;">'
                     f'{"+" if s["total_pnl"]>=0 else ""}Rs.{s["total_pnl"]:,.0f}</span>')
            return (
                f'<tr style="{bg}">'
                f'<td style="white-space:nowrap">{"&#9733; " if highlight else ""}{name}</td>'
                f'<td>{badge}</td>'
                f'<td>{s["positions"]}</td>'
                f'<td style="color:{wc}">{s["win_rate"]}%</td>'
                f'<td style="color:{pc}">Rs.{s["avg_pnl"]:,.0f}</td>'
                f'<td>{s["avg_shifts"]}</td>'
                f'<td style="color:#ef5350">Rs.{s["max_dd"]:,.0f}</td>'
                f'<td>{s["sharpe"]}</td>'
                f'<td style="color:#4fc3f7">{s["kelly"]}%</td>'
                f'</tr>'
            )

        rows  = row('Short Strangle', ps,  highlight=True)
        rows += row('Short Straddle', std)
        rows += row('Iron Condor',    ic)
        rows += row('Bear Call Spread', bcs)

        return f'''
        <div class="card">
            <div class="sec-title">&#128202; Strategy Comparison</div>
            <div style="overflow-x:auto">
            <table>
                <tr>
                    <th>Strategy</th><th>Total P&amp;L</th><th>Positions</th>
                    <th>Win %</th><th>Avg P&amp;L</th><th>Avg Shifts</th>
                    <th>Max DD</th><th>Sharpe</th><th>Kelly</th>
                </tr>
                {rows}
            </table>
            </div>
        </div>'''

    # ── Section 2: Shift analysis ──────────────────────────────────────────────

    def _section_shift_analysis(self, ps: pd.DataFrame, lt: pd.DataFrame) -> str:
        if ps.empty:
            return ''

        total = len(ps)

        def grp(n):
            sub = ps[ps['shift_count'] == n] if n < 2 else ps[ps['shift_count'] >= 2]
            cnt = len(sub)
            pct = cnt / total * 100 if total else 0
            avg = sub['final_pnl_rupees'].mean() if cnt > 0 else 0
            col = '#66bb6a' if avg >= 0 else '#ef5350'
            label = {0: 'Never adjusted', 1: 'Shifted once', 2: 'Shifted twice+'}[n]
            return (f'<div class="shift-row">'
                    f'<span style="color:#4fc3f7;font-weight:700">{cnt}</span> positions '
                    f'({pct:.0f}%) -- <em>{label}</em><br>'
                    f'Avg P&amp;L: <span style="color:{col};font-weight:700">Rs.{avg:,.0f}</span>'
                    f'</div>')

        left = grp(0) + grp(1) + grp(2)

        # Shift trigger breakdown from leg_transactions
        trigger_html = '<div style="color:#555;font-size:11px">No shift transactions found.</div>'
        if not lt.empty:
            shifts = lt[lt['time_type'] == 'OPEN'][
                lt[lt['time_type'] == 'OPEN']['shift_number'] > 0]
            if not shifts.empty:
                spot_ce = len(shifts[shifts['reason'].str.contains('SPOT_CE', na=False)])
                spot_pe = len(shifts[shifts['reason'].str.contains('SPOT_PE', na=False)])
                spot_bo = len(shifts[shifts['reason'].str.contains('SPOT_BOTH', na=False)])
                sig_sh  = len(shifts[shifts['reason'].str.contains('SIGNAL', na=False)])
                total_sh = spot_ce + spot_pe + spot_bo + sig_sh or 1
                def tr(label, cnt):
                    return (f'<div class="shift-row"><span style="color:#4fc3f7;font-weight:700">'
                            f'{cnt}</span> shifts -- <em>{label}</em> ({cnt/total_sh*100:.0f}%)</div>')
                trigger_html = (tr('CE delta breach (SPOT_CE)', spot_ce)
                              + tr('PE delta breach (SPOT_PE)', spot_pe)
                              + tr('Both legs breached (SPOT_BOTH)', spot_bo)
                              + tr('VRP signal flip (SIGNAL)', sig_sh))

        return f'''
        <div class="card">
            <div class="sec-title">&#128200; Short Strangle -- Shift Analysis</div>
            <div style="display:flex;gap:30px;flex-wrap:wrap">
                <div style="flex:1;min-width:220px">
                    <div style="color:#888;font-size:11px;text-transform:uppercase;
                                letter-spacing:1px;margin-bottom:10px">Shift Frequency</div>
                    {left}
                </div>
                <div style="flex:1;min-width:220px">
                    <div style="color:#888;font-size:11px;text-transform:uppercase;
                                letter-spacing:1px;margin-bottom:10px">Trigger Breakdown</div>
                    {trigger_html}
                </div>
            </div>
        </div>'''

    # ── Section 3: SVG multi-line chart ──────────────────────────────────────

    def _section_chart(self, ps, std, ic, bcs) -> str:
        W, H = 960, 320
        PAD_X, PAD_Y = 70, 30

        def cum_series(df):
            if df.empty:
                return []
            return np.cumsum(df.sort_values('entry_date')['final_pnl_rupees'].values).tolist()

        series = {
            'Short Strangle': (cum_series(ps),  '#4fc3f7', ''),
            'Short Straddle': (cum_series(std),  '#66bb6a', '6,3'),
            'Iron Condor':    (cum_series(ic),   '#ffd54f', '2,3'),
            'Bear Call':      (cum_series(bcs),  '#ff7043', '8,3,2,3'),
        }
        all_vals = [v for _, (cv, _, _) in series.items() for v in cv]
        if not all_vals:
            return '<div class="card"><p style="color:#555">No data to chart.</p></div>'

        lo, hi = min(all_vals), max(all_vals)
        span   = hi - lo or 1.0
        iw     = W - PAD_X * 2
        ih     = H - PAD_Y * 2

        def xp(i, n): return PAD_X + i / max(n - 1, 1) * iw
        def yp(v):    return PAD_Y + (1 - (v - lo) / span) * ih
        zero_y = yp(0)

        # Max drawdown shading for Strangle
        ps_cum = series['Short Strangle'][0]
        shading = ''
        if ps_cum:
            arr  = np.array(ps_cum)
            peak = np.maximum.accumulate(arr)
            dd   = arr - peak
            dd_e = int(np.argmin(dd))
            dd_s = int(np.argmax(arr[:dd_e + 1])) if dd_e > 0 else 0
            n    = len(arr)
            x0   = xp(dd_s, n); x1 = xp(dd_e, n)
            width_r = max(x1 - x0, 4)
            shading = (f'<rect x="{x0:.1f}" y="{PAD_Y}" width="{width_r:.1f}" '
                       f'height="{ih}" fill="rgba(255,68,68,0.07)" rx="2"/>'
                       f'<text x="{(x0+x1)/2:.1f}" y="{PAD_Y - 6}" fill="#ef5350" '
                       f'font-size="10" text-anchor="middle">max drawdown</text>')

        zero_line = (f'<line x1="{PAD_X}" y1="{zero_y:.1f}" '
                     f'x2="{W - PAD_X}" y2="{zero_y:.1f}" '
                     f'stroke="#333" stroke-dasharray="4,4" stroke-width="1"/>')

        lines = ''
        for label, (cv, color, dash) in series.items():
            if not cv:
                continue
            n   = len(cv)
            pts = ' '.join(f'{xp(i,n):.1f},{yp(v):.1f}' for i, v in enumerate(cv))
            da  = f'stroke-dasharray="{dash}"' if dash else ''
            lines += f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="2" {da}/>'
            # Label at right end
            lx = xp(n - 1, n) + 4
            ly = yp(cv[-1])
            lines += (f'<text x="{lx:.1f}" y="{ly + 4:.1f}" fill="{color}" '
                      f'font-size="10">{label}</text>')

        # Y-axis ticks
        y_labels = ''
        for frac in [0, 0.25, 0.5, 0.75, 1.0]:
            v = lo + frac * span
            y = yp(v)
            y_labels += (f'<line x1="{PAD_X-4}" y1="{y:.1f}" x2="{PAD_X}" y2="{y:.1f}" '
                         f'stroke="#444" stroke-width="1"/>'
                         f'<text x="{PAD_X-6}" y="{y+4:.1f}" fill="#888" '
                         f'font-size="10" text-anchor="end">Rs.{v:,.0f}</text>')

        svg = (f'<svg width="100%" viewBox="0 0 {W} {H}" '
               f'xmlns="http://www.w3.org/2000/svg" '
               f'style="background:linear-gradient(180deg,#12122a 0%,#0d0d1e 100%);'
               f'border-radius:8px;border:1px solid #2a2a44">'
               f'{shading}{zero_line}{y_labels}{lines}</svg>')

        return f'''
        <div class="card">
            <div class="sec-title">&#128200; Cumulative P&amp;L -- All Strategies</div>
            <div style="margin-top:12px">{svg}</div>
            <p style="color:#555;font-size:11px;margin-top:6px">
                Red-shaded = Short Strangle max drawdown period. Dashed = Straddle.
                Dotted = Condor. Dash-dot = Bear Call.
            </p>
        </div>'''

    # ── Section 4 generic: Week-by-week detail (any strategy) ────────────────

    def _section_week_detail_generic(
        self, ps: pd.DataFrame, lt: pd.DataFrame,
        title: str, id_prefix: str, col_label: str,
    ) -> str:
        if ps.empty:
            return ''

        ps_sorted = ps.sort_values('entry_date')
        best_pnl  = ps_sorted['final_pnl_rupees'].max()
        worst_pnl = ps_sorted['final_pnl_rupees'].min()

        rows_html = ''
        for _, row in ps_sorted.iterrows():
            pnl   = row['final_pnl_rupees']
            pc    = '#66bb6a' if pnl >= 0 else '#ef5350'
            bg    = 'rgba(102,187,106,0.06)' if pnl >= 0 else 'rgba(239,83,80,0.06)'
            bold  = 'font-weight:800' if pnl in (best_pnl, worst_pnl) else ''
            pid   = row['position_id']
            sc    = int(row.get('shift_count', 0))
            uid   = f'{id_prefix}_{pid}'.replace('-', '_')   # safe JS id

            # Always show leg sub-table (expandable) for all rows
            sub_html = ''
            if not lt.empty:
                sub_rows = lt[lt['position_id'] == pid]
                if not sub_rows.empty:
                    inner = ''
                    for _, tx in sub_rows.iterrows():
                        tc = '#66bb6a' if tx['cashflow_per_unit'] >= 0 else '#ef5350'
                        tt_col = {'OPEN': '#4fc3f7', 'CLOSE': '#ffd54f',
                                  'SETTLE': '#ab47bc'}.get(tx['time_type'], '#888')
                        inner += (
                            f'<tr>'
                            f'<td>{tx["date"]}</td>'
                            f'<td style="color:{tt_col}">{tx["time_type"]}</td>'
                            f'<td>{tx["option_type"]}</td>'
                            f'<td>{tx["strike"]:.0f}</td>'
                            f'<td>{tx["action"]}</td>'
                            f'<td>Rs.{tx["price_per_unit"]:.2f}</td>'
                            f'<td style="color:{tc}">Rs.{tx["cashflow_per_unit"]:.2f}</td>'
                            f'<td style="color:{tc}">Rs.{tx["cashflow_rupees"]:.0f}</td>'
                            f'<td style="color:#888;font-size:10px">{str(tx["reason"])[:40]}</td>'
                            f'</tr>'
                        )
                    sub_html = f'''
                    <tr id="{uid}" style="display:none">
                        <td colspan="9" style="padding:0 0 10px 20px">
                        <table style="font-size:11px;width:100%">
                            <tr><th>Date</th><th>Type</th><th>OType</th><th>Strike</th>
                                <th>Action</th><th>Price</th><th>CF/unit</th>
                                <th>CF Rs.</th><th>Reason</th></tr>
                            {inner}
                        </table>
                        </td>
                    </tr>'''

            # Toggle button always visible (even no-shift rows have entry/settle txns)
            toggle = (
                f'<button onclick="toggleSub(\'{uid}\')" '
                f'style="background:#1e2d3e;border:1px solid #4fc3f7;color:#4fc3f7;'
                f'border-radius:4px;padding:1px 6px;cursor:pointer;font-size:10px">'
                f'&#9776; {"" if sc == 0 else str(sc)+" shift"+"s" if sc > 1 else "1 shift"}'
                f'legs</button>'
            )

            ce_i   = row.get('ce_strike_initial', 0)
            pe_i   = row.get('pe_strike_initial', 0)
            credit = row.get('entry_credit_per_unit', 0)
            # For straddle: ce == pe, show once
            if ce_i == pe_i:
                strike_display = f'{ce_i:.0f}'
            else:
                strike_display = f'{ce_i:.0f} / {pe_i:.0f}'

            exit_r = row.get('exit_reason', '')
            exit_col = {'TARGET_HIT': '#66bb6a', 'EXPIRY': '#4fc3f7',
                        'SL_HIT': '#ef5350', 'SIGNAL_EXIT': '#ffd54f'}.get(exit_r, '#888')

            rows_html += (
                f'<tr style="background:{bg}">'
                f'<td style="{bold}">{pid}</td>'
                f'<td>{row["entry_date"]}</td>'
                f'<td>{row["expiry_date"]}</td>'
                f'<td>Rs.{row["entry_spot"]:,.0f}</td>'
                f'<td style="color:#aaa">{strike_display}</td>'
                f'<td>Rs.{credit:.2f}</td>'
                f'<td>{sc}</td>'
                f'<td><span style="color:{exit_col}">{exit_r}</span></td>'
                f'<td style="color:{pc};font-weight:700;{bold}">Rs.{pnl:,.0f}</td>'
                f'<td>{toggle}</td>'
                f'</tr>'
                + sub_html
            )

        color = '#4fc3f7' if 'Strangle' in title else '#66bb6a'
        return f'''
        <div class="card">
            <div class="sec-title" style="color:{color}">&#128203; Week-by-Week: {title}</div>
            <div style="overflow-x:auto">
            <table>
                <tr>
                    <th>Week</th><th>Entry</th><th>Expiry</th><th>Spot</th>
                    <th>{col_label}</th><th>Credit/unit</th>
                    <th>Shifts</th><th>Exit</th><th>P&amp;L</th><th>Legs</th>
                </tr>
                {rows_html}
            </table>
            </div>
        </div>'''

    # Keep old name as alias for backward compat
    def _section_week_detail(self, ps, lt):
        return self._section_week_detail_generic(ps, lt, 'Short Strangle', 'ss', 'CE / PE Strike')

    # ── Section 5: Monthly dual breakdown (Strangle + Straddle) ─────────────

    def _section_monthly_dual(self, ps: pd.DataFrame, std: pd.DataFrame) -> str:
        def monthly_table(df, label, color):
            if df.empty:
                return f'<p style="color:#555">No {label} data.</p>'
            d = df.copy()
            d['month'] = pd.to_datetime(d['entry_date']).dt.to_period('M').astype(str)
            m = d.groupby('month').agg(
                positions=('final_pnl_rupees', 'count'),
                total_shifts=('shift_count', 'sum'),
                wins=('final_pnl_rupees', lambda x: (x > 0).sum()),
                total_pnl=('final_pnl_rupees', 'sum'),
            ).reset_index()
            m['win_rate'] = (m['wins'] / m['positions'] * 100).round(1)
            best  = m['total_pnl'].max()
            worst = m['total_pnl'].min()
            rows = ''
            for _, r in m.iterrows():
                pc   = '#66bb6a' if r['total_pnl'] >= 0 else '#ef5350'
                bold = 'font-weight:800' if r['total_pnl'] in (best, worst) else ''
                rows += (f'<tr>'
                         f'<td style="{bold}">{r["month"]}</td>'
                         f'<td>{r["positions"]}</td>'
                         f'<td>{r["total_shifts"]}</td>'
                         f'<td>{r["win_rate"]}%</td>'
                         f'<td style="color:{pc};{bold}">Rs.{r["total_pnl"]:,.0f}</td>'
                         f'</tr>')
            return f'''
            <div style="flex:1;min-width:320px">
                <div style="color:{color};font-size:11px;font-weight:700;letter-spacing:1px;
                            text-transform:uppercase;margin-bottom:8px">{label}</div>
                <table>
                    <tr><th>Month</th><th>Pos</th><th>Shifts</th>
                        <th>Win%</th><th>P&amp;L</th></tr>
                    {rows}
                </table>
            </div>'''

        left  = monthly_table(ps,  'Short Strangle', '#4fc3f7')
        right = monthly_table(std, 'Short Straddle', '#66bb6a')

        return f'''
        <div class="card">
            <div class="sec-title">&#128197; Month-by-Month Breakdown</div>
            <div style="display:flex;gap:30px;flex-wrap:wrap">
                {left}
                {right}
            </div>
        </div>'''

    # Keep old name
    def _section_monthly(self, ps):
        return self._section_monthly_dual(ps, pd.DataFrame())

    # ── Section 6/7: Full leg log (any strategy) ─────────────────────────────

    def _section_leg_log(self, lt: pd.DataFrame, title: str = 'Short Strangle') -> str:
        if lt.empty:
            return ''
        color = '#4fc3f7' if 'Strangle' in title else '#66bb6a'
        rows = ''
        for _, tx in lt.sort_values(['position_id', 'date']).iterrows():
            cc = '#66bb6a' if tx['cashflow_per_unit'] >= 0 else '#ef5350'
            tt_col = {'OPEN': '#4fc3f7', 'CLOSE': '#ffd54f',
                      'SETTLE': '#ab47bc'}.get(tx['time_type'], '#888')
            rows += (f'<tr>'
                     f'<td>{tx["position_id"]}</td>'
                     f'<td>{tx["date"]}</td>'
                     f'<td style="color:{tt_col}">{tx["time_type"]}</td>'
                     f'<td>{tx["option_type"]}</td>'
                     f'<td>{tx["strike"]:.0f}</td>'
                     f'<td>{tx["action"]}</td>'
                     f'<td>Rs.{tx["price_per_unit"]:.2f}</td>'
                     f'<td style="color:{cc}">Rs.{tx["cashflow_per_unit"]:.2f}</td>'
                     f'<td style="color:{cc}">Rs.{tx["cashflow_rupees"]:.0f}</td>'
                     f'<td>{int(tx["shift_number"])}</td>'
                     f'<td style="color:#888;font-size:10px">{str(tx["reason"])[:50]}</td>'
                     f'</tr>')

        return f'''
        <div class="card">
            <div class="sec-title" style="color:{color}">&#128204; Full Leg Log — {title}</div>
            <div style="overflow-x:auto">
            <table style="font-size:11px">
                <tr>
                    <th>Position</th><th>Date</th><th>Type</th><th>OType</th>
                    <th>Strike</th><th>Action</th><th>Price</th>
                    <th>CF/unit</th><th>CF Rs.</th><th>Shift#</th><th>Reason</th>
                </tr>
                {rows}
            </table>
            </div>
        </div>'''

    # ── HTML shell ────────────────────────────────────────────────────────────

    @staticmethod
    def _shell(body: str, ts: str) -> str:
        return f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>F-Intel Weekly Dynamic Backtest Report</title>
<style>
  *,*::before,*::after{{box-sizing:border-box;margin:0;padding:0}}
  body{{font-family:'Segoe UI',Inter,system-ui,sans-serif;
        background:#0d0d1e;color:#e0e0e0;padding:24px;font-size:13px}}
  h1{{color:#4fc3f7;font-size:22px;margin-bottom:4px;letter-spacing:1px}}
  .sub{{color:#666;font-size:11px;margin-bottom:20px}}
  .card{{background:#12122a;border:1px solid #2a2a4a;border-radius:10px;
          padding:20px;margin-bottom:20px}}
  .sec-title{{color:#4fc3f7;font-size:12px;font-weight:700;letter-spacing:2px;
               text-transform:uppercase;margin-bottom:14px}}
  .shift-row{{margin-bottom:10px;padding:8px 12px;background:#0f0f23;
               border-left:3px solid #4fc3f7;border-radius:4px;}}
  table{{width:100%;border-collapse:collapse;font-size:12px;margin-top:4px}}
  th{{color:#888;text-align:left;padding:6px 10px;
      border-bottom:1px solid #2a2a4a;white-space:nowrap}}
  td{{padding:5px 10px;border-bottom:1px solid #1a1a35;white-space:nowrap}}
  tr:hover td{{background:#1a1a35}}
</style>
<script>
  function toggleSub(pid){{
    var el = document.getElementById('sub_' + pid);
    if(el) el.style.display = el.style.display === 'none' ? 'table-row' : 'none';
  }}
</script>
</head>
<body>
<h1>&#9889; F-Intel Weekly Dynamic Backtest Report</h1>
<div class="sub">Generated: {ts} | NIFTY Options | Lot Size: {LOT_SIZE} | Risk-Free: 7%</div>
{body}
</body>
</html>'''
