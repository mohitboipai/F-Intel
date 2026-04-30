import sys
import os
import time
import json
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np # Math support for arrays

# Add current directory to path to find local modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

from FyersAuth import FyersAuthenticator
from OptionAnalytics import OptionAnalytics
from SignalMemory import SignalMemory
from SharedDataCache import SharedDataCache
from NiftyHestonMC import HestonMath
from DataClient import DataHubClient
from RealizedVolEngine import RealizedVolEngine
from ConfluenceEngine import ConfluenceEngine
from OptionBuyerEngine import OptionBuyerEngine

# ────────────────────────────────────────────────────────
# NOTE: The old local StrategyEngine class has been retired.
# Strategy generation is now performed exclusively by
# SmartStrategyGenerator in StrategyEngine.py (BSM-backed,
# greeks-aware, POP-scored). Imported lazily inside the
# dashboard loop at _create_unified_dashboard().
# ────────────────────────────────────────────────────────


# ────────────────────────────────────────────────────────
#  GLOBAL UI CONSTANTS
# ────────────────────────────────────────────────────────
DARK_BG = '#0f0f19'
CARD_BG = '#1a1a2e'
ACCENT = '#4fc3f7'
RED = '#ff4444'
GREEN = '#66bb6a'
WHITE = '#ffffff'
MUTED = '#8888aa'

# ────────────────────────────────────────────────────────
#  LOCAL API TRACKING SERVER
# ────────────────────────────────────────────────────────
from http.server import HTTPServer, BaseHTTPRequestHandler
import threading
from urllib.parse import urlparse, parse_qs
from StrategyManager import StrategyManager
from TickDatabase import IntradayTickDB

_strategy_manager = StrategyManager()
_tick_db = IntradayTickDB()

def generate_tracked_html(tracked, df_chain):
    """
    Computes P&L for a list of strategies against a specific option chain DataFrame
    and returns the formatted HTML table.
    """
    if not tracked:
        return '<div style="color:#666;font-size:12px;text-align:center;padding:20px;">No strategies currently being tracked. Click "TEST LIVE" below to add one.</div>'
        
    # Convert chain to fast lookup dict: (type, strike) -> price
    chain_prices = {}
    if not df_chain.empty:
        for _, r in df_chain.iterrows():
            chain_prices[(r['type'], r['strike'])] = r['price']
            
    _rows = []
    _total_pnl = 0
    for t in tracked:
        current_val = 0
        entry_val = t.get('premium', 0)
        is_credit = t.get('type') == 'CREDIT'
        
        # Re-price legs
        for leg in t.get('legs', []):
            live_p = chain_prices.get((leg['type'], leg['strike']), leg['price'])
            mult = 1 if leg['action'] == 'SELL' else -1
            current_val += live_p * mult
        
        # P&L Calculation
        if is_credit:  
            pnl = entry_val - current_val
        else:
            pnl = current_val - abs(entry_val)
            
        pnl_pct = (pnl / max(1, abs(entry_val))) * 100
        _total_pnl += pnl
        
        c_color = GREEN if pnl > 0 else RED if pnl < 0 else WHITE
        _rows.append(f'''
        <tr style="border-bottom:1px solid #2a2a4a;">
            <td style="padding:10px 8px;text-align:left;">
                <div style="font-weight:700;color:{ACCENT};font-size:13px;">{t['name']}</div>
                <div style="color:{MUTED};font-size:10px;">{t.get('tracked_at','')}</div>
            </td>
            <td style="padding:10px 8px;text-align:right;color:{WHITE};font-weight:600;">₹{abs(entry_val):.1f}</td>
            <td style="padding:10px 8px;text-align:right;color:{WHITE};font-weight:600;">₹{abs(current_val):.1f}</td>
            <td style="padding:10px 8px;text-align:right;color:{c_color};font-weight:700;">₹{pnl:.2f} <br><span style="font-size:10px;">({pnl_pct:+.1f}%)</span></td>
            <td style="padding:10px 8px;text-align:center;"><button onclick="deleteStrategy('{t.get('id')}')" style="background:#442222;color:#ff8888;border:1px solid #662222;border-radius:4px;cursor:pointer;padding:2px 8px;font-size:10px;">X</button></td>
        </tr>''')
        
    tracked_table = f'<table style="width:100%;border-collapse:collapse;"><tr style="color:{MUTED};font-size:11px;text-transform:uppercase;border-bottom:1px solid #333;"><th style="text-align:left;padding:8px;">Strategy</th><th style="text-align:right;padding:8px;">Entry Prem</th><th style="text-align:right;padding:8px;">Live Prem</th><th style="text-align:right;padding:8px;">Live P&L</th><th></th></tr>{"".join(_rows)}</table>'
    t_color = GREEN if _total_pnl > 0 else RED if _total_pnl < 0 else WHITE
    
    return f'''
    <div style="display:flex;justify-content:space-between;margin-bottom:12px;align-items:baseline;">
        <div style="color:{WHITE};font-size:12px;font-weight:600;">ACTIVE VIRTUAL TRADES ({len(tracked)})</div>
        <div style="color:{t_color};font-size:16px;font-weight:700;">Net P&L: ₹{_total_pnl:+.1f}</div>
    </div>
    {tracked_table}
    '''

class StrategyAPIHandler(BaseHTTPRequestHandler):
    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST, GET, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == '/strategy_pnl_at':
            try:
                query = parse_qs(parsed.query)
                target_time = query.get('time', [''])[0]
                
                # Default to now if not provided
                if not target_time:
                    target_time = datetime.now().strftime('%H:%M')
                    
                # Fetch history
                hist_chain = _tick_db.get_chain_at_time("NSE:NIFTY50-INDEX", target_time)
                tracked = _strategy_manager.get_all_active_strategies()
                
                html_res = generate_tracked_html(tracked, hist_chain)
                
                self.send_response(200)
                self.send_header('Access-Control-Allow-Origin', '*')
                self.send_header('Content-type', 'text/html')
                self.end_headers()
                self.wfile.write(html_res.encode('utf-8'))
            except Exception as e:
                self.send_response(500)
                self.end_headers()
                self.wfile.write(str(e).encode('utf-8'))

    def do_POST(self):
        if self.path == '/track_strategy':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            try:
                strategy_data = json.loads(post_data.decode('utf-8'))
                
                # Use robust SQLite manager defined above
                _strategy_manager.track_strategy(strategy_data)

                self.send_response(200)
                self.send_header('Access-Control-Allow-Origin', '*')
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({'status': 'success'}).encode('utf-8'))
            except Exception as e:
                self.send_response(500)
                self.end_headers()
                self.wfile.write(json.dumps({'status': 'error', 'msg': str(e)}).encode('utf-8'))
                
        elif self.path == '/delete_strategy':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            try:
                req = json.loads(post_data.decode('utf-8'))
                _strategy_manager.delete_strategy(req.get('id'))
                
                self.send_response(200)
                self.send_header('Access-Control-Allow-Origin', '*')
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({'status': 'success'}).encode('utf-8'))
            except Exception as e:
                self.send_response(500)
                self.end_headers()

def run_strategy_server():
    server_address = ('localhost', 8081)
    httpd = HTTPServer(server_address, StrategyAPIHandler)
    httpd.serve_forever()

class VolatilityAnalyzer:
    def __init__(self):
        # Initialize Data Hub Client
        self.data_hub = DataHubClient()
        
        # Start the background API listener server for UI tracking controls
        self.api_thread = threading.Thread(target=run_strategy_server, daemon=True)
        self.api_thread.start()
        
        self.fyers = self._authenticate()
        self.analytics = OptionAnalytics()
        self.symbol = "NSE:NIFTY50-INDEX"
        self.spot_price = 0
        self.expiry_date = None # Current selected expiry

        # We need this for parse_and_filter compatibility, though we might override it
        self.selected_strikes = []

        # Unified expiry configuration (set once at startup)
        self._near_expiry = None   # Weekly expiry — used by all modules
        self._far_expiry = None    # Monthly expiry — used by IV surface + term structure
        self._expiries = []        # Full list [near, far, ...extras] for IV surface

        # Shared context / memory (persisted across module sessions)
        self.memory = SignalMemory()
        self.cache  = SharedDataCache(self.fyers, symbol=self.symbol)
        
        self.regime_engine = RealizedVolEngine()
        self.confluence_engine = ConfluenceEngine()
        self.buyer_engine = OptionBuyerEngine()

    def _authenticate(self):
        print("Authenticating with Fyers...")
        APP_ID = "QUTT4YYMIG-100"
        SECRET_ID = "ZG0WN2NL1B"
        REDIRECT_URI = "http://127.0.0.1:3000/callback"
        
        auth = FyersAuthenticator(APP_ID, SECRET_ID, REDIRECT_URI)
        fyers = auth.get_fyers_instance()
        if not fyers:
            print("Authentication Failed!")
            sys.exit(1)
        print("Authentication Successful.")
        return fyers

    def get_spot_price(self, verbose=True):
        # 1. Try Data Hub first
        hub_data = self.data_hub.get_latest_data()
        if hub_data and hub_data.get('spot', 0) > 0:
            self.spot_price = hub_data['spot']
            return {'price': self.spot_price}

        # 2. Fallback to direct Fyers (legacy)
        data = {"symbols": self.symbol}
        try:
            response = self.fyers.quotes(data=data)
            
            # Auto Re-Auth if Token Expired
            if response.get('code') == -15 or "token" in response.get('message', '').lower():
                print("Token expired during spot check. Re-authenticating...")
                self.fyers = self._authenticate()
                response = self.fyers.quotes(data=data)
            
            if response.get('s') == "ok":
                d = response['d'][0]['v']
                self.spot_price = d.get('lp', 0)
                return {
                    'price': self.spot_price,
                    'high': d.get('high_price', self.spot_price),
                    'low': d.get('low_price', self.spot_price),
                    'open': d.get('open_price', self.spot_price)
                }
            else:
                print(f"Error response from quotes: {response}")
        except Exception as e:
            print(f"Error fetching spot: {e}")
        return {'price': 0}

    def get_option_chain_data(self):
        # Strictly rely on Data Hub. If it's hitting rate limits (429), hitting Fyers
        # again directly here only punishes the limit further.
        hub_data = self.data_hub.get_latest_data()
        if hub_data and hub_data.get('chain'):
             return hub_data['chain']
        return None


    def parse_and_filter(self, data):
        if not data:
            return pd.DataFrame()
            
        options = data.get('optionsChain', [])
        records = []
        
        for item in options:
            strike = item.get('strike_price')
            
            # Filter by strike if selected AND if we are enforcing selection
            if self.selected_strikes:
                 match = False
                 for s in self.selected_strikes:
                     if abs(float(strike) - float(s)) < 0.1:
                         match = True
                         break
                 if not match:
                     continue
                
            raw_type = item.get('option_type', 'PE')
            if raw_type in ["CALL", "CE"]:
                option_type = "CE"
            else:
                option_type = "PE"
            
            records.append({
                'strike': float(strike),
                'type': option_type,
                'price': float(item.get('ltp', 0) or 0),
                'iv': float(item.get('iv', 0) or 0),
                'delta': float(item.get('delta', 0) or 0),
                'gamma': float(item.get('gamma', 0) or 0),
                'vega': float(item.get('vega', 0) or 0),
                'theta': float(item.get('theta', 0) or 0),
                'oi': int(item.get('oi', 0) or 0)
            })
            
        df = pd.DataFrame(records)
        return df

    # ================================================================
    # REUSABLE HELPERS (DRY — replaces 5 duplicated blocks)
    # ================================================================

    def _ensure_iv(self, row_iv, price, strike, T, option_type):
        """Recalculate IV if API value is suspect."""
        if 0.5 < row_iv < 200:
            return row_iv
        if price > 0.5:
            try:
                calc_iv = self.analytics.implied_volatility(price, self.spot_price, strike, T, 0.10, option_type)
                if 0 < calc_iv < 200:
                    return calc_iv
            except Exception:
                pass
        return row_iv if row_iv > 0 else 0

    def _filter_strikes(self, df, spot, range_pct=0.04):
        """Filter to strikes within range_pct of spot."""
        return df[(df['strike'] > spot * (1 - range_pct)) & (df['strike'] < spot * (1 + range_pct))]

    # ================================================================
    # SIGNAL SCORING ENGINE (replaces static if/elif decision tree)
    # ================================================================

    def _score_term(self, term_spread):
        """Term structure score: 0=backwardation(buy vol) → 100=contango(sell vol)."""
        # Map -5..+5 spread to 0..100
        return max(0, min(100, 50 + term_spread * 10))

    def _score_skew(self, skew_ratio):
        """Skew score: 0=steep put skew(fear) → 100=flat/call skew(comfort)."""
        # skew_ratio ~1.0 is flat (50), >1.2 is steep put (0), <0.8 is call skew (100)
        return max(0, min(100, 50 - (skew_ratio - 1.0) * 250))

    def _score_vrp(self, vrp):
        """VRP score: 0=negative VRP(buy vol) → 100=positive VRP(sell vol)."""
        # VRP range typically -5..+10
        return max(0, min(100, 50 + vrp * 8))

    def _score_regime(self, regime):
        """Regime score: directional bias from regime state."""
        regime_scores = {
            'COMPRESSION': 30,    # Caution — breakout ahead
            'EXPANSION': 40,      # Momentum — don't sell vol into trend
            'MEAN REVERSION': 75, # Vol collapsing — sell premium
            'NORMAL': 50          # Neutral
        }
        return regime_scores.get(regime, 50)

    def _score_percentile(self, hv_percentile):
        """Percentile score: high percentile → mean reversion sell bias."""
        return min(100, hv_percentile)

    def _score_signal(self, metrics):
        """Score each signal dimension 0-100, produce weighted composite.
        Only scores dimensions that have real data — skips defaults."""
        # Base weights for each dimension
        all_weights = {
            'term_structure': 0.25,
            'skew':           0.30,
            'vrp':            0.25,
            'regime':         0.20
        }
        
        scores = {}
        active_weights = {}
        
        # Only score dimensions that have real data
        if 'term_spread' in metrics and metrics['term_spread'] is not None:
            scores['term_structure'] = self._score_term(metrics['term_spread'])
            active_weights['term_structure'] = all_weights['term_structure']
        
        if 'skew_ratio' in metrics and metrics['skew_ratio'] is not None:
            scores['skew'] = self._score_skew(metrics['skew_ratio'])
            active_weights['skew'] = all_weights['skew']
        
        if 'vrp' in metrics and metrics['vrp'] is not None:
            scores['vrp'] = self._score_vrp(metrics['vrp'])
            active_weights['vrp'] = all_weights['vrp']
        
        if 'regime' in metrics and metrics['regime'] is not None:
            scores['regime'] = self._score_regime(metrics['regime'])
            active_weights['regime'] = all_weights['regime']
        
        # Normalize weights so active dimensions sum to 1.0
        if not active_weights:
            return {'scores': {}, 'composite': 50, 'action': 'NO DATA', 'strategy': 'Insufficient data', 'confidence': 0}
        
        weight_sum = sum(active_weights.values())
        normalized = {k: v / weight_sum for k, v in active_weights.items()}
        composite = sum(scores[k] * normalized[k] for k in scores)

        if composite > 65:
            action = "SELL VOL"
            strategy = "Iron Condors / Credit Spreads / Short Straddles"
        elif composite > 55:
            action = "LEAN SELL"
            strategy = "Covered Calls / Put Spreads"
        elif composite < 35:
            action = "BUY VOL"
            strategy = "Long Straddles / Debit Spreads"
        elif composite < 45:
            action = "LEAN BUY"
            strategy = "Calendar Spreads / Cheap Wings"
        else:
            action = "NEUTRAL"
            strategy = "Wait for clearer signal"

        confidence = abs(composite - 50) / 50

        # ── Alert dispatch ──────────────────────────────────────────────────────
        try:
            from AlertDispatcher import fire as _ad_fire
            if composite > 65:
                _ad_fire("VolatilityAnalyzer", "WARNING",
                         f"SELL VOL signal — composite {composite:.0f}",
                         f"Action: {action} | Strategy: {strategy}")
            elif composite < 35:
                _ad_fire("VolatilityAnalyzer", "WARNING",
                         f"BUY VOL signal — composite {composite:.0f}",
                         f"Action: {action}")
        except Exception:
            pass

        return {'scores': scores, 'composite': composite, 'action': action, 'strategy': strategy, 'confidence': confidence}

    def _format_signal_report(self, spot, iv, hv, signal_result, regime, iv_velocity_5d=0, iv_accel=0, vrp=0):
        """Print a rich formatted signal summary to console."""
        scores = signal_result['scores']
        comp = signal_result['composite']
        conf = signal_result['confidence']

        print("\n" + "═" * 60)
        print("  VOLATILITY INTELLIGENCE SUMMARY")
        print("═" * 60)
        print(f"  SPOT: {spot:,.0f}  |  ATM IV: {iv:.2f}%  |  20d HV: {hv:.2f}%")
        print(f"  VRP: {vrp:+.2f}%  |  Regime: {regime}")
        print()
        print("  SIGNAL SCORES (0=Buy Vol → 100=Sell Vol):")
        labels = {
            'term_structure': 'Term Structure',
            'skew':           'Skew          ',
            'vrp':            'VRP           ',
            'regime':         'Regime        '
        }
        for k, label in labels.items():
            if k not in scores:
                continue
            bar_len = int(scores[k] / 5)
            bar = "█" * bar_len + "░" * (20 - bar_len)
            print(f"    {label} {bar} {scores[k]:.0f}")
        print()
        print(f"  ▸ COMPOSITE: {comp:.0f} / 100")
        print(f"  ▸ ACTION: {signal_result['action']} (confidence: {conf:.0%})")
        print(f"  ▸ STRATEGY: {signal_result['strategy']}")
        if iv_velocity_5d != 0:
            accel_str = "accelerating" if iv_accel > 0.5 else "decelerating" if iv_accel < -0.5 else "steady"
            print(f"  ▸ Vol Velocity: {iv_velocity_5d:+.2f}%/5d ({accel_str})")
        print("═" * 60)

    def iv_surface_analysis(self):
        """Single entry point — launches the real-time IV Surface Intelligence Dashboard."""
        print("\n--- IV Surface Intelligence Dashboard ---")
        print("Fetching Spot Price...")
        self.get_spot_price()
        if self.spot_price <= 0:
            print("Error: Could not fetch Spot Price.")
            return

        expiries = self._expiries
        if len(expiries) < 2:
            print("Need at least 2 expiries for the surface. Please set expiries first.")
            return

        print(f"Symbol: {self.symbol}  Spot: {self.spot_price}")
        print(f"Using expiries: {', '.join(expiries)}")
        self._create_iv_dashboard(expiries)

    # ================================================================
    # IV DASHBOARD — DATA FETCHER
    # ================================================================
    def _iv_dashboard_fetch(self, expiries):
        """Shared data fetcher for IV surface methods. Returns list of point dicts."""
        points = []
        backup_strikes = self.selected_strikes
        self.selected_strikes = []

        try:
            self.get_spot_price()
            for exp in expiries:
                old_exp = self.expiry_date
                self.expiry_date = exp
                data = self.get_option_chain_data()
                df = self.parse_and_filter(data)
                self.expiry_date = old_exp

                if df.empty: continue

                T = self.analytics.get_time_to_expiry(exp)
                if T < 0.001: T = 0.001

                for _, row in df.iterrows():
                    strike = row['strike']
                    o_type = row['type']

                    if abs(strike - self.spot_price) > 1000: continue

                    iv = self._ensure_iv(row['iv'], row['price'], strike, T, o_type)

                    if iv > 0:
                        points.append({
                            'strike': strike, 'T': T, 'iv': iv,
                            'type': o_type, 'expiry': exp,
                            'days': round(T * 365, 1)
                        })
        finally:
            self.selected_strikes = backup_strikes

        return points

    # ================================================================
    # IV DASHBOARD — SPOT MOVEMENT PREDICTOR
    # ================================================================
    def _iv_predict_spot(self, points, expiries):
        """Analyze IV surface data to predict spot movement direction and magnitude."""
        spot = self.spot_price
        if not points or spot <= 0:
            return {
                'direction': 'NO DATA', 'confidence': 0, 'expected_move': 0,
                'expected_move_pct': 0, 'skew_signal': 'N/A', 'skew_ratio': 1.0,
                'term_signal': 'N/A', 'term_spread': 0, 'put_iv_avg': 0,
                'call_iv_avg': 0, 'atm_iv': 0, 'anomalous_strikes': [],
                'action': 'WAIT', 'strategy': 'Insufficient data',
                'near_atm': 0, 'far_atm': 0
            }

        # Group by expiry
        exp_data = {}
        for p in points:
            exp = p['expiry']
            if exp not in exp_data: exp_data[exp] = []
            exp_data[exp].append(p)

        # ATM IV per expiry
        def get_atm_iv(data):
            if not data: return 0
            dists = [abs(x['strike'] - spot) for x in data]
            return data[np.argmin(dists)]['iv']

        atm_ivs = {exp: get_atm_iv(exp_data.get(exp, [])) for exp in expiries}
        near_exp, far_exp = expiries[0], expiries[-1]
        near_atm = atm_ivs.get(near_exp, 0)
        far_atm = atm_ivs.get(far_exp, 0)

        # 1. TERM STRUCTURE
        term_spread = far_atm - near_atm
        if term_spread < -2: term_signal = "⚠️ BACKWARDATION — Big move imminent"
        elif term_spread < -1: term_signal = "🔶 Mild backwardation — Caution"
        elif term_spread > 1: term_signal = "✅ Contango — Normal, calm"
        else: term_signal = "➖ Flat"

        # 2. SKEW ANALYSIS (near-term)
        near_data = exp_data.get(near_exp, [])
        otm_puts = [p for p in near_data if p['strike'] < spot * 0.97 and p['type'] == 'PE']
        otm_calls = [p for p in near_data if p['strike'] > spot * 1.03 and p['type'] == 'CE']
        atm_range = [p for p in near_data if abs(p['strike'] - spot) < spot * 0.01]

        put_iv_avg = np.mean([p['iv'] for p in otm_puts]) if otm_puts else 0
        call_iv_avg = np.mean([p['iv'] for p in otm_calls]) if otm_calls else 0
        atm_iv = near_atm if near_atm > 0 else (np.mean([p['iv'] for p in atm_range]) if atm_range else 0)

        skew_ratio = put_iv_avg / atm_iv if atm_iv > 0 else 1.0
        if skew_ratio > 1.25: skew_signal = "🔴 STEEP PUT SKEW — Heavy downside hedging"
        elif skew_ratio > 1.10: skew_signal = "🟠 Elevated put skew — Mild fear"
        elif skew_ratio < 0.85: skew_signal = "🔵 CALL SKEW — Upside demand"
        elif skew_ratio < 0.95: skew_signal = "🟢 Mild call skew — Bullish tilt"
        else: skew_signal = "⚪ Balanced — No directional tilt"

        # 3. EXPECTED MOVE from ATM IV
        if atm_iv > 0:
            daily_vol = atm_iv / 100 * np.sqrt(1/365)
            expected_move = spot * daily_vol
            expected_move_pct = daily_vol * 100
        else:
            expected_move, expected_move_pct = 0, 0

        # 4. ANOMALOUS STRIKES (IV >> average = smart money positioning)
        all_ivs = [p['iv'] for p in points if p['iv'] > 0]
        if all_ivs:
            iv_mean = np.mean(all_ivs)
            iv_std = np.std(all_ivs)
            threshold = iv_mean + 1.5 * iv_std
            anomalous = [p for p in points if p['iv'] > threshold]
            anomalous_strikes = [
                {'strike': p['strike'], 'iv': p['iv'], 'type': p['type'],
                 'expiry': p['expiry'], 'excess': p['iv'] - iv_mean}
                for p in sorted(anomalous, key=lambda x: x['iv'], reverse=True)[:5]
            ]
        else:
            anomalous_strikes = []

        # 5. DIRECTION & CONFIDENCE
        score = 50  # neutral
        # Skew contribution (puts bid = bearish)
        if skew_ratio > 1.15: score -= (skew_ratio - 1.0) * 60
        elif skew_ratio < 0.9: score += (1.0 - skew_ratio) * 60
        # Call vs Put IV spread
        if put_iv_avg > 0 and call_iv_avg > 0:
            pc_skew = put_iv_avg - call_iv_avg
            score -= pc_skew * 3  # high put IV = bearish
        # Term structure
        if term_spread < -2: score -= 10  # backwardation = fear

        # Skew change tracking
        if not hasattr(self, '_skew_history'):
            self._skew_history = []
        self._skew_history.append({
            'time': datetime.now(), 'skew_ratio': skew_ratio,
            'term_spread': term_spread, 'atm_iv': atm_iv,
            'put_iv': put_iv_avg, 'call_iv': call_iv_avg, 'score': score
        })
        # Keep last 30 readings (5 minutes worth at 10s intervals)
        self._skew_history = self._skew_history[-30:]

        # Skew velocity (if we have history)
        skew_delta = 0
        if len(self._skew_history) >= 3:
            old_skew = self._skew_history[-3]['skew_ratio']
            skew_delta = skew_ratio - old_skew
            score -= skew_delta * 100  # rapidly steepening = bearish

        confidence = min(abs(score - 50) / 50, 1.0)
        if score > 60:
            direction = "BULLISH"
            action = "BUY CALLS / SELL PUTS"
        elif score > 55:
            direction = "LEAN BULLISH"
            action = "Bull Spreads / Sell Put Spreads"
        elif score < 40:
            direction = "BEARISH"
            action = "BUY PUTS / SELL CALLS"
        elif score < 45:
            direction = "LEAN BEARISH"
            action = "Bear Spreads / Buy Put Spreads"
        else:
            direction = "NEUTRAL"
            action = "Iron Condors / Straddles"

        # Strategy refinement
        if abs(term_spread) > 3: strategy = "Calendar Spreads exploit term structure"
        elif confidence > 0.6 and direction in ("BULLISH", "BEARISH"):
            strategy = "Directional debit spreads"
        elif confidence < 0.2:
            strategy = "Wait for clearer signal or sell premium"
        else:
            strategy = "Small directional bias — hedge accordingly"

        return {
            'direction': direction, 'confidence': confidence,
            'score': score, 'expected_move': expected_move,
            'expected_move_pct': expected_move_pct,
            'skew_signal': skew_signal, 'skew_ratio': skew_ratio,
            'skew_delta': skew_delta, 'term_signal': term_signal,
            'term_spread': term_spread, 'put_iv_avg': put_iv_avg,
            'call_iv_avg': call_iv_avg, 'atm_iv': atm_iv,
            'anomalous_strikes': anomalous_strikes,
            'action': action, 'strategy': strategy,
            'near_atm': near_atm, 'far_atm': far_atm
        }

    # ================================================================
    # IV DASHBOARD — DASH APPLICATION
    # ================================================================
    def _create_iv_dashboard(self, expiries):
        """Create and launch the real-time IV Surface Intelligence Dashboard using Plotly HTML."""
        import webbrowser
        from plotly.subplots import make_subplots
        import warnings
        import json
        import threading
        from http.server import HTTPServer, BaseHTTPRequestHandler
        from scipy.stats import norm
        import plotly.graph_objects as go
        from scipy.interpolate import griddata

        import sys
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))

        # Lightweight local API server removed to global scope
        html_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'iv_dashboard.html')

        print(f"\n  ╔══════════════════════════════════════════════════╗")
        print(f"  ║  IV SURFACE INTELLIGENCE DASHBOARD               ║")
        print(f"  ║  Auto-refreshing every 10 seconds                ║")
        print(f"  ║  Expiries: {', '.join(expiries):<36} ║")
        print(f"  ╚══════════════════════════════════════════════════╝")
        print(f"\n  Press Ctrl+C to stop.\n")

        first_run = True
        DARK_BG = '#0f0f19'
        CARD_BG = '#1a1a2e'
        ACCENT = '#4fc3f7'
        RED = '#ff4444'
        GREEN = '#66bb6a'
        YELLOW = '#ffd54f'
        WHITE = '#e0e0e0'
        MUTED = '#888'
        colors = ['#4fc3f7', '#ff7043', '#66bb6a', '#ab47bc', '#ffa726', '#ef5350']
        momentum_data: dict = {}   # populated per-cycle; safe default avoids NameError in HTML template

        try:
            while True:
                try:
                    # ── FETCH DATA ──
                    points = self._iv_dashboard_fetch(expiries)
                    spot = self.spot_price

                    if not points:
                        print("  Waiting for data...")
                        time.sleep(3)
                        continue

                    # ── PREDICTION ──
                    pred = self._iv_predict_spot(points, expiries)

                    # ── BUILD SUBPLOTS: 2D (left) + 3D (right) ──
                    fig = make_subplots(
                        rows=1, cols=2,
                        specs=[[{"type": "xy"}, {"type": "scene"}]],
                        subplot_titles=[
                            f"IV SMILE  |  {pred['direction']} ({pred['confidence']:.0%})",
                            f"3D SURFACE  |  Term: {pred['term_spread']:+.1f}%"
                        ],
                        horizontal_spacing=0.05
                    )

                    # ── 2D SMILE (col=1) ──
                    exp_data = {}
                    for p in points:
                        exp = p['expiry']
                        if exp not in exp_data: exp_data[exp] = []
                        exp_data[exp].append(p)

                    for i, exp in enumerate(expiries):
                        data = exp_data.get(exp, [])
                        if not data: continue
                        s_map = {}
                        for x in data:
                            k = x['strike']
                            if k not in s_map: s_map[k] = {'ivs': [], 'types': []}
                            s_map[k]['ivs'].append(x['iv'])
                            s_map[k]['types'].append(x['type'])

                        x_val = sorted(s_map.keys())
                        y_val = [np.mean(s_map[k]['ivs']) for k in x_val]
                        types_str = [', '.join(set(s_map[k]['types'])) for k in x_val]
                        color = colors[i % len(colors)]
                        atm = pred.get('near_atm', 0) if i == 0 else pred.get('far_atm', 0)

                        hover = [f"<b>Strike: {s:.0f}</b><br>IV: {iv:.2f}%<br>Type: {t}<br>Expiry: {exp}<br>ATM: {atm:.1f}%"
                                 for s, iv, t in zip(x_val, y_val, types_str)]

                        fig.add_trace(go.Scatter(
                            x=x_val, y=y_val, mode='lines+markers',
                            name=f"{exp} (ATM:{atm:.1f}%)",
                            line=dict(color=color, width=2.5),
                            marker=dict(size=6, color=color, line=dict(width=0.5, color='white')),
                            text=hover, hoverinfo='text'
                        ), row=1, col=1)

                    # Spot line on 2D
                    fig.add_vline(x=spot, line_dash="dash", line_color=YELLOW, line_width=2,
                                 annotation_text=f"Spot:{spot:.0f}", annotation_font_color=YELLOW,
                                 annotation_position="top right", row=1, col=1)

                    # Expected move band
                    if pred['expected_move'] > 0:
                        fig.add_vrect(x0=spot - pred['expected_move'], x1=spot + pred['expected_move'],
                                     fillcolor="rgba(255,213,79,0.08)", line_width=0,
                                     row=1, col=1)

                    # ── 3D SURFACE (col=2) ──
                    otm = [p for p in points if
                           (p['type'] == 'PE' and p['strike'] <= spot) or
                           (p['type'] == 'CE' and p['strike'] >= spot)]
                    if not otm: otm = points

                    xs = np.array([p['strike'] for p in otm])
                    ys = np.array([p['days'] for p in otm])
                    zs = np.array([p['iv'] for p in otm])
                    types = [p['type'] for p in otm]
                    exp_labels = [p['expiry'] for p in otm]

                    strike_grid = np.linspace(xs.min(), xs.max(), 50)
                    days_grid = np.linspace(ys.min(), ys.max(), 25)
                    sm, dm = np.meshgrid(strike_grid, days_grid)
                    try:
                        iv_mesh = griddata((xs, ys), zs, (sm, dm), method='cubic')
                        iv_nn = griddata((xs, ys), zs, (sm, dm), method='nearest')
                        iv_mesh = np.where(np.isnan(iv_mesh), iv_nn, iv_mesh)
                    except Exception:
                        iv_mesh = griddata((xs, ys), zs, (sm, dm), method='nearest')

                    fig.add_trace(go.Surface(
                        x=strike_grid, y=days_grid, z=iv_mesh,
                        colorscale='RdYlBu_r', opacity=0.75, showscale=True,
                        colorbar=dict(title='IV%', len=0.7, x=1.01, tickfont=dict(size=10)),
                        hovertemplate='Strike:%{x:.0f}<br>Days:%{y:.1f}<br>IV:%{z:.2f}%<extra></extra>'
                    ), row=1, col=2)

                    # Scatter points on 3D
                    hover_3d = [f"Strike:{s:.0f}<br>Days:{d:.1f}<br>IV:{iv:.2f}%<br>{t}<br>{e}"
                                for s, d, iv, t, e in zip(xs, ys, zs, types, exp_labels)]
                    mc = [RED if t == 'PE' else ACCENT for t in types]

                    fig.add_trace(go.Scatter3d(
                        x=xs, y=ys, z=zs, mode='markers',
                        marker=dict(size=3, color=mc, opacity=0.85, line=dict(width=0.3, color='white')),
                        text=hover_3d, hoverinfo='text', name='PE/CE Points'
                    ), row=1, col=2)

                    # ── LAYOUT ──
                    fig.update_layout(
                        height=550, width=1400,
                        paper_bgcolor=DARK_BG, plot_bgcolor='rgba(20,20,35,0.8)',
                        font=dict(color=WHITE, family='Inter, Segoe UI, sans-serif', size=11),
                        legend=dict(bgcolor='rgba(30,30,50,0.8)', font=dict(size=10, color=WHITE),
                                   x=0.01, y=0.99),
                        margin=dict(l=50, r=20, t=50, b=30),
                        hovermode='closest',
                        scene=dict(
                            xaxis=dict(title='Strike', backgroundcolor=DARK_BG, gridcolor='#333', color=MUTED),
                            yaxis=dict(title='Days', backgroundcolor=DARK_BG, gridcolor='#333', color=MUTED),
                            zaxis=dict(title='IV%', backgroundcolor=DARK_BG, gridcolor='#333', color=MUTED),
                            camera=dict(eye=dict(x=1.5, y=-1.8, z=0.8)),
                            bgcolor=DARK_BG
                        ),
                    )
                    fig.update_xaxes(gridcolor='rgba(100,100,100,0.15)', zeroline=False, title='Strike', row=1, col=1)
                    fig.update_yaxes(gridcolor='rgba(100,100,100,0.15)', zeroline=False, title='IV (%)', row=1, col=1)

                    # ── BUILD PREDICTION HTML PANEL ──
                    dir_color = GREEN if 'BULL' in pred['direction'] else RED if 'BEAR' in pred['direction'] else YELLOW
                    now_str = datetime.now().strftime('%H:%M:%S')

                    # Anomalous strikes HTML
                    anomalous_html = ""
                    if pred['anomalous_strikes']:
                        badges = ""
                        for a in pred['anomalous_strikes']:
                            bg = '#2a1a1a' if a['type'] == 'PE' else '#1a1a2a'
                            clr = RED if a['type'] == 'PE' else ACCENT
                            badges += (f'<span style="background:{bg};color:{clr};padding:3px 10px;'
                                      f'border-radius:6px;font-size:11px;border:1px solid {clr};margin-right:6px;">'
                                      f'{a["strike"]:.0f} {a["type"]} ({a["iv"]:.1f}% IV) [{a["expiry"]}]</span>')
                        anomalous_html = f'''<div style="margin-top:10px;">
                            <span style="color:{YELLOW};font-size:12px;font-weight:600;">🎯 SMART MONEY ZONES: </span>
                            {badges}</div>'''

                    prediction_panel = f'''
                    <div style="background:{CARD_BG};border-radius:12px;padding:18px;margin:10px 20px;
                                border:1px solid #2a2a4a;box-shadow:0 4px 20px rgba(0,0,0,0.3);
                                font-family:Inter,'Segoe UI',sans-serif;">
                        <div style="color:{ACCENT};font-size:13px;font-weight:700;letter-spacing:2px;
                                    margin-bottom:14px;">SPOT MOVEMENT PREDICTOR</div>
                        <div style="display:flex;gap:10px;flex-wrap:wrap;">
                            <!-- Direction -->
                            <div style="background:#12122a;border-radius:8px;padding:14px 18px;text-align:center;
                                        border:1px solid #2a2a4a;flex:1;min-width:150px;">
                                <div style="color:{MUTED};font-size:11px;font-weight:600;text-transform:uppercase;
                                            letter-spacing:1px;">DIRECTION</div>
                                <div style="font-size:22px;font-weight:700;color:{dir_color};">{pred['direction']}</div>
                                <div style="color:{MUTED};font-size:11px;">Confidence: {pred['confidence']:.0%}</div>
                            </div>
                            <!-- Expected Move -->
                            <div style="background:#12122a;border-radius:8px;padding:14px 18px;text-align:center;
                                        border:1px solid #2a2a4a;flex:1;min-width:150px;">
                                <div style="color:{MUTED};font-size:11px;font-weight:600;text-transform:uppercase;
                                            letter-spacing:1px;">EXPECTED 1-DAY MOVE</div>
                                <div style="font-size:22px;font-weight:700;color:{WHITE};">±{pred['expected_move']:.0f} pts</div>
                                <div style="color:{MUTED};font-size:11px;">±{pred['expected_move_pct']:.2f}%</div>
                            </div>
                            <!-- Skew -->
                            <div style="background:#12122a;border-radius:8px;padding:14px 18px;text-align:center;
                                        border:1px solid #2a2a4a;flex:1;min-width:150px;">
                                <div style="color:{MUTED};font-size:11px;font-weight:600;text-transform:uppercase;
                                            letter-spacing:1px;">SKEW RATIO</div>
                                <div style="font-size:22px;font-weight:700;color:{WHITE};">{pred['skew_ratio']:.3f}</div>
                                <div style="color:{MUTED};font-size:11px;">{pred['skew_signal']}</div>
                            </div>
                            <!-- Skew Delta -->
                            <div style="background:#12122a;border-radius:8px;padding:14px 18px;text-align:center;
                                        border:1px solid #2a2a4a;flex:1;min-width:150px;">
                                <div style="color:{MUTED};font-size:11px;font-weight:600;text-transform:uppercase;
                                            letter-spacing:1px;">SKEW Δ (30s)</div>
                                <div style="font-size:22px;font-weight:700;color:{RED if pred.get('skew_delta',0)>0.01 else GREEN if pred.get('skew_delta',0)<-0.01 else WHITE};">
                                    {pred.get('skew_delta',0):+.4f}</div>
                                <div style="color:{MUTED};font-size:11px;">{'↑ Puts bid' if pred.get('skew_delta',0)>0 else '↓ Calls bid' if pred.get('skew_delta',0)<0 else 'Stable'}</div>
                            </div>
                            <!-- Term Spread -->
                            <div style="background:#12122a;border-radius:8px;padding:14px 18px;text-align:center;
                                        border:1px solid #2a2a4a;flex:1;min-width:150px;">
                                <div style="color:{MUTED};font-size:11px;font-weight:600;text-transform:uppercase;
                                            letter-spacing:1px;">TERM SPREAD</div>
                                <div style="font-size:22px;font-weight:700;color:{RED if pred['term_spread']<-1 else GREEN if pred['term_spread']>1 else WHITE};">
                                    {pred['term_spread']:+.2f}%</div>
                                <div style="color:{MUTED};font-size:11px;">{pred['term_signal']}</div>
                            </div>
                            <!-- ATM IV -->
                            <div style="background:#12122a;border-radius:8px;padding:14px 18px;text-align:center;
                                        border:1px solid #2a2a4a;flex:1;min-width:150px;">
                                <div style="color:{MUTED};font-size:11px;font-weight:600;text-transform:uppercase;
                                            letter-spacing:1px;">ATM IV</div>
                                <div style="font-size:22px;font-weight:700;color:{WHITE};">{pred['atm_iv']:.2f}%</div>
                                <div style="color:{MUTED};font-size:11px;">Put:{pred['put_iv_avg']:.1f}% Call:{pred['call_iv_avg']:.1f}%</div>
                            </div>
                        </div>
                        {anomalous_html}
                        <!-- Action Bar -->
                        <div style="display:flex;align-items:center;gap:20px;padding:10px 16px;margin-top:12px;
                                    background:#12122a;border-radius:8px;border:1px solid #2a2a4a;">
                            <div><span style="color:{MUTED};font-size:12px;">⚡ ACTION: </span>
                                 <span style="color:{dir_color};font-size:14px;font-weight:700;">{pred['action']}</span></div>
                            <div><span style="color:{MUTED};font-size:12px;">📋 STRATEGY: </span>
                                 <span style="color:{WHITE};font-size:13px;">{pred['strategy']}</span></div>
                            <div style="display:flex;align-items:center;">
                            <div style="text-align:right;margin-right:15px;">
                                <div style="color:{MUTED};font-size:10px;font-weight:700;">DAY HIGH</div>
                                <div style="color:{WHITE};font-size:14px;font-weight:700;">{momentum_data.get('day_high', 0):.0f}</div>
                            </div>
                            <div style="text-align:right;margin-right:15px;">
                                <div style="color:{MUTED};font-size:10px;font-weight:700;">DAY LOW</div>
                                <div style="color:{WHITE};font-size:14px;font-weight:700;">{momentum_data.get('day_low', 0):.0f}</div>
                            </div>
                            <div style="text-align:right;">
                                <div style="color:{MUTED};font-size:10px;font-weight:700;">SPOT PRICE</div>
                                <div style="color:{ACCENT};font-size:20px;font-weight:900;">{spot:,.2f}</div>
                            </div>
                        </div>
                    </div>
                </div>
'''

                    # ── WRITE HTML FILE ──
                    plotly_html = fig.to_html(include_plotlyjs=True, full_html=False)

                    full_html = f'''<!DOCTYPE html>
<html><head>
<meta charset="utf-8">
<meta http-equiv="refresh" content="10">
<title>IV Surface Intelligence | {self.symbol}</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap" rel="stylesheet">
<style>
    body {{ margin:0; padding:10px 20px; background:{DARK_BG}; font-family:Inter,'Segoe UI',sans-serif; }}
    .header {{ display:flex; justify-content:space-between; align-items:center; padding:10px 20px;
               background:{CARD_BG}; border-radius:12px; margin-bottom:8px; border:1px solid #2a2a4a; }}
    .title {{ font-size:18px; font-weight:700; color:{ACCENT}; letter-spacing:2px; }}
    .spot {{ font-size:16px; font-weight:600; color:{YELLOW}; }}
    .live {{ color:{GREEN}; font-size:12px; font-weight:700; }}
</style>
</head><body>
    <div class="header">
        <div><span class="title">IV SURFACE INTELLIGENCE</span>
             <span style="color:{MUTED};font-size:14px;">  |  {self.symbol}</span></div>
        <div class="spot">Spot: {spot:,.0f}</div>
        <div><span class="live">⚡ LIVE</span>
             <span style="color:{MUTED};font-size:11px;"> Auto-refresh: 10s</span></div>
    </div>
    {plotly_html}
    {prediction_panel}
</body></html>'''

                    with open(html_path, 'w', encoding='utf-8') as f:
                        f.write(full_html)

                    if first_run:
                        webbrowser.open(f'file:///{html_path.replace(os.sep, "/")}')
                        first_run = False
                        print(f"  Dashboard opened at: file:///{html_path.replace(os.sep, '/')}")

                    print(f"  [{now_str}] Updated: {len(points)} pts | {pred['direction']} ({pred['confidence']:.0%}) | Spot:{spot:.0f}")
                    time.sleep(10)

                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    print(f"  Error: {e} — retrying in 5s...")
                    time.sleep(5)

        except KeyboardInterrupt:
            print("\n  Dashboard stopped.")
            # Cleanup
            try: os.remove(html_path)
            except: pass


    def iv_surface_live_2d(self):
        print("\n--- Real-Time 2D Smile Monitor (Live) ---")
        print("Support for Convergence Analysis (Near vs Far)")
        
        self.get_spot_price()
        print(f"Spot: {self.spot_price}")
        
        print("Enter Near Expiry (e.g., 2026-01-15):")
        near_exp = input("Near Expiry: ").strip()
        if not near_exp: return
        
        print("Enter Far Expiry (Optional, Press Enter to skip):")
        far_exp = input("Far Expiry: ").strip()
        
        import matplotlib.pyplot as plt
        from scipy.interpolate import make_interp_spline
        
        plt.ion()
        fig, ax = plt.figure(figsize=(10, 6)), plt.gca()
        
        previous_curve_near = None
        
        print("\nStarting Real-Time Loop... (Ctrl+C to Stop)")
        
        try:
            while True:
                # FETCH DATA Helper
                def get_curve(expiry):
                    old_exp = self.expiry_date
                    self.expiry_date = expiry
                    data = self.get_option_chain_data()
                    df = self.parse_and_filter(data)
                    self.expiry_date = old_exp
                    
                    if df.empty: return None, None, None
                    
                    T = self.analytics.get_time_to_expiry(expiry)
                    if T < 0.001: T = 0.001
                    spot = self.spot_price
                    
                    df = self._filter_strikes(df, spot, range_pct=0.06)
                    points = []
                    
                    for _, row in df.iterrows():
                        strike = row['strike']
                        o_type = row['type']
                        iv = self._ensure_iv(row['iv'], row['price'], strike, T, o_type)
                        
                        if iv > 0:
                            if (o_type == 'PE' and strike < spot) or (o_type == 'CE' and strike > spot):
                                points.append((strike, iv))
                                
                    if not points: return None, None, None
                    points.sort(key=lambda x: x[0])
                    strikes = [p[0] for p in points]
                    ivs = [p[1] for p in points]
                    
                    # Spline
                    try:
                        unique_strikes, unique_indices = np.unique(strikes, return_index=True)
                        unique_ivs = np.array(ivs)[unique_indices]
                        if len(unique_strikes) > 4:
                            x_smooth = np.linspace(min(unique_strikes), max(unique_strikes), 200)
                            spl = make_interp_spline(unique_strikes, unique_ivs, k=3)
                            y_smooth = spl(x_smooth)
                            return x_smooth, y_smooth, np.interp(spot, strikes, ivs)
                    except: pass
                    
                    return strikes, ivs, np.interp(spot, strikes, ivs)

                # Get Curves
                nx, ny, natm = get_curve(near_exp)
                fx, fy, fatm = (None, None, None)
                if far_exp:
                    fx, fy, fatm = get_curve(far_exp)
                
                if nx is None:
                    print("Waiting for data...")
                    time.sleep(2)
                    continue
                
                # Plot
                ax.clear()
                
                # Near Curve
                ax.plot(nx, ny, color='blue', linewidth=2, label=f"Near ({near_exp})")
                
                # Far Curve & Convergence
                nav_msg = ""
                if fx is not None:
                    ax.plot(fx, fy, color='purple', linewidth=2, label=f"Far ({far_exp})")
                    
                    # Convergence Logic
                    # Compare ATM difference
                    spread = fatm - natm
                    ax.fill_between(nx, ny, np.interp(nx, fx, fy), color='gray', alpha=0.1)
                    
                    nav_msg = f" | Term Spread: {spread:.2f}%"
                    if spread < 0: nav_msg += " (BACKWARDATION)"
                    else: nav_msg += " (CONTANGO)"
                
                # Ghost (Near only)
                if previous_curve_near is not None:
                     ax.plot(previous_curve_near[0], previous_curve_near[1], color='gray', linestyle='--', alpha=0.5, label='5m Ago')

                ax.axvline(self.spot_price, color='orange', linestyle=':', label='Spot')
                ax.set_title(f"LIVE VOLATILITY CONVERGENCE | Spot: {self.spot_price}{nav_msg}")
                ax.set_xlabel("Strike")
                ax.set_ylabel("IV (%)")
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                plt.draw()
                plt.pause(10)
                
                # Update Ghost every 5 minutes (fixed)
                if not hasattr(self, '_ghost_time') or (time.time() - self._ghost_time) > 300:
                    previous_curve_near = (nx, ny)
                    self._ghost_time = time.time()

        except KeyboardInterrupt:
            print("Stopped.")
            plt.close()


    def volatility_signal_analysis(self):
        while True:
            print("\n--- Volatility Signal Analysis Menu ---")
            print("1. Daily Volatility Cone (Mean Reversion)")
            print("2. Intraday Real-Time Monitor (VRP & Trend)")
            print("3. Back")
            
            c = input("Select: ")
            if c == '1':
                self.daily_volatility_cone()
            elif c == '2':
                self.intraday_volatility_monitor()
            elif c == '3':
                break

    def daily_volatility_cone(self):
        print("\n" + "="*70)
        print("  IV INTELLIGENCE ENGINE")
        print("="*70)
        
        # ============================================================
        # 1. FETCH 1-YEAR DAILY DATA
        # ============================================================
        print("\nFetching 1 Year of historical data...")
        closes = []
        highs = []
        lows = []
        
        try:
             today = datetime.now()
             start = today - pd.Timedelta(days=365)
             data = {
                "symbol": self.symbol, "resolution": "D", "date_format": "1",
                "range_from": start.strftime("%Y-%m-%d"),
                "range_to": today.strftime("%Y-%m-%d"),
                "cont_flag": "1"
            }
             r = self.fyers.history(data=data)
             
             if r.get('s') == 'ok':
                 candles = r['candles']
                 highs = [c[2] for c in candles]
                 lows = [c[3] for c in candles]
                 closes = [c[4] for c in candles]
                 print(f"Loaded {len(closes)} trading days.")
             else:
                 print(f"History Fetch Failed: {r}")
                 return
        except Exception as e:
            print(f"Error fetching history: {e}")
            return

        # ============================================================
        # 2. VOL METRICS
        # ============================================================
        rolling_hv = self.analytics.calculate_rolling_historical_volatility(closes, window=20)
        parkinson_hv = self.analytics.calculate_parkinson_volatility(highs, lows, window=20)
        
        if rolling_hv.empty or len(rolling_hv) < 40:
            print("Not enough data for analysis.")
            return
            
        current_hv = rolling_hv.iloc[-1]
        current_park_hv = parkinson_hv.iloc[-1] if not parkinson_hv.empty else 0
        mean_hv = rolling_hv.mean()
        min_hv = rolling_hv.min()
        max_hv = rolling_hv.max()
        hv_percentile = (rolling_hv < current_hv).mean() * 100
        
        print(f"\n--- HV Statistics (1 Year) ---")
        print(f"Current 20d HV:  {current_hv:.2f}%  (Parkinson: {current_park_hv:.2f}%)")
        print(f"1-Yr Mean:       {mean_hv:.2f}%")
        print(f"1-Yr Range:      {min_hv:.2f}% - {max_hv:.2f}%")
        print(f"HV Percentile:   {hv_percentile:.1f}%")

        # ============================================================
        # 3. REGIME DETECTOR (Bollinger Bandwidth on HV)
        # ============================================================
        bb_window = 20
        hv_sma = rolling_hv.rolling(bb_window).mean()
        hv_std = rolling_hv.rolling(bb_window).std()
        bb_upper = hv_sma + 2 * hv_std
        bb_lower = hv_sma - 2 * hv_std
        
        # Bandwidth = (Upper - Lower) / SMA * 100
        bandwidth = ((bb_upper - bb_lower) / hv_sma * 100).dropna()
        current_bw = bandwidth.iloc[-1] if not bandwidth.empty else 0
        
        # HV direction (slope of last 5 readings)
        hv_slope = 0
        if len(rolling_hv) >= 5:
            recent_5 = rolling_hv.iloc[-5:].values
            hv_slope = recent_5[-1] - recent_5[0]  # positive = rising
        
        # IV Change Velocity (5d and 10d)
        iv_velocity_5d = rolling_hv.diff(5).iloc[-1] if len(rolling_hv) >= 6 else 0
        iv_velocity_10d = rolling_hv.diff(10).iloc[-1] if len(rolling_hv) >= 11 else 0
        iv_accel = iv_velocity_5d - (iv_velocity_10d / 2)  # acceleration
        
        # Vol-of-Vol (stability of HV itself)
        hv_changes = rolling_hv.diff().dropna()
        vov = hv_changes.tail(20).std() if len(hv_changes) >= 20 else 0
        
        # Adaptive Regime Detection (percentile-based, not hardcoded)
        bw_percentile = (bandwidth < current_bw).mean() * 100 if not bandwidth.empty else 50
        
        if bw_percentile < 20:
            regime = "COMPRESSION"
            regime_desc = "Vol coiling tight — big move likely coming"
            if vov < hv_changes.std() * 0.5:
                regime_desc += " (stable compression — breakout likely)"
            else:
                regime_desc += " (unstable — false breakouts possible)"
        elif bw_percentile > 80 and hv_slope > 0:
            regime = "EXPANSION"
            regime_desc = "Vol expanding — trending, momentum in play"
        elif bw_percentile > 80 and hv_slope <= 0:
            regime = "MEAN REVERSION"
            regime_desc = "Post-spike collapse — vol reverting to mean"
        else:
            regime = "NORMAL"
            regime_desc = "No clear edge from vol structure"
        
        print(f"\n--- Regime Detection (Adaptive) ---")
        print(f"Bollinger Bandwidth: {current_bw:.1f}% (Percentile: {bw_percentile:.0f}%)")
        print(f"HV Slope (5d):       {hv_slope:+.2f}%")
        print(f"Vol-of-Vol:          {vov:.3f}")
        print(f"IV Velocity (5d):    {iv_velocity_5d:+.2f}%  Accel: {iv_accel:+.2f}")
        print(f"REGIME:              {regime}")
        print(f"                     {regime_desc}")
        
        # ============================================================
        # 4. MEAN REVERSION HALF-LIFE
        # ============================================================
        try:
            hv_vals = rolling_hv.dropna().values
            if len(hv_vals) > 20:
                # Autocorrelation at lag-1
                hv_centered = hv_vals - np.mean(hv_vals)
                autocorr = np.correlate(hv_centered[:-1], hv_centered[1:], mode='valid')[0]
                autocorr /= np.correlate(hv_centered[:-1], hv_centered[:-1], mode='valid')[0]
                
                if 0 < autocorr < 1:
                    lam = -np.log(autocorr)  # Mean reversion speed
                    half_life = np.log(2) / lam
                    print(f"\nMean Reversion Half-Life: {half_life:.1f} trading days")
                    
                    # Distance from mean
                    dist_from_mean = current_hv - mean_hv
                    if abs(dist_from_mean) > hv_std.iloc[-1]:
                        direction = "above" if dist_from_mean > 0 else "below"
                        print(f"HV is {abs(dist_from_mean):.1f}% {direction} mean → expect reversion in ~{half_life:.0f} days")
                    else:
                        print(f"HV is near mean — no strong reversion signal")
                else:
                    half_life = 0
                    print(f"\nMean Reversion: Autocorrelation too weak to estimate")
            else:
                half_life = 0
        except:
            half_life = 0

        # ============================================================
        # 5. GET CURRENT ATM IV
        # ============================================================
        print("\nFetching Current ATM IV...")
        self.get_spot_price()
        spot = self.spot_price
        
        exp = self._near_expiry
        print(f"Using expiry: {exp}")
        
        iv = 0
        old_exp = self.expiry_date
        self.expiry_date = exp
        try:
            d = self.get_option_chain_data()
            _df = self.parse_and_filter(d)
        except: _df = pd.DataFrame()
        self.expiry_date = old_exp

        if not _df.empty and spot > 0:
            _df['dist'] = abs(_df['strike'] - spot)
            row = _df.loc[_df['dist'].idxmin()]
            iv = row['iv']
            T_iv = self.analytics.get_time_to_expiry(exp)
            iv = self._ensure_iv(iv, row['price'], row['strike'], T_iv, row['type'])
        
        if iv <= 0:
            iv = float(input("Enter Manual ATM IV: ") or 15)
            
        print(f"Current ATM IV: {iv:.2f}%")
        
        # IV vs HV
        vrp = iv - current_hv
        z_score = self.analytics.calculate_z_score(iv, rolling_hv)
        
        print(f"VRP (IV - HV):   {vrp:+.2f}%")
        print(f"IV Z-Score:      {z_score:.2f}")
        
        # ============================================================
        # 6. PROBABILITY ENGINE
        # ============================================================
        print(f"\n--- Probability Engine ---")
        
        # Calculate forward 5-day realized moves at each historical point
        s = pd.Series(closes)
        log_rets = np.log(s / s.shift(1)).dropna()
        
        # For each day, compute forward 5-day cumulative return
        fwd_5d = (s.shift(-5) / s - 1) * 100  # Forward 5-day % move
        fwd_5d = fwd_5d.dropna()
        
        # Also compute rolling HV at each point to bucket by
        hv_at_each = rolling_hv.reindex(fwd_5d.index)
        
        # Build aligned DataFrame
        prob_df = pd.DataFrame({
            'hv': hv_at_each,
            'fwd_move': fwd_5d
        }).dropna()
        
        if len(prob_df) > 30:
            # Bucket by HV percentile (quintiles)
            prob_df['hv_bucket'] = pd.qcut(prob_df['hv'], 5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'], duplicates='drop')
            
            # Find which bucket current HV falls into
            hv_breaks = prob_df['hv'].quantile([0, 0.2, 0.4, 0.6, 0.8, 1.0]).values
            if current_hv <= hv_breaks[1]: current_bucket = 'Very Low'
            elif current_hv <= hv_breaks[2]: current_bucket = 'Low'
            elif current_hv <= hv_breaks[3]: current_bucket = 'Medium'
            elif current_hv <= hv_breaks[4]: current_bucket = 'High'
            else: current_bucket = 'Very High'
            
            # Get moves for current bucket
            bucket_moves = prob_df[prob_df['hv_bucket'] == current_bucket]['fwd_move']
            
            if len(bucket_moves) > 5:
                print(f"Current HV Bucket: '{current_bucket}' ({len(bucket_moves)} historical samples)")
                print(f"\nHistorical 5-Day Forward Moves when HV was '{current_bucket}':")
                
                pcts = [10, 25, 50, 75, 90]
                for p in pcts:
                    val = np.percentile(bucket_moves, p)
                    pts = abs(val / 100 * spot)
                    print(f"  {p:>3}th percentile:  {val:+.2f}%  ({pts:+.0f} pts)")
                
                # Probability calculations
                prob_1pct = (bucket_moves.abs() > 1.0).mean() * 100
                prob_2pct = (bucket_moves.abs() > 2.0).mean() * 100
                prob_up = (bucket_moves > 0).mean() * 100
                
                print(f"\n  P(|move| > 1% in 5d):  {prob_1pct:.0f}%")
                print(f"  P(|move| > 2% in 5d):  {prob_2pct:.0f}%")
                print(f"  P(up in 5d):           {prob_up:.0f}%")
                
                # Expected range
                p10 = np.percentile(bucket_moves, 10)
                p90 = np.percentile(bucket_moves, 90)
                print(f"\n  Expected 5-Day Range (80% confidence):")
                print(f"  [{spot * (1 + p10/100):.0f}] ─── [{spot:.0f}] ─── [{spot * (1 + p90/100):.0f}]")
                print(f"  ({p10:+.1f}%)          (spot)          ({p90:+.1f}%)")
            else:
                print(f"Not enough samples in bucket '{current_bucket}'.")
                bucket_moves = pd.Series(dtype=float)
        else:
            print("Not enough data for probability analysis.")
            bucket_moves = pd.Series(dtype=float)
            current_bucket = "N/A"

        # ============================================================
        # 7. SCORED SIGNAL SYSTEM
        # ============================================================
        signal_metrics = {
            'vrp': vrp,
            'regime': regime
        }
        signal_result = self._score_signal(signal_metrics)
        
        self._format_signal_report(
            spot, iv, current_hv, signal_result, regime,
            iv_velocity_5d=iv_velocity_5d, iv_accel=iv_accel, vrp=vrp
        )
        
        # ============================================================
        # 8. MEMORY ENGINE (Persist to JSON)
        # ============================================================
        memory_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'vol_memory.json')
        today_str = datetime.now().strftime("%Y-%m-%d")
        
        # Load existing memory
        memory = {}
        try:
            if os.path.exists(memory_file):
                with open(memory_file, 'r') as f:
                    memory = json.load(f)
        except: memory = {}
        
        # Count days in current regime
        days_in_regime = 1
        sorted_dates = sorted(memory.keys(), reverse=True)
        prev_regime = None
        for d_key in sorted_dates:
            if memory[d_key].get('regime') == regime:
                days_in_regime += 1
            else:
                prev_regime = memory[d_key].get('regime')
                break
        
        # Save today's snapshot (enhanced with new fields)
        memory[today_str] = {
            'iv': round(iv, 2),
            'hv': round(current_hv, 2),
            'parkinson': round(current_park_hv, 2),
            'regime': regime,
            'bandwidth': round(current_bw, 2),
            'z_score': round(z_score, 2),
            'percentile': round(hv_percentile, 1),
            'vrp': round(vrp, 2),
            'spot': round(spot, 2),
            'iv_velocity_5d': round(iv_velocity_5d, 2),
            'composite_score': round(signal_result['composite'], 1),
            'prev_regime': prev_regime
        }
        
        # Keep last 90 days only
        if len(memory) > 90:
            sorted_keys = sorted(memory.keys())
            for old_key in sorted_keys[:-90]:
                del memory[old_key]
        
        try:
            with open(memory_file, 'w') as f:
                json.dump(memory, f, indent=2)
            print(f"\n[Memory] Saved to {memory_file}")
        except Exception as e:
            print(f"[Memory] Save failed: {e}")
        
        print(f"[Memory] Days in '{regime}': {days_in_regime}")
        if prev_regime:
            print(f"[Memory] Previous regime: {prev_regime}")
        
        # Trend tracking from memory
        if len(sorted_dates) >= 3:
            recent_ivs = [memory[d]['iv'] for d in sorted_dates[:5] if 'iv' in memory.get(d, {})]
            if len(recent_ivs) >= 2:
                iv_trend = "RISING" if recent_ivs[0] > recent_ivs[-1] else "FALLING" if recent_ivs[0] < recent_ivs[-1] else "FLAT"
                print(f"[Memory] IV Trend ({len(recent_ivs)}d): {iv_trend} ({recent_ivs[-1]:.1f} → {recent_ivs[0]:.1f})")
        
        # Show recent memory table
        if len(sorted_dates) > 0:
            print(f"\n--- Recent Memory ---")
            print(f"{'Date':<12} | {'IV':<6} | {'HV':<6} | {'VRP':<6} | {'Regime':<16} | {'Score':<6}")
            print("-" * 65)
            for d_key in sorted_dates[:5]:
                m = memory[d_key]
                print(f"{d_key:<12} | {m.get('iv',0):<6.1f} | {m.get('hv',0):<6.1f} | {m.get('vrp',0):<+5.1f} | {m.get('regime',''):<16} | {m.get('composite_score','--'):<6}")

        # ============================================================
        # 9. DUAL-PANEL PLOT
        # ============================================================
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), gridspec_kw={'width_ratios': [3, 2]})
            
            # --- LEFT: HV with Bollinger Bands + IV ---
            x = range(len(rolling_hv))
            ax1.plot(rolling_hv.values, label='20d HV', color='blue', alpha=0.7)
            ax1.plot(parkinson_hv.values, label='Parkinson HV', color='orange', alpha=0.4, linestyle='--')
            
            # Bollinger Bands on HV
            valid_start = bb_window - 1  # Where BB values start
            bb_x = range(valid_start, len(rolling_hv))
            bb_up_vals = bb_upper.dropna().values
            bb_lo_vals = bb_lower.dropna().values
            sma_vals = hv_sma.dropna().values
            min_len = min(len(bb_x), len(bb_up_vals), len(bb_lo_vals), len(sma_vals))
            bb_x = range(valid_start, valid_start + min_len)
            
            ax1.fill_between(bb_x, bb_lo_vals[:min_len], bb_up_vals[:min_len], alpha=0.1, color='blue', label='BB Band')
            ax1.plot(bb_x, sma_vals[:min_len], 'b:', linewidth=0.8, alpha=0.5)
            
            # IV line
            ax1.axhline(iv, color='purple', linewidth=2, linestyle='-', label=f'ATM IV ({iv:.1f}%)')
            ax1.axhline(mean_hv, color='green', linewidth=1, linestyle=':', label=f'Mean HV ({mean_hv:.1f}%)')
            
            ax1.set_title(f'IV Intelligence: {self.symbol} | Regime: {regime}', fontsize=12, fontweight='bold')
            ax1.set_xlabel('Trading Days')
            ax1.set_ylabel('Annualized Volatility (%)')
            ax1.legend(fontsize=8)
            ax1.grid(True, alpha=0.3)
            
            # --- RIGHT: Probability Distribution ---
            if len(bucket_moves) > 5:
                ax2.hist(bucket_moves, bins=20, color='steelblue', alpha=0.7, edgecolor='white', density=True)
                ax2.axvline(0, color='black', linewidth=1, linestyle='-')
                ax2.axvline(bucket_moves.median(), color='red', linewidth=2, linestyle='--', label=f'Median: {bucket_moves.median():.2f}%')
                
                # Mark 1σ and 2σ
                std_move = bucket_moves.std()
                mean_move = bucket_moves.mean()
                ax2.axvline(mean_move - std_move, color='orange', linestyle=':', alpha=0.7)
                ax2.axvline(mean_move + std_move, color='orange', linestyle=':', alpha=0.7, label=f'±1σ ({std_move:.1f}%)')
                
                ax2.set_title(f'5-Day Forward Moves\nHV Bucket: {current_bucket} (n={len(bucket_moves)})', fontsize=11, fontweight='bold')
                ax2.set_xlabel('Forward 5-Day Move (%)')
                ax2.set_ylabel('Density')
                ax2.legend(fontsize=9)
                ax2.grid(True, alpha=0.3)
            else:
                ax2.text(0.5, 0.5, 'Insufficient data\nfor probability chart', ha='center', va='center', fontsize=14, transform=ax2.transAxes)
            
            plt.tight_layout()
            print("\nOpening IV Intelligence Plot...")
            plt.show()
            
        except ImportError:
            print("Matplotlib missing.")
            
        input("\nPress Enter to continue...")

    def intraday_volatility_monitor(self):
        print("\n--- Intraday Real-Time Monitor & VRP Scanner ---")
        
        # 0. Fetch Daily Baseline (for Robust VRP)
        print("Fetching Daily History for VRP Baseline...")
        daily_hv = 0
        try:
             today = datetime.now()
             start = today - pd.Timedelta(days=60)
             data = { "symbol": self.symbol, "resolution": "D", "date_format": "1", 
                     "range_from": start.strftime("%Y-%m-%d"), "range_to": today.strftime("%Y-%m-%d"), "cont_flag": "1" }
             r = self.fyers.history(data=data)
             if r.get('s') == 'ok':
                 closes = [c[4] for c in r['candles']]
                 if len(closes) > 20:
                     s = pd.Series(closes)
                     rets = np.log(s / s.shift(1)).dropna()
                     daily_hv = rets.tail(20).std() * np.sqrt(252) * 100
                     print(f"Daily 20d HV (Baseline): {daily_hv:.2f}%")
        except Exception as e: 
            print(f"Daily Fetch Info: {e}")

        # 1. Fetch Intraday History
        print("Fetching Intraday History (1-min candles)...")
        intra_closes = []
        try:
             today_str = datetime.now().strftime("%Y-%m-%d")
             data = { "symbol": self.symbol, "resolution": "1", "date_format": "1", 
                     "range_from": today_str, "range_to": today_str, "cont_flag": "1" }
             r = self.fyers.history(data=data)
             if r.get('s') == 'ok':
                 intra_closes = [float(c[4]) for c in r['candles']]
                 print(f"Loaded {len(intra_closes)} intraday minutes.")
             else:
                 print("No intraday history.")
        except: pass
        
        print("\nStarting Real-Time Analysis... (Ctrl+C to Stop)")
        
        exp = self._near_expiry
        self.expiry_date = exp
        print(f"Using expiry: {exp}")
        
        print(f"{'Time':<10} | {'Spot':<8} | {'IV':<6} | {'DailyVRP':<9} | {'IntraVRP':<9} | {'Skew':<6} | {'SIGNAL':<20}")
        print("-" * 90)
        
        iv_history = [] 
        spot_history = []
        
        try:
            while True:
                spot_data = self.get_spot_price()
                spot = spot_data.get('price', 0)
                if spot > 0:
                    intra_closes.append(spot)
                    spot_history.append(spot)
                    if len(intra_closes) > 300: intra_closes.pop(0) 
                    if len(spot_history) > 20: spot_history.pop(0)
                
                rv_intra = 0
                if len(intra_closes) > 10:
                    s = pd.Series(intra_closes)
                    rets = np.log(s / s.shift(1)).dropna()
                    rv_intra = rets.tail(20).std() * np.sqrt(252 * 375) * 100 

                d = self.get_option_chain_data()
                df = self.parse_and_filter(d)
                atm_iv = 0
                skew = 0
                
                if not df.empty and spot > 0:
                    df['dist'] = abs(df['strike'] - spot)
                    atm_row = df.loc[df['dist'].idxmin()]
                    T_intra = self.analytics.get_time_to_expiry(exp)
                    atm_iv = self._ensure_iv(atm_row['iv'], atm_row['price'], atm_row['strike'], T_intra, atm_row['type'])

                    try:
                        pe_df = df[df['type'] == 'PE']
                        ce_df = df[df['type'] == 'CE']
                        p_iv = pe_df.iloc[(pe_df['strike'] - spot * 0.95).abs().argmin()]['iv'] if not pe_df.empty else atm_iv
                        c_iv = ce_df.iloc[(ce_df['strike'] - spot * 1.05).abs().argmin()]['iv'] if not ce_df.empty else atm_iv
                        skew = p_iv - c_iv
                    except Exception: skew = 0
                
                daily_vrp = atm_iv - daily_hv if daily_hv > 0 else 0
                intra_vrp = atm_iv - rv_intra
                
                sig = "WAIT"
                if daily_vrp > 5.0:
                    sig = "SELL VOL (High Prem)"
                    if intra_vrp > 5.0: sig = "STRONG SELL (Crash?)"
                elif daily_vrp < -2.0:
                    sig = "BUY VOL (Cheap)"
                    
                spot_trend = "FLAT"
                if len(spot_history) >= 10:
                     avg_spot = sum(spot_history[-10:]) / 10
                     if spot > avg_spot: spot_trend = "UP"
                     elif spot < avg_spot: spot_trend = "DOWN"
                
                final_msg = f"{sig}"
                if spot_trend != "FLAT": final_msg += f" [{spot_trend}]"

                t_str = datetime.now().strftime("%H:%M:%S")
                print(f"{t_str}   | {spot:<8.2f} | {atm_iv:<6.2f} | {daily_vrp:<9.2f} | {intra_vrp:<9.2f} | {skew:<6.2f} | {final_msg:<20}")
                
                time.sleep(5)
                
        except KeyboardInterrupt:
            print("\nStopped.")

    def option_seller_advisor(self):
        print("\n" + "="*70)
        print("  OPTION SELLER ADVISOR — Range & Strike Selection")
        print("="*70)

        # ── Context from previous signals ──────────────────────────
        brief = self.memory.get_brief_text(self.spot_price)
        if brief:
            print(brief)
        self._seller_signal_id = None  # for dedup

        # ============================================================
        # 1. SETUP — Spot, Expiry, Option Chain
        # ============================================================
        print("\nFetching Spot Price...")
        self.get_spot_price()
        spot = self.spot_price
        if spot <= 0:
            print("Error: Could not fetch spot price.")
            return
        print(f"Spot: {spot:,.2f}")
        
        exp = self._near_expiry
        print(f"Using expiry: {exp}")
        T = self.analytics.get_time_to_expiry(exp)
        if T <= 0:
            print("Error: Expiry is in the past.")
            return
        DTE = max(1, int(T * 365))
        print(f"DTE: {DTE} days (T={T:.4f}y)")
        
        
        # ============================================================
        # 3. HISTORICAL BREACH ANALYSIS (Calculate once)
        # ============================================================
        print(f"\n--- Historical Breach Rate ({DTE}-Day Window) ---")
        breach_pct = 0
        total_samples = 0
        try:
            today = datetime.now()
            start = today - pd.Timedelta(days=365)
            hist_data = {
                "symbol": self.symbol, "resolution": "D", "date_format": "1",
                "range_from": start.strftime("%Y-%m-%d"),
                "range_to": today.strftime("%Y-%m-%d"),
                "cont_flag": "1"
            }
            r = self.fyers.history(data=hist_data)
            
            if r.get('s') == 'ok' and r.get('candles'):
                candles = r['candles']
                hist_closes = pd.Series([c[4] for c in candles])
                
                # Calculate rolling HV to estimate what EM would have been
                log_rets = np.log(hist_closes / hist_closes.shift(1)).dropna()
                rolling_hv = log_rets.rolling(20).std() * np.sqrt(252) * 100
                
                breach_count = 0
                actual_moves = []
                
                window = min(DTE, len(hist_closes) - 21)
                for i in range(20, len(hist_closes) - window):
                    hv_at_i = rolling_hv.iloc[i]
                    if pd.isna(hv_at_i) or hv_at_i <= 0:
                        continue
                    
                    s_i = hist_closes.iloc[i]
                    em_hist = s_i * (hv_at_i / 100) * np.sqrt(window / 252) * 0.85
                    
                    # Check max move in forward window
                    fwd_slice = hist_closes.iloc[i+1:i+1+window]
                    max_up = fwd_slice.max() - s_i
                    max_down = s_i - fwd_slice.min()
                    max_move = max(abs(max_up), abs(max_down))
                    actual_moves.append(max_move / s_i * 100)
                    
                    if max_move > em_hist:
                        breach_count += 1
                    total_samples += 1
                
                if total_samples > 0:
                    breach_pct = breach_count / total_samples * 100
                    print(f"  Samples: {total_samples} | Breach Rate: {breach_pct:.1f}% | Safe: {100 - breach_pct:.1f}%")
                    
                    if actual_moves:
                        avg_move = np.mean(actual_moves)
                        p90_move = np.percentile(actual_moves, 90)
                        print(f"  Avg max move: {avg_move:.2f}% | 90th pct: {p90_move:.2f}%")
                else:
                    print("  Insufficient data for breach analysis.")
            else:
                print("  Could not fetch historical data.")
        except Exception as e:
            print(f"  Breach analysis error: {e}")

        # ============================================================
        # REAL-TIME LOOP
        # ============================================================
        try:
            while True:
                print("\n" + "="*70)
                print(f"  UPDATE: {datetime.now().strftime('%H:%M:%S')} | Spot: {spot:.2f}")
                print("="*70)
                
                # Fetch fresh spot
                self.get_spot_price()
                spot = self.spot_price

                # Fetch option chain (using selected expiry)
                old_exp = self.expiry_date
                self.expiry_date = exp
                try:
                    raw = self.get_option_chain_data()
                    df = self.parse_and_filter(raw)
                except Exception as e:
                    print(f"Error fetching chain: {e}")
                    time.sleep(10)
                    continue
                self.expiry_date = old_exp
                
                if df.empty:
                    print("No option chain data available. Retrying...")
                    time.sleep(10)
                    continue
                
                # ============================================================
                # 2. EXPECTED MOVE — Straddle-Based + IV-Based
                # ============================================================
                # Find ATM separately for CE and PE (closest OTM or ATM)
                df['dist'] = abs(df['strike'] - spot)
                ce_df = df[df['type'] == 'CE'].copy()
                pe_df = df[df['type'] == 'PE'].copy()
                
                atm_strike = df.loc[df['dist'].idxmin(), 'strike']
                
                # Best ATM CE (nearest strike >= spot)
                ce_atm = ce_df[ce_df['strike'] >= spot]
                if ce_atm.empty:
                    ce_atm = ce_df
                atm_ce = ce_atm.loc[ce_atm['dist'].idxmin()] if not ce_atm.empty else None
                
                # Best ATM PE (nearest strike <= spot)
                pe_atm = pe_df[pe_df['strike'] <= spot]
                if pe_atm.empty:
                    pe_atm = pe_df
                atm_pe = pe_atm.loc[pe_atm['dist'].idxmin()] if not pe_atm.empty else None
                
                ce_price = atm_ce['price'] if atm_ce is not None else 0
                pe_price = atm_pe['price'] if atm_pe is not None else 0
                ce_strike = atm_ce['strike'] if atm_ce is not None else atm_strike
                pe_strike = atm_pe['strike'] if atm_pe is not None else atm_strike
                
                # ATM IV (try CE first, then PE)
                atm_iv = 0
                if atm_ce is not None:
                    atm_iv = self._ensure_iv(atm_ce['iv'], ce_price, ce_strike, T, 'CE')
                if atm_iv <= 0 and atm_pe is not None:
                    atm_iv = self._ensure_iv(atm_pe['iv'], pe_price, pe_strike, T, 'PE')
                
                # If either leg price is 0, compute it via BSM
                sigma_dec = atm_iv / 100 if atm_iv > 0 else 0.15
                if ce_price <= 0 and atm_iv > 0:
                    ce_price = self.analytics.black_scholes(spot, ce_strike, T, 0.07, sigma_dec, 'CE')
                if pe_price <= 0 and atm_iv > 0:
                    pe_price = self.analytics.black_scholes(spot, pe_strike, T, 0.07, sigma_dec, 'PE')
                
                straddle_price = ce_price + pe_price
                
                # EM from straddle (market's actual pricing)
                em_straddle = straddle_price
                em_straddle_pct = (em_straddle / spot) * 100
                
                # EM from IV (theoretical: 1-SD move)
                em_iv = spot * (atm_iv / 100) * np.sqrt(T) * 0.85
                em_iv_pct = (em_iv / spot) * 100
                
                print(f"  ATM Strike:      {atm_strike:.0f}")
                print(f"  ATM IV:          {atm_iv:.2f}%")
                print(f"  Straddle Price:  {straddle_price:.2f} (CE:{ce_price:.2f} + PE:{pe_price:.2f})")
                print(f"\n  Expected Move (Straddle):  ±{em_straddle:.0f} pts  ({em_straddle_pct:.2f}%)")
                print(f"  Range (Straddle): [{spot - em_straddle:.0f}] — [{spot:.0f}] — [{spot + em_straddle:.0f}]")
                
                # ============================================================
                # 3. HISTORICAL BREACH ANALYSIS (Calculated once outside loop is better, but reusing var is fine)
                # ============================================================
                if r.get('s') == 'ok' and total_samples > 0:
                     print(f"  Hist Breach Rate: {breach_pct:.1f}% (EM held {100-breach_pct:.1f}% safe)")
                
                # ============================================================
                # 4. OI-BASED RANGE BOUNDS — Max Pain, Walls
                # ============================================================
                print(f"\n--- OI-Based Range Bounds ---")
                max_pain = 0
                call_wall = 0
                put_wall = 0
                try:
                    from KeyLevelsEngine import KeyLevelsEngine
                    kle = KeyLevelsEngine()
                    # DIRECT CALCULATION using correct DF
                    max_pain = kle.calculate_max_pain(df)
                    walls = kle.calculate_oi_walls(df, spot)
                    call_wall = walls['call_wall']
                    put_wall = walls['put_wall']
                    pcr = kle.calculate_pcr(df)
                    
                    print(f"  Max Pain:    {max_pain}  (distance: {spot - max_pain:+.0f} pts)")
                    print(f"  Put Wall:    {put_wall}  (support)")
                    print(f"  Call Wall:   {call_wall}  (resistance)")
                    print(f"  PCR:         {pcr:.2f}  ({'Bearish' if pcr > 1.2 else 'Bullish' if pcr < 0.8 else 'Neutral'})")
                    
                    # Compare EM vs OI walls
                    em_upper = spot + em_straddle
                    em_lower = spot - em_straddle
                    if call_wall > 0 and call_wall < em_upper:
                        print(f"  ⚠ Call Wall ({call_wall}) is INSIDE EM upper")
                    if put_wall > 0 and put_wall > em_lower:
                        print(f"  ⚠ Put Wall ({put_wall}) is INSIDE EM lower")
                except Exception as e:
                    print(f"  OI analysis error: {e}")

                # ============================================================
                # 4b. OI VELOCITY ANALYSIS (Calculated via _compute_seller_data)
                # ============================================================
                # We skip manual printout here to avoid redundancy with the unified dashboard stats
                pass

                # oi_pressure / oi_pressure_score are computed in _compute_seller_data;
                # initialise safe defaults here so the print block below never raises NameError
                # if the velocity calculation was skipped (e.g. first run, no baseline OI yet).
                oi_pressure: str = 'NEUTRAL'
                oi_pressure_score: float = 0.0

                if oi_pressure == 'BULLISH':
                    print(f"  → Put writers building support, spot likely to hold/move UP")
                elif oi_pressure == 'BEARISH':
                    print(f"  → Call writers capping upside, spot likely to stall/move DOWN")
                else:
                    print(f"  → Balanced OI, range-bound likely")

                
                # ============================================================
                # 5. STRIKE SELECTION TABLE
                # ============================================================
                print(f"\n--- Strike Selection Table (DTE={DTE}) ---")
                print(f"{'Strike':>8} {'Type':>4} {'Dist':>7} {'P(Touch)':>9} {'P(OTM)':>8} {'Prem':>7} {'Theta':>7} {'Signal':>10}")
                print("-" * 72)
                
                r_rate = 0.07
                sigma = atm_iv / 100 if atm_iv > 0 else 0.15
                
                # Get unique strikes near ATM
                all_strikes = sorted(df['strike'].unique())
                atm_idx = min(range(len(all_strikes)), key=lambda i: abs(all_strikes[i] - spot))
                nearby = all_strikes[max(0, atm_idx - 10):atm_idx + 11]
                
                for strike in nearby:
                    for otype in ['CE', 'PE']:
                        row_df = df[(df['strike'] == strike) & (df['type'] == otype)]
                        if row_df.empty: continue
                        
                        row = row_df.iloc[0]
                        price = row['price']
                        if price < 0.5: continue
                        
                        dist = strike - spot
                        
                        # Skip ITM
                        if otype == 'CE' and dist < 0: continue
                        if otype == 'PE' and dist > 0: continue
                        
                        # BSM probabilities (ATM IV)
                        s = sigma 
                        try:
                            d1 = (np.log(spot / strike) + (r_rate + 0.5 * s**2) * T) / (s * np.sqrt(T))
                            d2 = d1 - s * np.sqrt(T)
                            from scipy.stats import norm
                            if otype == 'CE':
                                prob_otm = norm.cdf(-d2) * 100
                                prob_touch = min(100, 2 * norm.cdf(-abs(d2)) * 100)
                            else:
                                prob_otm = norm.cdf(d2) * 100
                                prob_touch = min(100, 2 * norm.cdf(-abs(d2)) * 100)
                        except:
                            prob_otm = 50; prob_touch = 50
                        
                        # Greeks (using per-strike IV if avail)
                        try:
                            row_iv = self._ensure_iv(row['iv'], price, strike, T, otype)
                            g_iv = row_iv / 100 if row_iv > 0 else sigma
                            greeks = self.analytics.calculate_greeks(spot, strike, T, r_rate, g_iv, otype)
                            theta = greeks.get('theta', 0)
                        except: theta = 0
                        
                        # Signal
                        signal = ""
                        if prob_otm >= 85: signal = "★ SAFE"
                        elif prob_otm >= 75: signal = "✓ GOOD"
                        elif prob_otm >= 60: signal = "~ OK"
                        else: signal = "✗ RISKY"
                        
                        print(f"{strike:>8.0f} {otype:>4} {dist:>+7.0f} {prob_touch:>8.1f}% {prob_otm:>7.1f}% {price:>7.2f} {theta:>7.2f} {signal:>10}")
                
                # ============================================================
                # 6. VISUAL RANGE MAP
                # ============================================================
                levels = {}
                levels['SPOT'] = spot
                if em_straddle > 0:
                    levels['EM-LO'] = spot - em_straddle
                    levels['EM-HI'] = spot + em_straddle
                if max_pain > 0: levels['MXPN'] = max_pain
                if put_wall > 0: levels['P.WALL'] = put_wall
                if call_wall > 0: levels['C.WALL'] = call_wall
                
                sorted_levels = sorted(levels.items(), key=lambda x: x[1])
                labels = "  ".join(f"{name:>7}" for name, _ in sorted_levels)
                values = "  ".join(f"{val:>7.0f}" for _, val in sorted_levels)
                
                if len(sorted_levels) >= 2:
                    lo = sorted_levels[0][1]; hi = sorted_levels[-1][1]
                    span = hi - lo if hi > lo else 1
                    bar_width = 60
                    bar = list("─" * bar_width)
                    for name, val in sorted_levels:
                        pos = int((val - lo) / span * (bar_width - 1))
                        pos = max(0, min(bar_width - 1, pos))
                        if name == 'SPOT': bar[pos] = '●'
                        elif 'EM' in name: bar[pos] = '│'
                        elif 'WALL' in name: bar[pos] = '┃'
                        else: bar[pos] = '◆'
                    print(f"\n  {labels}")
                    print(f"  {values}")
                    print(f"  {''.join(bar)}")
                    # Show OI pressure alongside range map
                    if oi_pressure != 'NEUTRAL':
                        arrow = '▲' if oi_pressure == 'BULLISH' else '▼'
                        print(f"  {arrow} OI Pressure: {oi_pressure} ({oi_pressure_score:+.0f})")
                
                # Sell zone recommendation
                if call_wall > 0 and put_wall > 0:
                    safe_ce = min(call_wall, spot + em_straddle) if em_straddle > 0 else call_wall
                    safe_pe = max(put_wall, spot - em_straddle) if em_straddle > 0 else put_wall
                    print(f"  SELL CE above: {safe_ce:.0f}  |  SELL PE below: {safe_pe:.0f}")

                    # ── Log/update seller range in SignalMemory (dedup) ─────
                    try:
                        seller_payload = {
                            'direction':    'NEUTRAL',
                            'action':       f"SELL CE>{safe_ce:.0f} / SELL PE<{safe_pe:.0f}",
                            'sell_ce_above':round(float(safe_ce), 1),
                            'sell_pe_below':round(float(safe_pe), 1),
                            'em':           round(float(em_straddle), 1),
                            'atm_iv':       round(float(atm_iv if 'atm_iv' in dir() else 0), 2),
                            'call_wall':    float(call_wall),
                            'put_wall':     float(put_wall),
                            'max_pain':     float(max_pain if 'max_pain' in dir() else 0),
                            'oi_pressure':  oi_pressure,
                            'oi_pressure_score': round(float(oi_pressure_score), 1),
                        }
                        if self._seller_signal_id is None:
                            self._seller_signal_id = self.memory.log_signal(
                                'OptionSellerAdvisor', seller_payload,
                                spot=spot, expiry=exp)
                        else:
                            self.memory.update_signal(self._seller_signal_id, seller_payload)
                        self.memory.update_context({
                            'seller_safe_range': [safe_pe, safe_ce]
                        }, spot=spot)
                    except Exception:
                        pass

                print("\nWaiting 30s for next update... (Ctrl+C to Exit)")
                time.sleep(30)
                
        except KeyboardInterrupt:
            print("\nExiting Option Seller Advisor...")
            return



    def _create_unified_dashboard(self, expiries):
        """Create and launch unified tabbed dashboard with all analysis modules."""
        import webbrowser
        import threading
        import socket
        import http.server
        from plotly.subplots import make_subplots
        import plotly.graph_objects as go
        from scipy.interpolate import griddata
        from scipy.stats import norm

        dashboard_dir = os.path.dirname(os.path.abspath(__file__))
        html_path = os.path.join(dashboard_dir, 'unified_dashboard.html')
        near_exp = self._near_expiry
        far_exp = self._far_expiry

        print(f"\n  ╔══════════════════════════════════════════════════╗")
        print(f"  ║  UNIFIED VOLATILITY DASHBOARD                    ║")
        print(f"  ║  Auto-refreshing every 15 seconds (HTTP)         ║")
        print(f"  ║  Expiries: {', '.join(expiries):<36} ║")
        print(f"  ╚══════════════════════════════════════════════════╝")
        print(f"\n  Press Ctrl+C to stop.\n")

        # ── Start local HTTP server (one-time, daemon thread) ──
        def _find_free_port():
            with socket.socket() as s:
                s.bind(('', 0))
                return s.getsockname()[1]

        _http_port = _find_free_port()
        _handler = http.server.SimpleHTTPRequestHandler

        _bt_cache = {}
        _bt_running = set()

        def _standalone_bt(strategy_type, days, sl_mult):
            _bt_running.add(strategy_type)
            try:
                from StrategyBacktester import OptionStrategyBacktester
                _bt  = OptionStrategyBacktester()
                _rpt = _bt.run(strategy_type, days=days, stop_loss_mult=sl_mult)
                _html_out = _rpt.to_html()
                _bt_cache[strategy_type] = _html_out
                _out_path = os.path.join(dashboard_dir, f'bt_{strategy_type}.html')
                with open(_out_path, 'w', encoding='utf-8') as _f:
                    _f.write(_html_out)
            except Exception as _e:
                _bt_cache[strategy_type] = f'<p style="color:#ff4444;">Backtest error: {_e}</p>'
            finally:
                _bt_running.discard(strategy_type)

        def _launch_backtest(strategy_type, days, sl_mult):
            if strategy_type not in _bt_running:
                import threading as _thr
                _thr.Thread(
                    target=_standalone_bt,
                    args=(strategy_type, days, sl_mult),
                    daemon=True
                ).start()

        class _QuietHandler(_handler):
            """Suppress request log spam, and handle POST for backtesting."""
            def log_message(self, *args): pass
            def log_request(self, *args): pass
            
            def do_POST(self):
                if self.path == '/bt_run':
                    content_len = int(self.headers.get('Content-Length', 0))
                    post_body = self.rfile.read(content_len)
                    try:
                        import json
                        data = json.loads(post_body)
                        _launch_backtest(data.get('type'), data.get('days', 365), data.get('sl', 2.0))
                        self.send_response(200)
                        self.send_header('Content-type', 'text/plain')
                        self.end_headers()
                        self.wfile.write(b"OK")
                    except Exception as e:
                        self.send_response(500)
                        self.end_headers()
                        self.wfile.write(str(e).encode())
                else:
                    self.send_response(404)
                    self.end_headers()

        _httpd = http.server.ThreadingHTTPServer(('127.0.0.1', _http_port), _QuietHandler)
        _httpd.timeout = 0.5
        # chdir to dashboard directory so the HTTP server serves the right files
        _orig_dir = os.getcwd()
        os.chdir(dashboard_dir)
        _http_thread = threading.Thread(
            target=lambda: _httpd.serve_forever(), daemon=True)
        _http_thread.start()
        _base_url = f'http://127.0.0.1:{_http_port}'
        print(f"  Local server: {_base_url}/")

        first_run = True
        DARK_BG = '#0f0f19'
        CARD_BG = '#1a1a2e'
        ACCENT = '#4fc3f7'
        RED = '#ff4444'
        GREEN = '#66bb6a'
        YELLOW = '#ffd54f'
        WHITE = '#e0e0e0'
        MUTED = '#888'
        colors = ['#4fc3f7', '#ff7043', '#66bb6a', '#ab47bc', '#ffa726', '#ef5350']

        # ── ONE-TIME DATA (cached across refreshes) ──
        vol_data = {}
        hist_cache = None
        breach_cache = None
        intra_closes = []
        baseline_oi = None
        heston_cache = {'params': None, 'ts': 0, 'ttl': 300}  # 5-min TTL
        prev_oi = None
        oi_time_series = []            # [(timestamp_str, net_ce_chg, net_pe_chg)]
        oi_velocity_history = []       # List of (timestamp, snapshot) for 15-min sliding window
        momentum_data = {'vwap': 0, 'ema': 0, 'status': 'N/A'} # Intraday price momentum
        oi_pressure = 'NEUTRAL'       # Directional OI pressure signal
        oi_pressure_score = 0         # Magnitude (-100 bearish to +100 bullish)
        # IV Trend history removed via User Request
        regime_history = []  # [(timestamp_str, regime, ml_prob), ...] — regime transition tracking
        _last_regime = None  # track previous regime for change detection
        _regime_changed_at = ''  # timestamp string of last regime change


        def _fetch_history_once():
            nonlocal hist_cache
            if hist_cache is not None:
                return hist_cache
            try:
                today = datetime.now()
                start = today - pd.Timedelta(days=365)
                data = {"symbol": self.symbol, "resolution": "D", "date_format": "1",
                        "range_from": start.strftime("%Y-%m-%d"),
                        "range_to": today.strftime("%Y-%m-%d"), "cont_flag": "1"}
                r = self.fyers.history(data=data)
                if r.get('s') == 'ok':
                    candles = r['candles']
                    hist_cache = {
                        'closes': [c[4] for c in candles],
                        'highs': [c[2] for c in candles],
                        'lows': [c[3] for c in candles],
                        'opens': [c[1] for c in candles]
                    }
                else:
                    hist_cache = None
            except:
                hist_cache = None
            return hist_cache

        def _fetch_intraday_baseline():
            nonlocal intra_closes, momentum_data
            try:
                today_str = datetime.now().strftime("%Y-%m-%d")
                data = {"symbol": self.symbol, "resolution": "1", "date_format": "1",
                        "range_from": today_str, "range_to": today_str, "cont_flag": "1"}
                r = self.fyers.history(data=data)
                if r.get('s') == 'ok':
                    candles = r['candles']
                    # Candle structure: [timestamp, open, high, low, close, volume]
                    intra_closes = [float(c[4]) for c in candles]
                    
                    if len(candles) > 0:
                        highs = [float(c[2]) for c in candles]
                        lows = [float(c[3]) for c in candles]
                        day_high = max(highs)
                        day_low = min(lows)

                        # VWAP: sum(typical_price * volume) / sum(volume)
                        vsum = 0
                        pv_sum = 0
                        for c in candles:
                            tp = (float(c[2]) + float(c[3]) + float(c[4])) / 3.0
                            vol = float(c[5])
                            pv_sum += tp * vol
                            vsum += vol
                        
                        vwap = pv_sum / vsum if vsum > 0 else float(candles[-1][4])
                        
                        # 9-period EMA
                        ema = 0
                        if len(intra_closes) >= 9:
                            ema = pd.Series(intra_closes).ewm(span=9, adjust=False).mean().iloc[-1]
                        else:
                            ema = vwap
                        
                        spot_now = intra_closes[-1]
                        status = "LONG" if spot_now > vwap and spot_now > ema else "SHORT" if spot_now < vwap and spot_now < ema else "NEUTRAL"
                        
                        momentum_data = {
                            'vwap': round(vwap, 2),
                            'ema': round(ema, 2),
                            'status': status,
                            'day_high': day_high,
                            'day_low': day_low
                        }
            except:
                pass

        def _compute_vol_intelligence(spot, momentum_data):
            """Compute vol cone / regime / VRP data. Returns dict."""
            nonlocal _last_regime, _regime_changed_at
            hist = _fetch_history_once()
            if not hist or len(hist['closes']) < 40:
                return None
            closes = hist['closes']
            highs = hist['highs']
            lows = hist['lows']

            rolling_hv = self.analytics.calculate_rolling_historical_volatility(closes, window=20)
            parkinson_hv = self.analytics.calculate_parkinson_volatility(highs, lows, window=20)
            if rolling_hv.empty or len(rolling_hv) < 40:
                return None

            current_hv = rolling_hv.iloc[-1]
            current_park = parkinson_hv.iloc[-1] if not parkinson_hv.empty else 0
            mean_hv = rolling_hv.mean()
            min_hv = rolling_hv.min()
            max_hv = rolling_hv.max()
            hv_percentile = (rolling_hv < current_hv).mean() * 100

            bb_window = 20
            hv_sma = rolling_hv.rolling(bb_window).mean()
            hv_std = rolling_hv.rolling(bb_window).std()
            bb_upper = hv_sma + 2 * hv_std
            bb_lower = hv_sma - 2 * hv_std
            bandwidth = ((bb_upper - bb_lower) / hv_sma * 100).dropna()
            current_bw = bandwidth.iloc[-1] if not bandwidth.empty else 0

            hv_slope = 0
            if len(rolling_hv) >= 5:
                recent_5 = rolling_hv.iloc[-5:].values
                hv_slope = recent_5[-1] - recent_5[0]

            iv_velocity_5d = rolling_hv.diff(5).iloc[-1] if len(rolling_hv) >= 6 else 0
            iv_velocity_10d = rolling_hv.diff(10).iloc[-1] if len(rolling_hv) >= 11 else 0
            iv_accel = iv_velocity_5d - (iv_velocity_10d / 2)

            hv_changes = rolling_hv.diff().dropna()
            vov = hv_changes.tail(20).std() if len(hv_changes) >= 20 else 0

            bw_percentile = (bandwidth < current_bw).mean() * 100 if not bandwidth.empty else 50

            # --- NEW STATISTICAL REGIME SYSTEM ---
            # Uses Bandwidth Percentile (coiling) and HV Trend + INTRADAY OVERLAY
            ml_prob = 0.50  # Kept for compatibility with return dict
            
            # Intraday momentum override
            m_stat = momentum_data.get('status', 'NEUTRAL') if momentum_data else 'NEUTRAL'
            
            # If intraday trend is strong and volume/slope matches
            if m_stat in ['LONG', 'SHORT'] and (iv_accel > 0 or hv_slope > 0):
                regime = "EXPANSION (LIVE)"
                regime_desc = f"Vol exploding intraday! Spot > VWAP/EMA. (BBw: {bw_percentile:.0f}th)"
            elif m_stat != 'NEUTRAL' and bw_percentile < 30:
                regime = "MOMENTUM / TRENDING"
                regime_desc = f"Trending directional breakout in progress."
            elif bw_percentile < 25:
                regime = "COMPRESSION"
                regime_desc = f"Vol tightly coiled (BBw: {bw_percentile:.0f}th) — Breakout likely"
            elif bw_percentile > 75 and hv_slope > 0:
                regime = "EXPANSION"
                regime_desc = f"Vol exploding (HV rising, BBw expanding)"
            elif hv_slope < -0.5:
                regime = "MEAN_REVERSION"
                regime_desc = f"Vol exhaustion (HV slope dropping)"
            else:
                regime = "NORMAL"
                regime_desc = "Standard vol flow — No extreme edge"

            # Track regime history for transition detection
            _now_ts = datetime.now().strftime('%H:%M:%S')
            if _last_regime is not None and regime != _last_regime:
                _regime_changed_at = _now_ts
            regime_history.append((_now_ts, regime, ml_prob))
            if len(regime_history) > 500:
                regime_history.pop(0)
            _prev_regime = _last_regime or regime
            _last_regime = regime

            # ATM IV from option chain
            iv = 0
            old_exp = self.expiry_date
            self.expiry_date = near_exp
            try:
                d = self.get_option_chain_data()
                _df = self.parse_and_filter(d)
            except:
                _df = pd.DataFrame()
            self.expiry_date = old_exp
            if not _df.empty and spot > 0:
                _df['dist'] = abs(_df['strike'] - spot)
                row = _df.loc[_df['dist'].idxmin()]
                T_iv = self.analytics.get_time_to_expiry(near_exp)
                iv = self._ensure_iv(row['iv'], row['price'], row['strike'], T_iv, row['type'])

            vrp = iv - current_hv if iv > 0 else 0
            z_score = self.analytics.calculate_z_score(iv, rolling_hv) if iv > 0 else 0
            # Signal scoring
            signal_metrics = {'vrp': vrp, 'regime': regime}
            signal_result = self._score_signal(signal_metrics)

            # Half-life
            half_life = 0
            try:
                hv_vals = rolling_hv.dropna().values
                if len(hv_vals) > 20:
                    hv_centered = hv_vals - np.mean(hv_vals)
                    autocorr = np.correlate(hv_centered[:-1], hv_centered[1:], mode='valid')[0]
                    autocorr /= np.correlate(hv_centered[:-1], hv_centered[:-1], mode='valid')[0]
                    if 0 < autocorr < 1:
                        half_life = np.log(2) / (-np.log(autocorr))
            except:
                pass

            # Build forecast rows from regime_history trend
            forecast_rows_html = ''
            if len(regime_history) >= 3:
                _recent_probs = [x[2] for x in regime_history[-10:]]
                _prob_trend = (_recent_probs[-1] - _recent_probs[0]) / max(1, len(_recent_probs))
                _horizons = [('5 min', 5), ('15 min', 15), ('30 min', 30)]
                _rows = []
                for _h_label, _h_steps in _horizons:
                    _est_p = max(0.01, min(0.99, ml_prob + _prob_trend * _h_steps))
                    _est_regime = 'COMPRESSION' if _est_p < 0.35 else 'EXPANSION' if _est_p > 0.65 else 'MEAN_REVERSION'
                    _rc = '#66bb6a' if _est_p < 0.35 else '#ef5350' if _est_p > 0.65 else '#ffa726'
                    _rows.append(f'<tr><td style="padding:4px 8px;color:#aaa;">{_h_label}</td>'
                                 f'<td style="padding:4px 8px;text-align:center;color:{_rc};font-weight:700;">{_est_p:.0%}</td>'
                                 f'<td style="padding:4px 8px;text-align:center;color:{_rc};">{_est_regime}</td></tr>')
                forecast_rows_html = ''.join(_rows)
            else:
                forecast_rows_html = '<tr><td colspan="3" style="padding:4px 8px;color:#555;">Collecting data...</td></tr>'

            return {
                'current_hv': current_hv, 'parkinson': current_park, 'mean_hv': mean_hv,
                'min_hv': min_hv, 'max_hv': max_hv, 'hv_percentile': hv_percentile,
                'regime': regime, 'regime_desc': regime_desc, 'ml_prob': ml_prob, 'bandwidth': current_bw,
                'bw_percentile': bw_percentile, 'hv_slope': hv_slope,
                'prev_regime': _prev_regime, 'regime_changed_at': _regime_changed_at,
                'forecast_rows': forecast_rows_html,
                'iv_velocity_5d': iv_velocity_5d, 'iv_accel': iv_accel, 'vov': vov,
                'iv': iv, 'vrp': vrp, 'z_score': z_score, 'half_life': half_life,
                'signal': signal_result, 'rolling_hv': rolling_hv
            }

        def _compute_vrp_data(spot, df_chain):
            """Compute VRP monitor data. Returns dict."""
            nonlocal intra_closes
            if spot > 0:
                intra_closes.append(spot)
                if len(intra_closes) > 300:
                    intra_closes.pop(0)

            rv_intra = 0
            if len(intra_closes) > 10:
                s = pd.Series(intra_closes)
                rets = np.log(s / s.shift(1)).dropna()
                rv_intra = rets.tail(20).std() * np.sqrt(252 * 375) * 100

            # Daily HV baseline
            daily_hv = 0
            hist = _fetch_history_once()
            if hist and len(hist['closes']) > 20:
                s = pd.Series(hist['closes'])
                rets = np.log(s / s.shift(1)).dropna()
                daily_hv = rets.tail(20).std() * np.sqrt(252) * 100

            atm_iv = 0
            skew = 0
            if not df_chain.empty and spot > 0:
                df_chain['dist'] = abs(df_chain['strike'] - spot)
                atm_row = df_chain.loc[df_chain['dist'].idxmin()]
                T_intra = self.analytics.get_time_to_expiry(near_exp)
                atm_iv = self._ensure_iv(atm_row['iv'], atm_row['price'], atm_row['strike'], T_intra, atm_row['type'])
                try:
                    pe_df = df_chain[df_chain['type'] == 'PE']
                    ce_df = df_chain[df_chain['type'] == 'CE']
                    p_iv = pe_df.iloc[(pe_df['strike'] - spot * 0.95).abs().argmin()]['iv'] if not pe_df.empty else atm_iv
                    c_iv = ce_df.iloc[(ce_df['strike'] - spot * 1.05).abs().argmin()]['iv'] if not ce_df.empty else atm_iv
                    skew = p_iv - c_iv
                except:
                    skew = 0

            daily_vrp = atm_iv - daily_hv if daily_hv > 0 else 0
            intra_vrp = atm_iv - rv_intra

            sig = "WAIT"
            if daily_vrp > 5.0:
                sig = "SELL VOL (High Prem)"
                if intra_vrp > 5.0: sig = "STRONG SELL"
            elif daily_vrp < -2.0:
                sig = "BUY VOL (Cheap)"

            return {
                'atm_iv': atm_iv, 'daily_hv': daily_hv, 'rv_intra': rv_intra,
                'daily_vrp': daily_vrp, 'intra_vrp': intra_vrp, 'skew': skew, 'signal': sig
            }

        def _compute_seller_data(spot, df_chain):
            """Compute seller advisor data. Returns dict."""
            nonlocal baseline_oi, prev_oi, breach_cache, oi_velocity_history, oi_pressure, oi_pressure_score

            T = self.analytics.get_time_to_expiry(near_exp)
            if T <= 0: return None
            DTE = max(1, int(T * 365))

            if df_chain.empty or spot <= 0: return None

            df_chain['dist'] = abs(df_chain['strike'] - spot)
            ce_df = df_chain[df_chain['type'] == 'CE'].copy()
            pe_df = df_chain[df_chain['type'] == 'PE'].copy()
            atm_strike = df_chain.loc[df_chain['dist'].idxmin(), 'strike']

            ce_atm = ce_df[ce_df['strike'] >= spot]
            if ce_atm.empty: ce_atm = ce_df
            atm_ce = ce_atm.loc[ce_atm['dist'].idxmin()] if not ce_atm.empty else None

            pe_atm = pe_df[pe_df['strike'] <= spot]
            if pe_atm.empty: pe_atm = pe_df
            atm_pe = pe_atm.loc[pe_atm['dist'].idxmin()] if not pe_atm.empty else None

            ce_price = atm_ce['price'] if atm_ce is not None else 0
            pe_price = atm_pe['price'] if atm_pe is not None else 0

            atm_iv = 0
            if atm_ce is not None:
                atm_iv = self._ensure_iv(atm_ce['iv'], ce_price, atm_ce['strike'], T, 'CE')
            if atm_iv <= 0 and atm_pe is not None:
                atm_iv = self._ensure_iv(atm_pe['iv'], pe_price, atm_pe['strike'], T, 'PE')

            sigma = atm_iv / 100 if atm_iv > 0 else 0.15
            if ce_price <= 0 and atm_iv > 0:
                ce_price = self.analytics.black_scholes(spot, atm_ce['strike'] if atm_ce is not None else atm_strike, T, 0.07, sigma, 'CE')
            if pe_price <= 0 and atm_iv > 0:
                pe_price = self.analytics.black_scholes(spot, atm_pe['strike'] if atm_pe is not None else atm_strike, T, 0.07, sigma, 'PE')

            straddle = ce_price + pe_price
            em_pct = (straddle / spot) * 100

            # --- 15-MINUTE VELOCITY TRACKER ---
            current_oi = {}
            for _, orow in df_chain.iterrows():
                key = (int(round(orow['strike'])), orow['type'])
                current_oi[key] = int(orow['oi'])

            _now_epoch = time.time()
            oi_velocity_history.append((_now_epoch, dict(current_oi)))
            # Keep 15m window
            oi_velocity_history = [x for x in oi_velocity_history if _now_epoch - x[0] <= 900]
            velocity_baseline = oi_velocity_history[0][1] if oi_velocity_history else current_oi

            all_strikes = sorted(df_chain['strike'].unique())
            all_int = [int(round(s)) for s in all_strikes]
            atm_idx = min(range(len(all_int)), key=lambda i: abs(all_int[i] - spot))
            nearby = all_int[max(0, atm_idx - 8): atm_idx + 9]

            if baseline_oi is None:
                baseline_oi = dict(current_oi)
            
            call_vel_added = 0
            call_vel_unwound = 0
            put_vel_added = 0
            put_vel_unwound = 0

            chain_rows = []
            for strike in nearby:
                ce_oi = current_oi.get((strike, 'CE'), 0)
                ce_base = baseline_oi.get((strike, 'CE'), 0)
                ce_vel_base = velocity_baseline.get((strike, 'CE'), 0)
                
                pe_oi = current_oi.get((strike, 'PE'), 0)
                pe_base = baseline_oi.get((strike, 'PE'), 0)
                pe_vel_base = velocity_baseline.get((strike, 'PE'), 0)

                ce_chg = ce_oi - ce_base     # Cumulative
                pe_chg = pe_oi - pe_base
                ce_vel = ce_oi - ce_vel_base # 15-min velocity
                pe_vel = pe_oi - pe_vel_base

                is_atm = abs(strike - spot) < 60
                chain_rows.append({
                    'strike': strike, 'ce_oi': ce_oi, 'pe_oi': pe_oi,
                    'ce_chg': ce_chg, 'pe_chg': pe_chg, 
                    'ce_vel': ce_vel, 'pe_vel': pe_vel, 
                    'is_atm': is_atm
                })
                
                if strike >= spot:
                    if ce_vel > 0: call_vel_added += ce_vel
                    elif ce_vel < 0: call_vel_unwound += abs(ce_vel)
                if strike <= spot:
                    if pe_vel > 0: put_vel_added += pe_vel
                    elif pe_vel < 0: put_vel_unwound += abs(pe_vel)

            # Compute Velocity Pressure
            total_force = (put_vel_added + call_vel_unwound) + (call_vel_added + put_vel_unwound)
            if total_force > 0:
                oi_pressure_score = (((put_vel_added + call_vel_unwound) - (call_vel_added + put_vel_unwound)) / total_force) * 100
            else:
                oi_pressure_score = 0
            
            oi_pressure = "BULLISH" if oi_pressure_score > 15 else "BEARISH" if oi_pressure_score < -15 else "NEUTRAL"

            # Max Pain + Walls (top-2 each side)
            max_pain = 0
            call_wall = 0
            put_wall  = 0
            call_wall_2 = 0
            put_wall_2  = 0
            pcr_val = 0
            try:
                from KeyLevelsEngine import KeyLevelsEngine
                kle = KeyLevelsEngine()
                max_pain = kle.calculate_max_pain(df_chain)
                walls = kle.calculate_oi_walls(df_chain, spot)
                call_wall   = walls['call_wall']
                put_wall    = walls['put_wall']
                call_wall_2 = walls.get('call_wall_2', 0)
                put_wall_2  = walls.get('put_wall_2', 0)
                pcr_val = kle.calculate_pcr(df_chain)
            except:
                pass

            # Strike selection
            strike_rows = []
            r_rate = 0.07
            for strike in nearby:
                for otype in ['CE', 'PE']:
                    row_df = df_chain[(df_chain['strike'] == strike) & (df_chain['type'] == otype)]
                    if row_df.empty: continue
                    row = row_df.iloc[0]
                    price = row['price']
                    if price < 0.5: continue
                    dist = strike - spot
                    if otype == 'CE' and dist < 0: continue
                    if otype == 'PE' and dist > 0: continue
                    try:
                        d1 = (np.log(spot / strike) + (r_rate + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
                        d2 = d1 - sigma * np.sqrt(T)
                        prob_otm = (norm.cdf(-d2) if otype == 'CE' else norm.cdf(d2)) * 100
                    except:
                        prob_otm = 50
                    try:
                        row_iv = self._ensure_iv(row['iv'], price, strike, T, otype)
                        g_iv = row_iv / 100 if row_iv > 0 else sigma
                        greeks = self.analytics.calculate_greeks(spot, strike, T, r_rate, g_iv, otype)
                        theta = greeks.get('theta', 0)
                    except:
                        theta = 0
                    signal = "★ SAFE" if prob_otm >= 85 else "✓ GOOD" if prob_otm >= 75 else "~ OK" if prob_otm >= 60 else "✗ RISKY"
                    strike_rows.append({'strike': strike, 'type': otype, 'dist': dist,
                                        'prob_otm': prob_otm, 'price': price, 'theta': theta, 'signal': signal})

            # Sell zones (based on EM + nearest wall)
            safe_ce = min(call_wall, spot + straddle) if call_wall > 0 and straddle > 0 else spot + straddle
            safe_pe = max(put_wall, spot - straddle)  if put_wall  > 0 and straddle > 0 else spot - straddle

            return {
                'DTE': DTE, 'atm_iv': atm_iv, 'straddle': straddle, 'em': straddle, 'em_pct': em_pct,
                'max_pain': max_pain,
                'call_wall': call_wall, 'put_wall': put_wall,
                'call_wall_2': call_wall_2, 'put_wall_2': put_wall_2,
                'sell_ce_above': safe_ce, 'sell_pe_below': safe_pe,
                'pcr': pcr_val,
                'chain_rows': chain_rows, 'strike_rows': strike_rows,
                'oi_pressure': oi_pressure, 'oi_pressure_score': oi_pressure_score
            }

        def _calibrate_heston(spot, df_chain, T_near):
            """Calibrate Heston params from live option chain. Cached 5 min."""
            nonlocal heston_cache
            now = time.time()
            if heston_cache['params'] and (now - heston_cache['ts']) < heston_cache['ttl']:
                return heston_cache['params']
            if df_chain.empty or spot <= 0 or T_near <= 0:
                return heston_cache['params']  # return stale if available
            try:
                from scipy.optimize import minimize as sp_minimize
                ce_df = df_chain[df_chain['type'] == 'CE'].copy()
                if ce_df.empty:
                    return heston_cache['params']
                ce_df['dist'] = abs(ce_df['strike'] - spot)
                # Use ~10 strikes around ATM for calibration
                cal_df = ce_df.nsmallest(10, 'dist')
                cal_df = cal_df[cal_df['price'] > 0.5]
                if len(cal_df) < 3:
                    return heston_cache['params']
                avg_iv = cal_df['iv'].mean() / 100.0
                v0_guess = max(0.005, avg_iv ** 2)
                r = 0.07
                initial = [2.0, v0_guess, v0_guess, -0.7, 0.3]
                bounds = [(0.1, 10.0), (0.001, 0.5), (0.001, 0.5), (-0.99, 0.99), (0.01, 5.0)]
                def objective(params):
                    k, th, v, rh, x = params
                    err = 0.0
                    for _, row in cal_df.iterrows():
                        try:
                            mp = HestonMath.price_vanilla_call(spot, row['strike'], T_near, r, k, th, v, rh, x)
                            if row['price'] > 0:
                                err += ((mp - row['price']) / row['price']) ** 2
                        except:
                            err += 1e4
                    return err
                result = sp_minimize(objective, initial, bounds=bounds, method='L-BFGS-B',
                                     options={'maxiter': 80, 'ftol': 1e-6})
                if result.success or result.fun < 10:
                    p = result.x
                    heston_cache['params'] = {'kappa': p[0], 'theta': p[1], 'v0': p[2], 'rho': p[3], 'xi': p[4]}
                    heston_cache['ts'] = now
                    print(f"  [Heston] Calibrated: κ={p[0]:.2f} θ={p[1]:.4f} v₀={p[2]:.4f} ρ={p[3]:.2f} ξ={p[4]:.2f}")
            except Exception as e:
                print(f"  [Heston] Calibration failed: {e}")
            return heston_cache['params']

        def _heston_mc_density(spot, params, atm_iv):
            """Run 50k Heston MC paths for Day 0/7/14 → PDF + stats."""
            if not params or spot <= 0:
                return None
            r = 0.07
            kappa, theta, v0, rho, xi = params['kappa'], params['theta'], params['v0'], params['rho'], params['xi']
            near_T = self.analytics.get_time_to_expiry(near_exp)
            near_days = max(1, int(near_T * 365))
            horizons = [
                {'label': f'Heston DTE {near_days}', 'days': near_days, 'color': ACCENT},
                {'label': 'Heston Day 7',  'days': 7,  'color': '#ff7043'},
                {'label': 'Heston Day 14', 'days': 14, 'color': '#66bb6a'},
            ]
            N_PATHS = 50000
            results = []
            for h in horizons:
                T = h['days'] / 365.0
                if T <= 0:
                    continue
                steps = max(10, h['days'] * 2)
                dt = T / steps
                # Vectorized Heston MC
                Z1 = np.random.normal(size=(N_PATHS, steps))
                Z3 = np.random.normal(size=(N_PATHS, steps))
                Z2 = rho * Z1 + np.sqrt(1 - rho**2) * Z3
                S = np.full(N_PATHS, spot)
                v = np.full(N_PATHS, v0)
                for t in range(steps):
                    v_pos = np.maximum(v, 0)
                    dS = (r - 0.5 * v_pos) * dt + np.sqrt(v_pos) * np.sqrt(dt) * Z1[:, t]
                    S = S * np.exp(dS)
                    dv = kappa * (theta - v_pos) * dt + xi * np.sqrt(v_pos) * np.sqrt(dt) * Z2[:, t]
                    v = v + dv
                # Terminal distribution → PDF via histogram
                terminal = S
                lo = np.percentile(terminal, 0.5)
                hi = np.percentile(terminal, 99.5)
                prices = np.linspace(lo, hi, 300)
                counts, edges = np.histogram(terminal, bins=300, range=(lo, hi), density=True)
                bin_centers = (edges[:-1] + edges[1:]) / 2
                # Stats from MC paths
                one_sigma_lo = np.percentile(terminal, 15.87)  # ~-1σ
                one_sigma_hi = np.percentile(terminal, 84.13)  # ~+1σ
                expected_move = (one_sigma_hi - one_sigma_lo) / 2
                prob_above = (terminal > spot).mean() * 100
                skewness = float(pd.Series(terminal).skew())
                kurtosis_val = float(pd.Series(terminal).kurtosis())
                results.append({
                    'label': h['label'], 'days': h['days'], 'color': h['color'],
                    'prices': bin_centers, 'pdf': counts,
                    'one_sigma_lo': one_sigma_lo, 'one_sigma_hi': one_sigma_hi,
                    'expected_move': expected_move,
                    'expected_move_pct': (expected_move / spot) * 100,
                    'prob_above': prob_above,
                    'skewness': skewness, 'kurtosis': kurtosis_val,
                    'terminal': terminal,  # keep for MC sim tab
                })
            return results if results else None

        def _compute_bsm_density(spot, atm_iv):
            """BSM log-normal density for comparison overlay."""
            if spot <= 0 or atm_iv <= 0:
                return None
            sigma_ann = atm_iv / 100
            r = 0.07
            near_T = self.analytics.get_time_to_expiry(near_exp)
            near_days = max(1, int(near_T * 365))
            horizons = [
                {'label': f'BSM DTE {near_days}', 'days': near_days, 'color': ACCENT},
                {'label': 'BSM Day 7',  'days': 7,  'color': '#ff7043'},
                {'label': 'BSM Day 14', 'days': 14, 'color': '#66bb6a'},
            ]
            results = []
            for h in horizons:
                T = h['days'] / 365.0
                if T <= 0:
                    continue
                sigma_t = sigma_ann * np.sqrt(T)
                mu = np.log(spot) + (r - 0.5 * sigma_ann**2) * T
                lo = spot * np.exp(-3.5 * sigma_t)
                hi = spot * np.exp(3.5 * sigma_t)
                prices = np.linspace(lo, hi, 300)
                pdf = (1 / (prices * sigma_t * np.sqrt(2 * np.pi))) * np.exp(
                    -((np.log(prices) - mu) ** 2) / (2 * sigma_t ** 2)
                )
                one_sigma_lo = spot * np.exp(-sigma_t)
                one_sigma_hi = spot * np.exp(sigma_t)
                expected_move = spot * (np.exp(sigma_t) - 1)
                prob_above = 1 - norm.cdf((np.log(spot) - mu) / sigma_t)
                results.append({
                    'label': h['label'], 'days': h['days'], 'color': h['color'],
                    'prices': prices, 'pdf': pdf,
                    'one_sigma_lo': one_sigma_lo, 'one_sigma_hi': one_sigma_hi,
                    'expected_move': expected_move,
                    'expected_move_pct': (expected_move / spot) * 100,
                    'prob_above': prob_above * 100,
                })
            return results if results else None

        def _compute_mc_simulation(spot, params):
            """Full MC simulation: fan chart, VaR/CVaR, vol path. 50k paths."""
            if not params or spot <= 0:
                return None
            r = 0.07
            kappa, theta, v0, rho, xi = params['kappa'], params['theta'], params['v0'], params['rho'], params['xi']
            near_T = self.analytics.get_time_to_expiry(near_exp)
            sim_days = 14
            T = sim_days / 365.0
            steps = sim_days * 2  # 2 steps per day
            N_PATHS = 50000
            dt = T / steps
            # Store full path history
            Z1 = np.random.normal(size=(N_PATHS, steps))
            Z3 = np.random.normal(size=(N_PATHS, steps))
            Z2 = rho * Z1 + np.sqrt(1 - rho**2) * Z3
            S_all = np.zeros((N_PATHS, steps + 1))
            v_all = np.zeros((N_PATHS, steps + 1))
            S_all[:, 0] = spot
            v_all[:, 0] = v0
            for t in range(steps):
                v_pos = np.maximum(v_all[:, t], 0)
                dS = (r - 0.5 * v_pos) * dt + np.sqrt(v_pos) * np.sqrt(dt) * Z1[:, t]
                S_all[:, t+1] = S_all[:, t] * np.exp(dS)
                dv = kappa * (theta - v_pos) * dt + xi * np.sqrt(v_pos) * np.sqrt(dt) * Z2[:, t]
                v_all[:, t+1] = v_all[:, t] + dv
            # Time axis in days
            time_days = np.linspace(0, sim_days, steps + 1)
            # Percentile bands
            pctiles = {}
            for p in [5, 10, 25, 50, 75, 90, 95]:
                pctiles[p] = np.percentile(S_all, p, axis=0)
            # Terminal stats
            terminal = S_all[:, -1]
            returns = (terminal - spot) / spot * 100
            var_1 = np.percentile(returns, 1)
            var_5 = np.percentile(returns, 5)
            cvar_1 = returns[returns <= var_1].mean() if (returns <= var_1).any() else var_1
            cvar_5 = returns[returns <= var_5].mean() if (returns <= var_5).any() else var_5
            max_dd = np.min((np.min(S_all, axis=1) - spot) / spot * 100)
            # Vol path stats
            vol_pctiles = {}
            vol_ann = np.sqrt(np.maximum(v_all, 0)) * 100  # annualized vol %
            for p in [25, 50, 75]:
                vol_pctiles[p] = np.percentile(vol_ann, p, axis=0)
            return {
                'time_days': time_days, 'pctiles': pctiles, 'terminal': terminal,
                'var_1': var_1, 'var_5': var_5, 'cvar_1': cvar_1, 'cvar_5': cvar_5,
                'max_dd': max_dd, 'prob_up': (terminal > spot).mean() * 100,
                'mean_ret': returns.mean(), 'vol_pctiles': vol_pctiles, 'vol_time': time_days,
                'median_terminal': np.median(terminal),
                'sim_days': sim_days, 'n_paths': N_PATHS,
            }

        # ── Intraday baseline will be fetched inside the loop ──

        try:
            while True:
                try:
                    # ── SHARED FETCH ──
                    self.get_spot_price()
                    spot = self.spot_price
                    if spot <= 0:
                        print("  Waiting for spot...")
                        time.sleep(3)
                        continue
                    now_str = datetime.now().strftime('%H:%M:%S')
                    _now_epoch = time.time()

                    # ── FRESH INTRADAY DATA ──
                    _fetch_intraday_baseline()

                    # ── IV SURFACE DATA (Tab 1) ──
                    points = self._iv_dashboard_fetch(expiries)
                    pred = self._iv_predict_spot(points, expiries) if points else {
                        'direction': 'NO DATA', 'confidence': 0, 'expected_move': 0,
                        'expected_move_pct': 0, 'skew_signal': 'N/A', 'skew_ratio': 1.0,
                        'term_signal': 'N/A', 'term_spread': 0, 'put_iv_avg': 0,
                        'call_iv_avg': 0, 'atm_iv': 0, 'anomalous_strikes': [],
                        'action': 'WAIT', 'strategy': 'Insufficient data',
                        'near_atm': 0, 'far_atm': 0, 'skew_delta': 0
                    }

                    # ── OPTION CHAIN FOR NEAR EXPIRY (Tab 3, 4) ──
                    old_exp = self.expiry_date
                    self.expiry_date = near_exp
                    try:
                        raw_chain = self.get_option_chain_data()
                        df_chain = self.parse_and_filter(raw_chain)
                        if not df_chain.empty:
                            _tick_db.record_snapshot(self.symbol, df_chain)
                    except:
                        df_chain = pd.DataFrame()
                    self.expiry_date = old_exp

                    # ── COMPUTE ALL TABS ──
                    vol_intel = _compute_vol_intelligence(spot, momentum_data if 'momentum_data' in locals() else None)
                    seller = _compute_seller_data(spot, df_chain.copy() if not df_chain.empty else df_chain)
                    # Track IV internally without saving a massive array
                    _live_iv = (vol_intel or {}).get('iv', 0)
                    _live_hv  = (vol_intel or {}).get('current_hv', 0)
                    
                    # ATM IV for prob density (prefer vol intel)
                    _pd_iv = _live_iv
                    # Heston calibration + MC density
                    T_near = self.analytics.get_time_to_expiry(near_exp)
                    heston_params = _calibrate_heston(spot, df_chain.copy() if not df_chain.empty else df_chain, T_near)
                    prob_density = _heston_mc_density(spot, heston_params, _pd_iv)
                    bsm_density = _compute_bsm_density(spot, _pd_iv)
                    # Fall back to BSM if Heston fails
                    if not prob_density:
                        prob_density = bsm_density
                        bsm_density = None  # no overlay needed

                    # ── REGIME SNAPSHOT ──
                    hist_cache = _fetch_history_once()
                    df_daily = pd.DataFrame(hist_cache) if hist_cache else pd.DataFrame()
                    rv_intra = (vol_intel or {}).get('rv_intra', 0)
                    momentum_vwap = momentum_data.get('vwap', 0) if 'momentum_data' in locals() else 0
                    regime_snapshot = self.regime_engine.get_regime_snapshot(spot, df_daily, _pd_iv, momentum_vwap, rv_intra)

                    # ── CONFLUENCE VERDICT ──
                    verdict_data = self.confluence_engine.evaluate(regime_snapshot, pred, seller, momentum_data if 'momentum_data' in locals() else None)

                    # ── BUYER SETUP ──
                    gex_accel = 0.0 # Will compute if needed, or default
                    buyer_setup = None
                    if not df_chain.empty and regime_snapshot:
                        buyer_setup = self.buyer_engine.generate_trade_setup(
                            confluence_verdict=verdict_data,
                            spot=spot,
                            vwap=momentum_vwap,
                            df_chain=df_chain,
                            gex_acceleration=gex_accel,
                            intraday_regime=regime_snapshot.get('regime', {}).get('name', '')
                        )

                    # ══════════════════════════════════════════
                    #  BUILD HTML
                    # ══════════════════════════════════════════

                    # ── TAB 1: IV SURFACE (reuse existing plotly logic) ──
                    iv_tab_html = ""
                    if points:
                        fig = make_subplots(rows=1, cols=2,
                            specs=[[{"type": "xy"}, {"type": "scene"}]],
                            subplot_titles=[
                                f"IV SMILE  |  {pred['direction']} ({pred['confidence']:.0%})",
                                f"3D SURFACE  |  Term: {pred['term_spread']:+.1f}%"
                            ], horizontal_spacing=0.05)

                        exp_data = {}
                        for p in points:
                            exp = p['expiry']
                            if exp not in exp_data: exp_data[exp] = []
                            exp_data[exp].append(p)

                        for i, exp in enumerate(expiries):
                            data = exp_data.get(exp, [])
                            if not data: continue
                            s_map = {}
                            for x in data:
                                k = x['strike']
                                if k not in s_map: s_map[k] = {'ivs': [], 'types': []}
                                s_map[k]['ivs'].append(x['iv'])
                                s_map[k]['types'].append(x['type'])
                            x_val = sorted(s_map.keys())
                            y_val = [np.mean(s_map[k]['ivs']) for k in x_val]
                            color = colors[i % len(colors)]
                            atm = pred.get('near_atm', 0) if i == 0 else pred.get('far_atm', 0)
                            fig.add_trace(go.Scatter(x=x_val, y=y_val, mode='lines+markers',
                                name=f"{exp} (ATM:{atm:.1f}%)",
                                line=dict(color=color, width=2.5),
                                marker=dict(size=6, color=color)), row=1, col=1)

                        fig.add_vline(x=spot, line_dash="dash", line_color=YELLOW, line_width=2,
                            annotation_text=f"Spot:{spot:.0f}", annotation_font_color=YELLOW,
                            annotation_position="top right", row=1, col=1)
                        if pred['expected_move'] > 0:
                            fig.add_vrect(x0=spot - pred['expected_move'], x1=spot + pred['expected_move'],
                                fillcolor="rgba(255,213,79,0.08)", line_width=0, row=1, col=1)

                        # 3D surface
                        otm = [p for p in points if
                            (p['type'] == 'PE' and p['strike'] <= spot) or
                            (p['type'] == 'CE' and p['strike'] >= spot)]
                        if not otm: otm = points
                        xs = np.array([p['strike'] for p in otm])
                        ys = np.array([p['days'] for p in otm])
                        zs = np.array([p['iv'] for p in otm])
                        types = [p['type'] for p in otm]
                        strike_grid = np.linspace(xs.min(), xs.max(), 50)
                        days_grid = np.linspace(ys.min(), ys.max(), 25)
                        sm, dm = np.meshgrid(strike_grid, days_grid)
                        try:
                            iv_mesh = griddata((xs, ys), zs, (sm, dm), method='cubic')
                            iv_nn = griddata((xs, ys), zs, (sm, dm), method='nearest')
                            iv_mesh = np.where(np.isnan(iv_mesh), iv_nn, iv_mesh)
                        except:
                            iv_mesh = griddata((xs, ys), zs, (sm, dm), method='nearest')
                        fig.add_trace(go.Surface(x=strike_grid, y=days_grid, z=iv_mesh,
                            colorscale='RdYlBu_r', opacity=0.75, showscale=True,
                            colorbar=dict(title='IV%', len=0.7, x=1.01)), row=1, col=2)
                        mc = [RED if t == 'PE' else ACCENT for t in types]
                        fig.add_trace(go.Scatter3d(x=xs, y=ys, z=zs, mode='markers',
                            marker=dict(size=3, color=mc, opacity=0.85), name='Points'), row=1, col=2)

                        fig.update_layout(height=480, width=1380, paper_bgcolor=DARK_BG,
                            plot_bgcolor='rgba(20,20,35,0.8)',
                            font=dict(color=WHITE, family='Inter, sans-serif', size=11),
                            legend=dict(bgcolor='rgba(30,30,50,0.8)', font=dict(size=10), x=0.01, y=0.99),
                            margin=dict(l=50, r=20, t=50, b=30), hovermode='closest',
                            scene=dict(
                                xaxis=dict(title='Strike', backgroundcolor=DARK_BG, gridcolor='#333'),
                                yaxis=dict(title='Days', backgroundcolor=DARK_BG, gridcolor='#333'),
                                zaxis=dict(title='IV%', backgroundcolor=DARK_BG, gridcolor='#333'),
                                camera=dict(eye=dict(x=1.5, y=-1.8, z=0.8)), bgcolor=DARK_BG))
                        fig.update_xaxes(gridcolor='rgba(100,100,100,0.15)', title='Strike', row=1, col=1)
                        fig.update_yaxes(gridcolor='rgba(100,100,100,0.15)', title='IV (%)', row=1, col=1)
                        iv_plotly = fig.to_html(include_plotlyjs=False, full_html=False)

                        # Prediction panel
                        dir_color = GREEN if 'BULL' in pred['direction'] else RED if 'BEAR' in pred['direction'] else YELLOW
                        iv_tab_html = f'''
                        {iv_plotly}
                        <div class="card" style="margin-top:10px;">
                            <div style="color:{ACCENT};font-size:13px;font-weight:700;letter-spacing:2px;margin-bottom:14px;">SPOT MOVEMENT PREDICTOR</div>
                            <div style="display:flex;gap:10px;flex-wrap:wrap;">
                                <div class="metric-box"><div class="metric-label">DIRECTION</div><div style="font-size:22px;font-weight:700;color:{dir_color};">{pred['direction']}</div><div class="metric-sub">Confidence: {pred['confidence']:.0%}</div></div>
                                <div class="metric-box"><div class="metric-label">EXPECTED 1-DAY MOVE</div><div style="font-size:22px;font-weight:700;color:{WHITE};">±{pred['expected_move']:.0f} pts</div><div class="metric-sub">±{pred['expected_move_pct']:.2f}%</div></div>
                                <div class="metric-box"><div class="metric-label">SKEW RATIO</div><div style="font-size:22px;font-weight:700;color:{WHITE};">{pred['skew_ratio']:.3f}</div><div class="metric-sub">{pred['skew_signal']}</div></div>
                                <div class="metric-box"><div class="metric-label">TERM SPREAD</div><div style="font-size:22px;font-weight:700;color:{RED if pred['term_spread']<-1 else GREEN if pred['term_spread']>1 else WHITE};">{pred['term_spread']:+.2f}%</div><div class="metric-sub">{pred['term_signal']}</div></div>
                                <div class="metric-box"><div class="metric-label">ATM IV</div><div style="font-size:22px;font-weight:700;color:{WHITE};">{pred['atm_iv']:.2f}%</div><div class="metric-sub">Put:{pred['put_iv_avg']:.1f}% Call:{pred['call_iv_avg']:.1f}%</div></div>
                            </div>
                        </div>'''
                    else:
                        iv_tab_html = '<div class="card"><p style="color:#888;">Waiting for IV surface data...</p></div>'

                    # ── TAB 2: VOL INTELLIGENCE (with IV Trend chart) ──
                    vol_tab_html = '<div class="card"><p style="color:#888;">Loading vol intelligence...</p></div>'
                    if vol_intel:
                        v = vol_intel
                        sig = v['signal']
                        regime_color = YELLOW if v['regime'] == 'COMPRESSION' else RED if v['regime'] == 'EXPANSION' else GREEN if v['regime'] == 'MEAN REVERSION' else MUTED
                        action_color = GREEN if 'SELL' in sig['action'] else RED if 'BUY' in sig['action'] else YELLOW

                        # Score bars
                        score_bars = ''
                        labels = {'term_structure': 'Term Structure', 'skew': 'Skew', 'vrp': 'VRP', 'regime': 'Regime'}
                        for k, label in labels.items():
                            if k in sig['scores']:
                                sc = sig['scores'][k]
                                pct = sc / 100 * 100
                                bar_color = GREEN if sc > 60 else RED if sc < 40 else YELLOW
                                score_bars += f'<div style="display:flex;align-items:center;gap:10px;margin:4px 0;"><span style="color:{MUTED};width:120px;font-size:12px;">{label}</span><div style="flex:1;height:8px;background:#222;border-radius:4px;"><div style="width:{pct}%;height:100%;background:{bar_color};border-radius:4px;"></div></div><span style="color:{WHITE};font-size:12px;width:30px;">{sc:.0f}</span></div>'

                        # ── Intraday regime signal (live IV z-score + velocity) ──
                        _iv_live      = v.get('iv', 0)
                        _iv_velocity  = v.get('iv_velocity_5d', 0)   # per-refresh IV drift proxy
                        _iv_zscore    = v.get('z_score', 0)           # IV z-score vs HV
                        _vrp_now      = v.get('vrp', 0)

                        # Derive actionable intraday premium-selling signal
                        if _iv_velocity < -0.8 and _iv_zscore < 0.5:
                            _intra_signal = 'SELL-PREMIUM'
                            _intra_color  = GREEN
                            _intra_reason = f'IV falling ({_iv_velocity:+.1f}%) + z-score={_iv_zscore:.1f}x → ideal premium seller environment'
                        elif _iv_velocity > 1.0 or _iv_zscore > 2.0:
                            _intra_signal = 'BUY-PREMIUM / AVOID SELL'
                            _intra_color  = RED
                            _intra_reason = f'IV rising ({_iv_velocity:+.1f}%) or elevated z-score={_iv_zscore:.1f}x → vol expansion risk'
                        elif _vrp_now > 3:
                            _intra_signal = 'SELL-PREMIUM'
                            _intra_color  = GREEN
                            _intra_reason = f'VRP={_vrp_now:+.1f}% → IV rich vs HV, premium sellers have edge'
                        elif _vrp_now < -2:
                            _intra_signal = 'DEFER / REDUCE SIZE'
                            _intra_color  = YELLOW
                            _intra_reason = f'Negative VRP={_vrp_now:+.1f}% → IV cheap, short vol not favored'
                        else:
                            _intra_signal = 'NEUTRAL'
                            _intra_color  = YELLOW
                            _intra_reason = f'Mixed signals — VRP={_vrp_now:+.1f}%, velocity={_iv_velocity:+.1f}%, z={_iv_zscore:.1f}x'

                        # Session phase (IST)
                        _now_hour = datetime.now().hour
                        _now_min  = datetime.now().minute
                        _session = 'PRE-OPEN' if _now_hour < 9 or (_now_hour == 9 and _now_min < 15) \
                            else 'OPENING AUCTION' if _now_hour == 9 and _now_min < 30 \
                            else 'MORNING' if _now_hour < 12 \
                            else 'MIDDAY' if _now_hour < 14 \
                            else 'CLOSING' if _now_hour < 15 or (_now_hour == 15 and _now_min < 30) \
                            else 'POST-MARKET'
                        _session_note = {
                            'OPENING AUCTION': 'Gap fills / reversals common — avoid naked sells at open',
                            'MORNING':  'Best window for straddle/strangle premium selling if IV elevated',
                            'MIDDAY':   'Low volatility window — theta decay favors sellers',
                            'CLOSING':  'Gamma spikes near expiry — hedge open positions',
                            'PRE-OPEN': 'Market closed — reference levels only',
                            'POST-MARKET': 'Market closed',
                        }.get(_session, '')

                        # DTE context
                        _dte_now = seller.get('DTE', 0) if seller else 0
                        _dte_color = RED if _dte_now <= 1 else YELLOW if _dte_now <= 3 else GREEN
                        _dte_note  = 'EXPIRY DAY — Extreme gamma risk!' if _dte_now == 0 \
                            else f'{_dte_now}d to weekly expiry'

                        vol_tab_html = f'''
                        <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;">
                            <div class="card">
                                <div style="color:{ACCENT};font-size:13px;font-weight:700;letter-spacing:2px;margin-bottom:14px;">HV STATISTICS (1-Year)</div>
                                <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:10px;">
                                    <div class="metric-box"><div class="metric-label">20d HV</div><div style="font-size:20px;font-weight:700;color:{WHITE};">{v['current_hv']:.2f}%</div><div class="metric-sub">Parkinson: {v['parkinson']:.2f}%</div></div>
                                    <div class="metric-box"><div class="metric-label">1-Yr Mean</div><div style="font-size:20px;font-weight:700;color:{WHITE};">{v['mean_hv']:.2f}%</div><div class="metric-sub">{v['min_hv']:.1f}% — {v['max_hv']:.1f}%</div></div>
                                    <div class="metric-box"><div class="metric-label">HV Percentile</div><div style="font-size:20px;font-weight:700;color:{YELLOW};">{v['hv_percentile']:.0f}%</div></div>
                                </div>
                                <div style="display:grid;grid-template-columns:1fr 1fr 1fr 1fr;gap:10px;margin-top:12px;">
                                    <div class="metric-box"><div class="metric-label">ATM IV</div><div style="font-size:18px;font-weight:700;color:{WHITE};">{v['iv']:.2f}%</div></div>
                                    <div class="metric-box"><div class="metric-label">VRP</div><div style="font-size:18px;font-weight:700;color:{GREEN if v['vrp']>2 else RED if v['vrp']<-2 else WHITE};">{v['vrp']:+.2f}%</div></div>
                                    <div class="metric-box"><div class="metric-label">Z-Score</div><div style="font-size:18px;font-weight:700;color:{WHITE};">{v['z_score']:.2f}</div></div>
                                    <div class="metric-box"><div class="metric-label">Half-Life</div><div style="font-size:18px;font-weight:700;color:{WHITE};">{v['half_life']:.0f}d</div></div>
                                </div>
                            </div>
                            <div class="card">
                                <div style="color:{ACCENT};font-size:13px;font-weight:700;letter-spacing:2px;margin-bottom:14px;">ML REGIME ENGINE (LSTM)</div>
                                <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-bottom:12px;">
                                    <div class="metric-box"><div class="metric-label">CURRENT REGIME</div><div style="font-size:20px;font-weight:700;color:{regime_color};">{v['regime']}</div><div class="metric-sub">{v['regime_desc']}</div></div>
                                    <div class="metric-box"><div class="metric-label">PRICE MOMENTUM</div><div style="font-size:20px;font-weight:700;color:{GREEN if momentum_data['status']=='LONG' else RED if momentum_data['status']=='SHORT' else YELLOW};">{momentum_data['status']}</div><div class="metric-sub">VWAP: {momentum_data['vwap']} | EMA: {momentum_data['ema']}</div></div>
                                </div>
                                <div style="margin:12px 0 8px;color:{MUTED};font-size:11px;font-weight:600;">EXPANSION PROBABILITY</div>
                                <div style="display:flex;align-items:center;gap:10px;margin:4px 0;">
                                    <span style="color:{MUTED};width:100px;font-size:12px;">P(Expansion)</span>
                                    <div style="flex:1;height:14px;background:#222;border-radius:4px;">
                                        <div style="width:{v.get('ml_prob', 0.5)*100}%;height:100%;background:{RED if v.get('ml_prob', 0.5) > 0.65 else YELLOW if v.get('ml_prob', 0.5) > 0.35 else GREEN};border-radius:4px;transition:width 0.5s;"></div>
                                    </div>
                                    <span style="color:{WHITE};font-size:16px;font-weight:700;width:50px;text-align:right;">{v.get('ml_prob', 0.5):.0%}</span>
                                </div>
                                <div style="margin-top:14px;color:{MUTED};font-size:11px;font-weight:600;">REGIME FORECAST (extrapolated from trend)</div>
                                <table style="width:100%;margin-top:6px;border-collapse:collapse;font-size:12px;">
                                    <tr style="color:{MUTED};border-bottom:1px solid #333;">
                                        <th style="text-align:left;padding:4px 8px;">Horizon</th>
                                        <th style="text-align:center;padding:4px 8px;">Est. P(Exp)</th>
                                        <th style="text-align:center;padding:4px 8px;">Expected Regime</th>
                                    </tr>
                                    {v.get('forecast_rows', '<tr><td colspan="3" style="padding:4px 8px;color:#555;">Collecting data...</td></tr>')}
                                </table>
                                <div style="margin-top:10px;color:{MUTED};font-size:10px;border-top:1px solid #333;padding-top:6px;">
                                    HV Slope: {v['hv_slope']:+.2f}% | BW Percentile: {v['bw_percentile']:.0f} | Samples: {len(regime_history)}
                                </div>
                            </div>
                        </div>

                        <div class="card" style="margin-top:12px;border:1px solid {_intra_color}44;">
                            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px;">
                                <div style="color:{ACCENT};font-size:13px;font-weight:700;letter-spacing:2px;">⚡ INTRADAY REGIME SIGNAL</div>
                                <div style="display:flex;gap:8px;align-items:center;">
                                    <span style="background:{_dte_color}22;color:{_dte_color};font-size:11px;font-weight:700;padding:3px 10px;border-radius:12px;border:1px solid {_dte_color}55;">{_dte_note}</span>
                                    <span style="background:#2a2a4a;color:{MUTED};font-size:11px;padding:3px 10px;border-radius:12px;">{_session}</span>
                                </div>
                            </div>
                            <div style="display:flex;gap:12px;align-items:stretch;flex-wrap:wrap;">
                                <div style="flex:0 0 auto;background:{_intra_color}18;border:1px solid {_intra_color}55;border-radius:8px;padding:14px 20px;text-align:center;min-width:180px;">
                                    <div class="metric-label">INTRADAY BIAS</div>
                                    <div style="font-size:20px;font-weight:900;color:{_intra_color};margin:6px 0;">{_intra_signal}</div>
                                </div>
                                <div style="flex:1;display:grid;grid-template-columns:1fr 1fr 1fr;gap:10px;">
                                    <div class="metric-box"><div class="metric-label">IV Velocity (5d)</div><div style="font-size:18px;font-weight:700;color:{RED if _iv_velocity > 0.5 else GREEN if _iv_velocity < -0.5 else WHITE};">{_iv_velocity:+.2f}%</div><div class="metric-sub">IV drift trend</div></div>
                                    <div class="metric-box"><div class="metric-label">IV Z-Score</div><div style="font-size:18px;font-weight:700;color:{RED if _iv_zscore > 2 else GREEN if _iv_zscore < 0 else WHITE};">{_iv_zscore:.2f}×</div><div class="metric-sub">vs 1yr mean</div></div>
                                    <div class="metric-box"><div class="metric-label">VRP Edge</div><div style="font-size:18px;font-weight:700;color:{GREEN if _vrp_now > 2 else RED if _vrp_now < -2 else WHITE};">{_vrp_now:+.1f}%</div><div class="metric-sub">IV – HV</div></div>
                                </div>
                            </div>
                            <div style="margin-top:10px;padding:8px 12px;background:#0d0d1e;border-radius:6px;color:{MUTED};font-size:11px;">
                                {_intra_reason}
                                {f' &nbsp;|&nbsp; <span style="color:{YELLOW};">{_session_note}</span>' if _session_note else ''}
                            </div>
                        </div>'''


                    # ── TAB 3: OPTION CHAIN ANALYSER (renamed + redesigned) ──
                    chain_tab_html = '<div class="card"><p style="color:#888;">Loading option chain data...</p></div>'
                    if seller:
                        s = seller
                        # OI chain table rows (using _compute_seller_data chain_rows)
                        oi_rows_html = ''
                        for r in s['chain_rows']:
                            is_call_wall   = (r['strike'] == s['call_wall']   and s['call_wall']   > 0)
                            is_put_wall    = (r['strike'] == s['put_wall']    and s['put_wall']    > 0)
                            is_call_wall_2 = (r['strike'] == s.get('call_wall_2', 0) and s.get('call_wall_2', 0) > 0)
                            is_put_wall_2  = (r['strike'] == s.get('put_wall_2',  0) and s.get('put_wall_2',  0) > 0)

                            bg = 'transparent'
                            if r['is_atm']:     bg = 'rgba(79,195,247,0.10)'
                            if is_call_wall:    bg = 'rgba(255,82,82,0.15)'
                            elif is_put_wall:   bg = 'rgba(102,187,106,0.15)'
                            elif is_call_wall_2: bg = 'rgba(255,82,82,0.06)'
                            elif is_put_wall_2:  bg = 'rgba(102,187,106,0.06)'

                            ce_v = f"+{r['ce_vel']:,}" if r['ce_vel'] > 0 else f"{r['ce_vel']:,}" if r['ce_vel'] < 0 else "·"
                            pe_v = f"+{r['pe_vel']:,}" if r['pe_vel'] > 0 else f"{r['pe_vel']:,}" if r['pe_vel'] < 0 else "·"
                            ce_color = GREEN if r['ce_vel'] > 0 else RED if r['ce_vel'] < 0 else MUTED
                            pe_color = GREEN if r['pe_vel'] > 0 else RED if r['pe_vel'] < 0 else MUTED

                            strike_marker = ' ◄ ATM' if r['is_atm'] else ''
                            if   is_call_wall:   strike_marker += ' [CALL WALL ①]'
                            elif is_call_wall_2: strike_marker += ' [CALL WALL ②]'
                            if   is_put_wall:    strike_marker += ' [PUT WALL ①]'
                            elif is_put_wall_2:  strike_marker += ' [PUT WALL ②]'
                            strike_color = "#4fc3f7" if r["is_atm"] else RED if (is_call_wall or is_call_wall_2) else GREEN if (is_put_wall or is_put_wall_2) else WHITE

                            oi_rows_html += (
                                f'<tr style="background:{bg};">'
                                f'<td style="text-align:right;">{r["ce_oi"]:,}</td>'
                                f'<td style="text-align:right;color:{ce_color};">{ce_v}</td>'
                                f'<td style="text-align:center;font-weight:700;color:{strike_color};">{r["strike"]}{strike_marker}</td>'
                                f'<td style="text-align:right;color:{pe_color};">{pe_v}</td>'
                                f'<td style="text-align:right;">{r["pe_oi"]:,}</td>'
                                f'</tr>'
                            )

                        # Strike selection table
                        strike_rows_html = ''
                        for sr in s['strike_rows']:
                            sig_color_s = GREEN if '★' in sr['signal'] else ACCENT if '✓' in sr['signal'] else YELLOW if '~' in sr['signal'] else RED
                            p_otm_color = GREEN if sr['prob_otm'] >= 85 else ACCENT if sr['prob_otm'] >= 75 else YELLOW if sr['prob_otm'] >= 60 else RED
                            strike_rows_html += (
                                f'<tr>'
                                f'<td>{sr["strike"]:.0f}</td>'
                                f'<td>{sr["type"]}</td>'
                                f'<td>{sr["dist"]:+.0f}</td>'
                                f'<td style="color:{p_otm_color};">{sr["prob_otm"]:.1f}%</td>'
                                f'<td>{sr["price"]:.2f}</td>'
                                f'<td>{sr["theta"]:.2f}</td>'
                                f'<td style="color:{sig_color_s};">{sr["signal"]}</td>'
                                f'</tr>'
                            )

                        # Pull sell zones from the dict (computed inside _compute_seller_data)
                        safe_ce = s.get('sell_ce_above', spot + s['em'])
                        safe_pe = s.get('sell_pe_below', spot - s['em'])
                        call_wall_2 = s.get('call_wall_2', 0)
                        put_wall_2  = s.get('put_wall_2',  0)
                        pcr_color = GREEN if s['pcr'] > 1.2 else RED if s['pcr'] < 0.8 else YELLOW
                        pcr_label = 'Bullish (Put Writing)' if s['pcr'] > 1.2 else 'Bearish (Call Writing)' if s['pcr'] < 0.8 else 'Neutral'

                        chain_tab_html = f'''
                        <div class="card" style="margin-bottom:12px;">
                            <div style="color:{ACCENT};font-size:13px;font-weight:700;letter-spacing:2px;margin-bottom:14px;">KEY METRICS &amp; SELL ZONES — DTE {s['DTE']} | {near_exp}</div>
                            <div style="display:flex;gap:10px;flex-wrap:wrap;margin-bottom:14px;">
                                <div class="metric-box"><div class="metric-label">ATM IV</div><div style="font-size:20px;font-weight:700;color:{WHITE};">{s['atm_iv']:.2f}%</div></div>
                                <div class="metric-box"><div class="metric-label">Straddle</div><div style="font-size:20px;font-weight:700;color:{WHITE};">{s['straddle']:.0f}</div></div>
                                <div class="metric-box"><div class="metric-label">Exp. Move</div><div style="font-size:20px;font-weight:700;color:{WHITE};">±{s['em']:.0f} ({s['em_pct']:.1f}%)</div></div>
                                <div class="metric-box"><div class="metric-label">Max Pain</div><div style="font-size:20px;font-weight:700;color:{YELLOW};">{s['max_pain']}</div></div>
                                <div class="metric-box" style="border-left:2px solid {GREEN};">
                                    <div class="metric-label">Put Wall ①</div>
                                    <div style="font-size:20px;font-weight:700;color:{GREEN};">{s['put_wall']}</div>
                                    {f'<div class="metric-sub" style="color:#66bb6a88;">② {put_wall_2}</div>' if put_wall_2 else ''}
                                </div>
                                <div class="metric-box" style="border-left:2px solid {RED};">
                                    <div class="metric-label">Call Wall ①</div>
                                    <div style="font-size:20px;font-weight:700;color:{RED};">{s['call_wall']}</div>
                                    {f'<div class="metric-sub" style="color:#ff525288;">② {call_wall_2}</div>' if call_wall_2 else ''}
                                </div>
                                <div class="metric-box"><div class="metric-label">15m OI VELOCITY</div><div style="font-size:20px;font-weight:700;color:{GREEN if oi_pressure=='BULLISH' else RED if oi_pressure=='BEARISH' else YELLOW};">{oi_pressure}</div><div class="metric-sub">Score: {oi_pressure_score:+.0f}</div></div>
                                <div class="metric-box"><div class="metric-label">PCR</div><div style="font-size:20px;font-weight:700;color:{pcr_color};">{s['pcr']:.2f}</div><div class="metric-sub">{pcr_label}</div></div>
                            </div>
                            <div class="action-bar">
                                <span style="color:{MUTED};font-size:12px;margin-right:8px;">⚡ SELL ZONES:</span>
                                <span style="color:{GREEN};font-size:16px;font-weight:700;">SELL CE &gt; {safe_ce:.0f}</span>
                                <span style="margin:0 20px;color:{MUTED};">|</span>
                                <span style="color:{RED};font-size:16px;font-weight:700;">SELL PE &lt; {safe_pe:.0f}</span>
                            </div>
                        </div>
                        <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;">
                            <div class="card" style="max-height:420px;overflow-y:auto;">
                                <div style="color:{ACCENT};font-size:13px;font-weight:700;letter-spacing:2px;margin-bottom:10px;">15-MINUTE OI VELOCITY CHAIN</div>
                                <table class="data-table"><thead><tr>
                                    <th style="text-align:right;">CE OI</th>
                                    <th style="text-align:right;">CE Vel</th>
                                    <th style="text-align:center;">Strike</th>
                                    <th style="text-align:right;">PE Vel</th>
                                    <th style="text-align:right;">PE OI</th>
                                </tr></thead><tbody>{oi_rows_html}</tbody></table>
                            </div>
                            <div class="card" style="max-height:420px;overflow-y:auto;">
                                <div style="color:{ACCENT};font-size:13px;font-weight:700;letter-spacing:2px;margin-bottom:10px;">STRIKE SELECTION TABLE</div>
                                <table class="data-table"><thead><tr>
                                    <th>Strike</th><th>Type</th><th>Dist</th>
                                    <th>P(OTM)</th><th>Premium</th><th>Theta</th><th>Signal</th>
                                </tr></thead><tbody>{strike_rows_html}</tbody></table>
                            </div>
                        </div>'''

                        # ── GEX chart ──
                        gex_html = ''
                        if not df_chain.empty:
                            try:
                                _lot = 75  # NIFTY lot size
                                _T_gex = self.analytics.get_time_to_expiry(near_exp) or (1 / 365)
                                _df_gex = df_chain.copy()

                                # Compute Gamma via BSM if missing
                                def _get_gamma(r):
                                    g = r['gamma']
                                    if g == 0:
                                        # Use ensure_iv to safely back-calculate IV if the API string was 0
                                        calc_iv = self._ensure_iv(r['iv'], r['price'], r['strike'], _T_gex, r['type'])
                                        sigma = calc_iv / 100.0 if calc_iv > 0 else 0.15
                                        _ratio = spot / r['strike'] if r['strike'] > 0 else 1.0
                                        if _ratio <= 0: _ratio = 1e-6  # guard log domain
                                        d1 = (np.log(_ratio) + (0.07 + 0.5 * sigma**2) * _T_gex) / (sigma * np.sqrt(max(_T_gex, 1e-6)))
                                        g = np.exp(-0.5 * d1**2) / (np.sqrt(2 * np.pi) * spot * sigma * np.sqrt(max(_T_gex, 1e-6)))
                                    return g

                                _df_gex['gamma_calc'] = _df_gex.apply(_get_gamma, axis=1)

                                _df_gex['gex'] = _df_gex.apply(
                                    lambda r: r['oi'] * r['gamma_calc'] * _lot * spot * spot / 10000
                                             * (1 if r['type'] == 'CE' else -1), axis=1)

                                _gex_by_strike = _df_gex.groupby('strike')['gex'].sum().reset_index()
                                _gex_by_strike = _gex_by_strike[abs(_gex_by_strike['gex']) > 1e4]
                                _gex_by_strike = _gex_by_strike.sort_values('strike', ascending=True)

                                _gex_colors = ['rgba(255,68,68,0.85)' if g > 0 else 'rgba(102,187,106,0.85)' for g in _gex_by_strike['gex']]
                                fig_gex = go.Figure(go.Bar(
                                    x=_gex_by_strike['gex'],
                                    y=_gex_by_strike['strike'].astype(str),
                                    orientation='h',
                                    marker=dict(color=_gex_colors, line=dict(width=1, color='rgba(255,255,255,0.1)')),
                                    text=[f"{g/1e6:.1f}M" if abs(g) >= 1e6 else f"{g/1e3:.0f}K" for g in _gex_by_strike['gex']],
                                    textposition='outside',
                                    textfont=dict(color=WHITE, size=10),
                                    hovertemplate='Strike: %{y}<br>Net GEX: %{x:,.0f}<extra></extra>',
                                ))
                                fig_gex.add_vline(x=0, line_color=WHITE, line_width=1.5, opacity=0.3)
                                _net_gex = _gex_by_strike['gex'].sum()
                                _pin_txt = "PINNING \u2194" if abs(_net_gex)<5e6 else "TRENDING \u2195" if _net_gex<0 else "STABILITY \u2194"
                                fig_gex.update_layout(
                                    height=max(350, len(_gex_by_strike) * 20),
                                    paper_bgcolor=DARK_BG, plot_bgcolor='rgba(15,15,25,0.7)',
                                    font=dict(color=WHITE, family='Inter, sans-serif', size=11),
                                    margin=dict(l=60, r=60, t=50, b=40),
                                    title=dict(
                                        text=f'Gamma Exposure (GEX) by Strike  |  Net: {"+" if _net_gex>=0 else ""}{_net_gex/1e6:.2f}M  |  <b>{_pin_txt}</b>',
                                        font=dict(color=ACCENT, size=13), x=0.01),
                                    xaxis=dict(gridcolor='rgba(255,255,255,0.05)', title='GEX (dealer exposure)', showline=True, linecolor='rgba(255,255,255,0.1)', zeroline=False),
                                    yaxis=dict(gridcolor='rgba(255,255,255,0.05)', title='Strike', type='category', showline=True, linecolor='rgba(255,255,255,0.1)'),
                                    bargap=0.15,
                                )
                                gex_html = fig_gex.to_html(include_plotlyjs=False, full_html=False)
                                # GEX Analysis summary
                                _gex_regime = 'LONG GAMMA (mean-reverting, sticky)' if _net_gex > 0 else 'SHORT GAMMA (trending, slippery)'
                                _gex_regime_color = GREEN if _net_gex > 0 else RED
                                _max_call_strike = _df_gex[_df_gex['type']=='CE'].groupby('strike')['oi'].sum().idxmax() if not _df_gex[_df_gex['type']=='CE'].empty else 0
                                _max_put_strike = _df_gex[_df_gex['type']=='PE'].groupby('strike')['oi'].sum().idxmax() if not _df_gex[_df_gex['type']=='PE'].empty else 0
                                _dist_call = abs(spot - _max_call_strike) if _max_call_strike else 0
                                _dist_put = abs(spot - _max_put_strike) if _max_put_strike else 0
                                # Find flip strike (where cumulative GEX changes sign)
                                _cum_gex = _gex_by_strike.sort_values('strike')['gex'].cumsum()
                                _flip_strike = 0
                                for _idx in range(1, len(_cum_gex)):
                                    if _cum_gex.iloc[_idx-1] * _cum_gex.iloc[_idx] < 0:
                                        _flip_strike = int(_gex_by_strike.sort_values('strike').iloc[_idx]['strike'])
                                        break
                                gex_analysis_html = f'''
                                <div style="margin-top:12px;padding:10px;background:rgba(15,15,25,0.5);border-radius:6px;">
                                    <div style="color:{ACCENT};font-size:12px;font-weight:700;margin-bottom:8px;">GEX ANALYSIS</div>
                                    <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px;">
                                        <div class="metric-box"><div class="metric-label">Dealer Position</div><div style="font-size:13px;font-weight:700;color:{_gex_regime_color};">{_gex_regime}</div></div>
                                        <div class="metric-box"><div class="metric-label">Call Wall (Resistance)</div><div style="font-size:15px;font-weight:700;color:{WHITE};">{_max_call_strike:.0f}</div><div class="metric-sub">{_dist_call:.0f} pts from spot</div></div>
                                        <div class="metric-box"><div class="metric-label">Put Wall (Support)</div><div style="font-size:15px;font-weight:700;color:{WHITE};">{_max_put_strike:.0f}</div><div class="metric-sub">{_dist_put:.0f} pts from spot</div></div>
                                    </div>
                                    <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-top:8px;">
                                        <div class="metric-box"><div class="metric-label">Net GEX</div><div style="font-size:15px;font-weight:700;color:{GREEN if _net_gex > 0 else RED};">{_net_gex/1e6:+.2f}M</div><div class="metric-sub">{_pin_txt}</div></div>
                                        <div class="metric-box"><div class="metric-label">GEX Flip Strike</div><div style="font-size:15px;font-weight:700;color:{YELLOW};">{_flip_strike if _flip_strike else 'N/A'}</div><div class="metric-sub">{'Above spot' if _flip_strike > spot else 'Below spot' if _flip_strike else ''}</div></div>
                                    </div>
                                </div>'''
                                chain_tab_html += f'<div class="card" style="margin-top:12px;padding:8px 16px;">{gex_html}{gex_analysis_html}</div>'
                            except Exception as e:
                                print(f"GEX Error: {e}")
                                pass
                    else:
                        chain_tab_html = '<div class="card"><p style="color:#888;">No option chain data available for analysis.</p></div>'


                    # ── TAB 4: PROBABILITY DENSITY ──
                    prob_tab_html = '<div class="card"><p style="color:#888;">Waiting for ATM IV data to compute probability density...</p></div>'
                    if prob_density:
                        def _hex_to_rgba(hex_c, alpha):
                            """Convert hex color to rgba string for Plotly."""
                            h = hex_c.lstrip('#')
                            return f'rgba({int(h[0:2],16)},{int(h[2:4],16)},{int(h[4:6],16)},{alpha})'

                        fig_pd = make_subplots(rows=1, cols=2,
                            specs=[[{"type": "xy"}, {"type": "scene"}]],
                            subplot_titles=[
                                'Price Probability Distribution by Horizon',
                                '3D Probability Surface (Price × Days)'
                            ], horizontal_spacing=0.05)

                        for pd_h in prob_density:
                            fig_pd.add_trace(go.Scatter(
                                x=pd_h['prices'], y=pd_h['pdf'], mode='lines',
                                name=pd_h['label'],
                                line=dict(color=pd_h['color'], width=2.5),
                                fill='tozeroy',
                                fillcolor=_hex_to_rgba(pd_h['color'], 0.08),
                            ), row=1, col=1)
                            # ±1σ shaded region
                            sigma_mask = (pd_h['prices'] >= pd_h['one_sigma_lo']) & (pd_h['prices'] <= pd_h['one_sigma_hi'])
                            sigma_prices = pd_h['prices'][sigma_mask]
                            sigma_pdf = pd_h['pdf'][sigma_mask]
                            if len(sigma_prices) > 0:
                                fig_pd.add_trace(go.Scatter(
                                    x=sigma_prices, y=sigma_pdf, mode='lines',
                                    line=dict(width=0), fill='tozeroy',
                                    fillcolor=_hex_to_rgba(pd_h['color'], 0.25),
                                    showlegend=False, hoverinfo='skip',
                                ), row=1, col=1)
                        # BSM overlay (dashed) for comparison
                        if bsm_density:
                            for bsm_h in bsm_density:
                                fig_pd.add_trace(go.Scatter(
                                    x=bsm_h['prices'], y=bsm_h['pdf'], mode='lines',
                                    name=bsm_h['label'],
                                    line=dict(color=bsm_h['color'], width=1.5, dash='dash'),
                                    opacity=0.5,
                                ), row=1, col=1)
                        # Spot line on 2D
                        fig_pd.add_vline(x=spot, line_dash='dash', line_color=YELLOW, line_width=2,
                            annotation_text=f'Spot:{spot:.0f}', annotation_font_color=YELLOW,
                            annotation_position='top right', row=1, col=1)

                        # ── 3D SURFACE: interpolate PDFs across days ──
                        all_days = sorted(set([h['days'] for h in prob_density]))
                        # Build a finer day grid from min to max
                        day_grid = np.linspace(min(all_days), max(all_days), 40)
                        # Use common price range (union of all horizons)
                        p_min = min(h['prices'][0] for h in prob_density)
                        p_max = max(h['prices'][-1] for h in prob_density)
                        price_grid = np.linspace(p_min, p_max, 150)
                        # Collect scatter points for griddata
                        xs, ys, zs = [], [], []
                        for pd_h in prob_density:
                            for p, z in zip(pd_h['prices'], pd_h['pdf']):
                                xs.append(p)
                                ys.append(pd_h['days'])
                                zs.append(z)
                        xs, ys, zs = np.array(xs), np.array(ys), np.array(zs)
                        pm, dm = np.meshgrid(price_grid, day_grid)
                        try:
                            z_mesh = griddata((xs, ys), zs, (pm, dm), method='cubic')
                            z_nn = griddata((xs, ys), zs, (pm, dm), method='nearest')
                            z_mesh = np.where(np.isnan(z_mesh), z_nn, z_mesh)
                            z_mesh = np.clip(z_mesh, 0, None)  # PDF can't be negative
                        except:
                            z_mesh = griddata((xs, ys), zs, (pm, dm), method='nearest')
                            z_mesh = np.clip(z_mesh, 0, None)

                        fig_pd.add_trace(go.Surface(
                            x=price_grid, y=day_grid, z=z_mesh,
                            colorscale='Turbo', opacity=0.85, showscale=True,
                            colorbar=dict(title='PDF', len=0.7, x=1.01),
                        ), row=1, col=2)
                        # Individual distribution lines on 3D surface
                        for pd_h in prob_density:
                            fig_pd.add_trace(go.Scatter3d(
                                x=pd_h['prices'], y=np.full_like(pd_h['prices'], pd_h['days']),
                                z=pd_h['pdf'], mode='lines',
                                line=dict(color=pd_h['color'], width=4),
                                name=pd_h['label'] + ' (3D)', showlegend=False,
                            ), row=1, col=2)

                        # ── Variance cone lines across days ──
                        sigma_ann = _pd_iv / 100
                        cone_days = np.linspace(1, max(h['days'] for h in prob_density), 60)
                        cone_T = cone_days / 365.0
                        mean_price = spot * np.exp((0.07 - 0.5 * sigma_ann**2) * cone_T)
                        sigma_t = sigma_ann * np.sqrt(cone_T)
                        z_floor = np.zeros_like(cone_days)
                        # Mean line
                        fig_pd.add_trace(go.Scatter3d(
                            x=mean_price, y=cone_days, z=z_floor, mode='lines',
                            line=dict(color=YELLOW, width=3, dash='dash'),
                            name='Mean', showlegend=False,
                        ), row=1, col=2)
                        # ±1σ lines
                        for sign, label in [(1, '+1σ'), (-1, '-1σ')]:
                            boundary = spot * np.exp(sign * sigma_t)
                            fig_pd.add_trace(go.Scatter3d(
                                x=boundary, y=cone_days, z=z_floor, mode='lines',
                                line=dict(color=ACCENT, width=3),
                                name=label, showlegend=False,
                            ), row=1, col=2)
                        # ±2σ lines
                        for sign, label in [(1, '+2σ'), (-1, '-2σ')]:
                            boundary = spot * np.exp(sign * 2 * sigma_t)
                            fig_pd.add_trace(go.Scatter3d(
                                x=boundary, y=cone_days, z=z_floor, mode='lines',
                                line=dict(color=RED, width=2, dash='dot'),
                                name=label, showlegend=False,
                            ), row=1, col=2)

                        fig_pd.update_layout(
                            height=480, width=1380, paper_bgcolor=DARK_BG,
                            plot_bgcolor='rgba(20,20,35,0.8)',
                            font=dict(color=WHITE, family='Inter, sans-serif', size=11),
                            legend=dict(bgcolor='rgba(30,30,50,0.8)', font=dict(size=10), x=0.01, y=0.99),
                            margin=dict(l=50, r=20, t=50, b=30), hovermode='closest',
                            scene=dict(
                                xaxis=dict(title='Price', backgroundcolor=DARK_BG, gridcolor='#333'),
                                yaxis=dict(title='Days', backgroundcolor=DARK_BG, gridcolor='#333'),
                                zaxis=dict(title='PDF', backgroundcolor=DARK_BG, gridcolor='#333'),
                                camera=dict(eye=dict(x=1.5, y=-1.8, z=0.8)), bgcolor=DARK_BG),
                        )
                        fig_pd.update_xaxes(gridcolor='rgba(100,100,100,0.15)', title='NIFTY Price', row=1, col=1)
                        fig_pd.update_yaxes(gridcolor='rgba(100,100,100,0.15)', title='Probability Density', row=1, col=1)
                        pd_plotly = fig_pd.to_html(include_plotlyjs=False, full_html=False)

                        heston_info = ''
                        hp = heston_cache.get('params')
                        is_heston = bool(hp)
                        if hp:
                            heston_info = f"κ={hp['kappa']:.2f} θ={hp['theta']:.4f} v₀={hp['v0']:.4f} ρ={hp['rho']:.2f} ξ={hp['xi']:.2f}"
                        model_label = 'Heston MC (50k paths)' if is_heston else 'BSM Log-Normal'
                        model_badge_color = ACCENT if is_heston else YELLOW
                        model_badge_icon  = '✓ Heston MC calibrated' if is_heston else '⚠ BSM fallback (Heston not converged)'

                        prob_tab_html = f'''
                        {pd_plotly}
                        <div style="margin-top:12px;padding:8px 14px;background:#12122a;border-radius:8px;border:1px solid #2a2a4a;display:flex;align-items:center;gap:12px;">
                            <span style="color:{model_badge_color};font-size:12px;font-weight:700;">{model_badge_icon}</span>
                            <span style="color:{MUTED};font-size:11px;">ATM IV: {_pd_iv:.2f}%</span>
                            {f'<span style="color:{MUTED};font-size:11px;">{heston_info}</span>' if heston_info else ''}
                        </div>'''

                        # ── Level Breach Probability panel ──
                        if prob_density and seller:
                            try:
                                from scipy.stats import norm as _norm
                                _s  = seller
                                _T  = self.analytics.get_time_to_expiry(near_exp)
                                _pd_iv_dec = _pd_iv / 100.0
                                _call_wall = float(_s['call_wall']) if _s['call_wall'] else 0
                                _put_wall  = float(_s['put_wall'])  if _s['put_wall']  else 0
                                _max_pain  = float(_s['max_pain'])  if _s['max_pain']  else spot
                                _em_pts    = _s['em']
                                _straddle  = _s['straddle']

                                def _p_above(lvl):
                                    if _T <= 0 or _pd_iv_dec <= 0: return 50.0
                                    d = (np.log(lvl/spot) - 0.5 * _pd_iv_dec**2 * _T) / (_pd_iv_dec * np.sqrt(_T))
                                    return (1 - _norm.cdf(d)) * 100

                                def _p_below(lvl):
                                    return 100 - _p_above(lvl)

                                def _p_range(lo, hi):
                                    return _p_above(lo) - _p_above(hi)

                                _p_cw = _p_above(_call_wall)  if _call_wall > 0 else None
                                _p_pw = _p_below(_put_wall)   if _put_wall  > 0 else None
                                _p_mp = _p_range(_max_pain - 50, _max_pain + 50)
                                _p_sfe = _p_range(spot - _straddle, spot + _straddle)  # straddle profit zone
                                _p_em  = _p_range(spot - _em_pts, spot + _em_pts)

                                def _breach_card(label, prob, good_is_low=True):
                                    if prob is None: return ''
                                    c = GREEN if (prob < 25 if good_is_low else prob > 75) else RED if (prob > 50 if good_is_low else prob < 50) else YELLOW
                                    return f'<div class="metric-box"><div class="metric-label">{label}</div><div style="font-size:22px;font-weight:700;color:{c};">{prob:.1f}%</div></div>'

                                # Calculate probability for SELL ZONES & SESSION LEVELS
                                _safe_ce = _s.get('sell_ce_above', 0)
                                _safe_pe = _s.get('sell_pe_below', 0)
                                _p_breach_ce = _p_above(_safe_ce) if _safe_ce > 0 else None
                                _p_breach_pe = _p_below(_safe_pe) if _safe_pe > 0 else None
                                
                                _d_high = momentum_data.get('day_high', 0)
                                _d_low  = momentum_data.get('day_low', 0)
                                _p_breach_high = _p_above(_d_high) if _d_high > 0 else None
                                _p_breach_low  = _p_below(_d_low)  if _d_low  > 0 else None

                                breach_html = f'''
                                <div class="card" style="margin-top:12px;">
                                    <div style="color:{ACCENT};font-size:13px;font-weight:700;letter-spacing:2px;margin-bottom:14px;">LEVEL BREACH PROBABILITY</div>
                                    <div style="margin-bottom:15px;">
                                        <div style="color:{MUTED};font-size:10px;font-weight:700;margin-bottom:8px;text-transform:uppercase;">Sell Zone Strike Breach (Option Chain Signals)</div>
                                        <div style="display:flex;gap:10px;flex-wrap:wrap;">
                                            {_breach_card("P(Break CE " + (str(int(_safe_ce)) if _safe_ce else "N/A") + ")", _p_breach_ce)}
                                            {_breach_card("P(Break PE " + (str(int(_safe_pe)) if _safe_pe else "N/A") + ")", _p_breach_pe)}
                                        </div>
                                        <div style="margin-top:6px;color:{MUTED};font-size:10px;">Sell CE above {int(_safe_ce) if _safe_ce else 'N/A'} | Sell PE below {int(_safe_pe) if _safe_pe else 'N/A'} (from Option Chain tab)</div>
                                    </div>
                                    <div style="margin-bottom:15px;border-top:1px solid #333;padding-top:12px;">
                                        <div style="color:{MUTED};font-size:10px;font-weight:700;margin-bottom:8px;text-transform:uppercase;">Session Levels (Intraday)</div>
                                        <div style="display:flex;gap:10px;flex-wrap:wrap;">
                                            {_breach_card("P(Breach DAY HIGH " + str(int(_d_high)) + ")", _p_breach_high)}
                                            {_breach_card("P(Breach DAY LOW " + str(int(_d_low)) + ")", _p_breach_low)}
                                        </div>
                                    </div>
                                    <div style="display:flex;gap:10px;flex-wrap:wrap;border-top:1px solid #333;padding-top:12px;">
                                        {_breach_card("P(In EM Range)", _p_em, good_is_low=False)}
                                        {_breach_card("P(Above Call Wall " + str(int(_call_wall)) + ")", _p_cw)}
                                        {_breach_card("P(Below Put Wall " + str(int(_put_wall)) + ")", _p_pw)}
                                    </div>
                                    <div style="margin-top:12px;padding:8px 14px;background:#12122a;border-radius:8px;border:1px solid #2a2a4a;display:flex;gap:12px;align-items:center;">
                                        <span style="color:{model_badge_color};font-size:12px;font-weight:700;">{model_badge_icon}</span>
                                        <span style="color:{MUTED};font-size:11px;">ATM IV: {_pd_iv:.2f}% | DTE: {_s['DTE']}</span>
                                    </div>
                                </div>'''
                                prob_tab_html += breach_html
                            except Exception:
                                pass

                    # ── TAB 5: STRATEGY ENGINE & LIVE P&L TRACKER ──
                    strategy_tab_html = '<div class="card"><p style="color:#888;">Initializing Strategy Engine...</p></div>'
                    try:
                        from StrategyEngine import (SmartStrategyGenerator, PayoffEngine,
                                                     build_strategy_card_html,
                                                     NIFTY_LOT, DARK_BG as _SE_BG)

                        # ── Market context ──────────────────────────────────────
                        _T_strat   = self.analytics.get_time_to_expiry(near_exp)
                        _iv_strat  = (vol_intel.get('iv', 15) if vol_intel else (pred.get('atm_iv', 15) if pred else 15))
                        _market_ctx = {
                            'T': _T_strat, 'iv': _iv_strat, 'atm_iv': _iv_strat,
                            'vrp':     vol_intel.get('vrp', 0) if vol_intel else 0,
                            'regime':  vol_intel.get('regime', 'NORMAL') if vol_intel else 'NORMAL',
                            'call_wall': float(seller.get('call_wall', 0)) if seller else 0,
                            'put_wall':  float(seller.get('put_wall', 0)) if seller else 0,
                            'em':   float(seller.get('em', spot * 0.015)) if seller else spot * 0.015,
                            'oi_pressure': oi_pressure,
                            'DTE': seller.get('DTE', 7) if seller else 7,
                        }
                        _cw = _market_ctx['call_wall']
                        _pw = _market_ctx['put_wall']
                        _dte_s  = _market_ctx['DTE']
                        _vrp_s  = _market_ctx['vrp']
                        _reg_s  = _market_ctx['regime']

                        # ── Context badge colors ──
                        _dte_clr    = RED if _dte_s <= 1 else YELLOW if _dte_s <= 3 else GREEN
                        _reg_clr    = YELLOW if _reg_s == 'COMPRESSION' else RED if _reg_s == 'EXPANSION' else '#64b5f6'
                        _vrp_clr    = GREEN if _vrp_s > 2 else RED if _vrp_s < -2 else YELLOW
                        _bias_lbl   = 'SELL PREMIUM' if _vrp_s > 3 else 'AVOID SELL' if _vrp_s < -2 else 'NEUTRAL'
                        _bias_clr   = GREEN if _bias_lbl == 'SELL PREMIUM' else RED if 'AVOID' in _bias_lbl else YELLOW

                        # ── Generate strategies ──────────────────────────────────
                        gen = SmartStrategyGenerator(spot, df_chain, _market_ctx, near_exp)
                        _strategies = gen.generate()

                        # ── Build per-strategy data: payoff chart + metrics ──────
                        _strat_data  = []   # list of dicts with name/score/html/metrics
                        _iv_dec = _iv_strat / 100.0
                        for _si, _so in enumerate(_strategies[:5]):
                            _pe    = PayoffEngine(_so, spot, _T_strat, _iv_dec,
                                                  call_wall=_cw, put_wall=_pw)
                            _pr    = np.linspace(spot * 0.85, spot * 1.15, 1000)
                            _max_p = _so.max_profit(_pr) * NIFTY_LOT
                            _max_l = _so.max_loss(_pr)   * NIFTY_LOT
                            _bes   = _so.breakevens(_pr)
                            _pop   = _so.pop(spot, _T_strat, _iv_dec) * 100
                            _ngrk  = _so.net_greeks(spot, _T_strat)
                            _nprem = _so.net_premium * NIFTY_LOT
                            _chart = _pe.build_payoff_chart().to_html(
                                include_plotlyjs=False, full_html=False,
                                config={'displayModeBar': False}
                            )
                            _greek_html = _pe.build_greeks_html()

                            # Leg rows for compact table
                            _leg_rows = ''
                            for _lg in _so.legs:
                                _ltype_clr = '#4fc3f7' if _lg.opt_type == 'CE' else '#ef9a9a'
                                _lact_clr  = RED if _lg.action == 'SELL' else GREEN
                                _leg_rows += (
                                    f'<tr style="border-bottom:1px solid #1e1e35;">'
                                    f'<td style="padding:4px 10px;color:{_lact_clr};font-weight:700;">{_lg.action}</td>'
                                    f'<td style="padding:4px 10px;color:{_ltype_clr};">{_lg.opt_type}</td>'
                                    f'<td style="padding:4px 10px;color:{WHITE};font-weight:600;">{_lg.strike:.0f}</td>'
                                    f'<td style="padding:4px 10px;color:{MUTED};">₹{_lg.entry_price:.1f}</td>'
                                    f'<td style="padding:4px 10px;color:{MUTED};">{_lg.lots} lot</td>'
                                    f'</tr>'
                                )

                            _pop_clr = GREEN if _pop >= 65 else YELLOW if _pop >= 50 else RED
                            _prem_clr = GREEN if _nprem > 0 else RED
                            _rr_str = f'1:{abs(_max_p / _max_l):.1f}' if _max_l != 0 and abs(_max_l) < 1e6 else '∞'
                            _bes_str = ' / '.join(f'{b:,.0f}' for b in _bes) if _bes else '—'
                            _payload = json.dumps(_so.to_dict()).replace('"', '&quot;')

                            _strat_data.append({
                                'name':    _so.name,
                                'score':   _so.score,
                                'type':    _so.strategy_type,
                                'pop':     _pop,
                                'pop_clr': _pop_clr,
                                'nprem':   _nprem,
                                'prem_clr': _prem_clr,
                                'max_p':   _max_p,
                                'max_l':   abs(_max_l),
                                'rr_str':  _rr_str,
                                'bes_str': _bes_str,
                                'chart_html': _chart,
                                'greek_html': _greek_html,
                                'ngrk':    _ngrk,
                                'leg_rows': _leg_rows,
                                'payload': _payload,
                                'idx':     _si,
                            })

                        # ── Standalone backtest runner migrated to outer scope ───────────

                        # ── Live P&L tracker data ────────────────────────────────
                        tracked   = _strategy_manager.get_all_active_strategies()
                        closed_h  = _strategy_manager.get_closed_strategies(limit=8)
                        summary   = _strategy_manager.get_summary()
                        tracked_html = generate_tracked_html(tracked, df_chain)
                        _total_pnl   = summary.get('total_realized_pnl', 0)
                        _total_clr   = GREEN if _total_pnl >= 0 else RED

                        _closed_rows = ''
                        for _ct in closed_h:
                            _pnl = _ct.get('pnl', 0) or 0
                            _pc  = GREEN if _pnl > 0 else RED
                            _closed_rows += (
                                f'<tr style="border-bottom:1px solid #1a1a2e;">'
                                f'<td style="padding:5px 8px;color:#ccc;">{_ct.get("name","")}</td>'
                                f'<td style="padding:5px 8px;color:{MUTED};">{(_ct.get("exit_time_str","") or "")[:10]}</td>'
                                f'<td style="padding:5px 8px;text-align:right;font-weight:700;color:{_pc};">'
                                f'{"+" if _pnl > 0 else ""}₹{_pnl:,.0f}</td>'
                                f'</tr>'
                            )

                        # ── Build strategy pills + hidden strategy panels ─────────
                        _pills_html  = ''
                        _panels_html = ''
                        _num_strats  = len(_strat_data)

                        for _sd in _strat_data:
                            _i         = _sd['idx']
                            _sc        = GREEN if _sd['score'] >= 70 else YELLOW if _sd['score'] >= 45 else RED
                            _active    = 'strat-pill-active' if _i == 0 else ''
                            _pstyle_attr = '' if _i == 0 else ' style="display:none;"'

                            _pills_html += f'''
                            <button class="strat-pill {_active}" onclick="selectStrat({_i})"
                                    id="pill-{_i}"
                                    style="background:{"#1e2a3a" if _i == 0 else "#12121e"};
                                           color:{_sc if _i == 0 else MUTED};
                                           border:1px solid {_sc if _i == 0 else "#2a2a4a"};
                                           border-radius:20px;padding:6px 16px;cursor:pointer;
                                           font-size:11px;font-weight:700;white-space:nowrap;
                                           transition:all 0.2s;">
                                {_sd["name"]}
                                <span style="font-size:10px;opacity:0.8;">({_sd["score"]})</span>
                            </button>'''

                            _panels_html += f'''
                            <div id="strat-panel-{_i}"{_pstyle_attr}>
                                <!-- Top metrics row -->
                                <div style="display:grid;grid-template-columns:repeat(6,1fr);
                                            gap:8px;margin-bottom:12px;">
                                    <div style="background:#0d0d20;border:1px solid {_sd["pop_clr"]}44;
                                                border-radius:8px;padding:12px 8px;text-align:center;">
                                        <div style="color:{MUTED};font-size:9px;font-weight:700;
                                                    text-transform:uppercase;margin-bottom:4px;">POP</div>
                                        <div style="font-size:28px;font-weight:900;
                                                    color:{_sd["pop_clr"]};line-height:1;">{_sd["pop"]:.0f}%</div>
                                    </div>
                                    <div style="background:#0d0d20;border:1px solid #2a2a4a;
                                                border-radius:8px;padding:12px 8px;text-align:center;">
                                        <div style="color:{MUTED};font-size:9px;font-weight:700;
                                                    text-transform:uppercase;margin-bottom:4px;">Net Premium</div>
                                        <div style="font-size:16px;font-weight:800;
                                                    color:{_sd["prem_clr"]};">
                                            {"+" if _sd["nprem"] > 0 else ""}₹{_sd["nprem"]:.0f}</div>
                                    </div>
                                    <div style="background:#0d0d20;border:1px solid #2a2a4a;
                                                border-radius:8px;padding:12px 8px;text-align:center;">
                                        <div style="color:{MUTED};font-size:9px;font-weight:700;
                                                    text-transform:uppercase;margin-bottom:4px;">Max Profit</div>
                                        <div style="font-size:16px;font-weight:800;color:{GREEN};">
                                            {"∞" if _sd["max_p"] > 1e6 else f"₹{_sd['max_p']:,.0f}"}</div>
                                    </div>
                                    <div style="background:#0d0d20;border:1px solid #2a2a4a;
                                                border-radius:8px;padding:12px 8px;text-align:center;">
                                        <div style="color:{MUTED};font-size:9px;font-weight:700;
                                                    text-transform:uppercase;margin-bottom:4px;">Max Loss</div>
                                        <div style="font-size:16px;font-weight:800;color:{RED};">
                                            {"∞" if _sd["max_l"] > 1e6 else f"₹{_sd['max_l']:,.0f}"}</div>
                                    </div>
                                    <div style="background:#0d0d20;border:1px solid #2a2a4a;
                                                border-radius:8px;padding:12px 8px;text-align:center;">
                                        <div style="color:{MUTED};font-size:9px;font-weight:700;
                                                    text-transform:uppercase;margin-bottom:4px;">R:R</div>
                                        <div style="font-size:16px;font-weight:800;color:{WHITE};">{_sd["rr_str"]}</div>
                                    </div>
                                    <div style="background:#0d0d20;border:1px solid #2a2a4a;
                                                border-radius:8px;padding:12px 8px;text-align:center;">
                                        <div style="color:{MUTED};font-size:9px;font-weight:700;
                                                    text-transform:uppercase;margin-bottom:4px;">Breakeven(s)</div>
                                        <div style="font-size:12px;font-weight:700;color:#ff7043;">{_sd["bes_str"]}</div>
                                    </div>
                                </div>

                                <!-- Greeks mini-bar -->
                                <div style="display:flex;gap:14px;background:#0a0a18;border-radius:6px;
                                            padding:8px 14px;margin-bottom:10px;flex-wrap:wrap;">
                                    {f'<span style="color:{MUTED};font-size:10px;">Δ <b style="color:{"#ef9a9a" if _sd["ngrk"]["delta"] < 0 else "#a5d6a7"};">{_sd["ngrk"]["delta"]:+.3f}</b></span>'}
                                    {f'<span style="color:{MUTED};font-size:10px;">Γ <b style="color:#aaa;">{_sd["ngrk"]["gamma"]:.4f}</b></span>'}
                                    {f'<span style="color:{MUTED};font-size:10px;">θ/d <b style="color:{"#a5d6a7" if _sd["ngrk"]["theta"] > 0 else "#ef9a9a"};">{_sd["ngrk"]["theta"]:+.1f}</b></span>'}
                                    {f'<span style="color:{MUTED};font-size:10px;">ν% <b style="color:#aaa;">{_sd["ngrk"]["vega"]:+.1f}</b></span>'}
                                    <span style="margin-left:auto;">
                                        <button onclick="trackStrategy('{_sd["payload"]}')"
                                                style="background:{GREEN};color:#000;border:none;
                                                       border-radius:6px;padding:5px 16px;font-size:11px;
                                                       font-weight:700;cursor:pointer;">⊕ Track Live</button>
                                    </span>
                                </div>

                                <!-- Payoff chart (full width) -->
                                {_sd["chart_html"]}

                                <!-- Leg breakdown + Greeks detail (collapsible) -->
                                <div style="display:grid;grid-template-columns:1fr 1.4fr;gap:10px;margin-top:10px;">
                                    <div>
                                        <div style="color:{MUTED};font-size:10px;font-weight:700;
                                                    text-transform:uppercase;margin-bottom:6px;">Legs</div>
                                        <table style="width:100%;border-collapse:collapse;font-size:12px;">
                                            <thead><tr style="color:{MUTED};border-bottom:1px solid #2a2a4a;">
                                                <th style="padding:4px 10px;text-align:left;">Action</th>
                                                <th style="padding:4px 10px;text-align:left;">Type</th>
                                                <th style="padding:4px 10px;text-align:left;">Strike</th>
                                                <th style="padding:4px 10px;text-align:left;">Price</th>
                                                <th style="padding:4px 10px;text-align:left;">Qty</th>
                                            </tr></thead>
                                            <tbody>{_sd["leg_rows"]}</tbody>
                                        </table>
                                    </div>
                                    <details>
                                        <summary style="color:{MUTED};font-size:10px;font-weight:700;
                                                        text-transform:uppercase;cursor:pointer;
                                                        padding:4px 0;">▶ Greeks per leg (1 lot)</summary>
                                        <div style="margin-top:6px;">{_sd["greek_html"]}</div>
                                    </details>
                                </div>
                            </div>'''

                        # ── Assemble full strategy tab HTML ──────────────────────
                        strategy_tab_html = f'''
                        <!-- ===================== CONTEXT BAR ===================== -->
                        <div style="display:flex;flex-wrap:wrap;gap:7px;align-items:center;
                                    background:linear-gradient(135deg,#12122a,#0e0e1e);
                                    border:1px solid #252540;border-radius:10px;
                                    padding:10px 16px;margin-bottom:14px;">
                            <span style="font-size:13px;font-weight:900;color:{ACCENT};
                                         letter-spacing:1px;margin-right:6px;">⚡ STRATEGY ENGINE</span>
                            <span style="background:{_dte_clr}1a;color:{_dte_clr};font-size:11px;
                                         font-weight:700;padding:3px 11px;border-radius:12px;
                                         border:1px solid {_dte_clr}44;">{_dte_s}d expiry</span>
                            <span style="background:{_reg_clr}1a;color:{_reg_clr};font-size:11px;
                                         font-weight:700;padding:3px 11px;border-radius:12px;
                                         border:1px solid {_reg_clr}44;">{_reg_s}</span>
                            <span style="background:{_vrp_clr}1a;color:{_vrp_clr};font-size:11px;
                                         padding:3px 11px;border-radius:12px;
                                         border:1px solid {_vrp_clr}44;">VRP {_vrp_s:+.1f}%</span>
                            <span style="background:#1a1a30;color:{MUTED};font-size:11px;
                                         padding:3px 11px;border-radius:12px;">IV {_iv_strat:.1f}%</span>
                            {'<span style="background:#1a1a30;color:#aaa;font-size:11px;padding:3px 11px;border-radius:12px;">⬆ CW ' + f'{_cw:.0f}' + '</span>' if _cw else ''}
                            {'<span style="background:#1a1a30;color:#aaa;font-size:11px;padding:3px 11px;border-radius:12px;">⬇ PW ' + f'{_pw:.0f}' + '</span>' if _pw else ''}
                            <div style="margin-left:auto;display:flex;align-items:center;gap:8px;">
                                <span style="font-size:12px;font-weight:900;padding:4px 16px;
                                             border-radius:12px;
                                             background:{_bias_clr}22;color:{_bias_clr};
                                             border:1px solid {_bias_clr}55;">{_bias_lbl}</span>
                            </div>
                        </div>

                        <!-- =================== MAIN GRID ========================= -->
                        <div style="display:grid;grid-template-columns:1fr 2.2fr;gap:14px;">

                            <!-- LEFT COLUMN: Tracker + closed trades -->
                            <div>
                                <div class="card" style="margin-bottom:12px;">
                                    <div style="display:flex;justify-content:space-between;
                                                align-items:center;margin-bottom:10px;">
                                        <span style="color:{ACCENT};font-size:12px;font-weight:700;
                                                     letter-spacing:2px;">LIVE P&amp;L TRACKER</span>
                                        <span style="color:{_total_clr};font-size:12px;font-weight:700;">
                                            {"+" if _total_pnl >= 0 else ""}₹{_total_pnl:,.0f} realized</span>
                                    </div>
                                    <div id="pnlTrackerContainer" style="max-height:280px;overflow-y:auto;">
                                        {tracked_html}
                                    </div>
                                    <div style="display:flex;gap:7px;margin-top:10px;flex-wrap:wrap;">
                                        <input type="time" id="histScrubTime"
                                               style="background:#0d0d20;color:{WHITE};
                                                      border:1px solid #333;border-radius:5px;
                                                      padding:3px 7px;font-size:11px;flex:1;">
                                        <button onclick="fetchHistoricalPNL()"
                                                style="background:{ACCENT};color:#000;border:none;
                                                       border-radius:5px;padding:4px 12px;font-size:11px;
                                                       font-weight:700;cursor:pointer;">Scrub</button>
                                        <button onclick="resetToLivePNL()"
                                                style="background:#1e1e30;color:{MUTED};border:none;
                                                       border-radius:5px;padding:4px 10px;font-size:11px;
                                                       cursor:pointer;">Live</button>
                                    </div>
                                </div>

                                <div class="card">
                                    <div style="color:{ACCENT};font-size:11px;font-weight:700;
                                                letter-spacing:2px;margin-bottom:8px;">CLOSED TRADES</div>
                                    <table style="width:100%;border-collapse:collapse;font-size:11px;">
                                        <thead>
                                            <tr style="color:{MUTED};border-bottom:1px solid #252540;">
                                                <th style="text-align:left;padding:4px 8px;">Strategy</th>
                                                <th style="text-align:left;padding:4px 8px;">Exit</th>
                                                <th style="text-align:right;padding:4px 8px;">P&amp;L</th>
                                            </tr>
                                        </thead>
                                        <tbody>
                                            {_closed_rows or f'<tr><td colspan="3" style="padding:8px;color:#444;text-align:center;">No closed trades yet</td></tr>'}
                                        </tbody>
                                    </table>
                                </div>
                            </div>

                            <!-- RIGHT COLUMN: Strategy selector + payoff -->
                            <div>
                                <div class="card">
                                    <!-- Strategy Selector Pills -->
                                    <div style="display:flex;gap:7px;flex-wrap:wrap;
                                                margin-bottom:14px;align-items:center;">
                                        <span style="color:{MUTED};font-size:10px;font-weight:700;
                                                     text-transform:uppercase;margin-right:4px;">Strategy:</span>
                                        {_pills_html}
                                    </div>
                                    <!-- Strategy content panels (one visible at a time) -->
                                    {_panels_html}
                                </div>
                            </div>
                        </div>

                        <!-- =================== BACKTEST PANEL ==================== -->
                        <div class="card" style="margin-top:14px;">
                            <div style="display:flex;justify-content:space-between;
                                        align-items:center;margin-bottom:10px;">
                                <div>
                                    <span style="color:{ACCENT};font-size:12px;font-weight:700;
                                                 letter-spacing:2px;">📊 BACKTESTER</span>
                                    <span style="color:{MUTED};font-size:10px;margin-left:10px;">
                                        NSE Bhavcopy (EOD) · Weekly Thursday expiry · Runs in background
                                    </span>
                                </div>
                            </div>
                            <div style="display:flex;gap:10px;flex-wrap:wrap;align-items:flex-end;
                                        background:#0a0a18;border-radius:8px;padding:10px 14px;
                                        margin-bottom:10px;">
                                <div>
                                    <div style="color:{MUTED};font-size:9px;margin-bottom:4px;
                                                text-transform:uppercase;">Strategy</div>
                                    <select id="bt-type"
                                            style="background:#12122a;color:{WHITE};border:1px solid #333;
                                                   border-radius:5px;padding:5px 10px;font-size:12px;">
                                        <option value="SHORT_STRADDLE">Short Straddle</option>
                                        <option value="SHORT_STRANGLE">Short Strangle</option>
                                        <option value="IRON_CONDOR">Iron Condor</option>
                                        <option value="BULL_PUT_SPREAD">Bull Put Spread</option>
                                        <option value="BEAR_CALL_SPREAD">Bear Call Spread</option>
                                    </select>
                                </div>
                                <div>
                                    <div style="color:{MUTED};font-size:9px;margin-bottom:4px;
                                                text-transform:uppercase;">Period</div>
                                    <select id="bt-days"
                                            style="background:#12122a;color:{WHITE};border:1px solid #333;
                                                   border-radius:5px;padding:5px 10px;font-size:12px;">
                                        <option value="180">6 months</option>
                                        <option value="365" selected>1 year</option>
                                        <option value="730">2 years</option>
                                    </select>
                                </div>
                                <div>
                                    <div style="color:{MUTED};font-size:9px;margin-bottom:4px;
                                                text-transform:uppercase;">Stop Loss</div>
                                    <select id="bt-sl"
                                            style="background:#12122a;color:{WHITE};border:1px solid #333;
                                                   border-radius:5px;padding:5px 10px;font-size:12px;">
                                        <option value="1.5">1.5× premium</option>
                                        <option value="2.0" selected>2× premium</option>
                                        <option value="3.0">3× premium</option>
                                        <option value="99">No SL</option>
                                    </select>
                                </div>
                                <button onclick="runBt()"
                                        style="background:linear-gradient(135deg,{ACCENT},{ACCENT}aa);
                                               color:#000;border:none;border-radius:7px;
                                               padding:8px 22px;font-size:12px;font-weight:800;
                                               cursor:pointer;letter-spacing:0.5px;">▶ Run</button>
                                <div id="bt-status"
                                     style="color:{YELLOW};font-size:11px;font-weight:600;
                                            display:flex;align-items:center;gap:6px;">
                                </div>
                            </div>
                            <div id="bt-results"></div>
                        </div>

                        <script>
                        // ── Strategy pill selector ───────────────────────────────
                        function selectStrat(idx) {{
                            var total = {_num_strats};
                            for (var i = 0; i < total; i++) {{
                                var panel = document.getElementById('strat-panel-' + i);
                                var pill  = document.getElementById('pill-' + i);
                                if (panel) panel.style.display = (i === idx) ? '' : 'none';
                                if (pill) {{
                                    pill.style.background = (i === idx) ? '#1e2a3a' : '#12121e';
                                    pill.style.color      = (i === idx) ? '#4fc3f7' : '#888';
                                    pill.style.border     = '1px solid ' + (i === idx ? '#4fc3f7' : '#2a2a4a');
                                }}
                            }}
                        }}

                        // ── Standalone backtest (no DataServer needed) ───────────
                        function runBt() {{
                            var type = document.getElementById('bt-type').value;
                            var days = parseInt(document.getElementById('bt-days').value);
                            var sl   = parseFloat(document.getElementById('bt-sl').value);
                            var stat = document.getElementById('bt-status');
                            var res  = document.getElementById('bt-results');
                            stat.innerHTML = '<span style="animation:pulse 1s infinite;">⏳</span> Running — may take 30-90 s…';
                            res.innerHTML = '';

                            // POST to the local HTTP server's /bt_run endpoint
                            fetch('/bt_run', {{
                                method: 'POST',
                                headers: {{'Content-Type': 'application/json'}},
                                body: JSON.stringify({{type: type, days: days, sl: sl}})
                            }}).then(function(r) {{ return r.text(); }})
                            .then(function(t) {{
                                stat.textContent = '⏳ Processing…';
                                pollBt(type);
                            }}).catch(function() {{
                                // Fallback: if no custom handler, notify
                                stat.textContent = '⚠ Refresh after 60s — running in background';
                            }});
                        }}

                        function pollBt(type) {{
                            var maxTries = 40, tries = 0;
                            var poll = setInterval(function() {{
                                tries++;
                                fetch('/bt_' + type + '.html?t=' + Date.now())
                                .then(function(r) {{
                                    if (r.ok) {{
                                        return r.text().then(function(html) {{
                                            if (html && html.length > 50) {{
                                                clearInterval(poll);
                                                document.getElementById('bt-status').textContent = '✓ Complete';
                                                document.getElementById('bt-results').innerHTML = html;
                                            }}
                                        }});
                                    }}
                                }}).catch(function() {{}});
                                if (tries >= maxTries) {{
                                    clearInterval(poll);
                                    document.getElementById('bt-status').textContent =
                                        '⚠ Still running — refresh the tab manually';
                                }}
                            }}, 3000);
                        }}

                        // ── Track strategy ───────────────────────────────────────
                        function trackStrategy(payloadStr) {{
                            var payload;
                            try {{ payload = JSON.parse(payloadStr.replace(/&quot;/g, '"')); }}
                            catch(e) {{ alert('Parse error: ' + e); return; }}

                            // Use relative path — works locally and via Cloudflare tunnel
                            var targets = ['/api/track_strategy'];
                            var attempt = function(idx) {{
                                if (idx >= targets.length) {{
                                    alert('Could not save — DataServer and local handler not available.');
                                    return;
                                }}
                                fetch(targets[idx], {{
                                    method: 'POST',
                                    headers: {{'Content-Type': 'application/json'}},
                                    body: JSON.stringify(payload)
                                }}).then(function(r) {{ return r.json(); }})
                                .then(function(d) {{
                                    if (d.ok) {{
                                        var btn = event.target;
                                        btn.textContent = '✓ Tracked!';
                                        btn.style.background = '#388e3c';
                                        setTimeout(function() {{
                                            btn.textContent = '⊕ Track Live';
                                            btn.style.background = '#66bb6a';
                                        }}, 2000);
                                    }}
                                }}).catch(function() {{ attempt(idx + 1); }});
                            }};
                            attempt(0);
                        }}
                        </script>'''

                    except Exception as e:
                        import traceback
                        strategy_tab_html = f'<div class="card"><p style="color:#ff4444;">Strategy Engine Error: {str(e)}</p><pre style="color:#888;font-size:10px;">{traceback.format_exc()[:600]}</pre></div>'


                        import plotly
                        _plotlyjs_cdn = '<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>'

                        # ── Market context for strategy generation ──
                        _T_strat  = self.analytics.get_time_to_expiry(near_exp)
                        _iv_strat = (vol_intel.get('iv', 15) if vol_intel else 15)
                        _market_ctx = {
                            'T':          _T_strat,
                            'iv':         _iv_strat,
                            'vrp':        vol_intel.get('vrp', 0) if vol_intel else 0,
                            'regime':     vol_intel.get('regime', 'NORMAL') if vol_intel else 'NORMAL',
                            'atm_iv':     _iv_strat,
                            'call_wall':  float(seller.get('call_wall', 0)) if seller else 0,
                            'put_wall':   float(seller.get('put_wall', 0)) if seller else 0,
                            'em':         float(seller.get('em', spot * 0.015)) if seller else spot * 0.015,
                            'oi_pressure': oi_pressure,
                            'DTE':        seller.get('DTE', 7) if seller else 7,
                        }
                        _call_wall_s = _market_ctx['call_wall']
                        _put_wall_s  = _market_ctx['put_wall']
                        _dte_s       = _market_ctx['DTE']
                        _iv_pct      = _iv_strat
                        _vrp_s       = _market_ctx['vrp']
                        _regime_s    = _market_ctx['regime']

                        # ── Context bar colors ──
                        _dte_clr = RED if _dte_s <= 1 else YELLOW if _dte_s <= 3 else GREEN
                        _regime_clr = YELLOW if _regime_s == 'COMPRESSION' else RED if _regime_s == 'EXPANSION' else GREEN
                        _vrp_clr = GREEN if _vrp_s > 2 else RED if _vrp_s < -2 else YELLOW
                        _intra_bias = 'SELL-PREMIUM' if _vrp_s > 3 else 'AVOID SELL' if _vrp_s < -2 else 'NEUTRAL'
                        _bias_clr   = GREEN if _intra_bias == 'SELL-PREMIUM' else RED if 'AVOID' in _intra_bias else YELLOW

                        # ── Strategy generation with payoff charts ──
                        gen = SmartStrategyGenerator(spot, df_chain, _market_ctx, near_exp)
                        _strategies = gen.generate()

                        strat_cards_html = ''
                        for _s_obj in _strategies[:4]:   # Top 4 by score
                            _payload = json.dumps(_s_obj.to_dict()).replace('"', '&quot;')
                            strat_cards_html += build_strategy_card_html(
                                _s_obj, spot, _T_strat, _iv_strat / 100,
                                call_wall=_call_wall_s, put_wall=_put_wall_s,
                                payload_json=_payload
                            )

                        # ── Live P&L tracker ──
                        tracked  = _strategy_manager.get_all_active_strategies()
                        closed_h = _strategy_manager.get_closed_strategies(limit=5)
                        summary  = _strategy_manager.get_summary()
                        tracked_html = generate_tracked_html(tracked, df_chain)

                        # Closed trades mini-table
                        _closed_rows = ''
                        for _ct in closed_h:
                            _pnl = _ct.get('pnl', 0) or 0
                            _pnl_clr = GREEN if _pnl > 0 else RED
                            _closed_rows += (
                                f'<tr><td style="padding:4px 8px;color:#aaa;">{_ct.get("name","")}</td>'
                                f'<td style="padding:4px 8px;color:{MUTED};">{_ct.get("exit_time_str","")[:10]}</td>'
                                f'<td style="padding:4px 8px;text-align:right;font-weight:700;color:{_pnl_clr};">'
                                f'{"+" if _pnl > 0 else ""}₹{_pnl:,.0f}</td></tr>'
                            )
                        _total_pnl    = summary.get('total_realized_pnl', 0)
                        _total_pnl_clr = GREEN if _total_pnl >= 0 else RED

                        buyer_setup_html = ""
                        if buyer_setup:
                            _bu = buyer_setup
                            _bc_color = GREEN if 'BULLISH' in _bu['verdict_base'] else RED if 'BEARISH' in _bu['verdict_base'] else YELLOW
                            buyer_setup_html = f'''
                            <div class="card" style="margin-bottom:12px; border:1px solid {_bc_color}66; background:rgba(20,20,35,0.7);">
                                <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px;">
                                    <div style="color:{_bc_color};font-size:14px;font-weight:900;letter-spacing:1px;">
                                        ⚡ OPTION BUYER SETUP
                                    </div>
                                    <span style="color:{WHITE};font-size:12px;font-weight:700;background:{_bc_color}33;padding:2px 8px;border-radius:4px;">
                                        {_bu['timing']}
                                    </span>
                                </div>
                                <div style="display:flex;justify-content:space-between;border-bottom:1px solid #333;padding-bottom:10px;margin-bottom:10px;">
                                    <div>
                                        <div style="color:{MUTED};font-size:11px;text-transform:uppercase;">Action</div>
                                        <div style="color:{WHITE};font-size:16px;font-weight:900;">{_bu['action']}</div>
                                    </div>
                                    <div style="text-align:right;">
                                        <div style="color:{MUTED};font-size:11px;text-transform:uppercase;">Entry Limit</div>
                                        <div style="color:{ACCENT};font-size:16px;font-weight:900;">₹{_bu['entry_limit']}</div>
                                    </div>
                                </div>
                                <div style="display:flex;justify-content:space-between;gap:10px;">
                                    <div style="flex:1;background:#0d0d1a;padding:8px;border-radius:6px;text-align:center;border:1px solid {GREEN}33;">
                                        <div style="color:{MUTED};font-size:10px;">Target 1 (0.5%)</div>
                                        <div style="color:{GREEN};font-weight:700;">₹{_bu['t1_premium']}</div>
                                    </div>
                                    <div style="flex:1;background:#0d0d1a;padding:8px;border-radius:6px;text-align:center;border:1px solid {GREEN}55;">
                                        <div style="color:{MUTED};font-size:10px;">Target 2 (1.0%)</div>
                                        <div style="color:{GREEN};font-weight:700;">₹{_bu['t2_premium']}</div>
                                    </div>
                                    <div style="flex:1;background:#0d0d1a;padding:8px;border-radius:6px;text-align:center;border:1px solid {RED}44;">
                                        <div style="color:{MUTED};font-size:10px;">Stop Loss</div>
                                        <div style="color:{RED};font-weight:700;">₹{_bu['sl_premium']}</div>
                                    </div>
                                </div>
                                <div style="margin-top:10px;text-align:center;color:{MUTED};font-size:10px;">
                                    VWAP Level: {_bu['vwap_hard_level']} | Delta Proxy: {_bu['delta_exposure']}
                                </div>
                            </div>
                            '''

                        strategy_tab_html = f'''
                        <!-- Context Bar -->
                        <div style="display:flex;flex-wrap:wrap;gap:8px;align-items:center;
                                    margin-bottom:12px;padding:10px 14px;background:#12122a;
                                    border-radius:8px;border:1px solid #2a2a4a;">
                            <span style="color:{ACCENT};font-size:12px;font-weight:700;margin-right:4px;">
                                STRATEGY ENGINE</span>
                            <span style="background:{_dte_clr}22;color:{_dte_clr};font-size:11px;
                                         font-weight:700;padding:2px 10px;border-radius:10px;
                                         border:1px solid {_dte_clr}44;">{_dte_s}d to expiry</span>
                            <span style="background:{_regime_clr}22;color:{_regime_clr};font-size:11px;
                                         font-weight:700;padding:2px 10px;border-radius:10px;
                                         border:1px solid {_regime_clr}44;">Regime: {_regime_s}</span>
                            <span style="background:{_vrp_clr}22;color:{_vrp_clr};font-size:11px;
                                         padding:2px 10px;border-radius:10px;border:1px solid {_vrp_clr}44;">
                                VRP {_vrp_s:+.1f}%</span>
                            <span style="background:#1a1a2e;color:{MUTED};font-size:11px;
                                         padding:2px 10px;border-radius:10px;">ATM IV {_iv_pct:.1f}%</span>
                            <span style="background:{_bias_clr}22;color:{_bias_clr};font-size:12px;
                                         font-weight:900;padding:2px 14px;border-radius:10px;
                                         border:1px solid {_bias_clr}55;margin-left:auto;">
                                ⚡ {_intra_bias}</span>
                        </div>

                        <!-- Main 2-column layout -->
                        <div style="display:grid;grid-template-columns:1fr 1.6fr;gap:14px;">
                            <!-- LEFT: P&L + Buyer Setup -->
                            <div>
                                {buyer_setup_html}
                                <div class="card" style="margin-bottom:12px;">
                                    <div style="display:flex;justify-content:space-between;
                                                align-items:center;margin-bottom:12px;">
                                        <div style="color:{ACCENT};font-size:13px;font-weight:700;
                                                    letter-spacing:2px;">LIVE P&amp;L TRACKER</div>
                                        <span style="color:{_total_pnl_clr};font-size:13px;font-weight:700;">
                                            Realized: {"+" if _total_pnl >= 0 else ""}₹{_total_pnl:,.0f}</span>
                                    </div>
                                    <div id="pnlTrackerContainer">
                                        {tracked_html}
                                    </div>
                                    <div style="display:flex;gap:8px;margin-top:10px;flex-wrap:wrap;">
                                        <input type="time" id="histScrubTime"
                                               style="background:#111;color:{WHITE};border:1px solid #444;
                                                      border-radius:4px;padding:3px 6px;font-size:12px;">
                                        <button onclick="fetchHistoricalPNL()"
                                                style="background:{ACCENT};color:#000;border:none;
                                                       border-radius:4px;padding:4px 10px;font-size:11px;
                                                       font-weight:600;cursor:pointer;">Scrub Time</button>
                                        <button onclick="resetToLivePNL()"
                                                style="background:#334;color:{WHITE};border:none;
                                                       border-radius:4px;padding:4px 10px;font-size:11px;
                                                       cursor:pointer;">Live</button>
                                    </div>
                                </div>

                                {f"""
                                <div class="card">
                                    <div style="color:{ACCENT};font-size:12px;font-weight:700;
                                                letter-spacing:2px;margin-bottom:8px;">CLOSED TRADES</div>
                                    <table style="width:100%;border-collapse:collapse;font-size:11px;">
                                        <thead><tr style="color:{MUTED};border-bottom:1px solid #333;">
                                            <th style="text-align:left;padding:4px 8px;">Strategy</th>
                                            <th style="text-align:left;padding:4px 8px;">Date</th>
                                            <th style="text-align:right;padding:4px 8px;">P&amp;L</th>
                                        </tr></thead>
                                        <tbody>{_closed_rows if _closed_rows else
                                               f'<tr><td colspan="3" style="padding:6px 8px;color:#555;">No closed trades yet.</td></tr>'}
                                        </tbody>
                                    </table>
                                </div>"""}
                            </div>

                            <!-- RIGHT: Strategy Builder + Payoff Charts -->
                            <div>
                                <div class="card" style="margin-bottom:12px;">
                                    <div style="color:{ACCENT};font-size:13px;font-weight:700;
                                                letter-spacing:2px;margin-bottom:12px;">
                                        RECOMMENDED STRATEGIES
                                        <span style="color:{MUTED};font-size:10px;font-weight:400;margin-left:8px;">
                                            Top {min(4, len(_strategies))} by score — payoff at 1 lot
                                        </span>
                                    </div>
                                    <div style="max-height:680px;overflow-y:auto;padding-right:6px;">
                                        {strat_cards_html or
                                         f'<div style="color:#666;">No strategies generated for current context.</div>'}
                                    </div>
                                </div>
                            </div>
                        </div>

                        <!-- BACKTEST PANEL (full width) -->
                        <div class="card" style="margin-top:14px;" id="bt-panel">
                            <div style="display:flex;justify-content:space-between;
                                        align-items:center;margin-bottom:12px;">
                                <div style="color:{ACCENT};font-size:13px;font-weight:700;letter-spacing:2px;">
                                    📊 STRATEGY BACKTESTER
                                    <span style="color:{MUTED};font-size:10px;font-weight:400;margin-left:8px;">
                                        NSE Bhavcopy — EOD prices — weekly Thursday expiry
                                    </span>
                                </div>
                            </div>
                            <div style="display:flex;gap:12px;flex-wrap:wrap;align-items:flex-end;
                                        background:#0d0d1e;padding:12px;border-radius:8px;margin-bottom:12px;">
                                <div>
                                    <div style="color:{MUTED};font-size:10px;margin-bottom:4px;">Strategy</div>
                                    <select id="bt-type" style="background:#12122a;color:{WHITE};
                                                                 border:1px solid #444;border-radius:4px;
                                                                 padding:5px 10px;font-size:12px;">
                                        <option value="SHORT_STRADDLE">Short Straddle</option>
                                        <option value="SHORT_STRANGLE">Short Strangle</option>
                                        <option value="IRON_CONDOR">Iron Condor</option>
                                        <option value="BULL_PUT_SPREAD">Bull Put Spread</option>
                                        <option value="BEAR_CALL_SPREAD">Bear Call Spread</option>
                                    </select>
                                </div>
                                <div>
                                    <div style="color:{MUTED};font-size:10px;margin-bottom:4px;">Period</div>
                                    <select id="bt-days" style="background:#12122a;color:{WHITE};
                                                                  border:1px solid #444;border-radius:4px;
                                                                  padding:5px 10px;font-size:12px;">
                                        <option value="180">6 months</option>
                                        <option value="365" selected>1 year</option>
                                        <option value="730">2 years</option>
                                    </select>
                                </div>
                                <div>
                                    <div style="color:{MUTED};font-size:10px;margin-bottom:4px;">Stop Loss</div>
                                    <select id="bt-sl" style="background:#12122a;color:{WHITE};
                                                               border:1px solid #444;border-radius:4px;
                                                               padding:5px 10px;font-size:12px;">
                                        <option value="1.5">1.5x premium</option>
                                        <option value="2.0" selected>2x premium</option>
                                        <option value="3.0">3x premium</option>
                                        <option value="100">No SL</option>
                                    </select>
                                </div>
                                <button onclick="runBacktest()"
                                        style="background:{ACCENT};color:#000;border:none;
                                               border-radius:6px;padding:7px 20px;font-size:12px;
                                               font-weight:700;cursor:pointer;">▶ Run Backtest</button>
                                <div id="bt-status" style="color:{YELLOW};font-size:12px;"></div>
                            </div>
                            <div id="bt-results"></div>
                        </div>

                        <script>
                        function runBacktest() {{
                            var type = document.getElementById('bt-type').value;
                            var days = document.getElementById('bt-days').value;
                            var sl   = document.getElementById('bt-sl').value;
                            document.getElementById('bt-status').textContent = '⏳ Running backtest...';
                            document.getElementById('bt-results').innerHTML = '';
                            fetch('/api/backtest', {{
                                method: 'POST',
                                headers: {{'Content-Type': 'application/json'}},
                                body: JSON.stringify({{strategy_type: type, days: parseInt(days),
                                                      stop_loss_mult: parseFloat(sl)}})
                            }}).then(r => r.json()).then(d => {{
                                if (d.ok) {{ pollBtStatus(type); }}
                                else {{ document.getElementById('bt-status').textContent = 'Error: ' + d.error; }}
                            }}).catch(e => {{
                                document.getElementById('bt-status').textContent = 'Backtest requires DataServer.';
                            }});
                        }}

                        function pollBtStatus(type) {{
                            var poll = setInterval(function() {{
                                fetch('/api/backtest_status?type=' + type)
                                .then(r => r.json()).then(d => {{
                                    if (d.status === 'DONE') {{
                                        clearInterval(poll);
                                        document.getElementById('bt-status').textContent = '✓ Done';
                                        document.getElementById('bt-results').innerHTML = d.html;
                                    }} else if (d.status === 'RUNNING') {{
                                        document.getElementById('bt-status').textContent = '⏳ Running...';
                                    }}
                                }}).catch(() => clearInterval(poll));
                            }}, 3000);
                        }}

                        function trackStrategy(payloadStr) {{
                            var payload;
                            try {{ payload = JSON.parse(payloadStr.replace(/&quot;/g, '"')); }}
                            catch(e) {{ alert('JSON parse error'); return; }}
                            fetch('/api/track_strategy', {{
                                method: 'POST',
                                headers: {{'Content-Type': 'application/json'}},
                                body: JSON.stringify(payload)
                            }}).then(r => r.json()).then(d => {{
                                if (d.ok) {{ alert('Strategy tracked! ID: ' + d.id); }}
                                else {{ alert('Error: ' + (d.error || 'Unknown')); }}
                            }}).catch(() => {{
                                alert('DataServer not running — strategy saved locally only.');
                            }});
                        }}
                        </script>'''

                    # ──────────────────────────────────────────
                    #  REGIME TAB HTML 
                    # ──────────────────────────────────────────
                    if regime_snapshot:
                        regime_tab_html = f'''
                        <div style="display:flex; flex-direction:column; gap:16px;">
                            <div class="card" style="border-left: 4px solid {ACCENT};">
                                <h2 style="color:{ACCENT}; margin-bottom: 8px;">MARKET REGIME: {regime_snapshot['regime']['name']}</h2>
                                <p style="color:{WHITE}; font-size:14px; margin-bottom: 12px;">{regime_snapshot['regime']['description']}</p>
                                <div style="color:{MUTED}; font-size:12px;">ACTION BIAS: <strong style="color:{WHITE};">{regime_snapshot['regime']['bias']}</strong> &nbsp;|&nbsp; VOL ACTION: <strong style="color:{GREEN}">{regime_snapshot['regime']['vol_action']}</strong></div>
                            </div>
                            
                            <div style="display:grid; grid-template-columns: 1fr 1fr 1fr; gap:12px;">
                                <div class="metric-box">
                                    <div class="metric-label">Consensus Realized Vol</div>
                                    <div style="font-size:24px; font-weight:700; color:{WHITE};">{regime_snapshot['rv']['consensus']:.2f}%</div>
                                    <div class="metric-sub">Trend: {regime_snapshot['rv']['trend']}</div>
                                </div>
                                <div class="metric-box">
                                    <div class="metric-label">Volatility Risk Premium</div>
                                    <div style="font-size:24px; font-weight:700; color:{GREEN if regime_snapshot['vrp']['iv_rv'] > 0 else RED};">{regime_snapshot['vrp']['iv_rv']:+.2f}%</div>
                                    <div class="metric-sub">IV vs Consensus RV</div>
                                </div>
                                <div class="metric-box">
                                    <div class="metric-label">Historical Vol (20d)</div>
                                    <div style="font-size:24px; font-weight:700; color:{WHITE};">{regime_snapshot['hv']['20d']:.2f}%</div>
                                    <div class="metric-sub">Percentile: {regime_snapshot['hv']['percentile']:.0f}%</div>
                                </div>
                            </div>
                            
                            <div class="card">
                                <h3 style="color:{ACCENT}; font-size:14px; margin-bottom:12px; border-bottom: 1px solid #333; padding-bottom:6px;">Realized Volatility Term Structure</h3>
                                <table class="data-table">
                                    <thead><tr><th style="text-align:left;">Estimator</th><th>5-Day</th><th>10-Day</th><th>20-Day</th><th>60-Day</th></tr></thead>
                                    <tbody>
                                        <tr>
                                            <td style="text-align:left; color:{WHITE};">Close-to-Close</td>
                                            <td>{regime_snapshot['rv']['5d']:.2f}%</td>
                                            <td>{regime_snapshot['rv']['10d']:.2f}%</td>
                                            <td>{regime_snapshot['rv']['20d']:.2f}%</td>
                                            <td>{regime_snapshot['rv']['60d']:.2f}%</td>
                                        </tr>
                                        <tr>
                                            <td style="text-align:left; color:{WHITE};">Parkinson (High/Low)</td>
                                            <td>-</td><td>-</td><td>{regime_snapshot['rv']['parkinson_20d']:.2f}%</td><td>-</td>
                                        </tr>
                                        <tr>
                                            <td style="text-align:left; color:{WHITE};">Garman-Klass (OHLC)</td>
                                            <td>-</td><td>-</td><td>{regime_snapshot['rv']['garman_klass_20d']:.2f}%</td><td>-</td>
                                        </tr>
                                        <tr>
                                            <td style="text-align:left; color:{WHITE};">Yang-Zhang (Gap+OHLC)</td>
                                            <td>-</td><td>-</td><td>{regime_snapshot['rv']['yang_zhang_20d']:.2f}%</td><td>-</td>
                                        </tr>
                                        <tr style="background:rgba(79,195,247,0.1);">
                                            <td style="text-align:left; font-weight:600; color:{ACCENT};">CONSENSUS (Avg)</td>
                                            <td colspan="4" style="text-align:center; font-weight:600; font-size:16px; color:{ACCENT};">{regime_snapshot['rv']['consensus']:.2f}%</td>
                                        </tr>
                                    </tbody>
                                </table>
                            </div>
                            
                            <div class="card" style="margin-top:4px;">
                                <div style="color:{MUTED}; font-size:12px; text-align:center;">Intraday Consensus RV: {regime_snapshot['rv']['intraday']:.2f}%</div>
                            </div>
                        </div>
                        '''
                    else:
                        regime_tab_html = "<div style='padding:40px; color:#ff4444; text-align:center;'>Waiting for sufficient daily history data (requires 20+ trading days).</div>"

                    # ── CONFLUENCE HEADER HTML ──
                    vd = verdict_data
                    v_color = GREEN if 'BULLISH' in vd['verdict'] else RED if 'BEARISH' in vd['verdict'] else YELLOW
                    verdict_html = f'''
                    <div style="display:flex; flex-direction:column; align-items:center; background:#0d0d1e; padding:8px 16px; border-radius:8px; border:1px solid {v_color}44;">
                        <div style="font-size:10px; color:{MUTED}; font-weight:700; text-transform:uppercase; letter-spacing:1px; margin-bottom:2px;">CONFLUENCE VERDICT</div>
                        <div style="font-size:16px; font-weight:900; color:{v_color}; text-shadow: 0 0 10px {v_color}44;">{vd['verdict']}</div>
                        <div style="font-size:10px; color:{WHITE}; margin-top:2px;">Confidence: {vd['confidence']:.0%} | Score: {vd['score']:.2f}</div>
                    </div>
                    '''

                    # Write fragment file that the running page will fetch
                    frag_path = html_path.replace('.html', '_fragment.html')
                    fragment_html = f'''
<div id="frag-regime">{regime_tab_html}</div>
<div id="frag-iv">{iv_tab_html}</div>
<div id="frag-vol">{vol_tab_html}</div>
<div id="frag-chain">{chain_tab_html}</div>
<div id="frag-prob">{prob_tab_html}</div>
<div id="frag-strat">{strategy_tab_html}</div>
<div id="frag-spot" data-spot="{spot:.0f}" data-time="{now_str}">
    <span id="frag-verdict-transfer" style="display:none;">{verdict_html}</span>
</div>'''
                    with open(frag_path, 'w', encoding='utf-8') as f:
                        f.write(fragment_html)

                    if first_run:
                        # Write full page only once (stable shell + live JS fetcher)
                        full_html = f'''<!DOCTYPE html>
<html><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<link rel="manifest" href="/static/manifest.json">
<meta name="theme-color" content="#0d1117">
<title>Unified Vol Dashboard | {self.symbol}</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap" rel="stylesheet">
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js" charset="utf-8"></script>
<style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{ background:{DARK_BG}; font-family:Inter,'Segoe UI',sans-serif; color:{WHITE}; padding:8px 16px; }}
    .header {{ display:flex; justify-content:space-between; align-items:center; padding:10px 20px;
               background:{CARD_BG}; border-radius:12px; margin-bottom:8px; border:1px solid #2a2a4a; }}
    .title {{ font-size:16px; font-weight:700; color:{ACCENT}; letter-spacing:2px; }}
    .tab-bar {{ display:flex; gap:4px; margin-bottom:8px; }}
    .tab-btn {{ padding:10px 24px; background:{CARD_BG}; border:1px solid #2a2a4a; border-bottom:none;
                border-radius:10px 10px 0 0; cursor:pointer; color:{MUTED}; font-size:13px;
                font-weight:600; letter-spacing:1px; transition:all 0.2s; }}
    .tab-btn:hover {{ color:{WHITE}; background:#1e1e38; }}
    .tab-btn.active {{ color:{ACCENT}; background:#12122a; border-color:{ACCENT}; border-bottom:2px solid {ACCENT}; }}
    .tab-content {{ display:none; }}
    .tab-content.active {{ display:block; }}
    .card {{ background:{CARD_BG}; border-radius:12px; padding:16px; border:1px solid #2a2a4a; }}
    .metric-box {{ background:#12122a; border-radius:8px; padding:12px 16px; text-align:center; border:1px solid #2a2a4a; flex:1; min-width:130px; }}
    .metric-label {{ color:{MUTED}; font-size:11px; font-weight:600; text-transform:uppercase; letter-spacing:1px; }}
    .metric-sub {{ color:{MUTED}; font-size:11px; }}
    .action-bar {{ display:flex; align-items:center; padding:10px 16px; background:#12122a; border-radius:8px; border:1px solid #2a2a4a; flex-wrap:wrap; gap:8px; }}
    .data-table {{ width:100%; border-collapse:collapse; font-size:12px; }}
    .data-table th {{ color:{MUTED}; font-size:11px; text-transform:uppercase; padding:6px 8px; border-bottom:1px solid #2a2a4a; text-align:right; }}
    .data-table td {{ padding:5px 8px; border-bottom:1px solid #1a1a2a; color:{WHITE}; text-align:right; }}
    .data-table tr:hover {{ background:rgba(79,195,247,0.04); }}
    #live-pulse {{ display:inline-block; width:8px; height:8px; border-radius:50%; background:{GREEN};
                   animation:pulse 1.5s infinite; margin-right:6px; }}
    @keyframes pulse {{ 0% {{opacity:1;}} 50% {{opacity:0.2;}} 100% {{opacity:1;}} }}
    #refresh-indicator {{ opacity:0; transition:opacity 0.3s; }}
    #refresh-indicator.show {{ opacity:1; }}
    
    @media (max-width: 768px) {{
        body {{ padding: 4px; font-size: 13px; }}
        .header {{ flex-direction: column; align-items: flex-start; gap: 10px; }}
        .header > div {{ flex-direction: column; align-items: flex-start !important; gap: 5px !important; }}
        .title {{ font-size: 14px; }}
        #spot-display {{ font-size: 26px !important; }}
        .tab-bar {{ overflow-x: auto; white-space: nowrap; -webkit-overflow-scrolling: touch; }}
        .tab-btn {{ flex: 0 0 auto; padding: 12px 16px; min-height: 44px; }}
        .card > div {{ grid-template-columns: 1fr !important; }}
        .metric-box {{ min-width: 100%; margin-bottom: 8px; }}
        .data-table {{ display: block; overflow-x: auto; white-space: nowrap; }}
        .plotly-graph-div {{ width: 100% !important; min-height: 300px; }}
        button, input {{ min-height: 44px; font-size: 16px; }}
        #verdict-container {{ margin-left: 0 !important; width: 100%; }}
        #verdict-container > div {{ width: 100%; }}
    }}
</style>
</head><body>
    <div class="header">
        <div style="display:flex;align-items:center;gap:20px;">
            <div><span class="title">UNIFIED VOLATILITY DASHBOARD</span>
                 <span style="color:{MUTED};font-size:13px;">  |  {self.symbol}</span></div>
        </div>
        <div style="display:flex;align-items:center;gap:20px;">
            <div id="spot-display" style="font-size:22px;font-weight:900;color:{ACCENT};">SPOT: {spot:,.2f}</div>
            <div style="display:flex;flex-direction:column;align-items:center;border-left:1px solid #333;padding-left:15px;">
                <div style="font-size:10px;color:{MUTED};font-weight:700;text-transform:uppercase;">Intraday Momentum</div>
                <div style="font-size:14px;font-weight:700;color:{GREEN if momentum_data['status']=='LONG' else RED if momentum_data['status']=='SHORT' else YELLOW};">
                    {momentum_data['status']} (V:{momentum_data['vwap']} | E:{momentum_data['ema']})
                </div>
            </div>
            <div id="verdict-container" style="margin-left:10px;">
                {verdict_html}
            </div>
        </div>
        <div>
            <span id="live-pulse"></span>
            <span style="color:{GREEN};font-size:12px;font-weight:700;">LIVE</span>
            <span id="time-display" style="color:{MUTED};font-size:11px;"> {now_str} | 15s refresh</span>
            <span id="refresh-indicator" style="color:{ACCENT};font-size:11px;margin-left:8px;">&#8635; Updating...</span>
        </div>
    </div>

    <div class="tab-bar">
        <div class="tab-btn active" onclick="switchTab('regime')">Regime Engine</div>
        <div class="tab-btn" onclick="switchTab('iv')">IV Surface</div>
        <div class="tab-btn" onclick="switchTab('vol')">Vol Intelligence</div>
        <div class="tab-btn" onclick="switchTab('chain')">Option Chain Analyser</div>
        <div class="tab-btn" onclick="switchTab('prob')">Prob Density</div>
        <div class="tab-btn" onclick="switchTab('strat')">Strategy Engine</div>
    </div>

    <div id="tab-regime" class="tab-content active">{regime_tab_html}</div>
    <div id="tab-iv" class="tab-content">{iv_tab_html}</div>
    <div id="tab-vol" class="tab-content">{vol_tab_html}</div>
    <div id="tab-chain" class="tab-content">{chain_tab_html}</div>
    <div id="tab-prob" class="tab-content">{prob_tab_html}</div>
    <div id="tab-strat" class="tab-content">{strategy_tab_html}</div>

    <script>
    var tabMap = {{'regime':0, 'iv':1,'vol':2,'chain':3,'prob':4, 'strat':5}};
    var activeTab = localStorage.getItem('volDashActiveTab') || 'regime';

    function switchTab(id) {{
        document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
        document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
        var el = document.getElementById('tab-' + id);
        if (el) el.classList.add('active');
        var idx = tabMap[id];
        if (idx !== undefined) {{
            document.querySelectorAll('.tab-btn')[idx].classList.add('active');
        }}
        activeTab = id;
        localStorage.setItem('volDashActiveTab', id);
    }}

    // Restore active tab on first load
    (function() {{
        var saved = localStorage.getItem('volDashActiveTab');
        if (saved && document.getElementById('tab-' + saved)) {{
            switchTab(saved);
        }}
    }})();

    // Seamless refresh — fetch fragment from DataServer (works locally and via Cloudflare tunnel)
    var FRAG_URL = '/fragment';

    function refreshContent() {{
        var indicator = document.getElementById('refresh-indicator');
        if (indicator) indicator.classList.add('show');

        var xhr = new XMLHttpRequest();
        xhr.open('GET', FRAG_URL + '?t=' + Date.now(), true);
        xhr.onload = function() {{
            if (xhr.status === 200) {{
                var tmp = document.createElement('div');
                tmp.innerHTML = xhr.responseText;
                // Swap each tab's content
                ['regime', 'iv', 'vol', 'chain', 'prob', 'strat'].forEach(function(id) {{
                    var fragEl = tmp.querySelector('#frag-' + id);
                    var tabEl = document.getElementById('tab-' + id);
                    if (fragEl && tabEl) {{
                        tabEl.innerHTML = fragEl.innerHTML;
                        executeScripts(tabEl);
                    }}
                }});
                // Update header spot + time
                var meta = tmp.querySelector('#frag-spot');
                if (meta) {{
                    var sVal = meta.getAttribute('data-spot');
                    var tVal = meta.getAttribute('data-time');
                    document.getElementById('spot-display').innerHTML = 'Spot: ' + Number(sVal).toLocaleString();
                    document.getElementById('time-display').innerHTML = tVal + ' | 15s refresh';
                }}
                // Re-activate current tab (restores button highlight after DOM swap)
                switchTab(activeTab);
            }}
            if (indicator) setTimeout(function(){{ indicator.classList.remove('show'); }}, 500);
        }};
        xhr.onerror = function() {{
            if (indicator) indicator.classList.remove('show');
        }};
        xhr.send();
    }}

    // Re-execute scripts injected via innerHTML (needed for Plotly charts to render)
    function executeScripts(container) {{
        container.querySelectorAll('script').forEach(function(old) {{
            var s = document.createElement('script');
            Array.from(old.attributes).forEach(function(a) {{ s.setAttribute(a.name, a.value); }});
            s.textContent = old.textContent;
            old.parentNode.replaceChild(s, old);
        }});
    }}

    window.isScrubbing = false;

    function trackStrategy(payloadStr) {{
        fetch('/api/track_strategy', {{
            method: 'POST',
            headers: {{'Content-Type': 'application/json'}},
            body: payloadStr
        }}).then(res => {{
            if(res.ok) {{
                console.log("Tracking started");
                if (!window.isScrubbing) refreshContent(); // instantly update UI
            }} else alert("Error starting tracking.");
        }}).catch(e => console.error("Error calling API:", e));
    }}
    
    function deleteStrategy(id) {{
        fetch('/api/close_strategy', {{
            method: 'POST',
            headers: {{'Content-Type': 'application/json'}},
            body: JSON.stringify({{id: id}})
        }}).then(res => {{
            if(res.ok) {{
                console.log("Strategy deleted");
                if (!window.isScrubbing) refreshContent(); // instantly update UI
            }}
        }});
    }}

    function fetchHistoricalPNL() {{
        const timeVal = document.getElementById('histScrubTime').value;
        if(!timeVal) return;
        window.isScrubbing = true;
        
        fetch(`/api/strategy_pnl_at?time=${{timeVal}}`) 
        .then(r => r.text())
        .then(html => {{
            document.getElementById('pnlTrackerContainer').innerHTML = html;
            document.getElementById('pnlTrackerContainer').style.boxShadow = "inset 0 0 10px rgba(255,165,0,0.3)"; 
            document.getElementById('pnlTrackerContainer').style.padding = "10px";
            document.getElementById('pnlTrackerContainer').style.borderRadius = "8px";
        }});
    }}

    function resetToLivePNL() {{
        window.isScrubbing = false;
        document.getElementById('histScrubTime').value = '';
        document.getElementById('pnlTrackerContainer').style.boxShadow = "none";
        document.getElementById('pnlTrackerContainer').style.padding = "0";
        refreshContent();
    }}

    // Real-Time WebSocket integration with DataHub
    var ws;
    function connectDataHub() {{
        var host = window.location.hostname;
        var port = window.location.port;
        var protocol = window.location.protocol === 'https:' ? 'wss://' : 'ws://';
        if (host === 'localhost' || host === '127.0.0.1') {{
            port = '8082';
        }}
        var wsUrl = protocol + host + (port ? ':' + port : '') + '/stream';
        ws = new WebSocket(wsUrl);
        
        ws.onopen = function() {{
            console.log("Connected to DataHub Real-Time Stream");
            document.getElementById('live-pulse').style.background = '{GREEN}';
            document.getElementById('time-display').innerHTML = 'CONNECTED | REAL-TIME';
        }};
        
        ws.onmessage = function(event) {{
            var msg = JSON.parse(event.data);
            if (msg.type === 'tick') {{
                // Update Spot Price instantly
                document.getElementById('spot-display').innerHTML = 'SPOT: ' + Number(msg.spot).toLocaleString(undefined, {{minimumFractionDigits: 2}});
                document.getElementById('spot-display').style.transition = 'color 0.2s';
                document.getElementById('spot-display').style.color = '{WHITE}';
                setTimeout(() => {{ document.getElementById('spot-display').style.color = '{ACCENT}'; }}, 200);
            }} else if (msg.type === 'chain' || msg.type === 'init') {{
                // Refresh full dashboard content when option chain updates (or on init)
                if (!window.isScrubbing) refreshContent();
            }}
        }};
        
        ws.onclose = function() {{
            console.log("Disconnected from DataHub. Retrying in 5s...");
            document.getElementById('live-pulse').style.background = '{RED}';
            document.getElementById('time-display').innerHTML = 'OFFLINE | RECONNECTING...';
            setTimeout(connectDataHub, 5000);
        }};
    }}

    // Start WebSocket connection
    connectDataHub();
    
    if ('serviceWorker' in navigator) {{
        window.addEventListener('load', function() {{
            navigator.serviceWorker.register('/static/sw.js').then(function(registration) {{
                console.log('ServiceWorker registration successful with scope: ', registration.scope);
            }}, function(err) {{
                console.log('ServiceWorker registration failed: ', err);
            }});
        }});
    }}
    </script>
</body></html>'''
                        with open(html_path, 'w', encoding='utf-8') as f:
                            f.write(full_html)
                        _dashboard_url = f'{_base_url}/unified_dashboard.html'
                        webbrowser.open(_dashboard_url)
                        first_run = False
                        print(f"  Dashboard opened: {_dashboard_url}")

                    print(f"  [{now_str}] Updated | {pred['direction']} ({pred['confidence']:.0%}) | Spot:{spot:.0f}")
                    import gc; gc.collect()  # free Plotly figure memory
                    time.sleep(15)

                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    print(f"  Error: {e} — retrying in 5s...")
                    time.sleep(5)

        except KeyboardInterrupt:
            print("\n  Dashboard stopped.")
            try: os.remove(html_path)
            except: pass


    def _setup_expiries(self):
        """Prompt user for expiries once, with auto-defaults for Thursdays. Returns True if valid."""
        print("\n" + "═" * 60)
        print("  EXPIRY CONFIGURATION")
        print("═" * 60)
        
        today = datetime.now().date()
        days_to_thu = 3 - today.weekday()
        if days_to_thu < 0: days_to_thu += 7
        
        default_near = (today + pd.Timedelta(days=days_to_thu)).strftime("%Y-%m-%d")
        default_far = (today + pd.Timedelta(days=days_to_thu + 28)).strftime("%Y-%m-%d")

        print(f"  Format: YYYY-MM-DD (Press Enter to use defaults)")
        print()

        near = input(f"  Near Expiry (weekly)  [{default_near}]: ").strip() or default_near
        far  = input(f"  Far Expiry  (monthly) [{default_far}]: ").strip() or default_far
        extras_raw = input("  Extra Expiries (optional, comma-sep): ").strip()
        extras = [x.strip() for x in extras_raw.split(',') if x.strip()] if extras_raw else []

        # Validate dates are future
        for exp_str in [near, far] + extras:
            try:
                exp_dt = datetime.strptime(exp_str, "%Y-%m-%d")
                if exp_dt.date() < datetime.now().date():
                    print(f"  Error: {exp_str} is in the past.")
                    return False
            except ValueError:
                print(f"  Error: Invalid date format '{exp_str}'. Use YYYY-MM-DD.")
                return False

        self._near_expiry = near
        self._far_expiry = far
        # Build full list: near, far, plus any extras (deduplicated, ordered)
        seen = set()
        self._expiries = []
        for e in [near, far] + extras:
            if e not in seen:
                self._expiries.append(e)
                seen.add(e)

        print(f"\n  ✓ Expiries locked: {', '.join(self._expiries)}")
        return True

    def run(self):
        print("\n" + "╔" + "═" * 58 + "╗")
        print("║" + "  UNIFIED VOLATILITY SURFACE ANALYZER".ljust(58) + "║")
        print("╚" + "═" * 58 + "╝")
        print(f"  Symbol: {self.symbol}")

        # Fetch spot once at startup
        print("  Fetching Spot Price...")
        self.get_spot_price()
        if self.spot_price > 0:
            print(f"  Spot: {self.spot_price:,.2f}")
        else:
            print("  Warning: Could not fetch spot price.")

        # Setup expiries
        while not self._setup_expiries():
            print("  Please try again.")

        # Launch unified dashboard (all analysis in one browser page)
        self._create_unified_dashboard(self._expiries)




if __name__ == "__main__":
    app = VolatilityAnalyzer()
    app.run()