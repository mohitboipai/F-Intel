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
    sys.stdout.reconfigure(encoding='utf-8') # type: ignore

from FyersAuth import FyersAuthenticator
from OptionAnalytics import OptionAnalytics
from SignalMemory import SignalMemory
from SharedDataCache import SharedDataCache
from NiftyHestonMC import HestonMath
from DataClient import DataHubClient
from RealizedVolEngine import RealizedVolEngine
from MasterSignalEngine import MasterSignalEngine
from OptionBuyerEngine import OptionBuyerEngine

try:
    import config as _cfg
except ImportError:
    _cfg = None

def _get_cfg(key, default):
    if _cfg is not None:
        return _cfg.get(key, default)
    return default

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
            live_p = float(chain_prices.get((leg['type'], leg['strike']), leg.get('price', 0)) or 0)
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
        
        from typing import Any
        self.fyers: Any = self._authenticate()
        self.analytics = OptionAnalytics()
        self.symbol = "NSE:NIFTY50-INDEX"
        self.spot_price: float = 0.0
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
        self.master_engine = MasterSignalEngine()
        self.buyer_engine = OptionBuyerEngine()

    def _authenticate(self):
        from fyers_auth_manager import get_fyers_instance
        return get_fyers_instance()

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
                response = self.fyers.quotes(data=data)  # type: ignore
            
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
        """
        4-tier IV resolution (Dumas-Fleming-Whaley 1998; Gatheral SVI 2004):
          Tier 1: Valid observable API IV (0.5% < iv < 200%).
          Tier 2: Newton-Raphson/Bisection numerical solve from market price.
          Tier 3: Moneyness quadratic smile approximation: IV(k) = a + b*k + c*k^2.
          Tier 4: Flat regime fallback: iv_fallback_flat from config.
        """
        # Tier 1: Valid API IV
        try:
            if row_iv is not None and not np.isnan(row_iv) and 0.5 < float(row_iv) < 200.0:
                return float(row_iv)
        except (ValueError, TypeError):
            pass

        r = _get_cfg("risk_free_rate", 0.051274)
        q = _get_cfg("dividend_yield", 0.0122) if (T * 365.0) > _get_cfg("dividend_dte_threshold", 7) else 0.0

        # Tier 2: Solve from option price via Analytics
        if price is not None and price > 0.5 and strike > 0 and T > 0:
            try:
                calc_iv = self.analytics.implied_volatility(
                    price, self.spot_price, strike, T, r, option_type, q=q
                )
                if 0.5 < calc_iv < 200.0:
                    return float(calc_iv)
            except Exception:
                pass

        # Tier 3: Quadratic smile fallback based on log-moneyness
        if self.spot_price > 0 and strike > 0:
            try:
                k = np.log(strike / self.spot_price)
                atm_base = _get_cfg("iv_fallback_flat", 0.15) * 100.0
                smile_iv = atm_base - 15.0 * k + 25.0 * (k ** 2)
                min_iv = _get_cfg("iv_fallback_min", 0.08) * 100.0
                max_iv = _get_cfg("iv_fallback_max", 0.80) * 100.0
                if min_iv <= smile_iv <= max_iv:
                    return float(smile_iv)
            except Exception:
                pass

        # Tier 4: Flat regime fallback from config
        fallback = _get_cfg("iv_fallback_flat", 0.15) * 100.0
        return float(fallback)

    def _filter_strikes(self, df, spot, range_pct=None):
        """Filter to strikes within range_pct of spot (defaults to config strike_filter_range)."""
        if range_pct is None:
            range_pct = _get_cfg("strike_filter_range", 0.05)
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
                        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                        font=dict(color=WHITE, family='Inter, Segoe UI, sans-serif', size=11),
                        legend=dict(bgcolor='rgba(30,30,50,0.8)', font=dict(size=10, color=WHITE),
                                   x=0.01, y=0.99),
                        margin=dict(l=50, r=20, t=50, b=30),
                        hovermode='closest',
                        scene=dict(
                            xaxis=dict(title='Strike', backgroundcolor='rgba(0,0,0,0)', gridcolor='#333', color=MUTED),
                            yaxis=dict(title='Days', backgroundcolor='rgba(0,0,0,0)', gridcolor='#333', color=MUTED),
                            zaxis=dict(title='IV%', backgroundcolor='rgba(0,0,0,0)', gridcolor='#333', color=MUTED),
                            camera=dict(eye=dict(x=1.5, y=-1.8, z=0.8)),
                            bgcolor='rgba(0,0,0,0)'
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
                
                if nx is None or ny is None or natm is None:
                    print("Waiting for data...")
                    time.sleep(2)
                    continue
                
                # Plot
                ax.clear()
                
                # Near Curve
                ax.plot(nx, ny, color='blue', linewidth=2, label=f"Near ({near_exp})")
                
                # Far Curve & Convergence
                nav_msg = ""
                if fx is not None and fy is not None and fatm is not None:
                    ax.plot(fx, fy, color='purple', linewidth=2, label=f"Far ({far_exp})")
                    
                    # Convergence Logic
                    # Compare ATM difference
                    spread = fatm - natm
                    ax.fill_between(nx, ny, np.interp(nx, fx, fy), color='gray', alpha=0.1) # type: ignore
                    
                    nav_msg = f" | Term Spread: {spread:.2f}%"
                    if spread < 0: nav_msg += " (BACKWARDATION)"
                    else: nav_msg += " (CONTANGO)"
                
                # Ghost (Near only)
                if previous_curve_near is not None and previous_curve_near[0] is not None:
                     ax.plot(previous_curve_near[0], previous_curve_near[1], color='gray', linestyle='--', alpha=0.5, label='5m Ago') # type: ignore

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
                hv_centered = hv_vals - hv_vals.mean() # type: ignore
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
                    val = float(np.percentile(list(bucket_moves.values), p)) # type: ignore
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
                p10 = float(np.percentile(list(bucket_moves.values), 10)) # type: ignore
                p90 = float(np.percentile(list(bucket_moves.values), 90)) # type: ignore
                print(f"\n  Expected 5-Day Range (80% confidence):")
                print(f"  [{spot * (1 + p10/100):.0f}] ─── [{spot:.0f}] ─── [{spot * (p90/100 + 1):.0f}]")
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
            'composite_score': round(float(signal_result.get('composite', 0)), 1), # type: ignore
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
                _r = _get_cfg("risk_free_rate", 0.051274)
                if ce_price <= 0 and atm_iv > 0:
                    ce_price = self.analytics.black_scholes(spot, ce_strike, T, _r, sigma_dec, 'CE')
                if pe_price <= 0 and atm_iv > 0:
                    pe_price = self.analytics.black_scholes(spot, pe_strike, T, _r, sigma_dec, 'PE')
                
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
                if total_samples > 0:
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
                
                r_rate = _get_cfg("risk_free_rate", 0.051274)
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
                            'sell_ce_above':round(float(safe_ce), 1), # type: ignore
                            'sell_pe_below':round(float(safe_pe), 1), # type: ignore
                            'em':           round(float(em_straddle), 1), # type: ignore
                            'atm_iv':       round(float(atm_iv if 'atm_iv' in dir() else 0), 2),
                            'call_wall':    float(call_wall),
                            'put_wall':     float(put_wall),
                            'max_pain':     float(max_pain if 'max_pain' in dir() else 0),
                            'oi_pressure':  oi_pressure,
                            'oi_pressure_score': round(oi_pressure_score, 1),
                        }
                        if self._seller_signal_id is None:
                            self._seller_signal_id = self.memory.log_signal(
                                'OptionSellerAdvisor', seller_payload,
                                spot=spot, expiry=exp or "")
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
            def log_message(self, *args): pass # type: ignore
            def log_request(self, *args): pass # type: ignore

            def copyfile(self, source, outputfile):
                try:
                    super().copyfile(source, outputfile)
                except (ConnectionAbortedError, ConnectionResetError, BrokenPipeError):
                    pass
            
            def do_GET(self):
                if self.path.startswith('/fragment'):
                    self.path = '/unified_dashboard_fragment.html'
                    return super().do_GET()
                elif self.path.startswith('/api/'):
                    try:
                        import urllib.request
                        url = f"http://127.0.0.1:8082{self.path}"
                        req = urllib.request.Request(url)
                        with urllib.request.urlopen(req, timeout=3) as resp:
                            data = resp.read()
                            self.send_response(resp.status)
                            self.send_header('Content-type', 'application/json')
                            self.send_header('Access-Control-Allow-Origin', '*')
                            self.end_headers()
                            self.wfile.write(data)
                            return
                    except Exception as e:
                        self.send_response(502)
                        self.end_headers()
                        self.wfile.write(str(e).encode())
                        return
                return super().do_GET()

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
        heston_cache: dict = {'params': None, 'ts': 0, 'ttl': 300}  # 5-min TTL
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
        _iv_history = []  # [(timestamp_str, atm_iv), ...] — rolling ATM IV cache for historical chart


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
            """Compute vol cone / regime / VRP data across multiple timeframes. Returns dict."""
            nonlocal _last_regime, _regime_changed_at, _iv_history
            hist = _fetch_history_once()
            if not hist or len(hist['closes']) < 40:
                return None
            closes = hist['closes']
            highs  = hist['highs']
            lows   = hist['lows']
            opens  = hist['opens']

            # ── Multi-timeframe Close-to-Close HV ──
            hv_series = {}
            for _w in [5, 10, 20, 60]:
                _s = self.analytics.calculate_rolling_historical_volatility(closes, window=_w)
                hv_series[_w] = _s

            rolling_hv = hv_series[20]  # primary 20d series
            if rolling_hv.empty or len(rolling_hv) < 40:
                return None

            # ── Parkinson (H/L) at multiple windows ──
            def _parkinson_hv(highs, lows, window):
                h = pd.Series(highs); l = pd.Series(lows)
                log_hl = (np.log(h / l) ** 2)
                pk = np.sqrt(log_hl.rolling(window).mean() / (4 * np.log(2))) * np.sqrt(252) * 100
                return pk

            # ── Garman-Klass (OHLC) at multiple windows ──
            def _garman_klass_hv(opens, highs, lows, closes, window):
                o = pd.Series(opens); h = pd.Series(highs); l = pd.Series(lows); c = pd.Series(closes)
                gk = (0.5 * (np.log(h/l)**2) - (2*np.log(2)-1) * (np.log(c/o)**2))
                return np.sqrt(gk.rolling(window).mean() * 252) * 100

            # ── Yang-Zhang (Gap + OHLC) at multiple windows ──
            def _yang_zhang_hv(opens, highs, lows, closes, window):
                o = pd.Series(opens); h = pd.Series(highs); l = pd.Series(lows); c = pd.Series(closes)
                k = 0.34 / (1.34 + (window + 1) / (window - 1))
                cc  = np.log(c / c.shift(1))
                oo  = np.log(o / o.shift(1))
                co  = np.log(c / o)
                oc  = np.log(o / c.shift(1))
                yz  = oo.rolling(window).var() + k * co.rolling(window).var() + (1 - k) * cc.rolling(window).var()
                return np.sqrt(yz * 252) * 100

            pk_20 = _parkinson_hv(highs, lows, 20)
            pk_60 = _parkinson_hv(highs, lows, 60)
            gk_20 = _garman_klass_hv(opens, highs, lows, closes, 20)
            gk_60 = _garman_klass_hv(opens, highs, lows, closes, 60)
            yz_20 = _yang_zhang_hv(opens, highs, lows, closes, 20)
            yz_60 = _yang_zhang_hv(opens, highs, lows, closes, 60)

            def _last(s): return float(s.iloc[-1]) if not s.empty and not pd.isna(s.iloc[-1]) else 0.0

            hv_vals = {
                'c2c_5':  _last(hv_series[5]),
                'c2c_10': _last(hv_series[10]),
                'c2c_20': _last(hv_series[20]),
                'c2c_60': _last(hv_series[60]),
                'pk_20':  _last(pk_20),
                'pk_60':  _last(pk_60),
                'gk_20':  _last(gk_20),
                'gk_60':  _last(gk_60),
                'yz_20':  _last(yz_20),
                'yz_60':  _last(yz_60),
            }
            # Parkinson at 5d / 10d
            hv_vals['pk_5']  = _last(_parkinson_hv(highs, lows, 5))
            hv_vals['pk_10'] = _last(_parkinson_hv(highs, lows, 10))

            current_hv  = hv_vals['c2c_20']
            current_park = hv_vals['pk_20']
            mean_hv = float(rolling_hv.mean())
            min_hv  = float(rolling_hv.min())
            max_hv  = float(rolling_hv.max())
            hv_percentile = float((rolling_hv < current_hv).mean() * 100)

            # Short-term vs medium-term regime comparison
            st_regime = 'RISING' if hv_vals['c2c_5'] > hv_vals['c2c_20'] else 'FALLING'
            mt_regime = 'ABOVE_MEAN' if hv_vals['c2c_20'] > mean_hv else 'BELOW_MEAN'

            # Bollinger Bands on 20d HV
            bb_window = 20
            hv_sma  = rolling_hv.rolling(bb_window).mean()
            hv_std  = rolling_hv.rolling(bb_window).std()
            bb_upper = hv_sma + 2 * hv_std
            bb_lower = hv_sma - 2 * hv_std
            bandwidth = ((bb_upper - bb_lower) / hv_sma * 100).dropna()
            current_bw   = float(bandwidth.iloc[-1]) if not bandwidth.empty else 0
            bw_percentile = float((bandwidth < current_bw).mean() * 100) if not bandwidth.empty else 50

            hv_slope = 0.0
            if len(rolling_hv) >= 5:
                recent_5 = rolling_hv.iloc[-5:].values
                hv_slope = float(recent_5[-1] - recent_5[0])

            iv_velocity_5d  = float(rolling_hv.diff(5).iloc[-1])  if len(rolling_hv) >= 6  else 0.0
            iv_velocity_10d = float(rolling_hv.diff(10).iloc[-1]) if len(rolling_hv) >= 11 else 0.0
            iv_accel = iv_velocity_5d - (iv_velocity_10d / 2)

            hv_changes = rolling_hv.diff().dropna()
            vov = float(hv_changes.tail(20).std()) if len(hv_changes) >= 20 else 0.0

            ml_prob = 0.50  # kept for backward compat

            # ── Intraday momentum override ──
            m_stat = momentum_data.get('status', 'NEUTRAL') if momentum_data else 'NEUTRAL'

            if m_stat in ['LONG', 'SHORT'] and (iv_accel > 0 or hv_slope > 0):
                regime = "EXPANSION (LIVE)"
                regime_desc = f"Vol exploding intraday! Spot > VWAP/EMA. (BBw: {bw_percentile:.0f}th)"
            elif m_stat != 'NEUTRAL' and bw_percentile < 30:
                regime = "MOMENTUM / TRENDING"
                regime_desc = "Trending directional breakout in progress."
            elif bw_percentile < 25:
                regime = "COMPRESSION"
                regime_desc = f"Vol tightly coiled (BBw: {bw_percentile:.0f}th) — Breakout likely"
            elif bw_percentile > 75 and hv_slope > 0:
                regime = "EXPANSION"
                regime_desc = "Vol exploding (HV rising, BBw expanding)"
            elif hv_slope < -0.5:
                regime = "MEAN_REVERSION"
                regime_desc = "Vol exhaustion (HV slope dropping)"
            else:
                regime = "NORMAL"
                regime_desc = "Standard vol flow — No extreme edge"

            # ── Regime transition log ──
            _now_ts = datetime.now().strftime('%H:%M:%S')
            if _last_regime is not None and regime != _last_regime:
                _regime_changed_at = _now_ts
            regime_history.append((_now_ts, regime, ml_prob))
            if len(regime_history) > 500:
                regime_history.pop(0)
            _prev_regime = _last_regime or regime
            _last_regime = regime

            # ── Regime transition log HTML (last 8 changes) ──
            _transition_log = []
            _last_seen = None
            for _ts, _rg, _ in regime_history:
                if _rg != _last_seen:
                    _transition_log.append((_ts, _rg))
                    _last_seen = _rg
            _recent_transitions = _transition_log[-8:]
            _trans_rows_html = ''
            for _ts, _rg in reversed(_recent_transitions):
                _rc = YELLOW if 'COMPRESS' in _rg else RED if 'EXPAND' in _rg else '#64b5f6' if 'MOMENT' in _rg else MUTED
                _trans_rows_html += (f'<tr><td style="padding:3px 8px;color:#888;font-size:10px;">{_ts}</td>'
                                     f'<td style="padding:3px 8px;font-weight:700;font-size:11px;color:{_rc};">{_rg}</td></tr>')
            if not _trans_rows_html:
                _trans_rows_html = '<tr><td colspan="2" style="padding:4px 8px;color:#555;">Building history...</td></tr>'

            # ── ATM IV from option chain (with caching) ──
            iv = 0.0
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

            # Cache IV history
            if iv > 0:
                _iv_history.append((_now_ts, iv))
                if len(_iv_history) > 252:
                    _iv_history.pop(0)

            vrp = iv - current_hv if iv > 0 else 0.0
            z_score = self.analytics.calculate_z_score(iv, rolling_hv) if iv > 0 else 0.0

            # ── VRP percentile vs 1-year ──
            vrp_series = pd.Series([float(i) - float(rolling_hv.iloc[max(0, j-1)])
                                    for j, i in enumerate(rolling_hv.values)
                                    if j > 0], dtype=float).dropna()
            vrp_percentile = (vrp_series < vrp).mean() * 100 if len(vrp_series) > 10 else 50.0

            signal_metrics = {'vrp': vrp, 'regime': regime}
            signal_result = self._score_signal(signal_metrics)

            # ── Half-life (mean-reversion speed) ──
            half_life = 0.0
            try:
                hv_vals_arr = rolling_hv.dropna().values
                if len(hv_vals_arr) > 20:
                    hv_centered = hv_vals_arr - np.mean(hv_vals_arr) # type: ignore
                    ac = np.correlate(hv_centered[:-1], hv_centered[1:], mode='valid')[0]
                    ac /= np.correlate(hv_centered[:-1], hv_centered[:-1], mode='valid')[0]
                    if 0 < ac < 1:
                        half_life = np.log(2) / (-np.log(ac))
            except:
                pass

            # ── Regime forecast rows ──
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
                'regime': regime, 'regime_desc': regime_desc, 'ml_prob': ml_prob,
                'bandwidth': current_bw, 'bw_percentile': bw_percentile, 'hv_slope': hv_slope,
                'st_regime': st_regime, 'mt_regime': mt_regime,
                'hv_vals': hv_vals,
                'prev_regime': _prev_regime, 'regime_changed_at': _regime_changed_at,
                'trans_rows': _trans_rows_html,
                'forecast_rows': forecast_rows_html,
                'iv_velocity_5d': iv_velocity_5d, 'iv_accel': iv_accel, 'vov': vov,
                'iv': iv, 'vrp': vrp, 'z_score': z_score, 'half_life': half_life,
                'vrp_percentile': vrp_percentile,
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
            _r = _get_cfg("risk_free_rate", 0.051274)
            if ce_price <= 0 and atm_iv > 0:
                ce_price = self.analytics.black_scholes(spot, atm_ce['strike'] if atm_ce is not None else atm_strike, T, _r, sigma, 'CE')
            if pe_price <= 0 and atm_iv > 0:
                pe_price = self.analytics.black_scholes(spot, atm_pe['strike'] if atm_pe is not None else atm_strike, T, _r, sigma, 'PE')

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
            r_rate = _get_cfg("risk_free_rate", 0.051274)
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
                r = _get_cfg("risk_free_rate", 0.051274)
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
            r = _get_cfg("risk_free_rate", 0.051274)
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
                T = float(h['days']) / 365.0
                if T <= 0:
                    continue
                steps = max(10, h['days'] * 2)
                dt = T / float(steps)
                # Vectorized Heston MC
                Z1 = np.random.normal(size=(N_PATHS, int(steps)))
                Z3 = np.random.normal(size=(N_PATHS, int(steps)))
                Z2 = rho * Z1 + np.sqrt(1 - rho**2) * Z3
                S = np.full(N_PATHS, spot)
                v = np.full(N_PATHS, v0)
                for t in range(int(steps)):
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
                skewness = float(pd.Series(list(terminal)).skew()) # type: ignore
                kurtosis_val = float(pd.Series(list(terminal)).kurtosis()) # type: ignore
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
            r = _get_cfg("risk_free_rate", 0.051274)
            near_T = self.analytics.get_time_to_expiry(near_exp)
            near_days = max(1, int(near_T * 365))
            horizons = [
                {'label': f'BSM DTE {near_days}', 'days': near_days, 'color': ACCENT},
                {'label': 'BSM Day 7',  'days': 7,  'color': '#ff7043'},
                {'label': 'BSM Day 14', 'days': 14, 'color': '#66bb6a'},
            ]
            results = []
            for h in horizons:
                T = float(h['days']) / 365.0
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
            r = _get_cfg("risk_free_rate", 0.051274)
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

                    # ── FETCH GREEK FLOWS (GEX / DEALER) ──
                    import urllib.request, json
                    gex_data, dealer_data = None, None
                    try:
                        gex_req = urllib.request.urlopen("http://127.0.0.1:8082/api/gex", timeout=1.0)
                        gex_data = json.loads(gex_req.read().decode())
                        dealer_req = urllib.request.urlopen("http://127.0.0.1:8082/api/dealer", timeout=1.0)
                        dealer_data = json.loads(dealer_req.read().decode())
                    except Exception as _e:
                        pass
                        
                    # ── UPDATE SIGNAL MEMORY WITH CONTEXT ──
                    if self.memory:
                        ctx = {}
                        if regime_snapshot:
                            ctx['regime'] = regime_snapshot.get('regime', {}).get('name', 'UNKNOWN')
                        if gex_data:
                            ctx['net_gex'] = gex_data.get('net_gex', 0)
                            ctx['net_vanna'] = gex_data.get('net_vanna', 0)
                            ctx['net_charm'] = gex_data.get('net_charm', 0)
                        if 'vol_intel' in locals() and vol_intel:
                            ctx['vrp'] = vol_intel.get('vrp', 0) if 'vrp' in vol_intel else (vol_intel.get('iv', 0) - vol_intel.get('current_hv', 0))
                        if 'momentum_data' in locals() and momentum_data:
                            ctx['momentum_status'] = momentum_data.get('status', 'NEUTRAL')
                            
                        self.memory.update_context(ctx)

                    # ── MASTER SIGNAL VERDICT ──
                    verdict_data = self.master_engine.evaluate(self.memory)

                    # ── BUYER SETUP ──
                    gex_accel = 0.0 # Will compute if needed, or default
                    buyer_setup = None
                    if not df_chain.empty and regime_snapshot:
                        buyer_setup = self.buyer_engine.generate_trade_setup(
                            confluence_verdict=verdict_data,
                            spot=spot,
                            vwap=float(momentum_vwap) if momentum_vwap else spot,
                            df_chain=df_chain,
                            gex_acceleration=gex_accel,
                            intraday_regime=regime_snapshot.get('regime', {}).get('name', '')
                        )
                        
                        if buyer_setup:
                            from signal_broadcaster import SignalBroadcaster
                            SignalBroadcaster.broadcast_trade(buyer_setup)

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
                        if float(pred.get('expected_move', 0)) > 0: # type: ignore
                            fig.add_vrect(x0=spot - float(pred.get('expected_move', 0)), x1=spot + float(pred.get('expected_move', 0)), # type: ignore
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

                        fig.update_layout(height=480, width=1380, paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)',
                            font=dict(color=WHITE, family='Inter, sans-serif', size=11),
                            legend=dict(bgcolor='rgba(30,30,50,0.8)', font=dict(size=10), x=0.01, y=0.99),
                            margin=dict(l=50, r=20, t=50, b=30), hovermode='closest',
                            scene=dict(
                                xaxis=dict(title='Strike', backgroundcolor='rgba(0,0,0,0)', gridcolor='#333'),
                                yaxis=dict(title='Days', backgroundcolor='rgba(0,0,0,0)', gridcolor='#333'),
                                zaxis=dict(title='IV%', backgroundcolor='rgba(0,0,0,0)', gridcolor='#333'),
                                camera=dict(eye=dict(x=1.5, y=-1.8, z=0.8)), bgcolor='rgba(0,0,0,0)'))
                        fig.update_xaxes(gridcolor='rgba(100,100,100,0.15)', title='Strike', row=1, col=1)
                        fig.update_yaxes(gridcolor='rgba(100,100,100,0.15)', title='IV (%)', row=1, col=1)
                        iv_plotly = fig.to_html(include_plotlyjs=False, full_html=False)

                        # Prediction panel
                        dir_color = GREEN if 'BULL' in str(pred.get('direction', '')) else RED if 'BEAR' in str(pred.get('direction', '')) else YELLOW
                        _ts = pred.get('term_spread', 0)
                        ts_val = float(_ts[0]) if isinstance(_ts, list) else float(_ts) # type: ignore
                        ts_color = RED if ts_val < -1 else GREEN if ts_val > 1 else WHITE
                        iv_tab_html = f'''
                        {iv_plotly}
                        <div class="card" style="margin-top:10px;">
                            <div style="color:{ACCENT};font-size:13px;font-weight:700;letter-spacing:2px;margin-bottom:14px;">SPOT MOVEMENT PREDICTOR</div>
                            <div style="display:flex;gap:10px;flex-wrap:wrap;">
                                <div class="metric-box"><div class="metric-label">DIRECTION</div><div style="font-size:22px;font-weight:700;color:{dir_color};">{pred['direction']}</div><div class="metric-sub">Confidence: {pred['confidence']:.0%}</div></div>
                                <div class="metric-box"><div class="metric-label">EXPECTED 1-DAY MOVE</div><div style="font-size:22px;font-weight:700;color:{WHITE};">±{pred['expected_move']:.0f} pts</div><div class="metric-sub">±{pred['expected_move_pct']:.2f}%</div></div>
                                <div class="metric-box"><div class="metric-label">SKEW RATIO</div><div style="font-size:22px;font-weight:700;color:{WHITE};">{pred['skew_ratio']:.3f}</div><div class="metric-sub">{pred['skew_signal']}</div></div>
                                <div class="metric-box"><div class="metric-label">TERM SPREAD</div><div style="font-size:22px;font-weight:700;color:{ts_color};">{pred['term_spread']:+.2f}%</div><div class="metric-sub">{pred['term_signal']}</div></div>
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

                        # ── VRP percentile for display ──
                        _vrp_pct = v.get('vrp_percentile', 50)
                        _vrp_pct_color = GREEN if _vrp_pct > 65 else RED if _vrp_pct < 35 else YELLOW
                        _hv = v.get('hv_vals', {})

                        # ── Short/medium-term regime colors ──
                        _st_rg = v.get('st_regime', 'STABLE')
                        _mt_rg = v.get('mt_regime', 'BELOW_MEAN')
                        _st_color = RED if _st_rg == 'RISING' else GREEN
                        _mt_color = RED if _mt_rg == 'ABOVE_MEAN' else GREEN

                        vol_tab_html = f'''
                        <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;">
                            <div class="card">
                                <div style="color:{ACCENT};font-size:13px;font-weight:700;letter-spacing:2px;margin-bottom:14px;">HV STATISTICS (1-Year)</div>
                                <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:10px;">
                                    <div class="metric-box"><div class="metric-label">20d HV (C2C)</div><div style="font-size:20px;font-weight:700;color:{WHITE};">{v['current_hv']:.2f}%</div><div class="metric-sub">Parkinson: {v['parkinson']:.2f}%</div></div>
                                    <div class="metric-box"><div class="metric-label">1-Yr Mean</div><div style="font-size:20px;font-weight:700;color:{WHITE};">{v['mean_hv']:.2f}%</div><div class="metric-sub">{v['min_hv']:.1f}% — {v['max_hv']:.1f}%</div></div>
                                    <div class="metric-box"><div class="metric-label">HV Percentile</div><div style="font-size:20px;font-weight:700;color:{YELLOW};">{v['hv_percentile']:.0f}%</div></div>
                                </div>
                                <div style="display:grid;grid-template-columns:1fr 1fr 1fr 1fr;gap:10px;margin-top:12px;">
                                    <div class="metric-box"><div class="metric-label">ATM IV</div><div style="font-size:18px;font-weight:700;color:{WHITE};">{v['iv']:.2f}%</div></div>
                                    <div class="metric-box"><div class="metric-label">VRP</div><div style="font-size:18px;font-weight:700;color:{GREEN if v['vrp']>2 else RED if v['vrp']<-2 else WHITE};">{v['vrp']:+.2f}%</div></div>
                                    <div class="metric-box"><div class="metric-label">VRP Rank</div><div style="font-size:18px;font-weight:700;color:{_vrp_pct_color};">{_vrp_pct:.0f}%ile</div><div class="metric-sub">vs 1yr window</div></div>
                                    <div class="metric-box"><div class="metric-label">Half-Life</div><div style="font-size:18px;font-weight:700;color:{WHITE};">{v['half_life']:.0f}d</div></div>
                                </div>
                                <div style="margin-top:14px;">
                                    <div style="color:{ACCENT};font-size:11px;font-weight:700;letter-spacing:1.5px;margin-bottom:8px;">REALIZED VOL CONE — ALL ESTIMATORS</div>
                                    <table class="data-table" style="font-size:11px;">
                                        <thead><tr><th style="text-align:left;">Estimator</th><th>5d</th><th>10d</th><th>20d</th><th>60d</th></tr></thead>
                                        <tbody>
                                            <tr><td style="text-align:left;color:{WHITE};">Close-to-Close</td><td>{_hv.get('c2c_5',0):.2f}%</td><td>{_hv.get('c2c_10',0):.2f}%</td><td>{_hv.get('c2c_20',0):.2f}%</td><td>{_hv.get('c2c_60',0):.2f}%</td></tr>
                                            <tr><td style="text-align:left;color:{WHITE};">Parkinson (H/L)</td><td>{_hv.get('pk_5',0):.2f}%</td><td>{_hv.get('pk_10',0):.2f}%</td><td>{_hv.get('pk_20',0):.2f}%</td><td>{_hv.get('pk_60',0):.2f}%</td></tr>
                                            <tr><td style="text-align:left;color:{WHITE};">Garman-Klass</td><td>-</td><td>-</td><td>{_hv.get('gk_20',0):.2f}%</td><td>{_hv.get('gk_60',0):.2f}%</td></tr>
                                            <tr><td style="text-align:left;color:{WHITE};">Yang-Zhang</td><td>-</td><td>-</td><td>{_hv.get('yz_20',0):.2f}%</td><td>{_hv.get('yz_60',0):.2f}%</td></tr>
                                            <tr style="background:rgba(79,195,247,0.08);"><td style="text-align:left;font-weight:700;color:{ACCENT};">ATM IV</td><td colspan="4" style="text-align:center;font-weight:700;color:{ACCENT};">{v['iv']:.2f}% (VRP: {v['vrp']:+.2f}% | {_vrp_pct:.0f}th pct)</td></tr>
                                        </tbody>
                                    </table>
                                </div>
                            </div>
                            <div class="card">
                                <div style="color:{ACCENT};font-size:13px;font-weight:700;letter-spacing:2px;margin-bottom:12px;">MULTI-TIMEFRAME REGIME MATRIX</div>
                                <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:10px;margin-bottom:12px;">
                                    <div class="metric-box" style="border-top:3px solid {GREEN if momentum_data['status']=='LONG' else RED if momentum_data['status']=='SHORT' else YELLOW};">
                                        <div class="metric-label">INTRADAY</div>
                                        <div style="font-size:16px;font-weight:800;color:{GREEN if momentum_data['status']=='LONG' else RED if momentum_data['status']=='SHORT' else YELLOW};">{momentum_data['status']}</div>
                                        <div class="metric-sub">VWAP/EMA momentum</div>
                                    </div>
                                    <div class="metric-box" style="border-top:3px solid {_st_color};">
                                        <div class="metric-label">SHORT-TERM (5d vs 20d)</div>
                                        <div style="font-size:16px;font-weight:800;color:{_st_color};">{_st_rg}</div>
                                        <div class="metric-sub">5d: {_hv.get('c2c_5',0):.2f}% vs 20d: {_hv.get('c2c_20',0):.2f}%</div>
                                    </div>
                                    <div class="metric-box" style="border-top:3px solid {_mt_color};">
                                        <div class="metric-label">MEDIUM-TERM (vs mean)</div>
                                        <div style="font-size:16px;font-weight:800;color:{_mt_color};">{_mt_rg}</div>
                                        <div class="metric-sub">20d: {_hv.get('c2c_20',0):.2f}% | Mean: {v['mean_hv']:.2f}%</div>
                                    </div>
                                </div>
                                <div style="margin-bottom:10px;">
                                    <div class="metric-box" style="text-align:left;padding:10px 14px;">
                                        <div class="metric-label">CURRENT REGIME</div>
                                        <div style="font-size:18px;font-weight:800;color:{regime_color};margin:4px 0;">{v['regime']}</div>
                                        <div style="font-size:11px;color:#aaa;">{v['regime_desc']}</div>
                                        <div style="font-size:10px;color:{MUTED};margin-top:4px;">BBw: {v['bw_percentile']:.0f}th pct | HV Slope: {v['hv_slope']:+.2f}% | Changed: {v.get('regime_changed_at','—')}</div>
                                    </div>
                                </div>
                                <div style="color:{ACCENT};font-size:11px;font-weight:700;letter-spacing:1.5px;margin-bottom:6px;">REGIME TRANSITION LOG</div>
                                <div style="max-height:140px;overflow-y:auto;background:#0a0a18;border-radius:6px;padding:4px;">
                                    <table style="width:100%;border-collapse:collapse;font-size:11px;">
                                        {v.get('trans_rows', '<tr><td colspan="2" style="padding:4px 8px;color:#555;">Building...</td></tr>')}
                                    </table>
                                </div>
                                <div style="margin-top:10px;display:grid;grid-template-columns:1fr 1fr;gap:8px;">
                                    <div class="metric-box"><div class="metric-label">Z-Score (IV)</div><div style="font-size:16px;font-weight:700;color:{WHITE};">{v['z_score']:.2f}×</div></div>
                                    <div class="metric-box"><div class="metric-label">Samples</div><div style="font-size:16px;font-weight:700;color:{MUTED};">{len(regime_history)}</div></div>
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
                                <div style="flex:1;display:grid;grid-template-columns:1fr 1fr 1fr 1fr;gap:10px;">
                                    <div class="metric-box"><div class="metric-label">IV Velocity (5d)</div><div style="font-size:18px;font-weight:700;color:{RED if _iv_velocity > 0.5 else GREEN if _iv_velocity < -0.5 else WHITE};">{_iv_velocity:+.2f}%</div><div class="metric-sub">IV drift trend</div></div>
                                    <div class="metric-box"><div class="metric-label">IV Z-Score</div><div style="font-size:18px;font-weight:700;color:{RED if _iv_zscore > 2 else GREEN if _iv_zscore < 0 else WHITE};">{_iv_zscore:.2f}×</div><div class="metric-sub">vs 1yr mean</div></div>
                                    <div class="metric-box"><div class="metric-label">VRP Edge</div><div style="font-size:18px;font-weight:700;color:{GREEN if _vrp_now > 2 else RED if _vrp_now < -2 else WHITE};">{_vrp_now:+.1f}%</div><div class="metric-sub">IV – HV</div></div>
                                    <div class="metric-box"><div class="metric-label">VRP Rank</div><div style="font-size:18px;font-weight:700;color:{_vrp_pct_color};">{_vrp_pct:.0f}th pct</div><div class="metric-sub">1-year window</div></div>
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
                                _lot = _get_cfg("nifty_lot_size", 65)  # NIFTY lot size
                                _r_gex = _get_cfg("risk_free_rate", 0.051274)
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
                                        d1 = (np.log(_ratio) + (_r_gex + 0.5 * sigma**2) * _T_gex) / (sigma * np.sqrt(max(_T_gex, 1e-6)))
                                        g = np.exp(-0.5 * d1**2) / (np.sqrt(2 * np.pi) * spot * sigma * np.sqrt(max(_T_gex, 1e-6)))
                                    return g

                                _df_gex['gamma_calc'] = _df_gex.apply(_get_gamma, axis=1)

                                _df_gex['gex'] = _df_gex.apply(
                                    lambda r: r['oi'] * r['gamma_calc'] * _lot * spot * spot / 10000
                                             * (1 if r['type'] == 'CE' else -1), axis=1)

                                _gex_by_strike = _df_gex.groupby('strike')['gex'].sum().reset_index()
                                _gex_by_strike = _gex_by_strike[abs(_gex_by_strike['gex']) > 1e4]
                                _gex_by_strike = pd.DataFrame(_gex_by_strike).sort_values(by='strike', ascending=True)

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
                                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
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
                                _dist_call = abs(spot - float(_max_call_strike)) if _max_call_strike else 0
                                _dist_put = abs(spot - float(_max_put_strike)) if _max_put_strike else 0
                                # Find flip strike (where cumulative GEX changes sign)
                                _cum_gex = _gex_by_strike.sort_values('strike')['gex'].cumsum()
                                _flip_strike = 0
                                for _idx in range(1, len(_cum_gex)):
                                    if _cum_gex.iloc[_idx-1] * _cum_gex.iloc[_idx] < 0:
                                        _flip_strike = int(_gex_by_strike.sort_values('strike').iloc[_idx]['strike'])
                                        break
                                _flip_dist_txt = f'{abs(spot - _flip_strike):.0f} pts {"above" if spot > _flip_strike else "below"}' if _flip_strike else 'N/A'
                                _flip_dist_c = GREEN if _flip_strike and spot > _flip_strike else RED if _flip_strike else MUTED
                                gex_analysis_html = f'''
                                <div style="margin-top:12px;padding:10px;background:rgba(15,15,25,0.5);border-radius:6px;">
                                    <div style="color:{ACCENT};font-size:12px;font-weight:700;margin-bottom:8px;">GEX ANALYSIS</div>
                                    <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px;">
                                        <div class="metric-box"><div class="metric-label">Dealer Position</div><div style="font-size:13px;font-weight:700;color:{_gex_regime_color};">{_gex_regime}</div></div>
                                        <div class="metric-box"><div class="metric-label">Call Wall (Resistance)</div><div style="font-size:15px;font-weight:700;color:{WHITE};">{_max_call_strike:.0f}</div><div class="metric-sub">{_dist_call:.0f} pts from spot</div></div>
                                        <div class="metric-box"><div class="metric-label">Put Wall (Support)</div><div style="font-size:15px;font-weight:700;color:{WHITE};">{_max_put_strike:.0f}</div><div class="metric-sub">{_dist_put:.0f} pts from spot</div></div>
                                    </div>
                                    <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px;margin-top:8px;">
                                        <div class="metric-box"><div class="metric-label">Net GEX</div><div style="font-size:15px;font-weight:700;color:{GREEN if _net_gex > 0 else RED};">{_net_gex/1e6:+.2f}M</div><div class="metric-sub">{_pin_txt}</div></div>
                                        <div class="metric-box"><div class="metric-label">GEX Flip Strike</div><div style="font-size:15px;font-weight:700;color:{YELLOW};">{_flip_strike if _flip_strike else 'N/A'}</div><div class="metric-sub">{'Above spot' if _flip_strike > spot else 'Below spot' if _flip_strike else ''}</div></div>
                                        <div class="metric-box"><div class="metric-label">Spot vs Flip</div><div style="font-size:15px;font-weight:700;color:{_flip_dist_c};">{_flip_dist_txt}</div><div class="metric-sub">dealer flip boundary</div></div>
                                    </div>
                                </div>'''

                                # ── Per-strike GEX Trade Management Table ──
                                _gex_mgmt_rows = ''
                                _sorted_gex = _gex_by_strike.sort_values('strike', ascending=False)
                                _gex_abs_max = _sorted_gex['gex'].abs().max() if not _sorted_gex.empty else 1
                                for _, _gr in _sorted_gex.iterrows():
                                    _sk = int(_gr['strike'])
                                    _gx = _gr['gex']
                                    _bar_w = int(abs(_gx) / _gex_abs_max * 100)
                                    _is_atm = abs(_sk - spot) < 100
                                    _is_call_w = (_sk == int(_max_call_strike))
                                    _is_put_w  = (_sk == int(_max_put_strike))
                                    _is_flip   = (_sk == _flip_strike)
                                    # Role
                                    if _is_flip:
                                        _role = 'GEX FLIP'
                                        _role_c = YELLOW
                                    elif _is_call_w:
                                        _role = 'CALL WALL'
                                        _role_c = RED
                                    elif _is_put_w:
                                        _role = 'PUT WALL'
                                        _role_c = GREEN
                                    elif _is_atm:
                                        _role = 'ATM ANCHOR'
                                        _role_c = ACCENT
                                    elif _gx > 0:
                                        _role = 'DEALER LONG Γ'
                                        _role_c = GREEN
                                    else:
                                        _role = 'DEALER SHORT Γ'
                                        _role_c = RED
                                    # Signal
                                    if _is_call_w:
                                        _sig = 'SELL CE above'
                                        _stop = f'Break > {_sk+50} → exit'
                                    elif _is_put_w:
                                        _sig = 'SELL PE below'
                                        _stop = f'Break < {_sk-50} → exit'
                                    elif _is_flip:
                                        _sig = 'Direction pivot'
                                        _stop = 'Spot crosses = bias shift'
                                    elif _gx < -2e6:
                                        _sig = 'Resistance zone'
                                        _stop = 'Close above = trend'
                                    elif _gx > 2e6:
                                        _sig = 'Support / bounce'
                                        _stop = 'Close below = weak'
                                    else:
                                        _sig = 'Watch'
                                        _stop = '—'
                                    _sk_color = _role_c
                                    _gex_str = f'{_gx/1e6:+.2f}M' if abs(_gx) >= 1e6 else f'{_gx/1e3:+.0f}K'
                                    _arrow = '↑' if _gx > 0 else '↓'
                                    _gex_mgmt_rows += (
                                        f'<tr style="background:{"rgba(79,195,247,0.07)" if _is_atm else "rgba(255,68,68,0.05)" if _is_call_w else "rgba(102,187,106,0.05)" if _is_put_w else "rgba(255,200,0,0.04)" if _is_flip else "transparent"};border-bottom:1px solid #1a1a2e;">'
                                        f'<td style="padding:5px 8px;font-weight:700;color:{_sk_color};">{_sk}{" ◄" if _is_atm else ""}</td>'
                                        f'<td style="padding:5px 8px;font-size:10px;color:{"#ff4444" if _gx < 0 else "#66bb6a"};">{_arrow} {_gex_str}</td>'
                                        f'<td style="padding:5px 8px;">'
                                        f'  <span style="font-size:9px;font-weight:700;color:{_role_c};background:{_role_c}22;padding:2px 6px;border-radius:8px;">{_role}</span>'
                                        f'</td>'
                                        f'<td style="padding:5px 8px;font-size:10px;color:#aaa;">{_sig}</td>'
                                        f'<td style="padding:5px 8px;font-size:10px;color:{MUTED};">{_stop}</td>'
                                        f'</tr>'
                                    )

                                gex_mgmt_html = f'''
                                <div style="margin-top:12px;">
                                    <div style="color:{ACCENT};font-size:11px;font-weight:700;letter-spacing:1.5px;margin-bottom:8px;">GEX TRADE MANAGEMENT — PER STRIKE</div>
                                    <div style="max-height:300px;overflow-y:auto;">
                                        <table style="width:100%;border-collapse:collapse;font-size:11px;">
                                            <thead><tr style="color:{MUTED};border-bottom:1px solid #2a2a4a;position:sticky;top:0;background:#0f0f19;">
                                                <th style="padding:5px 8px;text-align:left;">Strike</th>
                                                <th style="padding:5px 8px;text-align:left;">Net GEX</th>
                                                <th style="padding:5px 8px;text-align:left;">Role</th>
                                                <th style="padding:5px 8px;text-align:left;">Signal</th>
                                                <th style="padding:5px 8px;text-align:left;">Stop Zone</th>
                                            </tr></thead>
                                            <tbody>{_gex_mgmt_rows}</tbody>
                                        </table>
                                    </div>
                                </div>'''

                                chain_tab_html += f'<div class="card" style="margin-top:12px;padding:8px 16px;">{gex_html}{gex_analysis_html}{gex_mgmt_html}</div>'
                            except Exception as e:
                                print(f"GEX Error: {e}")
                                pass
                    else:
                        chain_tab_html = '<div class="card"><p style="color:#888;">No option chain data available for analysis.</p></div>'


                    # ── TAB 5: THETA DECAY EXPLORER & GREEKS DYNAMICS ──
                    theta_tab_html = '<div class="card"><p style="color:#888;">Waiting for option chain data to compute Theta Decay...</p></div>'
                    try:
                        if not df_chain.empty and spot > 0:
                            _lot_th = _get_cfg("nifty_lot_size", 65)
                            _r_th = _get_cfg("risk_free_rate", 0.051274)
                            _q_th = _get_cfg("dividend_yield", 0.0122)
                            _T_th = self.analytics.get_time_to_expiry(near_exp) or (7.0 / 365.0)
                            _curr_dte_th = max(_T_th * 365.0, 0.05)
                            _atm_iv_th = float(_pd_iv) if ('_pd_iv' in locals() and _pd_iv and _pd_iv > 0) else 14.0

                            # Filter strikes around spot (±8% band)
                            _lo_th, _hi_th = spot * 0.92, spot * 1.08
                            _df_th = df_chain[df_chain['strike'].between(_lo_th, _hi_th)].copy()

                            if not _df_th.empty:
                                _strikes_sorted = sorted(_df_th['strike'].unique())
                                _atm_strike_th = min(_strikes_sorted, key=lambda s: abs(s - spot))

                                _th_lookup = {}
                                for _, _row in _df_th.iterrows():
                                    _th_lookup[(_row['strike'], _row['type'])] = _row

                                _theta_table_rows = []
                                _ce_thetas_inr = []
                                _pe_thetas_inr = []
                                _straddle_thetas_inr = []
                                _daily_cushions_pts = []
                                _decay_yields_pct = []
                                _valid_strikes = []

                                _peak_theta_val = 0.0
                                _peak_strike = _atm_strike_th
                                _best_harvest_score = -1.0
                                _sweet_strike = _atm_strike_th
                                _sweet_cushion = 0.0
                                _sweet_yield = 0.0
                                _top_buyer_strike = _atm_strike_th
                                _top_buyer_conv = 0.0
                                _top_buyer_score = 0.0
                                _top_seller_strike = _atm_strike_th
                                _top_seller_score = 0.0
                                _atm_strad_prem = 0.0
                                _atm_strad_ext_pts = 0.0
                                _atm_strad_ext_inr = 0.0

                                def _calc_merton_theta(K_val, sig_val, otype):
                                    sqT = np.sqrt(max(_T_th, 1e-6))
                                    d1 = (np.log(spot / K_val) + (_r_th - _q_th + 0.5 * sig_val**2) * _T_th) / (sig_val * sqT)
                                    d2 = d1 - sig_val * sqT
                                    eqT = np.exp(-_q_th * _T_th)
                                    erT = np.exp(-_r_th * _T_th)
                                    pdf_d1 = norm.pdf(d1)
                                    gamma_val = eqT * pdf_d1 / (spot * sig_val * sqT)
                                    vega_val = spot * eqT * pdf_d1 * sqT / 100.0
                                    if otype == 'CE':
                                        delta_val = eqT * norm.cdf(d1)
                                        theta_val = (-spot * eqT * pdf_d1 * sig_val / (2.0 * sqT)
                                                     - _r_th * K_val * erT * norm.cdf(d2)
                                                     + _q_th * spot * eqT * norm.cdf(d1)) / 365.0
                                        charm_val = eqT * (-_q_th * norm.cdf(d1) + pdf_d1 * (2*(_r_th - _q_th)*_T_th - d2*sig_val*sqT)/(2*_T_th*sig_val*sqT)) / 365.0
                                    else:
                                        delta_val = eqT * (norm.cdf(d1) - 1.0)
                                        theta_val = (-spot * eqT * pdf_d1 * sig_val / (2.0 * sqT)
                                                     + _r_th * K_val * erT * norm.cdf(-d2)
                                                     - _q_th * spot * eqT * norm.cdf(-d1)) / 365.0
                                        charm_val = eqT * (_q_th * norm.cdf(-d1) + pdf_d1 * (2*(_r_th - _q_th)*_T_th - d2*sig_val*sqT)/(2*_T_th*sig_val*sqT)) / 365.0
                                    return theta_val, gamma_val, delta_val, charm_val, vega_val

                                for _K in _strikes_sorted:
                                    _ce_row = _th_lookup.get((_K, 'CE'))
                                    _pe_row = _th_lookup.get((_K, 'PE'))

                                    _ce_ltp = float(_ce_row['price']) if _ce_row is not None else 0.0
                                    _pe_ltp = float(_pe_row['price']) if _pe_row is not None else 0.0
                                    _ce_iv = float(_ce_row['iv']) if _ce_row is not None else 0.0
                                    _pe_iv = float(_pe_row['iv']) if _pe_row is not None else 0.0

                                    _ce_calc_iv = self._ensure_iv(_ce_iv, _ce_ltp, _K, _T_th, 'CE') if _ce_ltp > 0 else _atm_iv_th
                                    _pe_calc_iv = self._ensure_iv(_pe_iv, _pe_ltp, _K, _T_th, 'PE') if _pe_ltp > 0 else _atm_iv_th

                                    _sig_ce = max(_ce_calc_iv / 100.0, 0.02)
                                    _sig_pe = max(_pe_calc_iv / 100.0, 0.02)

                                    _th_ce, _gam_ce, _del_ce, _ch_ce, _vg_ce = _calc_merton_theta(_K, _sig_ce, 'CE')
                                    _th_pe, _gam_pe, _del_pe, _ch_pe, _vg_pe = _calc_merton_theta(_K, _sig_pe, 'PE')

                                    _th_ce_inr = _th_ce * _lot_th
                                    _th_pe_inr = _th_pe * _lot_th
                                    _th_strad_inr = _th_ce_inr + _th_pe_inr
                                    _th_strad_1h = _th_strad_inr / 6.25

                                    _strad_gam = _gam_ce + _gam_pe
                                    _del_net = _del_ce + _del_pe
                                    _vg_net = _vg_ce + _vg_pe

                                    _th_day_pts = abs(_th_strad_inr / _lot_th)
                                    _daily_cushion_pts = float(np.sqrt(max(2.0 * _th_day_pts / max(_strad_gam, 1e-7), 0.0)))
                                    _daily_cushion_pts = min(_daily_cushion_pts, 999.0)

                                    _strad_prem = max(_ce_ltp + _pe_ltp, 0.01)
                                    _intrinsic = abs(spot - _K)
                                    _strad_ext_pts = max(_strad_prem - _intrinsic, 0.0)
                                    _strad_ext_inr = _strad_ext_pts * _lot_th
                                    _ext_pct = (_strad_ext_pts / _strad_prem) * 100.0 if _strad_prem > 0 else 0.0

                                    _yield_pct = (_th_day_pts / _strad_prem) * 100.0
                                    _yield_pct = min(_yield_pct, 100.0)

                                    _cushion_ratio = _daily_cushion_pts / max(spot, 1000.0)
                                    _sell_score = min(max((_yield_pct * 0.4) + (min(_daily_cushion_pts, 250.0) / 250.0 * 30.0) + (20.0 if abs(_K - spot) <= 100 else 10.0), 0.0), 100.0)
                                    _convexity = (_strad_gam / max(_strad_prem, 1.0)) * 10000.0
                                    _buy_score = min(max(_convexity * 15.0 + (25.0 if abs(_K - spot) <= 75 else 10.0), 0.0), 100.0)

                                    if _convexity > _top_buyer_conv:
                                        _top_buyer_conv = _convexity
                                        _top_buyer_strike = _K
                                        _top_buyer_score = _buy_score

                                    _valid_strikes.append(_K)
                                    _ce_thetas_inr.append(_th_ce_inr)
                                    _pe_thetas_inr.append(_th_pe_inr)
                                    _straddle_thetas_inr.append(_th_strad_inr)
                                    _daily_cushions_pts.append(_daily_cushion_pts)
                                    _decay_yields_pct.append(_yield_pct)

                                    if abs(_th_strad_inr) > _peak_theta_val:
                                        _peak_theta_val = abs(_th_strad_inr)
                                        _peak_strike = _K

                                    _harvest_score = abs(_th_strad_inr) * _cushion_ratio * (1.0 + (_yield_pct / 100.0))
                                    if abs(_K - spot) <= (spot * 0.04) and _harvest_score > _best_harvest_score:
                                        _best_harvest_score = _harvest_score
                                        _sweet_strike = _K
                                        _sweet_cushion = _daily_cushion_pts
                                        _sweet_yield = _yield_pct
                                        _top_seller_strike = _K
                                        _top_seller_score = _sell_score

                                    _dist = _K - spot
                                    _dist_cls = GREEN if _dist > 0 else RED if _dist < 0 else ACCENT
                                    _dist_str = f"{_dist:+.0f}" if _dist != 0 else "ATM"

                                    if _sell_score >= 50.0 and _daily_cushion_pts >= 60.0:
                                        _edge_badge = f'<span class="badge-edge edge-seller" style="background:rgba(16,185,129,0.18);color:#10b981;font-size:10px;font-weight:700;padding:2px 6px;border-radius:4px;border:1px solid rgba(16,185,129,0.35);">SELLER ADV ({_sell_score:.0f})</span>'
                                    elif _buy_score >= 45.0:
                                        _edge_badge = f'<span class="badge-edge edge-buyer" style="background:rgba(2,132,199,0.18);color:#38bdf8;font-size:10px;font-weight:700;padding:2px 6px;border-radius:4px;border:1px solid rgba(56,189,248,0.35);">BUY CONVEX ({_buy_score:.0f})</span>'
                                    else:
                                        _edge_badge = f'<span class="badge-edge edge-neutral" style="background:rgba(148,163,184,0.15);color:#94a3b8;font-size:10px;font-weight:700;padding:2px 6px;border-radius:4px;">NEUTRAL</span>'

                                    if abs(_K - spot) <= 50 and _curr_dte_th <= 0.5:
                                        _ret_badge = f'<span style="background:rgba(239,68,68,0.2);color:#ef5350;font-size:10px;font-weight:800;padding:2px 6px;border-radius:4px;border:1px solid #ef5350;">EXIT / ROLL</span>'
                                    elif _ext_pct <= 20.0 and _strad_prem > 5.0:
                                        _ret_badge = f'<span style="background:rgba(255,213,79,0.2);color:#ffd54f;font-size:10px;font-weight:800;padding:2px 6px;border-radius:4px;border:1px solid #ffd54f;">TAKE PROFIT</span>'
                                    elif _daily_cushion_pts < 30.0 and abs(_K - spot) <= 150:
                                        _ret_badge = f'<span style="background:rgba(249,115,22,0.2);color:#fb923c;font-size:10px;font-weight:800;padding:2px 6px;border-radius:4px;border:1px solid #fb923c;">DEFEND</span>'
                                    elif _sell_score >= 55.0:
                                        _ret_badge = f'<span style="background:rgba(16,185,129,0.2);color:#10b981;font-size:10px;font-weight:800;padding:2px 6px;border-radius:4px;border:1px solid #10b981;">STAY IN TRADE</span>'
                                    else:
                                        _ret_badge = f'<span style="background:rgba(148,163,184,0.15);color:#94a3b8;font-size:10px;font-weight:700;padding:2px 6px;border-radius:4px;">HOLD</span>'

                                    _is_atm = (_K == _atm_strike_th)
                                    if _is_atm:
                                        _atm_strad_prem = _strad_prem
                                        _atm_strad_ext_pts = _strad_ext_pts
                                        _atm_strad_ext_inr = _strad_ext_inr

                                    _row_id = 'id="th-row-atm"' if _is_atm else ''
                                    _row_atm_attr = 'data-is-atm="true"' if _is_atm else ''
                                    _row_style = 'background:rgba(79,195,247,0.12);font-weight:700;border-left:3px solid #4fc3f7;cursor:pointer;' if _is_atm else 'cursor:pointer;'

                                    _theta_table_rows.append(
                                        f'<tr {_row_id} {_row_atm_attr} data-strike="{_K}" data-dist="{_dist}" '
                                        f'data-ce-ltp="{_ce_ltp:.2f}" data-pe-ltp="{_pe_ltp:.2f}" data-strad-ltp="{_strad_prem:.2f}" '
                                        f'data-ext-pts="{_strad_ext_pts:.2f}" data-ext-inr="{_strad_ext_inr:.0f}" '
                                        f'data-ce-theta="{_th_ce_inr:.2f}" data-pe-theta="{_th_pe_inr:.2f}" data-strad-theta="{_th_strad_inr:.2f}" '
                                        f'data-strad-gam="{_strad_gam:.6f}" data-delta="{_del_net:.4f}" data-vega="{_vg_net:.4f}" '
                                        f'data-cushion="{_daily_cushion_pts:.1f}" data-yield="{_yield_pct:.1f}" '
                                        f'data-sell-score="{_sell_score:.0f}" data-buy-score="{_buy_score:.0f}" '
                                        f'onclick="selectSimStrike({_K})" title="Click strike to simulate in Greek attribution engine" style="{_row_style}">'
                                        f'<td style="text-align:left;color:{WHITE};font-weight:700;">{_K:,.0f} {"(ATM)" if _is_atm else ""}</td>'
                                        f'<td style="color:{_dist_cls};">{_dist_str}</td>'
                                        f'<td style="color:#ffd54f;font-weight:700;">₹{_strad_prem:.1f}</td>'
                                        f'<td style="color:{YELLOW};">₹{_strad_ext_pts:.1f} <span style="color:{MUTED};font-size:10px;">({_ext_pct:.0f}%) / ₹{_strad_ext_inr:,.0f}</span></td>'
                                        f'<td>₹{_ce_ltp:.1f} <span style="color:{MUTED};font-size:10px;">({_ce_calc_iv:.1f}%)</span></td>'
                                        f'<td style="color:{RED};">₹{_th_ce_inr:,.0f}</td>'
                                        f'<td>₹{_pe_ltp:.1f} <span style="color:{MUTED};font-size:10px;">({_pe_calc_iv:.1f}%)</span></td>'
                                        f'<td style="color:{RED};">₹{_th_pe_inr:,.0f}</td>'
                                        f'<td style="color:{RED};font-weight:700;">₹{_th_strad_inr:,.0f}</td>'
                                        f'<td style="color:{YELLOW};">₹{_th_strad_1h:,.0f}</td>'
                                        f'<td>{_strad_gam:.5f}</td>'
                                        f'<td style="color:{GREEN};font-weight:700;">±{_daily_cushion_pts:.0f} pts</td>'
                                        f'<td style="color:{ACCENT};">{_yield_pct:.1f}%/d</td>'
                                        f'<td style="text-align:center;">{_edge_badge}</td>'
                                        f'<td style="text-align:center;">{_ret_badge}</td>'
                                        f'</tr>'
                                    )

                                _atm_idx = _valid_strikes.index(_atm_strike_th) if _atm_strike_th in _valid_strikes else 0
                                _atm_ce_inr = _ce_thetas_inr[_atm_idx]
                                _atm_pe_inr = _pe_thetas_inr[_atm_idx]
                                _atm_strad_inr = _straddle_thetas_inr[_atm_idx]
                                _atm_strad_1h = _atm_strad_inr / 6.25


                                _fig_th1 = make_subplots(
                                    rows=1, cols=2,
                                    subplot_titles=[
                                        "The Theta Cliff: Daily Decay (₹/lot) vs DTE",
                                        "Strike vs Daily Theta Profile (₹/lot)"
                                    ],
                                    horizontal_spacing=0.08
                                )

                                _dte_steps_th = [7.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.5, 0.25, 0.1]
                                for _tgt_strike, _tgt_name, _tgt_c in [
                                    (_atm_strike_th, f"ATM ({_atm_strike_th:.0f})", ACCENT),
                                    (_atm_strike_th + 200, f"OTM CE ({_atm_strike_th+200:.0f})", GREEN),
                                    (_atm_strike_th - 200, f"ITM CE ({_atm_strike_th-200:.0f})", YELLOW)
                                ]:
                                    _curve_y = []
                                    for _d in _dte_steps_th:
                                        _t_step = max(_d / 365.0, 1e-6)
                                        _sqT_s = np.sqrt(_t_step)
                                        _sig_s = max(_atm_iv_th / 100.0, 0.02)
                                        _d1_s = (np.log(spot/_tgt_strike) + (_r_th - _q_th + 0.5*_sig_s**2)*_t_step)/(_sig_s*_sqT_s)
                                        _d2_s = _d1_s - _sig_s*_sqT_s
                                        _th_s = (-spot * np.exp(-_q_th*_t_step) * norm.pdf(_d1_s) * _sig_s / (2*_sqT_s)
                                                 - _r_th * _tgt_strike * np.exp(-_r_th*_t_step) * norm.cdf(_d2_s)
                                                 + _q_th * spot * np.exp(-_q_th*_t_step) * norm.cdf(_d1_s)) / 365.0
                                        _curve_y.append(abs(_th_s * _lot_th))
                                    _fig_th1.add_trace(
                                        go.Scatter(
                                            x=_dte_steps_th, y=_curve_y,
                                            mode='lines+markers', name=_tgt_name,
                                            line=dict(color=_tgt_c, width=2.5),
                                            marker=dict(size=6)
                                        ),
                                        row=1, col=1
                                    )

                                _fig_th1.add_trace(
                                    go.Scatter(
                                        x=_valid_strikes, y=[abs(x) for x in _ce_thetas_inr],
                                        mode='lines+markers', name='Call Theta (₹)',
                                        line=dict(color=ACCENT, width=2),
                                        marker=dict(size=5)
                                    ),
                                    row=1, col=2
                                )
                                _fig_th1.add_trace(
                                    go.Scatter(
                                        x=_valid_strikes, y=[abs(x) for x in _pe_thetas_inr],
                                        mode='lines+markers', name='Put Theta (₹)',
                                        line=dict(color=RED, width=2),
                                        marker=dict(size=5)
                                    ),
                                    row=1, col=2
                                )
                                _fig_th1.add_trace(
                                    go.Scatter(
                                        x=_valid_strikes, y=[abs(x) for x in _straddle_thetas_inr],
                                        mode='lines', name='Straddle Theta (₹)',
                                        line=dict(color=YELLOW, width=2, dash='dash')
                                    ),
                                    row=1, col=2
                                )
                                _fig_th1.add_vline(x=spot, line_width=1.5, line_dash="dash", line_color="#ffffff", row=1, col=2)

                                _fig_th1.update_layout(
                                    height=350, autosize=True, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                                    font=dict(color=WHITE, family='Inter, sans-serif', size=10),
                                    legend=dict(bgcolor='rgba(18,18,42,0.85)', font=dict(size=9), orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
                                    margin=dict(l=40, r=20, t=40, b=30), hovermode='x unified'
                                )
                                _fig_th1.update_xaxes(title_text="Days to Expiry (DTE)", autorange="reversed", gridcolor='rgba(255,255,255,0.05)', row=1, col=1)
                                _fig_th1.update_yaxes(title_text="Decay (₹/lot/day)", gridcolor='rgba(255,255,255,0.05)', row=1, col=1)
                                _fig_th1.update_xaxes(title_text="Strike Price", gridcolor='rgba(255,255,255,0.05)', row=1, col=2)
                                _fig_th1.update_yaxes(title_text="Decay (₹/lot/day)", gridcolor='rgba(255,255,255,0.05)', row=1, col=2)
                                _plotly_th1 = _fig_th1.to_html(include_plotlyjs=False, full_html=False)

                                _fig_th2 = make_subplots(
                                    rows=1, cols=2,
                                    subplot_titles=[
                                        "Harvest Cushion: Daily Breakeven Move (±pts) vs Strike",
                                        "Strike × DTE 2D Decay Heatmap (₹/day)"
                                    ],
                                    horizontal_spacing=0.08
                                )

                                _fig_th2.add_trace(
                                    go.Scatter(
                                        x=_valid_strikes, y=_daily_cushions_pts,
                                        mode='lines+markers', name='Daily Cushion (±pts)',
                                        line=dict(color=GREEN, width=2.5),
                                        fill='tozeroy', fillcolor='rgba(16,185,129,0.1)'
                                    ),
                                    row=1, col=1
                                )
                                _fig_th2.add_vline(x=spot, line_width=1.5, line_dash="dash", line_color="#ffffff", row=1, col=1)

                                _heatmap_z = []
                                for _K in _valid_strikes:
                                    _row_z = []
                                    for _d in _dte_steps_th:
                                        _t_s = max(_d / 365.0, 1e-6)
                                        _sqT_s = np.sqrt(_t_s)
                                        _sig_s = max(_atm_iv_th / 100.0, 0.02)
                                        _d1_s = (np.log(spot/_K) + (_r_th - _q_th + 0.5*_sig_s**2)*_t_s)/(_sig_s*_sqT_s)
                                        _th_s = (-spot * np.exp(-_q_th*_t_s) * norm.pdf(_d1_s) * _sig_s / (2*_sqT_s)) / 365.0
                                        _row_z.append(round(abs(_th_s * _lot_th), 0))
                                    _heatmap_z.append(_row_z)

                                _fig_th2.add_trace(
                                    go.Heatmap(
                                        z=_heatmap_z,
                                        x=[f"{d}d" for d in _dte_steps_th],
                                        y=_valid_strikes,
                                        colorscale='Viridis',
                                        colorbar=dict(title=dict(text="₹/day", font=dict(size=10)), len=0.8, x=1.02),
                                        hoverongaps=False
                                    ),
                                    row=1, col=2
                                )

                                _fig_th2.update_layout(
                                    height=350, autosize=True, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                                    font=dict(color=WHITE, family='Inter, sans-serif', size=10),
                                    legend=dict(bgcolor='rgba(18,18,42,0.85)', font=dict(size=9)),
                                    margin=dict(l=40, r=20, t=40, b=30), hovermode='closest'
                                )
                                _fig_th2.update_xaxes(title_text="Strike Price", gridcolor='rgba(255,255,255,0.05)', row=1, col=1)
                                _fig_th2.update_yaxes(title_text="Breakeven Cushion (± pts)", gridcolor='rgba(255,255,255,0.05)', row=1, col=1)
                                _fig_th2.update_xaxes(title_text="DTE Horizon", gridcolor='rgba(255,255,255,0.05)', row=1, col=2)
                                _fig_th2.update_yaxes(title_text="Strike Price", gridcolor='rgba(255,255,255,0.05)', row=1, col=2)
                                _plotly_th2 = _fig_th2.to_html(include_plotlyjs=False, full_html=False)

                                # ── Quantitative Normalization & Granular Decay Calculations ──
                                _atm_idx = _valid_strikes.index(_atm_strike_th) if _atm_strike_th in _valid_strikes else 0
                                _atm_ce_inr = _ce_thetas_inr[_atm_idx] if _atm_idx < len(_ce_thetas_inr) else -1800.0
                                _atm_pe_inr = _pe_thetas_inr[_atm_idx] if _atm_idx < len(_pe_thetas_inr) else -1950.0
                                _atm_strad_inr = _straddle_thetas_inr[_atm_idx] if _atm_idx < len(_straddle_thetas_inr) else (_atm_ce_inr + _atm_pe_inr)
                                _atm_ce_1h = _atm_ce_inr / 6.25
                                _atm_pe_1h = _atm_pe_inr / 6.25
                                _atm_strad_1h = _atm_strad_inr / 6.25
                                _atm_ce_1m = _atm_ce_inr / 375.0
                                _atm_pe_1m = _atm_pe_inr / 375.0
                                _atm_strad_1m = _atm_strad_inr / 375.0

                                _atm_ce_row = _th_lookup.get((_atm_strike_th, 'CE'))
                                _atm_pe_row = _th_lookup.get((_atm_strike_th, 'PE'))
                                _atm_ce_ltp = float(_atm_ce_row['price']) if _atm_ce_row is not None else 0.0
                                _atm_pe_ltp = float(_atm_pe_row['price']) if _atm_pe_row is not None else 0.0
                                _atm_strad_prem = max(_atm_ce_ltp + _atm_pe_ltp, 0.01)

                                _atm_ce_ext_pts = max(0.0, _atm_ce_ltp - max(0.0, spot - _atm_strike_th))
                                _atm_ce_ext_inr = _atm_ce_ext_pts * _lot_th
                                _atm_pe_ext_pts = max(0.0, _atm_pe_ltp - max(0.0, _atm_strike_th - spot))
                                _atm_pe_ext_inr = _atm_pe_ext_pts * _lot_th
                                _atm_strad_ext_pts = _atm_ce_ext_pts + _atm_pe_ext_pts
                                _atm_strad_ext_inr = _atm_strad_ext_pts * _lot_th

                                _atm_ce_yield = (abs(_atm_ce_inr / _lot_th) / max(_atm_ce_ltp, 0.1)) * 100.0
                                _atm_pe_yield = (abs(_atm_pe_inr / _lot_th) / max(_atm_pe_ltp, 0.1)) * 100.0
                                _atm_strad_yield = (abs(_atm_strad_inr / _lot_th) / max(_atm_strad_prem, 0.1)) * 100.0

                                # ── Theta Asymmetry & Bleed Dominance Comparator ──
                                _atm_ce_inr_abs = abs(_atm_ce_inr)
                                _atm_pe_inr_abs = abs(_atm_pe_inr)
                                _atm_th_tot = _atm_ce_inr_abs + _atm_pe_inr_abs
                                _atm_ce_pct = (_atm_ce_inr_abs / max(_atm_th_tot, 1e-6)) * 100.0
                                _atm_pe_pct = (_atm_pe_inr_abs / max(_atm_th_tot, 1e-6)) * 100.0
                                _atm_th_ratio = _atm_ce_inr_abs / max(_atm_pe_inr_abs, 1e-6)

                                if _atm_pe_inr_abs > _atm_ce_inr_abs * 1.02:
                                    _th_leader = "PUTS"
                                    _th_diff_inr = _atm_pe_inr_abs - _atm_ce_inr_abs
                                    _th_diff_pts = _th_diff_inr / _lot_th
                                    _th_diff_pct = (_th_diff_inr / max(_atm_ce_inr_abs, 1.0)) * 100.0
                                    _th_verdict_col = "#ff7043"
                                    _th_verdict_text = f"PUT THETA IS HIGHER (+{_th_diff_pct:.1f}% vs Calls)"
                                    _th_insight = f"Put buyers bleeding faster (-₹{_th_diff_inr:,.0f}/d more). Put writing offers higher time-decay harvest than Call writing."
                                elif _atm_ce_inr_abs > _atm_pe_inr_abs * 1.02:
                                    _th_leader = "CALLS"
                                    _th_diff_inr = _atm_ce_inr_abs - _atm_pe_inr_abs
                                    _th_diff_pts = _th_diff_inr / _lot_th
                                    _th_diff_pct = (_th_diff_inr / max(_atm_pe_inr_abs, 1.0)) * 100.0
                                    _th_verdict_col = "#38bdf8"
                                    _th_verdict_text = f"CALL THETA IS HIGHER (+{_th_diff_pct:.1f}% vs Puts)"
                                    _th_insight = f"Call buyers bleeding faster (-₹{_th_diff_inr:,.0f}/d more). Call writing offers higher time-decay harvest than Put writing."
                                else:
                                    _th_leader = "BALANCED"
                                    _th_diff_inr = 0.0
                                    _th_diff_pts = 0.0
                                    _th_diff_pct = 0.0
                                    _th_verdict_col = "#ffd54f"
                                    _th_verdict_text = "THETA DECAY IS SYMMETRICAL"
                                    _th_insight = "Time bleed is evenly matched between Calls and Puts (neutral decay bias)."

                                # Chain-wide total decay (sum across all valid strikes)
                                _chain_ce_th_tot = sum([abs(x) for x in _ce_thetas_inr])
                                _chain_pe_th_tot = sum([abs(x) for x in _pe_thetas_inr])
                                _chain_tot_th = _chain_ce_th_tot + _chain_pe_th_tot
                                _chain_ce_pct = (_chain_ce_th_tot / max(_chain_tot_th, 1e-6)) * 100.0
                                _chain_pe_pct = (_chain_pe_th_tot / max(_chain_tot_th, 1e-6)) * 100.0

                                # Expected Move (1-sigma, Gatheral 2006 / Merton 1973)
                                _em_pts = spot * (_atm_iv_th / 100.0) * np.sqrt(max(_curr_dte_th / 365.0, 1e-4))
                                if _em_pts < 10.0: _em_pts = spot * 0.008

                                # Renormalized Alpha Metric (Bouchaud & Sornette 1994/2000)
                                _atm_sig = max(_atm_iv_th / 100.0, 0.02)
                                _atm_gam_ce = _calc_merton_theta(_atm_strike_th, _atm_sig, 'CE')[1]
                                _atm_gam_pe = _calc_merton_theta(_atm_strike_th, _atm_sig, 'PE')[1]
                                _atm_gam = _atm_gam_ce + _atm_gam_pe
                                _gamma_hazard_inr = 0.5 * _atm_gam * (_em_pts ** 2) * _lot_th
                                _renorm_alpha = abs(_atm_strad_inr) / max(_gamma_hazard_inr, 1.0)
                                _alpha_color = GREEN if _renorm_alpha >= 1.0 else YELLOW if _renorm_alpha >= 0.7 else RED
                                _alpha_label = "Alpha Edge Zone" if _renorm_alpha >= 1.0 else "Neutral Buffer" if _renorm_alpha >= 0.7 else "Gamma Hazard Zone"

                                # 9-Step Normalized Scenario Matrix (precalculated for instant render)
                                _scen_table_rows = []
                                for _z in [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]:
                                    _ds_z = round(_z * _em_pts, 1)
                                    _s_z = round(spot + _ds_z, 1)
                                    _d_ce_z = 0.5 * _ds_z + 0.5 * (_atm_gam / 2.0) * (_ds_z ** 2)
                                    _d_pe_z = -0.5 * _ds_z + 0.5 * (_atm_gam / 2.0) * (_ds_z ** 2)
                                    _ce_z = max(0.05, round(_atm_ce_ltp + _d_ce_z, 1))
                                    _pe_z = max(0.05, round(_atm_pe_ltp + _d_pe_z, 1))
                                    _st_z = round(_ce_z + _pe_z, 1)
                                    _pnl_s_pts = round(_atm_strad_prem - _st_z, 1)
                                    _pnl_s_inr = round(_pnl_s_pts * _lot_th, 0)
                                    _pnl_s_pct = round((_pnl_s_pts / max(_atm_strad_prem, 0.1)) * 100.0, 1)
                                    _pnl_b_inr = -_pnl_s_inr

                                    if abs(_z) >= 1.5:
                                        _cause = "Gamma Hazard (Curvature Loss)"
                                        _zone = "DANGER / DEFEND"
                                        _z_col = "#ef5350"
                                        _z_bg = "rgba(239,68,68,0.15)"
                                    elif abs(_z) >= 1.0:
                                        _cause = "Spot at Breakeven Edge"
                                        _zone = "BREAKEVEN EDGE"
                                        _z_col = "#ffd54f"
                                        _z_bg = "rgba(255,213,79,0.15)"
                                    elif abs(_z) >= 0.5:
                                        _cause = "Theta Buffering Spot Move"
                                        _zone = "SAFE / HARVEST"
                                        _z_col = "#10b981"
                                        _z_bg = "rgba(16,185,129,0.15)"
                                    else:
                                        _cause = "Pure Calendar Bleed"
                                        _zone = "MAX HARVEST"
                                        _z_col = "#10b981"
                                        _z_bg = "rgba(16,185,129,0.25)"

                                    _row_active = 'style="background:rgba(2,132,199,0.18); font-weight:700;"' if _z == 0.0 else ''
                                    _s_sign = '+' if _pnl_s_inr >= 0 else ''
                                    _b_sign = '+' if _pnl_b_inr >= 0 else ''
                                    _s_col = '#10b981' if _pnl_s_inr >= 0 else '#ef5350'
                                    _b_col = '#10b981' if _pnl_b_inr >= 0 else '#ef5350'

                                    _scen_table_rows.append(
                                        f'<tr data-z="{_z}" {_row_active}>'
                                        f'<td style="text-align:left; font-weight:800; color:{_z_col};">{_z:+.1f}σ</td>'
                                        f'<td>{_ds_z:+.1f} pts</td>'
                                        f'<td style="font-weight:700;">{_s_z:,.1f}</td>'
                                        f'<td style="color:#38bdf8;">₹{_ce_z:.1f}</td>'
                                        f'<td style="color:#ff7043;">₹{_pe_z:.1f}</td>'
                                        f'<td style="color:#ffd54f; font-weight:700;">₹{_st_z:.1f}</td>'
                                        f'<td style="color:{_s_col}; font-weight:800;">{_s_sign}₹{_pnl_s_inr:,.0f}</td>'
                                        f'<td style="color:{_s_col};">{_s_sign}{_pnl_s_pct:.1f}%</td>'
                                        f'<td style="color:{_b_col}; font-weight:800;">{_b_sign}₹{_pnl_b_inr:,.0f}</td>'
                                        f'<td style="text-align:left; color:#94a3b8; font-size:11px;">{_cause}</td>'
                                        f'<td style="text-align:center;"><span style="background:{_z_bg}; color:{_z_col}; font-size:10px; font-weight:800; padding:2px 8px; border-radius:4px; border:1px solid {_z_col}44;">{_zone}</span></td>'
                                        f'</tr>'
                                    )

                                _strike_options = []
                                for _s_opt in _valid_strikes:
                                    _sel = 'selected' if _s_opt == _atm_strike_th else ''
                                    _atm_tag = ' (ATM)' if _s_opt == _atm_strike_th else ''
                                    _strike_options.append(f'<option value="{_s_opt}" {_sel}>{_s_opt:,.0f}{_atm_tag}</option>')

                                theta_tab_html = f'''
                                <div style="display:flex; flex-direction:column; gap:12px;">
                                    <!-- CONTROLS & SELECTION BAR -->
                                    <div class="action-bar" style="justify-content:space-between; flex-wrap:wrap; gap:8px;">
                                        <div style="display:flex; align-items:center; gap:8px;">
                                            <span style="font-size:11px; font-weight:800; color:{MUTED}; text-transform:uppercase; letter-spacing:1px;">Target Strike:</span>
                                            <select id="sel-th-strike" onchange="selectSimStrike(parseFloat(this.value))" style="background:#12122a; color:{WHITE}; border:1px solid #0284c7; padding:4px 8px; border-radius:6px; font-size:11px; font-weight:700;">
                                                {''.join(_strike_options)}
                                            </select>
                                            <button class="btn" id="btn-th-autolock" onclick="toggleAutoLockATM(this)" title="Auto-center table around ATM strike and prevent jumping on refresh" style="display:flex; align-items:center; gap:5px; padding:4px 8px; font-size:11px; font-weight:700; background:rgba(16,185,129,0.18); color:#10b981; border:1px solid #10b981; border-radius:6px; cursor:pointer;">
                                                &#128274; Lock ATM: ON
                                            </button>
                                        </div>
                                        <div style="display:flex; align-items:center; gap:6px;">
                                            <span style="font-size:11px; font-weight:800; color:{MUTED}; text-transform:uppercase; letter-spacing:1px;">Decay Unit:</span>
                                            <div style="display:flex; background:#0c0c1e; padding:2px; border-radius:6px; border:1px solid #2a2a4a;">
                                                <button class="th-unit-btn active" id="btn-unit-inr" onclick="setThetaUnit('INR')" style="padding:3px 8px; font-size:10px; font-weight:700; border-radius:4px; border:none; cursor:pointer; background:#0284c7; color:#ffffff;">₹ / Lot (65)</button>
                                                <button class="th-unit-btn" id="btn-unit-pts" onclick="setThetaUnit('PTS')" style="padding:3px 8px; font-size:10px; font-weight:700; border-radius:4px; border:none; cursor:pointer; background:transparent; color:#94a3b8;">Pts / Share</button>
                                            </div>
                                        </div>
                                        <div style="display:flex; align-items:center; gap:6px;">
                                            <span style="font-size:11px; font-weight:800; color:{MUTED}; text-transform:uppercase;">View:</span>
                                            <button class="theta-type-btn active" id="btn-th-ce" onclick="toggleThetaView('CE')">CE</button>
                                            <button class="theta-type-btn" id="btn-th-pe" onclick="toggleThetaView('PE')">PE</button>
                                            <button class="theta-type-btn" id="btn-th-straddle" onclick="toggleThetaView('STRADDLE')">Straddle</button>
                                        </div>
                                        <div style="display:flex; align-items:center; gap:6px;">
                                            <span style="font-size:11px; font-weight:800; color:{MUTED}; text-transform:uppercase;">Focus:</span>
                                            <button class="theta-focus-btn active" id="btn-focus-all" onclick="setThetaTableFocus('all')">All</button>
                                            <button class="theta-focus-btn" id="btn-focus-seller" onclick="setThetaTableFocus('seller')">Seller</button>
                                            <button class="theta-focus-btn" id="btn-focus-buyer" onclick="setThetaTableFocus('buyer')">Buyer</button>
                                        </div>
                                        <div style="display:flex; align-items:center; gap:6px;">
                                            <span style="font-size:11px; font-weight:800; color:{MUTED}; text-transform:uppercase;">Model:</span>
                                            <button class="theta-model-btn active" id="btn-th-bsm" onclick="toggleThetaModel('bsm')">BSM</button>
                                            <button class="theta-model-btn" id="btn-th-heston" onclick="toggleThetaModel('heston')">Heston</button>
                                            <button class="theta-model-btn" id="btn-th-both" onclick="toggleThetaModel('both')">Both</button>
                                            <select id="sel-th-range" onchange="changeThetaRange(this.value)" style="background:#12122a; color:{WHITE}; border:1px solid #2a2a4a; padding:3px 6px; border-radius:6px; font-size:10px;">
                                                <option value="5">±5%</option>
                                                <option value="10" selected>±10%</option>
                                                <option value="15">±15%</option>
                                            </select>
                                            <button class="btn" id="btn-th-recalc" onclick="refreshThetaDecay(this)" title="Recalculate model" style="padding:4px 8px; font-size:10px; font-weight:700; background:#0284c7; color:#ffffff; border:none; border-radius:6px; cursor:pointer;">
                                                &#8635; Recalc
                                            </button>
                                        </div>
                                    </div>

                                    <!-- DEDICATED CALL vs PUT THETA COMPARISON CARD -->
                                    <div class="card" id="card-theta-comparison" style="border-left:4px solid {_th_verdict_col}; background:linear-gradient(135deg, rgba(18,18,42,0.95), rgba(10,14,28,0.95)); padding:14px; margin-bottom:4px;">
                                        <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:10px; margin-bottom:10px;">
                                            <div style="display:flex; align-items:center; gap:10px;">
                                                <span style="font-size:18px;">⚖️</span>
                                                <div>
                                                    <div style="font-size:11px; font-weight:800; color:{MUTED}; text-transform:uppercase; letter-spacing:1px;">CALL vs PUT THETA ASYMMETRY COMPARATOR</div>
                                                    <div id="th-asymmetry-verdict" style="font-size:16px; font-weight:900; color:{_th_verdict_col}; display:flex; align-items:center; gap:8px;">
                                                        <span id="th-verdict-title">{_th_verdict_text}</span>
                                                        <span id="th-leader-badge" style="font-size:10px; font-weight:800; padding:2px 8px; border-radius:4px; background:{_th_verdict_col}22; border:1px solid {_th_verdict_col}66; color:{_th_verdict_col};">{_th_leader} BLEED DOMINANCE</span>
                                                    </div>
                                                </div>
                                            </div>
                                            <div style="text-align:right;">
                                                <div style="font-size:11px; color:{MUTED};">Call / Put Ratio (CE/PE)</div>
                                                <div id="th-ratio-display" style="font-size:18px; font-weight:900; color:{WHITE}; font-family:'JetBrains Mono', monospace;">{_atm_th_ratio:.2f}x</div>
                                            </div>
                                        </div>

                                        <!-- Visual 2-Tone Asymmetry Gauge Bar -->
                                        <div style="margin-bottom:10px;">
                                            <div style="display:flex; justify-content:space-between; font-size:11px; font-weight:700; margin-bottom:4px;">
                                                <span style="color:#38bdf8;">CALL DECAY: <span id="th-ce-gauge-lbl">{_atm_ce_pct:.1f}%</span> (₹<span id="th-ce-val-lbl">{_atm_ce_inr_abs:,.0f}</span>/d)</span>
                                                <span id="th-gauge-center-strike" style="color:{MUTED}; font-size:10px;">TARGET STRIKE {_atm_strike_th:,.0f}</span>
                                                <span style="color:#ff7043;">PUT DECAY: <span id="th-pe-gauge-lbl">{_atm_pe_pct:.1f}%</span> (₹<span id="th-pe-val-lbl">{_atm_pe_inr_abs:,.0f}</span>/d)</span>
                                            </div>
                                            <div style="height:12px; border-radius:6px; background:#121226; overflow:hidden; display:flex; border:1px solid #2a2a4a; position:relative;">
                                                <div id="th-gauge-ce" style="width:{_atm_ce_pct:.1f}%; height:100%; background:linear-gradient(90deg, #0284c7, #38bdf8); transition:width 0.3s ease;"></div>
                                                <div id="th-gauge-pe" style="width:{_atm_pe_pct:.1f}%; height:100%; background:linear-gradient(90deg, #ff7043, #ef4444); transition:width 0.3s ease;"></div>
                                                <!-- Center 50% marker line -->
                                                <div style="position:absolute; left:50%; top:0; bottom:0; width:2px; background:rgba(255,255,255,0.4); transform:translateX(-50%); pointer-events:none;"></div>
                                            </div>
                                        </div>

                                        <!-- 3 Actionable Summary Boxes -->
                                        <div style="display:grid; grid-template-columns: repeat(3, 1fr); gap:8px;">
                                            <div class="metric-box" style="padding:8px 10px; text-align:left; background:rgba(255,255,255,0.02); border:1px solid rgba(255,255,255,0.06);">
                                                <div class="metric-label" style="color:{MUTED}; font-size:10px;">DECAY SPREAD (STRIKE)</div>
                                                <div id="th-diff-detail" style="font-size:13px; font-weight:800; color:{WHITE}; margin:2px 0;">Δ ₹{_th_diff_inr:,.0f} / lot ({_th_diff_pts:.1f} pts)</div>
                                                <div style="font-size:10px; color:{_th_verdict_col};" id="th-diff-sub">{_th_leader} bleeding faster</div>
                                            </div>
                                            <div class="metric-box" style="padding:8px 10px; text-align:left; background:rgba(255,255,255,0.02); border:1px solid rgba(255,255,255,0.06);">
                                                <div class="metric-label" style="color:{MUTED}; font-size:10px;">TRADER PLAYBOOK</div>
                                                <div id="th-insight-detail" style="font-size:11px; color:#e2e8f0; line-height:1.3; margin-top:2px;">{_th_insight}</div>
                                            </div>
                                            <div class="metric-box" style="padding:8px 10px; text-align:left; background:rgba(255,255,255,0.02); border:1px solid rgba(255,255,255,0.06);">
                                                <div class="metric-label" style="color:{MUTED}; font-size:10px;">CHAIN-WIDE MACRO THETA</div>
                                                <div style="font-size:12px; font-weight:800; color:{WHITE}; margin:2px 0;">
                                                    <span style="color:#38bdf8;">CE: {_chain_ce_pct:.0f}%</span> vs <span style="color:#ff7043;">PE: {_chain_pe_pct:.0f}%</span>
                                                </div>
                                                <div style="font-size:10px; color:{MUTED};">₹{_chain_ce_th_tot/1e5:.1f}L CE vs ₹{_chain_pe_th_tot/1e5:.1f}L PE total</div>
                                            </div>
                                        </div>
                                    </div>

                                    <!-- EXECUTIVE 3-WAY DECAY PANEL (Call vs Put vs Straddle) -->
                                    <div style="display:grid; grid-template-columns: repeat(3, 1fr); gap:10px;">
                                        <!-- CALL CARD -->
                                        <div class="card" style="border-top:4px solid #38bdf8; background:rgba(18,18,42,0.85); padding:12px;">
                                            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
                                                <div style="display:flex; align-items:center; gap:6px;">
                                                    <span style="font-size:13px; font-weight:900; color:#38bdf8; letter-spacing:1px;">CALL (CE) DECAY</span>
                                                    <span id="card-ce-strike" style="font-size:11px; color:#94a3b8; font-weight:700;">{_atm_strike_th:,.0f}</span>
                                                    <span id="card-ce-leader-tag" style="font-size:9px; font-weight:800; padding:2px 6px; border-radius:4px; {'background:rgba(56,189,248,0.2); color:#38bdf8; border:1px solid #38bdf8;' if _th_leader == 'CALLS' else 'background:rgba(255,255,255,0.06); color:#94a3b8;'}">{'🔥 HIGHER' if _th_leader == 'CALLS' else 'LOWER DECAY'}</span>
                                                </div>
                                                <div style="font-size:14px; font-weight:800; color:{WHITE};" id="card-ce-price">LTP: ₹{_atm_ce_ltp:.1f}</div>
                                            </div>
                                            <div style="display:grid; grid-template-columns: 1fr 1fr; gap:6px; margin-bottom:8px;">
                                                <div class="metric-box" style="padding:6px 8px; background:rgba(2,132,199,0.08); border:1px solid rgba(2,132,199,0.25);">
                                                    <div class="metric-label" style="color:#38bdf8;">Decay / Day</div>
                                                    <div id="card-ce-day" style="font-size:15px; font-weight:800; color:#ef5350;">-₹{abs(_atm_ce_inr):,.0f}</div>
                                                    <div id="card-ce-day-sub" class="metric-sub">-{abs(_atm_ce_inr/_lot_th):.1f} pts/d</div>
                                                </div>
                                                <div class="metric-box" style="padding:6px 8px; background:rgba(2,132,199,0.08); border:1px solid rgba(2,132,199,0.25);">
                                                    <div class="metric-label" style="color:#38bdf8;">Decay / Hour</div>
                                                    <div id="card-ce-hour" style="font-size:15px; font-weight:800; color:#ffd54f;">-₹{abs(_atm_ce_1h):,.0f}</div>
                                                    <div id="card-ce-hour-sub" class="metric-sub">-{abs(_atm_ce_1h/_lot_th):.2f} pts/h</div>
                                                </div>
                                            </div>
                                            <div style="display:grid; grid-template-columns: 1fr 1fr; gap:6px;">
                                                <div class="metric-box" style="padding:6px 8px;">
                                                    <div class="metric-label">Decay / Minute</div>
                                                    <div id="card-ce-min" style="font-size:13px; font-weight:800; color:#94a3b8;">-₹{abs(_atm_ce_1m):.2f}</div>
                                                    <div id="card-ce-min-sub" class="metric-sub">-{abs(_atm_ce_1m/_lot_th):.3f} pts/m</div>
                                                </div>
                                                <div class="metric-box" style="padding:6px 8px;">
                                                    <div class="metric-label">Till Expiry</div>
                                                    <div id="card-ce-exp" style="font-size:13px; font-weight:800; color:#38bdf8;">₹{_atm_ce_ext_inr:,.0f}</div>
                                                    <div id="card-ce-exp-sub" class="metric-sub">{_atm_ce_ext_pts:.1f} pts ext</div>
                                                </div>
                                            </div>
                                            <div style="margin-top:6px; font-size:10px; color:#94a3b8; display:flex; justify-content:space-between;">
                                                <span>Yield: <strong id="card-ce-yield" style="color:#38bdf8;">{_atm_ce_yield:.1f}%/d</strong></span>
                                                <span>Delta: <strong id="card-ce-delta" style="color:{WHITE};">+0.50</strong></span>
                                            </div>
                                        </div>

                                        <!-- PUT CARD -->
                                        <div class="card" style="border-top:4px solid #ff7043; background:rgba(18,18,42,0.85); padding:12px;">
                                            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
                                                <div style="display:flex; align-items:center; gap:6px;">
                                                    <span style="font-size:13px; font-weight:900; color:#ff7043; letter-spacing:1px;">PUT (PE) DECAY</span>
                                                    <span id="card-pe-strike" style="font-size:11px; color:#94a3b8; font-weight:700;">{_atm_strike_th:,.0f}</span>
                                                    <span id="card-pe-leader-tag" style="font-size:9px; font-weight:800; padding:2px 6px; border-radius:4px; {'background:rgba(255,112,67,0.2); color:#ff7043; border:1px solid #ff7043;' if _th_leader == 'PUTS' else 'background:rgba(255,255,255,0.06); color:#94a3b8;'}">{'🔥 HIGHER' if _th_leader == 'PUTS' else 'LOWER DECAY'}</span>
                                                </div>
                                                <div style="font-size:14px; font-weight:800; color:{WHITE};" id="card-pe-price">LTP: ₹{_atm_pe_ltp:.1f}</div>
                                            </div>
                                            <div style="display:grid; grid-template-columns: 1fr 1fr; gap:6px; margin-bottom:8px;">
                                                <div class="metric-box" style="padding:6px 8px; background:rgba(255,112,67,0.08); border:1px solid rgba(255,112,67,0.25);">
                                                    <div class="metric-label" style="color:#ff7043;">Decay / Day</div>
                                                    <div id="card-pe-day" style="font-size:15px; font-weight:800; color:#ef5350;">-₹{abs(_atm_pe_inr):,.0f}</div>
                                                    <div id="card-pe-day-sub" class="metric-sub">-{abs(_atm_pe_inr/_lot_th):.1f} pts/d</div>
                                                </div>
                                                <div class="metric-box" style="padding:6px 8px; background:rgba(255,112,67,0.08); border:1px solid rgba(255,112,67,0.25);">
                                                    <div class="metric-label" style="color:#ff7043;">Decay / Hour</div>
                                                    <div id="card-pe-hour" style="font-size:15px; font-weight:800; color:#ffd54f;">-₹{abs(_atm_pe_1h):,.0f}</div>
                                                    <div id="card-pe-hour-sub" class="metric-sub">-{abs(_atm_pe_1h/_lot_th):.2f} pts/h</div>
                                                </div>
                                            </div>
                                            <div style="display:grid; grid-template-columns: 1fr 1fr; gap:6px;">
                                                <div class="metric-box" style="padding:6px 8px;">
                                                    <div class="metric-label">Decay / Minute</div>
                                                    <div id="card-pe-min" style="font-size:13px; font-weight:800; color:#94a3b8;">-₹{abs(_atm_pe_1m):.2f}</div>
                                                    <div id="card-pe-min-sub" class="metric-sub">-{abs(_atm_pe_1m/_lot_th):.3f} pts/m</div>
                                                </div>
                                                <div class="metric-box" style="padding:6px 8px;">
                                                    <div class="metric-label">Till Expiry</div>
                                                    <div id="card-pe-exp" style="font-size:13px; font-weight:800; color:#ff7043;">₹{_atm_pe_ext_inr:,.0f}</div>
                                                    <div id="card-pe-exp-sub" class="metric-sub">{_atm_pe_ext_pts:.1f} pts ext</div>
                                                </div>
                                            </div>
                                            <div style="margin-top:6px; font-size:10px; color:#94a3b8; display:flex; justify-content:space-between;">
                                                <span>Yield: <strong id="card-pe-yield" style="color:#ff7043;">{_atm_pe_yield:.1f}%/d</strong></span>
                                                <span>Delta: <strong id="card-pe-delta" style="color:{WHITE};">-0.50</strong></span>
                                            </div>
                                        </div>

                                        <!-- STRADDLE COMBINED CARD -->
                                        <div class="card" style="border-top:4px solid #ffd54f; background:rgba(18,18,42,0.85); padding:12px;">
                                            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
                                                <div style="display:flex; align-items:center; gap:6px;">
                                                    <span style="font-size:13px; font-weight:900; color:#ffd54f; letter-spacing:1px;">STRADDLE (CE+PE)</span>
                                                    <span id="card-strad-strike" style="font-size:11px; color:#94a3b8; font-weight:700;">{_atm_strike_th:,.0f}</span>
                                                </div>
                                                <div style="font-size:14px; font-weight:800; color:#ffd54f;" id="card-strad-price">LTP: ₹{_atm_strad_prem:.1f}</div>
                                            </div>
                                            <div style="display:grid; grid-template-columns: 1fr 1fr; gap:6px; margin-bottom:8px;">
                                                <div class="metric-box" style="padding:6px 8px; background:rgba(255,213,79,0.08); border:1px solid rgba(255,213,79,0.25);">
                                                    <div class="metric-label" style="color:#ffd54f;">Combined / Day</div>
                                                    <div id="card-strad-day" style="font-size:15px; font-weight:800; color:#ef5350;">-₹{abs(_atm_strad_inr):,.0f}</div>
                                                    <div id="card-strad-day-sub" class="metric-sub">-{abs(_atm_strad_inr/_lot_th):.1f} pts/d</div>
                                                </div>
                                                <div class="metric-box" style="padding:6px 8px; background:rgba(255,213,79,0.08); border:1px solid rgba(255,213,79,0.25);">
                                                    <div class="metric-label" style="color:#ffd54f;">Combined / Hour</div>
                                                    <div id="card-strad-hour" style="font-size:15px; font-weight:800; color:#ffd54f;">-₹{abs(_atm_strad_1h):,.0f}</div>
                                                    <div id="card-strad-hour-sub" class="metric-sub">-{abs(_atm_strad_1h/_lot_th):.2f} pts/h</div>
                                                </div>
                                            </div>
                                            <div style="display:grid; grid-template-columns: 1fr 1fr; gap:6px;">
                                                <div class="metric-box" style="padding:6px 8px;">
                                                    <div class="metric-label">Combined / Minute</div>
                                                    <div id="card-strad-min" style="font-size:13px; font-weight:800; color:#94a3b8;">-₹{abs(_atm_strad_1m):.2f}</div>
                                                    <div id="card-strad-min-sub" class="metric-sub">-{abs(_atm_strad_1m/_lot_th):.3f} pts/m</div>
                                                </div>
                                                <div class="metric-box" style="padding:6px 8px;">
                                                    <div class="metric-label">Total Expiry Bleed</div>
                                                    <div id="card-strad-exp" style="font-size:13px; font-weight:800; color:#ffd54f;">₹{_atm_strad_ext_inr:,.0f}</div>
                                                    <div id="card-strad-exp-sub" class="metric-sub">{_atm_strad_ext_pts:.1f} pts ext</div>
                                                </div>
                                            </div>
                                            <div style="margin-top:6px; font-size:10px; color:#94a3b8; display:flex; justify-content:space-between; align-items:center;">
                                                <span>Yield: <strong id="card-strad-yield" style="color:#ffd54f;">{_atm_strad_yield:.1f}%/d</strong></span>
                                                <span>Renorm Alpha: <strong id="card-strad-alpha" style="color:{_alpha_color}; font-weight:800;">{_renorm_alpha:.2f} ({_alpha_label})</strong></span>
                                            </div>
                                        </div>
                                    </div>

                                    <!-- GREEK CAUSE & EFFECT ATTRIBUTION DOCK -->
                                    <div class="card" style="padding:12px; border-left:4px solid #0284c7; background:rgba(18,18,42,0.75);">
                                        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px; flex-wrap:wrap; gap:6px;">
                                            <div style="display:flex; align-items:center; gap:8px;">
                                                <span style="color:#38bdf8; font-size:12px; font-weight:900; letter-spacing:1px;">🔬 GREEK CAUSE & EFFECT ATTRIBUTION ENGINE</span>
                                                <span style="background:rgba(2,132,199,0.2); color:#38bdf8; font-size:10px; font-weight:700; padding:2px 8px; border-radius:12px; border:1px solid rgba(2,132,199,0.35);">Merton 1973 · Bouchaud-Sornette 2000</span>
                                            </div>
                                            <div style="font-size:11px; color:{MUTED};">Decomposing option price change: <strong style="color:{WHITE};">ΔP = ΘΔt + ΔΔS + ½Γ(ΔS)² + VΔσ</strong></div>
                                        </div>

                                        <div style="display:grid; grid-template-columns: repeat(4, 1fr); gap:8px; margin-bottom:8px;">
                                            <div class="metric-box" style="border:1px solid rgba(16,185,129,0.3); background:rgba(16,185,129,0.06); text-align:left; padding:8px 10px;">
                                                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:2px;">
                                                    <span style="color:#10b981; font-weight:800; font-size:10px;">1. THETA (ΘΔt)</span>
                                                    <span id="attr-th-rate" style="font-size:10px; color:#10b981; font-weight:700;">-₹{abs(_atm_strad_inr):,.0f}/d</span>
                                                </div>
                                                <div id="attr-th-val" style="font-size:16px; font-weight:800; color:#10b981; margin-bottom:2px;">+₹0</div>
                                                <div style="font-size:9px; color:#94a3b8; line-height:1.2;">Pure calendar time bleed. Steady profit for seller; steady loss for buyer.</div>
                                            </div>

                                            <div class="metric-box" style="border:1px solid rgba(56,189,248,0.3); background:rgba(56,189,248,0.06); text-align:left; padding:8px 10px;">
                                                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:2px;">
                                                    <span style="color:#38bdf8; font-weight:800; font-size:10px;">2. DELTA (ΔΔS)</span>
                                                    <span id="attr-del-rate" style="font-size:10px; color:#38bdf8; font-weight:700;">Net: 0.00</span>
                                                </div>
                                                <div id="attr-del-val" style="font-size:16px; font-weight:800; color:#38bdf8; margin-bottom:2px;">₹0</div>
                                                <div style="font-size:9px; color:#94a3b8; line-height:1.2;">Linear directional impact. Straddle net delta is near zero at ATM.</div>
                                            </div>

                                            <div class="metric-box" style="border:1px solid rgba(255,213,79,0.3); background:rgba(255,213,79,0.06); text-align:left; padding:8px 10px;">
                                                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:2px;">
                                                    <span style="color:#ffd54f; font-weight:800; font-size:10px;">3. GAMMA (½ΓΔS²)</span>
                                                    <span id="attr-gam-rate" style="font-size:10px; color:#ffd54f; font-weight:700;">Γ: {_atm_gam:.4f}</span>
                                                </div>
                                                <div id="attr-gam-val" style="font-size:16px; font-weight:800; color:#ffd54f; margin-bottom:2px;">-₹0</div>
                                                <div style="font-size:9px; color:#94a3b8; line-height:1.2;">Curvature drag. Accelerates against seller on large moves; boosts buyer.</div>
                                            </div>

                                            <div class="metric-box" style="border:1px solid rgba(192,132,252,0.3); background:rgba(192,132,252,0.06); text-align:left; padding:8px 10px;">
                                                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:2px;">
                                                    <span style="color:#c084fc; font-weight:800; font-size:10px;">4. VEGA (VΔσ)</span>
                                                    <span id="attr-veg-rate" style="font-size:10px; color:#c084fc; font-weight:700;">₹1,950/1%</span>
                                                </div>
                                                <div id="attr-veg-val" style="font-size:16px; font-weight:800; color:#c084fc; margin-bottom:2px;">₹0</div>
                                                <div style="font-size:9px; color:#94a3b8; line-height:1.2;">Implied volatility shock. Vol expansion hurts seller; vol crush helps.</div>
                                            </div>
                                        </div>

                                        <div style="background:#0a0a18; padding:6px 10px; border-radius:6px; font-size:11px; display:flex; justify-content:space-between; align-items:center; border:1px solid #2a2a4a;">
                                            <div>
                                                <span style="color:#94a3b8; font-weight:700;">ATTRIBUTION FORMULA:</span>
                                                <span style="color:{WHITE}; font-family:monospace; margin-left:6px;" id="formula-text">ΔPrice = Θ(₹0) + Δ(₹0) + ½Γ(₹0) + V(₹0) = ₹0</span>
                                            </div>
                                            <div style="display:flex; align-items:center; gap:6px;">
                                                <span style="color:#94a3b8;">Expected Move (1σ):</span>
                                                <strong style="color:#ffd54f;" id="attr-em-pts">±{_em_pts:.1f} pts</strong>
                                            </div>
                                        </div>
                                    </div>

                                    <!-- NORMALIZED SPOT SHIFT & TIME-TRAVEL FORECASTER -->
                                    <div class="card" style="padding:12px; border-left:4px solid #10b981; background:rgba(18,18,42,0.75);">
                                        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:10px; flex-wrap:wrap; gap:6px;">
                                            <div style="display:flex; align-items:center; gap:8px;">
                                                <span style="color:#10b981; font-size:12px; font-weight:900; letter-spacing:1px;">⚡ NORMALIZED SPOT SHIFT & TIME-TRAVEL FORECASTER</span>
                                                <span style="background:rgba(16,185,129,0.2); color:#10b981; font-size:10px; font-weight:700; padding:2px 8px; border-radius:12px; border:1px solid rgba(16,185,129,0.35);">Real-Time P&L & Retention Engine</span>
                                            </div>
                                            <div>
                                                <span style="color:#38bdf8; font-size:11px; font-weight:700; cursor:pointer; text-decoration:underline;" onclick="resetSimShocks()">Reset All Shocks</span>
                                            </div>
                                        </div>

                                        <!-- Sliders Grid -->
                                        <div style="display:grid; grid-template-columns: repeat(4, 1fr); gap:10px; margin-bottom:10px; background:rgba(10,10,24,0.65); padding:10px; border-radius:8px; border:1px solid rgba(255,255,255,0.06);">
                                            <!-- Spot Shift -->
                                            <div style="display:flex; flex-direction:column; gap:5px;">
                                                <div style="display:flex; justify-content:space-between; font-size:11px;">
                                                    <span style="color:#10b981; font-weight:800;">📈 Spot Shift (ΔS & Z):</span>
                                                    <span id="lbl-sim-spot" style="color:#10b981; font-weight:800;">0 pts (0.0σ)</span>
                                                </div>
                                                <input type="range" id="slider-sim-spot" min="-300" max="300" step="5" value="0" oninput="onThetaSimSliderChange()" style="width:100%; accent-color:#10b981; cursor:pointer;">
                                                <div style="display:flex; gap:3px; flex-wrap:wrap;">
                                                    <button type="button" class="th-quick-btn" onclick="setSimSpotPreset(-150)">-150</button>
                                                    <button type="button" class="th-quick-btn" onclick="setSimSpotPreset(-100)">-100</button>
                                                    <button type="button" class="th-quick-btn" onclick="setSimSpotPreset(-50)">-50</button>
                                                    <button type="button" class="th-quick-btn" onclick="setSimSpotPreset(0)">0 (ATM)</button>
                                                    <button type="button" class="th-quick-btn" onclick="setSimSpotPreset(50)">+50</button>
                                                    <button type="button" class="th-quick-btn" onclick="setSimSpotPreset(100)">+100</button>
                                                    <button type="button" class="th-quick-btn" onclick="setSimSpotPreset(150)">+150</button>
                                                </div>
                                            </div>

                                            <!-- Minutes -->
                                            <div style="display:flex; flex-direction:column; gap:5px;">
                                                <div style="display:flex; justify-content:space-between; font-size:11px;">
                                                    <span style="color:#38bdf8; font-weight:800;">⏱ Forward Time (Min):</span>
                                                    <span id="lbl-sim-min" style="color:#38bdf8; font-weight:800;">+0 min</span>
                                                </div>
                                                <input type="range" id="slider-sim-min" min="0" max="375" step="5" value="0" oninput="onThetaSimSliderChange()" style="width:100%; accent-color:#0284c7; cursor:pointer;">
                                                <div style="display:flex; gap:3px; flex-wrap:wrap;">
                                                    <button type="button" class="th-quick-btn" onclick="setSimTimePreset(0)">Now</button>
                                                    <button type="button" class="th-quick-btn" onclick="setSimTimePreset(15)">+15m</button>
                                                    <button type="button" class="th-quick-btn" onclick="setSimTimePreset(30)">+30m</button>
                                                    <button type="button" class="th-quick-btn" onclick="setSimTimePreset(60)">+60m</button>
                                                    <button type="button" class="th-quick-btn" onclick="setSimTimePreset(180)">+3h</button>
                                                    <button type="button" class="th-quick-btn" onclick="setSimTimePreset(375)">Close (3:30)</button>
                                                </div>
                                            </div>

                                            <!-- Days -->
                                            <div style="display:flex; flex-direction:column; gap:5px;">
                                                <div style="display:flex; justify-content:space-between; font-size:11px;">
                                                    <span style="color:#ffd54f; font-weight:800;">📅 Forward Days (Overnight):</span>
                                                    <span id="lbl-sim-days" style="color:#ffd54f; font-weight:800;">+0.0 days</span>
                                                </div>
                                                <input type="range" id="slider-sim-days" min="0" max="{max(_curr_dte_th, 1.0):.1f}" step="0.1" value="0" oninput="onThetaSimSliderChange()" style="width:100%; accent-color:#ffd54f; cursor:pointer;">
                                                <div style="display:flex; gap:3px; flex-wrap:wrap;">
                                                    <button type="button" class="th-quick-btn" onclick="setSimDaysPreset(0)">0d</button>
                                                    <button type="button" class="th-quick-btn" onclick="setSimDaysPreset(0.5)">+0.5d</button>
                                                    <button type="button" class="th-quick-btn" onclick="setSimDaysPreset(1.0)">+1.0d</button>
                                                    <button type="button" class="th-quick-btn" onclick="setSimDaysPreset({_curr_dte_th:.1f})">Till Expiry</button>
                                                </div>
                                            </div>

                                            <!-- IV Shift -->
                                            <div style="display:flex; flex-direction:column; gap:5px;">
                                                <div style="display:flex; justify-content:space-between; font-size:11px;">
                                                    <span style="color:#c084fc; font-weight:800;">🌪 IV Shift (Δσ):</span>
                                                    <span id="lbl-sim-iv" style="color:#c084fc; font-weight:800;">0.0%</span>
                                                </div>
                                                <input type="range" id="slider-sim-iv" min="-5.0" max="5.0" step="0.2" value="0" oninput="onThetaSimSliderChange()" style="width:100%; accent-color:#c084fc; cursor:pointer;">
                                                <div style="display:flex; gap:3px; flex-wrap:wrap;">
                                                    <button type="button" class="th-quick-btn" onclick="setSimIVPreset(-3.0)">-3% Crush</button>
                                                    <button type="button" class="th-quick-btn" onclick="setSimIVPreset(-1.0)">-1%</button>
                                                    <button type="button" class="th-quick-btn" onclick="setSimIVPreset(0)">0%</button>
                                                    <button type="button" class="th-quick-btn" onclick="setSimIVPreset(1.0)">+1%</button>
                                                    <button type="button" class="th-quick-btn" onclick="setSimIVPreset(3.0)">+3% Spike</button>
                                                </div>
                                            </div>
                                        </div>

                                        <!-- Outcome Strip -->
                                        <div style="display:grid; grid-template-columns: repeat(4, 1fr); gap:8px;">
                                            <div class="metric-box" style="border:1px solid rgba(2,132,199,0.35); background:rgba(2,132,199,0.08);">
                                                <div class="metric-label" style="color:#38bdf8;">Projected Straddle LTP</div>
                                                <div id="sim-res-price" style="font-size:16px; font-weight:800; color:{WHITE};">₹{_atm_strad_prem:.1f}</div>
                                                <div id="sim-res-price-sub" class="metric-sub">Base: ₹{_atm_strad_prem:.1f} (Δ: ₹0.0)</div>
                                            </div>
                                            <div class="metric-box" style="border:1px solid rgba(16,185,129,0.35); background:rgba(16,185,129,0.08);">
                                                <div class="metric-label" style="color:#10b981;">Seller P&L (1 Lot)</div>
                                                <div id="sim-res-seller-pnl" style="font-size:16px; font-weight:800; color:#10b981;">+₹0 (+0.0%)</div>
                                                <div id="sim-res-seller-sub" class="metric-sub">Theta decay captured</div>
                                            </div>
                                            <div class="metric-box" style="border:1px solid rgba(239,68,68,0.35); background:rgba(239,68,68,0.08);">
                                                <div class="metric-label" style="color:#ef5350;">Buyer P&L (1 Lot)</div>
                                                <div id="sim-res-buyer-pnl" style="font-size:16px; font-weight:800; color:#ef5350;">-₹0 (-0.0%)</div>
                                                <div id="sim-res-buyer-sub" class="metric-sub">Convexity vs decay bleed</div>
                                            </div>
                                            <div class="metric-box" id="sim-res-action-card" style="border:1px solid rgba(16,185,129,0.4); background:rgba(16,185,129,0.12);">
                                                <div class="metric-label" style="color:#10b981;">Position Retention Guide</div>
                                                <div id="sim-res-action-title" style="font-size:14px; font-weight:900; color:#10b981;">STAY IN POSITION</div>
                                                <div id="sim-res-action-desc" class="metric-sub" style="color:{WHITE};">Extrinsic decay active · Safe cushion</div>
                                            </div>
                                        </div>
                                    </div>

                                    <!-- 9-STEP NORMALIZED SCENARIO MATRIX (Gatheral 2006 · Merton 1973) -->
                                    <div class="card" style="padding:10px;">
                                        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
                                            <div style="color:{ACCENT}; font-size:11px; font-weight:800; letter-spacing:1px;">9-STEP RESEARCH-NORMALIZED SCENARIO MATRIX (GATHERAL 2006 · MERTON 1973)</div>
                                            <div style="font-size:10px; color:{MUTED};">Universal standard deviation scaling (Z = -2.0σ to +2.0σ) · Call, Put, Straddle Payoffs</div>
                                        </div>
                                        <div style="overflow-x:auto;">
                                            <table class="data-table" id="scenarios-table">
                                                <thead>
                                                    <tr style="position:sticky; top:0; background:{CARD_BG}; z-index:2;">
                                                        <th style="text-align:left;">Displacement (Z)</th>
                                                        <th>Spot Shift</th>
                                                        <th>Nifty Index</th>
                                                        <th style="color:#38bdf8;">Call LTP (₹)</th>
                                                        <th style="color:#ff7043;">Put LTP (₹)</th>
                                                        <th style="color:#ffd54f;">Straddle LTP (₹)</th>
                                                        <th style="color:#10b981;">Seller P&L (₹)</th>
                                                        <th style="color:#10b981;">Seller %</th>
                                                        <th style="color:#ef5350;">Buyer P&L (₹)</th>
                                                        <th style="text-align:left;">Dominant Greek Cause</th>
                                                        <th style="text-align:center;">Status / Advice</th>
                                                    </tr>
                                                </thead>
                                                <tbody>
                                                    {''.join(_scen_table_rows)}
                                                </tbody>
                                            </table>
                                        </div>
                                    </div>

                                    <!-- VISUALIZATION ROW 1 -->
                                    <div class="card" style="padding:10px;">
                                        <div style="min-height:350px; width:100%;" id="theta-chart-row1">
                                            {_plotly_th1}
                                        </div>
                                    </div>

                                    <!-- VISUALIZATION ROW 2 -->
                                    <div class="card" style="padding:10px;">
                                        <div style="min-height:350px; width:100%;" id="theta-chart-row2">
                                            {_plotly_th2}
                                        </div>
                                    </div>

                                    <!-- PER-STRIKE ACTIONABLE MATRIX TABLE -->
                                    <div class="card">
                                        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:10px;">
                                            <div style="color:{ACCENT}; font-size:12px; font-weight:800; letter-spacing:1.5px;">PER-STRIKE THETA DECAY & HARVEST MATRIX</div>
                                            <div style="font-size:11px; color:{MUTED};">Straddle LTP · Expiry Horizon · CE/PE Burn · Gamma · Daily Cushion · Trade Verdict & Advice</div>
                                        </div>
                                        <div id="theta-table-container" style="max-height:360px; overflow-y:auto; position:relative; scroll-behavior:smooth;">
                                            <table class="data-table" id="theta-decay-table">
                                                <thead>
                                                    <tr style="position:sticky; top:0; background:{CARD_BG}; z-index:2;">
                                                        <th style="text-align:left;">Strike</th>
                                                        <th>Dist</th>
                                                        <th style="color:#ffd54f;">Straddle LTP</th>
                                                        <th style="color:{YELLOW};">Expiry Horizon</th>
                                                        <th>CE LTP (IV)</th>
                                                        <th>CE θ/Day</th>
                                                        <th>PE LTP (IV)</th>
                                                        <th>PE θ/Day</th>
                                                        <th>Straddle θ/Day</th>
                                                        <th>1-Hr Burn</th>
                                                        <th>Gamma (Γ)</th>
                                                        <th>Daily Cushion</th>
                                                        <th>Decay Yield</th>
                                                        <th style="text-align:center;">Edge Verdict</th>
                                                        <th style="text-align:center;">Position Advice</th>
                                                    </tr>
                                                </thead>
                                                <tbody>
                                                    {''.join(_theta_table_rows)}
                                                </tbody>
                                            </table>
                                        </div>
                                    </div>
                                </div>'''

                    except Exception as _th_e:
                        theta_tab_html = f'<div class="card"><p style="color:#ff4444;">Theta Decay module error: {str(_th_e)}</p></div>'


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
                        _r_cone = _get_cfg("risk_free_rate", 0.051274)
                        mean_price = spot * np.exp((_r_cone - 0.5 * sigma_ann**2) * cone_T)
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
                            height=480, width=1380, paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)',
                            font=dict(color=WHITE, family='Inter, sans-serif', size=11),
                            legend=dict(bgcolor='rgba(30,30,50,0.8)', font=dict(size=10), x=0.01, y=0.99),
                            margin=dict(l=50, r=20, t=50, b=30), hovermode='closest',
                            scene=dict(
                                xaxis=dict(title='Price', backgroundcolor='rgba(0,0,0,0)', gridcolor='#333'),
                                yaxis=dict(title='Days', backgroundcolor='rgba(0,0,0,0)', gridcolor='#333'),
                                zaxis=dict(title='PDF', backgroundcolor='rgba(0,0,0,0)', gridcolor='#333'),
                                camera=dict(eye=dict(x=1.5, y=-1.8, z=0.8)), bgcolor='rgba(0,0,0,0)'),
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
                                
                                _d_high = float(momentum_data.get('day_high', 0))
                                _d_low  = float(momentum_data.get('day_low', 0))
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

                    # ── Strategy Engine tab removed from dashboard ──
                    # StrategyManager / trackStrategy() / /api/track_strategy endpoint
                    # remain active for strategy_builder.html standalone page.

                    # ── MARKET MAKER POSITIONING TAB (TRI-MODEL COMPARISON) ──
                    mm_tab_html = '<div class="card"><p style="color:#888;">Loading dealer positioning...</p></div>'
                    try:
                        if not df_chain.empty and spot > 0:
                            _lot_mm = _get_cfg("nifty_lot_size", 65)
                            _r_mm = _get_cfg("risk_free_rate", 0.051274)
                            _T_mm = self.analytics.get_time_to_expiry(near_exp) or (1 / 365)
                            _df_mm = df_chain.copy()

                            # Compute gamma + delta + vega via BSM per leg
                            def _bsm_greeks(row):
                                try:
                                    calc_iv = self._ensure_iv(row['iv'], row['price'], row['strike'], _T_mm, row['type'])
                                    sigma = max(calc_iv / 100.0, 0.01)
                                    K = row['strike']; S = spot; r = _r_mm
                                    if K <= 0 or S <= 0:
                                        return pd.Series({'gamma': 0.0, 'delta': 0.0, 'vega': 0.0})
                                    sqT = np.sqrt(max(_T_mm, 1e-6))
                                    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * _T_mm) / (sigma * sqT)
                                    d2 = d1 - sigma * sqT
                                    from scipy.stats import norm
                                    phi = np.exp(-0.5 * d1**2) / np.sqrt(2 * np.pi)
                                    gamma = phi / (S * sigma * sqT)
                                    delta = norm.cdf(d1) if row['type'] == 'CE' else norm.cdf(d1) - 1.0
                                    vega  = S * phi * sqT / 100.0  # vega per 1% IV move
                                    return pd.Series({'gamma': gamma, 'delta': delta, 'vega': vega})
                                except:
                                    return pd.Series({'gamma': 0.0, 'delta': 0.0, 'vega': 0.0})

                            _greeks = _df_mm.apply(_bsm_greeks, axis=1)
                            _df_mm[['gamma_mm', 'delta_mm', 'vega_mm']] = _greeks

                            # Common base scaling: Spot^2 * 0.01
                            _scaling = spot * (spot * 0.01)

                            # ── 1. MODEL A: STANDARD (Wall Street Prior: Long CE +1, Short PE -1) ──
                            # CE: Dealer buys Call (+1). Delta > 0 -> DEX > 0. Gamma > 0 -> GEX > 0
                            # PE: Dealer sells Put (-1). Delta < 0 -> DEX = (-1)*Delta = +|Delta| > 0. Gamma > 0 -> GEX = (-1)*Gamma < 0
                            _df_mm['dex_std'] = _df_mm.apply(
                                lambda r: r['delta_mm'] * r['oi'] * _lot_mm if r['type'] == 'CE'
                                else -r['delta_mm'] * r['oi'] * _lot_mm, axis=1)
                            _df_mm['gex_std'] = _df_mm.apply(
                                lambda r: r['gamma_mm'] * r['oi'] * _lot_mm * _scaling if r['type'] == 'CE'
                                else -r['gamma_mm'] * r['oi'] * _lot_mm * _scaling, axis=1)
                            _df_mm['gex_sh_std'] = _df_mm.apply(
                                lambda r: r['gamma_mm'] * r['oi'] * _lot_mm * (spot * 0.01) if r['type'] == 'CE'
                                else -r['gamma_mm'] * r['oi'] * _lot_mm * (spot * 0.01), axis=1)

                            # ── 2. MODEL B: INVERTED (Retail Speculative: Short CE -1, Long PE +1) ──
                            # Retail buys calls / institutions sell puts -> Dealer is Short CE (-1), Long PE (+1)
                            _df_mm['dex_inv'] = -_df_mm['dex_std']
                            _df_mm['gex_inv'] = -_df_mm['gex_std']
                            _df_mm['gex_sh_inv'] = -_df_mm['gex_sh_std']

                            # ── 3. MODEL C: DUAL-SHORT (NSE Option Writers / Straddles: Short CE -1, Short PE -1) ──
                            # CE: Dealer sells Call (-1). Delta > 0 -> DEX = -Delta < 0. Gamma > 0 -> GEX = -Gamma < 0
                            # PE: Dealer sells Put (-1).  Delta < 0 -> DEX = -Delta = +|Delta| > 0. Gamma > 0 -> GEX = -Gamma < 0
                            _df_mm['dex_dual'] = _df_mm.apply(
                                lambda r: -r['delta_mm'] * r['oi'] * _lot_mm if r['type'] == 'CE'
                                else -r['delta_mm'] * r['oi'] * _lot_mm, axis=1)
                            _df_mm['gex_dual'] = _df_mm.apply(
                                lambda r: -r['gamma_mm'] * r['oi'] * _lot_mm * _scaling, axis=1)
                            _df_mm['gex_sh_dual'] = _df_mm.apply(
                                lambda r: -r['gamma_mm'] * r['oi'] * _lot_mm * (spot * 0.01), axis=1)

                            # ── AGGREGATES & METRICS ──
                            # Model A (Standard)
                            _net_dex_std = _df_mm['dex_std'].sum()
                            _net_gex_std = _df_mm['gex_std'].sum()
                            _hedge_std_val = -_df_mm['gex_sh_std'].sum()
                            _std_regime = 'LONG GAMMA (Pinning ↔)' if _net_gex_std > 0 else 'SHORT GAMMA (Trending ↕)'
                            _std_regime_c = GREEN if _net_gex_std > 0 else RED
                            _std_dex_bias = 'BUY SPOT' if _net_dex_std > 0 else 'SELL SPOT'
                            _std_dex_c = GREEN if _net_dex_std > 0 else RED
                            _hedge_std_action = f"SELL {abs(_hedge_std_val)/1000:.1f}K sh (dampens move)" if _hedge_std_val < 0 else f"BUY {abs(_hedge_std_val)/1000:.1f}K sh (chases move)"

                            # Zero Gamma Flip Strike for Standard
                            _sk_std = _df_mm.groupby('strike')['gex_std'].sum().sort_index()
                            _sk_strikes = [float(k) for k in _sk_std.index]
                            _sk_vals = [float(v) for v in _sk_std.values]
                            _flip_std = None
                            for _i in range(len(_sk_vals) - 1):
                                if _sk_vals[_i] * _sk_vals[_i + 1] < 0:
                                    _flip_val = _sk_strikes[_i] if abs(_sk_vals[_i]) < abs(_sk_vals[_i + 1]) else _sk_strikes[_i + 1]
                                    _flip_std = round(_flip_val)
                                    break
                            _flip_std_txt = f"{_flip_std} ({abs(spot - _flip_std):.0f} pts {'above' if spot > _flip_std else 'below'})" if _flip_std else "No Flip Point"

                            # Model B (Inverted)
                            _net_dex_inv = _df_mm['dex_inv'].sum()
                            _net_gex_inv = _df_mm['gex_inv'].sum()
                            _hedge_inv_val = -_df_mm['gex_sh_inv'].sum()
                            _inv_regime = 'PUT SUPPORT (Pinning ↔)' if _net_gex_inv > 0 else 'CALL SQUEEZE RISK (Breakout ↑)'
                            _inv_regime_c = GREEN if _net_gex_inv > 0 else RED
                            _inv_dex_bias = 'BUY SPOT' if _net_dex_inv > 0 else 'SELL SPOT'
                            _inv_dex_c = GREEN if _net_dex_inv > 0 else RED
                            _hedge_inv_action = f"BUY {abs(_hedge_inv_val)/1000:.1f}K sh (amplifies rally)" if _hedge_inv_val > 0 else f"SELL {abs(_hedge_inv_val)/1000:.1f}K sh (chases drop)"
                            _flip_inv_txt = f"{_flip_std} (Inverted Flip Level)" if _flip_std else "No Flip Point"

                            # Model C (Dual-Short / Option Writers)
                            _net_dex_dual = _df_mm['dex_dual'].sum()
                            _net_gex_dual = _df_mm['gex_dual'].sum()
                            _hedge_dual_val = -_df_mm['gex_sh_dual'].sum()
                            _dual_regime = 'SHORT VOLATILITY (Straddle Acceleration ⚡)'
                            _dual_regime_c = RED
                            _dual_dex_bias = 'PUT HEAVY (Downside Drag)' if _net_dex_dual > 0 else 'CALL HEAVY (Upside Drag)'
                            _dual_dex_c = GREEN if _net_dex_dual > 0 else RED
                            _hedge_dual_action = f"BUY {abs(_hedge_dual_val)/1000:.1f}K sh (chases breakout)" if _hedge_dual_val > 0 else f"SELL {abs(_hedge_dual_val)/1000:.1f}K sh (chases breakdown)"
                            
                            # Max Gamma Valley Strike for Dual-Short (Straddle Center)
                            _sk_dual = _df_mm.groupby('strike')['gex_dual'].sum()
                            _max_valley_strike = round(float(str(_sk_dual.idxmin()))) if not _sk_dual.empty else 0
                            _dist_valley = abs(spot - _max_valley_strike) if _max_valley_strike else 0

                            # ── Consensus & Synthesis ──
                            _consensus_title = "MARKET MAKER DYNAMICS CONSENSUS"
                            _hedges = [_hedge_std_val, _hedge_inv_val, _hedge_dual_val]
                            _sell_hedges = sum(1 for h in _hedges if h < 0)
                            if _sell_hedges >= 2:
                                _consensus_flow = "DAMPENING FLOW (Dealers forced to SELL spot into 1% rally)"
                                _consensus_flow_c = GREEN
                            else:
                                _consensus_flow = "ACCELERATING FLOW (Dealers forced to BUY spot into 1% rally)"
                                _consensus_flow_c = RED

                            if _net_gex_std > 0:
                                _consensus_desc = f"<b>Mean Reverting Pinning:</b> Standard Wall St model indicates net positive dealer gamma ({_net_gex_std/1e6:+.1f}M), which dampens volatility. However, if spot moves >{_dist_valley:.0f} pts away from straddle center ({_max_valley_strike}), Dual-Short option writing dynamics take over and amplify momentum."
                            else:
                                _consensus_desc = f"<b>Slippery Regime Alert:</b> Both Standard and Dual-Short models indicate negative dealer gamma. Market makers are net short volatility and will be forced to chase momentum in the direction of the break."

                            # ── Per-strike Summary Table ──
                            _ce_oi_map = _df_mm[_df_mm['type'] == 'CE'].set_index('strike')['oi']
                            _pe_oi_map = _df_mm[_df_mm['type'] == 'PE'].set_index('strike')['oi']

                            _by_strike = _df_mm.groupby('strike').agg(
                                gex_std=('gex_std', 'sum'),
                                gex_inv=('gex_inv', 'sum'),
                                gex_dual=('gex_dual', 'sum'),
                                dex_dual=('dex_dual', 'sum')
                            ).reset_index().sort_values('strike', ascending=False)

                            _by_strike['oi_ce'] = _by_strike['strike'].map(_ce_oi_map).fillna(0)
                            _by_strike['oi_pe'] = _by_strike['strike'].map(_pe_oi_map).fillna(0)

                            _mm_rows = ''
                            for _, _mr in _by_strike.iterrows():
                                _msk = int(_mr['strike'])
                                _g_std = _mr['gex_std'] / 1e6
                                _g_inv = _mr['gex_inv'] / 1e6
                                _g_dual = _mr['gex_dual'] / 1e6
                                _d_dual = _mr['dex_dual'] / 1000
                                _ce_k = _mr['oi_ce'] / 1000
                                _pe_k = _mr['oi_pe'] / 1000
                                _dist = _msk - spot

                                _c_std = GREEN if _g_std > 0 else RED
                                _c_inv = GREEN if _g_inv > 0 else RED
                                _c_dual = RED
                                _c_d_dual = GREEN if _d_dual > 0 else RED

                                _is_atm_mm = abs(_dist) < 75
                                _mm_rows += (
                                    f'<tr style="background:{"rgba(79,195,247,0.08)" if _is_atm_mm else "transparent"};border-bottom:1px solid #1a1a2e;">'
                                    f'<td style="padding:6px 10px;font-weight:700;color:{ACCENT if _is_atm_mm else WHITE};">{_msk}{" ◄ ATM" if _is_atm_mm else ""}</td>'
                                    f'<td style="padding:6px 10px;text-align:right;color:{MUTED};">{_dist:+.0f}</td>'
                                    f'<td style="padding:6px 10px;text-align:right;color:{WHITE};">{_ce_k:.0f}K / {_pe_k:.0f}K</td>'
                                    f'<td style="padding:6px 10px;text-align:right;font-weight:600;color:{_c_std};">{"+" if _g_std>0 else ""}{_g_std:.1f}M</td>'
                                    f'<td style="padding:6px 10px;text-align:right;font-weight:600;color:{_c_inv};">{"+" if _g_inv>0 else ""}{_g_inv:.1f}M</td>'
                                    f'<td style="padding:6px 10px;text-align:right;font-weight:600;color:{_c_dual};">{_g_dual:.1f}M</td>'
                                    f'<td style="padding:6px 10px;text-align:right;font-weight:600;color:{_c_d_dual};">{"+" if _d_dual>0 else ""}{_d_dual:.1f}K</td>'
                                    f'</tr>'
                                )

                            mm_tab_html = f'''
                            <!-- Market Maker Gamma Pinning & Order Flow Absorption Terminal -->
                            <div id="gamma-explosion-root" class="gamma-explosion-wrap">
                                <!-- 1. Interactive Real-Time Candlesticks & GEX Bands Chart (Plotly) -->
                                <div class="ge-chart-card">
                                    <div class="ge-card-header" style="margin-bottom:8px;">
                                        <div class="ge-card-title">
                                            <span style="letter-spacing:1px; font-weight:700;">LIVE NIFTY CANDLESTICKS &amp; INSTITUTIONAL GEX BANDS</span>
                                        </div>
                                        <div id="ge-chart-legend" style="display:flex; gap:12px; font-size:11px; font-family:var(--font-mono);">
                                            <span style="color:#00e676;">🟩 Call Wall</span>
                                            <span style="color:#ff3366;">🟥 Put Wall</span>
                                            <span style="color:#00e5ff;">🟦 Gamma Flip</span>
                                            <span style="color:#ffd54f;">🟨 Pin Corridor</span>
                                        </div>
                                    </div>
                                    <div id="ge-interactive-chart" style="width:100%; height:380px;"></div>
                                </div>

                                <!-- 2. Dual Spotlight Grid: Reel 1 & Reel 2 Models -->
                                <div class="ge-grid">
                                    <!-- Reel 1 Spotlight: Dominant Pin & Duration Timer (quantedoptions) -->
                                    <div id="ge-reel1-spotlight" class="ge-card" style="border-left: 3px solid var(--accent-amber);">
                                        <div class="ge-card-header">
                                            <div class="ge-card-title">
                                                <span style="color:var(--accent-amber); font-weight:800;">⚡ REEL 1: SESSION DOMINANT GAMMA PIN</span>
                                            </div>
                                            <span id="ge-reel1-status-pill" class="ge-pin-badge low">HOLDING CEILING</span>
                                        </div>
                                        <div style="display:grid; grid-template-columns: repeat(auto-fit, minmax(115px, 1fr)); gap:10px; margin-top:6px;">
                                            <div class="ge-stat-box"><span class="lbl">PINNED STRIKE</span><span id="ge-r1-strike" class="val" style="color:#ffd54f;">--</span><span class="sub">MAGNETIC ANCHOR</span></div>
                                            <div class="ge-stat-box"><span class="lbl">PIN DURATION</span><span id="ge-r1-duration" class="val" style="color:#00e5ff;">--</span><span class="sub">SESSION CLOCK</span></div>
                                            <div class="ge-stat-box"><span class="lbl">MM GAMMA</span><span id="ge-r1-gamma" class="val" style="color:#00e676;">--</span><span class="sub">STABILIZING FLOW</span></div>
                                            <div class="ge-stat-box"><span class="lbl">PULL FORCE</span><span id="ge-r1-pull" class="val">--</span><span class="sub">PROXIMITY SCORE</span></div>
                                            <div class="ge-stat-box"><span class="lbl">CORRIDOR</span><span id="ge-r1-corridor" class="val">--</span><span class="sub">±50 PT RANGE</span></div>
                                        </div>
                                        <div id="ge-r1-unwind-alert" style="margin-top:10px; padding:10px 14px; background:rgba(0, 229, 255, 0.05); border-radius:6px; border:1px solid rgba(0, 229, 255, 0.25); font-size:12px; color:#ddd;">
                                            Scanning session dominant pin...
                                        </div>
                                    </div>

                                    <!-- Reel 2 Spotlight: Retest & 100-Point Squeeze Setup (aleksrosme) -->
                                    <div id="ge-reel2-spotlight" class="ge-card" style="border-left: 3px solid var(--accent-cyan);">
                                        <div class="ge-card-header">
                                            <div class="ge-card-title">
                                                <span style="color:var(--accent-cyan); font-weight:800;">⚡ REEL 2: GEX RETEST &amp; 100-PT SQUEEZE</span>
                                            </div>
                                            <div id="ge-radar-status" class="ge-status-pill monitoring">MONITORING</div>
                                        </div>
                                        <div id="ge-r2-headline" style="font-size:13px; font-weight:700; color:#fff; min-height:36px;">
                                            Monitoring GEX Barrier Retest...
                                        </div>
                                        <div class="ge-target-ladder" style="margin-top:8px;">
                                            <div class="ge-target-box trigger"><span class="lbl">TRIGGER ENTRY</span><span id="ge-target-trigger" class="val">--</span><span class="sub">BREAKOUT</span></div>
                                            <div class="ge-target-box" style="border-color:rgba(255,51,102,0.3);"><span class="lbl" style="color:#ff3366;">STOP LOSS</span><span id="ge-target-stop" class="val" style="color:#ff3366;">--</span><span class="sub">15-20 PT RISK</span></div>
                                            <div class="ge-target-box t1"><span class="lbl">PRIMARY T1</span><span id="ge-target-t1" class="val">--</span><span class="sub">+50 PTS</span></div>
                                            <div class="ge-target-box t2"><span class="lbl">EXPLOSION T2</span><span id="ge-target-t2" class="val">--</span><span class="sub">+100 PTS</span></div>
                                        </div>
                                        <div style="display:flex; justify-content:space-between; align-items:center; margin-top:10px; padding:8px 12px; background:rgba(0,0,0,0.3); border-radius:6px;">
                                            <div style="font-size:12px; color:#ccc;">R:R Expectancy: <span id="ge-target-rr" class="num-mono" style="color:#00e676; font-weight:800; font-size:14px;">--</span></div>
                                            <div id="ge-hedge-flow" class="num-mono" style="font-size:11px;"></div>
                                        </div>
                                    </div>
                                </div>

                                <!-- 3. Strike GEX Ladder & Pin Corridor -->
                                <div class="ge-card">
                                    <div class="ge-card-header">
                                        <div class="ge-card-title">
                                            <span style="font-weight:700;">STRIKE GEX LADDER &amp; INSTITUTIONAL DISTRIBUTION</span>
                                        </div>
                                        <span class="num-mono" style="font-size:11px; color:{MUTED};">Signed Dealer Gamma (₹ Cr)</span>
                                    </div>
                                    <div id="ge-ladder-container" class="ge-ladder-wrap">
                                        <div style="color:{MUTED}; font-size:12px; padding:10px;">Loading strike GEX ladder...</div>
                                    </div>
                                </div>
                            </div>

                            <div style="margin-top: 24px; margin-bottom: 12px; border-top: 1px solid rgba(255,255,255,0.08); padding-top: 16px;">
                                <div style="font-size: 13px; font-weight: 800; letter-spacing: 1.2px; color: {ACCENT}; text-transform: uppercase;">
                                    TRI-MODEL DEALER POSITIONING COMPARISON
                                </div>
                            </div>

                            <!-- Tri-Model Comparative Cards -->
                            <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px;margin-bottom:14px;">
                                <!-- Model A: Standard -->
                                <div class="card" style="border-top:4px solid {_std_regime_c};position:relative;">
                                    <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:8px;">
                                        <div style="color:{ACCENT};font-size:12px;font-weight:800;letter-spacing:1.5px;">1. STANDARD (WALL ST)</div>
                                        <span style="background:rgba(79,195,247,0.15);color:{ACCENT};font-size:10px;font-weight:700;padding:2px 6px;border-radius:4px;">SPX OVERWRITE</span>
                                    </div>
                                    <div style="font-size:11px;color:{MUTED};margin-bottom:12px;">Dealers Long Calls (+1) | Short Puts (-1)</div>
                                    
                                    <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-bottom:10px;">
                                        <div class="metric-box">
                                            <div class="metric-label">Dealer DEX</div>
                                            <div style="font-size:15px;font-weight:800;color:{_std_dex_c};">{_std_dex_bias}</div>
                                            <div class="metric-sub">{_net_dex_std/1000:+.1f}K sh</div>
                                        </div>
                                        <div class="metric-box">
                                            <div class="metric-label">Dealer GEX</div>
                                            <div style="font-size:15px;font-weight:800;color:{_std_regime_c};">{_net_gex_std/1e6:+.1f}M</div>
                                            <div class="metric-sub">{_std_regime[:10]}</div>
                                        </div>
                                    </div>
                                    <div style="background:#0c0c1e;padding:8px 10px;border-radius:6px;font-size:11px;border:1px solid #22223a;">
                                        <div style="display:flex;justify-content:space-between;margin-bottom:4px;">
                                            <span style="color:{MUTED};">Flip Strike:</span>
                                            <strong style="color:{WHITE};">{_flip_std_txt}</strong>
                                        </div>
                                        <div style="display:flex;justify-content:space-between;">
                                            <span style="color:{MUTED};">+1% Hedge Flow:</span>
                                            <strong style="color:{GREEN if 'SELL' in _hedge_std_action else RED};">{_hedge_std_action}</strong>
                                        </div>
                                    </div>
                                </div>

                                <!-- Model B: Inverted -->
                                <div class="card" style="border-top:4px solid {_inv_regime_c};position:relative;">
                                    <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:8px;">
                                        <div style="color:{ACCENT};font-size:12px;font-weight:800;letter-spacing:1.5px;">2. INVERTED (SPECULATIVE)</div>
                                        <span style="background:rgba(255,214,0,0.15);color:{YELLOW};font-size:10px;font-weight:700;padding:2px 6px;border-radius:4px;">RETAIL MANIA</span>
                                    </div>
                                    <div style="font-size:11px;color:{MUTED};margin-bottom:12px;">Dealers Short Calls (-1) | Long Puts (+1)</div>
                                    
                                    <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-bottom:10px;">
                                        <div class="metric-box">
                                            <div class="metric-label">Dealer DEX</div>
                                            <div style="font-size:15px;font-weight:800;color:{_inv_dex_c};">{_inv_dex_bias}</div>
                                            <div class="metric-sub">{_net_dex_inv/1000:+.1f}K sh</div>
                                        </div>
                                        <div class="metric-box">
                                            <div class="metric-label">Dealer GEX</div>
                                            <div style="font-size:15px;font-weight:800;color:{_inv_regime_c};">{_net_gex_inv/1e6:+.1f}M</div>
                                            <div class="metric-sub">{_inv_regime[:12]}</div>
                                        </div>
                                    </div>
                                    <div style="background:#0c0c1e;padding:8px 10px;border-radius:6px;font-size:11px;border:1px solid #22223a;">
                                        <div style="display:flex;justify-content:space-between;margin-bottom:4px;">
                                            <span style="color:{MUTED};">Flip Strike:</span>
                                            <strong style="color:{WHITE};">{_flip_inv_txt}</strong>
                                        </div>
                                        <div style="display:flex;justify-content:space-between;">
                                            <span style="color:{MUTED};">+1% Hedge Flow:</span>
                                            <strong style="color:{RED if 'BUY' in _hedge_inv_action else GREEN};">{_hedge_inv_action}</strong>
                                        </div>
                                    </div>
                                </div>

                                <!-- Model C: Dual-Short (NSE Reality) -->
                                <div class="card" style="border-top:4px solid {_dual_regime_c};position:relative;">
                                    <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:8px;">
                                        <div style="color:{ACCENT};font-size:12px;font-weight:800;letter-spacing:1.5px;">3. DUAL-SHORT (NSE WRITERS)</div>
                                        <span style="background:rgba(255,61,0,0.15);color:{RED};font-size:10px;font-weight:700;padding:2px 6px;border-radius:4px;">THETA HARVEST</span>
                                    </div>
                                    <div style="font-size:11px;color:{MUTED};margin-bottom:12px;">Dealers Short Calls (-1) & Short Puts (-1)</div>
                                    
                                    <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-bottom:10px;">
                                        <div class="metric-box">
                                            <div class="metric-label">Direction Bias</div>
                                            <div style="font-size:15px;font-weight:800;color:{_dual_dex_c};">{_dual_dex_bias}</div>
                                            <div class="metric-sub">{_net_dex_dual/1000:+.1f}K net</div>
                                        </div>
                                        <div class="metric-box">
                                            <div class="metric-label">Dealer GEX</div>
                                            <div style="font-size:15px;font-weight:800;color:{_dual_regime_c};">{_net_gex_dual/1e6:.1f}M</div>
                                            <div class="metric-sub">SHORT VOLATILITY</div>
                                        </div>
                                    </div>
                                    <div style="background:#0c0c1e;padding:8px 10px;border-radius:6px;font-size:11px;border:1px solid #22223a;">
                                        <div style="display:flex;justify-content:space-between;margin-bottom:4px;">
                                            <span style="color:{MUTED};">Straddle Center:</span>
                                            <strong style="color:{WHITE};">{_max_valley_strike} ({_dist_valley:.0f} pts away)</strong>
                                        </div>
                                        <div style="display:flex;justify-content:space-between;">
                                            <span style="color:{MUTED};">+1% Hedge Flow:</span>
                                            <strong style="color:{RED};">{_hedge_dual_action}</strong>
                                        </div>
                                    </div>
                                </div>
                            </div>

                            <!-- Consensus & Strategy Implication Banner -->
                            <div class="card" style="margin-bottom:14px;border-left:4px solid {ACCENT};background:#0e1022;">
                                <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;">
                                    <span style="color:{ACCENT};font-size:12px;font-weight:800;letter-spacing:1px;">{_consensus_title}</span>
                                    <span style="font-size:11px;font-weight:700;color:{_consensus_flow_c};">{_consensus_flow}</span>
                                </div>
                                <div style="font-size:12px;color:#ccc;line-height:1.5;">{_consensus_desc}</div>
                            </div>

                            <!-- Per-Strike Comparative Table -->
                            <div class="card">
                                <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;">
                                    <div style="color:{ACCENT};font-size:12px;font-weight:800;letter-spacing:1.5px;">PER-STRIKE TRI-MODEL EXPOSURE COMPARISON</div>
                                    <div style="font-size:11px;color:{MUTED};">CE/PE OI · Standard GEX · Inverted GEX · Dual-Short GEX & DEX</div>
                                </div>
                                <div style="max-height:380px;overflow-y:auto;">
                                    <table style="width:100%;border-collapse:collapse;font-size:11px;">
                                        <thead>
                                            <tr style="color:{MUTED};border-bottom:1px solid #2a2a4a;position:sticky;top:0;background:{CARD_BG};z-index:2;">
                                                <th style="padding:6px 10px;text-align:left;">Strike</th>
                                                <th style="padding:6px 10px;text-align:right;">Dist</th>
                                                <th style="padding:6px 10px;text-align:right;">CE / PE OI</th>
                                                <th style="padding:6px 10px;text-align:right;">Std GEX (M)</th>
                                                <th style="padding:6px 10px;text-align:right;">Inv GEX (M)</th>
                                                <th style="padding:6px 10px;text-align:right;">Dual GEX (M)</th>
                                                <th style="padding:6px 10px;text-align:right;">Dual DEX (K)</th>
                                            </tr>
                                        </thead>
                                        <tbody>{_mm_rows}</tbody>
                                    </table>
                                </div>
                            </div>'''
                    except Exception as _mm_e:
                        mm_tab_html = f'<div class="card"><p style="color:#ff4444;">Dealer positioning error: {str(_mm_e)}</p></div>'

                    # ── REGIME TAB HTML (SIMPLIFIED ACTIONABLE VOLATILITY PLAYBOOK) ──
                    if regime_snapshot:
                        _vrp_val = regime_snapshot['vrp']['iv_rv']
                        if _vrp_val > 1.5:
                            _vrp_badge = "OPTIONS EXPENSIVE · SELLERS FAVORED"
                            _vrp_badge_col = GREEN
                            _vrp_sub = "Implied Vol is higher than actual market movement. Theta harvesting & credit spreads favored."
                            _rec_trades = "Delta-Neutral Short Straddles, Iron Condors, OTM Credit Spreads"
                            _rec_risk = "Favorable decay backdrop; hedge delta if spot breaches Call/Put walls."
                        elif _vrp_val < -1.5:
                            _vrp_badge = "OPTIONS CHEAP · BUYERS FAVORED"
                            _vrp_badge_col = RED
                            _vrp_sub = "Implied Vol is cheaper than actual price movement. Directional & breakout setups favored."
                            _rec_trades = "Long Straddles, Directional Debit Spreads, Gamma Squeeze breakout calls/puts"
                            _rec_risk = "Avoid naked option selling; volatility expansion risk is elevated."
                        else:
                            _vrp_badge = "FAIR VALUE VOLATILITY · BALANCED"
                            _vrp_badge_col = YELLOW
                            _vrp_sub = "IV closely tracks Realized Vol. Play selective tactical setups with disciplined stops."
                            _rec_trades = "Calendar Spreads, Ratio Spreads, Defined-Risk Rangebound plays"
                            _rec_risk = "Monitor dealer Gamma Flip strike for regime inflection."

                        _rv_cons = regime_snapshot['rv']['consensus']
                        _rv_intra = regime_snapshot['rv'].get('intraday', 0.0)
                        _rv_5d = regime_snapshot['rv'].get('5d', 0.0)
                        _rv_20d = regime_snapshot['rv'].get('20d', 0.0)
                        _rv_60d = regime_snapshot['rv'].get('60d', 0.0)
                        _hv_20d = regime_snapshot['hv']['20d']
                        _hv_pctile = regime_snapshot['hv']['percentile']
                        _rv_trend = regime_snapshot['rv']['trend']
                        _rv_trend_col = GREEN if 'COMPRESS' in _rv_trend.upper() or 'FALL' in _rv_trend.upper() else RED if 'EXPAND' in _rv_trend.upper() or 'RIS' in _rv_trend.upper() else YELLOW

                        regime_tab_html = f'''
                        <div style="display:flex; flex-direction:column; gap:14px;">
                            <!-- Institutional Gamma Explosion Quick-Launch Banner -->
                            <div class="ge-quick-banner" onclick="switchTab('mm')">
                                <div style="display:flex; align-items:center; gap:14px;">
                                    <span style="display:inline-flex; width:10px; height:10px; border-radius:50%; background:#00f0ff; box-shadow:0 0 10px #00f0ff; animation:pulseBadgeCyan 1.5s infinite;"></span>
                                    <div>
                                        <div style="font-weight:800; font-size:13px; color:#00f0ff; letter-spacing:0.5px; display:flex; align-items:center; gap:8px;">
                                            <span>⚡ NEW: MARKET MAKER GAMMA EXPLOSION & PINNING TERMINAL</span>
                                            <span style="font-size:10px; background:rgba(0,240,255,0.25); color:#00f0ff; padding:2px 8px; border-radius:4px; font-weight:700;">LIVE MODEL</span>
                                        </div>
                                        <div style="font-size:12px; color:#cbd5e1; margin-top:3px;">
                                            Real-time Dealer Gamma Pins, Duration Timer (4h+), GEX Retest Absorption & Directional Squeeze Targets.
                                        </div>
                                    </div>
                                </div>
                                <div style="display:flex; align-items:center; gap:6px; background:#00f0ff; color:#060812; font-weight:800; font-size:11px; padding:8px 16px; border-radius:6px; letter-spacing:0.5px; text-transform:uppercase;">
                                    LAUNCH TERMINAL &rarr;
                                </div>
                            </div>

                            <!-- Executive Market Regime Banner -->
                            <div class="card" style="border-left: 4px solid {ACCENT}; background:linear-gradient(135deg, rgba(18,18,42,0.95), rgba(10,14,28,0.95)); padding:16px;">
                                <div style="display:flex; justify-content:space-between; align-items:flex-start; flex-wrap:wrap; gap:10px; margin-bottom:8px;">
                                    <div>
                                        <div style="font-size:11px; font-weight:800; color:{MUTED}; text-transform:uppercase; letter-spacing:1px; margin-bottom:3px;">ACTIVE MARKET REGIME</div>
                                        <h2 style="color:{ACCENT}; font-size:22px; font-weight:900; margin:0 0 4px 0;">{regime_snapshot['regime']['name']}</h2>
                                    </div>
                                    <div style="display:flex; gap:8px; align-items:center;">
                                        <span style="background:rgba(255,255,255,0.06); color:{WHITE}; font-size:11px; font-weight:700; padding:4px 10px; border-radius:6px; border:1px solid #2a2a4a;">BIAS: <strong style="color:{ACCENT};">{regime_snapshot['regime']['bias']}</strong></span>
                                        <span style="background:{GREEN}18; color:{GREEN}; font-size:11px; font-weight:800; padding:4px 10px; border-radius:6px; border:1px solid {GREEN}44;">VOL: {regime_snapshot['regime']['vol_action']}</span>
                                    </div>
                                </div>
                                <p style="color:#cbd5e1; font-size:13px; line-height:1.5; margin:0 0 10px 0;">{regime_snapshot['regime']['description']}</p>
                            </div>

                            <!-- 3 Core Quantitative Metrics Grid -->
                            <div style="display:grid; grid-template-columns: repeat(3, 1fr); gap:12px;">
                                <div class="metric-box" style="padding:14px; text-align:left; background:rgba(18,18,42,0.85); border-top:3px solid {ACCENT};">
                                    <div class="metric-label" style="font-size:11px; color:{MUTED};">CONSENSUS REALIZED VOL</div>
                                    <div style="font-size:26px; font-weight:900; color:{WHITE}; margin:4px 0; font-family:'JetBrains Mono', monospace;">{_rv_cons:.2f}%</div>
                                    <div class="metric-sub" style="color:{_rv_trend_col}; font-weight:700;">Trend: {_rv_trend}</div>
                                </div>
                                <div class="metric-box" style="padding:14px; text-align:left; background:rgba(18,18,42,0.85); border-top:3px solid {_vrp_badge_col};">
                                    <div class="metric-label" style="font-size:11px; color:{MUTED};">VOLATILITY RISK PREMIUM (VRP)</div>
                                    <div style="font-size:26px; font-weight:900; color:{_vrp_badge_col}; margin:4px 0; font-family:'JetBrains Mono', monospace;">{_vrp_val:+.2f}%</div>
                                    <div class="metric-sub" style="color:{_vrp_badge_col}; font-weight:700;">{_vrp_badge}</div>
                                </div>
                                <div class="metric-box" style="padding:14px; text-align:left; background:rgba(18,18,42,0.85); border-top:3px solid #ffd54f;">
                                    <div class="metric-label" style="font-size:11px; color:{MUTED};">HISTORICAL VOLATILITY (20D)</div>
                                    <div style="font-size:26px; font-weight:900; color:{WHITE}; margin:4px 0; font-family:'JetBrains Mono', monospace;">{_hv_20d:.2f}%</div>
                                    <div class="metric-sub" style="color:#ffd54f; font-weight:700;">Rank: {_hv_pctile:.0f}th Percentile (1-Yr)</div>
                                </div>
                            </div>

                            <!-- Actionable Trading Playbook (3 Clean Pillars) -->
                            <div class="card" style="padding:14px; background:rgba(18,18,42,0.85);">
                                <div style="font-size:12px; font-weight:800; color:{ACCENT}; text-transform:uppercase; letter-spacing:1px; margin-bottom:10px;">🎯 ACTIONABLE TRADING PLAYBOOK FOR THIS REGIME</div>
                                <div style="display:grid; grid-template-columns: repeat(3, 1fr); gap:10px;">
                                    <div style="background:rgba(255,255,255,0.03); border:1px solid #2a2a4a; border-radius:8px; padding:12px;">
                                        <div style="font-size:11px; font-weight:800; color:{GREEN}; margin-bottom:4px;">1. RECOMMENDED STRATEGIES</div>
                                        <div style="font-size:13px; font-weight:700; color:{WHITE}; line-height:1.4;">{_rec_trades}</div>
                                        <div style="font-size:11px; color:{MUTED}; margin-top:4px;">Aligned with current VRP and regime dynamics.</div>
                                    </div>
                                    <div style="background:rgba(255,255,255,0.03); border:1px solid #2a2a4a; border-radius:8px; padding:12px;">
                                        <div style="font-size:11px; font-weight:800; color:#ffd54f; margin-bottom:4px;">2. RISK & DEFENSE BOUNDARY</div>
                                        <div style="font-size:12px; color:#e2e8f0; line-height:1.4;">{_rec_risk}</div>
                                    </div>
                                    <div style="background:rgba(255,255,255,0.03); border:1px solid #2a2a4a; border-radius:8px; padding:12px;">
                                        <div style="font-size:11px; font-weight:800; color:#38bdf8; margin-bottom:4px;">3. TIMING & INTRADAY CONTEXT</div>
                                        <div style="font-size:12px; color:#e2e8f0; line-height:1.4;">Intraday RV: <strong style="color:{WHITE};">{_rv_intra:.2f}%</strong>. {_vrp_sub}</div>
                                    </div>
                                </div>
                            </div>

                            <!-- Streamlined Volatility Horizon Strip -->
                            <div class="card" style="padding:14px; background:rgba(18,18,42,0.85);">
                                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:10px;">
                                    <div style="font-size:12px; font-weight:800; color:{ACCENT}; text-transform:uppercase; letter-spacing:1px;">📊 REALIZED VOLATILITY TERM STRUCTURE HORIZON</div>
                                    <div style="font-size:11px; color:{MUTED};">Consensus Benchmark: <strong style="color:{ACCENT};">{_rv_cons:.2f}%</strong></div>
                                </div>
                                <div style="display:grid; grid-template-columns: repeat(4, 1fr); gap:8px;">
                                    <div class="metric-box" style="padding:10px; text-align:center;">
                                        <div class="metric-label" style="font-size:10px;">INTRADAY (1D)</div>
                                        <div style="font-size:18px; font-weight:800; color:{WHITE}; margin:2px 0;">{_rv_intra:.2f}%</div>
                                        <div class="metric-sub">Session Realized</div>
                                    </div>
                                    <div class="metric-box" style="padding:10px; text-align:center;">
                                        <div class="metric-label" style="font-size:10px;">SHORT-TERM (5D)</div>
                                        <div style="font-size:18px; font-weight:800; color:{WHITE}; margin:2px 0;">{_rv_5d:.2f}%</div>
                                        <div class="metric-sub">Weekly Trend</div>
                                    </div>
                                    <div class="metric-box" style="padding:10px; text-align:center;">
                                        <div class="metric-label" style="font-size:10px;">MEDIUM-TERM (20D)</div>
                                        <div style="font-size:18px; font-weight:800; color:{ACCENT}; margin:2px 0;">{_rv_20d:.2f}%</div>
                                        <div class="metric-sub">Monthly Base</div>
                                    </div>
                                    <div class="metric-box" style="padding:10px; text-align:center;">
                                        <div class="metric-label" style="font-size:10px;">QUARTERLY (60D)</div>
                                        <div style="font-size:18px; font-weight:800; color:{WHITE}; margin:2px 0;">{_rv_60d:.2f}%</div>
                                        <div class="metric-sub">Quarterly Anchor</div>
                                    </div>
                                </div>
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
<div id="frag-theta">{theta_tab_html}</div>
<div id="frag-prob">{prob_tab_html}</div>
<div id="frag-mm">{mm_tab_html}</div>
<div id="frag-spot" data-spot="{spot:.0f}" data-time="{now_str}">
    <span id="frag-verdict-transfer" style="display:none;">{verdict_html}</span>
</div>'''
                    with open(frag_path, 'w', encoding='utf-8') as f:
                        f.write(fragment_html)

                    # Write full page (clean modular shell + live client controllers)
                    full_html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta name="theme-color" content="#06080d">
    <title>F-Intel | Quantitative Volatility & Options Terminal</title>
    
    <!-- Google Fonts & Plotly -->
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700&family=Outfit:wght@400;500;600;700;800&family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <script src="https://cdn.plot.ly/plotly-2.35.2.min.js" charset="utf-8"></script>

    <!-- PWA Manifest -->
    <link rel="manifest" href="/static/manifest.json">

    <!-- Stylesheets -->
    <link rel="stylesheet" href="/static/css/theme.css">
    <link rel="stylesheet" href="/static/css/layout.css">
    <link rel="stylesheet" href="/static/css/gamma_explosion.css">
</head>
<body>
    <!-- Hardware-Accelerated Quant Lattice Background -->
    <canvas id="quant-bg-canvas"></canvas>

    <!-- Top Navigation Header -->
    <header class="top-nav">
        <div class="brand-section">
            <div class="brand-logo">
                <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" style="color:var(--accent-cyan);">
                    <polyline points="22 12 18 12 15 21 9 3 6 12 2 12"></polyline>
                </svg>
                <span>F-INTEL</span>
            </div>
            <span class="brand-badge">PRO QUANT</span>
            <div id="top-verdict-pill" style="display:inline-flex;">{verdict_html}</div>
        </div>

        <div class="market-strip">
            <div id="spot-display" class="spot-pill">
                SPOT: <span class="spot-val">{spot:,.2f}</span>
            </div>

            <div class="system-badges">
                <span id="ws-status-badge" class="status-indicator" style="background:rgba(255,255,255,0.08); color:var(--text-muted); border:1px solid var(--border-subtle);">
                    <span class="pulse-dot"></span> LIVE
                </span>
                <span id="time-display" class="num-mono" style="font-size:12px; color:var(--text-muted);">
                    &#128339; {now_str}
                </span>
            </div>

            <div class="fx-toggle-wrap">
                <span>FX</span>
                <label class="fx-toggle" title="Toggle Ambient Background Animation">
                    <input type="checkbox" id="fx-toggle-input" checked>
                    <span class="fx-slider"></span>
                </label>
            </div>
        </div>
    </header>

    <!-- Tab Navigation -->
    <nav class="tab-navigation">
        <button class="tab-btn active" data-tab="regime"><span>REGIME SYSTEM</span></button>
        <button class="tab-btn" data-tab="iv"><span>IV SURFACE & SMILE</span></button>
        <button class="tab-btn" data-tab="vol"><span>REALIZED VOL & VRP</span></button>
        <button class="tab-btn" data-tab="chain"><span>OPTION CHAIN & GREEKS</span></button>
        <button class="tab-btn" data-tab="theta"><span>THETA DECAY & SIMULATOR</span></button>
        <button class="tab-btn" data-tab="prob"><span>PROBABILITY CONE</span></button>
        <button class="tab-btn mm-tab-btn" data-tab="mm">
            <span style="display:inline-flex; align-items:center; gap:6px;">
                <span style="color:var(--accent-cyan);">⚡</span>
                <span style="font-weight:700;">GAMMA EXPLOSION & MM</span>
                <span class="pulse-badge" style="font-size:9px; font-weight:800; padding:2px 6px; border-radius:4px; background:var(--accent-cyan); color:#060812; letter-spacing:0.5px;">NEW</span>
            </span>
        </button>
    </nav>

    <!-- Main Workspace Container -->
    <main class="workspace-container" id="dash-container">
        <section id="tab-regime" class="tab-content active">{regime_tab_html}</section>
        <section id="tab-iv" class="tab-content">{iv_tab_html}</section>
        <section id="tab-vol" class="tab-content">{vol_tab_html}</section>
        <section id="tab-chain" class="tab-content">{chain_tab_html}</section>
        <section id="tab-theta" class="tab-content">{theta_tab_html}</section>
        <section id="tab-prob" class="tab-content">{prob_tab_html}</section>
        <section id="tab-mm" class="tab-content">{mm_tab_html}</section>
    </main>

    <!-- Client Scripts -->
    <script src="/static/js/background_canvas.js"></script>
    <script src="/static/js/theta_simulator.js"></script>
    <script src="/static/js/gamma_explosion_terminal.js"></script>
    <script src="/static/js/dashboard_core.js"></script>
</body>
</html>'''
                    with open(html_path, 'w', encoding='utf-8') as f:
                        f.write(full_html)
                    try:
                        tpl_dest = os.path.join(dashboard_dir, 'templates', 'unified_dashboard.html')
                        with open(tpl_dest, 'w', encoding='utf-8') as f:
                            f.write(full_html)
                    except Exception:
                        pass
                    if first_run:
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

        session_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fintel_session.json")
        near, far, extras = None, None, []
        if os.path.exists(session_file):
            for _ in range(5):
                try:
                    with open(session_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        near = data.get("near_expiry")
                        far = data.get("far_expiry")
                        if near and far:
                            break
                except Exception:
                    pass
                time.sleep(0.5)

        if near and far:
            print(f"  Loaded expiries from fintel_session.json")
        else:
            print(f"  Format: YYYY-MM-DD (Press Enter to use defaults)\n")
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