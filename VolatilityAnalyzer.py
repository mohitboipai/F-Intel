import sys
import os
import time
import json
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np # Math support for arrays
from scipy.stats import norm

# Add current directory to path to find local modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8') # type: ignore
    except AttributeError:
        pass

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
DARK_BG = '#131722'
CARD_BG = '#1e222d'
ACCENT  = '#2962ff'
RED     = '#ef5350'
GREEN   = '#26a69a'
YELLOW  = '#ff9800'
WHITE   = '#d1d4dc'
MUTED   = '#868993'

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
        self._last_net_gex = 0.0
        self._last_dte = 99.0

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

    def compute_seller_data(self, arg1=None, arg2=None, df_chain=None, spot=0.0, T=None, DTE=None, near_exp=None, baseline_oi=None, velocity_baseline=None):
        """Compute seller advisor data and enriched merged option chain rows."""
        if isinstance(arg1, pd.DataFrame):
            df_chain = arg1
            if arg2 is not None and spot == 0.0:
                spot = float(arg2)
        elif isinstance(arg2, pd.DataFrame):
            df_chain = arg2
            if arg1 is not None and spot == 0.0:
                spot = float(arg1)
        elif df_chain is None and isinstance(arg1, (int, float)):
            spot = float(arg1)

        if df_chain is None or df_chain.empty:
            return None

        if spot <= 0:
            if hasattr(self, 'spot_price') and self.spot_price > 0:
                spot = self.spot_price
            else:
                spot = float(np.median(df_chain['strike'].values.astype(float)))

        if T is None:
            if near_exp:
                try:
                    T = self.analytics.get_time_to_expiry(near_exp)
                except Exception:
                    T = 7 / 365
            elif hasattr(self, '_near_expiry') and self._near_expiry:
                try:
                    T = self.analytics.get_time_to_expiry(self._near_expiry)
                except Exception:
                    T = 7 / 365
            else:
                T = 7 / 365

        if T <= 0:
            T = 1 / 365
        if DTE is None:
            DTE = max(1, int(T * 365))

        df_chain = df_chain.copy()
        df_chain['dist'] = abs(df_chain['strike'] - spot)
        ce_df = df_chain[df_chain['type'] == 'CE'].copy()
        pe_df = df_chain[df_chain['type'] == 'PE'].copy()
        atm_strike = float(df_chain.loc[int(df_chain['dist'].idxmin()), 'strike']) if len(df_chain) > 0 else spot

        ce_atm = ce_df[ce_df['strike'] >= spot]
        if len(ce_atm) == 0: ce_atm = ce_df
        atm_ce = ce_atm.loc[int(ce_atm['dist'].idxmin())] if len(ce_atm) > 0 else None

        pe_atm = pe_df[pe_df['strike'] <= spot]
        if len(pe_atm) == 0: pe_atm = pe_df
        atm_pe = pe_atm.loc[int(pe_atm['dist'].idxmin())] if len(pe_atm) > 0 else None

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
        em_pct = (straddle / spot) * 100 if spot > 0 else 0.0

        current_oi = {}
        for row_tup in df_chain[['strike', 'type', 'oi']].itertuples(index=False):
            key = (round(float(row_tup.strike)), str(row_tup.type))
            current_oi[key] = int(getattr(row_tup, 'oi', 0) or 0)

        if baseline_oi is None:
            baseline_oi = dict(current_oi)
        if velocity_baseline is None:
            velocity_baseline = dict(current_oi)

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
        except Exception:
            pass

        all_strikes = sorted(df_chain['strike'].unique())
        all_int = [round(float(s)) for s in all_strikes]
        atm_idx = min(range(len(all_int)), key=lambda i: abs(all_int[i] - spot)) if all_int else 0
        nearby_set = set(all_int[max(0, atm_idx - 12): min(len(all_int), atm_idx + 13)])
        for _w in [call_wall, put_wall, call_wall_2, put_wall_2, max_pain]:
            if _w:
                _w_int = round(float(_w))
                if _w_int in all_int:
                    nearby_set.add(_w_int)
        nearby = sorted(list(nearby_set))

        call_vel_added = 0
        call_vel_unwound = 0
        put_vel_added = 0
        put_vel_unwound = 0

        chain_map = {}
        for crow_dict in df_chain.to_dict('records'):
            chain_map[(round(float(crow_dict['strike'])), str(crow_dict['type']))] = crow_dict

        _lot = _get_cfg("nifty_lot_size", 65)
        _r_gex = _get_cfg("risk_free_rate", 0.051274)
        _q_gex = _get_cfg("dividend_yield", 0.0122)
        _T_gex = max(T, 1 / 365)
        _sqT = np.sqrt(max(_T_gex, 1e-6))

        chain_rows = []
        for strike in nearby:
            ce_row = chain_map.get((strike, 'CE'))
            pe_row = chain_map.get((strike, 'PE'))

            ce_oi = current_oi.get((strike, 'CE'), 0)
            ce_base = baseline_oi.get((strike, 'CE'), 0)
            ce_vel_base = velocity_baseline.get((strike, 'CE'), 0)

            pe_oi = current_oi.get((strike, 'PE'), 0)
            pe_base = baseline_oi.get((strike, 'PE'), 0)
            pe_vel_base = velocity_baseline.get((strike, 'PE'), 0)

            ce_chg = ce_oi - ce_base
            pe_chg = pe_oi - pe_base
            ce_vel = ce_oi - ce_vel_base
            pe_vel = pe_oi - pe_vel_base

            ce_price = float(ce_row['price']) if ce_row is not None and not pd.isna(ce_row['price']) else 0.0
            pe_price = float(pe_row['price']) if pe_row is not None and not pd.isna(pe_row['price']) else 0.0

            ce_iv_val = float(ce_row['iv']) if ce_row is not None and not pd.isna(ce_row['iv']) and ce_row['iv'] > 0 else 0.0
            pe_iv_val = float(pe_row['iv']) if pe_row is not None and not pd.isna(pe_row['iv']) and pe_row['iv'] > 0 else 0.0
            if ce_iv_val <= 0 and ce_price > 0:
                ce_iv_val = self._ensure_iv(ce_iv_val, ce_price, strike, _T_gex, 'CE')
            if pe_iv_val <= 0 and pe_price > 0:
                pe_iv_val = self._ensure_iv(pe_iv_val, pe_price, strike, _T_gex, 'PE')

            sig_ce = max(ce_iv_val / 100.0, 0.02) if ce_iv_val > 0 else sigma
            d1_ce = (np.log(spot / strike) + (_r_gex - _q_gex + 0.5 * sig_ce**2) * _T_gex) / (sig_ce * _sqT)
            gamma_ce = np.exp(-_q_gex * _T_gex) * np.exp(-0.5 * d1_ce**2) / (np.sqrt(2 * np.pi) * spot * sig_ce * _sqT) if (spot > 0 and strike > 0) else 0.0
            ce_contracts = (ce_oi / _lot) if ce_oi > 100_000 else float(ce_oi)
            ce_gex_lots = +1.0 * ce_contracts * gamma_ce * 50.0
            ce_gex_cr = (+1.0 * ce_contracts * _lot * gamma_ce * spot * spot * 0.01) / 1e7

            sig_pe = max(pe_iv_val / 100.0, 0.02) if pe_iv_val > 0 else sigma
            d1_pe = (np.log(spot / strike) + (_r_gex - _q_gex + 0.5 * sig_pe**2) * _T_gex) / (sig_pe * _sqT)
            gamma_pe = np.exp(-_q_gex * _T_gex) * np.exp(-0.5 * d1_pe**2) / (np.sqrt(2 * np.pi) * spot * sig_pe * _sqT) if (spot > 0 and strike > 0) else 0.0
            pe_contracts = (pe_oi / _lot) if pe_oi > 100_000 else float(pe_oi)
            pe_gex_lots = -1.0 * pe_contracts * gamma_pe * 50.0
            pe_gex_cr = (-1.0 * pe_contracts * _lot * gamma_pe * spot * spot * 0.01) / 1e7

            dist_ce = strike - spot
            try:
                d1_ce_otm = (np.log(spot / strike) + (_r_gex + 0.5 * sig_ce**2) * _T_gex) / (sig_ce * _sqT)
                d2_ce_otm = d1_ce_otm - sig_ce * _sqT
                ce_prob_otm = norm.cdf(-d2_ce_otm) * 100.0 if dist_ce >= 0 else norm.cdf(d2_ce_otm) * 100.0
            except Exception:
                ce_prob_otm = 50.0

            try:
                greeks_ce = self.analytics.calculate_greeks(spot, strike, _T_gex, _r_gex, sig_ce, 'CE')
                ce_theta = greeks_ce.get('theta', 0.0)
            except Exception:
                ce_theta = 0.0

            if dist_ce >= 0:
                ce_signal = "★ SAFE" if ce_prob_otm >= 85 else "✓ GOOD" if ce_prob_otm >= 75 else "~ OK" if ce_prob_otm >= 60 else "✗ RISKY"
            else:
                ce_signal = "ITM"

            dist_pe = spot - strike
            try:
                d1_pe_otm = (np.log(spot / strike) + (_r_gex + 0.5 * sig_pe**2) * _T_gex) / (sig_pe * _sqT)
                d2_pe_otm = d1_pe_otm - sig_pe * _sqT
                pe_prob_otm = norm.cdf(d2_pe_otm) * 100.0 if dist_pe >= 0 else norm.cdf(-d2_pe_otm) * 100.0
            except Exception:
                pe_prob_otm = 50.0

            try:
                greeks_pe = self.analytics.calculate_greeks(spot, strike, _T_gex, _r_gex, sig_pe, 'PE')
                pe_theta = greeks_pe.get('theta', 0.0)
            except Exception:
                pe_theta = 0.0

            if dist_pe >= 0:
                pe_signal = "★ SAFE" if pe_prob_otm >= 85 else "✓ GOOD" if pe_prob_otm >= 75 else "~ OK" if pe_prob_otm >= 60 else "✗ RISKY"
            else:
                pe_signal = "ITM"

            net_gex_lots = ce_gex_lots + pe_gex_lots
            net_gex_cr = ce_gex_cr + pe_gex_cr
            is_atm = abs(strike - spot) < 60

            chain_rows.append({
                'strike': strike,
                'ce_oi': ce_oi, 'pe_oi': pe_oi,
                'ce_chg': ce_chg, 'pe_chg': pe_chg, 
                'ce_vel': ce_vel, 'pe_vel': pe_vel,
                'ce_price': ce_price, 'pe_price': pe_price,
                'ce_iv': ce_iv_val, 'pe_iv': pe_iv_val,
                'ce_gex_lots': ce_gex_lots, 'pe_gex_lots': pe_gex_lots,
                'net_gex_lots': net_gex_lots,
                'ce_gex_cr': ce_gex_cr, 'pe_gex_cr': pe_gex_cr,
                'net_gex_cr': net_gex_cr,
                'is_atm': is_atm,
                'ce_prob_otm': ce_prob_otm, 'ce_theta': ce_theta, 'ce_signal': ce_signal,
                'pe_prob_otm': pe_prob_otm, 'pe_theta': pe_theta, 'pe_signal': pe_signal
            })
            
            if strike >= spot:
                if ce_vel > 0: call_vel_added += ce_vel
                elif ce_vel < 0: call_vel_unwound += abs(ce_vel)
            if strike <= spot:
                if pe_vel > 0: put_vel_added += pe_vel
                elif pe_vel < 0: put_vel_unwound += abs(pe_vel)

        total_force = (put_vel_added + call_vel_unwound) + (call_vel_added + put_vel_unwound)
        if total_force > 0:
            oi_pressure_score = (((put_vel_added + call_vel_unwound) - (call_vel_added + put_vel_unwound)) / total_force) * 100
        else:
            oi_pressure_score = 0
        
        oi_pressure = "BULLISH" if oi_pressure_score > 15 else "BEARISH" if oi_pressure_score < -15 else "NEUTRAL"

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

    _compute_seller_data = compute_seller_data

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

        # ── 0DTE Short-Gamma Cascade Guard ───────────────────────────────────────
        # When DTE <= 1.2 and net dealer GEX is negative, dealer delta-hedging accelerates
        # directional moves into cascades (e.g. 70-80 pt drops where OTM puts double).
        # Selling vol (Iron Condors / Credit Spreads) into an active short-gamma cascade is catastrophic.
        _chk_net_gex = metrics.get('net_gex', getattr(self, '_last_net_gex', 0.0))
        _chk_dte = metrics.get('dte', getattr(self, '_last_dte', 99.0))
        if _chk_dte <= 1.2 and _chk_net_gex < -1e5:
            action = "BUY VOL (0DTE SQUEEZE)"
            strategy = "0DTE Short Gamma Cascade — Long Convexity / Momentum Puts/Calls"
            composite = min(composite, 25.0)
        elif composite > 65:
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

    def _format_signal_report(self, spot, iv, hv, signal_result, regime, iv_velocity_5d: float = 0.0, iv_accel: float = 0.0, vrp: float = 0.0):
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
                if getattr(self, '_last_dte', 99.0) <= 1.2 and getattr(self, '_last_net_gex', 0.0) < -1e5:
                    sig = "0DTE GAMMA SQUEEZE (BUY VOL)"
                elif daily_vrp > 5.0:
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
    _BUILDER_TERMINAL_SHELL = '''
<div class="sb-workspace">
    <!-- Left Sidebar: Presets, Live Positions, AI Wizard -->
    <aside class="sb-sidebar">
        <div class="sb-nav-tabs">
            <button class="sb-subtab-btn active" data-subtab="sb-subtab-ready" onclick="StrategyBuilder.switchSubTab('sb-subtab-ready')">Presets</button>
            <button class="sb-subtab-btn" data-subtab="sb-subtab-portfolio" onclick="StrategyBuilder.switchSubTab('sb-subtab-portfolio')">Positions <span id="sb-pos-badge" style="background:#00e5ff; color:#131722; border-radius:8px; padding:1px 5px; font-size:9px; margin-left:3px; display:none;">0</span></button>
            <button class="sb-subtab-btn" data-subtab="sb-subtab-ai" onclick="StrategyBuilder.switchSubTab('sb-subtab-ai')">AI Wizard</button>
            <button class="sb-subtab-btn" onclick="window.open('/builder', '_blank')" style="color:#00e5ff; font-weight:700;">Backtest Studio ↗</button>
        </div>

        <!-- Presets SubTab -->
        <div id="sb-subtab-ready" class="sb-subtab-content active">
            <div class="sb-category">
                <div class="sb-category-title" style="color:#00e676;">Bullish</div>
                <button class="sb-preset-btn" onclick="StrategyBuilder.buildStrategy('Bull Call Spread')">Bull Call Spread <span>↗</span></button>
                <button class="sb-preset-btn" onclick="StrategyBuilder.buildStrategy('Bull Put Spread')">Bull Put Spread <span>↗</span></button>
            </div>
            <div class="sb-category">
                <div class="sb-category-title" style="color:#ff3366;">Bearish</div>
                <button class="sb-preset-btn" onclick="StrategyBuilder.buildStrategy('Bear Put Spread')">Bear Put Spread <span>↘</span></button>
                <button class="sb-preset-btn" onclick="StrategyBuilder.buildStrategy('Bear Call Spread')">Bear Call Spread <span>↘</span></button>
            </div>
            <div class="sb-category">
                <div class="sb-category-title" style="color:#00e5ff;">Neutral / Vol Selling</div>
                <button class="sb-preset-btn" onclick="StrategyBuilder.buildStrategy('Short Straddle')">Short Straddle <span>🎯</span></button>
                <button class="sb-preset-btn" onclick="StrategyBuilder.buildStrategy('Short Strangle')">Short Strangle <span>🎯</span></button>
                <button class="sb-preset-btn" onclick="StrategyBuilder.buildStrategy('Iron Condor')">Iron Condor <span>🛡</span></button>
                <button class="sb-preset-btn" onclick="StrategyBuilder.buildStrategy('Iron Butterfly')">Iron Butterfly <span>🛡</span></button>
            </div>
            <div class="sb-category">
                <div class="sb-category-title" style="color:#ffd54f;">Ratio Backspreads</div>
                <button class="sb-preset-btn" onclick="StrategyBuilder.buildStrategy('Call Ratio Backspread')">Call Ratio Backspread <span>🚀</span></button>
                <button class="sb-preset-btn" onclick="StrategyBuilder.buildStrategy('Put Ratio Backspread')">Put Ratio Backspread <span>⚡</span></button>
            </div>
        </div>

        <!-- Portfolio SubTab -->
        <div id="sb-subtab-portfolio" class="sb-subtab-content">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;">
                <span style="font-size:11px; font-weight:700; color:#fff;">Live Paper Trades</span>
                <button onclick="StrategyBuilder.openOrderbook()" style="background:transparent; border:1px solid #2a2e39; color:#868993; border-radius:4px; padding:2px 6px; font-size:10px; cursor:pointer;">P&L History</button>
            </div>
            <div id="sb-active-positions-container">
                <div style="color:#868993; font-size:12px; text-align:center; padding: 24px 0;">Loading paper positions...</div>
            </div>
        </div>

        <!-- AI Wizard SubTab -->
        <div id="sb-subtab-ai" class="sb-subtab-content">
            <button id="sb-btn-fetch-wizard" onclick="StrategyBuilder.fetchWizardRecommendation()" style="width:100%; padding:8px 12px; background:linear-gradient(135deg, #00e5ff, #00b0ff); color:#000; border:none; border-radius:6px; font-size:11px; font-weight:800; cursor:pointer; margin-bottom:8px;">⚡ Suggest AI Strategy</button>
            <div style="font-size:10px; font-weight:800; color:#868993; text-transform:uppercase; margin-bottom:6px;">Today's Live Recommendations</div>
            <div id="sb-ai-presets-container" style="display:flex; flex-direction:column; gap:8px;">
                <div style="color:#868993; font-size:11px; text-align:center; padding: 18px 0;">No AI recommendations yet today.</div>
            </div>
        </div>
    </aside>

    <!-- Main Strategy Builder Panel -->
    <div class="sb-dashboard">
        <!-- Top Row: Payoff Chart + Strategy Summary & Greeks -->
        <div class="sb-top-grid">
            <!-- Chart Card -->
            <div class="sb-card" style="display:flex; flex-direction:column;">
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:10px;">
                    <span style="font-size:12px; font-weight:800; color:#00e5ff; letter-spacing:0.8px;">INTERACTIVE PAYOFF PROFILE</span>
                    <span style="font-size:11px; color:#868993;">NIFTY Spot: <strong id="sb-spot-display" style="color:#fff; font-family:var(--font-mono);">--</strong></span>
                </div>
                <div class="sb-controls-row">
                    <div class="sb-slider-group">
                        <label><span>Spot Shift</span> <span id="sb-val-target-spot" style="font-family:var(--font-mono); color:#00e676;">+0.0%</span></label>
                        <input type="range" id="sb-slide-spot" min="-15" max="15" value="0" step="0.5">
                    </div>
                    <div class="sb-slider-group">
                        <label><span>Target Date</span> <span id="sb-val-target-date" style="font-family:var(--font-mono);">T+0D</span></label>
                        <input type="range" id="sb-slide-date" min="0" max="30" value="0" step="1">
                    </div>
                    <div class="sb-slider-group">
                        <label><span>Target Time</span> <span id="sb-val-target-time" style="font-family:var(--font-mono);">15:30</span></label>
                        <input type="range" id="sb-slide-time" min="0" max="25" value="25" step="1">
                    </div>
                </div>
                <div id="sb-payoff-chart" style="height:320px; width:100%;"></div>
                <div style="text-align:center; font-size:13px; font-weight:700; color:#868993; margin-top:4px;">
                    Projected P&L: <span id="sb-projected-profit" style="font-family:var(--font-mono); font-size:15px; color:#00e676;">₹0</span>
                </div>
            </div>

            <!-- Summary Card -->
            <div class="sb-card" style="display:flex; flex-direction:column; justify-content:space-between;">
                <div>
                    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:12px;">
                        <h3 id="sb-summary-strategy-name" style="font-size:14px; font-weight:800; color:#fff; margin:0;">Custom Strategy</h3>
                        <span class="sb-badge sb-badge-cyan">BSM MODEL</span>
                    </div>
                    <div class="sb-summary-grid">
                        <div class="sb-stat-box">
                            <div class="sb-stat-label">Max Profit</div>
                            <div class="sb-stat-value" id="sb-stat-max-profit" style="color:#00e676;">₹0</div>
                        </div>
                        <div class="sb-stat-box">
                            <div class="sb-stat-label">Max Loss</div>
                            <div class="sb-stat-value" id="sb-stat-max-loss" style="color:#ff3366;">₹0</div>
                        </div>
                        <div class="sb-stat-box">
                            <div class="sb-stat-label">Prob of Profit</div>
                            <div class="sb-stat-value" id="sb-stat-pop" style="color:#00e5ff;">0.0%</div>
                        </div>
                        <div class="sb-stat-box">
                            <div class="sb-stat-label">Net Premium</div>
                            <div class="sb-stat-value" id="sb-stat-premium" style="color:#ffd54f;">₹0</div>
                        </div>
                    </div>
                    <div style="background:var(--bg-canvas,#131722); padding:10px 12px; border-radius:6px; border:1px solid var(--border-subtle,#2a2e39); margin-bottom:12px;">
                        <div style="display:flex; justify-content:space-between; font-size:11px;">
                            <span style="color:#868993;">Expiry Breakevens:</span>
                            <span id="sb-stat-breakevens" style="font-family:var(--font-mono); font-weight:700; color:#fff;">-</span>
                        </div>
                    </div>
                </div>
                <div>
                    <div style="font-size:10px; font-weight:800; color:#868993; text-transform:uppercase; margin-bottom:6px;">Net Strategy Greeks</div>
                    <div class="sb-greeks-row">
                        <div class="sb-greek-box"><div class="sb-greek-label">Delta (Δ)</div><div class="sb-greek-value" id="sb-greek-delta">0.00</div></div>
                        <div class="sb-greek-box"><div class="sb-greek-label">Theta (θ)</div><div class="sb-greek-value" id="sb-greek-theta">0.00</div></div>
                        <div class="sb-greek-box"><div class="sb-greek-label">Gamma (Γ)</div><div class="sb-greek-value" id="sb-greek-gamma">0.0000</div></div>
                        <div class="sb-greek-box"><div class="sb-greek-label">Vega (ν)</div><div class="sb-greek-value" id="sb-greek-vega">0.00</div></div>
                    </div>
                </div>
            </div>
        </div>

        <!-- Middle Row: Leg Builder Basket -->
        <div class="sb-card">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:12px;">
                <div style="font-size:12px; font-weight:800; color:#00e5ff; letter-spacing:0.8px;">ACTIVE STRATEGY LEGS</div>
                <div style="display:flex; gap:8px;">
                    <button onclick="StrategyBuilder.clearBasket()" style="padding:4px 10px; background:transparent; border:1px solid #2a2e39; color:#868993; border-radius:4px; font-size:11px; cursor:pointer;">Clear All</button>
                    <button onclick="StrategyBuilder.deployStrategy()" style="padding:4px 12px; background:#00e676; color:#131722; font-weight:800; border:none; border-radius:4px; font-size:11px; cursor:pointer;">Deploy Paper Trade 🚀</button>
                </div>
            </div>
            <table class="sb-table">
                <thead>
                    <tr>
                        <th>Side</th>
                        <th>Lots (x65)</th>
                        <th>Expiry</th>
                        <th>Strike</th>
                        <th>Type</th>
                        <th>Entry Price</th>
                        <th>IV</th>
                        <th>Del</th>
                    </tr>
                </thead>
                <tbody id="sb-basket-body">
                    <tr><td colspan="8" style="text-align:center; color:#868993; padding: 24px 0;">No active legs. Click preset on left or +B / +S from option chain below.</td></tr>
                </tbody>
            </table>
        </div>

        <!-- Bottom Row: Option Chain with Quick Add Buttons -->
        <div class="sb-card">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:10px;">
                <div style="font-size:12px; font-weight:800; color:#00e5ff; letter-spacing:0.8px;">OPTION CHAIN & QUICK LEG SELECTOR</div>
                <div style="display:flex; gap:10px; align-items:center;">
                    <select id="sb-expiry-select" style="padding:4px 8px; border-radius:4px; background:#131722; color:#fff; border:1px solid #2a2e39; font-size:11px;"></select>
                    <div class="sb-action-btn-group">
                        <button id="sb-view-ltp" class="active" onclick="StrategyBuilder.setChainView('LTP')">LTP & OI</button>
                        <button id="sb-view-greeks" onclick="StrategyBuilder.setChainView('GREEKS')">GREEKS</button>
                    </div>
                </div>
            </div>
            <div class="sb-chain-wrapper">
                <table class="sb-chain-table">
                    <thead>
                        <tr>
                            <th>Call OI</th>
                            <th>Call Δ</th>
                            <th>Call LTP</th>
                            <th class="sb-strike-col">Strike</th>
                            <th>Put LTP</th>
                            <th>Put Δ</th>
                            <th>Put OI</th>
                        </tr>
                    </thead>
                    <tbody id="sb-chain-body">
                        <tr><td colspan="7" style="padding: 24px 0; color:#868993;">Loading chain data...</td></tr>
                    </tbody>
                </table>
            </div>
        </div>
    </div>
</div>

<!-- Orderbook Modal -->
<div id="sb-orderbook-modal" class="sb-modal-overlay">
    <div class="sb-modal-box">
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:14px;">
            <h3 style="font-size:14px; font-weight:800; color:#fff; margin:0;">Exited Paper Positions Directory</h3>
            <button onclick="StrategyBuilder.closeOrderbook()" style="background:transparent; border:none; color:#868993; font-size:18px; cursor:pointer;">✕</button>
        </div>
        <table class="sb-table">
            <thead>
                <tr>
                    <th>Exit Time</th>
                    <th>Strategy</th>
                    <th>Req. Margin</th>
                    <th>Realized P&L</th>
                </tr>
            </thead>
            <tbody id="sb-history-body"></tbody>
        </table>
    </div>
</div>
'''

    def _create_unified_dashboard(self, expiries, single_run=False):
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
        DARK_BG = '#131722'
        CARD_BG = '#1e222d'
        ACCENT = '#00e5ff'
        RED = '#ff3366'
        GREEN = '#00e676'
        YELLOW = '#ffd54f'
        WHITE = '#ffffff'
        MUTED = '#868993'
        colors = ['#00e5ff', '#ff7043', '#00e676', '#b388ff', '#ffd54f', '#ff3366']

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
                        'date': [datetime.fromtimestamp(c[0]).strftime('%Y-%m-%d') for c in candles],
                        'timestamp': [c[0] for c in candles],
                        'closes': [float(c[4]) for c in candles],
                        'highs': [float(c[2]) for c in candles],
                        'lows': [float(c[3]) for c in candles],
                        'opens': [float(c[1]) for c in candles]
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
            if getattr(self, '_last_dte', 99.0) <= 1.2 and getattr(self, '_last_net_gex', 0.0) < -1e5:
                sig = "0DTE GAMMA SQUEEZE (BUY VOL)"
            elif daily_vrp > 5.0:
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
            if df_chain.empty or spot <= 0: return None

            current_oi = {}
            for _, orow in df_chain.iterrows():
                key = (int(round(orow['strike'])), orow['type'])
                current_oi[key] = int(orow.get('oi', 0) or 0)

            _now_epoch = time.time()
            oi_velocity_history.append((_now_epoch, dict(current_oi)))
            oi_velocity_history = [x for x in oi_velocity_history if _now_epoch - x[0] <= 900]
            velocity_baseline = oi_velocity_history[0][1] if oi_velocity_history else current_oi
            if baseline_oi is None:
                baseline_oi = dict(current_oi)

            res = self.compute_seller_data(df_chain=df_chain, spot=spot, T=T, near_exp=near_exp,
                                           baseline_oi=baseline_oi, velocity_baseline=velocity_baseline)
            if res:
                oi_pressure = res.get('oi_pressure', 'NEUTRAL')
                oi_pressure_score = res.get('oi_pressure_score', 0)
            return res

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
        _gc_counter = 0

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
                    
                    # ATM IV for prob density (prefer vol intel, fallback to HV or 12.0)
                    _pd_iv = _live_iv if _live_iv > 0 else (_live_hv if _live_hv > 0 else 12.0)
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
                        
                        # Minion mode removed per user request

                    # ══════════════════════════════════════════
                    #  BUILD HTML
                    # ══════════════════════════════════════════

                    # ── TAB 1: IV SURFACE (Interactive Real-Time & Rewind Terminal) ──
                    _IV_TERMINAL_SHELL = '''
<div id="iv-surface-card" class="iv-terminal-card">
    <!-- Top Header -->
    <div class="iv-header">
        <div class="iv-title-wrap">
            <span class="iv-title">
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
                    <circle cx="12" cy="12" r="10"></circle>
                    <path d="M8 12a4 4 0 0 1 8 0"></path>
                </svg>
                REAL-TIME IV SURFACE &amp; EXHAUSTION TERMINAL
            </span>
            <span id="iv-rewind-status-badge" class="iv-status-badge">LIVE STREAMING</span>
        </div>
        <div class="iv-header-right">
            <div class="iv-spot-pill">
                DAY OPEN: <span id="iv-open-val" style="color:#fff;">--</span>
            </div>
            <div class="iv-spot-pill">
                SPOT: <span id="iv-spot-val" style="color:#fff;">--</span>
            </div>
            <span id="iv-snap-time" style="font-family:var(--font-mono); font-size:11px; color:var(--text-muted);">--:--:--</span>
        </div>
    </div>

    <!-- Mathematical Movement & Exhaustion Panel -->
    <div class="iv-exhaustion-panel">
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:12px;">
            <div style="display:flex; align-items:center; gap:10px;">
                <div style="font-family:var(--font-mono); font-size:12px; font-weight:700; color:#00e5ff; letter-spacing:1px;">
                    QUANTITATIVE VOLATILITY BUDGET &amp; CONSUMPTION GAUGES
                </div>
                <span style="font-family:var(--font-mono); font-size:10px; color:#868993; background:rgba(255,255,255,0.06); padding:2px 8px; border-radius:3px;">1-DAY &amp; WEEKLY (1σ)</span>
            </div>
            <span id="iv-regime-tag" class="iv-regime-tag consolidation">CALCULATING</span>
        </div>

        <div class="iv-exhaustion-grid">
            <!-- 1-Day Expected Move -->
            <div class="iv-metric-tile">
                <div class="iv-metric-label">EXPECTED 1-DAY MOVE (1σ)</div>
                <div id="iv-exp-move-pts" class="iv-metric-value">±-- pts</div>
                <div id="iv-exp-move-pct" class="iv-metric-sub">±--%</div>
            </div>
            <!-- Intraday Range -->
            <div class="iv-metric-tile">
                <div class="iv-metric-label">INTRADAY RANGE (H - L)</div>
                <div id="iv-range-pts" class="iv-metric-value">-- pts</div>
                <div id="iv-range-sub" class="iv-metric-sub">H: -- | L: --</div>
            </div>
            <!-- 1-Day 1σ Boundaries -->
            <div class="iv-metric-tile">
                <div class="iv-metric-label">1-DAY 1σ BOUNDARIES</div>
                <div style="display:flex; gap:12px; margin-top:2px;">
                    <div>
                        <span style="font-size:10px; color:#868993;">UPPER:</span>
                        <div id="iv-boundary-upper" style="font-family:var(--font-mono); font-size:13px; font-weight:700; color:#ffa726;">--</div>
                    </div>
                    <div>
                        <span style="font-size:10px; color:#868993;">LOWER:</span>
                        <div id="iv-boundary-lower" style="font-family:var(--font-mono); font-size:13px; font-weight:700; color:#ffa726;">--</div>
                    </div>
                </div>
                <div class="iv-metric-sub" style="margin-top:4px;">Daily standard deviation band</div>
            </div>
            <!-- Weekly Expected Move -->
            <div class="iv-metric-tile" style="border-left: 2px solid rgba(124, 77, 255, 0.4);">
                <div class="iv-metric-label" style="color:#b388ff;">WEEKLY EXPECTED MOVE (1σ)</div>
                <div id="iv-weekly-exp-pts" class="iv-metric-value" style="color:#e0e0ff;">±-- pts</div>
                <div id="iv-weekly-exp-pct" class="iv-metric-sub">±--% (5-Day Horizon)</div>
            </div>
            <!-- Weekly Realized & 5D Range -->
            <div class="iv-metric-tile" style="border-left: 2px solid rgba(124, 77, 255, 0.4);">
                <div class="iv-metric-label" style="color:#b388ff;">WEEKLY REALIZED MOVE &amp; RANGE</div>
                <div id="iv-weekly-realized-pts" class="iv-metric-value">-- pts</div>
                <div id="iv-weekly-range-sub" class="iv-metric-sub">5D Range: -- pts (H: -- | L: --)</div>
            </div>
            <!-- Weekly 1σ Boundaries -->
            <div class="iv-metric-tile" style="border-left: 2px solid rgba(124, 77, 255, 0.4);">
                <div class="iv-metric-label" style="color:#b388ff;">WEEKLY 1σ BOUNDARIES</div>
                <div style="display:flex; gap:12px; margin-top:2px;">
                    <div>
                        <span style="font-size:10px; color:#868993;">UPPER:</span>
                        <div id="iv-weekly-boundary-upper" style="font-family:var(--font-mono); font-size:13px; font-weight:700; color:#b388ff;">--</div>
                    </div>
                    <div>
                        <span style="font-size:10px; color:#868993;">LOWER:</span>
                        <div id="iv-weekly-boundary-lower" style="font-family:var(--font-mono); font-size:13px; font-weight:700; color:#b388ff;">--</div>
                    </div>
                </div>
                <div class="iv-metric-sub" style="margin-top:4px;">Expiry / Weekly volatility band</div>
            </div>
        </div>

        <!-- 1-Day Gauge Bar -->
        <div class="iv-gauge-wrap" style="margin-bottom:12px;">
            <div class="iv-gauge-labels">
                <span style="display:flex; align-items:center; gap:6px;">
                    <span style="color:#00e5ff; font-weight:700;">1-DAY BUDGET:</span>
                    <span>Spot Net Move: <b id="iv-consumption-pct-label" style="color:#00e5ff;">0%</b></span>
                    <span style="color:var(--text-muted);">|</span>
                    <span>Intraday Range Traversed: <b id="iv-range-consumption-label" style="color:#94a3b8;">0%</b></span>
                </span>
                <span style="color:#ffd54f;">100% (1σ Barrier)</span>
            </div>
            <div class="iv-gauge-track">
                <div id="iv-gauge-bar-range" class="iv-gauge-bar-range" title="Intraday High-Low Range Traversed"></div>
                <div id="iv-gauge-bar-fill" class="iv-gauge-bar" title="Current Spot Net Displacement"></div>
                <div class="iv-gauge-marker-1sigma" title="1-Sigma Statistical Barrier (80% scale)"></div>
            </div>
        </div>

        <!-- Weekly Gauge Bar -->
        <div class="iv-gauge-wrap">
            <div class="iv-gauge-labels">
                <span style="display:flex; align-items:center; gap:6px;">
                    <span style="color:#b388ff; font-weight:700;">WEEKLY BUDGET:</span>
                    <span>Weekly Realized Move: <b id="iv-weekly-consumption-label" style="color:#b388ff;">0%</b></span>
                    <span style="color:var(--text-muted);">|</span>
                    <span>5-Day Range Traversed: <b id="iv-weekly-range-consumption-label" style="color:#94a3b8;">0%</b></span>
                </span>
                <span style="color:#ffd54f;">100% (Weekly 1σ Barrier)</span>
            </div>
            <div class="iv-gauge-track">
                <div id="iv-weekly-bar-range" class="iv-gauge-bar-range" title="Weekly High-Low Range Traversed"></div>
                <div id="iv-weekly-bar-fill" class="iv-gauge-bar" title="Weekly Realized Net Displacement"></div>
                <div class="iv-gauge-marker-1sigma" title="Weekly 1-Sigma Statistical Barrier (80% scale)"></div>
            </div>
            <div class="iv-gauge-legend">
                <div class="iv-gauge-legend-items">
                    <span><span class="legend-box solid" style="background:#00e5ff;"></span> Solid Fill: Current Spot Net Move</span>
                    <span><span class="legend-box" style="background:rgba(0, 229, 255, 0.25); border:1px solid #00e5ff;"></span> Shaded Area: Total Range Traversed</span>
                </div>
                <span>Markers: Gold line indicates 1.0σ full volatility budget</span>
            </div>
        </div>
    </div>

    <!-- Rewind Scrubber & Preset Controls -->
    <div class="iv-rewind-toolbar">
        <div style="font-family:var(--font-mono); font-size:11px; font-weight:700; color:#868993;">
            REWIND MEMORY:
        </div>
        <div class="iv-rewind-slider-wrap">
            <input type="range" id="iv-rewind-slider" class="iv-slider" min="0" max="1" value="1" disabled>
            <span id="iv-rewind-time-label" style="font-family:var(--font-mono); font-size:12px; font-weight:700; color:#00e5ff; min-width:65px;">LIVE</span>
        </div>

        <div class="iv-btn-group">
            <button class="iv-tool-btn iv-preset-btn active" data-preset="live">LIVE</button>
            <button class="iv-tool-btn iv-preset-btn" data-preset="5m">-5m</button>
            <button class="iv-tool-btn iv-preset-btn" data-preset="15m">-15m</button>
            <button class="iv-tool-btn iv-preset-btn" data-preset="30m">-30m</button>
            <button class="iv-tool-btn iv-preset-btn" data-preset="60m">-1h</button>
            <button class="iv-tool-btn iv-preset-btn" data-preset="open">DAY OPEN</button>
        </div>

        <div class="iv-btn-group" title="Select baseline for Smile &amp; ΔIV comparison">
            <span style="font-size:10px; color:#868993; padding:4px 6px; align-self:center;">BASELINE:</span>
            <button class="iv-tool-btn iv-baseline-btn active" data-baseline="open">DAY OPEN</button>
            <button class="iv-tool-btn iv-baseline-btn" data-baseline="prev">PRIOR SNAP</button>
        </div>

        <button id="btn-iv-return-live" class="iv-btn-live">
            ↺ RETURN TO LIVE
        </button>
    </div>

    <!-- Total IV Shifts Strip -->
    <div class="iv-shifts-strip">
        <div class="iv-shift-pill">
            <div class="iv-shift-title">ATM IV &amp; Δ SHIFT</div>
            <div id="iv-shift-atm-val" class="iv-shift-val">--%</div>
        </div>
        <div class="iv-shift-pill">
            <div class="iv-shift-title">SKEW RATIO (PUT/CALL)</div>
            <div id="iv-shift-skew-val" class="iv-shift-val">--</div>
        </div>
        <div class="iv-shift-pill">
            <div class="iv-shift-title">TERM SPREAD</div>
            <div id="iv-shift-ts-val" class="iv-shift-val">--%</div>
        </div>
        <div class="iv-shift-pill">
            <div class="iv-shift-title">WING IV (PUT vs CALL)</div>
            <div id="iv-shift-wings-val" class="iv-shift-val">--</div>
        </div>
    </div>

    <!-- Charts Grid: 2D Smile + 3D Surface -->
    <div class="iv-charts-grid">
        <!-- 2D Smile Chart -->
        <div class="iv-chart-box">
            <div class="iv-chart-header">
                <span class="iv-chart-title">2D VOLATILITY SMILE &amp; Δ IV PER STRIKE</span>
                <span style="font-size:10px; font-family:var(--font-mono); color:var(--text-muted);">Active vs Baseline Overlay</span>
            </div>
            <div id="iv-smile-plot" class="iv-plot-canvas"></div>
        </div>

        <!-- 3D Surface Chart -->
        <div class="iv-chart-box">
            <div class="iv-chart-header">
                <span class="iv-chart-title">3D VOLATILITY SURFACE MESH</span>
                <span style="font-size:10px; font-family:var(--font-mono); color:var(--text-muted);">Strike × DTE × IV (%)</span>
            </div>
            <div id="iv-surface-3d-plot" class="iv-plot-canvas"></div>
        </div>
    </div>
</div>
'''
                    iv_tab_html = _IV_TERMINAL_SHELL

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
                        <div class="card" style="padding:16px; background:var(--bg-surface, #1e222d); border:1px solid var(--border-card, #363c4e);">
                            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:14px; flex-wrap:wrap; gap:8px;">
                                <div style="color:{ACCENT}; font-size:12px; font-weight:800; letter-spacing:1px; text-transform:uppercase;">
                                    📈 1-YEAR HISTORICAL VOLATILITY & ESTIMATOR CONE
                                </div>
                                <div style="font-size:11px; color:{MUTED};">
                                    Historical Percentile: <strong style="color:{YELLOW}; font-family:var(--font-mono);">{v['hv_percentile']:.0f}%</strong> · Half-Life: <strong style="color:{WHITE}; font-family:var(--font-mono);">{v['half_life']:.0f}d</strong>
                                </div>
                            </div>
                            <div style="display:grid; grid-template-columns: repeat(4, 1fr); gap:10px; margin-bottom:14px;">
                                <div class="metric-box"><div class="metric-label">20D HV (C2C)</div><div style="font-size:20px; font-weight:700; color:{WHITE}; font-family:var(--font-mono);">{v['current_hv']:.2f}%</div><div class="metric-sub">Parkinson: {v['parkinson']:.2f}%</div></div>
                                <div class="metric-box"><div class="metric-label">1-YR MEAN HV</div><div style="font-size:20px; font-weight:700; color:{WHITE}; font-family:var(--font-mono);">{v['mean_hv']:.2f}%</div><div class="metric-sub">{v['min_hv']:.1f}% &mdash; {v['max_hv']:.1f}% Range</div></div>
                                <div class="metric-box"><div class="metric-label">LIVE ATM IV</div><div style="font-size:20px; font-weight:700; color:{ACCENT}; font-family:var(--font-mono);">{v['iv']:.2f}%</div><div class="metric-sub">Z-Score: {v['z_score']:.2f}&times;</div></div>
                                <div class="metric-box"><div class="metric-label">ANNUAL VRP</div><div style="font-size:20px; font-weight:700; color:{GREEN if v['vrp']>2 else RED if v['vrp']<-2 else WHITE}; font-family:var(--font-mono);">{v['vrp']:+.2f}%</div><div class="metric-sub">{_vrp_pct:.0f}th %ile Rank</div></div>
                            </div>
                            <div>
                                <div style="color:{ACCENT}; font-size:11px; font-weight:700; letter-spacing:1px; margin-bottom:8px;">REALIZED VOLATILITY CONE &mdash; ALL ESTIMATORS</div>
                                <table class="data-table" style="font-size:11px; width:100%;">
                                    <thead><tr><th style="text-align:left;">Estimator</th><th>5D Horizon</th><th>10D Horizon</th><th>20D Base</th><th>60D Macro</th></tr></thead>
                                    <tbody>
                                        <tr><td style="text-align:left; color:{WHITE}; font-weight:600;">Close-to-Close (C2C)</td><td style="font-family:var(--font-mono);">{_hv.get('c2c_5',0):.2f}%</td><td style="font-family:var(--font-mono);">{_hv.get('c2c_10',0):.2f}%</td><td style="font-family:var(--font-mono);">{_hv.get('c2c_20',0):.2f}%</td><td style="font-family:var(--font-mono);">{_hv.get('c2c_60',0):.2f}%</td></tr>
                                        <tr><td style="text-align:left; color:{WHITE}; font-weight:600;">Parkinson (High / Low)</td><td style="font-family:var(--font-mono);">{_hv.get('pk_5',0):.2f}%</td><td style="font-family:var(--font-mono);">{_hv.get('pk_10',0):.2f}%</td><td style="font-family:var(--font-mono);">{_hv.get('pk_20',0):.2f}%</td><td style="font-family:var(--font-mono);">{_hv.get('pk_60',0):.2f}%</td></tr>
                                        <tr><td style="text-align:left; color:{WHITE}; font-weight:600;">Garman-Klass</td><td style="font-family:var(--font-mono);">&mdash;</td><td style="font-family:var(--font-mono);">&mdash;</td><td style="font-family:var(--font-mono);">{_hv.get('gk_20',0):.2f}%</td><td style="font-family:var(--font-mono);">{_hv.get('gk_60',0):.2f}%</td></tr>
                                        <tr><td style="text-align:left; color:{WHITE}; font-weight:600;">Yang-Zhang (Drift-Independent)</td><td style="font-family:var(--font-mono);">&mdash;</td><td style="font-family:var(--font-mono);">&mdash;</td><td style="font-family:var(--font-mono);">{_hv.get('yz_20',0):.2f}%</td><td style="font-family:var(--font-mono);">{_hv.get('yz_60',0):.2f}%</td></tr>
                                        <tr style="background:rgba(0,229,255,0.08);"><td style="text-align:left; font-weight:800; color:{ACCENT};">Live ATM Implied Volatility (IV)</td><td colspan="4" style="text-align:center; font-weight:800; color:{ACCENT}; font-family:var(--font-mono);">{v['iv']:.2f}% (Annual VRP: {v['vrp']:+.2f}% | {_vrp_pct:.0f}th Percentile Rank)</td></tr>
                                    </tbody>
                                </table>
                            </div>
                        </div>'''


                    # ── TAB 3: OPTION CHAIN ANALYSER (renamed + redesigned) ──
                    chain_tab_html = '<div class="card"><p style="color:#888;">Loading option chain data...</p></div>'
                    if seller:
                        s = seller
                        safe_ce = s.get('sell_ce_above', spot + s['em'])
                        safe_pe = s.get('sell_pe_below', spot - s['em'])
                        call_wall_1 = s.get('call_wall', 0)
                        put_wall_1  = s.get('put_wall',  0)
                        call_wall_2 = s.get('call_wall_2', 0)
                        put_wall_2  = s.get('put_wall_2',  0)
                        max_pain_val = s.get('max_pain', 0)
                        pcr_color = GREEN if s['pcr'] > 1.2 else RED if s['pcr'] < 0.8 else YELLOW
                        pcr_label = 'Bullish (Put Writing)' if s['pcr'] > 1.2 else 'Bearish (Call Writing)' if s['pcr'] < 0.8 else 'Neutral'

                        # ── 1. GEX Calculations & Chart upfront ──
                        gex_html = ''
                        gex_analysis_html = ''
                        _net_gex_lots = 0.0
                        _net_gex_crores = 0.0
                        _pin_txt = "PINNING ↔ (Mean-Reverting)"
                        _regime_badge_col = GREEN
                        _flip_strike = 0
                        _flip_dist_txt = 'N/A'
                        _flip_dist_c = MUTED
                        _hedge_action = 'Balanced Dealer Inventory'
                        _hedge_act_txt = 'N/A'
                        _dist_call = abs(spot - float(call_wall_1)) if call_wall_1 else 0
                        _dist_put = abs(spot - float(put_wall_1)) if put_wall_1 else 0

                        if not df_chain.empty:
                            try:
                                _lot = _get_cfg("nifty_lot_size", 65)
                                _T_gex = self.analytics.get_time_to_expiry(near_exp) or (1 / 365)
                                
                                from calculations.GexEngine import GexEngine
                                gex_eng = GexEngine(lot_size=_lot, positioning_model='standard')
                                _df_chain_input = df_chain.copy()
                                if 'dte' not in _df_chain_input.columns:
                                    _df_chain_input['dte'] = max(_T_gex * 365.0, 0.5)
                                gex_res = gex_eng.calculate_gex(_df_chain_input, spot)

                                _net_gex = float(gex_res.get('net_gex', 0.0))
                                self._last_net_gex = _net_gex
                                self._last_dte = max(_T_gex * 365.0, 0.0)
                                _net_gex_crores = _net_gex / 1e7
                                _flip_strike = round(float(gex_res.get('zero_gamma_level', 0.0)))

                                # Sync walls with GexEngine
                                _cw_res = gex_res.get('call_wall', 0.0)
                                _pw_res = gex_res.get('put_wall', 0.0)
                                if _cw_res: call_wall_1 = _cw_res
                                if _pw_res: put_wall_1 = _pw_res
                                _dist_call = abs(spot - float(call_wall_1)) if call_wall_1 else 0
                                _dist_put = abs(spot - float(put_wall_1)) if put_wall_1 else 0

                                _is_long_gamma = (_net_gex >= 0)
                                _pin_txt = "PINNING ↔ (Mean-Reverting)" if _is_long_gamma else "ACCELERATING ↕ (Breakout Risk)"
                                _regime_badge_col = GREEN if _is_long_gamma else RED
                                _hedge_action = f"Dealers Long GEX (+₹{abs(_net_gex_crores):.1f} Cr) — Dampens Volatility" if _net_gex >= 0 else f"Dealers Short GEX (-₹{abs(_net_gex_crores):.1f} Cr) — Accelerates Breakouts"

                                # Strike distribution within ±550 band in ₹ Crores
                                prof_dict = gex_res.get('profile', {})
                                _gex_band_lo, _gex_band_hi = spot - 550, spot + 550

                                strike_keys = sorted([s for s in prof_dict.keys() if _gex_band_lo <= s <= _gex_band_hi])
                                if not strike_keys:
                                    strike_keys = sorted(list(prof_dict.keys()))

                                _strikes_list = [float(s) for s in strike_keys]
                                _gex_cr_list = [float(prof_dict[s]) / 1e7 for s in strike_keys]
                                _gex_colors = [GREEN if g >= 0 else RED for g in _gex_cr_list]

                                _s_min = min(_strikes_list) if _strikes_list else spot - 500
                                _s_max = max(_strikes_list) if _strikes_list else spot + 500

                                fig_gex = go.Figure(go.Bar(
                                    x=_gex_cr_list,
                                    y=_strikes_list,
                                    orientation='h',
                                    marker=dict(color=_gex_colors, line=dict(width=1, color='rgba(255,255,255,0.15)')),
                                    text=[f"{'+' if g>=0 else ''}{g:.1f} Cr" for g in _gex_cr_list],
                                    textposition='outside',
                                    textfont=dict(color=WHITE, size=10),
                                    hovertemplate='<b>Strike: %{y}</b><br>Net GEX: %{x:+.2f} Cr<extra></extra>',
                                ))

                                # 1. Pinning Corridor Shaded Area (Put Wall 1 to Call Wall 1)
                                if call_wall_1 and put_wall_1 and call_wall_1 > put_wall_1:
                                    fig_gex.add_hrect(
                                        y0=max(_s_min, float(put_wall_1)), y1=min(_s_max, float(call_wall_1)),
                                        fillcolor='rgba(0, 229, 255, 0.05)',
                                        line_width=1, line_color='rgba(0, 229, 255, 0.25)', line_dash='dot',
                                        layer='below'
                                    )

                                # 2. 1-Sigma Expected Move Shaded Area
                                _em_val = float(s.get('em', 0) or 0)
                                if _em_val > 0 and _strikes_list:
                                    fig_gex.add_hrect(
                                        y0=max(_s_min, spot - _em_val), y1=min(_s_max, spot + _em_val),
                                        fillcolor='rgba(255, 214, 0, 0.035)',
                                        line_width=1, line_color='rgba(255, 214, 0, 0.2)', line_dash='dash',
                                        layer='below'
                                    )

                                # 3. Key Institutional Wall Lines
                                if call_wall_1 and (_s_min <= call_wall_1 <= _s_max):
                                    fig_gex.add_hline(
                                        y=float(call_wall_1), line_dash='dot', line_color='#ff3366', line_width=1.8,
                                        annotation_text=f"🔴 CALL WALL ①: {call_wall_1}",
                                        annotation_position="top right",
                                        annotation_font=dict(color='#ff3366', size=10, family='Inter, sans-serif'),
                                        annotation_bgcolor='rgba(13,17,38,0.9)'
                                    )
                                if put_wall_1 and (_s_min <= put_wall_1 <= _s_max):
                                    fig_gex.add_hline(
                                        y=float(put_wall_1), line_dash='dot', line_color='#00e676', line_width=1.8,
                                        annotation_text=f"🟢 PUT WALL ①: {put_wall_1}",
                                        annotation_position="bottom right",
                                        annotation_font=dict(color='#00e676', size=10, family='Inter, sans-serif'),
                                        annotation_bgcolor='rgba(13,17,38,0.9)'
                                    )
                                if call_wall_2 and (_s_min <= call_wall_2 <= _s_max) and call_wall_2 != call_wall_1:
                                    fig_gex.add_hline(
                                        y=float(call_wall_2), line_dash='dot', line_color='#ff7043', line_width=1.2,
                                        annotation_text=f"▲ CW ②: {call_wall_2} (Squeeze Target)",
                                        annotation_position="top right",
                                        annotation_font=dict(color='#ff7043', size=9, family='Inter, sans-serif'),
                                        annotation_bgcolor='rgba(13,17,38,0.9)'
                                    )
                                if put_wall_2 and (_s_min <= put_wall_2 <= _s_max) and put_wall_2 != put_wall_1:
                                    fig_gex.add_hline(
                                        y=float(put_wall_2), line_dash='dot', line_color='#81c784', line_width=1.2,
                                        annotation_text=f"▼ PW ②: {put_wall_2} (Cascade Target)",
                                        annotation_position="bottom right",
                                        annotation_font=dict(color='#81c784', size=9, family='Inter, sans-serif'),
                                        annotation_bgcolor='rgba(13,17,38,0.9)'
                                    )
                                if _flip_strike and (_s_min <= _flip_strike <= _s_max):
                                    fig_gex.add_hline(
                                        y=float(_flip_strike), line_dash='dashdot', line_color='#ffea00', line_width=1.5,
                                        annotation_text=f"⚡ ZERO-GAMMA FLIP: {_flip_strike}",
                                        annotation_position="top left",
                                        annotation_font=dict(color='#ffea00', size=10, family='Inter, sans-serif'),
                                        annotation_bgcolor='rgba(13,17,38,0.9)'
                                    )
                                fig_gex.add_hline(
                                    y=float(spot), line_dash='dash', line_color='#00e5ff', line_width=2.2,
                                    annotation_text=f"● LIVE SPOT: {spot:.1f}",
                                    annotation_position="top right",
                                    annotation_font=dict(color='#00e5ff', size=11, family='Inter, sans-serif'),
                                    annotation_bgcolor='rgba(13,17,38,0.9)'
                                )
                                fig_gex.update_layout(
                                    height=520,
                                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                                    font=dict(color=WHITE, family='Inter, sans-serif', size=11),
                                    margin=dict(l=65, r=85, t=50, b=40),
                                    title=dict(
                                        text=f'GEX DISTRIBUTION & Spot Movement Trajectory | Net GEX: {"+" if _net_gex_crores>=0 else ""}{_net_gex_crores:+.1f} Cr',
                                        font=dict(color=ACCENT, size=13), x=0.01),
                                    xaxis=dict(gridcolor='rgba(255,255,255,0.05)', title='Net GEX (₹ Crores)', showline=True, linecolor='rgba(255,255,255,0.1)', zeroline=True, zerolinecolor='rgba(255,255,255,0.25)', zerolinewidth=1.5),
                                    yaxis=dict(
                                        range=[_s_min - 25, _s_max + 25],
                                        autorange=False,
                                        tickmode='array', tickvals=_strikes_list, ticktext=[str(int(s)) for s in _strikes_list],
                                        gridcolor='rgba(255,255,255,0.05)', title='Strike Price', showline=True, linecolor='rgba(255,255,255,0.1)'),
                                    bargap=0.18,
                                 )
                                gex_html = fig_gex.to_html(include_plotlyjs=False, full_html=False, div_id='gex-distribution-chart', default_height='520px', default_width='100%')

                                # Movement trajectory synthesis
                                _cw2_txt = f" → CW ② ({call_wall_2})" if call_wall_2 else ""
                                _pw2_txt = f" → PW ② ({put_wall_2})" if put_wall_2 else ""
                                _traj_summary = (
                                    f"Dealer hedging dampens volatility within <strong>{put_wall_1} – {call_wall_1}</strong> pinning corridor. "
                                    f"Upside breakout &gt; {call_wall_1+25:.0f} triggers dealer short squeeze{_cw2_txt}. "
                                    f"Downside breakdown &lt; {put_wall_1-25:.0f} flips dealer gamma, triggering cascade risk{_pw2_txt}."
                                ) if _is_long_gamma else (
                                    f"Dealer short gamma accelerates price momentum away from <strong>{_flip_strike or 'flip level'}</strong>. "
                                    f"Upside squeeze target: <strong>{call_wall_1}</strong> (then {call_wall_2 or 'higher'}). "
                                    f"Downside cascade target: <strong>{put_wall_1}</strong> (then {put_wall_2 or 'lower'})."
                                )

                                # Strike zone contextual analysis
                                _cw_gex_hint = f"Dealers short calls (sell rallies) · Squeeze trigger > {call_wall_1+25:.0f}"
                                _atm_gex_hint = f"Max gamma concentration · Magnet attraction toward Max Pain ({max_pain_val})"
                                _pw_gex_hint = f"Dealers long puts (buy dips) · Waterfall breakdown < {put_wall_1-25:.0f}"
                                _flip_hint = f"Inflection threshold: Spot {'ABOVE' if spot > (_flip_strike or 0) else 'BELOW'} flip ({_flip_strike}) · {'Stabilizing' if _is_long_gamma else 'Breakout Acceleration'}"

                                gex_analysis_html = f'''
                                <div style="margin-top:12px;padding:14px 16px;background:rgba(15,15,25,0.75);border-radius:8px;border:1px solid rgba(0,229,255,0.2);">
                                    <!-- Header Strip -->
                                    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px;flex-wrap:wrap;gap:8px;">
                                        <div style="color:{ACCENT};font-size:12px;font-weight:900;letter-spacing:1px;display:flex;align-items:center;gap:6px;">
                                            <span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:{ACCENT};box-shadow:0 0 8px {ACCENT};"></span>
                                            SPOT MOVEMENT TRAJECTORY &amp; REAL-TIME INTRADAY PROJECTION
                                        </div>
                                        <div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap;">
                                            <span id="spot-move-regime-badge" style="font-size:10px;font-weight:800;color:{_regime_badge_col};background:{_regime_badge_col}18;padding:3px 10px;border-radius:4px;border:1px solid {_regime_badge_col}44;">{_pin_txt}</span>
                                            <span id="live-net-gex-badge" style="font-size:10px;font-weight:800;color:{GREEN if _net_gex_crores>=0 else RED};background:rgba(255,255,255,0.06);padding:3px 10px;border-radius:4px;border:1px solid {GREEN if _net_gex_crores>=0 else RED};">Net GEX: {f"{_net_gex_crores:+.1f} Cr"}</span>
                                        </div>
                                    </div>

                                    <!-- 4 Landmark Boundary Metric Boxes -->
                                    <div style="display:grid;grid-template-columns:repeat(auto-fit, minmax(200px, 1fr));gap:8px;margin-bottom:12px;">
                                        <div class="metric-box" style="padding:8px 10px;border-left:2px solid #00e676;">
                                            <div class="metric-label" style="color:#00e676;font-size:10px;font-weight:700;">🟢 PUT WALL ① (Demand Floor)</div>
                                            <div id="spot-move-put-wall" style="font-size:16px;font-weight:800;color:{WHITE};">{put_wall_1}</div>
                                            <div id="spot-move-put-dist" class="metric-sub" style="color:#00e676;font-size:10px;">-{_dist_put:.0f} pts · Cascade &lt; {put_wall_1-25:.0f}{_pw2_txt}</div>
                                        </div>
                                        <div class="metric-box" style="padding:8px 10px;border-left:2px solid #ff3366;">
                                            <div class="metric-label" style="color:#ff3366;font-size:10px;font-weight:700;">🔴 CALL WALL ① (Supply Ceiling)</div>
                                            <div id="spot-move-call-wall" style="font-size:16px;font-weight:800;color:{WHITE};">{call_wall_1}</div>
                                            <div id="spot-move-call-dist" class="metric-sub" style="color:#ff3366;font-size:10px;">+{_dist_call:.0f} pts · Squeeze &gt; {call_wall_1+25:.0f}{_cw2_txt}</div>
                                        </div>
                                        <div class="metric-box" style="padding:8px 10px;border-left:2px solid #ffd54f;">
                                            <div class="metric-label" style="color:#ffd54f;font-size:10px;font-weight:700;">⚡ ZERO-GAMMA FLIP LEVEL</div>
                                            <div id="spot-move-flip-strike" style="font-size:16px;font-weight:800;color:{WHITE};">{_flip_strike or 'N/A'}</div>
                                            <div id="spot-move-flip-dist" class="metric-sub" style="color:{_flip_dist_c};font-size:10px;">{_flip_dist_txt} ({'Long Gamma Pinning' if _is_long_gamma else 'Short Gamma Breakout'})</div>
                                        </div>
                                        <div class="metric-box" style="padding:8px 10px;border-left:2px solid #00e5ff;">
                                            <div class="metric-label" style="color:#00e5ff;font-size:10px;font-weight:700;">🎯 1-SIGMA EXPECTED MOVE</div>
                                            <div id="spot-move-expected-move" style="font-size:16px;font-weight:800;color:{WHITE};">±{_em_val:.0f} pts</div>
                                            <div id="spot-move-expected-range" class="metric-sub" style="color:#94a3b8;font-size:10px;">Range: {spot - _em_val:.0f} – {spot + _em_val:.0f} | Pain: {max_pain_val}</div>
                                        </div>
                                    </div>

                                    <!-- Real-Time Intraday Projection & Cross-Strike Dynamics Matrix -->
                                    <div style="display:grid;grid-template-columns:repeat(auto-fit, minmax(320px, 1fr));gap:10px;margin-bottom:10px;">
                                        <!-- Panel 1: Real-Time Intraday Projection -->
                                        <div style="padding:10px 12px;background:rgba(0,0,0,0.35);border-radius:6px;border:1px solid rgba(255,255,255,0.06);">
                                            <div style="font-size:11px;font-weight:800;color:#00e5ff;margin-bottom:6px;display:flex;justify-content:space-between;">
                                                <span>⚡ INTRADAY PROJECTION ENGINE</span>
                                                <span id="proj-bias-badge" style="color:{GREEN if _is_long_gamma else RED};font-size:10px;">{'PIN / MEAN-REVERTING' if _is_long_gamma else 'DIRECTIONAL EXPANSION'}</span>
                                            </div>
                                            <div style="font-size:11px;color:#ccc;line-height:1.5;">
                                                <div>• Projected Corridor: <strong id="proj-corridor" style="color:#ffffff;">{put_wall_1} — {call_wall_1}</strong> ({call_wall_1 - put_wall_1:.0f} pts)</div>
                                                <div>• Primary Magnet Target: <strong id="proj-target" style="color:#ffd54f;">{max_pain_val} (Max Pain)</strong></div>
                                                <div>• Dealer Hedging Flow: <span id="proj-flow" style="color:{_regime_badge_col};font-weight:700;">{_hedge_act_txt}</span> per 50pt Nifty advance</div>
                                                <div>• Squeeze Acceleration Trigger: <span id="proj-trigger" style="color:#ff3366;font-weight:700;">&gt; {call_wall_1+25:.0f}</span> (Upside) / <span style="color:#00e676;font-weight:700;">&lt; {put_wall_1-25:.0f}</span> (Downside)</div>
                                            </div>
                                        </div>

                                        <!-- Panel 2: What's Happening Across Strikes & What It Means -->
                                        <div style="padding:10px 12px;background:rgba(0,0,0,0.35);border-radius:6px;border:1px solid rgba(255,255,255,0.06);">
                                            <div style="font-size:11px;font-weight:800;color:#ffd54f;margin-bottom:6px;">
                                                🔍 REAL-TIME STRIKE DYNAMICS &amp; MEANING
                                            </div>
                                            <div style="font-size:11px;color:#bbb;line-height:1.45;">
                                                <div style="margin-bottom:4px;">
                                                    <span style="color:#ff6688;font-weight:700;">Call Wall ({call_wall_1}):</span> {_cw_gex_hint}
                                                </div>
                                                <div style="margin-bottom:4px;">
                                                    <span style="color:#00e5ff;font-weight:700;">ATM ({round(spot/50)*50}):</span> {_atm_gex_hint}
                                                </div>
                                                <div style="margin-bottom:4px;">
                                                    <span style="color:#00e676;font-weight:700;">Put Wall ({put_wall_1}):</span> {_pw_gex_hint}
                                                </div>
                                                <div>
                                                    <span style="color:#ffd54f;font-weight:700;">Flip Line ({_flip_strike or 'N/A'}):</span> {_flip_hint}
                                                </div>
                                            </div>
                                        </div>
                                    </div>

                                    <!-- Bottom Outlook Strip -->
                                    <div style="padding:8px 12px;background:rgba(0,229,255,0.06);border-radius:6px;border:1px solid rgba(0,229,255,0.15);font-size:11px;color:#cbd5e1;line-height:1.5;">
                                        <span style="color:#00e5ff;font-weight:800;margin-right:4px;">PREDICTIVE OUTLOOK:</span>
                                        <span id="spot-move-outlook-text">{_traj_summary}</span>
                                    </div>
                                </div>'''
                            except Exception as e:
                                print(f"GEX Error: {e}")

                        # ── 2. Master Option Chain Table Rows ──
                        max_ce_oi = max([r.get('ce_oi', 0) for r in s['chain_rows']] + [1])
                        max_pe_oi = max([r.get('pe_oi', 0) for r in s['chain_rows']] + [1])

                        oi_rows_html = ''
                        for r in s['chain_rows']:
                            is_cw1 = (r['strike'] == call_wall_1 and call_wall_1 > 0)
                            is_pw1 = (r['strike'] == put_wall_1  and put_wall_1 > 0)
                            is_cw2 = (r['strike'] == call_wall_2 and call_wall_2 > 0)
                            is_pw2 = (r['strike'] == put_wall_2  and put_wall_2 > 0)
                            is_mp  = (r['strike'] == max_pain_val and max_pain_val > 0)
                            is_atm = r.get('is_atm', False)

                            if is_cw1:
                                row_style = "background:rgba(255,51,102,0.18);box-shadow:inset 0 0 12px rgba(255,51,102,0.3);border-top:1.5px solid #ff3366;border-bottom:1.5px solid #ff3366;"
                            elif is_pw1:
                                row_style = "background:rgba(0,230,118,0.18);box-shadow:inset 0 0 12px rgba(0,230,118,0.3);border-top:1.5px solid #00e676;border-bottom:1.5px solid #00e676;"
                            elif is_atm:
                                row_style = "background:rgba(0,229,255,0.12);box-shadow:inset 0 0 8px rgba(0,229,255,0.25);border-top:1.5px solid #00e5ff;border-bottom:1.5px solid #00e5ff;"
                            elif is_cw2:
                                row_style = "background:rgba(255,51,102,0.08);border-top:1px solid rgba(255,51,102,0.25);border-bottom:1px solid rgba(255,51,102,0.25);"
                            elif is_pw2:
                                row_style = "background:rgba(0,230,118,0.08);border-top:1px solid rgba(0,230,118,0.25);border-bottom:1px solid rgba(0,230,118,0.25);"
                            elif is_mp:
                                row_style = "background:rgba(255,214,0,0.08);"
                            else:
                                row_style = "background:transparent;border-bottom:1px solid rgba(255,255,255,0.04);"

                            ce_itm = (r['strike'] < spot)
                            pe_itm = (r['strike'] > spot)
                            ce_cell_bg = "rgba(255,255,255,0.025)" if ce_itm else "transparent"
                            pe_cell_bg = "rgba(255,255,255,0.025)" if pe_itm else "transparent"

                            ce_oi_pct = min(100, int((r['ce_oi'] / max_ce_oi) * 100))
                            pe_oi_pct = min(100, int((r['pe_oi'] / max_pe_oi) * 100))
                            ce_oi_bg = f"background:linear-gradient(to left, rgba(255,82,82,0.25) {ce_oi_pct}%, {ce_cell_bg} {ce_oi_pct}%);"
                            pe_oi_bg = f"background:linear-gradient(to right, rgba(0,230,118,0.25) {pe_oi_pct}%, {pe_cell_bg} {pe_oi_pct}%);"

                            ce_v = f"+{r['ce_vel']:,}" if r['ce_vel'] > 0 else f"{r['ce_vel']:,}" if r['ce_vel'] < 0 else "·"
                            pe_v = f"+{r['pe_vel']:,}" if r['pe_vel'] > 0 else f"{r['pe_vel']:,}" if r['pe_vel'] < 0 else "·"
                            ce_v_col = GREEN if r['ce_vel'] > 0 else RED if r['ce_vel'] < 0 else MUTED
                            pe_v_col = GREEN if r['pe_vel'] > 0 else RED if r['pe_vel'] < 0 else MUTED

                            ce_gex_val = r.get('ce_gex_cr', 0.0)
                            pe_gex_val = r.get('pe_gex_cr', 0.0)
                            ce_gex_txt = f"{ce_gex_val:+.1f} Cr" if abs(ce_gex_val) >= 0.05 else "·"
                            pe_gex_txt = f"{pe_gex_val:+.1f} Cr" if abs(pe_gex_val) >= 0.05 else "·"
                            ce_gex_col = GREEN if ce_gex_val > 0 else RED if ce_gex_val < 0 else MUTED
                            pe_gex_col = RED if pe_gex_val < 0 else GREEN if pe_gex_val > 0 else MUTED

                            ce_price_txt = f"₹{r['ce_price']:.2f}" if r.get('ce_price', 0) > 0 else "·"
                            pe_price_txt = f"₹{r['pe_price']:.2f}" if r.get('pe_price', 0) > 0 else "·"
                            ce_iv_txt = f"{r['ce_iv']:.1f}%" if r.get('ce_iv', 0) > 0 else "·"
                            pe_iv_txt = f"{r['pe_iv']:.1f}%" if r.get('pe_iv', 0) > 0 else "·"

                            dist_from_spot = r['strike'] - spot
                            dist_label = f"+{dist_from_spot:.0f}" if dist_from_spot > 0 else f"{dist_from_spot:.0f}"

                            strike_badges = []
                            if is_atm:
                                strike_badges.append(f'<span style="background:rgba(0,229,255,0.25);color:#00e5ff;font-size:9px;font-weight:900;padding:2px 6px;border-radius:3px;border:1px solid #00e5ff;">◄ ATM</span>')
                            if is_cw1:
                                strike_badges.append(f'<span style="background:rgba(255,51,102,0.3);color:#ff3366;font-size:9px;font-weight:900;padding:2px 6px;border-radius:3px;border:1px solid #ff3366;box-shadow:0 0 8px #ff3366;">🔴 CALL WALL ①</span>')
                            elif is_cw2:
                                strike_badges.append(f'<span style="background:rgba(255,51,102,0.15);color:#ff6688;font-size:9px;font-weight:700;padding:1px 5px;border-radius:3px;border:1px solid rgba(255,51,102,0.4);">CALL WALL ②</span>')
                            if is_pw1:
                                strike_badges.append(f'<span style="background:rgba(0,230,118,0.3);color:#00e676;font-size:9px;font-weight:900;padding:2px 6px;border-radius:3px;border:1px solid #00e676;box-shadow:0 0 8px #00e676;">🟢 PUT WALL ①</span>')
                            elif is_pw2:
                                strike_badges.append(f'<span style="background:rgba(0,230,118,0.15);color:#55e088;font-size:9px;font-weight:700;padding:1px 5px;border-radius:3px;border:1px solid rgba(0,230,118,0.4);">PUT WALL ②</span>')
                            if is_mp and not is_atm:
                                strike_badges.append(f'<span style="background:rgba(255,214,0,0.2);color:#ffd54f;font-size:9px;font-weight:800;padding:2px 6px;border-radius:3px;border:1px solid #ffd54f;">🟡 MAX PAIN</span>')

                            # Seller Greek & Probability metrics for merged option chain
                            ce_potm_val = r.get('ce_prob_otm', 50.0)
                            ce_potm_col = GREEN if ce_potm_val >= 85 else ACCENT if ce_potm_val >= 75 else YELLOW if ce_potm_val >= 60 else RED
                            ce_potm_txt = f"{ce_potm_val:.1f}%" if not ce_itm else "—"

                            ce_sig = r.get('ce_signal', '')
                            ce_sig_col = GREEN if '★' in ce_sig else ACCENT if '✓' in ce_sig else YELLOW if '~' in ce_sig else RED
                            ce_sig_badge = f'<span style="color:{ce_sig_col};font-weight:700;font-size:10px;">{ce_sig}</span>' if not ce_itm else f'<span style="color:#64748b;font-size:10px;">ITM</span>'

                            ce_th_val = r.get('ce_theta', 0.0)
                            ce_th_txt = f"{ce_th_val:.1f}" if ce_th_val != 0 else "·"

                            pe_potm_val = r.get('pe_prob_otm', 50.0)
                            pe_potm_col = GREEN if pe_potm_val >= 85 else ACCENT if pe_potm_val >= 75 else YELLOW if pe_potm_val >= 60 else RED
                            pe_potm_txt = f"{pe_potm_val:.1f}%" if not pe_itm else "—"

                            pe_sig = r.get('pe_signal', '')
                            pe_sig_col = GREEN if '★' in pe_sig else ACCENT if '✓' in pe_sig else YELLOW if '~' in pe_sig else RED
                            pe_sig_badge = f'<span style="color:{pe_sig_col};font-weight:700;font-size:10px;">{pe_sig}</span>' if not pe_itm else f'<span style="color:#64748b;font-size:10px;">ITM</span>'

                            pe_th_val = r.get('pe_theta', 0.0)
                            pe_th_txt = f"{pe_th_val:.1f}" if pe_th_val != 0 else "·"

                            strike_badges_html = " ".join(strike_badges)
                            strike_main_col = "#00e5ff" if is_atm else RED if (is_cw1 or is_cw2) else GREEN if (is_pw1 or is_pw2) else YELLOW if is_mp else WHITE
                            atm_attr = ' id="row-atm" class="glow-atm" data-is-atm="true"' if is_atm else ''

                            oi_rows_html += (
                                f'<tr data-strike="{r["strike"]}"{atm_attr} style="{row_style}">'
                                f'<td class="col-seller" style="text-align:center;padding:7px 6px;background:{ce_cell_bg};">{ce_sig_badge}</td>'
                                f'<td class="col-seller" style="text-align:right;padding:7px 6px;font-family:monospace;color:#ffab91;background:{ce_cell_bg};font-size:11px;">{ce_th_txt}</td>'
                                f'<td class="col-gex" data-call-gex="{r["strike"]}" style="text-align:right;padding:7px 8px;font-family:monospace;color:{ce_gex_col};background:{ce_cell_bg};font-weight:700;">{ce_gex_txt}</td>'
                                f'<td class="col-gex" style="text-align:right;padding:7px 8px;font-family:monospace;color:{WHITE};{ce_oi_bg};font-weight:600;">{r["ce_oi"]:,}</td>'
                                f'<td class="col-base" style="text-align:right;padding:7px 8px;font-family:monospace;color:{ce_v_col};background:{ce_cell_bg};font-weight:600;">{ce_v}</td>'
                                f'<td class="col-base" style="text-align:right;padding:7px 8px;font-family:monospace;color:{WHITE};background:{ce_cell_bg};font-weight:700;">{ce_price_txt}</td>'
                                f'<td class="col-base" style="text-align:right;padding:7px 8px;font-family:monospace;color:#ffd54f;background:{ce_cell_bg};font-size:11px;">{ce_iv_txt}</td>'
                                f'<td class="col-strike strike-cell" data-strike="{r["strike"]}" style="text-align:center;padding:7px 12px;font-weight:900;background:rgba(18,22,46,0.9);border-left:1px solid rgba(255,255,255,0.08);border-right:1px solid rgba(255,255,255,0.08);">'
                                f'  <div style="display:flex;align-items:center;justify-content:center;gap:6px;flex-wrap:wrap;">'
                                f'    <span style="font-size:14px;color:{strike_main_col};">{r["strike"]}</span>'
                                f'    <span style="font-size:10px;color:#64748b;">({dist_label})</span>'
                                f'    {strike_badges_html}'
                                f'  </div>'
                                f'</td>'
                                f'<td class="col-base" style="text-align:left;padding:7px 8px;font-family:monospace;color:#ffd54f;background:{pe_cell_bg};font-size:11px;">{pe_iv_txt}</td>'
                                f'<td class="col-base" style="text-align:left;padding:7px 8px;font-family:monospace;color:{WHITE};background:{pe_cell_bg};font-weight:700;">{pe_price_txt}</td>'
                                f'<td class="col-base" style="text-align:right;padding:7px 8px;font-family:monospace;color:{pe_v_col};background:{pe_cell_bg};font-weight:600;">{pe_v}</td>'
                                f'<td class="col-gex" style="text-align:right;padding:7px 8px;font-family:monospace;color:{WHITE};{pe_oi_bg};font-weight:600;">{r["pe_oi"]:,}</td>'
                                f'<td class="col-gex" data-put-gex="{r["strike"]}" style="text-align:right;padding:7px 8px;font-family:monospace;color:{pe_gex_col};background:{pe_cell_bg};font-weight:700;">{pe_gex_txt}</td>'
                                f'<td class="col-seller" style="text-align:right;padding:7px 6px;font-family:monospace;color:#ffab91;background:{pe_cell_bg};font-size:11px;">{pe_th_txt}</td>'
                                f'<td class="col-seller" style="text-align:center;padding:7px 6px;background:{pe_cell_bg};">{pe_sig_badge}</td>'
                                f'</tr>'
                            )

                        # Keep strike selection rows for standalone references
                        strike_rows_html = ''
                        for sr in s.get('strike_rows', []):
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

                        chain_tab_html = f'''
                        <!-- 1. KEY METRICS & SELL ZONES -->
                        <div class="card" style="margin-bottom:12px;">
                            <div style="color:{ACCENT};font-size:13px;font-weight:700;letter-spacing:2px;margin-bottom:14px;">KEY METRICS &amp; SELL ZONES — DTE {s['DTE']} | {near_exp}</div>
                            <div style="display:flex;gap:10px;flex-wrap:wrap;margin-bottom:14px;">
                                <div class="metric-box"><div class="metric-label">ATM IV</div><div style="font-size:20px;font-weight:700;color:{WHITE};">{s['atm_iv']:.2f}%</div></div>
                                <div class="metric-box"><div class="metric-label">Straddle</div><div style="font-size:20px;font-weight:700;color:{WHITE};">{s['straddle']:.0f}</div></div>
                                <div class="metric-box"><div class="metric-label">Exp. Move</div><div style="font-size:20px;font-weight:700;color:{WHITE};">±{s['em']:.0f} ({s['em_pct']:.1f}%)</div></div>
                                <div class="metric-box"><div class="metric-label">Max Pain</div><div style="font-size:20px;font-weight:700;color:{YELLOW};">{max_pain_val}</div></div>
                                <div class="metric-box" id="chain-net-gex-box" style="border-left:2px solid {'#00e676' if _net_gex_crores>=0 else '#ff3366'};">
                                    <div class="metric-label">LIVE NET GEX</div>
                                    <div id="chain-net-gex-val" style="font-size:20px;font-weight:700;color:{'#00e676' if _net_gex_crores>=0 else '#ff3366'};">{'+' if _net_gex_crores>=0 else ''}{_net_gex_crores:+.1f} Cr</div>
                                    <div class="metric-sub" id="chain-gex-regime-sub">{'Long Gamma Pin' if _net_gex_crores>=0 else 'Short Gamma Trend'}</div>
                                </div>
                                <div class="metric-box" style="border-left:2px solid {GREEN};">
                                    <div class="metric-label">Put Wall ①</div>
                                    <div style="font-size:20px;font-weight:700;color:{GREEN};">{put_wall_1}</div>
                                    {f'<div class="metric-sub" style="color:#66bb6a88;">② {put_wall_2}</div>' if put_wall_2 else ''}
                                </div>
                                <div class="metric-box" style="border-left:2px solid {RED};">
                                    <div class="metric-label">Call Wall ①</div>
                                    <div style="font-size:20px;font-weight:700;color:{RED};">{call_wall_1}</div>
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

                        <!-- 2. MASTER OPTION CHAIN & GEX WALL MATRIX (WITH INTEGRATED STRIKE SELECTION) -->
                        <div class="card" style="margin-bottom:14px;padding:0;overflow:hidden;border:1px solid rgba(0,229,255,0.2);border-radius:10px;box-shadow:0 6px 24px rgba(0,0,0,0.4);">
                            <div style="display:flex;justify-content:space-between;align-items:center;padding:12px 18px;background:rgba(18,22,46,0.95);border-bottom:1px solid rgba(255,255,255,0.08);flex-wrap:wrap;gap:8px;">
                                <div style="display:flex;align-items:center;gap:10px;">
                                    <div style="font-size:14px;font-weight:900;letter-spacing:1.5px;color:#00e5ff;">MASTER OPTION CHAIN &amp; SELLER MATRIX</div>
                                    <span style="font-size:10px;background:rgba(0,229,255,0.12);color:#00e5ff;padding:2px 8px;border-radius:4px;font-weight:700;">LIVE NIFTY · EXPIRY {near_exp}</span>
                                </div>
                                <div style="display:flex;gap:6px;align-items:center;flex-wrap:wrap;">
                                    <span style="font-size:10px;color:#64748b;font-weight:700;margin-right:4px;">VIEW:</span>
                                    <button type="button" id="btn-chain-all" class="chain-mode-btn active" onclick="setChainMode('all')" style="padding:4px 10px;font-size:10px;font-weight:800;border-radius:4px;background:rgba(0,229,255,0.15);color:#00e5ff;border:1px solid #00e5ff;cursor:pointer;">COMPREHENSIVE</button>
                                    <button type="button" id="btn-chain-seller" class="chain-mode-btn" onclick="setChainMode('seller')" style="padding:4px 10px;font-size:10px;font-weight:800;border-radius:4px;background:rgba(255,255,255,0.05);color:#94a3b8;border:1px solid #333a60;cursor:pointer;">PRO SELLER</button>
                                    <button type="button" id="btn-chain-gex" class="chain-mode-btn" onclick="setChainMode('gex')" style="padding:4px 10px;font-size:10px;font-weight:800;border-radius:4px;background:rgba(255,255,255,0.05);color:#94a3b8;border:1px solid #333a60;cursor:pointer;">CLASSIC GEX</button>
                                </div>
                                <div style="display:flex;gap:12px;font-size:11px;font-family:'JetBrains Mono',monospace;flex-wrap:wrap;">
                                    <span style="color:#ff3366;">🔴 CALL WALL / RESISTANCE</span>
                                    <span style="color:#00e676;">🟢 PUT WALL / SUPPORT</span>
                                    <span style="color:#00e5ff;">🔵 ATM PIVOT</span>
                                    <span style="color:#ffd54f;">🟡 MAX PAIN PIN</span>
                                </div>
                            </div>
                            <div id="master-chain-container" style="max-height:560px;overflow-y:auto;">
                                <table id="master-chain-table" class="data-table" style="width:100%;margin:0;border-collapse:collapse;font-size:12px;">
                                    <thead style="position:sticky;top:0;z-index:10;background:#0d1124;box-shadow:0 2px 8px rgba(0,0,0,0.7);">
                                        <tr style="border-bottom:1px solid #222744;">
                                            <th id="th-calls-header" colspan="7" style="text-align:center;color:#ff5252;background:rgba(255,51,102,0.1);letter-spacing:1px;font-size:11px;padding:6px;">CALLS (RESISTANCE / PREMIUM SELLERS)</th>
                                            <th style="text-align:center;color:#00e5ff;background:rgba(0,229,255,0.12);letter-spacing:1px;font-size:11px;padding:6px;">STRIKE</th>
                                            <th id="th-puts-header" colspan="7" style="text-align:center;color:#00e676;background:rgba(0,230,118,0.1);letter-spacing:1px;font-size:11px;padding:6px;">PUTS (SUPPORT / PREMIUM SELLERS)</th>
                                        </tr>
                                        <tr style="border-bottom:1px solid #333a60;font-size:11px;color:#94a3b8;">
                                            <th class="col-seller" style="text-align:center;padding:7px 6px;">Signal</th>
                                            <th class="col-seller" style="text-align:right;padding:7px 6px;">Theta</th>
                                            <th class="col-gex" style="text-align:right;padding:7px 8px;">Call GEX</th>
                                            <th class="col-gex" style="text-align:right;padding:7px 8px;">Call OI</th>
                                            <th class="col-base" style="text-align:right;padding:7px 8px;">15m Vel</th>
                                            <th class="col-base" style="text-align:right;padding:7px 8px;">Call LTP</th>
                                            <th class="col-base" style="text-align:right;padding:7px 8px;">IV</th>
                                            <th class="col-strike" style="text-align:center;padding:7px 12px;color:#fff;font-weight:800;">Strike Price</th>
                                            <th class="col-base" style="text-align:left;padding:7px 8px;">IV</th>
                                            <th class="col-base" style="text-align:left;padding:7px 8px;">Put LTP</th>
                                            <th class="col-base" style="text-align:right;padding:7px 8px;">15m Vel</th>
                                            <th class="col-gex" style="text-align:right;padding:7px 8px;">Put OI</th>
                                            <th class="col-gex" style="text-align:right;padding:7px 8px;">Put GEX</th>
                                            <th class="col-seller" style="text-align:right;padding:7px 6px;">Theta</th>
                                            <th class="col-seller" style="text-align:center;padding:7px 6px;">Signal</th>
                                        </tr>
                                    </thead>
                                    <tbody id="master-chain-tbody">
                                        {oi_rows_html}
                                    </tbody>
                                </table>
                            </div>
                        </div>

                        <!-- 3. GEX DISTRIBUTION BAR CHART & VOLATILITY CONTOURS -->
                        <div class="card" style="margin-bottom:14px;padding:16px 20px;border:1px solid rgba(0,229,255,0.25);border-radius:10px;box-shadow:0 6px 24px rgba(0,0,0,0.4);">
                            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px;flex-wrap:wrap;gap:8px;">
                                <div style="display:flex;align-items:center;gap:8px;">
                                    <div style="width:8px;height:8px;border-radius:50%;background:#00e5ff;box-shadow:0 0 8px #00e5ff;"></div>
                                    <div style="font-size:13px;font-weight:900;letter-spacing:1.5px;color:#00e5ff;">
                                        GEX DISTRIBUTION &amp; VOLATILITY CONTOURS
                                    </div>
                                </div>
                                <span style="font-size:11px;color:#94a3b8;font-family:'JetBrains Mono',monospace;">
                                    Full Width Contours · Strike Selection Merged in Master Chain
                                </span>
                            </div>
                            <!-- Real-Time Intraday Projection & Strike Dynamics Panel -->
                            {gex_analysis_html}

                            <div id="gex-chart-container" style="min-height:540px; height:540px; width:100%; position:relative; overflow:hidden; margin-top:14px;">
                                {gex_html}
                            </div>
                        </div>

                        <!-- 5. OPTION BUYER RADAR -->
                        <div id="gex-rebalance-card" class="card" style="margin-bottom:20px;border:1px solid #1e2438;background:#0d111e;border-radius:12px;padding:24px;box-shadow:0 4px 20px rgba(0,0,0,0.25);">
                            <!-- Top Header: Title, Subtitle, Badges -->
                            <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:12px;margin-bottom:20px;border-bottom:1px solid #1e2438;padding-bottom:16px;">
                                <div style="display:flex;align-items:center;gap:12px;">
                                    <div style="width:8px;height:8px;border-radius:50%;background:#38bdf8;"></div>
                                    <div>
                                        <div style="font-size:15px;font-weight:800;color:#f8fafc;letter-spacing:0.5px;display:flex;align-items:center;gap:10px;">
                                            <span>Option Buyer Radar</span>
                                            <span style="font-size:11px;color:#94a3b8;background:#171d30;border:1px solid #232b45;padding:2px 8px;border-radius:4px;font-weight:600;">Breakout Engine</span>
                                        </div>
                                        <div id="gr-desc" style="font-size:12px;color:#94a3b8;margin-top:4px;">
                                            Live momentum breakout detector tracking dealer hedging pressure.
                                        </div>
                                    </div>
                                </div>
                                <div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap;">
                                    <span id="gr-expected-move-badge" style="font-size:11px;font-weight:700;padding:5px 12px;border-radius:6px;background:#171d30;color:#38bdf8;border:1px solid #232b45;letter-spacing:0.3px;">
                                        EXPECTED MOVE: --
                                    </span>
                                    <span id="gr-archetype-badge" style="font-size:11px;font-weight:700;padding:5px 12px;border-radius:6px;background:#171d30;color:#cbd5e1;border:1px solid #232b45;letter-spacing:0.3px;">
                                        0DTE Breakout
                                    </span>
                                    <span id="gr-status-badge" style="font-size:11px;font-weight:700;padding:5px 12px;border-radius:6px;background:#171d30;color:#f59e0b;border:1px solid #232b45;letter-spacing:0.3px;">
                                        COILING
                                    </span>
                                    <span id="gr-direction-badge" style="font-size:11px;font-weight:700;padding:5px 12px;border-radius:6px;background:#171d30;color:#10b981;border:1px solid #232b45;">
                                        CALL BUY (CE)
                                    </span>
                                    <span id="gr-tier-badge" style="display:none;font-size:11px;font-weight:700;padding:5px 12px;border-radius:6px;background:#171d30;color:#cbd5e1;border:1px solid #232b45;"></span>
                                </div>
                            </div>

                            <!-- 4 Key Reference Levels Grid (Clean, Spacious, Plain English) -->
                            <div style="display:grid;grid-template-columns:repeat(auto-fit, minmax(180px, 1fr));gap:14px;margin-bottom:20px;">
                                <!-- Level 1: Current Spot -->
                                <div class="metric-box" style="padding:14px 16px;background:#13182b;border:1px solid #1e2438;border-radius:8px;">
                                    <div class="metric-label" style="font-size:11px;font-weight:700;color:#94a3b8;letter-spacing:0.5px;">SPOT PRICE</div>
                                    <div id="gr-spot-val" style="font-size:24px;font-weight:800;color:#ffffff;font-family:'JetBrains Mono',monospace;margin:4px 0;">--</div>
                                    <div id="gr-spot-sub" class="metric-sub" style="font-size:11px;color:#64748b;">Live NIFTY Index</div>
                                </div>

                                <!-- Level 2: Breakout Trigger -->
                                <div class="metric-box" style="padding:14px 16px;background:#13182b;border:1px solid #1e2438;border-radius:8px;">
                                    <div class="metric-label" style="font-size:11px;font-weight:700;color:#f59e0b;letter-spacing:0.5px;">BREAKOUT TRIGGER</div>
                                    <div id="gr-trigger-val" style="font-size:24px;font-weight:800;color:#f59e0b;font-family:'JetBrains Mono',monospace;margin:4px 0;">--</div>
                                    <div id="gr-trigger-sub" class="metric-sub" style="font-size:11px;color:#94a3b8;">Entry above this level</div>
                                </div>

                                <!-- Level 3: Projected Target -->
                                <div class="metric-box" style="padding:14px 16px;background:#13182b;border:1px solid #1e2438;border-radius:8px;">
                                    <div class="metric-label" style="font-size:11px;font-weight:700;color:#10b981;letter-spacing:0.5px;">PROJECTED TARGET</div>
                                    <div id="gr-target-val" style="font-size:24px;font-weight:800;color:#10b981;font-family:'JetBrains Mono',monospace;margin:4px 0;">--</div>
                                    <div id="gr-target-sub" class="metric-sub" style="font-size:11px;color:#94a3b8;">Dealer rebalance target</div>
                                </div>

                                <!-- Level 4: Major Resistance Ceiling -->
                                <div class="metric-box" style="padding:14px 16px;background:#13182b;border:1px solid #1e2438;border-radius:8px;">
                                    <div class="metric-label" style="font-size:11px;font-weight:700;color:#cbd5e1;letter-spacing:0.5px;">MAJOR RESISTANCE</div>
                                    <div id="gr-fortress-val" style="font-size:24px;font-weight:800;color:#e2e8f0;font-family:'JetBrains Mono',monospace;margin:4px 0;">--</div>
                                    <div id="gr-fortress-sub" class="metric-sub" style="font-size:11px;color:#64748b;">Key open interest ceiling</div>
                                </div>
                            </div>

                            <!-- Runway Progress Bar (Clean & Quiet) -->
                            <div style="background:#13182b;border:1px solid #1e2438;border-radius:8px;padding:14px 18px;margin-bottom:20px;">
                                <div style="display:flex;justify-content:space-between;align-items:center;font-size:12px;color:#94a3b8;margin-bottom:10px;">
                                    <span style="display:flex;align-items:center;gap:8px;">
                                        <span style="color:#f59e0b;font-weight:700;">Trigger: <span id="gr-bar-start">--</span></span>
                                        <span style="color:#64748b;">&bull;</span>
                                        <span style="color:#38bdf8;font-weight:700;">Runway: <span id="gr-bar-runway">-- pts</span></span>
                                    </span>
                                    <span style="font-weight:700;color:#38bdf8;">
                                        Progress: <span id="gr-progress-pct" style="font-family:'JetBrains Mono',monospace;">0%</span>
                                    </span>
                                    <span style="color:#10b981;font-weight:700;">Target: <span id="gr-bar-target">--</span></span>
                                </div>
                                <div style="height:6px;background:#0b0e17;border-radius:3px;overflow:hidden;margin-bottom:8px;">
                                    <div id="gr-progress-fill" style="width:0%;height:100%;background:#38bdf8;border-radius:3px;transition:width 0.4s ease;"></div>
                                </div>
                                <div style="display:flex;justify-content:space-between;font-size:11px;color:#64748b;">
                                    <span>Trigger Level</span>
                                    <span id="gr-fuel-indicator" style="color:#64748b;">Dealer Hedging Volume: --</span>
                                    <span>Target Objective</span>
                                </div>
                            </div>

                            <!-- Dual Strike Recommendation Grid (Clean, Spacious Cards) -->
                            <div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:20px;">
                                <!-- Option 1: Primary ATM Strike -->
                                <div style="background:#13182b;border:1px solid #1e2438;border-radius:10px;padding:18px;">
                                    <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:14px;">
                                        <div>
                                            <span style="font-size:10px;font-weight:700;color:#38bdf8;background:#17223b;padding:3px 8px;border-radius:4px;letter-spacing:0.5px;">PRIMARY (ATM) &bull; BALANCED</span>
                                            <div id="gr-p-strike-name" style="font-size:20px;font-weight:800;color:#ffffff;font-family:'JetBrains Mono',monospace;margin-top:6px;">-- CE</div>
                                        </div>
                                        <div style="text-align:right;">
                                            <div style="font-size:10px;color:#64748b;">Current Price</div>
                                            <div id="gr-p-ltp" style="font-size:20px;font-weight:800;color:#38bdf8;font-family:'JetBrains Mono',monospace;">₹--</div>
                                        </div>
                                    </div>
                                    <!-- Clean Price Levels -->
                                    <div style="display:grid;grid-template-columns:repeat(4, 1fr);gap:8px;text-align:center;">
                                        <div style="background:#0e1322;padding:8px 6px;border-radius:6px;border:1px solid #1a2035;">
                                            <div style="font-size:10px;color:#94a3b8;font-weight:600;">BUY AT</div>
                                            <div id="gr-p-buy" style="font-size:13px;font-weight:700;color:#ffffff;font-family:'JetBrains Mono',monospace;margin-top:2px;">₹--</div>
                                        </div>
                                        <div style="background:#0e1322;padding:8px 6px;border-radius:6px;border:1px solid #1a2035;">
                                            <div style="font-size:10px;color:#10b981;font-weight:600;">TARGET 1</div>
                                            <div id="gr-p-t1" style="font-size:13px;font-weight:700;color:#10b981;font-family:'JetBrains Mono',monospace;margin-top:2px;">₹--</div>
                                            <div id="gr-p-t1-pct" style="font-size:10px;color:#10b981;">+--%</div>
                                        </div>
                                        <div style="background:#0e1322;padding:8px 6px;border-radius:6px;border:1px solid #1a2035;">
                                            <div style="font-size:10px;color:#38bdf8;font-weight:600;">TARGET 2</div>
                                            <div id="gr-p-t2" style="font-size:13px;font-weight:700;color:#38bdf8;font-family:'JetBrains Mono',monospace;margin-top:2px;">₹--</div>
                                            <div id="gr-p-t2-pct" style="font-size:10px;color:#38bdf8;">+--%</div>
                                        </div>
                                        <div style="background:#0e1322;padding:8px 6px;border-radius:6px;border:1px solid #1a2035;">
                                            <div style="font-size:10px;color:#ef4444;font-weight:600;">STOP LOSS</div>
                                            <div id="gr-p-sl" style="font-size:13px;font-weight:700;color:#ef4444;font-family:'JetBrains Mono',monospace;margin-top:2px;">₹--</div>
                                            <div id="gr-p-sl-pct" style="font-size:10px;color:#ef4444;">---%</div>
                                        </div>
                                    </div>
                                    <div style="font-size:11px;color:#64748b;margin-top:12px;">
                                        Balanced delta (~0.50) &bull; Lower time decay risk
                                    </div>
                                </div>

                                <!-- Option 2: Momentum OTM Strike -->
                                <div style="background:#13182b;border:1px solid #1e2438;border-radius:10px;padding:18px;">
                                    <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:14px;">
                                        <div>
                                            <span style="font-size:10px;font-weight:700;color:#cbd5e1;background:#17223b;padding:3px 8px;border-radius:4px;letter-spacing:0.5px;">MOMENTUM (OTM) &bull; HIGH LEVERAGE</span>
                                            <div id="gr-o-strike-name" style="font-size:20px;font-weight:800;color:#ffffff;font-family:'JetBrains Mono',monospace;margin-top:6px;">-- CE</div>
                                        </div>
                                        <div style="text-align:right;">
                                            <div style="font-size:10px;color:#64748b;">Current Price</div>
                                            <div id="gr-o-ltp" style="font-size:20px;font-weight:800;color:#cbd5e1;font-family:'JetBrains Mono',monospace;">₹--</div>
                                        </div>
                                    </div>
                                    <!-- Clean Price Levels -->
                                    <div style="display:grid;grid-template-columns:repeat(4, 1fr);gap:8px;text-align:center;">
                                        <div style="background:#0e1322;padding:8px 6px;border-radius:6px;border:1px solid #1a2035;">
                                            <div style="font-size:10px;color:#94a3b8;font-weight:600;">BUY AT</div>
                                            <div id="gr-o-buy" style="font-size:13px;font-weight:700;color:#ffffff;font-family:'JetBrains Mono',monospace;margin-top:2px;">₹--</div>
                                        </div>
                                        <div style="background:#0e1322;padding:8px 6px;border-radius:6px;border:1px solid #1a2035;">
                                            <div style="font-size:10px;color:#10b981;font-weight:600;">TARGET 1</div>
                                            <div id="gr-o-t1" style="font-size:13px;font-weight:700;color:#10b981;font-family:'JetBrains Mono',monospace;margin-top:2px;">₹--</div>
                                            <div id="gr-o-t1-pct" style="font-size:10px;color:#10b981;">+--%</div>
                                        </div>
                                        <div style="background:#0e1322;padding:8px 6px;border-radius:6px;border:1px solid #1a2035;">
                                            <div style="font-size:10px;color:#38bdf8;font-weight:600;">TARGET 2</div>
                                            <div id="gr-o-t2" style="font-size:13px;font-weight:700;color:#38bdf8;font-family:'JetBrains Mono',monospace;margin-top:2px;">₹--</div>
                                            <div id="gr-o-t2-pct" style="font-size:10px;color:#38bdf8;">+--%</div>
                                        </div>
                                        <div style="background:#0e1322;padding:8px 6px;border-radius:6px;border:1px solid #1a2035;">
                                            <div style="font-size:10px;color:#ef4444;font-weight:600;">STOP LOSS</div>
                                            <div id="gr-o-sl" style="font-size:13px;font-weight:700;color:#ef4444;font-family:'JetBrains Mono',monospace;margin-top:2px;">₹--</div>
                                            <div id="gr-o-sl-pct" style="font-size:10px;color:#ef4444;">---%</div>
                                        </div>
                                    </div>
                                    <div style="font-size:11px;color:#64748b;margin-top:12px;">
                                        <span id="gr-o-status-tag" style="color:#94a3b8;">Lower premium cost &bull; Higher percentage upside on momentum</span>
                                    </div>
                                </div>
                            </div>

                            <!-- Bottom Action Summary (Clean & Clear) -->
                            <div style="background:#13182b;border:1px solid #1e2438;border-radius:8px;padding:14px 18px;display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:10px;">
                                <div style="display:flex;align-items:center;gap:10px;">
                                    <span style="font-size:11px;font-weight:700;color:#f59e0b;text-transform:uppercase;letter-spacing:0.5px;">Action Plan:</span>
                                    <span id="gr-action-text" style="font-size:13px;font-weight:600;color:#ffffff;">Monitoring Nifty for trigger proximity...</span>
                                </div>
                                <div style="font-size:11px;color:#64748b;" id="gr-update-ts">
                                    Updated: --:--:--
                                </div>
                            </div>
                        </div>

                        <!-- 3. REAL-TIME OI DYNAMICS & ORDER FLOW RADAR -->
                        <div class="card" id="oi-velocity-card" style="margin-top:16px;margin-bottom:14px;padding:0;overflow:hidden;border:1px solid rgba(0,229,255,0.25);border-radius:10px;box-shadow:0 6px 24px rgba(0,0,0,0.45);background:rgba(13,17,36,0.95);">
                            <!-- Top Header Toolbar -->
                            <div style="display:flex;justify-content:space-between;align-items:center;padding:12px 18px;background:rgba(18,22,46,0.98);border-bottom:1px solid rgba(255,255,255,0.08);flex-wrap:wrap;gap:10px;">
                                <div style="display:flex;align-items:center;gap:10px;flex-wrap:wrap;">
                                    <div style="font-size:14px;font-weight:900;letter-spacing:1.5px;color:#00e5ff;">OI DYNAMICS &amp; ORDER FLOW RADAR</div>
                                    <span id="oi-rewind-status-badge" style="font-size:10px;background:rgba(0,230,118,0.15);color:#00e676;border:1px solid rgba(0,230,118,0.4);padding:3px 8px;border-radius:4px;font-weight:800;letter-spacing:0.5px;">LIVE STREAMING</span>
                                    <button type="button" id="btn-return-live" style="display:none;padding:3px 10px;font-size:10px;font-weight:800;border-radius:4px;background:rgba(0,229,255,0.2);color:#00e5ff;border:1px solid #00e5ff;cursor:pointer;">↺ RETURN TO LIVE</button>
                                </div>

                                <div style="display:flex;gap:12px;align-items:center;flex-wrap:wrap;">
                                    <!-- Metric Mode: Total Cumulative OI vs Absolute Delta Volume vs Rate of Change -->
                                    <div style="display:flex;align-items:center;gap:4px;background:rgba(255,255,255,0.03);padding:2px 4px;border-radius:6px;border:1px solid rgba(255,255,255,0.08);">
                                        <button type="button" id="btn-metric-total" class="oi-metric-btn" style="padding:4px 9px;font-size:10px;font-weight:800;border-radius:4px;background:transparent;color:#94a3b8;border:1px solid transparent;cursor:pointer;">TOTAL OI</button>
                                        <button type="button" id="btn-metric-delta" class="oi-metric-btn active" style="padding:4px 9px;font-size:10px;font-weight:800;border-radius:4px;background:rgba(0,229,255,0.18);color:#00e5ff;border:1px solid #00e5ff;cursor:pointer;">Δ OI VOLUME</button>
                                        <button type="button" id="btn-metric-rate" class="oi-metric-btn" style="padding:4px 9px;font-size:10px;font-weight:800;border-radius:4px;background:transparent;color:#94a3b8;border:1px solid transparent;cursor:pointer;">RATE / MIN</button>
                                    </div>

                                    <!-- Strike Field of View Range -->
                                    <div style="display:flex;align-items:center;gap:4px;background:rgba(255,255,255,0.03);padding:2px 4px;border-radius:6px;border:1px solid rgba(255,255,255,0.08);">
                                        <span style="font-size:10px;color:#64748b;font-weight:700;margin-left:2px;">RANGE:</span>
                                        <button type="button" id="btn-range-500" class="oi-range-btn" style="padding:3px 6px;font-size:10px;font-weight:700;border-radius:4px;background:transparent;color:#94a3b8;border:none;cursor:pointer;">±10</button>
                                        <button type="button" id="btn-range-1000" class="oi-range-btn active" style="padding:3px 7px;font-size:10px;font-weight:800;border-radius:4px;background:rgba(0,229,255,0.15);color:#00e5ff;border:1px solid rgba(0,229,255,0.3);cursor:pointer;">±20 (1k)</button>
                                        <button type="button" id="btn-range-1500" class="oi-range-btn" style="padding:3px 6px;font-size:10px;font-weight:700;border-radius:4px;background:transparent;color:#94a3b8;border:none;cursor:pointer;">±30</button>
                                        <button type="button" id="btn-range-all" class="oi-range-btn" style="padding:3px 6px;font-size:10px;font-weight:700;border-radius:4px;background:transparent;color:#94a3b8;border:none;cursor:pointer;">ALL</button>
                                    </div>

                                    <!-- Timeframe Selector -->
                                    <div style="display:flex;align-items:center;gap:4px;">
                                        <button type="button" id="btn-tf-1m" class="oi-tf-btn" style="padding:4px 8px;font-size:10px;font-weight:800;border-radius:4px;background:rgba(255,255,255,0.04);color:#94a3b8;border:1px solid rgba(255,255,255,0.1);cursor:pointer;">1m</button>
                                        <button type="button" id="btn-tf-3m" class="oi-tf-btn" style="padding:4px 8px;font-size:10px;font-weight:800;border-radius:4px;background:rgba(255,255,255,0.04);color:#94a3b8;border:1px solid rgba(255,255,255,0.1);cursor:pointer;">3m</button>
                                        <button type="button" id="btn-tf-5m" class="oi-tf-btn active" style="padding:4px 8px;font-size:10px;font-weight:800;border-radius:4px;background:rgba(0,229,255,0.18);color:#00e5ff;border:1px solid #00e5ff;cursor:pointer;">5m</button>
                                        <button type="button" id="btn-tf-15m" class="oi-tf-btn" style="padding:4px 8px;font-size:10px;font-weight:800;border-radius:4px;background:rgba(255,255,255,0.04);color:#94a3b8;border:1px solid rgba(255,255,255,0.1);cursor:pointer;">15m</button>
                                    </div>
                                </div>
                            </div>

                            <!-- Whole-Day Macro Context Strip -->
                            <div style="display:flex;justify-content:space-between;align-items:center;padding:7px 18px;background:rgba(12,16,34,0.9);border-bottom:1px solid rgba(255,255,255,0.06);flex-wrap:wrap;gap:10px;font-size:11px;">
                                <div style="display:flex;align-items:center;gap:14px;flex-wrap:wrap;">
                                    <span style="color:#64748b;font-weight:800;letter-spacing:0.5px;">DAY TOTALS:</span>
                                    <span style="color:#94a3b8;">Total Calls: <strong id="oi-macro-ce-val" style="color:#00e5ff;font-family:'JetBrains Mono',monospace;">--</strong></span>
                                    <span style="color:#94a3b8;">Total Puts: <strong id="oi-macro-pe-val" style="color:#00e676;font-family:'JetBrains Mono',monospace;">--</strong></span>
                                    <span style="color:#94a3b8;">Day PCR: <strong id="oi-macro-pcr-val" style="color:#ffd54f;font-family:'JetBrains Mono',monospace;">--</strong></span>
                                </div>
                                <span id="oi-macro-bias-badge" style="font-size:10px;font-weight:800;letter-spacing:0.5px;padding:2px 8px;border-radius:4px;border:1px solid rgba(255,255,255,0.1);color:#cbd5e1;background:rgba(255,255,255,0.03);">WHOLE-DAY MACRO: BALANCED</span>
                            </div>

                            <!-- Multi-Timeframe Rate Cards Row -->
                            <div style="padding:10px 18px;background:rgba(10,14,30,0.6);border-bottom:1px solid rgba(255,255,255,0.06);">
                                <div style="display:grid;grid-template-columns:repeat(auto-fit, minmax(170px, 1fr));gap:8px;">
                                    <div id="tf-card-1m" style="background:rgba(15,23,42,0.6);border:1px solid rgba(255,255,255,0.08);border-radius:6px;padding:6px 10px;cursor:pointer;transition:all 0.2s;">
                                        <div style="display:flex;justify-content:space-between;align-items:center;font-size:10px;color:#64748b;font-weight:700;">
                                            <span>1m RATE</span>
                                            <span id="tf-bias-1m" style="color:#94a3b8;font-weight:800;">--</span>
                                        </div>
                                        <div id="tf-val-1m" style="font-size:15px;font-weight:900;color:#94a3b8;margin-top:2px;font-family:'JetBrains Mono',monospace;">--</div>
                                    </div>
                                    <div id="tf-card-3m" style="background:rgba(15,23,42,0.6);border:1px solid rgba(255,255,255,0.08);border-radius:6px;padding:6px 10px;cursor:pointer;transition:all 0.2s;">
                                        <div style="display:flex;justify-content:space-between;align-items:center;font-size:10px;color:#64748b;font-weight:700;">
                                            <span>3m RATE</span>
                                            <span id="tf-bias-3m" style="color:#94a3b8;font-weight:800;">--</span>
                                        </div>
                                        <div id="tf-val-3m" style="font-size:15px;font-weight:900;color:#94a3b8;margin-top:2px;font-family:'JetBrains Mono',monospace;">--</div>
                                    </div>
                                    <div id="tf-card-5m" style="background:rgba(0,229,255,0.08);border:1px solid #00e5ff;border-radius:6px;padding:6px 10px;cursor:pointer;transition:all 0.2s;">
                                        <div style="display:flex;justify-content:space-between;align-items:center;font-size:10px;color:#64748b;font-weight:700;">
                                            <span>5m RATE</span>
                                            <span id="tf-bias-5m" style="color:#00e5ff;font-weight:800;">--</span>
                                        </div>
                                        <div id="tf-val-5m" style="font-size:15px;font-weight:900;color:#00e5ff;margin-top:2px;font-family:'JetBrains Mono',monospace;">--</div>
                                    </div>
                                    <div id="tf-card-15m" style="background:rgba(15,23,42,0.6);border:1px solid rgba(255,255,255,0.08);border-radius:6px;padding:6px 10px;cursor:pointer;transition:all 0.2s;">
                                        <div style="display:flex;justify-content:space-between;align-items:center;font-size:10px;color:#64748b;font-weight:700;">
                                            <span>15m RATE</span>
                                            <span id="tf-bias-15m" style="color:#94a3b8;font-weight:800;">--</span>
                                        </div>
                                        <div id="tf-val-15m" style="font-size:15px;font-weight:900;color:#94a3b8;margin-top:2px;font-family:'JetBrains Mono',monospace;">--</div>
                                    </div>
                                </div>
                            </div>

                            <!-- FOCUSED ADVISORY: MAJOR LEVELS & WHERE BIG MONEY MOVED -->
                            <div style="padding:12px 18px;background:rgba(14,18,40,0.92);border-bottom:1px solid rgba(255,255,255,0.06);">
                                <div style="display:grid;grid-template-columns:repeat(auto-fit, minmax(320px, 1fr));gap:12px;">
                                    <!-- Panel A: Major Structural Levels (Call Wall, Put Wall, ATM) -->
                                    <div style="background:rgba(18,24,52,0.85);border:1px solid rgba(255,255,255,0.08);border-radius:8px;padding:10px 14px;">
                                        <div style="font-size:11px;font-weight:800;color:#ffd54f;margin-bottom:8px;letter-spacing:0.5px;">🏛️ MAJOR STRUCTURAL LEVELS</div>
                                        <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;font-family:'JetBrains Mono',monospace;">
                                            <div style="background:rgba(255,51,102,0.06);border:1px solid rgba(255,51,102,0.25);border-radius:6px;padding:6px 8px;">
                                                <div style="font-size:10px;color:#94a3b8;">CALL WALL (Ceiling)</div>
                                                <div id="major-call-wall-strike" style="font-size:14px;font-weight:900;color:#ff3366;margin-top:2px;">--</div>
                                                <div id="major-call-wall-delta" style="font-size:10px;color:#cbd5e1;margin-top:2px;">--</div>
                                            </div>
                                            <div style="background:rgba(0,230,118,0.06);border:1px solid rgba(0,230,118,0.25);border-radius:6px;padding:6px 8px;">
                                                <div style="font-size:10px;color:#94a3b8;">PUT WALL (Floor)</div>
                                                <div id="major-put-wall-strike" style="font-size:14px;font-weight:900;color:#00e676;margin-top:2px;">--</div>
                                                <div id="major-put-wall-delta" style="font-size:10px;color:#cbd5e1;margin-top:2px;">--</div>
                                            </div>
                                        </div>
                                        <div id="major-levels-status-text" style="font-size:11px;color:#94a3b8;margin-top:8px;line-height:1.4;">
                                            Monitoring major structural wall defenses...
                                        </div>
                                    </div>

                                    <!-- Panel B: Outlier Hotspots (Where Major Change Happened) -->
                                    <div style="background:rgba(18,24,52,0.85);border:1px solid rgba(0,229,255,0.2);border-radius:8px;padding:10px 14px;">
                                        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;">
                                            <span style="font-size:11px;font-weight:800;color:#00e5ff;letter-spacing:0.5px;">⚡ WHERE MAJOR CHANGE HAPPENED</span>
                                            <span id="oi-net-pressure-pill" style="font-size:10px;font-weight:800;color:#ffd54f;background:rgba(255,213,79,0.12);border:1px solid #ffd54f;padding:2px 6px;border-radius:4px;font-family:'JetBrains Mono',monospace;">NET: --</span>
                                        </div>
                                        <!-- Call vs Put Flow Breakdown Cards -->
                                        <div style="display:grid;grid-template-columns:1fr 1fr;gap:6px;margin-bottom:6px;font-size:10px;font-family:'JetBrains Mono',monospace;">
                                            <div style="background:rgba(255,51,102,0.05);border:1px solid rgba(255,51,102,0.2);border-radius:5px;padding:4px 6px;">
                                                <div style="color:#ff3366;font-weight:800;margin-bottom:2px;">CALL FLOW</div>
                                                <div id="oi-call-write-hotspot" style="color:#cbd5e1;">Writing: --</div>
                                                <div id="oi-call-unwind-hotspot" style="color:#cbd5e1;">Covering: --</div>
                                            </div>
                                            <div style="background:rgba(0,230,118,0.05);border:1px solid rgba(0,230,118,0.2);border-radius:5px;padding:4px 6px;">
                                                <div style="color:#00e676;font-weight:800;margin-bottom:2px;">PUT FLOW</div>
                                                <div id="oi-put-write-hotspot" style="color:#cbd5e1;">Writing: --</div>
                                                <div id="oi-put-unwind-hotspot" style="color:#cbd5e1;">Dumping: --</div>
                                            </div>
                                        </div>
                                        <div id="oi-hotspots-container" style="display:flex;flex-direction:column;gap:5px;">
                                            <div style="font-size:11px;color:#64748b;">Accumulating consecutive ticks to identify volume surge strikes...</div>
                                        </div>
                                        <div id="oi-advisory-narrative" style="font-size:11px;color:#cbd5e1;line-height:1.4;margin-top:6px;border-top:1px dashed rgba(255,255,255,0.08);padding-top:6px;">
                                            Analyzing major positioning shifts...
                                        </div>
                                    </div>
                                </div>
                            </div>

                            <!-- Historical Rewind Toolbar -->
                            <div style="display:flex;justify-content:space-between;align-items:center;padding:8px 18px;background:rgba(15,20,38,0.95);border-bottom:1px solid rgba(255,255,255,0.06);flex-wrap:wrap;gap:10px;">
                                <div style="display:flex;align-items:center;gap:8px;flex:1;min-width:280px;">
                                    <span style="font-size:11px;color:#94a3b8;font-weight:700;white-space:nowrap;">REWIND MEMORY:</span>
                                    <input type="range" id="oi-rewind-slider" min="0" max="10" value="10" style="flex:1;accent-color:#00e5ff;cursor:pointer;">
                                    <span id="oi-rewind-time-label" style="font-size:11px;font-family:'JetBrains Mono',monospace;color:#00e5ff;font-weight:800;min-width:65px;text-align:right;">LIVE</span>
                                </div>
                                <div style="display:flex;align-items:center;gap:5px;flex-wrap:wrap;">
                                    <span style="font-size:10px;color:#64748b;font-weight:700;">PRESETS:</span>
                                    <button type="button" id="btn-preset-live" class="oi-preset-btn active" style="padding:3px 8px;font-size:10px;font-weight:800;border-radius:4px;background:rgba(0,230,118,0.15);color:#00e676;border:1px solid rgba(0,230,118,0.4);cursor:pointer;">LIVE</button>
                                    <button type="button" id="btn-preset-1m" class="oi-preset-btn" style="padding:3px 7px;font-size:10px;font-weight:700;border-radius:4px;background:rgba(255,255,255,0.04);color:#94a3b8;border:1px solid rgba(255,255,255,0.1);cursor:pointer;">-1m</button>
                                    <button type="button" id="btn-preset-3m" class="oi-preset-btn" style="padding:3px 7px;font-size:10px;font-weight:700;border-radius:4px;background:rgba(255,255,255,0.04);color:#94a3b8;border:1px solid rgba(255,255,255,0.1);cursor:pointer;">-3m</button>
                                    <button type="button" id="btn-preset-5m" class="oi-preset-btn" style="padding:3px 7px;font-size:10px;font-weight:700;border-radius:4px;background:rgba(255,255,255,0.04);color:#94a3b8;border:1px solid rgba(255,255,255,0.1);cursor:pointer;">-5m</button>
                                    <button type="button" id="btn-preset-10m" class="oi-preset-btn" style="padding:3px 7px;font-size:10px;font-weight:700;border-radius:4px;background:rgba(255,255,255,0.04);color:#94a3b8;border:1px solid rgba(255,255,255,0.1);cursor:pointer;">-10m</button>
                                    <button type="button" id="btn-preset-15m" class="oi-preset-btn" style="padding:3px 7px;font-size:10px;font-weight:700;border-radius:4px;background:rgba(255,255,255,0.04);color:#94a3b8;border:1px solid rgba(255,255,255,0.1);cursor:pointer;">-15m</button>
                                    <button type="button" id="btn-preset-30m" class="oi-preset-btn" style="padding:3px 7px;font-size:10px;font-weight:700;border-radius:4px;background:rgba(255,255,255,0.04);color:#94a3b8;border:1px solid rgba(255,255,255,0.1);cursor:pointer;">-30m</button>
                                </div>
                            </div>

                            <!-- Bidirectional Plotly Chart Canvas -->
                            <div style="padding:10px 14px;background:rgba(8,11,26,0.85);">
                                <div id="oi-velocity-chart" style="width:100%;min-height:560px;"></div>
                            </div>

                            <!-- Bottom Flow Legend Guide Strip -->
                            <div style="display:flex;justify-content:space-between;align-items:center;padding:8px 18px;background:rgba(15,20,38,0.95);border-top:1px solid rgba(255,255,255,0.06);font-size:11px;font-family:'JetBrains Mono',monospace;flex-wrap:wrap;gap:8px;">
                                <div style="display:flex;gap:14px;flex-wrap:wrap;">
                                    <span style="color:#ff3366;">🔴 CALL WRITING (RESISTANCE)</span>
                                    <span style="color:#00e5ff;">🔵 CALL SHORT COVERING</span>
                                    <span style="color:#00e676;">🟢 PUT WRITING (SUPPORT)</span>
                                    <span style="color:#ff9100;">🟠 PUT CAPITULATION</span>
                                </div>
                                <div style="color:#94a3b8;font-size:10px;">
                                    <span>░░ DOTTED GHOST BARS: CURRENT LIVE FLOW (WHEN REWOUND)</span>
                                </div>
                            </div>
                        </div>
'''
                    else:
                        chain_tab_html = '<div class="card"><p style="color:#888;">No option chain data available for analysis.</p></div>'


                    # ── TAB 5: THETA DECAY EXPLORER & GREEKS DYNAMICS ──
                    theta_tab_html = '<div class="card"><p style="color:#888;">Waiting for option chain data to compute Theta Decay...</p></div>'
                    try:
                        if not df_chain.empty and spot > 0:
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
                                _ce_thetas_pts = []
                                _pe_thetas_pts = []
                                _straddle_thetas_pts = []
                                _daily_cushions_pts = []
                                _decay_yields_pct = []
                                _valid_strikes = []
                                _ce_deltas = []
                                _pe_deltas = []
                                _net_deltas = []
                                _gammas = []
                                _ce_vegas_pts = []
                                _pe_vegas_pts = []
                                _straddle_vegas_pts = []

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

                                    _ce_intr = max(0.0, spot - _K)
                                    _pe_intr = max(0.0, _K - spot)
                                    _ce_ext_pts = max(0.0, _ce_ltp - _ce_intr)
                                    _pe_ext_pts = max(0.0, _pe_ltp - _pe_intr)

                                    _ce_intr_pct = (_ce_intr / _ce_ltp * 100.0) if _ce_ltp > 0 else 0.0
                                    _ce_ext_pct = (_ce_ext_pts / _ce_ltp * 100.0) if _ce_ltp > 0 else 100.0
                                    _pe_intr_pct = (_pe_intr / _pe_ltp * 100.0) if _pe_ltp > 0 else 0.0
                                    _pe_ext_pct = (_pe_ext_pts / _pe_ltp * 100.0) if _pe_ltp > 0 else 100.0

                                    # Extrinsic boundary clamping: Daily decay cannot exceed remaining extrinsic value
                                    _th_ce_day = min(abs(_th_ce), _ce_ext_pts) if _ce_ext_pts > 0 else abs(_th_ce)
                                    _th_pe_day = min(abs(_th_pe), _pe_ext_pts) if _pe_ext_pts > 0 else abs(_th_pe)

                                    _th_strad_pts = _th_ce_day + _th_pe_day
                                    _strad_gam = _gam_ce + _gam_pe
                                    _del_net = _del_ce + _del_pe
                                    _vg_net = _vg_ce + _vg_pe

                                    _daily_cushion_pts = float(np.sqrt(max(2.0 * _th_strad_pts / max(_strad_gam, 1e-7), 0.0)))
                                    _daily_cushion_pts = min(_daily_cushion_pts, 999.0)

                                    _strad_prem = max(_ce_ltp + _pe_ltp, 0.01)
                                    _yield_pct = (_th_strad_pts / _strad_prem) * 100.0
                                    _yield_pct = min(_yield_pct, 100.0)

                                    _valid_strikes.append(_K)
                                    _ce_thetas_pts.append(_th_ce_day)
                                    _pe_thetas_pts.append(_th_pe_day)
                                    _straddle_thetas_pts.append(_th_strad_pts)
                                    _daily_cushions_pts.append(_daily_cushion_pts)
                                    _decay_yields_pct.append(_yield_pct)
                                    _ce_deltas.append(_del_ce)
                                    _pe_deltas.append(_del_pe)
                                    _net_deltas.append(_del_net)
                                    _gammas.append(_strad_gam)
                                    _ce_vegas_pts.append(_vg_ce)
                                    _pe_vegas_pts.append(_vg_pe)
                                    _straddle_vegas_pts.append(_vg_net)

                                    _is_atm = (_K == _atm_strike_th)
                                    _row_id = 'id="th-row-atm"' if _is_atm else ''
                                    _row_atm_attr = 'data-is-atm="true"' if _is_atm else ''
                                    _row_style = 'background:rgba(79,195,247,0.12);font-weight:700;border-left:3px solid #00e5ff;' if _is_atm else ''

                                    _dist = _K - spot
                                    _dist_cls = GREEN if _dist > 0 else RED if _dist < 0 else ACCENT
                                    _dist_str = f"{_dist:+.0f}" if _dist != 0 else "ATM"

                                    _ce_comp_html = (
                                        f'<div style="display:flex;flex-direction:column;gap:1px;font-size:10.5px;font-family:var(--font-mono);">'
                                        f'<div><span style="color:#868993;">Int:</span> <span style="color:#ffffff;">{_ce_intr:.1f} pts</span> <span style="color:#64748b;font-size:9.5px;">({_ce_intr_pct:.0f}%)</span></div>'
                                        f'<div><span style="color:#868993;">Ext:</span> <strong style="color:#00e5ff;">{_ce_ext_pts:.1f} pts</strong> <span style="color:#00e5ff;font-size:9.5px;">({_ce_ext_pct:.0f}%)</span></div>'
                                        f'</div>'
                                    )

                                    _ce_greeks_html = (
                                        f'<div style="display:flex;flex-direction:column;gap:1px;font-size:10.5px;font-family:var(--font-mono);">'
                                        f'<div><span style="color:#868993;">Δ:</span> <strong style="color:#00e5ff;">{_del_ce:+.2f}</strong> · <span style="color:#868993;">Θ:</span> <strong style="color:#ef5350;">-{_th_ce_day:.1f}</strong> <span style="color:#64748b;font-size:9.5px;">pts/d</span></div>'
                                        f'<div><span style="color:#868993;">Vega:</span> <strong style="color:#00e676;">+{_vg_ce:.1f}</strong> <span style="color:#64748b;font-size:9.5px;">pts/1%</span></div>'
                                        f'</div>'
                                    )

                                    _pe_comp_html = (
                                        f'<div style="display:flex;flex-direction:column;gap:1px;font-size:10.5px;font-family:var(--font-mono);">'
                                        f'<div><span style="color:#868993;">Int:</span> <span style="color:#ffffff;">{_pe_intr:.1f} pts</span> <span style="color:#64748b;font-size:9.5px;">({_pe_intr_pct:.0f}%)</span></div>'
                                        f'<div><span style="color:#868993;">Ext:</span> <strong style="color:#ff7043;">{_pe_ext_pts:.1f} pts</strong> <span style="color:#ff7043;font-size:9.5px;">({_pe_ext_pct:.0f}%)</span></div>'
                                        f'</div>'
                                    )

                                    _pe_greeks_html = (
                                        f'<div style="display:flex;flex-direction:column;gap:1px;font-size:10.5px;font-family:var(--font-mono);">'
                                        f'<div><span style="color:#868993;">Δ:</span> <strong style="color:#ff7043;">{_del_pe:+.2f}</strong> · <span style="color:#868993;">Θ:</span> <strong style="color:#ef5350;">-{_th_pe_day:.1f}</strong> <span style="color:#64748b;font-size:9.5px;">pts/d</span></div>'
                                        f'<div><span style="color:#868993;">Vega:</span> <strong style="color:#00e676;">+{_vg_pe:.1f}</strong> <span style="color:#64748b;font-size:9.5px;">pts/1%</span></div>'
                                        f'</div>'
                                    )

                                    _theta_table_rows.append(
                                        f'<tr {_row_id} {_row_atm_attr} style="{_row_style}border-bottom:1px solid #1a1a2e;">'
                                        f'<td style="padding:6px 8px;text-align:left;font-weight:700;color:{ACCENT if _is_atm else WHITE};">{int(_K)}{" ◄ ATM" if _is_atm else ""}</td>'
                                        f'<td style="padding:6px 8px;text-align:right;color:{_dist_cls};font-family:var(--font-mono);">{_dist_str}</td>'
                                        f'<td style="padding:6px 8px;text-align:right;color:#00e5ff;font-family:var(--font-mono);">{_ce_ltp:.1f} <span style="font-size:10px;color:#868993;">({_ce_calc_iv:.1f}%)</span></td>'
                                        f'<td style="padding:6px 8px;text-align:left;">{_ce_comp_html}</td>'
                                        f'<td style="padding:6px 8px;text-align:left;">{_ce_greeks_html}</td>'
                                        f'<td style="padding:6px 8px;text-align:right;color:#ff7043;font-family:var(--font-mono);">{_pe_ltp:.1f} <span style="font-size:10px;color:#868993;">({_pe_calc_iv:.1f}%)</span></td>'
                                        f'<td style="padding:6px 8px;text-align:left;">{_pe_comp_html}</td>'
                                        f'<td style="padding:6px 8px;text-align:left;">{_pe_greeks_html}</td>'
                                        f'<td style="padding:6px 8px;text-align:right;font-weight:800;color:#ffd54f;font-family:var(--font-mono); font-size:11px;">-{_th_strad_pts:.1f} pts/d</td>'
                                        f'<td style="padding:6px 8px;text-align:right;font-weight:700;color:#00e676;font-family:var(--font-mono);">±{_daily_cushion_pts:.0f} pts</td>'
                                        f'</tr>'
                                    )

                                _atm_idx = _valid_strikes.index(_atm_strike_th) if _atm_strike_th in _valid_strikes else 0
                                _atm_ce_pts_abs = _ce_thetas_pts[_atm_idx] if _atm_idx < len(_ce_thetas_pts) else 5.4
                                _atm_pe_pts_abs = _pe_thetas_pts[_atm_idx] if _atm_idx < len(_pe_thetas_pts) else 5.6
                                _atm_strad_day_pts = _straddle_thetas_pts[_atm_idx] if _atm_idx < len(_straddle_thetas_pts) else (_atm_ce_pts_abs + _atm_pe_pts_abs)
                                _atm_strad_1h_pts = _atm_strad_day_pts / 6.25
                                _atm_strad_vega = _straddle_vegas_pts[_atm_idx] if _atm_idx < len(_straddle_vegas_pts) else 19.5

                                _atm_del_ce = _ce_deltas[_atm_idx] if _atm_idx < len(_ce_deltas) else 0.5
                                _atm_del_pe = _pe_deltas[_atm_idx] if _atm_idx < len(_pe_deltas) else -0.5
                                _atm_del_net = _net_deltas[_atm_idx] if _atm_idx < len(_net_deltas) else (_atm_del_ce + _atm_del_pe)

                                _atm_ce_row = _th_lookup.get((_atm_strike_th, 'CE'))
                                _atm_pe_row = _th_lookup.get((_atm_strike_th, 'PE'))
                                _atm_ce_ltp = float(_atm_ce_row['price']) if _atm_ce_row is not None else 0.0
                                _atm_pe_ltp = float(_atm_pe_row['price']) if _atm_pe_row is not None else 0.0
                                _atm_strad_prem = max(_atm_ce_ltp + _atm_pe_ltp, 0.01)

                                _atm_strad_yield = (_atm_strad_day_pts / max(_atm_strad_prem, 0.1)) * 100.0
                                _atm_strad_cushion = _daily_cushions_pts[_atm_idx] if _atm_idx < len(_daily_cushions_pts) else 60.0

                                # Intraday High/Low Watermark Context Tracker (persistent)
                                _intraday_greeks_file = os.path.join(dashboard_dir, "fintel_greeks_intraday.json")
                                _today_str = datetime.now().strftime("%Y-%m-%d")
                                _now_time_str = datetime.now().strftime("%H:%M:%S")
                                _intraday_data = {}
                                if os.path.exists(_intraday_greeks_file):
                                    try:
                                        with open(_intraday_greeks_file, 'r', encoding='utf-8') as _igf:
                                            _intraday_data = json.load(_igf)
                                    except Exception:
                                        _intraday_data = {}

                                if _intraday_data.get("date") == _today_str:
                                    _th_intraday_min = min(float(_intraday_data.get("atm_theta_min", _atm_strad_day_pts)), _atm_strad_day_pts)
                                    _th_intraday_max = max(float(_intraday_data.get("atm_theta_max", _atm_strad_day_pts)), _atm_strad_day_pts)
                                    _vg_intraday_min = min(float(_intraday_data.get("atm_vega_min", _atm_strad_vega)), _atm_strad_vega)
                                    _vg_intraday_max = max(float(_intraday_data.get("atm_vega_max", _atm_strad_vega)), _atm_strad_vega)
                                else:
                                    _th_intraday_min = _atm_strad_day_pts
                                    _th_intraday_max = _atm_strad_day_pts
                                    _vg_intraday_min = _atm_strad_vega
                                    _vg_intraday_max = _atm_strad_vega

                                _intraday_to_save = {
                                    "date": _today_str,
                                    "atm_theta_min": round(_th_intraday_min, 2),
                                    "atm_theta_max": round(_th_intraday_max, 2),
                                    "atm_vega_min": round(_vg_intraday_min, 2),
                                    "atm_vega_max": round(_vg_intraday_max, 2),
                                    "last_updated": _now_time_str
                                }
                                try:
                                    with open(_intraday_greeks_file, 'w', encoding='utf-8') as _igf:
                                        json.dump(_intraday_to_save, _igf, indent=2)
                                except Exception:
                                    pass

                                _th_range = max(_th_intraday_max - _th_intraday_min, 0.1)
                                _th_pctile_day = min(max((_atm_strad_day_pts - _th_intraday_min) / _th_range * 100.0, 0.0), 100.0)

                                # Theoretical baseline ATM straddle decay pace for this DTE:
                                # Black-Scholes ATM Straddle Theta ~ (2 * S * sigma) / sqrt(2 * pi * 365 * DTE)
                                _theo_atm_theta = (2.0 * spot * (_atm_iv_th / 100.0)) / np.sqrt(2.0 * np.pi * 365.0 * max(_curr_dte_th, 0.05))

                                # ── TWO CLEAN DUAL-AXIS GREEKS VISUALIZERS ──
                                # Plot 1: Theta Bleed Curve & Vega Sensitivity across Strikes
                                _fig_th_bleed = make_subplots(specs=[[{"secondary_y": True}]])
                                _fig_th_bleed.add_trace(go.Scatter(
                                    x=_valid_strikes, y=_ce_thetas_pts,
                                    mode='lines+markers', name='Call Decay (pts/d)',
                                    line=dict(color='#00e5ff', width=2),
                                    marker=dict(size=4),
                                    hovertemplate="<b>%{x:,.0f}</b> Call Bleed: <b>-%{y:.1f} pts/d</b><extra></extra>"
                                ), secondary_y=False)
                                _fig_th_bleed.add_trace(go.Scatter(
                                    x=_valid_strikes, y=_pe_thetas_pts,
                                    mode='lines+markers', name='Put Decay (pts/d)',
                                    line=dict(color='#ff7043', width=2),
                                    marker=dict(size=4),
                                    hovertemplate="<b>%{x:,.0f}</b> Put Bleed: <b>-%{y:.1f} pts/d</b><extra></extra>"
                                ), secondary_y=False)
                                _fig_th_bleed.add_trace(go.Scatter(
                                    x=_valid_strikes, y=_straddle_thetas_pts,
                                    mode='lines', name='Straddle Bleed (pts/d)',
                                    line=dict(color='#ffd54f', width=2.5),
                                    hovertemplate="<b>%{x:,.0f}</b> Straddle Bleed: <b>-%{y:.1f} pts/d</b><extra></extra>"
                                ), secondary_y=False)
                                _fig_th_bleed.add_trace(go.Scatter(
                                    x=_valid_strikes, y=_straddle_vegas_pts,
                                    mode='lines+markers', name='Straddle Vega (pts/1% IV)',
                                    line=dict(color='#00e676', width=2, dash='dot'),
                                    marker=dict(size=4),
                                    hovertemplate="<b>%{x:,.0f}</b> Vega: <b>±%{y:.1f} pts/1%</b><extra></extra>"
                                ), secondary_y=True)

                                # Intraday ATM Bleed Corridor Watermark
                                _fig_th_bleed.add_hrect(
                                    y0=_th_intraday_min, y1=_th_intraday_max,
                                    fillcolor="rgba(255, 213, 79, 0.08)", line_width=1, line_dash="dot", line_color="rgba(255, 213, 79, 0.3)",
                                    annotation_text=f"Day ATM Range ({_th_intraday_min:.1f} - {_th_intraday_max:.1f} pts/d)",
                                    annotation_position="top left", annotation_font=dict(size=8, color="#ffd54f"),
                                    secondary_y=False
                                )
                                _fig_th_bleed.add_vline(x=spot, line_width=1.5, line_dash="dash", line_color="#00e5ff", annotation_text=f"SPOT {spot:,.0f}", annotation_position="top right", annotation_font=dict(size=9, color="#00e5ff", family="JetBrains Mono, monospace"))
                                _fig_th_bleed.update_layout(
                                    title=dict(text="THETA BLEED (PTS/DAY) & VEGA SENSITIVITY (PTS/1% IV)", font=dict(color="#00e5ff", size=11, family="Inter, sans-serif")),
                                    height=310, autosize=True, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                                    font=dict(color=WHITE, family='Inter, sans-serif', size=10),
                                    legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1, bgcolor='rgba(0,0,0,0)', font=dict(size=9, color='#94a3b8')),
                                    margin=dict(l=45, r=45, t=35, b=25), hovermode='x unified'
                                )
                                _fig_th_bleed.update_xaxes(gridcolor='rgba(255,255,255,0.05)', showgrid=True, tickfont=dict(color='#cbd5e1', size=9, family='JetBrains Mono, monospace'))
                                _fig_th_bleed.update_yaxes(title=dict(text="Decay (pts/day)", font=dict(color="#868993", size=10)), gridcolor='rgba(255,255,255,0.05)', showgrid=True, tickfont=dict(color='#cbd5e1', size=9, family='JetBrains Mono, monospace'), secondary_y=False)
                                _fig_th_bleed.update_yaxes(title=dict(text="Vega (pts/1% IV)", font=dict(color="#00e676", size=10)), showgrid=False, tickfont=dict(color='#00e676', size=9, family='JetBrains Mono, monospace'), secondary_y=True)
                                _plotly_th_bleed = _fig_th_bleed.to_html(include_plotlyjs=False, full_html=False, div_id='theta-plotly-bleed')

                                # Plot 2: Directional Delta Slope & Gamma Convexity Curve
                                _fig_th_delta = make_subplots(specs=[[{"secondary_y": True}]])
                                _fig_th_delta.add_trace(go.Scatter(
                                    x=_valid_strikes, y=_ce_deltas,
                                    mode='lines+markers', name='Call Delta (Δ)',
                                    line=dict(color='#00e5ff', width=2),
                                    marker=dict(size=4),
                                    hovertemplate="<b>%{x:,.0f}</b> Call Δ: <b>%{y:+.2f}</b><extra></extra>"
                                ), secondary_y=False)
                                _fig_th_delta.add_trace(go.Scatter(
                                    x=_valid_strikes, y=_pe_deltas,
                                    mode='lines+markers', name='Put Delta (Δ)',
                                    line=dict(color='#ff7043', width=2),
                                    marker=dict(size=4),
                                    hovertemplate="<b>%{x:,.0f}</b> Put Δ: <b>%{y:+.2f}</b><extra></extra>"
                                ), secondary_y=False)
                                _fig_th_delta.add_trace(go.Scatter(
                                    x=_valid_strikes, y=_net_deltas,
                                    mode='lines', name='Net Delta (Δ)',
                                    line=dict(color='#ffffff', width=1.5, dash='dash'),
                                    hovertemplate="<b>%{x:,.0f}</b> Net Δ: <b>%{y:+.2f}</b><extra></extra>"
                                ), secondary_y=False)
                                _fig_th_delta.add_trace(go.Scatter(
                                    x=_valid_strikes, y=_gammas,
                                    mode='lines', name='Gamma Convexity (Γ)',
                                    line=dict(color='#c084fc', width=2.5),
                                    fill='tozeroy', fillcolor='rgba(192,132,252,0.10)',
                                    hovertemplate="<b>%{x:,.0f}</b> Gamma: <b>%{y:.5f}</b><extra></extra>"
                                ), secondary_y=True)

                                _fig_th_delta.add_hline(y=0, line_width=1, line_dash="solid", line_color="rgba(255,255,255,0.2)", secondary_y=False)
                                _fig_th_delta.add_vline(x=spot, line_width=1.5, line_dash="dash", line_color="#00e5ff", annotation_text=f"SPOT {spot:,.0f}", annotation_position="top right", annotation_font=dict(size=9, color="#00e5ff", family="JetBrains Mono, monospace"))
                                _fig_th_delta.update_layout(
                                    title=dict(text="DIRECTIONAL DELTA (Δ) & GAMMA (Γ) CONVEXITY", font=dict(color="#c084fc", size=11, family="Inter, sans-serif")),
                                    height=310, autosize=True, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                                    font=dict(color=WHITE, family='Inter, sans-serif', size=10),
                                    legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1, bgcolor='rgba(0,0,0,0)', font=dict(size=9, color='#94a3b8')),
                                    margin=dict(l=45, r=45, t=35, b=25), hovermode='x unified'
                                )
                                _fig_th_delta.update_xaxes(gridcolor='rgba(255,255,255,0.05)', showgrid=True, tickfont=dict(color='#cbd5e1', size=9, family='JetBrains Mono, monospace'))
                                _fig_th_delta.update_yaxes(title=dict(text="Delta (Δ)", font=dict(color="#868993", size=10)), range=[-1.05, 1.05], gridcolor='rgba(255,255,255,0.05)', showgrid=True, tickfont=dict(color='#cbd5e1', size=9, family='JetBrains Mono, monospace'), secondary_y=False)
                                _fig_th_delta.update_yaxes(title=dict(text="Gamma (Γ)", font=dict(color="#c084fc", size=10)), showgrid=False, tickfont=dict(color='#c084fc', size=9, family='JetBrains Mono, monospace'), secondary_y=True)
                                _plotly_th_gamma = _fig_th_delta.to_html(include_plotlyjs=False, full_html=False, div_id='theta-plotly-gamma')

                                # Asymmetry calculation (Points)
                                _th_diff_pts = _atm_pe_pts_abs - _atm_ce_pts_abs
                                if _atm_pe_pts_abs > _atm_ce_pts_abs * 1.02:
                                    _th_leader = "PUTS"
                                    _th_diff_pct = (_th_diff_pts / max(_atm_ce_pts_abs, 0.1)) * 100.0
                                    _th_verdict_col = "#ff7043"
                                    _th_verdict_text = f"PUTS BLEEDING FASTER (+{_th_diff_pct:.1f}% vs Calls)"
                                    _th_insight = f"Put buyers bleed faster (-{_th_diff_pts:.1f} pts/d more). Put writing offers higher time-decay harvest."
                                elif _atm_ce_pts_abs > _atm_pe_pts_abs * 1.02:
                                    _th_leader = "CALLS"
                                    _th_diff_pts = _atm_ce_pts_abs - _atm_pe_pts_abs
                                    _th_diff_pct = (_th_diff_pts / max(_atm_pe_pts_abs, 0.1)) * 100.0
                                    _th_verdict_col = "#00e5ff"
                                    _th_verdict_text = f"CALLS BLEEDING FASTER (+{_th_diff_pct:.1f}% vs Puts)"
                                    _th_insight = f"Call buyers bleed faster (-{_th_diff_pts:.1f} pts/d more). Call writing offers higher time-decay harvest."
                                else:
                                    _th_leader = "BALANCED"
                                    _th_diff_pts = 0.0
                                    _th_verdict_col = "#ffd54f"
                                    _th_verdict_text = "THETA DECAY IS SYMMETRICAL"
                                    _th_insight = "Time bleed is evenly matched between Calls and Puts."

                                # 25Δ Skew calculation
                                _idx_25c = min(range(len(_ce_deltas)), key=lambda i: abs(_ce_deltas[i] - 0.25)) if _ce_deltas else 0
                                _idx_25p = min(range(len(_pe_deltas)), key=lambda i: abs(_pe_deltas[i] - (-0.25))) if _pe_deltas else 0
                                _strike_25c = _valid_strikes[_idx_25c] if _idx_25c < len(_valid_strikes) else _atm_strike_th
                                _strike_25p = _valid_strikes[_idx_25p] if _idx_25p < len(_valid_strikes) else _atm_strike_th
                                _row_25c = _th_lookup.get((_strike_25c, 'CE'))
                                _row_25p = _th_lookup.get((_strike_25p, 'PE'))
                                _iv_25c = float(_row_25c['iv']) if _row_25c is not None and float(_row_25c.get('iv', 0)) > 0 else _atm_iv_th
                                _iv_25p = float(_row_25p['iv']) if _row_25p is not None and float(_row_25p.get('iv', 0)) > 0 else _atm_iv_th
                                _skew_25d = _iv_25p - _iv_25c

                                _atm_sig = max(_atm_iv_th / 100.0, 0.02)
                                _atm_gam_ce_val = _calc_merton_theta(_atm_strike_th, _atm_sig, 'CE')[1]
                                _atm_gam_pe_val = _calc_merton_theta(_atm_strike_th, _atm_sig, 'PE')[1]
                                _atm_gam_val_exact = _atm_gam_ce_val + _atm_gam_pe_val

                                # 50pt move alpha test (in points)
                                _em_pts = 50.0
                                _gamma_hazard_pts = 0.5 * _atm_gam_val_exact * (_em_pts ** 2)
                                _renorm_alpha = _atm_strad_day_pts / max(_gamma_hazard_pts, 0.01)
                                _alpha_color = GREEN if _renorm_alpha >= 1.0 else YELLOW if _renorm_alpha >= 0.7 else RED
                                _alpha_label = "Positive Theta Edge" if _renorm_alpha >= 1.0 else "Neutral Buffer" if _renorm_alpha >= 0.7 else "Gamma Hazard Zone"

                                theta_tab_html = f'''
                                <div style="display:flex; flex-direction:column; gap:14px;">

                                    <!-- 1. EXECUTIVE REAL-TIME DECAY & DIRECTION COCKPIT (3 CARDS IN POINTS) -->
                                    <div style="display:grid; grid-template-columns: repeat(3, 1fr); gap:12px;">
                                        <!-- Card 1: Net ATM Straddle Bleed -->
                                        <div class="card" style="border-top:3px solid #00e676; padding:14px; background:rgba(30,34,45,0.7);">
                                            <div style="display:flex; justify-content:space-between; align-items:center;">
                                                <span style="font-size:10px; font-weight:800; color:#00e676; text-transform:uppercase; letter-spacing:1px;">ATM STRADDLE BLEED (POINTS)</span>
                                                <span style="font-size:9px; font-weight:800; padding:2px 6px; border-radius:4px; background:rgba(0,230,118,0.15); color:#00e676; font-family:var(--font-mono);">{_curr_dte_th:.1f} DTE</span>
                                            </div>
                                            <div id="card-strad-day" style="font-size:26px; font-weight:900; color:#00e676; margin:6px 0 2px 0; font-family:var(--font-mono);">
                                                -{_atm_strad_day_pts:.1f} pts <span style="font-size:13px; font-weight:700; color:#ffffff;">/ day</span>
                                            </div>
                                            <div style="font-size:12px; font-weight:700; color:#ffd54f; margin-bottom:10px; font-family:var(--font-mono);">
                                                <span id="card-strad-hour">-{_atm_strad_1h_pts:.2f} pts / trading hr</span> · <span id="card-strad-prem">Prem: {_atm_strad_prem:.1f} pts</span>
                                            </div>
                                            <div style="background:#131722; border-radius:6px; padding:8px 10px; border:1px solid var(--border-subtle, #2a2e39); font-size:11px;">
                                                <div style="display:flex; justify-content:space-between; margin-bottom:3px;">
                                                    <span style="color:#868993;">Today's Intraday Range:</span>
                                                    <strong style="color:#ffffff; font-family:var(--font-mono);">{_th_intraday_min:.1f} — {_th_intraday_max:.1f} pts/d <span style="color:#ffd54f; font-size:10px;">({_th_pctile_day:.0f}% of range)</span></strong>
                                                </div>
                                                <div style="display:flex; justify-content:space-between;">
                                                    <span style="color:#868993;">Expiry Decay Pace:</span>
                                                    <strong style="color:#00e676; font-family:var(--font-mono);">{_atm_strad_yield:.1f}% / day <span style="color:#868993; font-size:10px;">(Theo: ~{_theo_atm_theta:.1f} pts/d)</span></strong>
                                                </div>
                                            </div>
                                        </div>

                                        <!-- Card 2: Put vs Call Decay Asymmetry -->
                                        <div class="card" style="border-top:3px solid {_th_verdict_col}; padding:14px; background:rgba(30,34,45,0.7);">
                                            <div style="display:flex; justify-content:space-between; align-items:center;">
                                                <span style="font-size:10px; font-weight:800; color:{_th_verdict_col}; text-transform:uppercase; letter-spacing:1px;">PUT-CALL DECAY ASYMMETRY</span>
                                                <span id="card-asym-verdict" style="font-size:9px; font-weight:800; padding:2px 6px; border-radius:4px; background:{_th_verdict_col}22; color:{_th_verdict_col}; font-family:var(--font-mono);">{_th_leader} BLEEDING FASTER</span>
                                            </div>
                                            <div id="card-asym-diff" style="font-size:26px; font-weight:900; color:{_th_verdict_col}; margin:6px 0 2px 0; font-family:var(--font-mono);">
                                                {'+' if _th_diff_pts > 0 else ''}{_th_diff_pts:.1f} pts <span style="font-size:13px; font-weight:700; color:#ffffff;">/ day edge</span>
                                            </div>
                                            <div style="font-size:12px; font-weight:700; color:#868993; margin-bottom:10px;">
                                                Call: <strong style="color:#00e5ff;">-{_atm_ce_pts_abs:.1f} pts/d</strong> · Put: <strong style="color:#ff7043;">-{_atm_pe_pts_abs:.1f} pts/d</strong>
                                            </div>
                                            <div style="background:#131722; border-radius:6px; padding:8px 10px; border:1px solid var(--border-subtle, #2a2e39); font-size:11px;">
                                                <div style="display:flex; justify-content:space-between; margin-bottom:3px;">
                                                    <span style="color:#868993;">ATM Net Delta (Δ):</span>
                                                    <strong style="color:{WHITE}; font-family:var(--font-mono);">{_atm_del_net:+.2f} <span style="color:#868993; font-size:10px;">(C: {_atm_del_ce:+.2f} | P: {_atm_del_pe:+.2f})</span></strong>
                                                </div>
                                                <div style="display:flex; justify-content:space-between;">
                                                    <span style="color:#868993;">25Δ Skew (Put - Call IV):</span>
                                                    <strong style="color:{WHITE}; font-family:var(--font-mono);">{_skew_25d:+.1f}% <span style="color:{_th_verdict_col}; font-size:10px;">({_iv_25p:.1f}% vs {_iv_25c:.1f}%)</span></strong>
                                                </div>
                                            </div>
                                        </div>

                                        <!-- Card 3: Break-Even Cushion & Greeks Sensitivity -->
                                        <div class="card" style="border-top:3px solid #00e5ff; padding:14px; background:rgba(30,34,45,0.7);">
                                            <div style="display:flex; justify-content:space-between; align-items:center;">
                                                <span style="font-size:10px; font-weight:800; color:#00e5ff; text-transform:uppercase; letter-spacing:1px;">DAILY BREAK-EVEN CUSHION</span>
                                                <span style="font-size:9px; font-weight:800; padding:2px 6px; border-radius:4px; background:rgba(0,229,255,0.15); color:#00e5ff; font-family:var(--font-mono);">THETA vs GAMMA &amp; VEGA</span>
                                            </div>
                                            <div id="card-cushion-pts" style="font-size:26px; font-weight:900; color:#ffffff; margin:6px 0 2px 0; font-family:var(--font-mono);">
                                                ±{_atm_strad_cushion:.1f} PTS
                                            </div>
                                            <div style="font-size:12px; font-weight:700; color:#00e5ff; margin-bottom:10px; font-family:var(--font-mono);">
                                                Safe Zone: <strong style="color:#ffffff;">{spot - _atm_strad_cushion:,.0f} – {spot + _atm_strad_cushion:,.0f}</strong>
                                            </div>
                                            <div style="background:#131722; border-radius:6px; padding:8px 10px; border:1px solid var(--border-subtle, #2a2e39); font-size:11px;">
                                                <div style="display:flex; justify-content:space-between; margin-bottom:3px;">
                                                    <span style="color:#868993;">Straddle Vega (ν):</span>
                                                    <strong style="color:#00e676; font-family:var(--font-mono);">±{_atm_strad_vega:.1f} pts <span style="color:#868993; font-size:10px;">per 1% IV shift</span></strong>
                                                </div>
                                                <div style="display:flex; justify-content:space-between;">
                                                    <span style="color:#868993;">ATM Gamma (Γ):</span>
                                                    <strong style="color:#c084fc; font-family:var(--font-mono);">{_atm_gam_val_exact:.5f} Γ <span style="color:{_alpha_color}; font-size:10px;">({_alpha_label})</span></strong>
                                                </div>
                                            </div>
                                        </div>
                                    </div>

                                    <!-- 2. TWO CLEAN DUAL-AXIS GREEKS VISUALIZERS -->
                                    <div style="display:grid; grid-template-columns: 1fr 1fr; gap:12px;">
                                        <div class="card" style="padding:12px; background:rgba(30,34,45,0.7); border:1px solid var(--border-card, #363c4e);">
                                            {_plotly_th_bleed}
                                        </div>
                                        <div class="card" style="padding:12px; background:rgba(30,34,45,0.7); border:1px solid var(--border-card, #363c4e);">
                                            {_plotly_th_gamma}
                                        </div>
                                    </div>

                                    <!-- 3. STREAMLINED 10-COLUMN PER-STRIKE OPTION PRICE COMPOSITION & GREEKS MATRIX -->
                                    <div class="card" style="padding:14px; background:var(--bg-surface, #1e222d); border:1px solid var(--border-card, #363c4e);">
                                        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:10px; flex-wrap:wrap; gap:8px;">
                                            <div style="font-size:12px; font-weight:800; color:#00e5ff; text-transform:uppercase; letter-spacing:1px;">
                                                PER-STRIKE OPTION PRICE COMPOSITION &amp; GREEKS DYNAMICS
                                            </div>
                                            <div style="font-size:11px; color:#868993;">
                                                Real-Time Decomposition into Intrinsic (Moneyness) &amp; Extrinsic (Time/Vol) premium in <strong style="color:#00e5ff;">Pure Points</strong>
                                            </div>
                                        </div>
                                        <div id="theta-table-container" style="max-height:420px; overflow-y:auto; scroll-behavior:smooth;">
                                            <table class="data-table" id="theta-decay-table" style="width:100%; border-collapse:collapse; font-size:11px;">
                                                <thead>
                                                    <tr style="position:sticky; top:0; background:#1e222d; z-index:2; border-bottom:1px solid #2a2a4a; color:#868993;">
                                                        <th style="padding:7px 8px; text-align:left;">Strike</th>
                                                        <th style="padding:7px 8px; text-align:right;">Dist</th>
                                                        <th style="padding:7px 8px; text-align:right; color:#00e5ff;">Call LTP (IV)</th>
                                                        <th style="padding:7px 8px; text-align:left; color:#00e5ff;">Call Composition</th>
                                                        <th style="padding:7px 8px; text-align:left; color:#00e5ff;">Call Greeks (Δ, Θ, ν)</th>
                                                        <th style="padding:7px 8px; text-align:right; color:#ff7043;">Put LTP (IV)</th>
                                                        <th style="padding:7px 8px; text-align:left; color:#ff7043;">Put Composition</th>
                                                        <th style="padding:7px 8px; text-align:left; color:#ff7043;">Put Greeks (Δ, Θ, ν)</th>
                                                        <th style="padding:7px 8px; text-align:right; color:#ffd54f;">Straddle Bleed</th>
                                                        <th style="padding:7px 8px; text-align:right; color:#00e676;">BE Cushion</th>
                                                    </tr>
                                                </thead>
                                                <tbody id="theta-decay-tbody">
                                                    {''.join(_theta_table_rows)}
                                                </tbody>
                                            </table>
                                        </div>
                                    </div>
                                </div>'''

                    except Exception as _th_e:
                        theta_tab_html = f'<div class="card"><p style="color:#ff4444;">Theta Decay module error: {str(_th_e)}</p></div>'


                    # ── Probability Cone tab removed per user request ──
                    prob_tab_html = ''

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

                            _is_shares = bool((_df_mm['oi'].max() > 100_000)) if not _df_mm.empty else False
                            _total_shares = _df_mm['oi'] if _is_shares else (_df_mm['oi'] * _lot_mm)

                            # ── INSTITUTIONAL DEALER GEX POSITIONING & HEDGING FLOWS ──
                            # Standard dealer positioning:
                            # Dealers are liquidity providers. 
                            # CE: Dealer Long Call delta & gamma (+1).
                            # PE: Dealer Short Put (+delta, -gamma) (-1).
                            _df_mm['gex_lots_50pt'] = _df_mm.apply(
                                lambda r: (50.0 * r['gamma_mm'] * (r['oi'] if _is_shares else r['oi'] * _lot_mm) / _lot_mm) if r['type'] == 'CE'
                                else (-50.0 * r['gamma_mm'] * (r['oi'] if _is_shares else r['oi'] * _lot_mm) / _lot_mm), axis=1)
                            _df_mm['gex_cr_100pt'] = _df_mm.apply(
                                lambda r: (100.0 * r['gamma_mm'] * (r['oi'] if _is_shares else r['oi'] * _lot_mm) * spot / 1e7) if r['type'] == 'CE'
                                else (-100.0 * r['gamma_mm'] * (r['oi'] if _is_shares else r['oi'] * _lot_mm) * spot / 1e7), axis=1)
                            _df_mm['dex_lots'] = _df_mm.apply(
                                lambda r: (r['delta_mm'] * (r['oi'] if _is_shares else r['oi'] * _lot_mm) / _lot_mm) if r['type'] == 'CE'
                                else (-r['delta_mm'] * (r['oi'] if _is_shares else r['oi'] * _lot_mm) / _lot_mm), axis=1)

                            _net_gex_lots = float(_df_mm['gex_lots_50pt'].sum())
                            _net_gex_crores = float(_df_mm['gex_cr_100pt'].sum())
                            _net_dex_lots = float(_df_mm['dex_lots'].sum())
                            _net_dex_crores = float((_net_dex_lots * _lot_mm * spot) / 1e7)
                            _hedge_lots_50pt = -_net_gex_lots

                            # Dealer Gamma Regime
                            if _net_gex_crores >= 0:
                                _regime_title = "LONG GAMMA · PINNING REGIME"
                                _regime_badge = "STABILIZING FLOW"
                                _regime_badge_col = GREEN
                                _regime_desc = "Dealers counter-trade price swings (sell into rallies, buy dips). Volatility is dampened; mean-reversion & pinning expected within corridor."
                                _hedge_act_txt = f"SELL ₹{abs(_net_gex_crores * 0.5):.1f} Cr"
                                _hedge_act_col = GREEN
                                _hedge_impact = "Dampens rally momentum"
                            else:
                                _regime_title = "SHORT GAMMA · BREAKOUT REGIME"
                                _regime_badge = "SLIPPERY ACCELERATION"
                                _regime_badge_col = RED
                                _regime_desc = "Dealers chase momentum (buy into rallies, sell into breakdowns). Move magnification risk is elevated; expect rapid breakout acceleration."
                                _hedge_act_txt = f"BUY ₹{abs(_net_gex_crores * 0.5):.1f} Cr"
                                _hedge_act_col = RED
                                _hedge_impact = "Magnifies rally breakout"

                            # Dealer Net Delta Bias
                            if _net_dex_crores < 0:
                                _dex_title = "SHORT DELTA"
                                _dex_badge = "BULLISH SQUEEZE BIAS"
                                _dex_badge_col = "#00e5ff"
                                _dex_desc = f"Dealers short ₹{abs(_net_dex_crores):.1f} Cr notional. Upward breakout triggers forced short covering."
                            else:
                                _dex_title = "LONG DELTA"
                                _dex_badge = "BEARISH DRAG BIAS"
                                _dex_badge_col = "#ff9100"
                                _dex_desc = f"Dealers long ₹{abs(_net_dex_crores):.1f} Cr notional. Downward break triggers forced long liquidation."

                            # Key Institutional Landmarks
                            _ce_df = _df_mm[_df_mm['type'] == 'CE']
                            _pe_df = _df_mm[_df_mm['type'] == 'PE']
                            _s_obj = locals().get('seller') or locals().get('s')
                            _call_wall = int(_s_obj.get('call_wall', 0)) if isinstance(_s_obj, dict) and _s_obj.get('call_wall') else (int(_ce_df.loc[_ce_df['oi'].idxmax(), 'strike']) if not _ce_df.empty else 0)
                            _put_wall = int(_s_obj.get('put_wall', 0)) if isinstance(_s_obj, dict) and _s_obj.get('put_wall') else (int(_pe_df.loc[_pe_df['oi'].idxmax(), 'strike']) if not _pe_df.empty else 0)
                            _max_pain = int(_s_obj.get('max_pain', 0)) if isinstance(_s_obj, dict) and _s_obj.get('max_pain') else 0
                            if _max_pain == 0 and not df_chain.empty:
                                try:
                                    from KeyLevelsEngine import KeyLevelsEngine
                                    _kle = KeyLevelsEngine()
                                    _max_pain = int(_kle.calculate_max_pain(df_chain))
                                except Exception:
                                    _max_pain = round(spot / 50) * 50

                            _up_barrier = _call_wall + 25 if _call_wall > 0 else 0
                            _down_barrier = _put_wall - 25 if _put_wall > 0 else 0

                            # Per-strike summary table
                            _ce_map = _ce_df.set_index('strike')['oi']
                            _pe_map = _pe_df.set_index('strike')['oi']
                            _by_strike = _df_mm.groupby('strike').agg(
                                gex_lots=('gex_lots_50pt', 'sum'),
                                gex_cr=('gex_cr_100pt', 'sum'),
                                dex_lots=('dex_lots', 'sum')
                            ).reset_index().sort_values('strike', ascending=False)
                            _by_strike['oi_ce'] = _by_strike['strike'].map(_ce_map).fillna(0)
                            _by_strike['oi_pe'] = _by_strike['strike'].map(_pe_map).fillna(0)

                            _mm_rows = ''
                            _atm_mm_strike = min(_by_strike['strike'], key=lambda s: abs(s - spot)) if not _by_strike.empty else 0
                            for _, _mr in _by_strike.iterrows():
                                _msk = int(_mr['strike'])
                                _dist = _msk - spot
                                _ce_oi = int(_mr['oi_ce'])
                                _pe_oi = int(_mr['oi_pe'])
                                _g_cr = float(_mr['gex_cr'])
                                _d_l = float(_mr['dex_lots'])
                                _d_cr = float((_d_l * _lot_mm * spot) / 1e7)

                                _is_atm = (_msk == _atm_mm_strike)
                                _is_cw = (_msk == _call_wall)
                                _is_pw = (_msk == _put_wall)
                                _is_mp = (_msk == _max_pain)

                                if _is_cw:
                                    _role_badge = '<span style="background:rgba(255,51,102,0.2);color:#ff3366;font-weight:700;padding:2px 6px;border-radius:4px;">CALL WALL (Ceiling)</span>'
                                elif _is_pw:
                                    _role_badge = '<span style="background:rgba(0,230,118,0.2);color:#00e676;font-weight:700;padding:2px 6px;border-radius:4px;">PUT WALL (Floor)</span>'
                                elif _is_mp:
                                    _role_badge = '<span style="background:rgba(255,214,0,0.2);color:#ffd600;font-weight:700;padding:2px 6px;border-radius:4px;">MAX PAIN (Pin)</span>'
                                elif _is_atm:
                                    _role_badge = '<span style="background:rgba(79,195,247,0.2);color:#4fc3f7;font-weight:700;padding:2px 6px;border-radius:4px;">ATM ZONE</span>'
                                elif _g_cr > 2.0:
                                    _role_badge = '<span style="color:#00e676;">PIN ABSORPTION</span>'
                                elif _g_cr < -2.0:
                                    _role_badge = '<span style="color:#ff3366;">SQUEEZE ACCEL</span>'
                                else:
                                    _role_badge = '<span style="color:#777;">TRANSITION</span>'

                                _row_id = 'id="mm-row-atm"' if _is_atm else ''
                                _row_atm_attr = 'data-is-atm="true"' if _is_atm else ''
                                _row_bg = "rgba(79,195,247,0.08)" if _is_atm else ("rgba(255,51,102,0.05)" if _is_cw else ("rgba(0,230,118,0.05)" if _is_pw else "transparent"))
                                _mm_rows += (
                                    f'<tr {_row_id} {_row_atm_attr} style="background:{_row_bg};border-bottom:1px solid #1a1a2e;">'
                                    f'<td style="padding:7px 10px;font-weight:700;color:{ACCENT if _is_atm else WHITE};">{_msk}{" ◄ ATM" if _is_atm else ""}</td>'
                                    f'<td style="padding:7px 10px;text-align:right;color:{MUTED};">{_dist:+.0f}</td>'
                                    f'<td style="padding:7px 10px;text-align:right;color:{WHITE};">{_ce_oi:,}</td>'
                                    f'<td style="padding:7px 10px;text-align:right;color:{WHITE};">{_pe_oi:,}</td>'
                                    f'<td style="padding:7px 10px;text-align:right;font-weight:700;color:{GREEN if _g_cr>0 else RED};">{_g_cr:+.1f} Cr</td>'
                                    f'<td style="padding:7px 10px;text-align:right;font-weight:600;color:{"#00e5ff" if _d_cr<0 else "#ff9100"};">{_d_cr:+.1f} Cr</td>'
                                    f'<td style="padding:7px 10px;text-align:left;">{_role_badge}</td>'
                                    f'</tr>'
                                )

                            mm_tab_html = f'''
                            <!-- Market Maker Gamma Pinning & Order Flow Absorption Terminal -->
                            <div id="gamma-explosion-root" class="gamma-explosion-wrap">
                                <!-- Strike GEX Ladder & Pin Corridor -->
                                <div class="ge-card">
                                    <div class="ge-card-header">
                                        <div class="ge-card-title">
                                            <span style="font-weight:700;">STRIKE GEX LADDER &amp; GEX DISTRIBUTION</span>
                                        </div>
                                        <span class="num-mono" style="font-size:11px; color:{MUTED};">Signed Dealer Gamma (₹ Cr)</span>
                                    </div>
                                    <div id="ge-ladder-container" class="ge-ladder-wrap">
                                        <div style="color:{MUTED}; font-size:12px; padding:10px;">Loading strike GEX ladder...</div>
                                    </div>
                                </div>
                            </div>

                            <div style="margin-top: 24px; margin-bottom: 14px; border-top: 1px solid rgba(255,255,255,0.08); padding-top: 16px;">
                                <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:8px;">
                                    <div style="font-size: 13px; font-weight: 800; letter-spacing: 1.2px; color: {ACCENT}; text-transform: uppercase;">
                                        GEX &amp; DEALER POSITIONING
                                    </div>
                                    <span style="font-size:11px; color:{MUTED};">Dealer Gamma &amp; Delta Exposure in ₹ Crores (Notional)</span>
                                </div>
                            </div>

                            <!-- Executive Metric Cards -->
                            <div style="display:grid;grid-template-columns:repeat(auto-fit, minmax(220px, 1fr));gap:12px;margin-bottom:14px;">
                                <!-- Card 1: Gamma Regime -->
                                <div id="dealer-regime-card" class="card" style="border-top:4px solid {_regime_badge_col};">
                                    <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:6px;">
                                        <div style="color:{ACCENT};font-size:11px;font-weight:800;letter-spacing:1px;">DEALER GAMMA REGIME</div>
                                        <span id="dealer-regime-badge" style="background:rgba(255,255,255,0.06);color:{_regime_badge_col};font-size:10px;font-weight:800;padding:2px 6px;border-radius:4px;">{_regime_badge}</span>
                                    </div>
                                    <div id="dealer-gex-val" style="font-size:20px;font-weight:800;color:{_regime_badge_col};margin-bottom:2px;">{_net_gex_crores:+.1f} Cr</div>
                                    <div style="font-size:11px;color:{MUTED};margin-bottom:8px;">Net GEX per 100-pt move</div>
                                    <div id="dealer-regime-desc" style="font-size:11px;color:#aaa;line-height:1.4;">{_regime_desc}</div>
                                </div>

                                <!-- Card 2: Net Delta Exposure -->
                                <div id="dealer-dex-card" class="card" style="border-top:4px solid {_dex_badge_col};">
                                    <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:6px;">
                                        <div style="color:{ACCENT};font-size:11px;font-weight:800;letter-spacing:1px;">DEALER DELTA (DEX)</div>
                                        <span id="dealer-dex-badge" style="background:rgba(255,255,255,0.06);color:{_dex_badge_col};font-size:10px;font-weight:800;padding:2px 6px;border-radius:4px;">{_dex_title}</span>
                                    </div>
                                    <div id="dealer-dex-val" style="font-size:20px;font-weight:800;color:{_dex_badge_col};margin-bottom:2px;">₹{_net_dex_crores:+.1f} Cr</div>
                                    <div style="font-size:11px;color:{MUTED};margin-bottom:8px;">Notional Exposure</div>
                                    <div id="dealer-dex-desc" style="font-size:11px;color:#aaa;line-height:1.4;">{_dex_desc}</div>
                                </div>

                                <!-- Card 3: 50-pt Hedge Requirement -->
                                <div id="dealer-hedge-card" class="card" style="border-top:4px solid {_hedge_act_col};">
                                    <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:6px;">
                                        <div style="color:{ACCENT};font-size:11px;font-weight:800;letter-spacing:1px;">50-PT HEDGE FLOW</div>
                                        <span id="dealer-hedge-badge" style="background:rgba(255,255,255,0.06);color:{_hedge_act_col};font-size:10px;font-weight:800;padding:2px 6px;border-radius:4px;">REBALANCING</span>
                                    </div>
                                    <div id="dealer-hedge-val" style="font-size:20px;font-weight:800;color:{_hedge_act_col};margin-bottom:2px;">{_hedge_act_txt}</div>
                                    <div style="font-size:11px;color:{MUTED};margin-bottom:8px;">Required on a +50 pt Nifty advance</div>
                                    <div id="dealer-hedge-desc" style="font-size:11px;color:#aaa;line-height:1.4;">Dealers forced flow {'dampens upside momentum' if _net_gex_crores >= 0 else 'accelerates upside momentum'}.</div>
                                </div>

                                <!-- Card 4: Pin Anchor / Max Pain -->
                                <div id="dealer-pain-card" class="card" style="border-top:4px solid {YELLOW};">
                                    <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:6px;">
                                        <div style="color:{ACCENT};font-size:11px;font-weight:800;letter-spacing:1px;">EXPIRY PIN ANCHOR</div>
                                        <span style="background:rgba(255,214,0,0.15);color:{YELLOW};font-size:10px;font-weight:800;padding:2px 6px;border-radius:4px;">MAX PAIN</span>
                                    </div>
                                    <div id="dealer-pain-val" style="font-size:20px;font-weight:800;color:{YELLOW};margin-bottom:2px;">{_max_pain}</div>
                                    <div id="dealer-pain-dist" style="font-size:11px;color:{MUTED};margin-bottom:8px;">{abs(spot - _max_pain):.0f} pts away ({'Above' if spot > _max_pain else 'Below'} Spot)</div>
                                    <div style="font-size:11px;color:#aaa;line-height:1.4;">Option sellers' maximum profitability strike; acts as magnetic gravity near expiry.</div>
                                </div>
                            </div>

                            <!-- Tactical Corridor Banner -->
                            <div class="card" style="margin-bottom:14px;border-left:4px solid {ACCENT};background:#0e1022;">
                                <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;flex-wrap:wrap;gap:8px;">
                                    <span style="color:{ACCENT};font-size:12px;font-weight:800;letter-spacing:1px;">TACTICAL EXPIRY CORRIDOR &amp; VOLATILITY TRIGGERS</span>
                                    <div style="display:flex;gap:8px;font-size:11px;">
                                        <span id="dealer-upper-barrier" style="background:rgba(255,51,102,0.15);color:#ff3366;padding:2px 8px;border-radius:4px;font-weight:700;">Upper Barrier: {_up_barrier}</span>
                                        <span id="dealer-lower-barrier" style="background:rgba(0,230,118,0.15);color:#00e676;padding:2px 8px;border-radius:4px;font-weight:700;">Lower Barrier: {_down_barrier}</span>
                                    </div>
                                </div>
                                <div id="dealer-tactical-text" style="font-size:12px;color:#ccc;line-height:1.5;">
                                    Safe Pinning Corridor: <strong style="color:{WHITE};">{_put_wall} — {_call_wall}</strong> ({_call_wall - _put_wall:.0f} pts). 
                                    A sustained push <strong>above {_up_barrier}</strong> triggers a violent Call Gamma Squeeze as dealers cover short strikes. 
                                    A decisive break <strong>below {_down_barrier}</strong> sparks rapid put unwinding and cascade selling.
                                </div>
                            </div>

                            <!-- Per-Strike Inventory & Dealer Exposure Table -->
                            <div class="card">
                                <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;">
                                    <div style="color:{ACCENT};font-size:12px;font-weight:800;letter-spacing:1.5px;">PER-STRIKE INVENTORY &amp; DEALER EXPOSURE</div>
                                    <div style="font-size:11px;color:{MUTED};">Open Interest · Signed Dealer Gamma (₹ Cr) &amp; Delta Exposure</div>
                                </div>
                                <div id="dealer-inventory-container" style="max-height:420px;overflow-y:auto;scroll-behavior:smooth;">
                                    <table style="width:100%;border-collapse:collapse;font-size:11px;">
                                        <thead>
                                            <tr style="color:{MUTED};border-bottom:1px solid #2a2a4a;position:sticky;top:0;background:{CARD_BG};z-index:2;">
                                                <th style="padding:7px 10px;text-align:left;">Strike</th>
                                                <th style="padding:7px 10px;text-align:right;">Dist (pts)</th>
                                                <th style="padding:7px 10px;text-align:right;">Call OI</th>
                                                <th style="padding:7px 10px;text-align:right;">Put OI</th>
                                                <th style="padding:7px 10px;text-align:right;">Dealer Gamma (₹ Cr)</th>
                                                <th style="padding:7px 10px;text-align:right;">Dealer Delta (₹ Cr)</th>
                                                <th style="padding:7px 10px;text-align:left;">Dealer Role / Wall</th>
                                            </tr>
                                        </thead>
                                        <tbody id="dealer-inventory-tbody">{_mm_rows}</tbody>
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

                        # ── ECONOMETRIC & STRANGLE SIZING METRICS ──
                        _econ = regime_snapshot.get('econometric', {})
                        _sz = regime_snapshot.get('strangle_sizing', {})

                        _semi = _econ.get('semi_variance', {})
                        _rv_plus_pct = _semi.get('rv_plus_pct', 50.0)
                        _rv_minus_pct = _semi.get('rv_minus_pct', 50.0)
                        _vai = _semi.get('vai', 0.0)
                        _semi_bias = _semi.get('bias', 'BALANCED')
                        _semi_interp = _semi.get('interpretation', 'Symmetric volatility dispersion')

                        _har = _econ.get('har_forecast', {})
                        _har_1d = _har.get('forecast_1d', _rv_5d)
                        _har_5d = _har.get('forecast_5d', _rv_20d)

                        _jump = _econ.get('jump_decomposition', {})
                        _jump_ratio_pct = _jump.get('jump_ratio_pct', 8.5)
                        _jump_badge = _jump.get('action_badge', 'STRUCTURAL_FLOW')
                        _jump_desc = _jump.get('description', 'Continuous volatility flow')

                        _fvrp = _econ.get('forward_vrp', {})
                        _fvrp_val = _fvrp.get('vrp_5d', _vrp_val)
                        _fvrp_verdict = _fvrp.get('verdict', 'FAVORABLE_PREMIUM')
                        _atm_iv = float(regime_snapshot.get('atm_iv', 0) or _econ.get('atm_iv', 0) or _pd_iv or _live_iv or _live_hv or _hv_20d or 12.0)

                        _sz_opt_lots = _sz.get('optimal_lots', 14)
                        _sz_base_lots = _sz.get('base_lots', 11)
                        _sz_ce_lots = _sz.get('ce_lots', 7)
                        _sz_pe_lots = _sz.get('pe_lots', 7)
                        _sz_badge = _sz.get('verdict_badge', '⚡ INCREASE SIZE (125%)')
                        _sz_color = _sz.get('verdict_color', GREEN)
                        _sz_deploy_pct = _sz.get('deployment_pct', 125.0)
                        _sz_stress_loss = _sz.get('stress_test', {}).get('estimated_loss_inr', 18000.0)
                        _sz_stress_pct = _sz.get('stress_test', {}).get('risk_pct_of_capital', 0.90)

                        # ── MULTI-HORIZON MACRO & PRICE-GROUNDED VRP METRICS ──
                        _macro = regime_snapshot.get('macro', {})
                        _macro_stage = _macro.get('stage', 'MACRO MID-RANGE')
                        _macro_min = _macro.get('hv_min', _hv_20d * 0.7)
                        _macro_max = _macro.get('hv_max', _hv_20d * 1.5)
                        _curve_badge = _macro.get('curve_badge', 'CONTANGO (Theta Favorable)')
                        _curve_color = _macro.get('curve_color', GREEN)

                        _pvrp = regime_snapshot.get('price_vrp', {})
                        _lot = int(_pvrp.get('lot_size', _get_cfg("nifty_lot_size", 65)))

                        # ── DAILY 1D METRICS ──
                        _daily_iv_pts = _pvrp.get('daily_iv_pts', 0.0)
                        if _daily_iv_pts <= 0:
                            _daily_iv_pts = spot * (_atm_iv / 100.0) / 15.874
                        _daily_rv_pts = _pvrp.get('daily_rv_pts', 0.0)
                        if _daily_rv_pts <= 0:
                            _daily_rv_pts = spot * (_rv_cons / 100.0) / 15.874
                        _daily_vrp_pts = _pvrp.get('daily_vrp_pts', _daily_iv_pts - _daily_rv_pts)
                        _daily_iv_inr = _pvrp.get('daily_iv_inr', _daily_iv_pts * _lot)
                        _daily_rv_inr = _pvrp.get('daily_rv_inr', _daily_rv_pts * _lot)
                        _daily_vrp_inr = _pvrp.get('daily_vrp_inr', _daily_vrp_pts * _lot)
                        _daily_edge_pct = (_daily_vrp_pts / _daily_rv_pts * 100.0) if _daily_rv_pts > 0 else 0.0

                        # ── WEEKLY 5 DTE STRADDLE METRICS ──
                        _straddle_iv_pts = _pvrp.get('straddle_iv_pts', 0.0)
                        if _straddle_iv_pts <= 0:
                            _straddle_iv_pts = 0.8 * spot * (_atm_iv / 100.0) * ((5.0 / 365.0) ** 0.5)
                        _straddle_rv_pts = _pvrp.get('straddle_rv_pts', 0.0)
                        if _straddle_rv_pts <= 0:
                            _straddle_rv_pts = 0.8 * spot * (_rv_cons / 100.0) * ((5.0 / 365.0) ** 0.5)
                        _straddle_vrp_pts = _pvrp.get('straddle_vrp_pts', _straddle_iv_pts - _straddle_rv_pts)
                        _straddle_iv_inr = _pvrp.get('straddle_iv_inr', _straddle_iv_pts * _lot)
                        _straddle_rv_inr = _pvrp.get('straddle_rv_inr', _straddle_rv_pts * _lot)
                        _straddle_vrp_inr = _pvrp.get('straddle_vrp_inr', _straddle_vrp_pts * _lot)
                        _straddle_edge_pct = _pvrp.get('straddle_edge_pct', ((_straddle_iv_pts - _straddle_rv_pts) / _straddle_rv_pts * 100.0) if _straddle_rv_pts > 0 else 0.0)

                        # Weekly 5D 1-Sigma Expected Move (Spot * IV * sqrt(5/365))
                        _weekly_iv_pts = spot * (_atm_iv / 100.0) * ((5.0 / 365.0) ** 0.5)
                        _weekly_rv_pts = spot * (_rv_5d / 100.0) * ((5.0 / 365.0) ** 0.5)
                        _weekly_iv_inr = _weekly_iv_pts * _lot

                        _edge_ratio = _pvrp.get('sellers_edge_ratio', (_atm_iv / _rv_cons) if _rv_cons > 0 else 1.0)
                        _notional_per_lot = (spot * _lot) / 100_000.0

                        _hpts = regime_snapshot.get('horizon_points', {})
                        _pts_1d = _hpts.get('1d', _daily_rv_pts)
                        _pts_5d = _hpts.get('5d', spot * (_rv_5d / 100.0) / 15.874)
                        _pts_20d = _hpts.get('20d', spot * (_rv_20d / 100.0) / 15.874)
                        _pts_60d = _hpts.get('60d', spot * (_rv_60d / 100.0) / 15.874)
                        _pts_1y = _hpts.get('1y', spot * (_hv_20d / 100.0) / 15.874)

                        _har_1d_pts = spot * (_har_1d / 100.0) / 15.874
                        _har_5d_pts = spot * (_har_5d / 100.0) / 15.874

                        # ── MULTI-HORIZON TERM STRUCTURE COMPARISON & PAST NUMBERS ──
                        _vol_hist = regime_snapshot.get('history', {})
                        _rv_10d = float(regime_snapshot['rv'].get('10d', (_rv_5d + _rv_20d)/2.0))
                        _pts_10d = spot * (_rv_10d / 100.0) / 15.874

                        _term_rows = [
                            ('1D Session (Intraday)', '1 Day', float(_rv_intra if _rv_intra > 0 else _rv_cons), float(_atm_iv), float(_pts_1d), float(_pts_1d * _lot)),
                            ('5D Weekly (5 DTE)', '5 Days', float(_rv_5d), float(_atm_iv), float(_pts_5d), float(_pts_5d * _lot)),
                            ('10D Bi-Weekly', '10 Days', float(_rv_10d), float(_atm_iv), float(_pts_10d), float(_pts_10d * _lot)),
                            ('20D Monthly (Benchmark)', '20 Days', float(_rv_cons), float(_atm_iv), float(_pts_20d), float(_pts_20d * _lot)),
                            ('60D Quarterly', '60 Days', float(_rv_60d), float(_atm_iv), float(_pts_60d), float(_pts_60d * _lot)),
                            ('252D Annual Macro', '1 Year', float(_hv_20d), float(_atm_iv), float(_pts_1y), float(_pts_1y * _lot)),
                        ]

                        _term_table_rows_html = ''
                        for _h_title, _h_period, _h_rv, _h_iv_val, _h_pt, _h_inr_val in _term_rows:
                            _h_vrp = _h_iv_val - _h_rv
                            _vrp_color = '#00e676' if _h_vrp > 0.5 else ('#ff3366' if _h_vrp < -0.5 else '#ffd54f')
                            _vrp_badge = 'PREMIUM OVERPRICED' if _h_vrp > 0.5 else ('DISCOUNTED CHEAP' if _h_vrp < -0.5 else 'FAIR VALUE')
                            _term_table_rows_html += f'''
                            <tr style="border-bottom:1px solid rgba(255,255,255,0.05); font-family:var(--font-mono, monospace);">
                                <td style="padding:9px 12px; text-align:left; font-family:'Inter', sans-serif; font-weight:700; color:{WHITE};">{_h_title}</td>
                                <td style="padding:9px 12px; text-align:center; color:#64748b;">{_h_period}</td>
                                <td style="padding:9px 12px; text-align:center; font-weight:800; color:#00e5ff;">{_h_rv:.2f}%</td>
                                <td style="padding:9px 12px; text-align:center; font-weight:700; color:#ffd54f;">{_h_iv_val:.2f}%</td>
                                <td style="padding:9px 12px; text-align:center; font-weight:800; color:{_vrp_color};">{_h_vrp:+.2f}%</td>
                                <td style="padding:9px 12px; text-align:right; font-weight:700; color:{WHITE};">±{_h_pt:.1f} pts</td>
                                <td style="padding:9px 12px; text-align:right; font-weight:800; color:#38bdf8;">±₹{_h_inr_val:,.0f}</td>
                                <td style="padding:9px 12px; text-align:left;">
                                    <span style="background:{_vrp_color}18; color:{_vrp_color}; font-size:10px; font-weight:800; padding:2px 8px; border-radius:4px; border:1px solid {_vrp_color}44;">
                                        {_vrp_badge}
                                    </span>
                                </td>
                            </tr>
                            '''

                        # ── PAST NUMBERS & HISTORICAL VOLATILITY EVOLUTION CHART ──
                        _p_cur_rv20 = _vol_hist.get('cur_rv20', _rv_cons)
                        _p_yest_rv20 = _vol_hist.get('yesterday_rv20', _rv_cons)
                        _p_d5_rv20 = _vol_hist.get('d5_ago_rv20', _rv_cons)
                        _p_d20_rv20 = _vol_hist.get('d20_ago_rv20', _rv_cons)

                        _chg_yest = _p_cur_rv20 - _p_yest_rv20
                        _chg_d5 = _p_cur_rv20 - _p_d5_rv20
                        _chg_d20 = _p_cur_rv20 - _p_d20_rv20

                        try:
                            fig_vol_history = go.Figure()
                            _h_dates = _vol_hist.get('dates', [])
                            _h_rv20_vals = _vol_hist.get('rv20', [])
                            _h_rv5_vals = _vol_hist.get('rv5', [])

                            if _h_dates and len(_h_dates) == len(_h_rv20_vals):
                                fig_vol_history.add_trace(
                                    go.Scatter(
                                        x=_h_dates,
                                        y=_h_rv20_vals,
                                        mode='lines+markers',
                                        name='20D Realized Vol (RV)',
                                        line=dict(color='#00e5ff', width=2.5),
                                        marker=dict(size=4, color='#00e5ff'),
                                        hovertemplate="<b>%{x|%d %b %Y}</b><br>20D Realized Vol: <b>%{y:.2f}%</b><extra></extra>"
                                    )
                                )
                                if _h_rv5_vals and len(_h_rv5_vals) == len(_h_dates):
                                    fig_vol_history.add_trace(
                                        go.Scatter(
                                            x=_h_dates,
                                            y=_h_rv5_vals,
                                            mode='lines',
                                            name='5D Short-Term RV',
                                            line=dict(color='#ffd54f', width=1.8, dash='dot'),
                                            hovertemplate="<b>%{x|%d %b %Y}</b><br>5D Short-Term RV: <b>%{y:.2f}%</b><extra></extra>"
                                        )
                                    )
                                fig_vol_history.add_trace(
                                    go.Scatter(
                                        x=_h_dates,
                                        y=[float(_atm_iv)] * len(_h_dates),
                                        mode='lines',
                                        name=f"Live ATM IV ({_atm_iv:.1f}%)",
                                        line=dict(color='#c084fc', width=1.8, dash='dash'),
                                        hovertemplate="<b>Live ATM IV Benchmark</b>: <b>%{y:.2f}%</b><extra></extra>"
                                    )
                                )
                                _med_val = _macro.get('hv_median', _hv_20d)
                                if _med_val > 0:
                                    fig_vol_history.add_trace(
                                        go.Scatter(
                                            x=_h_dates,
                                            y=[float(_med_val)] * len(_h_dates),
                                            mode='lines',
                                            name=f"1Y Median ({_med_val:.1f}%)",
                                            line=dict(color='#64748b', width=1.2, dash='dot'),
                                            hovertemplate="<b>1-Year Vol Median</b>: <b>%{y:.2f}%</b><extra></extra>"
                                        )
                                    )

                            fig_vol_history.update_layout(
                                height=280,
                                margin=dict(l=45, r=40, t=20, b=25),
                                paper_bgcolor='rgba(0,0,0,0)',
                                plot_bgcolor='rgba(0,0,0,0)',
                                showlegend=True,
                                legend=dict(
                                    orientation="h",
                                    yanchor="bottom",
                                    y=1.02,
                                    xanchor="right",
                                    x=1,
                                    font=dict(size=10, color="#94a3b8", family="JetBrains Mono, monospace"),
                                    bgcolor='rgba(15,23,42,0.6)'
                                ),
                                xaxis=dict(
                                    title=dict(text="Date", font=dict(color="#868993", size=10)),
                                    showgrid=True,
                                    gridcolor='rgba(255, 255, 255, 0.04)',
                                    tickfont=dict(size=10, color="#868993", family="JetBrains Mono, monospace"),
                                    type='date',
                                    tickformat='%d %b',
                                    hoverformat='%d %b %Y'
                                ),
                                yaxis=dict(
                                    title=dict(text="Annualized Vol (%)", font=dict(color="#00e5ff", size=10)),
                                    showgrid=True,
                                    gridcolor='rgba(255, 255, 255, 0.04)',
                                    tickfont=dict(size=10, color="#868993", family="JetBrains Mono, monospace")
                                ),
                                hovermode="x unified"
                            )
                            vol_history_chart_html = fig_vol_history.to_html(include_plotlyjs=False, full_html=False, div_id='vol-history-evolution-plot', default_height='280px', default_width='100%')
                        except Exception as _hist_err:
                            vol_history_chart_html = f'<div style="color:#ff4444; padding:20px;">History chart error: {_hist_err}</div>'

                        regime_tab_html = f'''
                        <div style="display:flex; flex-direction:column; gap:14px;">

                            <!-- 1. Price-Grounded VRP & Expected Move Matrix -->
                            <div class="card" style="padding:16px; background:var(--bg-surface, #1e222d); border:1px solid var(--border-card, #363c4e);">
                                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:12px; flex-wrap:wrap; gap:8px;">
                                    <div style="font-size:12px; font-weight:800; color:{ACCENT}; text-transform:uppercase; letter-spacing:1px;">
                                        💰 VOLATILITY RISK PREMIUM IN PRICE (POINTS & RUPEES)
                                    </div>
                                    <div style="font-size:11px; color:{MUTED};">
                                        NIFTY Spot: <strong style="color:{WHITE}; font-family:var(--font-mono);">{spot:,.2f}</strong> · Lot Size: <strong style="color:{ACCENT};">{_lot} Qty</strong>
                                    </div>
                                </div>
                                <div style="display:grid; grid-template-columns: repeat(4, 1fr); gap:10px;">
                                    <!-- Card 1: Daily Points VRP Buffer & Edge -->
                                    <div class="metric-box" style="padding:14px; text-align:left; background:rgba(30, 34, 45, 0.7); border-top:3px solid {'#00e676' if _daily_vrp_pts > 0 else '#ff3366'}; display:flex; flex-direction:column; justify-content:space-between;">
                                        <div>
                                            <div style="display:flex; justify-content:space-between; align-items:center;">
                                                <div class="metric-label" style="font-size:10px;">DAILY VRP PRICE BUFFER (1D)</div>
                                                <span style="font-size:9px; font-weight:800; padding:2px 6px; border-radius:4px; background:{'rgba(0,230,118,0.15)' if _daily_vrp_pts > 0 else 'rgba(255,51,102,0.15)'}; color:{'#00e676' if _daily_vrp_pts > 0 else '#ff3366'}; font-family:var(--font-mono);">
                                                    {'+' if _daily_edge_pct > 0 else ''}{_daily_edge_pct:.0f}% EDGE
                                                </span>
                                            </div>
                                            <div style="font-size:24px; font-weight:900; color:{'#00e676' if _daily_vrp_pts > 0 else '#ff3366'}; margin:4px 0 2px 0; font-family:var(--font-mono);">
                                                {_daily_vrp_pts:+.1f} PTS <span style="font-size:13px; font-weight:700; color:{WHITE};">/ day</span>
                                            </div>
                                            <div style="font-size:13px; font-weight:800; color:{'#00e676' if _daily_vrp_inr >= 0 else '#ff3366'}; margin-bottom:10px; font-family:var(--font-mono);">
                                                {'+₹' if _daily_vrp_inr >= 0 else '-₹'}{abs(_daily_vrp_inr):,.0f} <span style="font-size:10px; font-weight:600; color:{MUTED};">/ lot edge ({_lot} qty)</span>
                                            </div>
                                        </div>
                                        <div style="background:#131722; border-radius:6px; padding:8px 10px; border:1px solid var(--border-subtle, #2a2e39); font-size:11px;">
                                            <div style="display:flex; justify-content:space-between; margin-bottom:3px;">
                                                <span style="color:{MUTED};">Priced 1D Move (IV):</span>
                                                <strong style="color:{WHITE}; font-family:var(--font-mono);">±{_daily_iv_pts:.1f} pts <span style="color:{ACCENT}; font-size:10px;">(₹{_daily_iv_inr:,.0f})</span></strong>
                                            </div>
                                            <div style="display:flex; justify-content:space-between;">
                                                <span style="color:{MUTED};">Realized Reality (RV):</span>
                                                <strong style="color:{WHITE}; font-family:var(--font-mono);">±{_daily_rv_pts:.1f} pts <span style="color:#94a3b8; font-size:10px;">(₹{_daily_rv_inr:,.0f})</span></strong>
                                            </div>
                                        </div>
                                        <div class="metric-sub" style="color:{'#00e676' if _daily_vrp_pts > 0 else '#ff3366'}; font-weight:700; margin-top:8px;">
                                            {'⚡ Sellers harvest +' + f'{_daily_vrp_pts:.1f} pts daily theta buffer' if _daily_vrp_pts > 0 else '⚠️ Buyers hold statistical vol discount (-' + f'{abs(_daily_vrp_pts):.1f} pts)'}
                                        </div>
                                    </div>

                                    <!-- Card 2: Weekly Straddle Mispricing & Exact Values -->
                                    <div class="metric-box" style="padding:14px; text-align:left; background:rgba(30, 34, 45, 0.7); border-top:3px solid {'#00e676' if _straddle_vrp_pts > 0 else '#ff3366'}; display:flex; flex-direction:column; justify-content:space-between;">
                                        <div>
                                            <div style="display:flex; justify-content:space-between; align-items:center;">
                                                <div class="metric-label" style="font-size:10px;">WEEKLY ATM STRADDLE (5 DTE)</div>
                                                <span style="font-size:9px; font-weight:800; padding:2px 6px; border-radius:4px; background:{'rgba(0,230,118,0.15)' if _straddle_vrp_pts > 0 else 'rgba(255,51,102,0.15)'}; color:{'#00e676' if _straddle_vrp_pts > 0 else '#ff3366'}; font-family:var(--font-mono);">
                                                    {'+' if _straddle_edge_pct > 0 else ''}{_straddle_edge_pct:.1f}% MISPRICING
                                                </span>
                                            </div>
                                            <div style="font-size:24px; font-weight:900; color:{WHITE}; margin:4px 0 2px 0; font-family:var(--font-mono);">
                                                {_straddle_vrp_pts:+.1f} PTS <span style="font-size:13px; font-weight:700; color:{'#00e676' if _straddle_vrp_pts > 0 else '#ff3366'};">EDGE</span>
                                            </div>
                                            <div style="font-size:13px; font-weight:800; color:{'#00e676' if _straddle_vrp_inr >= 0 else '#ff3366'}; margin-bottom:10px; font-family:var(--font-mono);">
                                                {'+₹' if _straddle_vrp_inr >= 0 else '-₹'}{abs(_straddle_vrp_inr):,.0f} <span style="font-size:10px; font-weight:600; color:{MUTED};">/ lot edge ({_lot} qty)</span>
                                            </div>
                                        </div>
                                        <div style="background:#131722; border-radius:6px; padding:8px 10px; border:1px solid var(--border-subtle, #2a2e39); font-size:11px;">
                                            <div style="display:flex; justify-content:space-between; margin-bottom:3px;">
                                                <span style="color:{MUTED};">Market Price (at IV):</span>
                                                <strong style="color:{WHITE}; font-family:var(--font-mono);">{_straddle_iv_pts:.1f} pts <span style="color:{ACCENT}; font-size:10px;">(₹{_straddle_iv_inr:,.0f})</span></strong>
                                            </div>
                                            <div style="display:flex; justify-content:space-between;">
                                                <span style="color:{MUTED};">Fair Value (at RV):</span>
                                                <strong style="color:{WHITE}; font-family:var(--font-mono);">{_straddle_rv_pts:.1f} pts <span style="color:#94a3b8; font-size:10px;">(₹{_straddle_rv_inr:,.0f})</span></strong>
                                            </div>
                                        </div>
                                        <div class="metric-sub" style="color:{'#00e676' if _straddle_vrp_pts > 0 else '#ff3366'}; font-weight:700; margin-top:8px;">
                                            {'💰 Option Sellers Edge: Premium Overpriced by ' + f'{_straddle_edge_pct:+.1f}%' if _straddle_vrp_pts > 0 else '🎯 Option Buyers Edge: Straddle Discounted by ' + f'{abs(_straddle_edge_pct):.1f}%'}
                                        </div>
                                    </div>

                                    <!-- Card 3: Weekly 5-Day Expected Move Range -->
                                    <div class="metric-box" style="padding:14px; text-align:left; background:rgba(30, 34, 45, 0.7); border-top:3px solid {ACCENT}; display:flex; flex-direction:column; justify-content:space-between;">
                                        <div>
                                            <div style="display:flex; justify-content:space-between; align-items:center;">
                                                <div class="metric-label" style="font-size:10px;">WEEKLY 5-DAY RANGE (1σ)</div>
                                                <span style="font-size:9px; font-weight:800; padding:2px 6px; border-radius:4px; background:rgba(0,229,255,0.15); color:{ACCENT};">
                                                    68.3% PROBABILITY
                                                </span>
                                            </div>
                                            <div style="font-size:24px; font-weight:900; color:{WHITE}; margin:4px 0 2px 0; font-family:var(--font-mono);">
                                                ±{_weekly_iv_pts:.0f} PTS
                                            </div>
                                            <div style="font-size:13px; font-weight:800; color:{ACCENT}; margin-bottom:10px; font-family:var(--font-mono);">
                                                ₹{_weekly_iv_inr:,.0f} <span style="font-size:10px; font-weight:600; color:{MUTED};">1σ range / lot ({_lot} qty)</span>
                                            </div>
                                        </div>
                                        <div style="background:#131722; border-radius:6px; padding:8px 10px; border:1px solid var(--border-subtle, #2a2e39); font-size:11px;">
                                            <div style="display:flex; justify-content:space-between; margin-bottom:3px;">
                                                <span style="color:{MUTED};">Lower Bound:</span>
                                                <strong style="color:#ff3366; font-family:var(--font-mono);">{spot - _weekly_iv_pts:,.0f}</strong>
                                            </div>
                                            <div style="display:flex; justify-content:space-between;">
                                                <span style="color:{MUTED};">Upper Bound:</span>
                                                <strong style="color:#00e676; font-family:var(--font-mono);">{spot + _weekly_iv_pts:,.0f}</strong>
                                            </div>
                                        </div>
                                        <div class="metric-sub" style="color:{MUTED}; font-weight:600; margin-top:8px;">
                                            Priced weekly ATM break-even corridor on spot
                                        </div>
                                    </div>

                                    <!-- Card 4: Overpricing Multiple / Ratio -->
                                    <div class="metric-box" style="padding:14px; text-align:left; background:rgba(30, 34, 45, 0.7); border-top:3px solid {'#00e676' if _edge_ratio > 1.15 else '#ffd54f' if _edge_ratio >= 0.85 else '#ff3366'}; display:flex; flex-direction:column; justify-content:space-between;">
                                        <div>
                                            <div style="display:flex; justify-content:space-between; align-items:center;">
                                                <div class="metric-label" style="font-size:10px;">SELLERS' PREMIUM MULTIPLE</div>
                                                <span style="font-size:9px; font-weight:800; padding:2px 6px; border-radius:4px; background:{'rgba(0,230,118,0.15)' if _edge_ratio > 1.15 else 'rgba(255,213,79,0.15)' if _edge_ratio >= 0.85 else 'rgba(255,51,102,0.15)'}; color:{'#00e676' if _edge_ratio > 1.15 else '#ffd54f' if _edge_ratio >= 0.85 else '#ff3366'}; font-family:var(--font-mono);">
                                                    {_atm_iv - _rv_cons:+.1f}% VRP SPREAD
                                                </span>
                                            </div>
                                            <div style="font-size:24px; font-weight:900; color:{'#00e676' if _edge_ratio > 1.15 else '#ffd54f' if _edge_ratio >= 0.85 else '#ff3366'}; margin:4px 0 2px 0; font-family:var(--font-mono);">
                                                {_edge_ratio:.2f}x
                                            </div>
                                            <div style="font-size:13px; font-weight:800; color:{WHITE}; margin-bottom:10px; font-family:var(--font-mono);">
                                                IV {_atm_iv:.1f}% <span style="font-size:10px; font-weight:600; color:{MUTED};">vs RV {_rv_cons:.1f}%</span>
                                            </div>
                                        </div>
                                        <div style="background:#131722; border-radius:6px; padding:8px 10px; border:1px solid var(--border-subtle, #2a2e39); font-size:11px;">
                                            <div style="display:flex; justify-content:space-between; margin-bottom:3px;">
                                                <span style="color:{MUTED};">Premium Status:</span>
                                                <strong style="color:{'#00e676' if _edge_ratio > 1.0 else '#ff3366'}; font-family:var(--font-mono);">{f'+{(_edge_ratio - 1.0)*100:.1f}% Overpriced' if _edge_ratio > 1.0 else f'{(_edge_ratio - 1.0)*100:.1f}% Discounted'}</strong>
                                            </div>
                                            <div style="display:flex; justify-content:space-between;">
                                                <span style="color:{MUTED};">Optimal Action:</span>
                                                <strong style="color:{ACCENT};">{'Credit & Theta Harvest' if _edge_ratio > 1.10 else 'Debit & Directional' if _edge_ratio < 0.90 else 'Selective Tactical'}</strong>
                                            </div>
                                        </div>
                                        <div class="metric-sub" style="color:{MUTED}; font-weight:600; margin-top:8px;">
                                            Implied vs Realized Volatility dispersion ratio
                                        </div>
                                    </div>
                                </div>
                            </div>

                            <!-- 2. Multi-Horizon Volatility Term Structure Matrix -->
                            <div class="card" style="padding:16px; background:var(--bg-surface, #1e222d); border:1px solid var(--border-card, #363c4e);">
                                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:12px; flex-wrap:wrap; gap:8px;">
                                    <div>
                                        <div style="font-size:12px; font-weight:800; color:{ACCENT}; text-transform:uppercase; letter-spacing:1px;">
                                            📐 MULTI-HORIZON VOLATILITY TERM STRUCTURE MATRIX
                                        </div>
                                        <div style="font-size:11px; color:{MUTED}; margin-top:2px;">
                                            Term Slope (5D - 60D): <strong style="color:{'#ff3366' if (_rv_5d - _rv_60d) > 1 else '#00e676' if (_rv_5d - _rv_60d) < -1 else ACCENT}; font-family:var(--font-mono);">{_rv_5d - _rv_60d:+.2f}%</strong> ({'Inverted / Backwardation' if (_rv_5d - _rv_60d) > 1 else 'Normal Contango' if (_rv_5d - _rv_60d) < -1 else 'Flat Curve'}) · Consensus RV: <strong style="color:{WHITE}; font-family:var(--font-mono);">{_rv_cons:.2f}%</strong>
                                        </div>
                                    </div>
                                    <div style="font-size:11px; color:{MUTED};">
                                        Lot Size: <strong style="color:{WHITE};">{_lot} Qty</strong>
                                    </div>
                                </div>
                                <div style="overflow-x:auto;">
                                    <table style="width:100%; border-collapse:collapse; font-size:11px;">
                                        <thead>
                                            <tr style="background:#131722; color:{MUTED}; border-bottom:1px solid var(--border-subtle, #2a2e39);">
                                                <th style="padding:8px 12px; text-align:left;">Horizon & Tenor</th>
                                                <th style="padding:8px 12px; text-align:center;">Days</th>
                                                <th style="padding:8px 12px; text-align:center; color:#00e5ff;">Realized Vol (RV)</th>
                                                <th style="padding:8px 12px; text-align:center; color:#ffd54f;">Live Implied Vol (IV)</th>
                                                <th style="padding:8px 12px; text-align:center;">VRP Spread (IV - RV)</th>
                                                <th style="padding:8px 12px; text-align:right;">Expected Move (Pts)</th>
                                                <th style="padding:8px 12px; text-align:right; color:#38bdf8;">Expected Move (₹/Lot)</th>
                                                <th style="padding:8px 12px; text-align:left;">Premium State</th>
                                            </tr>
                                        </thead>
                                        <tbody>
                                            {_term_table_rows_html}
                                        </tbody>
                                    </table>
                                </div>
                            </div>

                            <!-- 3. Past Numbers & Historical Volatility Evolution -->
                            <div class="card" style="padding:16px; background:var(--bg-surface, #1e222d); border:1px solid var(--border-card, #363c4e);">
                                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:12px; flex-wrap:wrap; gap:8px;">
                                    <div>
                                        <div style="font-size:12px; font-weight:800; color:{ACCENT}; text-transform:uppercase; letter-spacing:1px;">
                                            📈 HISTORICAL VOLATILITY EVOLUTION & PAST NUMBERS
                                        </div>
                                        <div style="font-size:11px; color:{MUTED}; margin-top:2px;">
                                            Compare current realized volatility directly against yesterday, 5 days ago, 20 days ago, and 1-year macro range
                                        </div>
                                    </div>
                                    <div style="display:flex; gap:6px; flex-wrap:wrap;">
                                        <span style="background:rgba(0, 229, 255, 0.1); color:#00e5ff; font-size:10px; font-weight:800; padding:3px 8px; border-radius:4px; border:1px solid rgba(0,229,255,0.25);">
                                            Current 20D RV: <strong>{_p_cur_rv20:.2f}%</strong>
                                        </span>
                                        <span style="background:rgba(255,255,255,0.04); color:{'#ff3366' if _chg_yest > 0 else '#00e676'}; font-size:10px; font-weight:700; padding:3px 8px; border-radius:4px;">
                                            Yesterday: <strong>{_p_yest_rv20:.2f}%</strong> ({_chg_yest:+.2f}%)
                                        </span>
                                        <span style="background:rgba(255,255,255,0.04); color:{'#ff3366' if _chg_d5 > 0 else '#00e676'}; font-size:10px; font-weight:700; padding:3px 8px; border-radius:4px;">
                                            5D Ago: <strong>{_p_d5_rv20:.2f}%</strong> ({_chg_d5:+.2f}%)
                                        </span>
                                        <span style="background:rgba(255,255,255,0.04); color:{'#ff3366' if _chg_d20 > 0 else '#00e676'}; font-size:10px; font-weight:700; padding:3px 8px; border-radius:4px;">
                                            20D Ago: <strong>{_p_d20_rv20:.2f}%</strong> ({_chg_d20:+.2f}%)
                                        </span>
                                        <span style="background:rgba(255,213,79,0.1); color:#ffd54f; font-size:10px; font-weight:800; padding:3px 8px; border-radius:4px; border:1px solid rgba(255,213,79,0.25);">
                                            1Y Range: <strong>{_macro_min:.1f}% — {_macro_max:.1f}%</strong> ({_hv_pctile:.0f}%ile)
                                        </span>
                                    </div>
                                </div>
                                {vol_history_chart_html}
                            </div>

                            <!-- 4. 1-Year Realized Volatility Cone & HV Statistics -->
                            {vol_tab_html}
                        </div>
                        '''
                    else:
                        regime_tab_html = "<div style='padding:40px; color:#ff4444; text-align:center;'>Waiting for sufficient daily history data (requires 20+ trading days).</div>"

                    # Write fragment file that the running page will fetch
                    builder_tab_html = self._BUILDER_TERMINAL_SHELL
                    frag_path = html_path.replace('.html', '_fragment.html')
                    fragment_html = f'''
<div id="frag-regime">{regime_tab_html}</div>
<div id="frag-iv">{iv_tab_html}</div>
<div id="frag-vol"></div>
<div id="frag-chain">{chain_tab_html}</div>
<div id="frag-theta">{theta_tab_html}</div>
<div id="frag-prob">{prob_tab_html}</div>
<div id="frag-mm">{mm_tab_html}</div>
<div id="frag-builder">{builder_tab_html}</div>
<div id="frag-spot" data-spot="{spot:.0f}" data-time="{now_str}"></div>'''
                    # Atomic write to prevent lock contention on Windows
                    frag_tmp = frag_path + '.tmp'
                    with open(frag_tmp, 'w', encoding='utf-8') as f:
                        f.write(fragment_html)
                    try:
                        os.replace(frag_tmp, frag_path)
                    except Exception:
                        with open(frag_path, 'w', encoding='utf-8') as f:
                            f.write(fragment_html)

                    # Write full page (clean modular shell + live client controllers)
                    full_html = f'''<!DOCTYPE html>
<html lang="en" data-theme="dark">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta name="theme-color" content="#131722" id="theme-color-meta">
    <title>F-Intel | Quantitative Volatility & Options Terminal</title>
    
    <!-- Google Fonts & Plotly -->
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700&family=Outfit:wght@400;500;600;700;800&family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <script src="https://cdn.plot.ly/plotly-2.35.2.min.js" charset="utf-8"></script>

    <!-- PWA Manifest -->
    <link rel="manifest" href="/static/manifest.json">

    <!-- Stylesheets -->
    <link rel="stylesheet" href="/static/css/theme.css?v=20260918_v2">
    <link rel="stylesheet" href="/static/css/layout.css?v=20260918_v2">
    <link rel="stylesheet" href="/static/css/gamma_explosion.css?v=20260918_v2">
    <link rel="stylesheet" href="/static/css/iv_surface.css?v=20260918_v2">
    <link rel="stylesheet" href="/static/css/strategy_builder.css?v=20260922_v1">
</head>
<body>

    <!-- Top Navigation Header -->
    <header class="top-nav">
        <div class="brand-section">
            <div class="brand-logo">
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
                    <polyline points="22 12 18 12 15 21 9 3 6 12 2 12"></polyline>
                </svg>
                <span>F-INTEL</span>
            </div>
        </div>

        <div class="market-strip">
            <div id="spot-display" class="spot-pill">
                SPOT: <span class="spot-val">{spot:,.2f}</span>
            </div>

            <div class="system-badges">
                <span id="ws-status-badge" class="status-indicator">
                    <span class="pulse-dot"></span> LIVE
                </span>
                <span id="time-display" class="num-mono" style="font-size:12px; color:var(--text-muted);">{now_str}</span>
            </div>
        </div>
    </header>

    <!-- Tab Navigation -->
    <nav class="tab-navigation">
        <button class="tab-btn active" data-tab="regime"><span>REGIME & VOLATILITY</span></button>
        <button class="tab-btn" data-tab="iv"><span>IV SURFACE</span></button>
        <button class="tab-btn" data-tab="chain"><span>OPTION CHAIN & GREEKS</span></button>
        <button class="tab-btn" data-tab="theta"><span>GREEKS</span></button>
        <button class="tab-btn" data-tab="mm"><span>GAMMA EXPLOSION & MM</span></button>
        <button class="tab-btn" data-tab="builder"><span>STRATEGY BUILDER</span></button>
    </nav>

    <!-- Main Workspace Container -->
    <main class="workspace-container" id="dash-container">
        <section id="tab-regime" class="tab-content active">{regime_tab_html}</section>
        <section id="tab-iv" class="tab-content">{iv_tab_html}</section>
        <section id="tab-vol" class="tab-content" style="display:none;"></section>
        <section id="tab-chain" class="tab-content">{chain_tab_html}</section>
        <section id="tab-theta" class="tab-content">{theta_tab_html}</section>
        <section id="tab-mm" class="tab-content">{mm_tab_html}</section>
        <section id="tab-builder" class="tab-content">{builder_tab_html}</section>
    </main>

    <!-- Client Scripts -->
    <!-- background_canvas.js removed: FX animation disabled in new flat design -->
    <script src="/static/js/theta_simulator.js?v=20260918_v2"></script>
    <script src="/static/js/gamma_explosion_terminal.js?v=20260918_v2"></script>
    <script src="/static/js/advanced_vol_terminal.js?v=20260918_v2"></script>
    <script src="/static/js/gex_rebalance_radar.js?v=20260918_v2"></script>
    <script src="/static/js/oi_velocity_radar.js?v=20260918_v2"></script>
    <script src="/static/js/iv_surface_terminal.js?v=20260918_v2"></script>
    <script src="/static/js/strategy_builder_terminal.js?v=20260922_v1"></script>
    <script src="/static/js/dashboard_core.js?v=20260918_v2"></script>
</body>
</html>'''
                    # Atomic write for unified_dashboard.html
                    html_tmp = html_path + '.tmp'
                    with open(html_tmp, 'w', encoding='utf-8') as f:
                        f.write(full_html)
                    try:
                        os.replace(html_tmp, html_path)
                    except Exception:
                        with open(html_path, 'w', encoding='utf-8') as f:
                            f.write(full_html)

                    try:
                        tpl_dest = os.path.join(dashboard_dir, 'templates', 'unified_dashboard.html')
                        tpl_tmp = tpl_dest + '.tmp'
                        with open(tpl_tmp, 'w', encoding='utf-8') as f:
                            f.write(full_html)
                        try:
                            os.replace(tpl_tmp, tpl_dest)
                        except Exception:
                            with open(tpl_dest, 'w', encoding='utf-8') as f:
                                f.write(full_html)
                    except Exception:
                        pass

                    try:
                        frag_dest = os.path.join(dashboard_dir, 'unified_dashboard_fragment.html')
                        frag_tmp = frag_dest + '.tmp'
                        with open(frag_tmp, 'w', encoding='utf-8') as f:
                            f.write(fragment_html)
                        try:
                            os.replace(frag_tmp, frag_dest)
                        except Exception:
                            with open(frag_dest, 'w', encoding='utf-8') as f:
                                f.write(fragment_html)
                    except Exception:
                        pass

                    if first_run:
                        _dashboard_url = "http://127.0.0.1:8082/"
                        try:
                            import urllib.request
                            with urllib.request.urlopen("http://127.0.0.1:8082/health", timeout=1.0):
                                pass
                        except Exception:
                            _dashboard_url = f'{_base_url}/unified_dashboard.html'

                        webbrowser.open(_dashboard_url)
                        first_run = False
                        print(f"  Dashboard opened: {_dashboard_url}")

                    print(f"  [{now_str}] Updated | {pred['direction']} ({pred['confidence']:.0%}) | Spot:{spot:.0f}")
                    _gc_counter += 1
                    if _gc_counter % 10 == 0:
                        import gc; gc.collect()  # free Plotly figure memory
                    if single_run:
                        break
                    time.sleep(1)

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
        
        default_near = (today + timedelta(days=days_to_thu)).strftime("%Y-%m-%d")
        default_far = (today + timedelta(days=days_to_thu + 28)).strftime("%Y-%m-%d")

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