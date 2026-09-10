import sys

# Ensure stdout can print Unicode (like ✓) on Windows consoles
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')  # type: ignore
    except AttributeError:
        pass
        
import os
import time
import json
import threading
import pandas as pd
from flask import Flask, jsonify, send_file, request
from flask_cors import CORS
from flask_sock import Sock
from datetime import datetime
from fyers_auth_manager import get_fyers_instance, get_access_token
from fyers_apiv3.FyersWebsocket import data_ws
from OptionAnalytics import OptionAnalytics
try:
    import config
except ImportError:
    config = None

# --- CONFIG ---
from dotenv import load_dotenv
load_dotenv()

# Hardcoded secrets removed to .env
APP_ID = os.getenv("FYERS_APP_ID")
SYMBOL = "NSE:NIFTY50-INDEX"
PORT = 8082
CHAIN_REFRESH_INTERVAL = 60 # Seconds (Option chain rate limits are strict, 1 per min)

app = Flask(__name__)
CORS(app)  # type: ignore
sock = Sock(app)

class DataHub:
    def __init__(self):
        self.fyers = None
        self.access_token = None
        self.latest_data = {
            "spot": 0,
            "chain": {},
            "last_update": "",
            "status": "Initializing",
            "tick_count": 0
        }
        self.clients = set()
        self.lock = threading.Lock()
        self._stop_event = threading.Event()

    def authenticate(self):
        print("DataHub: Authenticating...")
        self.fyers = get_fyers_instance()
        if self.fyers:
            self.access_token = get_access_token()
            print("DataHub: Authentication Successful.")
            return True
        return False

    def broadcast(self, data):
        """Send data to all connected WebSocket clients (thread-safe)."""
        msg = json.dumps(data)
        dead = []
        # Snapshot clients without holding lock during send
        with self.lock:
            clients_snapshot = list(self.clients)
        for client in clients_snapshot:
            try:
                client.send(msg)
            except Exception:
                dead.append(client)
        # Remove dead clients under lock using set arithmetic
        if dead:
            with self.lock:
                self.clients -= set(dead)

    # --- Fyers WebSocket (Real-time Ticks) ---
    def on_ticks(self, ticks):
        for tick in ticks:
            if tick.get('symbol') == SYMBOL:
                lp = tick.get('lp')
                if lp:
                    with self.lock:
                        self.latest_data["spot"] = lp
                        self.latest_data["tick_count"] += 1
                        self.latest_data["last_update"] = datetime.now().strftime("%H:%M:%S")
                    # Push spot update to UI immediately
                    self.broadcast({"type": "tick", "spot": lp, "time": self.latest_data["last_update"]})

    def start_fyers_ws(self):
        print("DataHub: Starting Fyers WebSocket...")
        fyers_ws = data_ws.FyersDataSocket(
            access_token=f"{APP_ID}:{self.access_token}",
            log_path=".",
            litemode=True,
            on_message=self.on_ticks
        )
        fyers_ws.subscribe(symbols=[SYMBOL], data_type="symbolData")
        fyers_ws.keep_running()

    # --- Option Chain Polling ---
    def chain_loop(self):
        while not self._stop_event.is_set():
            try:
                if self.fyers:
                    # REST Fallback for Spot Price if WebSocket is silent or as a backup
                    try:
                        q_res = self.fyers.quotes({"symbols": SYMBOL})
                        if q_res and q_res.get('s') == 'ok' and q_res.get('d'):
                            lp = q_res['d'][0]['v']['lp']
                            with self.lock:
                                self.latest_data["spot"] = lp
                                self.latest_data["last_update"] = datetime.now().strftime("%H:%M:%S")
                            self.broadcast({"type": "tick", "spot": lp, "time": self.latest_data["last_update"]})
                    except Exception as e:
                        print(f"DataHub: Spot Fallback Error: {e}")

                    c_res = self.fyers.optionchain({"symbol": SYMBOL, "strikecount": 50})
                    if c_res and c_res.get('s') == 'ok':
                        chain = c_res.get('data', {})
                        with self.lock:
                            self.latest_data["chain"] = chain
                            self.latest_data["status"] = "Live"
                        # Push chain update to UI
                        self.broadcast({"type": "chain", "chain": chain})
            except Exception as e:
                print(f"DataHub: Chain Error: {e}")
            time.sleep(CHAIN_REFRESH_INTERVAL)

    def start(self):
        if not self.authenticate(): return
        
        # Start Chain Poller
        threading.Thread(target=self.chain_loop, daemon=True).start()
        
        # Start Fyers WS
        threading.Thread(target=self.start_fyers_ws, daemon=True).start()

hub = DataHub()

# ── SharedDataCache bridge ────────────────────────────────────────────────────
# A lightweight adapter so HestonCalibrator / PricingRouter can read the live
# spot and chain from DataHub without an extra Fyers API call.
from SharedDataCache import SharedDataCache as _SDC

class _DataHubCacheAdapter:
    """
    Thin adapter exposing the SharedDataCache interface on top of DataHub.
    Only the methods actually used by HestonCalibrator and PricingRouter
    need to be implemented — spot, raw_chain, T, and heston_params.
    """
    HESTON_TTL = 300

    def __init__(self, hub_ref):
        self._hub           = hub_ref
        self._heston_params = None
        self._heston_ts     = 0.0
        self._T             = 7 / 365   # default

    # ── Spot ─────────────────────────────────────────────────────────
    @property
    def _spot(self) -> float:
        return self._hub.latest_data.get('spot', 0.0)

    def get_spot(self) -> float:
        return self._spot

    # ── Raw chain ──────────────────────────────────────────────────────
    def set_raw_chain(self, chain_dict: dict):
        """Called by chain_loop after a successful chain fetch."""
        pass   # data lives in hub.latest_data['chain'] — get_raw_chain reads it

    def get_raw_chain(self) -> dict | None:
        chain = self._hub.latest_data.get('chain', {})
        return chain if chain else None

    # ── T (DTE in years) ───────────────────────────────────────────────
    def set_T(self, T: float):
        self._T = max(0.0, T)

    def get_T(self) -> float:
        return self._T

    # ── Heston params (5-min TTL) ──────────────────────────────────────
    def get_heston_params(self) -> dict | None:
        if self._heston_params is None:
            return None
        if time.time() - self._heston_ts > self.HESTON_TTL:
            return None
        return self._heston_params

    def set_heston_params(self, params: dict):
        self._heston_params = params
        self._heston_ts     = time.time()


hub_cache = _DataHubCacheAdapter(hub)

# Register hub_cache with PricingRouter so it can find Heston params without Fyers
try:
    from PricingRouter import register_shared_cache
    register_shared_cache(hub_cache)
except Exception:
    pass

@sock.route('/stream')
def stream(ws):
    with hub.lock:
        hub.clients.add(ws)
    # Send initial state
    with hub.lock:
        ws.send(json.dumps({"type": "init", "data": hub.latest_data}))
    
    while True:
        data = ws.receive() # Keep connection alive
        if data is None: break

@app.route('/get_data', methods=['GET'])
def get_data():
    with hub.lock:
        return jsonify(hub.latest_data)

@app.route('/', methods=['GET'])
@app.route('/unified_dashboard.html', methods=['GET'])
def index():
    response = send_file('unified_dashboard.html')
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response

@app.route('/fragment', methods=['GET'])
def serve_fragment():
    """Serve the latest pre-rendered dashboard fragment for refreshContent()."""
    try:
        response = send_file('unified_dashboard_fragment.html')
        response.headers['Cache-Control'] = 'no-store'
        return response
    except Exception:
        return '<div id="frag-regime"></div>', 200

@app.route('/static/manifest.json', methods=['GET'])
def serve_manifest():
    return send_file('static/manifest.json')

@app.route('/static/sw.js', methods=['GET'])
def serve_sw():
    response = send_file('static/sw.js')
    response.headers['Cache-Control'] = 'no-cache'
    return response

@app.route('/static/icon.png', methods=['GET'])
def serve_icon():
    return send_file('static/icon.png')

# ─────────────────────────────────────────────
# Strategy Engine API Endpoints
# ─────────────────────────────────────────────
from flask import request

# Lazy imports (avoid circular/slow imports at module load)
_bt_cache = {}          # strategy_type → latest BacktestReport
_bt_running = {}        # strategy_type → True/False

# ─────────────────────────────────────────────────────────────────────────────
# Strategy Wizard API  (Enhancement 3A)
# ─────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────
# SENSIBULL BUILDER API (Enhancement 4)
# ─────────────────────────────────────────────────────────────────────────────
@app.route('/api/chain/live', methods=['GET'])
def api_chain_live():
    """
    Returns the live option chain enriched with BSM Greeks.
    """
    try:
        from StrategyEngine import bsm_greeks, bsm_implied_volatility
        
        spot = hub.latest_data.get('spot', 0.0)
        chain_raw = hub.latest_data.get('chain', {})
        options = chain_raw.get('optionsChain', [])
        T = hub_cache.get_T()

        enriched_options = []
        for o in options:
            strike = float(o.get('strike_price', 0))
            opt_type = 'CE' if o.get('option_type', '') in ('CE', 'CALL') else 'PE'
            ltp = float(o.get('ltp', 0) or 0)
            iv = float(o.get('iv', 0) or 0)
            oi = int(o.get('oi', 0) or 0)
            
            if iv == 0 and ltp > 0 and spot > 0:
                iv = bsm_implied_volatility(ltp, spot, strike, T, 0.07, opt_type) * 100.0

            # Calculate Greeks if IV > 0
            greeks = {'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0}
            if iv > 0 and spot > 0:
                greeks = bsm_greeks(spot, strike, T, 0.07, iv/100.0, opt_type)
                
            enriched_options.append({
                'strike': strike,
                'type': opt_type,
                'ltp': ltp,
                'iv': iv,
                'oi': oi,
                'greeks': greeks
            })

        return jsonify({
            'ok': True,
            'spot': spot,
            'T': T,
            'expiry': chain_raw.get('expiry', 'Current'),
            'options': enriched_options
        })
    except Exception as e:
        import traceback
        return jsonify({'ok': False, 'error': str(e), 'trace': traceback.format_exc()}), 500

@app.route('/api/builder/analyze', methods=['POST'])
def api_builder_analyze():
    """
    Analyzes an arbitrary basket of option legs.
    Returns: Payload for chart (T+0, Expiry), Greeks, Summary Stats.
    """
    try:
        from StrategyEngine import OptionLeg, Strategy
        import numpy as np

        body = request.get_json(force=True) or {}
        legs_raw = body.get('legs', [])
        target_date_offset = float(body.get('target_date_offset', 0)) # days from now
        target_spot_offset = float(body.get('target_spot_offset', 0)) # percent (e.g. 2 for +2%)
        
        spot = hub.latest_data.get('spot', 0.0)
        T_now = hub_cache.get_T()
        
        if spot <= 0:
            return jsonify({'ok': False, 'error': 'Spot not available'}), 503

        # Parse Legs
        legs = []
        for lg in legs_raw:
            iv_raw = lg.get('iv', 15.0)
            # No IV bump anymore, just use actual IV
            adjusted_iv = max(0.01, iv_raw) / 100.0
            
            legs.append(OptionLeg(
                opt_type    = lg.get('type', 'CE'),
                action      = lg.get('action', 'SELL'),
                strike      = float(lg.get('strike', 0)),
                entry_price = float(lg.get('price', 0)),
                iv          = adjusted_iv,
                lots        = int(lg.get('lots', 1)),
                expiry      = lg.get('expiry', ''),
            ))
            
        strategy = Strategy("Custom Builder", legs, "CUSTOM")
        
        if not legs:
            return jsonify({'ok': True, 'empty': True})

        # Base Analysis
        T_target = max(0.0, T_now - (target_date_offset / 365.0))
        
        # Spot Range (±10%) around the targeted offset
        grid_center = spot * (1 + target_spot_offset / 100.0)
        range_pct = 0.10
        spot_range = np.linspace(grid_center * (1 - range_pct), grid_center * (1 + range_pct), 50)
        
        # Payoffs
        pnl_expiry = strategy.payoff_at_expiry(spot_range)
        pnl_target = strategy.payoff_now_bsm(spot_range, T=T_target)
        
        # Stats
        max_profit = strategy.max_profit(spot_range)
        max_loss = strategy.max_loss(spot_range)
        breakevens = strategy.breakevens(spot_range)
        
        # Target Stats
        projected_pnl = float(strategy.payoff_now_bsm(np.array([grid_center]), T=T_target)[0])
        
        # Calculate target breakevens (roots of pnl_target)
        target_breakevens = []
        for i in range(len(spot_range) - 1):
            if pnl_target[i] * pnl_target[i+1] < 0:
                # Linear interpolation
                x0, y0 = spot_range[i], pnl_target[i]
                x1, y1 = spot_range[i+1], pnl_target[i+1]
                be = x0 - y0 * (x1 - x0) / (y1 - y0)
                target_breakevens.append(float(be))
        
        # Find ATM IV for POP
        atm_iv = 15.0
        if legs:
            atm_iv = sum(l.iv for l in legs) / len(legs)
            
        pop = strategy.pop(spot, T_now, atm_iv) * 100
        net_premium = strategy.net_premium_lots
        
        # Serialize infinities to strings to prevent JS JSON.parse errors
        import math
        if math.isinf(max_profit): max_profit = "Infinity" if max_profit > 0 else "-Infinity"
        if math.isinf(max_loss): max_loss = "Infinity" if max_loss > 0 else "-Infinity"
        
        # Greeks
        net_greeks = strategy.net_greeks(spot, T_target)

        return jsonify({
            'ok': True,
            'atm_iv': atm_iv,
            'summary': {
                'max_profit': max_profit,
                'max_loss': max_loss,
                'breakevens': breakevens,
                'target_breakevens': target_breakevens,
                'projected_pnl': projected_pnl,
                'pop': pop,
                'net_premium': net_premium
            },
            'greeks': net_greeks,
            'chart': {
                'spot_prices': spot_range.tolist(),
                'pnl_expiry': pnl_expiry.tolist(),
                'pnl_target': pnl_target.tolist(),
                'T_target': T_target
            }
        })
    except Exception as e:
        import traceback
        return jsonify({'ok': False, 'error': str(e), 'trace': traceback.format_exc()}), 500

@app.route('/api/wizard', methods=['POST'])
def api_wizard():
    """
    POST /api/wizard
    """
    try:
        from StrategyWizard import StrategyWizard
        from StrategyEngine import OptionLeg, Strategy, PayoffEngine
        from WizardHistoryManager import WizardHistoryManager
        history_mgr = WizardHistoryManager()
        import numpy as np
        
        body = request.get_json(force=True) or {}
        view       = body.get('view', 'NEUTRAL').upper()
        risk       = body.get('risk', 'MODERATE').upper()
        capital    = float(body.get('capital', 100000))
        conviction = body.get('conviction', 'MODERATE').upper()

        spot  = hub.latest_data.get('spot', 0.0)
        chain = hub.latest_data.get('chain', {})
        if spot <= 0:
            return jsonify({'ok': False, 'error': 'Spot price not available'}), 503

        import pandas as pd
        df_chain = pd.DataFrame([
            {
                'strike': float(o.get('strike_price', 0)),
                'type':   'CE' if o.get('option_type', '') in ('CE', 'CALL') else 'PE',
                'price':  float(o.get('ltp', 0) or 0),
                'iv':     float(o.get('iv', 0) or 0),
                'oi':     int(o.get('oi', 0) or 0),
            }
            for o in chain.get('optionsChain', [])
        ])

        atm_iv = 15.0
        skew_ratio = 1.0
        T = hub_cache.get_T()
        if not df_chain.empty:
            dist = abs(df_chain['strike'] - spot)
            atm_row = df_chain.loc[dist.idxmin()]
            atm_val = atm_row.get('iv')
            if hasattr(atm_val, 'iloc'): atm_val = atm_val.iloc[0]
            atm_iv  = float(str(atm_val)) if atm_val is not None else 15.0
            pe_ivs = df_chain[df_chain['type'] == 'PE']['iv'].replace(0, np.nan).dropna()
            skew_ratio = (pe_ivs.median() / atm_iv) if atm_iv > 0 and not pe_ivs.empty else 1.0

        expiry = chain.get('expiry', '')
        if not expiry:
            for o in chain.get('optionsChain', []):
                e = o.get('expiry', '')
                if e:
                    expiry = e
                    break
        
        from KeyLevelsEngine import KeyLevelsEngine
        kle = KeyLevelsEngine()
        walls = kle.calculate_oi_walls(df_chain, spot)
        
        gex_snap = _gex_snapshot
        oi_bias = gex_snap.get('oi_surge', {}).get('bias', 0.0)
        if oi_bias > 0.1:
            oi_pressure = 'BULLISH'
        elif oi_bias < -0.1:
            oi_pressure = 'BEARISH'
        else:
            oi_pressure = 'NEUTRAL'
            
        term_spread = 0.0 # TODO: compute from front vs next-expiry ATM IV

        market_context = {
            'T':              T,
            'iv':             atm_iv,
            'atm_iv':         atm_iv,
            'vrp':            0.0,
            'regime':         gex_snap.get('regime', 'COMPRESSION'),
            'call_wall':      walls.get('call_wall', 0),
            'put_wall':       walls.get('put_wall', 0),
            'em':             spot * (atm_iv / 100) * np.sqrt(T),
            'oi_pressure':    oi_pressure,
            'DTE':            max(1, int(T * 365)),
            'skew_ratio':     skew_ratio,
            'explosion_score': gex_snap.get('score', 0),
            'term_spread':    term_spread,
            'heston_params':  hub_cache.get_heston_params(),
        }

        wizard = StrategyWizard(spot, df_chain, market_context, expiry)
        recs   = wizard.recommend(view, risk, capital, conviction)

        results = []
        for rec in recs:
            strat = rec['strategy']
            d     = strat.to_dict()

            sigma = atm_iv / 100.0
            pop   = strat.pop(spot, T, sigma) * 100

            engine     = PayoffEngine(strat, spot, T, sigma)
            risk_dials = engine.build_risk_dial_data()

            results.append({
                'name':          strat.name,
                'score':         rec['score'],
                'view_match':    rec['view_match'],
                'max_loss_inr':  round(rec['max_loss_inr'], 0),
                'lots_possible': rec['lots_possible'],
                'reasoning':     rec['reasoning'],
                'legs':          d['legs'],
                'net_premium':   round(strat.net_premium, 2),
                'pop':           round(pop, 1),
                'risk_dials':    risk_dials,
            })

        # Add top recommendation to history if we have any
        if results:
            top = results[0]
            # rationale is a list of strings, join it
            rationale_str = " ".join(top['reasoning']) if isinstance(top['reasoning'], list) else str(top['reasoning'])
            # Create a more contextual rationale string based on the dashboard metrics
            regime_str = market_context.get('regime', 'UNKNOWN')
            cw = market_context.get('call_wall', 0)
            pw = market_context.get('put_wall', 0)
            full_rationale = f"Dashboard shows {regime_str} regime. Spot at {spot:.0f} (CW: {cw}, PW: {pw}). {rationale_str}"
            
            history_mgr.add_recommendation(top['name'], top['legs'], full_rationale, top['score'])

        return jsonify({'ok': True, 'strategies': results})

    except Exception as e:
        import traceback
        return jsonify({'ok': False, 'error': str(e),
                        'trace': traceback.format_exc()}), 500

@app.route('/api/wizard/history', methods=['GET'])
def api_wizard_history():
    try:
        from WizardHistoryManager import WizardHistoryManager
        mgr = WizardHistoryManager()
        return jsonify({'ok': True, 'history': mgr.get_today_history()})
    except Exception as e:
        import traceback
        return jsonify({'ok': False, 'error': str(e), 'trace': traceback.format_exc()}), 500

@app.route('/api/scenario', methods=['POST'])
def api_scenario():
    try:
        from ScenarioEngine import ScenarioEngine
        from StrategyEngine import OptionLeg, Strategy
        import numpy as np

        body         = request.get_json(force=True) or {}
        strat_dict   = body.get('strategy', {})
        range_pct    = float(body.get('spot_range_pct', 10)) / 100.0
        n_spot       = int(body.get('n_spot', 21))

        legs_raw = strat_dict.get('legs', [])
        if not legs_raw:
            return jsonify({'ok': False, 'error': 'No legs provided'}), 400

        legs = []
        for lg in legs_raw:
            iv_raw = lg.get('iv', 15.0)
            iv_dec = (iv_raw / 100.0) if iv_raw > 1 else float(iv_raw)
            legs.append(OptionLeg(
                opt_type    = lg.get('type', 'CE'),
                action      = lg.get('action', 'SELL'),
                strike      = float(lg.get('strike', 0)),
                entry_price = float(lg.get('price', 0)),
                iv          = iv_dec,
                lots        = int(lg.get('lots', 1)),
                expiry      = lg.get('expiry', ''),
            ))
        strat_type = strat_dict.get('type', 'CREDIT')
        strat_name = strat_dict.get('name', 'Custom')
        strategy   = Strategy(strat_name, legs, strat_type)

        spot  = hub.latest_data.get('spot', 0.0)
        T     = hub_cache.get_T()
        sigma = float(strat_dict.get('net_premium', 0)) / max(spot, 1) if spot > 0 else 0.15
        
        chain = hub.latest_data.get('chain', {})
        if chain.get('optionsChain'):
            import pandas as pd
            df_c = pd.DataFrame(chain.get('optionsChain', []))
            if not df_c.empty and 'iv' in df_c.columns and 'strike_price' in df_c.columns:
                df_c['dist'] = abs(df_c['strike_price'].astype(float) - spot)
                atm_val = df_c.loc[df_c['dist'].idxmin(), 'iv']
                if hasattr(atm_val, 'iloc'): atm_val = atm_val.iloc[0]
                atm_iv = float(str(atm_val)) if atm_val is not None else 15.0
                sigma  = atm_iv / 100.0

        engine = ScenarioEngine(strategy, spot, T, max(0.01, sigma))
        grid   = engine.compute_grid(spot_pct_range=(-range_pct, range_pct),
                                     n_spot=n_spot)
        ladder = engine.spot_ladder(spot_pct_range=(-range_pct, range_pct),
                                    n=n_spot)

        return jsonify({'ok': True, 'grid': grid, 'ladder': ladder})

    except Exception as e:
        import traceback
        return jsonify({'ok': False, 'error': str(e),
                        'trace': traceback.format_exc()}), 500

# GEX / Signals / Regime / Dealer API
# ─────────────────────────────────────────────────────────────────────────────

_gex_snapshot: dict = {
    "score": 0, "regime": "UNKNOWN", "net_gex": 0,
    "strikes": [], "direction": "NEUTRAL", "last_update": "",
    # New fields from calculations/GexEngine
    "zero_gamma_level": 0, "dealer_long_pct": 0, "dealer_short_pct": 0,
    "spot_gamma": 0, "forward_gex": 0,
}
_gex_lock = threading.Lock()

# In-memory dealer snapshot (refreshed together with GEX)
_dealer_snapshot: dict = {
    "net_dex": 0, "net_gex_shares": 0, "net_vanna": 0, "net_charm": 0,
    "hedge_spot_up": 0, "hedge_iv_up": 0, "hedge_1day": 0,
    "strike_dex": [], "strike_gex": [], "last_update": "",
}
_dealer_lock = threading.Lock()


def _parse_chain_to_df(chain: dict) -> "pd.DataFrame":
    """Convert Fyers optionsChain dict to a clean DataFrame for calculations/."""
    import pandas as pd
    rows = chain.get("optionsChain", [])
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame([{
        "strike": float(r.get("strike_price", 0)),
        "type":   r.get("option_type", "CE"),
        "oi":     float(r.get("oi", 0) or 0),  # type: ignore
        "volume": float(r.get("volume", 0) or 0),  # type: ignore
        "iv":     float(r.get("iv", 0) or 0),  # type: ignore
        "price":  float(r.get("ltp", 0) or 0),  # type: ignore
        "dte":    float(r.get("dte", 1) or 1),  # type: ignore
    } for r in rows])


def _gex_refresh_loop(interval: int = 60):
    """Background thread: refresh GEX + Dealer snapshots every `interval` seconds."""
    while True:
        try:
            spot  = hub.latest_data.get("spot", 0)
            chain = hub.latest_data.get("chain", {})
            if not spot or not chain:
                time.sleep(5)
                continue
            
            if spot > 0 and chain:
                import pandas as pd
                from calculations.GexEngine import GexEngine
                from calculations.DealerPositionEngine import DealerPositionEngine

                df = _parse_chain_to_df(chain)
                if not df.empty:
                    # ── GEX via calculations/GexEngine ──────────────────────
                    _lot = config.get("nifty_lot_size", 65) if config else 65
                    gex_eng = GexEngine(lot_size=_lot, positioning_model='standard')
                    gex_res = gex_eng.calculate_gex(df, spot)

                    profile   = gex_res.get("profile")
                    net_gex   = gex_res.get("net_gex", 0)
                    direction = "POSITIVE" if net_gex > 0 else "NEGATIVE"

                    strike_list = []
                    if profile is not None and not profile.empty:
                        for strike, gex_val in profile.items():
                            strike_list.append({"strike": float(strike), "gex": float(gex_val)})

                    regime = "POSITIVE GEX" if net_gex > 0 else "NEGATIVE GEX"
                    score  = round(abs(net_gex) / 1e9, 2)
                    # Try GammaExplosionModel for full context (non-fatal)
                    try:
                        from GammaExplosionModel import GammaExplosionModel
                        gm = GammaExplosionModel()
                        gm.fyers = hub.fyers
                        gm.spot_price = spot
                        gm_df = gm.parse_chain(chain)
                        if not gm_df.empty:
                            T = hub_cache.get_T()
                            gm_res = gm.analyze(gm_df, spot, T)
                            
                            with _gex_lock:
                                _gex_snapshot.update(gm_res)
                    except Exception as e:
                        print(f"[GEX] GammaExplosionModel error: {e}")

                    with _gex_lock:
                        # Keep calculations/GexEngine fields as fallback/additions
                        if "score" not in _gex_snapshot:
                            _gex_snapshot["score"] = score
                        if "regime" not in _gex_snapshot:
                            _gex_snapshot["regime"] = regime
                        if "net_gex" not in _gex_snapshot:
                            _gex_snapshot["net_gex"] = float(net_gex)
                        
                        _gex_snapshot.update({
                            "zero_gamma_level": gex_res.get("zero_gamma_level", 0),
                            "dealer_long_pct":  round(gex_res.get("dealer_long_pct", 0), 1),
                            "dealer_short_pct": round(gex_res.get("dealer_short_pct", 0), 1),
                            "spot_gamma":       gex_res.get("spot_gamma", 0),
                            "forward_gex":      gex_res.get("forward_gex", 0),
                            "last_update":      datetime.now().strftime("%H:%M:%S"),
                        })
                    print(f"[GEX] net={_gex_snapshot.get('net_gex', 0):.0f}, flip={_gex_snapshot.get('gex_flip_point', gex_res.get('zero_gamma_level', 0)):.0f}")

                    # ── Dealer Inventory via DealerPositionEngine ──────────
                    try:
                        dep   = DealerPositionEngine(lot_size=_lot)
                        d_res = dep.calculate_dealer_inventory(df, spot)
                        sp    = d_res.get("strike_profile", {})

                        def _to_list(s):
                            if s is None or (hasattr(s, "empty") and s.empty):
                                return []
                            return [{"strike": float(k), "value": float(v)} for k, v in s.items()]

                        with _dealer_lock:
                            _dealer_snapshot.update({
                                "net_dex":        round(d_res.get("net_delta_exposure", 0), 0),
                                "net_gex_shares": round(d_res.get("net_gamma_shares", 0), 0),
                                "net_vanna":      round(d_res.get("net_vanna_exposure", 0), 0),
                                "net_charm":      round(d_res.get("net_charm_exposure", 0), 0),
                                "hedge_spot_up":  round(d_res.get("projected_hedging", {}).get("buy_shares_if_spot_up_1pct", 0), 0),
                                "hedge_iv_up":    round(d_res.get("projected_hedging", {}).get("buy_shares_if_iv_up_1pct", 0), 0),
                                "hedge_1day":     round(d_res.get("projected_hedging", {}).get("buy_shares_if_1_day_passes", 0), 0),
                                "strike_dex":     _to_list(sp.get("dex")),
                                "strike_gex":     _to_list(sp.get("gex")),
                                "last_update":    datetime.now().strftime("%H:%M:%S"),
                            })
                        print(f"[DEALER] DEX={d_res.get('net_delta_exposure',0):.0f}")
                    except Exception as _de:
                        print(f"[DEALER] Error (non-fatal): {_de}")

        except Exception as _e:
            print(f"[GEX] Refresh error (non-fatal): {_e}")
        time.sleep(interval)


@app.route('/api/gex', methods=['GET'])
def api_gex():
    """
    GET /api/gex — GEX snapshot from calculations/GexEngine.
    { ok, net_gex, regime, direction, zero_gamma_level,
      dealer_long_pct, dealer_short_pct, spot_gamma,
      strikes:[{strike, gex}], last_update }
    """
    with _gex_lock:
        snap = dict(_gex_snapshot)
    return jsonify({"ok": True, **snap})


@app.route('/api/dealer', methods=['GET'])
def api_dealer():
    """
    GET /api/dealer — Dealer inventory snapshot from DealerPositionEngine.
    { ok, net_dex, net_gex_shares, net_vanna, net_charm,
      hedge_spot_up, hedge_iv_up, hedge_1day,
      strike_dex:[{strike,value}], strike_gex:[{strike,value}], last_update }
    """
    with _dealer_lock:
        snap = dict(_dealer_snapshot)
    return jsonify({"ok": True, **snap})




@app.route('/api/signals', methods=['GET'])
def api_signals():
    """
    GET /api/signals
    Returns the full SignalMemory store: active signals, resolved signals, context.

    Response shape:
      { ok, active:[...], resolved:[...], context:{...} }
    """
    try:
        from SignalMemory import SignalMemory
        mem = SignalMemory()
        data = mem._data
        return jsonify({
            "ok":       True,
            "active":   data.get("active_signals", []),
            "resolved": data.get("resolved_signals", [])[-30:],  # last 30
            "context":  data.get("context", {}),
        })
    except Exception as e:
        import traceback
        return jsonify({"ok": False, "error": str(e),
                        "trace": traceback.format_exc()}), 500


@app.route('/api/regime', methods=['GET'])
def api_regime():
    """
    GET /api/regime
    Returns a snapshot of the current volatility regime, VRP, RV estimates,
    ATM IV, and cached Heston parameters.

    Response shape:
      { ok, regime, atm_iv, vrp, rv_5d, rv_20d, dte_days, heston:{...}, last_update }
    """
    try:
        import pandas as pd

        spot  = hub.latest_data.get("spot", 0.0)
        chain = hub.latest_data.get("chain", {})
        T     = hub_cache.get_T()
        dte_days = max(1, round(T * 365))

        # ── ATM IV from live chain ─────────────────────────────────────
        atm_iv = 0.0
        if chain.get("optionsChain") and spot > 0:
            df_c = pd.DataFrame(chain["optionsChain"])
            if not df_c.empty and "iv" in df_c.columns and "strike_price" in df_c.columns:
                df_c["dist"] = abs(df_c["strike_price"].astype(float) - spot)
                row = df_c.loc[df_c["dist"].idxmin()]
                iv_val = row.get("iv", 0)
                if isinstance(iv_val, pd.Series):
                    iv_val = iv_val.iloc[0]
                atm_iv = float(iv_val or 0)

        # ── RV estimates from SignalMemory context ─────────────────────
        rv_5d = rv_20d = vrp = 0.0
        regime_label = "UNKNOWN"
        try:
            from SignalMemory import SignalMemory
            ctx = SignalMemory()._data.get("context", {})
            rv_5d  = float(ctx.get("rv_5d")  or 0)
            rv_20d = float(ctx.get("rv_20d") or 0)
            vrp    = float(ctx.get("vrp")    or 0)
            regime_label = ctx.get("regime") or "UNKNOWN"
        except Exception:
            pass

        # ── Derive regime if still unknown ────────────────────────────
        if regime_label == "UNKNOWN" and atm_iv > 0:
            if vrp < -2:
                regime_label = "UNDERPRICED"
            elif vrp > 3:
                regime_label = "OVERPRICED"
            else:
                regime_label = "COMPRESSION"

        heston = hub_cache.get_heston_params() or {}

        return jsonify({
            "ok":          True,
            "regime":      regime_label,
            "atm_iv":      round(atm_iv, 2),
            "vrp":         round(vrp, 2),
            "rv_5d":       round(rv_5d, 2),
            "rv_20d":      round(rv_20d, 2),
            "dte_days":    dte_days,
            "spot":        spot,
            "heston":      heston,
            "last_update": hub.latest_data.get("last_update"),
        })

    except Exception as e:
        import traceback
        return jsonify({"ok": False, "error": str(e),
                        "trace": traceback.format_exc()}), 500


# ─────────────────────────────────────────────────────────────────────────────
# Legacy Backtest Routes  (used by the dashboard's bt tab)
# /bt_run  →  triggers async backtest (same logic as /api/backtest)
# /bt_<type>.html  →  polls and serves the HTML result
# ─────────────────────────────────────────────────────────────────────────────


@app.route('/api/confluence', methods=['GET'])
def api_confluence():
    try:
        from MasterSignalEngine import MasterSignalEngine
        from SignalMemory import SignalMemory
        
        mem = SignalMemory()
        mse = MasterSignalEngine()
        verdict = mse.evaluate(mem)
        
        return jsonify({'ok': True, 'confluence': verdict})
    except Exception as e:
        import traceback
        return jsonify({'ok': False, 'error': str(e), 'trace': traceback.format_exc()}), 500


@app.route('/api/health/engines', methods=['GET'])
def health_engines():
    try:
        from SignalMemory import SignalMemory
        mem = SignalMemory()
        context = mem.get_context()
        
        chain = hub.latest_data.get('chain', {})
        chain_update = chain.get('last_update', 'Never')
        
        return jsonify({
            'ok': True,
            'gex_engine': _gex_snapshot.get('last_update', 'Never'),
            'dealer_engine': _dealer_snapshot.get('last_update', 'Never'),
            'signal_memory': context.get('last_updated', 'Never'),
            'chain_cache': chain_update
        })
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)}), 500



# ─────────────────────────────────────────────────────────────────────────────
# PORTFOLIO MANAGER API (Paper Trading)
# ─────────────────────────────────────────────────────────────────────────────
try:
    from PortfolioManager import PortfolioManager
    portfolio_mgr = PortfolioManager()
except ImportError:
    portfolio_mgr = None

@app.route('/api/portfolio/deploy', methods=['POST'])
def api_portfolio_deploy():
    if not portfolio_mgr:
        return jsonify({'ok': False, 'error': 'PortfolioManager not loaded'}), 500
    try:
        body = request.get_json(force=True) or {}
        legs = body.get('legs', [])
        name = body.get('name', 'Custom Strategy')
        spot = hub.latest_data.get('spot', 0.0)
        
        if not legs:
            return jsonify({'ok': False, 'error': 'No legs provided'}), 400
            
        pos = portfolio_mgr.deploy(name, legs, spot)
        return jsonify({'ok': True, 'position': pos})
    except Exception as e:
        import traceback
        return jsonify({'ok': False, 'error': str(e), 'trace': traceback.format_exc()}), 500

@app.route('/api/portfolio/status', methods=['GET'])
def api_portfolio_status():
    if not portfolio_mgr:
        return jsonify({'ok': False, 'error': 'PortfolioManager not loaded'}), 500
    try:
        # Build live chain map: { 'CE': {strike: ltp}, 'PE': {strike: ltp} }
        chain_raw = hub.latest_data.get('chain', {})
        options = chain_raw.get('optionsChain', [])
        live_chain = {'CE': {}, 'PE': {}}
        for o in options:
            strike = float(o.get('strike_price', 0))
            opt_type = 'CE' if o.get('option_type', '') in ('CE', 'CALL') else 'PE'
            ltp = float(o.get('ltp', 0) or 0)
            live_chain[opt_type][strike] = ltp
            
        spot = hub.latest_data.get('spot', 0.0)
        T_now = hub_cache.get_T()
        atm_iv = 15.0
        
        # Approximate ATM IV
        if options:
            import pandas as pd
            df_c = pd.DataFrame(options)
            if not df_c.empty and 'iv' in df_c.columns and 'strike_price' in df_c.columns:
                df_c['dist'] = abs(df_c['strike_price'].astype(float) - spot)
                atm_val = df_c.loc[df_c['dist'].idxmin(), 'iv']
                if hasattr(atm_val, 'iloc'): atm_val = atm_val.iloc[0]
                atm_iv = float(str(atm_val)) if atm_val is not None else 15.0
                
        status = portfolio_mgr.get_status(live_chain, spot, T_now, atm_iv/100.0)
        return jsonify({'ok': True, 'portfolio': status})
    except Exception as e:
        import traceback
        return jsonify({'ok': False, 'error': str(e), 'trace': traceback.format_exc()}), 500

@app.route('/api/portfolio/exit', methods=['POST'])
def api_portfolio_exit():
    if not portfolio_mgr:
        return jsonify({'ok': False, 'error': 'PortfolioManager not loaded'}), 500
    try:
        body = request.get_json(force=True) or {}
        pos_id = body.get('id')
        if not pos_id:
            return jsonify({'ok': False, 'error': 'No position ID provided'}), 400
            
        chain_raw = hub.latest_data.get('chain', {})
        options = chain_raw.get('optionsChain', [])
        live_chain = {'CE': {}, 'PE': {}}
        for o in options:
            strike = float(o.get('strike_price', 0))
            opt_type = 'CE' if o.get('option_type', '') in ('CE', 'CALL') else 'PE'
            ltp = float(o.get('ltp', 0) or 0)
            live_chain[opt_type][strike] = ltp
            
        pos = portfolio_mgr.exit_position(pos_id, live_chain)
        if pos:
            return jsonify({'ok': True, 'position': pos})
        return jsonify({'ok': False, 'error': 'Position not found'}), 404
    except Exception as e:
        import traceback
        return jsonify({'ok': False, 'error': str(e), 'trace': traceback.format_exc()}), 500

# ─────────────────────────────────────────────────────────────────────────────
# OPTIONAL TOKEN AUTHENTICATION MIDDLEWARE
# Set FINTEL_TOKEN in .env to enable.  The dashboard passes it via
# X-FIntel-Token header or ?token= query param.
# Skip /health and /ready to allow liveness probes without auth.
# ─────────────────────────────────────────────────────────────────────────────
_FINTEL_TOKEN = os.getenv("FINTEL_TOKEN", "").strip()
_AUTH_EXEMPT  = {"/", "/fragment", "/builder", "/health", "/ready",
                 "/static/manifest.json", "/static/sw.js", "/static/icon.png"}

@app.before_request
def _check_token():
    if not _FINTEL_TOKEN:
        return  # Auth disabled — no token configured
    from flask import abort
    path = request.path
    if path in _AUTH_EXEMPT:
        return  # Skip auth for liveness probes
    # WebSocket upgrade requests skip REST auth
    if request.headers.get("Upgrade", "").lower() == "websocket":
        return
    token = (request.headers.get("X-FIntel-Token", "")
             or request.args.get("token", ""))
    if token != _FINTEL_TOKEN:
        abort(401)


# ─────────────────────────────────────────────────────────────────────────────
# KEY LEVELS ENDPOINT
# ─────────────────────────────────────────────────────────────────────────────
@app.route('/api/key_levels', methods=['GET'])
def api_key_levels():
    """
    GET /api/key_levels
    Returns Max Pain, PCR, OI walls (call_wall, put_wall x2) from live chain.
    { ok, max_pain, pcr, call_wall, call_wall_2, put_wall, put_wall_2,
      total_call_oi, total_put_oi, last_update }
    """
    try:
        import pandas as pd
        from KeyLevelsEngine import KeyLevelsEngine

        chain = hub.latest_data.get('chain', {})
        spot  = hub.latest_data.get('spot', 0.0)
        options = chain.get('optionsChain', [])
        if not options or not spot:
            return jsonify({'ok': True, 'max_pain': 0, 'pcr': 0,
                            'call_wall': 0, 'put_wall': 0,
                            'call_wall_2': 0, 'put_wall_2': 0,
                            'total_call_oi': 0, 'total_put_oi': 0,
                            'last_update': hub.latest_data.get('last_update', '')})

        # Build lightweight KLE and reuse existing chain data
        kle = KeyLevelsEngine()
        kle.fyers = hub.fyers
        # Parse chain into the KLE-compatible DataFrame
        records = []
        for o in options:
            otype = 'CE' if o.get('option_type', '') in ('CE', 'CALL') else 'PE'
            records.append({
                'strike': float(o.get('strike_price', 0)),
                'type':   otype,
                'ltp':    float(o.get('ltp', 0) or 0),
                'oi':     float(o.get('oi', 0) or 0),
                'iv':     float(o.get('iv', 0) or 0),
                'delta':  float(o.get('delta', 0) or 0),
                'gamma':  float(o.get('gamma', 0) or 0),
            })
        df = pd.DataFrame(records)

        max_pain  = kle.calculate_max_pain(df) if not df.empty else 0
        pcr       = kle.calculate_pcr(df) if not df.empty else 0
        walls     = kle.calculate_oi_walls(df, spot)

        total_call_oi = int(df[df['type'] == 'CE']['oi'].sum()) if not df.empty else 0
        total_put_oi  = int(df[df['type'] == 'PE']['oi'].sum()) if not df.empty else 0

        return jsonify({
            'ok':            True,
            'max_pain':      float(max_pain),
            'pcr':           round(float(pcr), 3),
            'call_wall':     walls.get('call_wall', 0),
            'call_wall_2':   walls.get('call_wall_2', 0),
            'put_wall':      walls.get('put_wall', 0),
            'put_wall_2':    walls.get('put_wall_2', 0),
            'total_call_oi': total_call_oi,
            'total_put_oi':  total_put_oi,
            'last_update':   hub.latest_data.get('last_update', ''),
        })
    except Exception as e:
        import traceback
        return jsonify({'ok': False, 'error': str(e),
                        'trace': traceback.format_exc()}), 500


# ─────────────────────────────────────────────────────────────────────────────
# REALIZED VOLATILITY ENDPOINT
# ─────────────────────────────────────────────────────────────────────────────
@app.route('/api/rv', methods=['GET'])
def api_rv():
    """
    GET /api/rv
    Returns RV estimates from SignalMemory context + ATM IV.
    { ok, rv_5d, rv_20d, atm_iv, vrp, regime, last_update }
    """
    try:
        import pandas as pd
        from SignalMemory import SignalMemory

        ctx = SignalMemory()._data.get('context', {})
        spot  = hub.latest_data.get('spot', 0.0)
        chain = hub.latest_data.get('chain', {})

        atm_iv = 0.0
        if chain.get('optionsChain') and spot > 0:
            df_c = pd.DataFrame(chain['optionsChain'])
            if not df_c.empty and 'iv' in df_c.columns and 'strike_price' in df_c.columns:
                df_c['dist'] = abs(df_c['strike_price'].astype(float) - spot)
                row = df_c.loc[df_c['dist'].idxmin()]
                iv_val = row.get('iv', 0)
                if isinstance(iv_val, pd.Series): iv_val = iv_val.iloc[0]
                atm_iv = float(iv_val or 0)

        return jsonify({
            'ok':          True,
            'rv_5d':       round(float(ctx.get('rv_5d') or 0), 2),
            'rv_20d':      round(float(ctx.get('rv_20d') or 0), 2),
            'atm_iv':      round(atm_iv, 2),
            'vrp':         round(float(ctx.get('vrp') or 0), 2),
            'regime':      ctx.get('regime', 'UNKNOWN'),
            'last_update': ctx.get('last_updated', hub.latest_data.get('last_update', '')),
        })
    except Exception as e:
        import traceback
        return jsonify({'ok': False, 'error': str(e),
                        'trace': traceback.format_exc()}), 500


# ─────────────────────────────────────────────────────────────────────────────
# CONSOLIDATED CHAIN STRIKES ENDPOINT (one row per strike for Builder tab)
# ─────────────────────────────────────────────────────────────────────────────
@app.route('/api/chain/strikes', methods=['GET'])
def api_chain_strikes():
    """
    GET /api/chain/strikes
    Returns one-row-per-strike format for the Strategy Builder table.
    Each row: { strike, ce: {ltp,iv,oi,delta,gamma,theta,vega}, pe: {...} }
    Optional ?range_pct=10  (default 10 -- +/-10% from spot)
    { ok, spot, strikes: [...], last_update }
    """
    try:
        from StrategyEngine import bsm_greeks, bsm_implied_volatility
        import pandas as pd

        range_pct = float(request.args.get('range_pct', 10)) / 100.0
        spot      = hub.latest_data.get('spot', 0.0)
        chain_raw = hub.latest_data.get('chain', {})
        options   = chain_raw.get('optionsChain', [])
        T         = hub_cache.get_T()

        if not spot or not options:
            return jsonify({'ok': True, 'spot': spot, 'strikes': [],
                            'last_update': hub.latest_data.get('last_update', '')})

        # Build lookup: { (strike, type) -> option_dict }
        lookup = {}
        for o in options:
            s  = float(o.get('strike_price', 0))
            ot = 'CE' if o.get('option_type', '') in ('CE', 'CALL') else 'PE'
            lookup[(s, ot)] = o

        # Find strikes within range
        lo, hi = spot * (1 - range_pct), spot * (1 + range_pct)
        strikes_in_range = sorted({s for (s, _) in lookup.keys() if lo <= s <= hi})

        def _enrich(o, otype):
            if o is None:
                return {'ltp': 0, 'iv': 0, 'oi': 0, 'delta': 0,
                        'gamma': 0, 'theta': 0, 'vega': 0}
            ltp  = float(o.get('ltp', 0) or 0)
            iv   = float(o.get('iv', 0) or 0)
            oi   = int(o.get('oi', 0) or 0)
            strike = float(o.get('strike_price', 0))
            if iv == 0 and ltp > 0 and spot > 0:
                iv = bsm_implied_volatility(ltp, spot, strike, T, 0.07, otype) * 100.0
            greeks = {'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0}
            if iv > 0 and spot > 0:
                greeks = bsm_greeks(spot, strike, T, 0.07, iv / 100.0, otype)
            return {'ltp': ltp, 'iv': round(float(iv), 2), 'oi': oi,
                    'delta': round(float(greeks['delta']), 4),
                    'gamma': round(float(greeks['gamma']), 6),
                    'theta': round(float(greeks['theta']), 4),
                    'vega':  round(float(greeks['vega']), 4)}

        result = []
        for s in strikes_in_range:
            result.append({
                'strike': s,
                'is_atm': abs(s - spot) == min(abs(k - spot) for k in strikes_in_range),
                'ce': _enrich(lookup.get((s, 'CE')), 'CE'),
                'pe': _enrich(lookup.get((s, 'PE')), 'PE'),
            })

        return jsonify({
            'ok':          True,
            'spot':        spot,
            'T':           T,
            'strikes':     result,
            'last_update': hub.latest_data.get('last_update', ''),
        })
    except Exception as e:
        import traceback
        return jsonify({'ok': False, 'error': str(e),
                        'trace': traceback.format_exc()}), 500


# ─────────────────────────────────────────────────────────────────────────────
# SESSION METADATA ENDPOINT
# ─────────────────────────────────────────────────────────────────────────────
@app.route('/api/session', methods=['GET'])
def api_session():
    """
    GET /api/session
    Returns current session metadata: tunnel URL, expiries, session start time.
    """
    try:
        import pathlib
        session_path = pathlib.Path(__file__).parent / 'fintel_session.json'
        session_data = {}
        if session_path.exists():
            try:
                session_data = json.loads(session_path.read_text(encoding='utf-8'))
            except Exception:
                pass
        return jsonify({'ok': True, **session_data,
                        'server_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')})
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)}), 500


# ─────────────────────────────────────────────────────────────────────────────
# READINESS PROBE — 503 until spot + chain are populated
# ─────────────────────────────────────────────────────────────────────────────
@app.route('/ready', methods=['GET'])
def ready():
    """Startup readiness gate for deployment health checks."""
    spot  = hub.latest_data.get('spot', 0)
    chain = hub.latest_data.get('chain', {})
    if spot > 0 and chain.get('optionsChain'):
        return jsonify({'ok': True, 'spot': spot,
                        'status': hub.latest_data.get('status', 'Live')}), 200
    return jsonify({'ok': False, 'reason': 'Waiting for data',
                    'spot': spot, 'chain_populated': bool(chain.get('optionsChain'))}), 503


# ─────────────────────────────────────────────────────────────────────────────
# PORTFOLIO HISTORY ENDPOINT
# ─────────────────────────────────────────────────────────────────────────────
@app.route('/api/portfolio/history', methods=['GET'])
def api_portfolio_history():
    """GET /api/portfolio/history — closed position history."""
    if not portfolio_mgr:
        return jsonify({'ok': False, 'error': 'PortfolioManager not loaded'}), 500
    try:
        history = portfolio_mgr.data.get('history', [])
        return jsonify({'ok': True, 'history': history[-50:]})
    except Exception as e:
        import traceback
        return jsonify({'ok': False, 'error': str(e),
                        'trace': traceback.format_exc()}), 500


@app.route('/builder')
def builder_page():
    return send_file('strategy_builder.html')


# ─────────────────────────────────────────────────────────────────────────────
# THETA DECAY EXPLORER  —  /api/theta_decay
# ─────────────────────────────────────────────────────────────────────────────
@app.route('/api/theta_decay', methods=['GET'])
def api_theta_decay():
    """
    GET /api/theta_decay
    Computes a Strike × DTE grid of BSM and/or Heston Greeks (Θ, Δ, Γ, V).

    Query params:
        iv        – implied volatility % (default: from ATM chain or 15)
        r         – risk-free rate       (default: 0.07)
        q         – dividend yield       (default: 0.0122)
        opt_type  – CE or PE             (default: CE)
        model     – bsm | heston | both  (default: both)
        range_pct – strike band ±%       (default: 10)
        dte_steps – comma-sep DTE list   (default: 7,6,5,4,3,2,1,0.5,0.25,0.1)

    Response:
        { ok, spot, strikes:[float], dte_steps:[float],
          series: { bsm: { theta, delta, gamma, vega }, heston: {...} } }
        Each series key is a 2-D list [strike_idx][dte_idx].
    """
    try:
        from OptionAnalytics import OptionAnalytics
        from scipy.stats import norm
        import numpy as np

        # ── Read params ──────────────────────────────────────────────────────
        spot       = float(request.args.get('spot', 0) or hub.latest_data.get('spot', 0) or 24000)
        iv_pct     = float(request.args.get('iv', 0) or 15)
        r          = float(request.args.get('r', 0.07))
        q          = float(request.args.get('q', 0.0122))
        opt_type   = request.args.get('opt_type', 'CE').upper()
        model_req  = request.args.get('model', 'both').lower()   # bsm | heston | both
        range_pct  = float(request.args.get('range_pct', 10)) / 100.0
        dt_min     = float(request.args.get('dt_min', 0) or 0)
        dt_days    = float(request.args.get('dt_days', 0) or 0)
        d_spot     = float(request.args.get('d_spot', 0) or 0)
        d_iv       = float(request.args.get('d_iv', 0) or 0)
        lot_size   = 65.0

        raw_dte    = request.args.get('dte_steps', '7,6,5,4,3,2,1,0.5,0.25,0.1')
        dte_steps  = [float(x) for x in raw_dte.split(',') if x.strip()]

        # ── Option chain from live feed / cache ──────────────────────────────
        chain_raw = hub.latest_data.get('chain', {})
        options   = chain_raw.get('optionsChain', [])
        if not options and hub_cache:
            try:
                cached_chain = hub_cache.get_raw_chain() or {}
                options = cached_chain.get('optionsChain', [])
            except Exception:
                pass

        # Time to expiry (current) for IV inversion
        T_current = 7.0 / 365.0
        if hub_cache:
            try:
                T_val = hub_cache.get_T()
                if T_val and T_val > 0:
                    T_current = T_val
            except Exception:
                pass

        # ── Strikes selection (within ±range_pct of spot) ───────────────────
        lo, hi = spot * (1.0 - range_pct), spot * (1.0 + range_pct)
        if options:
            all_chain_strikes = sorted({
                float(o.get('strike_price', 0))
                for o in options
                if lo <= float(o.get('strike_price', 0) or 0) <= hi
            })
            atm_strike = min(all_chain_strikes, key=lambda s: abs(s - spot)) if all_chain_strikes else round(spot / 50.0) * 50.0

            # Cap strikes to ~25 for high performance while guaranteeing ATM strike
            if len(all_chain_strikes) > 25:
                step = max(1, len(all_chain_strikes) // 25)
                selected = all_chain_strikes[::step]
                if atm_strike not in selected:
                    selected.append(atm_strike)
                raw_strikes = sorted(selected)
            else:
                raw_strikes = all_chain_strikes
        else:
            # Fallback synthetic grid
            atm_strike = round(spot / 50.0) * 50.0
            raw_strikes = [atm_strike + i * 50 for i in range(-12, 13)]
            raw_strikes = [s for s in raw_strikes if lo <= s <= hi]

        strikes = sorted(raw_strikes)
        if not strikes:
            strikes = [round(spot / 50.0) * 50.0]
        atm_strike = min(strikes, key=lambda s: abs(s - spot))

        # ── Build fast lookup dict: (strike, type) -> option row ─────────────
        lookup = {}
        for o in options:
            try:
                s_val = float(o.get('strike_price', 0) or 0)
                ot_val = 'CE' if o.get('option_type', '') in ('CE', 'CALL') else 'PE'
                lookup[(s_val, ot_val)] = o
            except Exception:
                pass

        # ATM chain IV as baseline fallback
        atm_opt = lookup.get((atm_strike, opt_type)) or lookup.get((atm_strike, 'CE')) or {}
        chain_atm_iv = float(atm_opt.get('iv', 0) or 0)
        default_iv = chain_atm_iv if chain_atm_iv > 0.5 else (iv_pct if iv_pct > 0.5 else 15.0)

        # ── Resolve Live IV & LTP per strike (Chain -> Parity -> BSM -> Fallback) ──
        analytics = OptionAnalytics()
        iv_per_strike = {}
        iv_source_per_strike = {}
        ltp_per_strike = {}

        for K in strikes:
            if opt_type == 'STRADDLE':
                ce_opt = lookup.get((K, 'CE')) or {}
                pe_opt = lookup.get((K, 'PE')) or {}
                ltp_ce = float(ce_opt.get('ltp', 0) or 0)
                ltp_pe = float(pe_opt.get('ltp', 0) or 0)
                ltp_per_strike[K] = round(ltp_ce + ltp_pe, 2)

                iv_ce = float(ce_opt.get('iv', 0) or 0)
                iv_pe = float(pe_opt.get('iv', 0) or 0)
                if iv_ce > 0.5 and iv_pe > 0.5:
                    iv_per_strike[K] = round((iv_ce + iv_pe) / 2.0, 2)
                    iv_source_per_strike[K] = 'chain'
                elif iv_ce > 0.5:
                    iv_per_strike[K] = round(iv_ce, 2)
                    iv_source_per_strike[K] = 'chain'
                elif iv_pe > 0.5:
                    iv_per_strike[K] = round(iv_pe, 2)
                    iv_source_per_strike[K] = 'chain'
                else:
                    iv_per_strike[K] = round(default_iv, 2)
                    iv_source_per_strike[K] = 'fallback'
            else:
                opt = lookup.get((K, opt_type))
                opp_type = 'PE' if opt_type == 'CE' else 'CE'
                opp_opt = lookup.get((K, opp_type))

                ltp = float(opt.get('ltp', 0) or 0) if opt else 0.0
                iv_chain = float(opt.get('iv', 0) or 0) if opt else 0.0
                opp_iv = float(opp_opt.get('iv', 0) or 0) if opp_opt else 0.0
                opp_ltp = float(opp_opt.get('ltp', 0) or 0) if opp_opt else 0.0

                ltp_per_strike[K] = ltp

                # 1. Primary: Direct IV shown in option chain for this strike
                if iv_chain > 0.5:
                    iv_per_strike[K] = round(iv_chain, 2)
                    iv_source_per_strike[K] = 'chain'
                # 2. Put-Call Parity: In broker feeds (e.g. Fyers/NSE), ITM options often
                # report IV=0; by put-call parity, ITM IV == OTM IV at same strike
                elif opp_iv > 0.5:
                    iv_per_strike[K] = round(opp_iv, 2)
                    iv_source_per_strike[K] = 'parity'
                # 3. Fallback calculation using BSM implied volatility from live price (LTP)
                elif ltp > 0 and spot > 0:
                    T_calc = max(float(T_current or 0), 1e-4)
                    calc_iv = analytics.implied_volatility(ltp, spot, K, T_calc, r, opt_type, q=q)
                    if calc_iv and 0.5 < calc_iv < 250.0:
                        iv_per_strike[K] = round(float(calc_iv), 2)
                        iv_source_per_strike[K] = 'bsm_calc'
                    elif opp_ltp > 0:
                        opp_calc = analytics.implied_volatility(opp_ltp, spot, K, T_calc, r, opp_type, q=q)
                        if opp_calc and 0.5 < opp_calc < 250.0:
                            iv_per_strike[K] = round(float(opp_calc), 2)
                            iv_source_per_strike[K] = 'bsm_calc'
                        else:
                            iv_per_strike[K] = round(default_iv, 2)
                            iv_source_per_strike[K] = 'fallback'
                    else:
                        iv_per_strike[K] = round(default_iv, 2)
                        iv_source_per_strike[K] = 'fallback'
                else:
                    iv_per_strike[K] = round(default_iv, 2)
                    iv_source_per_strike[K] = 'fallback'

        # ── Greeks & Pricing Calculation Helper (Merton 1973 BS with continuous div yield) ──
        def _bsm_calc_price(spot_val, K_val, T_val, r_val, sig_val, otype, q=0.0):
            if T_val <= 0 or spot_val <= 0 or K_val <= 0:
                return max(0.0, spot_val - K_val) if otype == 'CE' else max(0.0, K_val - spot_val)
            d1 = (np.log(spot_val / K_val) + (r_val - q + 0.5 * sig_val ** 2) * T_val) / (sig_val * np.sqrt(T_val))
            d2 = d1 - sig_val * np.sqrt(T_val)
            if otype == 'CE':
                return spot_val * np.exp(-q * T_val) * norm.cdf(d1) - K_val * np.exp(-r_val * T_val) * norm.cdf(d2)
            else:
                return K_val * np.exp(-r_val * T_val) * norm.cdf(-d2) - spot_val * np.exp(-q * T_val) * norm.cdf(-d1)

        def _bsm_calc_greeks(spot_val, K_val, T_val, r_val, sig_val, otype, q=0.0):
            if T_val <= 0 or sig_val <= 0 or K_val <= 0 or spot_val <= 0:
                return {'delta': 0.0, 'gamma': 0.0, 'vega': 0.0, 'theta': 0.0}
            d1 = (np.log(spot_val / K_val) + (r_val - q + 0.5 * sig_val ** 2) * T_val) / (sig_val * np.sqrt(T_val))
            d2 = d1 - sig_val * np.sqrt(T_val)
            pdf_d1 = norm.pdf(d1)
            exp_qT = np.exp(-q * T_val)
            exp_rT = np.exp(-r_val * T_val)
            gamma_val = exp_qT * pdf_d1 / (spot_val * sig_val * np.sqrt(T_val))
            vega_val  = (spot_val * exp_qT * pdf_d1 * np.sqrt(T_val)) / 100.0  # per 1% vol
            if otype == 'CE':
                delta_val = exp_qT * norm.cdf(d1)
                theta_val = (-spot_val * exp_qT * pdf_d1 * sig_val / (2.0 * np.sqrt(T_val))
                             - r_val * K_val * exp_rT * norm.cdf(d2)
                             + q * spot_val * exp_qT * norm.cdf(d1)) / 365.0
            else:
                delta_val = exp_qT * (norm.cdf(d1) - 1.0)
                theta_val = (-spot_val * exp_qT * pdf_d1 * sig_val / (2.0 * np.sqrt(T_val))
                             + r_val * K_val * exp_rT * norm.cdf(-d2)
                             - q * spot_val * exp_qT * norm.cdf(-d1)) / 365.0
            return {
                'delta': round(float(delta_val), 4),
                'gamma': round(float(gamma_val), 6),
                'vega':  round(float(vega_val),  4),
                'theta': round(float(theta_val), 4)
            }

        # ── BSM Greeks Computation Grid across Strikes & DTE ─────────────────
        bsm_series = {'theta': [], 'delta': [], 'gamma': [], 'vega': []}
        for K in strikes:
            sig = iv_per_strike[K] / 100.0
            row = {'theta': [], 'delta': [], 'gamma': [], 'vega': []}
            for dte in dte_steps:
                T_dte = max(dte / 365.0, 1e-5)
                if opt_type == 'STRADDLE':
                    g_ce = _bsm_calc_greeks(spot, K, T_dte, r, sig, 'CE', q=q)
                    g_pe = _bsm_calc_greeks(spot, K, T_dte, r, sig, 'PE', q=q)
                    row['theta'].append(round(g_ce['theta'] + g_pe['theta'], 4))
                    row['delta'].append(round(g_ce['delta'] + g_pe['delta'], 4))
                    row['gamma'].append(round(g_ce['gamma'] + g_pe['gamma'], 6))
                    row['vega'].append(round(g_ce['vega'] + g_pe['vega'], 4))
                else:
                    g = _bsm_calc_greeks(spot, K, T_dte, r, sig, opt_type, q=q)
                    for key in row:
                        row[key].append(g[key])
            for key in bsm_series:
                bsm_series[key].append(row[key])

        # ── Heston Model Computation Grid (optional / live calibrated) ───────
        heston_series = None
        if model_req in ('heston', 'both'):
            try:
                from NiftyHestonMC import HestonMath
                heston_params = None

                # Try SharedDataCache first
                try:
                    sc = hub_cache
                    heston_params = sc.get_heston_params() if sc else None
                except Exception:
                    pass

                # Fallback: lightweight defaults (calibrated typical NSE Nifty)
                if not heston_params:
                    atm_sig = default_iv / 100.0
                    heston_params = {
                        'kappa': 2.0, 'theta': atm_sig ** 2,
                        'v0': atm_sig ** 2, 'rho': -0.7, 'xi': 0.5
                    }

                h_series = {'theta': [], 'delta': [], 'gamma': [], 'vega': []}
                bump = spot * 0.001   # 0.1% spot bump for finite-diff Greeks

                kp      = float(heston_params['kappa'])
                th      = float(heston_params['theta'])
                v0_base = float(heston_params['v0'])
                rho     = float(heston_params['rho'])
                xi      = float(heston_params['xi'])

                for K in strikes:
                    strike_sig = iv_per_strike[K] / 100.0
                    row = {'theta': [], 'delta': [], 'gamma': [], 'vega': []}
                    for dte in dte_steps:
                        T  = max(dte / 365.0, 1e-5)
                        dT = min(1e-4, T * 0.2)
                        T_plus  = T + dT
                        T_minus = max(T - dT, 1e-6)

                        try:
                            sub_types = ['CE', 'PE'] if opt_type == 'STRADDLE' else [opt_type]
                            t_th, t_del, t_gam, t_veg = 0.0, 0.0, 0.0, 0.0
                            for st in sub_types:
                                parity       = 0.0
                                ce       = max(HestonMath.price_vanilla_call(spot, K, T, r, kp, th, v0_base, rho, xi), 0.0)
                                ceU      = max(HestonMath.price_vanilla_call(spot + bump, K, T, r, kp, th, v0_base, rho, xi), 0.0)
                                ceD      = max(HestonMath.price_vanilla_call(spot - bump, K, T, r, kp, th, v0_base, rho, xi), 0.0)
                                ce_plus  = max(HestonMath.price_vanilla_call(spot, K, T_plus, r, kp, th, v0_base, rho, xi), 0.0)
                                ce_minus = max(HestonMath.price_vanilla_call(spot, K, T_minus, r, kp, th, v0_base, rho, xi), 0.0)

                                if st == 'PE':
                                    parity       = K * np.exp(-r * T) - spot * np.exp(-q * T)
                                    parity_plus  = K * np.exp(-r * T_plus) - spot * np.exp(-q * T_plus)
                                    parity_minus = K * np.exp(-r * T_minus) - spot * np.exp(-q * T_minus)
                                    ce       = ce + parity
                                    ceU      = ceU + parity
                                    ceD      = ceD + parity
                                    ce_plus  = ce_plus + parity_plus
                                    ce_minus = ce_minus + parity_minus

                                d_val = (ceU - ceD) / (2.0 * bump)
                                g_val = (ceU - 2.0 * ce + ceD) / (bump ** 2)
                                th_val = -(ce_plus - ce_minus) / (2.0 * dT * 365.0)

                                v0_up = (np.sqrt(max(v0_base, 1e-6)) + 0.01) ** 2
                                ceV = max(HestonMath.price_vanilla_call(spot, K, T, r, kp, th, v0_up, rho, xi), 0.0)
                                if st == 'PE':
                                    ceV = ceV + parity
                                v_val = (ceV - ce) / 100.0

                                t_del += d_val
                                t_gam += g_val
                                t_th  += th_val
                                t_veg += v_val

                            row['theta'].append(round(float(t_th), 4))
                            row['delta'].append(round(float(t_del), 4))
                            row['gamma'].append(round(float(t_gam), 6))
                            row['vega'].append(round(float(t_veg), 4))
                        except Exception:
                            if opt_type == 'STRADDLE':
                                g_ce = _bsm_calc_greeks(spot, K, T, r, strike_sig, 'CE', q=q)
                                g_pe = _bsm_calc_greeks(spot, K, T, r, strike_sig, 'PE', q=q)
                                row['theta'].append(round(g_ce['theta'] + g_pe['theta'], 4))
                                row['delta'].append(round(g_ce['delta'] + g_pe['delta'], 4))
                                row['gamma'].append(round(g_ce['gamma'] + g_pe['gamma'], 6))
                                row['vega'].append(round(g_ce['vega'] + g_pe['vega'], 4))
                            else:
                                g = _bsm_calc_greeks(spot, K, T, r, strike_sig, opt_type, q=q)
                                for key in row:
                                    row[key].append(g[key])

                    for key in h_series:
                        h_series[key].append(row[key])

                heston_series = h_series
            except Exception as he:
                print(f"[theta_decay] Heston branch failed (non-fatal): {he}")

        # ── Extended Decision, Straddle & Forward Simulation Analytics ────────
        strikes_analytics = {}
        target_dT = (dt_days + (dt_min / 375.0)) / 365.0
        T_proj = max(float(T_current or (7.0 / 365.0)) - target_dT, 1e-6)
        spot_proj = max(spot + d_spot, 10.0)

        # Baseline RV for edge assessment
        rv_consensus = 13.0
        try:
            if hub_cache and hasattr(hub_cache, 'get_rv'):
                r_val = hub_cache.get_rv()
                if r_val and r_val > 1.0:
                    rv_consensus = float(r_val)
        except Exception:
            pass

        # Expected Move (1 standard deviation move based on ATM IV and DTE)
        atm_dte_years = max(float(T_current or (7.0 / 365.0)), 1e-4)
        atm_iv_dec = (default_iv / 100.0)
        expected_move_pts = round(float(spot * atm_iv_dec * np.sqrt(atm_dte_years)), 1)
        if expected_move_pts < 10.0:
            expected_move_pts = round(spot * 0.008, 1)

        for i, K in enumerate(strikes):
            ce_o = lookup.get((K, 'CE')) or {}
            pe_o = lookup.get((K, 'PE')) or {}
            c_ltp = float(ce_o.get('ltp', 0) or 0)
            p_ltp = float(pe_o.get('ltp', 0) or 0)
            st_ltp = round(c_ltp + p_ltp, 2)

            # Active LTP according to opt_type
            active_ltp = st_ltp if opt_type == 'STRADDLE' else (c_ltp if opt_type == 'CE' else p_ltp)
            
            # Intrinsic & Extrinsic time value
            if opt_type == 'STRADDLE':
                intr = abs(spot - K)
            elif opt_type == 'CE':
                intr = max(0.0, spot - K)
            else:
                intr = max(0.0, K - spot)
            
            extrinsic_decay_pts = round(max(0.0, active_ltp - intr), 2)
            extrinsic_decay_inr = round(extrinsic_decay_pts * lot_size, 0)
            extrinsic_pct = round((extrinsic_decay_pts / max(active_ltp, 0.1)) * 100.0, 1)

            # Greeks at current market
            th_day_pts = abs(bsm_series['theta'][i][0]) if bsm_series['theta'][i] else 0.0
            gamma_val = bsm_series['gamma'][i][0] if bsm_series['gamma'][i] else 1e-6
            delta_val = bsm_series['delta'][i][0] if bsm_series['delta'][i] else 0.0
            vega_val = bsm_series['vega'][i][0] if bsm_series['vega'][i] else 0.0

            # Daily Breakeven Cushion Move (pts)
            cushion_pts = round(float(np.sqrt(max(2.0 * th_day_pts / max(gamma_val, 1e-7), 0.0))), 1)
            decay_yield = round((th_day_pts / max(active_ltp, 0.1)) * 100.0, 1)

            # Forward Simulation Pricing (at T_proj, spot_proj, sig_proj)
            k_sig_base = iv_per_strike[K] / 100.0
            k_sig_proj = max(0.02, (iv_per_strike[K] + d_iv) / 100.0)
            
            if opt_type == 'STRADDLE':
                p_curr_model = _bsm_calc_price(spot, K, max(float(T_current or 0.01), 1e-5), r, k_sig_base, 'CE', q) + \
                               _bsm_calc_price(spot, K, max(float(T_current or 0.01), 1e-5), r, k_sig_base, 'PE', q)
                p_proj_model = _bsm_calc_price(spot_proj, K, T_proj, r, k_sig_proj, 'CE', q) + \
                               _bsm_calc_price(spot_proj, K, T_proj, r, k_sig_proj, 'PE', q)
            else:
                p_curr_model = _bsm_calc_price(spot, K, max(float(T_current or 0.01), 1e-5), r, k_sig_base, opt_type, q)
                p_proj_model = _bsm_calc_price(spot_proj, K, T_proj, r, k_sig_proj, opt_type, q)

            model_dp = round(p_proj_model - p_curr_model, 2)
            model_dp_inr = round(model_dp * lot_size, 0)
            proj_ltp = round(max(0.05, active_ltp + model_dp), 2)

            # Greek Attribution for this strike
            th_attr_pts = round(-th_day_pts * (dt_days + (dt_min / 375.0)), 2)
            del_attr_pts = round(delta_val * d_spot, 2)
            gam_attr_pts = round(0.5 * gamma_val * (d_spot ** 2), 2)
            veg_attr_pts = round(vega_val * d_iv, 2)
            taylor_dp_pts = round(th_attr_pts + del_attr_pts + gam_attr_pts + veg_attr_pts, 2)

            # Granular Call & Put Greeks and Decays
            g_ce = _bsm_calc_greeks(spot, K, max(float(T_current or 0.01), 1e-5), r, k_sig_base, 'CE', q)
            g_pe = _bsm_calc_greeks(spot, K, max(float(T_current or 0.01), 1e-5), r, k_sig_base, 'PE', q)
            ce_th_day = abs(g_ce['theta'])
            pe_th_day = abs(g_pe['theta'])
            strad_th_day = round(ce_th_day + pe_th_day, 2)

            ce_intr = max(0.0, spot - K)
            pe_intr = max(0.0, K - spot)
            ce_ext = max(0.0, c_ltp - ce_intr)
            pe_ext = max(0.0, p_ltp - pe_intr)
            st_ext = ce_ext + pe_ext

            ce_decay = {
                'per_day_pts': round(ce_th_day, 2),
                'per_day_inr': round(ce_th_day * lot_size, 0),
                'per_hour_inr': round((ce_th_day * lot_size) / 6.25, 1),
                'per_min_inr': round((ce_th_day * lot_size) / 375.0, 2),
                'till_expiry_pts': round(ce_ext, 1),
                'till_expiry_inr': round(ce_ext * lot_size, 0),
                'yield_pct': round((ce_th_day / max(c_ltp, 0.1)) * 100.0, 1),
                'delta': g_ce['delta'],
                'gamma': g_ce['gamma'],
                'vega': g_ce['vega']
            }

            pe_decay = {
                'per_day_pts': round(pe_th_day, 2),
                'per_day_inr': round(pe_th_day * lot_size, 0),
                'per_hour_inr': round((pe_th_day * lot_size) / 6.25, 1),
                'per_min_inr': round((pe_th_day * lot_size) / 375.0, 2),
                'till_expiry_pts': round(pe_ext, 1),
                'till_expiry_inr': round(pe_ext * lot_size, 0),
                'yield_pct': round((pe_th_day / max(p_ltp, 0.1)) * 100.0, 1),
                'delta': g_pe['delta'],
                'gamma': g_pe['gamma'],
                'vega': g_pe['vega']
            }

            straddle_decay = {
                'per_day_pts': strad_th_day,
                'per_day_inr': round(strad_th_day * lot_size, 0),
                'per_hour_inr': round((strad_th_day * lot_size) / 6.25, 1),
                'per_min_inr': round((strad_th_day * lot_size) / 375.0, 2),
                'till_expiry_pts': round(st_ext, 1),
                'till_expiry_inr': round(st_ext * lot_size, 0),
                'yield_pct': round((strad_th_day / max(st_ltp, 0.1)) * 100.0, 1),
                'delta': round(g_ce['delta'] + g_pe['delta'], 4),
                'gamma': round(g_ce['gamma'] + g_pe['gamma'], 6),
                'vega': round(g_ce['vega'] + g_pe['vega'], 4)
            }

            # Renormalized Alpha Metric (Theta / 1-sigma Gamma Risk)
            gamma_hazard_inr = 0.5 * straddle_decay['gamma'] * (expected_move_pts ** 2) * lot_size
            renorm_alpha = round(straddle_decay['per_day_inr'] / max(gamma_hazard_inr, 1.0), 2)

            # Buy vs Sell Edge Scoring
            s_yield_score = min(40.0, decay_yield * 2.5)
            s_cushion_score = min(35.0, (cushion_pts / max(spot * 0.008, 1.0)) * 25.0)
            s_vol_score = min(25.0, max(0.0, (iv_per_strike[K] - rv_consensus) * 6.0))
            sell_edge_score = int(round(s_yield_score + s_cushion_score + s_vol_score))

            convexity = (gamma_val / max(active_ltp, 0.5)) * 1000.0
            b_conv_score = min(45.0, convexity * 15.0)
            vel_req = th_day_pts / (max(abs(delta_val), 0.08) * 6.25)
            b_vel_score = max(0.0, 35.0 - vel_req * 1.2)
            b_vol_score = min(20.0, max(0.0, (rv_consensus - iv_per_strike[K]) * 5.0))
            buy_edge_score = int(round(b_conv_score + b_vel_score + b_vol_score))

            if sell_edge_score >= 70:
                trade_verdict = 'STRONG SELL'
                trade_color = '#10b981'
            elif buy_edge_score >= 70:
                trade_verdict = 'PRIME BUY'
                trade_color = '#4fc3f7'
            elif cushion_pts < 60 and float(T_current or 0) < 1.0 / 365.0:
                trade_verdict = 'GAMMA HAZARD'
                trade_color = '#ef5350'
            elif sell_edge_score >= 55:
                trade_verdict = 'FAIR SELL'
                trade_color = '#66bb6a'
            elif buy_edge_score >= 55:
                trade_verdict = 'RUNNER BUY'
                trade_color = '#29b6f6'
            else:
                trade_verdict = 'NEUTRAL'
                trade_color = '#888888'

            if extrinsic_pct <= 25.0:
                retention_status = 'TAKE PROFIT'
                retention_reason = f'{100-extrinsic_pct:.0f}% decay captured; tail risk remaining'
                retention_color = '#ffd54f'
            elif cushion_pts < 75 and float(T_current or 0) < 1.0 / 365.0:
                retention_status = 'EXIT / ROLL'
                retention_reason = 'Gamma explosion risk outweighs theta'
                retention_color = '#ef5350'
            elif abs(spot - K) > 0.75 * cushion_pts:
                retention_status = 'DEFEND / HEDGE'
                retention_reason = 'Spot near breakeven limit; adjust or cut'
                retention_color = '#ff9800'
            else:
                retention_status = 'STAY (OPTIMAL)'
                retention_reason = 'Healthy theta harvest with safe cushion'
                retention_color = '#10b981'

            strikes_analytics[str(K)] = {
                'ce_ltp': c_ltp,
                'pe_ltp': p_ltp,
                'straddle_ltp': st_ltp,
                'active_ltp': active_ltp,
                'intrinsic_pts': round(intr, 2),
                'extrinsic_pts': extrinsic_decay_pts,
                'extrinsic_inr': extrinsic_decay_inr,
                'extrinsic_pct': extrinsic_pct,
                'th_day_pts': th_day_pts,
                'th_day_inr': round(th_day_pts * lot_size, 0),
                'th_1h_inr': round((th_day_pts * lot_size) / 6.25, 1),
                'gamma': gamma_val,
                'delta': delta_val,
                'vega': vega_val,
                'cushion_pts': cushion_pts,
                'decay_yield': decay_yield,
                'ce_decay': ce_decay,
                'pe_decay': pe_decay,
                'straddle_decay': straddle_decay,
                'renorm_alpha': renorm_alpha,
                'normalized_moneyness': round(float((K - spot) / max(expected_move_pts, 1.0)), 2),
                'proj_ltp': proj_ltp,
                'model_dp_pts': model_dp,
                'model_dp_inr': model_dp_inr,
                'attribution': {
                    'theta_pts': th_attr_pts,
                    'delta_pts': del_attr_pts,
                    'gamma_pts': gam_attr_pts,
                    'vega_pts': veg_attr_pts,
                    'sum_pts': taylor_dp_pts,
                    'theta_inr': round(th_attr_pts * lot_size, 0),
                    'delta_inr': round(del_attr_pts * lot_size, 0),
                    'gamma_inr': round(gam_attr_pts * lot_size, 0),
                    'vega_inr': round(veg_attr_pts * lot_size, 0),
                    'sum_inr': round(taylor_dp_pts * lot_size, 0),
                },
                'sell_edge_score': sell_edge_score,
                'buy_edge_score': buy_edge_score,
                'trade_verdict': trade_verdict,
                'trade_color': trade_color,
                'retention_status': retention_status,
                'retention_reason': retention_reason,
                'retention_color': retention_color
            }

        atm_str = str(atm_strike)
        atm_analytics = strikes_analytics.get(atm_str, {})

        # ── 9-Step Normalized Scenario Matrix (Z in [-2.0 to +2.0] Standard Deviations) ──
        normalized_scenarios = []
        atm_opt_ce = lookup.get((atm_strike, 'CE')) or {}
        atm_opt_pe = lookup.get((atm_strike, 'PE')) or {}
        atm_ce_ltp = float(atm_opt_ce.get('ltp', 0) or 0)
        atm_pe_ltp = float(atm_opt_pe.get('ltp', 0) or 0)
        atm_strad_base = atm_ce_ltp + atm_pe_ltp
        atm_sig = iv_per_strike.get(atm_strike, default_iv) / 100.0
        atm_sig_proj = max(0.02, (iv_per_strike.get(atm_strike, default_iv) + d_iv) / 100.0)

        base_ce_m = _bsm_calc_price(spot, atm_strike, atm_dte_years, r, atm_sig, 'CE', q)
        base_pe_m = _bsm_calc_price(spot, atm_strike, atm_dte_years, r, atm_sig, 'PE', q)

        for z in [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]:
            ds_sim = round(float(z * expected_move_pts), 1)
            s_sim = max(spot + ds_sim, 10.0)
            p_ce_sim = _bsm_calc_price(s_sim, atm_strike, T_proj, r, atm_sig_proj, 'CE', q)
            p_pe_sim = _bsm_calc_price(s_sim, atm_strike, T_proj, r, atm_sig_proj, 'PE', q)

            d_ce = p_ce_sim - base_ce_m
            d_pe = p_pe_sim - base_pe_m

            proj_ce = round(max(0.05, atm_ce_ltp + d_ce), 1)
            proj_pe = round(max(0.05, atm_pe_ltp + d_pe), 1)
            proj_st = round(proj_ce + proj_pe, 1)

            pnl_seller_pts = round(atm_strad_base - proj_st, 1)
            pnl_seller_inr = round(pnl_seller_pts * lot_size, 0)
            pnl_seller_pct = round((pnl_seller_pts / max(atm_strad_base, 0.1)) * 100.0, 1)

            pnl_buyer_inr = -pnl_seller_inr
            pnl_buyer_pct = -pnl_seller_pct

            if abs(z) >= 1.5:
                dom_cause = 'Gamma Hazard (Curvature Loss)'
                scen_status = 'DANGER / DEFEND'
                scen_color = '#ef5350'
            elif abs(z) >= 1.0:
                dom_cause = 'Spot at Breakeven Edge'
                scen_status = 'BREAKEVEN EDGE'
                scen_color = '#ffd54f'
            elif abs(z) >= 0.5:
                dom_cause = 'Theta Buffering Spot Move'
                scen_status = 'SAFE / HARVEST'
                scen_color = '#10b981'
            else:
                dom_cause = 'Pure Calendar Bleed'
                scen_status = 'MAXIMUM HARVEST'
                scen_color = '#10b981'

            normalized_scenarios.append({
                'z_score': z,
                'spot_shift': ds_sim,
                'spot_level': round(s_sim, 1),
                'ce_ltp': proj_ce,
                'pe_ltp': proj_pe,
                'straddle_ltp': proj_st,
                'seller_pnl_inr': pnl_seller_inr,
                'seller_pnl_pct': pnl_seller_pct,
                'buyer_pnl_inr': pnl_buyer_inr,
                'buyer_pnl_pct': pnl_buyer_pct,
                'dominant_cause': dom_cause,
                'status': scen_status,
                'color': scen_color
            })

        # ── Build response payload ───────────────────────────────────────────
        series_out = {'bsm': bsm_series}
        if heston_series is not None:
            series_out['heston'] = heston_series

        return jsonify({
            'ok':                   True,
            'spot':                 spot,
            'atm_strike':           atm_strike,
            'expected_move':        expected_move_pts,
            'iv_used':              round(default_iv, 2),
            'opt_type':             opt_type,
            'model':                model_req,
            'strikes':              strikes,
            'dte_steps':            dte_steps,
            'iv_per_strike':        {str(K): iv_per_strike[K] for K in strikes},
            'iv_source_per_strike': {str(K): iv_source_per_strike[K] for K in strikes},
            'ltp_per_strike':       {str(K): ltp_per_strike[K] for K in strikes},
            'series':               series_out,
            'strikes_analytics':    strikes_analytics,
            'atm_analytics':        atm_analytics,
            'normalized_scenarios': normalized_scenarios,
            'simulation_params': {
                'dt_min':   dt_min,
                'dt_days':  dt_days,
                'd_spot':   d_spot,
                'd_iv':     d_iv,
            }
        })


    except Exception as e:
        import traceback
        return jsonify({'ok': False, 'error': str(e),
                        'trace': traceback.format_exc()}), 500


@app.route('/health', methods=['GET'])
def health():
    """Quick liveness probe — returns spot, status, last tick time, and server time."""
    return jsonify({
        'ok':          True,
        'spot':        hub.latest_data.get('spot', 0),
        'status':      hub.latest_data.get('status', 'Unknown'),
        'last_update': hub.latest_data.get('last_update'),          # last Fyers tick HH:MM:SS
        'tick_count':  hub.latest_data.get('tick_count', 0),
        'server_time': datetime.now().strftime('%H:%M:%S'),          # live server clock
    })


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION & PROFILES ENDPOINTS
# ─────────────────────────────────────────────────────────────────────────────
@app.route('/api/config', methods=['GET'])
def get_config():
    """GET /api/config — returns active configuration and available profiles."""
    if not config:
        return jsonify({'ok': False, 'error': 'config module not loaded'}), 500
    try:
        active_cfg = dict(config._active)
        profiles = []
        try:
            cfg_file = config._CONFIG_PATH
            if cfg_file.exists():
                with open(cfg_file, 'r') as f:
                    raw = json.load(f)
                profiles = list(raw.get("profiles", {}).keys())
        except Exception:
            pass
        return jsonify({
            'ok': True,
            'config': active_cfg,
            'profiles': profiles,
            'active_profile': getattr(config, '_active_profile', 'default')
        })
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)}), 500

@app.route('/api/config', methods=['PATCH', 'POST'])
def update_config():
    """PATCH/POST /api/config — update configuration values."""
    if not config:
        return jsonify({'ok': False, 'error': 'config module not loaded'}), 500
    try:
        data = request.get_json() or {}
        if not data:
            return jsonify({'ok': False, 'error': 'No data provided'}), 400
        config.save_config(data)
        return jsonify({
            'ok': True,
            'message': 'Configuration updated',
            'config': dict(config._active)
        })
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)}), 500

@app.route('/api/config/profile/<profile_name>', methods=['POST'])
def switch_profile(profile_name):
    """POST /api/config/profile/<profile_name> — switch active profile (in-memory)."""
    if not config:
        return jsonify({'ok': False, 'error': 'config module not loaded'}), 500
    try:
        success = config.load_profile(profile_name)
        if success:
            return jsonify({
                'ok': True,
                'message': f'Switched to profile {profile_name}',
                'profile': profile_name,
                'config': dict(config._active)
            })
        else:
            return jsonify({
                'ok': False,
                'error': f'Profile "{profile_name}" not found or failed to load'
            }), 404
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)}), 500


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    hub.start()
    print(f"DataHub Real-Time server starting on port {PORT}...")

    # ── Background Heston Calibrator ──────────────────────────────────
    try:
        from HestonCalibrator import HestonCalibrator
        _calibrator = HestonCalibrator(hub_cache, hub.fyers)
        _calibrator.start()
    except Exception as _e:
        print(f"[DataServer] HestonCalibrator could not start (non-fatal): {_e}")

    # ── Background GEX Refresher ───────────────────────────────────────
    threading.Thread(target=_gex_refresh_loop, args=(120,),
                     daemon=True, name="GEXRefresher").start()
    print("[DataServer] GEX refresh thread started (interval=120s).")

    app.run(port=PORT, debug=False, use_reloader=False)

