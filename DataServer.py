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
from flask import Flask, jsonify, send_file
from flask_cors import CORS
from flask_sock import Sock
from datetime import datetime
from fyers_auth_manager import get_fyers_instance, get_access_token
from fyers_apiv3.FyersWebsocket import data_ws
from OptionAnalytics import OptionAnalytics

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
def index():
    return send_file('unified_dashboard.html')

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
                    gex_eng = GexEngine(lot_size=65, positioning_model='standard')
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
                        dep   = DealerPositionEngine(lot_size=65)
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

@app.route('/builder')
def builder_page():
    return send_file('strategy_builder.html')

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
    threading.Thread(target=_gex_refresh_loop, args=(300,),
                     daemon=True, name="GEXRefresher").start()
    print("[DataServer] GEX refresh thread started (interval=300s).")

    app.run(port=PORT, debug=False, use_reloader=False)

