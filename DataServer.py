import sys

# Ensure stdout can print Unicode (like ✓) on Windows consoles
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
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
from FyersAuth import FyersAuthenticator
from fyers_apiv3.FyersWebsocket import data_ws
from OptionAnalytics import OptionAnalytics

# --- CONFIG ---
APP_ID = "QUTT4YYMIG-100"
SECRET_ID = "ZG0WN2NL1B"
REDIRECT_URI = "http://127.0.0.1:3000/callback"
SYMBOL = "NSE:NIFTY50-INDEX"
PORT = 8082
CHAIN_REFRESH_INTERVAL = 60 # Seconds (Option chain rate limits are strict, 1 per min)

app = Flask(__name__)
CORS(app)
sock = Sock(app)

class DataHub:
    def __init__(self):
        self.fyers = None
        self.access_token = None
        self.latest_data = {
            "spot": 0,
            "chain": {},
            "last_update": None,
            "status": "Initializing",
            "tick_count": 0
        }
        self.clients = set()
        self.lock = threading.Lock()
        self._stop_event = threading.Event()

    def authenticate(self):
        print("DataHub: Authenticating...")
        auth = FyersAuthenticator(APP_ID, SECRET_ID, REDIRECT_URI)
        self.fyers = auth.get_fyers_instance()
        if self.fyers:
            self.access_token = open("access_token.txt", "r").read().strip()
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
                    # REST Fallback for Spot Price if WebSocket is silent
                    if self.latest_data["spot"] == 0:
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
        self._T = max(0.0, float(T))

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

@app.route('/api/track_strategy', methods=['POST'])
def api_track_strategy():
    """Save a strategy to the DB for live tracking."""
    try:
        from StrategyManager import StrategyManager
        data = request.get_json(force=True)
        mgr  = StrategyManager()
        sid  = mgr.track_strategy(data, entry_spot=data.get('entry_spot', 0))
        return jsonify({'ok': True, 'id': sid})
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)}), 500

@app.route('/api/close_strategy', methods=['POST'])
def api_close_strategy():
    """Mark a tracked strategy as closed and record P&L."""
    try:
        from StrategyManager import StrategyManager
        data       = request.get_json(force=True)
        strategy_id = data.get('id')
        exit_prem  = float(data.get('exit_premium', 0))
        exit_spot  = float(data.get('exit_spot', 0))
        mgr  = StrategyManager()
        pnl  = mgr.close_strategy(strategy_id, exit_prem, exit_spot)
        return jsonify({'ok': True, 'pnl': pnl})
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)}), 500

@app.route('/api/delete_strategy', methods=['POST'])
def api_delete_strategy():
    """Delete a tracked strategy (alias used by the inline strategy tab)."""
    try:
        from StrategyManager import StrategyManager
        data = request.get_json(force=True)
        sid  = data.get('id')
        mgr  = StrategyManager()
        if hasattr(mgr, 'delete_strategy'):
            mgr.delete_strategy(sid)
        else:
            mgr.close_strategy(sid, 0, 0)
        return jsonify({'ok': True})
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)}), 500

@app.route('/api/strategy_pnl_at', methods=['GET'])
def api_strategy_pnl_at():
    """Return historical P&L HTML for a given intraday time (P&L scrubber)."""
    try:
        from StrategyManager import StrategyManager
        from TickDatabase import IntradayTickDB
        target_time = request.args.get('time', '')
        if not target_time:
            from datetime import datetime as _dt
            target_time = _dt.now().strftime('%H:%M')
        tick_db = IntradayTickDB()
        hist_chain = tick_db.get_chain_at_time('NSE:NIFTY50-INDEX', target_time)
        mgr = StrategyManager()
        tracked = mgr.get_all_active_strategies()
        if not tracked:
            html = '<div style="color:#666;font-size:12px;text-align:center;padding:20px;">No strategies tracked.</div>'
        else:
            rows = []
            chain_prices = {}
            if not hist_chain.empty:
                for _, r in hist_chain.iterrows():
                    chain_prices[(r['type'], r['strike'])] = r['price']
            for t in tracked:
                entry_val = t.get('premium', 0)
                current_val = sum(
                    chain_prices.get((lg['type'], lg['strike']), lg['price']) * (1 if lg['action'] == 'SELL' else -1)
                    for lg in t.get('legs', [])
                )
                pnl = entry_val - current_val if t.get('type') == 'CREDIT' else current_val - abs(entry_val)
                c = '#66bb6a' if pnl > 0 else '#ff4444'
                rows.append(f'<div style="padding:6px 0;border-bottom:1px solid #333;">'
                            f'<b style="color:#4fc3f7">{t["name"]}</b> '
                            f'<span style="color:{c};font-weight:700">P&L: &#8377;{pnl:+.2f}</span></div>')
            html = ''.join(rows)
        return html, 200, {'Content-Type': 'text/html'}
    except Exception as e:
        return f'<div style="color:#ff4444">Error: {e}</div>', 500, {'Content-Type': 'text/html'}



@app.route('/api/strategies', methods=['GET'])
def api_strategies():
    """Return all active tracked strategies."""
    try:
        from StrategyManager import StrategyManager
        mgr = StrategyManager()
        active = mgr.get_all_active_strategies()
        closed = mgr.get_closed_strategies(limit=10)
        summary = mgr.get_summary()
        return jsonify({'ok': True, 'active': active, 'closed': closed, 'summary': summary})
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)}), 500

@app.route('/api/backtest', methods=['POST'])
def api_backtest():
    """
    Launch an async backtest. Returns immediately with a job token.
    Poll /api/backtest_status?type=<strategy_type> for completion.
    """
    try:
        data          = request.get_json(force=True)
        strategy_type = data.get('strategy_type', 'SHORT_STRADDLE')
        days          = int(data.get('days', 365))
        stop_loss     = float(data.get('stop_loss_mult', 2.0))
        atm_iv        = float(data.get('atm_iv_pct', 14.0))
        wing_width    = int(data.get('wing_width', 150))

        if _bt_running.get(strategy_type):
            return jsonify({'ok': True, 'status': 'RUNNING', 'type': strategy_type})

        def _run():
            from StrategyBacktester import OptionStrategyBacktester
            from StrategyManager import StrategyManager
            _bt_running[strategy_type] = True
            try:
                bt = OptionStrategyBacktester()
                report = bt.run(strategy_type, days=days, stop_loss_mult=stop_loss,
                                atm_iv_pct=atm_iv, wing_width=wing_width)
                _bt_cache[strategy_type] = report
                # Persist to DB
                try:
                    mgr = StrategyManager()
                    mgr.save_backtest(strategy_type, report.to_json(), report.stats,
                                      report.start_date, report.end_date)
                except Exception:
                    pass
            finally:
                _bt_running[strategy_type] = False

        threading.Thread(target=_run, daemon=True).start()
        return jsonify({'ok': True, 'status': 'STARTED', 'type': strategy_type})

    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)}), 500

@app.route('/api/backtest_status', methods=['GET'])
def api_backtest_status():
    """Return cached backtest HTML result for a strategy type."""
    strategy_type = request.args.get('type', 'SHORT_STRADDLE')
    if _bt_running.get(strategy_type):
        return jsonify({'ok': True, 'status': 'RUNNING'})
    report = _bt_cache.get(strategy_type)
    if not report:
        return jsonify({'ok': True, 'status': 'NONE'})
    return jsonify({
        'ok': True,
        'status': 'DONE',
        'html': report.to_html(),
        'stats': report.stats,
    })


# ─────────────────────────────────────────────────────────────────────────────
# Strategy Wizard API  (Enhancement 3)
# ─────────────────────────────────────────────────────────────────────────────

@app.route('/api/wizard', methods=['POST'])
def api_wizard():
    """
    POST /api/wizard
    View-driven strategy recommendations.

    Body JSON:
      view       : BULLISH | BEARISH | NEUTRAL | VOLATILE | NON-VOLATILE
      risk       : CONSERVATIVE | MODERATE | AGGRESSIVE
      capital    : float (INR)
      conviction : LOW | MODERATE | HIGH
    """
    try:
        from StrategyWizard import StrategyWizard
        from StrategyEngine import PayoffEngine, OptionAnalytics as _OA
        import pandas as pd
        import numpy as np

        body = request.get_json(force=True) or {}
        view       = body.get('view', 'NEUTRAL').upper()
        risk       = body.get('risk', 'MODERATE').upper()
        capital    = float(body.get('capital', 100000))
        conviction = body.get('conviction', 'MODERATE').upper()

        # Pull live data from DataHub
        spot  = hub.latest_data.get('spot', 0.0)
        chain = hub.latest_data.get('chain', {})
        if spot <= 0:
            return jsonify({'ok': False, 'error': 'Spot price not available'}), 503

        # Parse chain to DataFrame
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

        # Derive context scalars from chain
        atm_iv = 15.0
        skew_ratio = 1.0
        T = hub_cache.get_T()
        if not df_chain.empty:
            dist = abs(df_chain['strike'] - spot)
            atm_row = df_chain.loc[dist.idxmin()]
            atm_iv  = float(atm_row.get('iv', 15.0) or 15.0)
            # Skew ratio: median put IV / ATM IV
            pe_ivs = df_chain[df_chain['type'] == 'PE']['iv'].replace(0, np.nan).dropna()
            skew_ratio = float(pe_ivs.median() / atm_iv) if atm_iv > 0 and not pe_ivs.empty else 1.0

        expiry = chain.get('expiry', '')
        if not expiry:
            # Guess near expiry from chain row data
            for o in chain.get('optionsChain', []):
                e = o.get('expiry', '')
                if e:
                    expiry = e
                    break

        market_context = {
            'T':              T,
            'iv':             atm_iv,
            'atm_iv':         atm_iv,
            'vrp':            0.0,
            'regime':         'COMPRESSION',
            'call_wall':      0,
            'put_wall':       0,
            'em':             spot * (atm_iv / 100) * np.sqrt(T),
            'oi_pressure':    'NEUTRAL',
            'DTE':            max(1, int(T * 365)),
            'skew_ratio':     skew_ratio,
            'explosion_score': 0,
            'term_spread':    0,
            'heston_params':  hub_cache.get_heston_params(),
        }

        wizard = StrategyWizard(spot, df_chain, market_context, expiry)
        recs   = wizard.recommend(view, risk, capital, conviction)

        results = []
        for rec in recs:
            strat = rec['strategy']
            d     = strat.to_dict()

            # POP
            sigma = atm_iv / 100.0
            pop   = strat.pop(spot, T, sigma) * 100

            # Risk dials
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

        return jsonify({'ok': True, 'strategies': results})

    except Exception as e:
        import traceback
        return jsonify({'ok': False, 'error': str(e),
                        'trace': traceback.format_exc()}), 500


# ─────────────────────────────────────────────────────────────────────────────
# Scenario Engine API  (Enhancement 3B)
# ─────────────────────────────────────────────────────────────────────────────

@app.route('/api/scenario', methods=['POST'])
def api_scenario():
    """
    POST /api/scenario
    Compute P&L scenario grid and spot ladder for a given strategy.

    Body JSON:
      strategy        : dict as returned by /api/wizard (legs, net_premium, etc.)
      spot_range_pct  : int (default 10 → ±10%)
      n_spot          : int (default 21)
    """
    try:
        from ScenarioEngine import ScenarioEngine
        from StrategyEngine import OptionLeg, Strategy
        import numpy as np

        body         = request.get_json(force=True) or {}
        strat_dict   = body.get('strategy', {})
        range_pct    = float(body.get('spot_range_pct', 10)) / 100.0
        n_spot       = int(body.get('n_spot', 21))

        # Reconstruct Strategy from JSON legs
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

        # Market state
        spot  = hub.latest_data.get('spot', 0.0)
        T     = hub_cache.get_T()
        sigma = float(strat_dict.get('net_premium', 0)) / max(spot, 1) if spot > 0 else 0.15
        # Use ATM IV from chain as sigma if available
        chain = hub.latest_data.get('chain', {})
        if chain.get('optionsChain'):
            import pandas as pd
            df_c = pd.DataFrame(chain.get('optionsChain', []))
            if not df_c.empty and 'iv' in df_c.columns and 'strike_price' in df_c.columns:
                df_c['dist'] = abs(df_c['strike_price'].astype(float) - spot)
                atm_iv = float(df_c.loc[df_c['dist'].idxmin(), 'iv'] or 15.0)
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


# ─────────────────────────────────────────────────────────────────────────────
# GEX / Signals / Regime / Dealer API
# ─────────────────────────────────────────────────────────────────────────────

_gex_snapshot: dict = {
    "score": 0, "regime": "UNKNOWN", "net_gex": 0,
    "strikes": [], "direction": "NEUTRAL", "last_update": None,
    # New fields from calculations/GexEngine
    "zero_gamma_level": 0, "dealer_long_pct": 0, "dealer_short_pct": 0,
    "spot_gamma": 0, "forward_gex": 0,
}
_gex_lock = threading.Lock()

# In-memory dealer snapshot (refreshed together with GEX)
_dealer_snapshot: dict = {
    "net_dex": 0, "net_gex_shares": 0, "net_vanna": 0, "net_charm": 0,
    "hedge_spot_up": 0, "hedge_iv_up": 0, "hedge_1day": 0,
    "strike_dex": [], "strike_gex": [], "last_update": None,
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
        "oi":     float(r.get("oi", 0) or 0),
        "volume": float(r.get("volume", 0) or 0),
        "iv":     float(r.get("iv", 0) or 0),
        "price":  float(r.get("ltp", 0) or 0),
        "dte":    float(r.get("dte", 1) or 1),
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
                    gex_eng = GexEngine(lot_size=75, positioning_model='standard')
                    gex_res = gex_eng.calculate_gex(df, spot)

                    profile   = gex_res.get("profile")
                    net_gex   = gex_res.get("net_gex", 0)
                    direction = "POSITIVE" if net_gex > 0 else "NEGATIVE"

                    strike_list = []
                    if profile is not None and not profile.empty:
                        for strike, gex_val in profile.items():
                            strike_list.append({"strike": float(strike), "gex": float(gex_val)})

                    # Try GammaExplosionModel for regime label + composite score (non-fatal)
                    regime = "POSITIVE GEX" if net_gex > 0 else "NEGATIVE GEX"
                    score  = round(abs(net_gex) / 1e9, 2)
                    try:
                        from GammaExplosionModel import GammaExplosionModel
                        gm = GammaExplosionModel(fyers_instance=hub.fyers)
                        gm.spot_price = spot
                        gm_df = gm.parse_chain(chain) if hasattr(gm, "parse_chain") else pd.DataFrame()
                        if not gm_df.empty:
                            gm_res = gm.run_analysis(gm_df)
                            regime = gm_res.get("regime", regime)
                            score  = gm_res.get("composite_score", score)
                    except Exception:
                        pass

                    with _gex_lock:
                        _gex_snapshot.update({
                            "score":            score,
                            "regime":           regime,
                            "net_gex":          float(net_gex),
                            "direction":        direction,
                            "strikes":          strike_list,
                            "zero_gamma_level": gex_res.get("zero_gamma_level", 0),
                            "dealer_long_pct":  round(gex_res.get("dealer_long_pct", 0), 1),
                            "dealer_short_pct": round(gex_res.get("dealer_short_pct", 0), 1),
                            "spot_gamma":       gex_res.get("spot_gamma", 0),
                            "forward_gex":      gex_res.get("forward_gex", 0),
                            "last_update":      datetime.now().strftime("%H:%M:%S"),
                        })
                    print(f"[GEX] net={net_gex:.0f}, flip={gex_res.get('zero_gamma_level',0):.0f}, regime={regime}")

                    # ── Dealer Inventory via DealerPositionEngine ──────────
                    try:
                        dep   = DealerPositionEngine(lot_size=75)
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
                atm_iv = float(row.get("iv", 0) or 0)

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

@app.route('/bt_run', methods=['POST'])
def bt_run():
    """Legacy backtest trigger used by unified_dashboard.html."""
    try:
        data          = request.get_json(force=True) or {}
        strategy_type = data.get('type', 'SHORT_STRADDLE').upper()
        days          = int(data.get('days', 365))
        stop_loss     = float(data.get('sl', 2.0))

        if _bt_running.get(strategy_type):
            return 'RUNNING', 200

        def _run():
            from StrategyBacktester import OptionStrategyBacktester
            _bt_running[strategy_type] = True
            try:
                bt     = OptionStrategyBacktester()
                report = bt.run(strategy_type, days=days, stop_loss_mult=stop_loss)
                _bt_cache[strategy_type] = report
                # Also persist static HTML file so /bt_<type>.html can serve it
                html_path = f'bt_{strategy_type}.html'
                with open(html_path, 'w', encoding='utf-8') as _f:
                    _f.write(report.to_html())
            except Exception as _e:
                print(f'[bt_run] backtest error: {_e}')
            finally:
                _bt_running[strategy_type] = False

        threading.Thread(target=_run, daemon=True).start()
        return 'STARTED', 200
    except Exception as e:
        return str(e), 500


@app.route('/bt_<path:fname>.html', methods=['GET'])
def bt_serve_html(fname):
    """Legacy backtest result poller: serves the pre-rendered HTML file."""
    import os
    strategy_type = fname.upper()
    # If a cached report exists in memory, serve that
    report = _bt_cache.get(strategy_type)
    if report:
        try:
            return report.to_html(), 200, {'Content-Type': 'text/html; charset=utf-8'}
        except Exception:
            pass
    # Otherwise try the static file on disk
    html_path = f'bt_{strategy_type}.html'
    if os.path.exists(html_path):
        return send_file(html_path)
    # Still running or not started yet
    if _bt_running.get(strategy_type):
        return '', 202   # 202 = not ready, dashboard keeps polling
    return '', 404


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

