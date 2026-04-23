import os
import time
import json
import threading
from flask import Flask, jsonify
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
# GEX / Signals / Regime  API  (Institutional UI — Phase 0)
# ─────────────────────────────────────────────────────────────────────────────

# In-memory GEX snapshot refreshed by the background thread below
_gex_snapshot: dict = {
    "score": 0, "regime": "UNKNOWN", "net_gex": 0,
    "strikes": [], "direction": "NEUTRAL", "last_update": None,
}
_gex_lock = threading.Lock()


def _gex_refresh_loop(interval: int = 300):
    """Background thread: refresh GEX snapshot every `interval` seconds."""
    while True:
        try:
            if hub.fyers and hub.latest_data.get("chain"):
                from GammaExplosionModel import GammaExplosionModel
                import pandas as pd
                gm = GammaExplosionModel(fyers_instance=hub.fyers)
                gm.spot_price = hub.latest_data.get("spot", 0)
                chain = hub.latest_data["chain"]
                df = gm.parse_chain(chain) if hasattr(gm, "parse_chain") else pd.DataFrame()
                if df.empty:
                    # Fallback: parse optionsChain manually
                    rows = chain.get("optionsChain", [])
                    if rows:
                        df = pd.DataFrame([{
                            "strike":     float(r.get("strike_price", 0)),
                            "type":       r.get("option_type", "CE"),
                            "oi":         int(r.get("oi", 0) or 0),
                            "iv":         float(r.get("iv", 0) or 0),
                            "price":      float(r.get("ltp", 0) or 0),
                        } for r in rows])
                if not df.empty and gm.spot_price > 0:
                    result = gm.run_analysis(df)      # returns full dict
                    snap = {
                        "score":       result.get("composite_score", 0),
                        "regime":      result.get("regime", "NORMAL"),
                        "net_gex":     result.get("net_gex", 0),
                        "direction":   result.get("explosion_direction", "NEUTRAL"),
                        "strikes":     result.get("gex_by_strike", []),
                        "last_update": datetime.now().strftime("%H:%M:%S"),
                    }
                    with _gex_lock:
                        _gex_snapshot.update(snap)
                    print(f"[GEX] Refreshed — score={snap['score']}, regime={snap['regime']}")
        except Exception as _e:
            print(f"[GEX] Refresh error (non-fatal): {_e}")
        time.sleep(interval)


@app.route('/api/gex', methods=['GET'])
def api_gex():
    """
    GET /api/gex
    Returns the cached GammaExplosionModel snapshot.
    Refreshed every 5 minutes by _gex_refresh_loop().

    Response shape:
      { ok, score, regime, net_gex, direction, strikes:[{strike,gex,label}], last_update }
    """
    with _gex_lock:
        snap = dict(_gex_snapshot)
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

