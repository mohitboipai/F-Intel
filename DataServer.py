from sqlalchemy.sql.elements import True_
from aiohttp.web import route
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

# --- CONFIG ---
APP_ID = "QUTT4YYMIG-100"
SECRET_ID = "ZG0WN2NL1B"
REDIRECT_URI = "http://127.0.0.1:3000/callback"
SYMBOL = "NSE:NIFTY50-INDEX"
PORT = 8082
CHAIN_REFRESH_INTERVAL = 15 # Seconds (Option chain is heavy, poll slower)

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
        """Send data to all connected WebSocket clients."""
        msg = json.dumps(data)
        with self.lock:
            # Create a copy of the clients set to avoid modification during iteration
            for client in list(self.clients):
                try:
                    client.send(msg)
                except:
                    self.clients.remove(client)

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

if __name__ == "__main__":
    hub.start()
    print(f"DataHub Real-Time server starting on port {PORT}...")
    app.run(port=PORT, debug=False, use_reloader=False)
