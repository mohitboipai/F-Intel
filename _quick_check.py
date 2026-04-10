"""
_quick_check.py - F-Intel system health check
All output written to quick_check.log to avoid Windows terminal encoding issues.
"""
import sys, os, socket, json, time
sys.path.insert(0, '.')

LOG_FILE = "quick_check.log"
_lines = []

def log(s=""):
    _lines.append(s)
    print(s)  # also show in terminal best-effort

def flush():
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(_lines) + "\n")

results = []

def check(label, fn):
    try:
        msg = fn()
        tag = "[PASS]"
    except Exception as e:
        msg = str(e)
        tag = "[FAIL]"
    log(f"  {tag}  {label}: {msg}")
    results.append((tag, label, msg))
    flush()

log("=" * 60)
log("  F-INTEL QUICK SYSTEM CHECK")
log(f"  {time.strftime('%Y-%m-%d %H:%M:%S')}")
log("=" * 60)

# ── 1. TOKEN ─────────────────────────────────────────────────────
log("\n[1] TOKEN")
def _token():
    tok = open('access_token.txt').read().strip()
    assert len(tok) > 50, "Token suspiciously short"
    age_hrs = (time.time() - os.path.getmtime('access_token.txt')) / 3600
    if age_hrs > 20:
        raise Exception(f"Token is {age_hrs:.1f}h old -- Fyers tokens expire at midnight; re-run FyersAuth")
    return f"OK, age={age_hrs:.1f}h"
check("access_token.txt age", _token)

# ── 2. FYERS API ──────────────────────────────────────────────────
log("\n[2] FYERS API")
fyers_instance = None
def _profile():
    global fyers_instance
    from fyers_apiv3 import fyersModel
    tok = open('access_token.txt').read().strip()
    fyers_instance = fyersModel.FyersModel(client_id='QUTT4YYMIG-100', token=tok, log_path='.')
    r = fyers_instance.get_profile()
    if r.get('s') != 'ok':
        raise Exception(f"API error code={r.get('code')} msg={r.get('message')}")
    name = r['data'].get('name', r['data'].get('fy_id', '?'))
    return f"Authenticated as '{name}'"
check("fyers.get_profile()", _profile)

# ── 3. SPOT PRICE ─────────────────────────────────────────────────
log("\n[3] LIVE SPOT")
def _spot():
    if fyers_instance is None:
        raise Exception("Fyers not initialised (see section 2)")
    r = fyers_instance.quotes(data={"symbols": "NSE:NIFTY50-INDEX"})
    if r.get('s') != 'ok':
        raise Exception(f"quotes error: {r}")
    lp = r['d'][0]['v']['lp']
    return f"NIFTY lp={lp}"
check("NIFTY spot quote", _spot)

# ── 4. OPTION CHAIN ───────────────────────────────────────────────
log("\n[4] OPTION CHAIN")
def _chain():
    if fyers_instance is None:
        raise Exception("Fyers not initialised")
    r = fyers_instance.optionchain(data={"symbol": "NSE:NIFTY50-INDEX", "strikecount": 5, "timestamp": ""})
    if r.get('s') != 'ok':
        raise Exception(f"chain error: {r.get('message', r)}")
    rows = r.get('data', {}).get('optionsChain', [])
    spot = r.get('data', {}).get('lp', 0)
    return f"spot={spot}, {len(rows)} option rows returned"
check("optionchain (5 strikes)", _chain)

# ── 5. PORTS ──────────────────────────────────────────────────────
log("\n[5] PORTS")
def _port(p):
    def _fn():
        s = socket.socket(); s.settimeout(1)
        try:
            s.connect(('127.0.0.1', p))
            s.close()
            return "OPEN"
        except:
            return "CLOSED -- process not running"
    return _fn
check("port 8082 (DataServer WS/HTTP)", _port(8082))
check("port 8081 (VA local API)", _port(8081))

# ── 6. DATASERVER HTTP ────────────────────────────────────────────
log("\n[6] DATASERVER HTTP")
def _ds_status():
    import urllib.request
    # DataServer has no /status — use /get_data as the liveness probe
    try:
        r = urllib.request.urlopen('http://127.0.0.1:8082/get_data', timeout=2)
        d = json.loads(r.read())
        return f"HTTP {r.status}: status={d.get('status','?')} spot={d.get('spot',0)} ticks={d.get('tick_count',0)}"
    except Exception as e:
        return f"Not reachable ({e.__class__.__name__}) -- is DataServer.py running?"
check("/get_data liveness (DataServer)", _ds_status)

def _ds_data():
    import urllib.request
    r = urllib.request.urlopen('http://127.0.0.1:8082/get_data', timeout=2)
    d = json.loads(r.read())
    spot = d.get('spot', 0)
    ts = d.get('timestamp', 'no-ts')
    return f"spot={spot}, ts={ts}"
check("/get_data live snapshot", _ds_data)

# ── 7. DATABASE FILES ─────────────────────────────────────────────
log("\n[7] FILES")
def _dbf(fname):
    def _fn():
        sz = os.path.getsize(fname)
        age_h = (time.time() - os.path.getmtime(fname)) / 3600
        return f"{sz//1024} KB, last-write {age_h:.1f}h ago"
    return _fn
check("strategies.db", _dbf("strategies.db"))
check("intraday_chain.db", _dbf("intraday_chain.db"))
check("market_memory.json", _dbf("market_memory.json"))
check("alerts.jsonl", _dbf("alerts.jsonl"))

# ── 8. STRATEGY ENGINE ────────────────────────────────────────────
log("\n[8] STRATEGY ENGINE")
def _se():
    import numpy as np, pandas as pd
    from StrategyEngine import SmartStrategyGenerator, build_strategy_card_html
    spot = 24000.0
    rows = [{'strike': float(s), 'type': t, 'price': max(2.0, abs(spot-s)+10), 'iv':15.0}
            for s in range(22000, 26500, 50) for t in ('CE','PE')]
    df = pd.DataFrame(rows)
    ctx = {'T':7/365,'iv':15.0,'atm_iv':15.0,'vrp':2.0,'regime':'COMPRESSION',
           'call_wall':24500,'put_wall':23500,'em':250.0,'oi_pressure':'NEUTRAL','DTE':7}
    gen = SmartStrategyGenerator(spot, df, ctx, '2026-04-17')
    strats = gen.generate()
    top = strats[0]
    # Make sure build_strategy_card_html is importable (Issue 1 fix)
    assert callable(build_strategy_card_html)
    return f"{len(strats)} strategies, top={top.name} score={top.score}"
check("SmartStrategyGenerator + build_strategy_card_html", _se)

# ── 9. ALERT DISPATCHER ───────────────────────────────────────────
log("\n[9] ALERT DISPATCHER")
def _alert():
    from AlertDispatcher import fire
    fire('QuickCheck', 'INFO', 'System health check passed', '')
    lines = open('alerts.jsonl', encoding='utf-8').readlines()
    last = json.loads(lines[-1])
    return f"Last event: source={last['source']} level={last['level']}"
check("AlertDispatcher.fire() -> alerts.jsonl", _alert)

# ── 10. RECENT LOG ERRORS ─────────────────────────────────────────
log("\n[10] RECENT LOG ERRORS (last 100 fyersApi.log lines)")
def _log_errors():
    lines = open('fyersApi.log', encoding='utf-8', errors='replace').readlines()
    errors = [l for l in lines[-100:] if '"level":"ERROR"' in l]
    if not errors:
        return "No errors in last 100 lines -- clean"
    try:
        last = json.loads(errors[-1])
        msg = last.get('message', '')
        ts = last.get('timestamp', '')
        return f"{len(errors)} errors; last at {ts}: {msg[:100]}"
    except:
        return f"{len(errors)} errors; last: {errors[-1][:120]}"
check("fyersApi.log error scan", _log_errors)

# ── SUMMARY ───────────────────────────────────────────────────────
fails  = [r for r in results if r[0] == "[FAIL]"]
warns  = [r for r in results if r[0] == "[WARN]"]
passes = [r for r in results if r[0] == "[PASS]"]

log("\n" + "=" * 60)
log(f"  RESULT: {len(passes)} passed | {len(warns)} warnings | {len(fails)} failed")
log("=" * 60)
if fails:
    log("\n  ACTION REQUIRED:")
    for _, label, msg in fails:
        log(f"    FAIL  {label}")
        log(f"          {msg}")
if not fails:
    log("\n  All checks passed. Run .\\start_analyzer.ps1 to launch the system.")
log("")
flush()
print(f"\n  Full results written to: {os.path.abspath(LOG_FILE)}")
