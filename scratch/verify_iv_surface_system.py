"""
scratch/verify_iv_surface_system.py
Tests:
1. IvSurfaceEngine snapshot pushing, expected move vs day open consumption calculations,
   rewind logic, delta smile overlay, and 3D surface meshgrid.
2. DataServer Flask test client endpoints: /api/iv-surface and /api/iv_surface.
3. Static files integrity (iv_surface.css, iv_surface_terminal.js, dashboard_core.js).
"""
import sys
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        if hasattr(sys.stdout, 'reconfigure'):
            getattr(sys.stdout, 'reconfigure')(encoding='utf-8')
    except (AttributeError, Exception):
        pass
import os
import time
import json
import pandas as pd
import numpy as np

# Ensure project root is in path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

print(f"Testing F-Intel IV Surface & Exhaustion System from: {ROOT_DIR}")

# ─────────────────────────────────────────────────────────────────────────────
# 1. TEST IvSurfaceEngine
# ─────────────────────────────────────────────────────────────────────────────
print("\n--- 1. Testing calculations/IvSurfaceEngine.py ---")
from calculations.IvSurfaceEngine import IvSurfaceEngine

engine = IvSurfaceEngine(max_history=100)

# Build synthetic option chain around spot 23200
spot = 23200.0
day_open = 23150.0 # moved +50 pts from open
day_high = 23260.0
day_low  = 23120.0

strikes = [22800 + i * 50 for i in range(17)] # 22800 to 23600
chain_rows = []
for k in strikes:
    # 2 expiries: DTE 2 (weekly) and DTE 9 (next weekly)
    for dte in [2.0, 9.0]:
        # Put smile: higher IV for OTM puts (k < spot)
        # Call smile: slight skew
        moneyness = (k - spot) / spot
        base_iv = 12.5 + (0.5 if dte > 5 else 0.0)
        ce_iv = base_iv + 15.0 * (moneyness ** 2) - 2.0 * moneyness
        pe_iv = base_iv + 25.0 * (moneyness ** 2) - 8.0 * moneyness
        
        chain_rows.append({
            'strike': float(k), 'type': 'CE', 'iv': float(ce_iv), 'dte': dte,
            'oi': 100000, 'volume': 50000, 'price': 100.0
        })
        chain_rows.append({
            'strike': float(k), 'type': 'PE', 'iv': float(pe_iv), 'dte': dte,
            'oi': 120000, 'volume': 60000, 'price': 100.0
        })

df_chain = pd.DataFrame(chain_rows)

# Push Snapshot 1 (simulate morning open)
snap1 = engine.push_snapshot(
    df=df_chain,
    spot=23150.0,
    day_open=23150.0,
    day_high=23160.0,
    day_low=23140.0
)
assert snap1 is not None, "Failed to create snapshot 1"
print("Snapshot 1 (Open):", {
    'spot': snap1.get('spot'),
    'day_open': snap1.get('day_open'),
    'atm_iv': snap1.get('atm_iv'),
    'expected_move': snap1.get('expected_move')
})

# Simulate 1 second time passage
time.sleep(0.1)

# Push Snapshot 2 (midday move up +50 pts)
snap2 = engine.push_snapshot(
    df=df_chain,
    spot=23200.0,
    day_open=day_open,
    day_high=day_high,
    day_low=day_low
)
assert snap2 is not None, "Failed to create snapshot 2"

em = snap2.get('expected_move', {})
ex = snap2.get('exhaustion', {})
print("\nSnapshot 2 (Midday +50 pts):")
print(f"  Spot: {snap2.get('spot')} vs Day Open: {snap2.get('day_open')}")
print(f"  Expected 1-Day Move: ±{em.get('expected_move_pts'):.1f} pts (±{em.get('expected_move_pct'):.2f}%)")
print(f"  Displacement Since Open: {em.get('spot_vs_open_pts'):+.1f} pts ({em.get('spot_vs_open_pct'):+.2f}%)")
print(f"  Intraday Range: {em.get('high_low_range_pts'):.1f} pts (H: {em.get('day_high')} / L: {em.get('day_low')})")
print(f"  1-Sigma Boundaries: Upper={em.get('upper_1sigma'):.0f}, Lower={em.get('lower_1sigma'):.0f}")
print(f"  Net Displacement Consumption: {ex.get('net_consumption_pct'):.1f}%")
print(f"  Total Range Consumption: {ex.get('range_consumption_pct'):.1f}%")
print(f"  Status Tag: {ex.get('status')}")

assert em.get('expected_move_pts') > 0, "Expected move must be positive"
assert ex.get('net_consumption_pct') >= 0, "Net consumption must be >= 0"
assert any(k in ex.get('status', '') for k in ['CONSOLIDATION', 'NORMAL', 'EXHAUSTION', 'EXPANSION']), f"Unexpected status tag: {ex.get('status')}"

# Test Rewind & Comparison
print("\nTesting Rewind & Baseline Smile / Surface...")
rewind_res = engine.get_surface_data(rewind_ts=snap1['ts'], baseline_mode='open')
assert rewind_res.get('ok') is True, "Rewind failed"
assert 'smile_2d' in rewind_res, "Missing smile_2d"
assert 'surface_3d' in rewind_res, "Missing surface_3d"
assert 'total_iv_shifts' in rewind_res, "Missing total_iv_shifts"
assert 'history_index' in rewind_res, "Missing history_index"
print("  Smile strikes count:", len(rewind_res['smile_2d']['active_strikes']))
print("  3D Surface strikes:", len(rewind_res['surface_3d']['strikes']), "dtes:", len(rewind_res['surface_3d']['dtes']))
print("  History Index count:", len(rewind_res['history_index']))
print("  Total IV Shifts:", rewind_res['total_iv_shifts'])
print("✓ IvSurfaceEngine passed all tests!")

# ─────────────────────────────────────────────────────────────────────────────
# 2. TEST DataServer Endpoints
# ─────────────────────────────────────────────────────────────────────────────
print("\n--- 2. Testing DataServer.py Endpoints ---")
import DataServer

# Wire the engine into DataServer's global if not yet populated
DataServer._iv_surface_engine = engine

with DataServer.app.test_client() as client:
    # Test GET /api/iv-surface
    res = client.get('/api/iv-surface')
    print("GET /api/iv-surface status code:", res.status_code)
    assert res.status_code == 200, f"Expected 200, got {res.status_code}"
    data = json.loads(res.data.decode('utf-8'))
    assert data.get('ok') is True, "API returned ok: False"
    assert 'expected_move' in data, "Missing expected_move in API response"
    assert 'exhaustion' in data, "Missing exhaustion in API response"
    print("  API Response expected move:", data['expected_move']['expected_move_pts'], "pts")
    print("  API Response consumption:", data['exhaustion']['net_consumption_pct'], "%")

    # Test GET /api/iv_surface alias
    res_alias = client.get('/api/iv_surface')
    assert res_alias.status_code == 200, f"Alias /api/iv_surface failed with {res_alias.status_code}"

    # Test GET /api/iv-surface with rewind_ts and baseline
    res_rewind = client.get(f'/api/iv-surface?rewind_ts={snap1["ts"]}&baseline=open')
    assert res_rewind.status_code == 200, "Rewind API call failed"
    rewind_json = json.loads(res_rewind.data.decode('utf-8'))
    assert rewind_json.get('ok') is True, "Rewind JSON returned ok: False"
    print("  Rewind API call success!")

print("✓ DataServer API endpoints verified!")

# ─────────────────────────────────────────────────────────────────────────────
# 3. TEST Static Files Integrity
# ─────────────────────────────────────────────────────────────────────────────
print("\n--- 3. Testing Static Assets & Preservations ---")

css_path = os.path.join(ROOT_DIR, "static", "css", "iv_surface.css")
assert os.path.exists(css_path), "Missing static/css/iv_surface.css"
assert os.path.getsize(css_path) > 500, "iv_surface.css is too small"
print("✓ static/css/iv_surface.css exists and is populated.")

js_path = os.path.join(ROOT_DIR, "static", "js", "iv_surface_terminal.js")
assert os.path.exists(js_path), "Missing static/js/iv_surface_terminal.js"
assert os.path.getsize(js_path) > 1000, "iv_surface_terminal.js is too small"
print("✓ static/js/iv_surface_terminal.js exists and is populated.")

core_js_path = os.path.join(ROOT_DIR, "static", "js", "dashboard_core.js")
with open(core_js_path, "r", encoding="utf-8") as f:
    core_content = f.read()

assert "iv-surface-card" in core_content, "dashboard_core.js missing iv-surface-card preservation"
assert "iv_surface_update" in core_content, "dashboard_core.js missing iv_surface_update WebSocket dispatch"
assert "IvSurfaceTerminal" in core_content, "dashboard_core.js missing window.IvSurfaceTerminal hook"
print("✓ dashboard_core.js correctly preserves #iv-surface-card and dispatches WebSocket updates.")

va_path = os.path.join(ROOT_DIR, "VolatilityAnalyzer.py")
with open(va_path, "r", encoding="utf-8") as f:
    va_content = f.read()

assert "iv-surface-card" in va_content, "VolatilityAnalyzer.py missing iv-surface-card"
assert "iv_surface.css" in va_content, "VolatilityAnalyzer.py missing iv_surface.css"
assert "iv_surface_terminal.js" in va_content, "VolatilityAnalyzer.py missing iv_surface_terminal.js"
print("✓ VolatilityAnalyzer.py correctly incorporates iv-surface-card, iv_surface.css, and iv_surface_terminal.js.")

print("\n========================================================")
print("ALL VERIFICATIONS PASSED SUCCESSFULLY!")
print("========================================================")
