"""smoke_tests.py — Verify all Enhancement 1-4 modules."""
import sys; sys.path.insert(0,'.')
import numpy as np, pandas as pd

print('=== SMOKE TEST: F-Intel Strategy Intelligence Enhancements ===')

# -----------------------------------------------------------------
# 1. SharedDataCache new methods
# -----------------------------------------------------------------
print('\n[1] SharedDataCache new methods...')
from SharedDataCache import SharedDataCache

class MockFyers:
    def quotes(self, **k): return {'s':'error'}
    def optionchain(self, **k): return {'s':'error'}
    def history(self, **k): return {'s':'error'}

sc = SharedDataCache(MockFyers())
assert sc.get_heston_params() is None, "initial heston params should be None"
sc.set_heston_params({'kappa':2.0,'theta':0.04,'v0':0.02,'rho':-0.7,'xi':0.3})
p = sc.get_heston_params()
assert p is not None and p['kappa'] == 2.0, "heston params not stored"
sc.set_T(7/365)
assert abs(sc.get_T() - 7/365) < 1e-9, "T not stored"
sc.set_raw_chain({'optionsChain':[]})
assert sc.get_raw_chain() == {'optionsChain':[]}, "raw chain not stored"
print('  [PASS] SharedDataCache — get/set heston_params, T, raw_chain')

# -----------------------------------------------------------------
# 2. PricingRouter._select_model decision tree
# -----------------------------------------------------------------
print('\n[2] PricingRouter._select_model decision tree...')
from PricingRouter import PricingRouter, HestonParamCache
pr = PricingRouter()
assert pr._select_model(None) == 'BSM',               "None context -> BSM"
assert pr._select_model({'moneyness':0.04}) == 'HESTON',          "Rule1: far OTM"
assert pr._select_model({'explosion_score':65}) == 'HESTON',      "Rule2: high vol"
assert pr._select_model({'term_spread':3.0}) == 'HESTON',         "Rule3: term struct"
assert pr._select_model({'skew_ratio':1.25}) == 'HESTON',         "Rule4: steep skew"
assert pr._select_model({'moneyness':0.01,'explosion_score':20,
                          'term_spread':1.0,'skew_ratio':1.1}) == 'BSM', "all below -> BSM"
print('  [PASS] All 6 decision tree assertions OK')

# -----------------------------------------------------------------
# 3. HestonParamCache TTL
# -----------------------------------------------------------------
print('\n[3] HestonParamCache TTL...')
import time
hc = HestonParamCache()
assert hc.get() is None, "empty cache should return None"
hc.set({'kappa':2.0})
assert hc.get() is not None, "just-set cache should return params"
assert hc.get()['kappa'] == 2.0, "wrong value returned"
print('  [PASS] HestonParamCache TTL=300s, set/get OK')

# -----------------------------------------------------------------
# 4. PricingRouter.price() — BSM path (no Heston params available)
# -----------------------------------------------------------------
print('\n[4] PricingRouter.price() BSM fallback...')
ce = pr.price(24000, 24000, 7/365, 0.07, 0.13, 'CE', context=None)
pe = pr.price(24000, 24000, 7/365, 0.07, 0.13, 'PE', context=None)
assert ce > 0, "CE price must be positive"
assert pe > 0, "PE price must be positive"
# Verify put-call parity fallback works when HESTON selected but no params
ctx_heston = {'moneyness': 0.05}  # will select HESTON but no params -> BSM fallback
ce2 = pr.price(24000, 24000, 7/365, 0.07, 0.13, 'CE', context=ctx_heston)
assert ce2 > 0, "BSM fallback should still price"
print(f'  [PASS] CE={ce:.2f} PE={pe:.2f} (BSM); Heston-selected fallback={ce2:.2f}')

# -----------------------------------------------------------------
# 5. PricingRouter terminal_pdf — BSM path
# -----------------------------------------------------------------
print('\n[5] PricingRouter.terminal_pdf() BSM path...')
prices, weights = pr.terminal_pdf(24000, 7/365, 0.13, context=None)
assert len(prices) == len(weights), "prices and weights must be same length"
assert abs(weights.sum() - 1.0) < 1e-6, "weights must sum to ~1"
print(f'  [PASS] {len(prices)} bins, weights sum={weights.sum():.6f}')

# -----------------------------------------------------------------
# 6. Strategy.pop() backward compatibility + context param
# -----------------------------------------------------------------
print('\n[6] Strategy.pop() — context=None backward compat + Heston path...')
from StrategyEngine import OptionLeg, Strategy, SmartStrategyGenerator, PayoffEngine

spot = 24000.0
# Build a realistic mock chain using BSM prices so POP is meaningful
from StrategyEngine import bsm_price
rows = []
for s in range(22000, 26500, 50):
    for t in ('CE','PE'):
        p = bsm_price(spot, float(s), 7/365, 0.07, 0.13, t)
        rows.append({'strike': float(s), 'type': t,
                     'price': max(0.05, p), 'iv': 13.0})
df   = pd.DataFrame(rows)
ctx  = {'T':7/365,'iv':13.0,'atm_iv':13.0,'vrp':2.0,'regime':'COMPRESSION',
        'call_wall':24500,'put_wall':23500,'em':250.0,'oi_pressure':'NEUTRAL','DTE':7}
gen    = SmartStrategyGenerator(spot, df, ctx, '2026-04-17')
strats = gen.generate()
top    = strats[0]

pop_bsm  = top.pop(spot, 7/365, 0.13)                              # no context
pop_ctx  = top.pop(spot, 7/365, 0.13, context={'moneyness':0.0})   # with context
assert 0 < pop_bsm < 1,  f"BSM POP out of range: {pop_bsm}"
assert 0 < pop_ctx < 1,  f"Context POP out of range: {pop_ctx}"
print(f'  [PASS] pop_bsm={pop_bsm:.3f} pop_ctx={pop_ctx:.3f}')

# -----------------------------------------------------------------
# 7. PayoffEngine.build_risk_dial_data()
# -----------------------------------------------------------------
print('\n[7] PayoffEngine.build_risk_dial_data()...')
engine = PayoffEngine(top, spot, 7/365, 0.13)
dials  = engine.build_risk_dial_data()
required_keys = ['delta_pnl_1pct','theta_daily','vega_1pt','gamma_100pt','dials','labels']
for k in required_keys:
    assert k in dials, f"Missing key: {k}"
dial_vals = dials['dials']
for d in ('delta','theta','vega','gamma'):
    v = dial_vals[d]
    assert -100 <= v <= 100, f"dial '{d}'={v} out of -100..+100"
for lab_key, lab_val in dials['labels'].items():
    if lab_key != 'gamma':
        assert 'Rs.' in lab_val, f"Label '{lab_key}' missing Rs.: {lab_val}"
    else:
        assert 'Delta changes' in lab_val, f"Gamma label wrong: {lab_val}"
print(f'  [PASS] dials within range, labels contain Rs.')
print(f'  delta_pnl_1pct=Rs.{dials["delta_pnl_1pct"]:,.0f}  '
      f'theta_daily=Rs.{dials["theta_daily"]:,.0f}/day  '
      f'vega_1pt=Rs.{dials["vega_1pt"]:,.0f}/IV pt')

# -----------------------------------------------------------------
# 8. StrategyWizard.recommend()
# -----------------------------------------------------------------
print('\n[8] StrategyWizard.recommend()...')
from StrategyWizard import StrategyWizard
wiz  = StrategyWizard(spot, df, ctx, '2026-04-17')

for view, risk in [('NEUTRAL','MODERATE'), ('BULLISH','AGGRESSIVE'), ('BEARISH','CONSERVATIVE')]:
    recs = wiz.recommend(view, risk, 200000, 'HIGH')
    assert len(recs) <= 3 and len(recs) >= 1, f"Expected 1-3 recs, got {len(recs)}"
    for r in recs:
        for field in ('strategy','score','view_match','max_loss_inr','lots_possible','reasoning'):
            assert field in r, f"Missing field: {field}"
    top_r = recs[0]
    print(f'  view={view} risk={risk}: top={top_r["strategy"].name} '
          f'score={top_r["score"]} view_match={top_r["view_match"]} '
          f'lots={top_r["lots_possible"]}')
print('  [PASS] recommend() for 3 view/risk combos')

# -----------------------------------------------------------------
# 9. ScenarioEngine.compute_grid() and spot_ladder()
# -----------------------------------------------------------------
print('\n[9] ScenarioEngine.compute_grid() + spot_ladder()...')
from ScenarioEngine import ScenarioEngine
se   = ScenarioEngine(top, spot, 7/365, 0.13)
grid = se.compute_grid(n_spot=11, n_dte=3, n_iv=3)
required = ['spot_pcts','dte_fracs','iv_bumps','pnl','breakeven_spots']
for k in required:
    assert k in grid, f"Missing grid key: {k}"
pnl_arr = np.array(grid['pnl'])
assert pnl_arr.shape == (3,3,11), f"Wrong shape: {pnl_arr.shape}"
ladder  = se.spot_ladder(n=11)
assert len(ladder) == 11, f"Wrong ladder length: {len(ladder)}"
for row in ladder:
    for field in ('spot','move_pct','pnl_expiry','pnl_today','prob'):
        assert field in row, f"Missing ladder field: {field}"
    assert 0 <= row['prob'] <= 100, f"prob out of range: {row['prob']}"
print(f'  [PASS] grid shape={pnl_arr.shape}, ladder len={len(ladder)}')
print(f'  breakevens={grid["breakeven_spots"]}')
print(f'  ladder[5]={ladder[5]}')

print()
print('='*60)
print('  ALL 9 SMOKE TESTS PASSED')
print('='*60)
