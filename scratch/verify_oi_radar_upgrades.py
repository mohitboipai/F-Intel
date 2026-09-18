import sys
import os
import json
import time

# Ensure project root is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
reconf = getattr(sys.stdout, 'reconfigure', None)
if callable(reconf):
    try:
        reconf(encoding='utf-8')
    except Exception:
        pass

from calculations.OiVelocityEngine import OiVelocityEngine

def test_oi_velocity_engine():
    print("=" * 60)
    print("TEST 1: OiVelocityEngine - Straddle Unwinding (Conflict Resolution)")
    print("=" * 60)

    engine = OiVelocityEngine(lot_size=65)

    t0 = 1700000000.0
    t1 = t0 + 300.0 # 5 min later

    # Snapshot 0
    snap0 = {
        'ts': t0,
        'time_str': '09:20:00',
        'spot': 23205.0,
        'oi_map': {
            (23000, 'CE'): 50000,
            (23000, 'PE'): 120000,
            (23100, 'CE'): 60000,
            (23100, 'PE'): 80000,
            (23200, 'CE'): 100000,
            (23200, 'PE'): 100000,
            (23300, 'CE'): 90000,
            (23300, 'PE'): 50000,
            (23400, 'CE'): 110000,
            (23400, 'PE'): 30000,
            (23500, 'CE'): 150000,
            (23500, 'PE'): 20000,
        }
    }

    # Snapshot 1: Exact User Case: CE unwinding at 23200 (-16,093) & PE liquidation at 23200 (-20,786)
    snap1 = {
        'ts': t1,
        'time_str': '09:25:00',
        'spot': 23210.0,
        'oi_map': {
            (23000, 'CE'): 52000,
            (23000, 'PE'): 120000,
            (23100, 'CE'): 60000,
            (23100, 'PE'): 78000,
            (23200, 'CE'): 100000 - 16093,  # 83907 (-16,093)
            (23200, 'PE'): 100000 - 20786,  # 79214 (-20,786)
            (23300, 'CE'): 92000,
            (23300, 'PE'): 49000,
            (23400, 'CE'): 111000,
            (23400, 'PE'): 30000,
            (23500, 'CE'): 150000,
            (23500, 'PE'): 20000,
        }
    }

    res = engine.calculate_velocity([snap0, snap1], timeframe='5m', spot=23210.0)

    assert res['ok'] is True, "Calculation failed!"
    print(f"Calculation OK: {res['ok']}")
    print(f"Total Call OI: {res['total_call_oi']:,}")
    print(f"Total Put OI:  {res['total_put_oi']:,}")
    print(f"Day PCR:       {res['day_pcr']:.2f}")

    assert res['total_call_oi'] > 0, "Total Call OI missing or zero"
    assert res['total_put_oi'] > 0, "Total Put OI missing or zero"
    assert res['day_pcr'] > 0, "Day PCR missing or zero"

    # Verify Hotspots
    hotspots = res.get('hotspots', [])
    print(f"\nHotspots generated ({len(hotspots)}):")
    for h in hotspots:
        print(f" - [{h.get('type')}] Strike {h.get('strike')} ({h.get('color')}): {h.get('desc')}")

    types = [h.get('type') for h in hotspots]
    assert 'STRADDLE_UNWINDING' in types, "STRADDLE_UNWINDING hotspot was not generated for matching CE/PE unwinds!"
    assert 'CALL_SHORT_COVERING' not in types, "Contradictory CALL_SHORT_COVERING was not suppressed!"
    assert 'PUT_CAPITULATION' not in types, "Contradictory PUT_CAPITULATION was not suppressed!"

    # Verify Call & Put flow
    analysis = res.get('analysis', {})
    call_flow = analysis.get('call_flow', {})
    put_flow = analysis.get('put_flow', {})

    print(f"\nCall Flow: Writing={call_flow.get('top_writing_strike')} ({call_flow.get('top_writing_delta')}), "
          f"Unwinding={call_flow.get('top_unwinding_strike')} ({call_flow.get('top_unwinding_delta')})")
    print(f"Put Flow:  Writing={put_flow.get('top_writing_strike')} ({put_flow.get('top_writing_delta')}), "
          f"Unwinding={put_flow.get('top_unwinding_strike')} ({put_flow.get('top_unwinding_delta')})")

    assert call_flow.get('top_unwinding_strike') == 23200, f"Expected Call top unwind at 23200, got {call_flow.get('top_unwinding_strike')}"
    assert put_flow.get('top_unwinding_strike') == 23200, f"Expected Put top unwind at 23200, got {put_flow.get('top_unwinding_strike')}"

    print(f"\nNarrative:\n{analysis.get('narrative')}")
    assert 'Straddle Unwinding' in analysis.get('narrative'), "Narrative did not mention Straddle Unwinding!"

    print("\n" + "=" * 60)
    print("TEST 2: OiVelocityEngine - Straddle Pinning (Dual Writing)")
    print("=" * 60)

    # Snapshot 1 with dual writing at 23200 (+15,000 CE, +18,000 PE)
    snap1_write = {
        'ts': t1,
        'time_str': '09:25:00',
        'spot': 23200.0,
        'oi_map': {
            (23000, 'CE'): 50000,
            (23000, 'PE'): 120000,
            (23200, 'CE'): 115000,  # +15,000
            (23200, 'PE'): 118000,  # +18,000
            (23500, 'CE'): 150000,
            (23500, 'PE'): 20000,
        }
    }

    res_write = engine.calculate_velocity([snap0, snap1_write], timeframe='5m', spot=23200.0)
    hotspots_write = res_write.get('hotspots', [])
    types_write = [h.get('type') for h in hotspots_write]
    print(f"Hotspots for dual write: {types_write}")
    assert 'STRADDLE_PINNING' in types_write, "STRADDLE_PINNING hotspot was not generated for matching CE/PE writes!"
    print(f"Narrative: {res_write.get('analysis', {}).get('narrative')}")

    print("\n" + "=" * 60)
    print("TEST 3: Verify VolatilityAnalyzer.py (Template Source Generator)")
    print("=" * 60)

    required_ids = [
        'btn-metric-total',
        'btn-metric-delta',
        'btn-metric-rate',
        'oi-macro-ce-val',
        'oi-macro-pe-val',
        'oi-macro-pcr-val',
        'oi-macro-bias-badge',
        'oi-call-write-hotspot',
        'oi-call-unwind-hotspot',
        'oi-put-write-hotspot',
        'oi-put-unwind-hotspot',
        'oi-velocity-card',
        'oi-hotspots-container',
        'oi-advisory-narrative',
        'oi-velocity-chart',
    ]

    with open('VolatilityAnalyzer.py', 'r', encoding='utf-8') as f:
        content = f.read()

    missing = [i for i in required_ids if f'id="{i}"' not in content]
    if missing:
        print(f"FAILED: VolatilityAnalyzer.py is missing IDs: {missing}")
        assert False, f"Missing IDs in VolatilityAnalyzer.py: {missing}"
    else:
        print(f"PASSED: VolatilityAnalyzer.py has all {len(required_ids)} required IDs!")

    print("\nALL VERIFICATIONS PASSED SUCCESSFULLY!")

if __name__ == '__main__':
    test_oi_velocity_engine()
