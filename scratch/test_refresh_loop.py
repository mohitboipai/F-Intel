import sys
import os
import json
import urllib.request
import traceback
import pandas as pd

sys.path.insert(0, '.')

try:
    from DataServer import _parse_chain_to_df, _gex_rebalance_engine, _gex_rebalance_snapshot, hub_cache
    from calculations.GexEngine import GexEngine

    # Fetch live data from running server
    req = urllib.request.urlopen("http://127.0.0.1:8082/get_data")
    data = json.loads(req.read().decode('utf-8'))
    spot = data.get("spot", 0)
    chain = data.get("chain", {})

    print(f"Spot: {spot}")
    df = _parse_chain_to_df(chain)
    print(f"DF shape: {df.shape}")

    # Test GexEngine
    gex_eng = GexEngine(lot_size=65, positioning_model='standard')
    gex_res = gex_eng.calculate_gex(df, spot)
    print("GexEngine result keys:", list(gex_res.keys()))
    print("GexEngine net_gex:", gex_res.get('net_gex'))
    print("GexEngine zero_gamma_level:", gex_res.get('zero_gamma_level'))
    print("GexEngine call_wall:", gex_res.get('call_wall'))
    print("GexEngine put_wall:", gex_res.get('put_wall'))

    # Test GexRebalanceEngine
    dte_val = hub_cache.get_T() * 365.0
    print(f"dte_val: {dte_val}")
    
    if _gex_rebalance_engine is not None:
        gr_payload = _gex_rebalance_engine.evaluate(
            df, spot,
            oi_velocity_data=None,
            dte=dte_val
        )
        print("GexRebalanceEngine payload:")
        print(json.dumps(gr_payload, indent=2, default=str))
    else:
        print("GexRebalanceEngine is not initialized")

except Exception as e:
    print(f"ERROR: {e}")
    traceback.print_exc()
