import sys
import traceback
import urllib.request
import json
import pandas as pd

sys.path.insert(0, '.')

try:
    from calculations.GexRebalanceEngine import GexRebalanceEngine
    from DataServer import hub, _parse_chain_to_df, _gex_rebalance_engine

    print("Checking _gex_rebalance_engine:", _gex_rebalance_engine)

    # Fetch latest chain from DataServer or local endpoint
    req = urllib.request.urlopen("http://127.0.0.1:8082/get_data")
    data = json.loads(req.read().decode('utf-8'))
    spot = data.get("spot", 0)
    chain = data.get("chain", {})

    print(f"Spot: {spot}, Chain strikes count: {len(chain)}")
    
    df = _parse_chain_to_df(chain)
    print("Parsed df shape:", df.shape)
    if not df.empty:
        print("Columns in df:", df.columns.tolist())
        print("Head of df:")
        print(df.head(2))

        engine = GexRebalanceEngine(lot_size=65)
        res = engine.evaluate(df, spot)
        print("Evaluate success! Result:")
        print(json.dumps(res, indent=2, default=str))

except Exception as e:
    print(f"ERROR: {e}")
    traceback.print_exc()
