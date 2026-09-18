import sys
import os
os.environ.setdefault("CI", "1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
import numpy as np
from RealizedVolEngine import RealizedVolEngine

engine = RealizedVolEngine()

# Create synthetic 260 days of OHLC data to test
dates = pd.date_range(end=pd.Timestamp.now(), periods=260)
base = 23000.0 + np.cumsum(np.random.normal(5, 100, 260))
highs = base + np.random.uniform(50, 150, 260)
lows = base - np.random.uniform(50, 150, 260)
opens = base + np.random.normal(0, 30, 260)
closes = base + np.random.normal(0, 40, 260)

df = pd.DataFrame({
    'open': opens,
    'high': highs,
    'low': lows,
    'close': closes
}, index=dates)

snap = engine.get_regime_snapshot(spot=23280.0, df_daily=df, atm_iv=11.5, intra_rv=8.2)

assert snap is not None, "get_regime_snapshot returned None"

print("Snapshot keys:", list(snap.keys()))
print("Macro:", snap['macro'])
print("Price VRP:", snap['price_vrp'])
print("Horizon points:", snap['horizon_points'])
print("Regime:", snap['regime'])

