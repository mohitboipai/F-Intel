import sys
import os
import time

sys.path.insert(0, os.path.abspath('.'))

from VolatilityAnalyzer import VolatilityAnalyzer

print("Testing single-cycle dashboard generation...")
va = VolatilityAnalyzer()
va.get_spot_price()
print(f"Spot price: {va.spot_price}")

# Check that near and far expiries load
va._setup_expiries()
print(f"Expiries: {va._expiries}")

print("Verification complete.")
