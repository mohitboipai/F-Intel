import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from VolatilityAnalyzer import VolatilityAnalyzer

app = VolatilityAnalyzer()
app.get_spot_price()
app._setup_expiries()
print("Running single-pass update...")
app._create_unified_dashboard(app._expiries, single_run=True)
print("Single-pass update complete!")
