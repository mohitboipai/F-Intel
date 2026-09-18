import os
import sys

# Ensure root in path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from VolatilityAnalyzer import VolatilityAnalyzer

app = VolatilityAnalyzer()
app.get_spot_price()
app._setup_expiries()
print("Regenerating dashboard with Lot Size 65 and Enhanced VRP Matrix...")
app._create_unified_dashboard(app._expiries, single_run=True)
print("Dashboard regenerated successfully!")
