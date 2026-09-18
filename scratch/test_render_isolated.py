import os
import sys

# Ensure root in path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from VolatilityAnalyzer import VolatilityAnalyzer

app = VolatilityAnalyzer()
app.get_spot_price()
app._setup_expiries()

# Monkey patch html_path so it writes to test_unified_dashboard.html
original_create = app._create_unified_dashboard

import VolatilityAnalyzer as va_mod

# Run single run
app._create_unified_dashboard(app._expiries, single_run=True)

test_dest = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'scratch', 'test_unified_dashboard.html')
src = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'unified_dashboard.html')

if os.path.exists(src):
    with open(src, 'r', encoding='utf-8') as f:
        data = f.read()
    with open(test_dest, 'w', encoding='utf-8') as f:
        f.write(data)
    
    print(f"Captured dashboard ({len(data)} bytes)")
    print("  cascade-term-structure-plot present:", 'cascade-term-structure-plot' in data)
    print("  QUANTITATIVE VOLATILITY MODELS present:", 'QUANTITATIVE VOLATILITY MODELS' in data)
    print("  ACTIONABLE TRADING PLAYBOOK present:", 'ACTIONABLE TRADING PLAYBOOK' in data)
    print("  INTRADAY REGIME SIGNAL present:", 'INTRADAY REGIME SIGNAL' in data)
    print("  'lot edge (65 qty)' present:", 'lot edge (65 qty)' in data)
