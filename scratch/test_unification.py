import pandas as pd
from calculations.GexEngine import GexEngine
from VolatilityAnalyzer import VolatilityAnalyzer

strikes = [23000, 23100, 23200, 23300, 23400]
rows = []
for s in strikes:
    rows.append({'strike': s, 'type': 'CE', 'oi': 45000, 'iv': 14.5, 'price': 120, 'delta': 0.5, 'gamma': 0.0012, 'vega': 10, 'theta': -5, 'dte': 2.0})
    rows.append({'strike': s, 'type': 'PE', 'oi': 40000, 'iv': 15.0, 'price': 110, 'delta': -0.5, 'gamma': 0.0012, 'vega': 10, 'theta': -5, 'dte': 2.0})
df = pd.DataFrame(rows)
spot = 23200.0

# 1. GexEngine (Model B)
gex_eng = GexEngine(lot_size=65, positioning_model='standard')
res1 = gex_eng.calculate_gex(df, spot)

# 2. VolatilityAnalyzer compute_seller_data
va = VolatilityAnalyzer.__new__(VolatilityAnalyzer)
va.selected_strikes = []
class MockAnalytics:
    def calculate_greeks(self, *a, **k):
        return {'theta': -5.0}
    def get_time_to_expiry(self, *a, **k):
        return 2.0 / 365.0
va.analytics = MockAnalytics()  # type: ignore
va._ensure_iv = lambda row_iv, price, strike, T, option_type: 14.5  # type: ignore

res2 = va.compute_seller_data(df_chain=df, spot=spot, T=2.0/365.0, baseline_oi={}, velocity_baseline={})

print('=== GexEngine (Model B) ===')
print('Net GEX Crores:', f"{res1['net_gex'] / 1e7:+.2f} Cr")
print('Zero Gamma Level:', res1['zero_gamma_level'])
print('Call Wall:', res1['call_wall'])
print('Put Wall:', res1['put_wall'])

print('\n=== compute_seller_data (Option Chain Tab) ===')
if res2 and 'chain_rows' in res2:
    net_cr_sum = sum(r['net_gex_cr'] for r in res2['chain_rows'])
    print(f'Net GEX Sum (₹ Cr): {net_cr_sum:+.2f} Cr')
    for r in res2['chain_rows']:
        print(f"Strike {r['strike']}: CE GEX={r['ce_gex_cr']:+.2f} Cr, PE GEX={r['pe_gex_cr']:+.2f} Cr, Net={r['net_gex_cr']:+.2f} Cr")
