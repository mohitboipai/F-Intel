import os

path = os.path.join(os.path.dirname(__file__), 'test_unified_dashboard.html')
with open(path, 'r', encoding='utf-8') as f:
    html = f.read()

idx1 = html.find('id="tab-regime"')
idx2 = html.find('</section>', idx1)
sec = html[idx1:idx2]
print(f'Regime Section length: {len(sec):,} chars')

checks = [
    ('ACTIVE MARKET REGIME', 'ACTIVE MARKET REGIME' in sec),
    ('VRP IN PRICE (POINTS & RUPEES)', 'VOLATILITY RISK PREMIUM IN PRICE' in sec),
    ('DAILY VRP PRICE BUFFER (1D)', 'DAILY VRP PRICE BUFFER (1D)' in sec),
    ('WEEKLY ATM STRADDLE (5 DTE)', 'WEEKLY ATM STRADDLE (5 DTE)' in sec),
    ('WEEKLY 5-DAY RANGE (1-Sigma)', 'WEEKLY 5-DAY RANGE' in sec),
    ('SELLERS\' PREMIUM MULTIPLE', 'SELLERS\' PREMIUM MULTIPLE' in sec),
    ('CASCADE LINE PLOT (Plotly)', 'cascade-term-structure-plot' in sec),
    ('1-YEAR REALIZED VOL CONE TABLE', 'REALIZED VOLATILITY CONE' in sec),
    ('QUANT VOL MODELS BRIEF (REMOVED)', 'QUANTITATIVE VOLATILITY MODELS' in sec),
    ('ADVANCED VOL METERS / CORSI (REMOVED)', 'ADVANCED VOLATILITY MODELING' in sec),
    ('MULTI-TIMEFRAME REGIME MATRIX (REMOVED)', 'MULTI-TIMEFRAME REGIME MATRIX' in sec),
    ('REGIME TRANSITION LOG (REMOVED)', 'REGIME TRANSITION LOG' in sec),
    ('ACTIONABLE TRADING PLAYBOOK (REMOVED)', 'ACTIONABLE TRADING PLAYBOOK' in sec),
    ('INTRADAY REGIME SIGNAL (REMOVED)', 'INTRADAY REGIME SIGNAL' in sec),
    ('LOT SIZE 65 QTY CONVERSION', '65 Qty' in sec and 'lot edge (65 qty)' in sec),
]

all_pass = True
for name, res in checks:
    status = "PASS" if (res if "REMOVED" not in name else not res) else "FAIL"
    if status == "FAIL":
        all_pass = False
    print(f"  [{status}] {name}")

print(f"\nOverall: {'ALL TESTS PASSED' if all_pass else 'SOME TESTS FAILED'}")
