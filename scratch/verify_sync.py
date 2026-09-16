with open('templates/unified_dashboard.html', 'r', encoding='utf-8') as f:
    tpl = f.read()

print('Corridor in tpl:', 'INSTITUTIONAL GEX TRADING CORRIDOR' in tpl)
print('P(OTM) in tpl:', 'P(OTM)' in tpl)
print('gex-chart-container in tpl:', 'gex-chart-container' in tpl)
print('gex-distribution-chart in tpl:', 'gex-distribution-chart' in tpl)
print('th-calls-header colspan=7:', 'id="th-calls-header" colspan="7"' in tpl)
print('th-puts-header colspan=7:', 'id="th-puts-header" colspan="7"' in tpl)

with open('static/js/dashboard_core.js', 'r', encoding='utf-8') as f:
    js = f.read()

print('Corridor runway bar injected in JS:', 'gex-corridor-runway-bar" style=' in js)
print('updateGexChartSpot in JS:', 'updateGexChartSpot' in js)
print('Plotly resize on chain in JS:', "name === 'chain'" in js and "window.Plotly.Plots.resize(gexChart)" in js)
print('showSeller ? 2 in JS:', 'showSeller ? 2 : 0' in js)
