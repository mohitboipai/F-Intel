import json
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

# load json
try:
    with open('market_memory.json', 'r') as f:
        data = json.load(f)
except Exception as e:
    print("Error reading JSON:", e)
    exit(1)

signals = data.get('active_signals', []) + data.get('resolved_signals', [])
seller_signals = [s for s in signals if s['source'] == 'OptionSellerAdvisor']

if not seller_signals:
    print("No OptionSellerAdvisor signals found in market_memory.json.")
    exit(0)

# flatten
rows = []
for s in seller_signals:
    try:
        rows.append({
            'Timestamp': pd.to_datetime(s['timestamp']),
            'Spot': s['spot_at_log'],
            'Sell CE Above': s['signal'].get('sell_ce_above'),
            'Sell PE Below': s['signal'].get('sell_pe_below'),
            'Max Pain': s['signal'].get('max_pain'),
            'Action': s['signal'].get('action')
        })
    except Exception as e:
        continue

df = pd.DataFrame(rows)
df = df.sort_values('Timestamp').drop_duplicates(subset=['Timestamp'])

# Resample to higher timeframe (15-minute intervals) for cleaner visualization
df.set_index('Timestamp', inplace=True)
# Forward-fill missing numeric values to keep the zone lines continuous, then drop where Spot is entirely missing
df_resampled = df.resample('15T').ffill()
df_resampled = df_resampled.dropna(subset=['Spot']).reset_index()

print(f"Loaded {len(df)} discrete Sell Zone signal states.")
print(f"Resampled to {len(df_resampled)} data points (15-minute timeframe).")

excel_file = 'SellZoneAnalysis_15Min.xlsx'
writer = pd.ExcelWriter(excel_file, engine='xlsxwriter')
df_resampled['Timestamp_str'] = df_resampled['Timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
df_out = df_resampled[['Timestamp_str', 'Spot', 'Sell CE Above', 'Sell PE Below', 'Max Pain', 'Action']]

df_out.to_excel(writer, sheet_name='Sell Zones', index=False)

workbook = writer.book
worksheet = writer.sheets['Sell Zones']

# Format columns to be wider
worksheet.set_column('A:A', 20)
worksheet.set_column('B:E', 15)
worksheet.set_column('F:F', 30)

max_row = len(df_out)
if max_row > 0:
    # create a chart
    chart = workbook.add_chart({'type': 'line'})

    # Spot (Column B=1)
    chart.add_series({
        'name':       ['Sell Zones', 0, 1],
        'categories': ['Sell Zones', 1, 0, max_row, 0],
        'values':     ['Sell Zones', 1, 1, max_row, 1],
        'line':       {'color': 'black', 'width': 2.5}
    })
    # Sell CE Above (Column C=2)
    chart.add_series({
        'name':       ['Sell Zones', 0, 2],
        'categories': ['Sell Zones', 1, 0, max_row, 0],
        'values':     ['Sell Zones', 1, 2, max_row, 2],
        'line':       {'color': '#d62728', 'dash_type': 'dash'}
    })
    # Sell PE Below (Column D=3)
    chart.add_series({
        'name':       ['Sell Zones', 0, 3],
        'categories': ['Sell Zones', 1, 0, max_row, 0],
        'values':     ['Sell Zones', 1, 3, max_row, 3],
        'line':       {'color': '#2ca02c', 'dash_type': 'dash'}
    })

    chart.set_title({'name': 'Spot vs Sell Zones Alignment'})
    chart.set_x_axis({'name': 'Time', 'label_position': 'low'})
    chart.set_y_axis({'name': 'NIFTY Level'})
    chart.set_size({'width': 1000, 'height': 500})
    chart.set_legend({'position': 'bottom'})

    worksheet.insert_chart('H2', chart)

writer.close()
print(f"Successfully generated {excel_file} with chart visualization.")
