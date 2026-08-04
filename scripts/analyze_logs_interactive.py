import json
import pandas as pd
import plotly.graph_objects as go
import warnings

warnings.filterwarnings("ignore")

# 1. Load JSON Data
try:
    with open('market_memory.json', 'r') as f:
        data = json.load(f)
except Exception as e:
    print("Error reading JSON:", e)
    exit(1)

signals = data.get('active_signals', []) + data.get('resolved_signals', [])
seller_signals = [s for s in signals if s['source'] == 'OptionSellerAdvisor' and 'spot_at_log' in s]

if not seller_signals:
    print("No OptionSellerAdvisor signals found in market_memory.json.")
    exit(0)

# 2. Extract Data
rows = []
for s in seller_signals:
    try:
        rows.append({
            'Timestamp': pd.to_datetime(s['timestamp']),
            'Spot': float(s['spot_at_log']),
            'Sell CE Above': float(s['signal'].get('sell_ce_above')),
            'Sell PE Below': float(s['signal'].get('sell_pe_below'))
        })
    except Exception as e:
        continue

df = pd.DataFrame(rows)
df = df.sort_values('Timestamp').drop_duplicates(subset=['Timestamp']).reset_index(drop=True)

# 3. Detect Signal Changes & Group into Blocks
# A signal change occurs when either CE or PE differs from the previous row
df['SignalChange'] = (df['Sell CE Above'].shift() != df['Sell CE Above']) | \
                     (df['Sell PE Below'].shift() != df['Sell PE Below'])
# The cumsum() generates a unique ID for each contiguous block of identical signals
df['BlockID'] = df['SignalChange'].cumsum()

# Group by the BlockID to compute OHLC of the Spot price during that signal condition
grouped = df.groupby('BlockID').agg(
    StartTime=('Timestamp', 'first'),
    EndTime=('Timestamp', 'last'),
    Open=('Spot', 'first'),
    High=('Spot', 'max'),
    Low=('Spot', 'min'),
    Close=('Spot', 'last'),
    SellCE=('Sell CE Above', 'first'),
    SellPE=('Sell PE Below', 'first')
).reset_index()

print(f"Processed {len(df)} total logs into {len(grouped)} distinct signal blocks (candlesticks).")

# 4. Plot with Plotly
fig = go.Figure()

# Plot Spot OHLC Candlesticks
fig.add_trace(go.Candlestick(
    x=grouped['StartTime'],
    open=grouped['Open'],
    high=grouped['High'],
    low=grouped['Low'],
    close=grouped['Close'],
    name='Spot OHLC'
))

# Plot 'Sell CE Above' Line using Step shape
fig.add_trace(go.Scatter(
    x=grouped['StartTime'], 
    y=grouped['SellCE'], 
    mode='lines+markers', 
    line=dict(color='red', width=2, dash='dash'), 
    name='Sell CE Above (Resistance)', 
    line_shape='hv' # Step line mapping
))

# Plot 'Sell PE Below' Line using Step shape
fig.add_trace(go.Scatter(
    x=grouped['StartTime'], 
    y=grouped['SellPE'], 
    mode='lines+markers', 
    line=dict(color='green', width=2, dash='dash'), 
    name='Sell PE Below (Support)', 
    line_shape='hv' # Step line mapping
))

# 5. Format & Layout
fig.update_layout(
    title='Option Seller Zones vs Spot Movement (Grouped by Signal Block)',
    yaxis_title='NIFTY Level',
    xaxis_title='Time of Signal Block Start',
    xaxis_rangeslider_visible=True,
    template='plotly_dark',
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)

# Fix x-axis formatting for datetime
fig.update_xaxes(
    type='category', 
    categoryorder='category ascending',
    tickangle=45
)

# Convert StartTime to string if grouping categorically so gaps (nights/weekends) aren't rendered
# Plotly Candlesticks inherently render dates linearly. By using string 'x', it skips empty hours.
fig.data[0].x = grouped['StartTime'].dt.strftime('%m-%d %H:%M')
fig.data[1].x = grouped['StartTime'].dt.strftime('%m-%d %H:%M')
fig.data[2].x = grouped['StartTime'].dt.strftime('%m-%d %H:%M')

out_file = 'SellZone_Candlesticks.html'
fig.write_html(out_file)
print(f"Successfully generated interactive chart: {out_file}")
