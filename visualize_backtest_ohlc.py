import json
import pandas as pd
import plotly.graph_objects as go
import warnings
import os
import sys

warnings.filterwarnings("ignore")

print("1. Loading OptionSellerAdvisor logs from market_memory.json...")
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

# Extract into DataFrame
rows = []
for s in seller_signals:
    try:
        rows.append({
            'Timestamp': pd.to_datetime(s['timestamp']),
            'SellCE': float(s['signal'].get('sell_ce_above')),
            'SellPE': float(s['signal'].get('sell_pe_below'))
        })
    except Exception as e:
        continue

df_sig = pd.DataFrame(rows)
df_sig = df_sig.dropna(subset=['SellCE', 'SellPE'])
df_sig = df_sig.sort_values('Timestamp').drop_duplicates(subset=['Timestamp'])

min_date = df_sig['Timestamp'].min()
max_date = df_sig['Timestamp'].max()

min_date_str = (min_date - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
max_date_str = (max_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

print(f"Signal Date Range: {min_date_str} to {max_date_str}")
print("2. Fetching true 15-minute OHLC from Fyers API...")

try:
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from FyersAuth import FyersAuthenticator
    
    auth = FyersAuthenticator("QUTT4YYMIG-100", "ZG0WN2NL1B", "http://127.0.0.1:3000/callback")
    fyers = auth.get_fyers_instance()
    
    hist_data = {
        "symbol": "NSE:NIFTY50-INDEX",
        "resolution": "15",
        "date_format": "1",
        "range_from": min_date_str,
        "range_to": max_date_str,
        "cont_flag": "1"
    }
    res = fyers.history(data=hist_data)
    
    if res.get('s') == 'ok' and 'candles' in res:
        df_ohlc = pd.DataFrame(res['candles'], columns=['Stamp', 'Open', 'High', 'Low', 'Close', 'Vol'])
        # Fyers gives UTC unix epoch, Nifty opens at 3:45 AM UTC (which is 09:15 AM IST)
        # Convert to local India Standard Time matching the signal logs
        df_ohlc['Timestamp'] = pd.to_datetime(df_ohlc['Stamp'], unit='s') + pd.Timedelta(hours=5, minutes=30)
        df_ohlc = df_ohlc[['Timestamp', 'Open', 'High', 'Low', 'Close']]
    else:
        print(f"Failed to fetch Fyers history: {res}")
        exit(1)
except Exception as e:
    print(f"Exception while connecting to Fyers API: {e}")
    exit(1)

print(f"Loaded {len(df_ohlc)} candlestick intervals from Fyers.")
print("3. Merging Intraday signals securely onto timeframe backbone via forward fill...")

# To merge successfully, both must be sorted by Timestamp
df_ohlc = df_ohlc.sort_values('Timestamp')
df_sig = df_sig.sort_values('Timestamp')

# We use pandas.merge_asof to match each 15-minute candle timestamp with the *last available* signal BEFORE that timestamp.
df_merged = pd.merge_asof(
    df_ohlc, 
    df_sig,
    on='Timestamp',
    direction='backward'
)

# Optional: If signals were strictly for active intervals, we drop ones where SellCE/PE are massively outdated (e.g. days ago)
# But since market_memory logs signal presence dynamically, we'll keep the ffill mapping.
df_merged = df_merged.dropna(subset=['SellCE', 'SellPE'])

if df_merged.empty:
    print("Merge failed or no overlap between Logs Date range and API Date range!")
    exit(1)

print(f"Aligned {len(df_merged)} interval logs. Plotting...")

fig = go.Figure()

# Plot OHLC Fyers true continuous timeframe
fig.add_trace(go.Candlestick(
    x=df_merged['Timestamp'],
    open=df_merged['Open'],
    high=df_merged['High'],
    low=df_merged['Low'],
    close=df_merged['Close'],
    name='15-Min Spot (Fyers)'
))

# Plot 'Sell CE Above' Dynamic Line mapped accurately over standard timeframes
fig.add_trace(go.Scatter(
    x=df_merged['Timestamp'], 
    y=df_merged['SellCE'], 
    mode='lines', 
    line=dict(color='red', width=2, dash='dash'), 
    name='Sell CE Above (Resistance)', 
    line_shape='hv'
))

# Plot 'Sell PE Below' Dynamic Line mapped accurately over standard timeframes
fig.add_trace(go.Scatter(
    x=df_merged['Timestamp'], 
    y=df_merged['SellPE'], 
    mode='lines', 
    line=dict(color='green', width=2, dash='dash'), 
    name='Sell PE Below (Support)', 
    line_shape='hv'
))

# Clean gaps for nights and weekends from Plotly interactive display
fig.update_xaxes(
    rangebreaks=[
        dict(bounds=["15:30", "09:15"]), # hide non-trading hours
        dict(bounds=["sat", "mon"]),     # hide weekends
    ],
    rangeslider_visible=True
)

fig.update_layout(
    title='15m Candlestick Chart (Spot) with Intraday OptionSeller Zones',
    yaxis_title='NIFTY Level',
    xaxis_title='Date & Time',
    template='plotly_dark',
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    height=800
)

out_file = 'TrueBacktest_15Min_Plot.html'
fig.write_html(out_file)
print(f"\n=========================================================")
print(f"Successfully generated comprehensive chart: {out_file}")
print(f"=========================================================")
