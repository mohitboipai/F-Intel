import os
import sys
import glob
import traceback
import pandas as pd
import numpy as np

def calculate_max_pain(df):
    if df.empty:
        return 0
    
    strikes = sorted(df['Strike'].unique())
    calls = df[df['Type'] == 'CE'].set_index('Strike')
    puts = df[df['Type'] == 'PE'].set_index('Strike')
    
    max_pain_strike = 0
    max_pain_value = float('inf')
    
    for K in strikes:
        total_pain = 0
        for call_strike, row in calls.iterrows():
            if K > call_strike:
                total_pain += row['OI'] * (K - call_strike)
        for put_strike, row in puts.iterrows():
            if K < put_strike:
                total_pain += row['OI'] * (put_strike - K)
                
        if total_pain < max_pain_value:
            max_pain_value = total_pain
            max_pain_strike = K
            
    return max_pain_strike


def main():
    bhav_dir = os.path.join("historical_data", "oi_bhavcopy")
    csv_files = sorted(glob.glob(os.path.join(bhav_dir, "fo_bhav_*.csv")))
    print(f"Found {len(csv_files)} bhavcopy files.")

    records = []

    for idx, file in enumerate(csv_files):
        print(f"[{idx+1}/{len(csv_files)}] Processing {os.path.basename(file)} ...")
        try:
            # We only need specific columns to save memory
            usecols = ['TradDt', 'TckrSymb', 'FinInstrmTp', 'StrkPric', 'OptnTp', 'ClsPric', 'UndrlygPric', 'OpnIntrst', 'XpryDt']
            df = pd.read_csv(file, usecols=usecols, engine='c')
            
            # Filter for NIFTY options
            df = df[(df['TckrSymb'] == 'NIFTY') & (df['FinInstrmTp'] == 'IDO')]
            if df.empty:
                continue
            
            date_str = df['TradDt'].iloc[0]
            # Convert XpryDt to dt
            df['XpryDt'] = pd.to_datetime(df['XpryDt'])
            # Get earliest expiry
            near_exp = df['XpryDt'].min()
            
            # Only use near expiry data to mimic intraday OptionSellerAdvisor focus
            df_near = df[df['XpryDt'] == near_exp]
            
            spot = df_near['UndrlygPric'].dropna().iloc[0]
            if pd.isna(spot) or spot <= 0:
                continue

            # Standardize columnnames to match logic expectations
            df_filtered = pd.DataFrame({
                'Strike': df_near['StrkPric'],
                'Type': df_near['OptnTp'],
                'OI': df_near['OpnIntrst'],
                'Price': df_near['ClsPric']
            })

            # Calculate Max Pain
            max_pain = calculate_max_pain(df_filtered)

            # OI Walls
            calls = df_filtered[df_filtered['Type'] == 'CE']
            puts = df_filtered[df_filtered['Type'] == 'PE']

            call_wall = 0
            calls_above = calls[calls['Strike'] > spot]
            if not calls_above.empty:
                top_calls = calls_above.groupby('Strike')['OI'].sum().nlargest(1)
                if not top_calls.empty:
                    call_wall = top_calls.index[0]

            put_wall = 0
            puts_below = puts[puts['Strike'] < spot]
            if not puts_below.empty:
                top_puts = puts_below.groupby('Strike')['OI'].sum().nlargest(1)
                if not top_puts.empty:
                    put_wall = top_puts.index[0]

            # ATM Straddle
            atm_strike = calls.iloc[(calls['Strike'] - spot).abs().argmin()]['Strike'] if not calls.empty else spot
            
            ce_atm = calls[calls['Strike'] == atm_strike]
            pe_atm = puts[puts['Strike'] == atm_strike]
            
            ce_price = ce_atm['Price'].sum() if not ce_atm.empty else 0
            pe_price = pe_atm['Price'].sum() if not pe_atm.empty else 0
            em_straddle = ce_price + pe_price

            # Define Safe ranges
            # Safe CE: Minimum of (Call Wall) and (Spot + Straddle EM)
            safe_ce = call_wall
            if em_straddle > 0:
                if call_wall > 0:
                     safe_ce = min(call_wall, spot + em_straddle)
                else:
                     safe_ce = spot + em_straddle

            # Safe PE: Maximum of (Put Wall) and (Spot - Straddle EM)
            safe_pe = put_wall
            if em_straddle > 0:
                if put_wall > 0:
                     safe_pe = max(put_wall, spot - em_straddle)
                else:
                     safe_pe = spot - em_straddle

            records.append({
                'Date': pd.to_datetime(date_str),
                'Spot': spot,
                'Sell CE Above': safe_ce,
                'Sell PE Below': safe_pe,
                'Max Pain': max_pain,
                'Action': f"SELL CE>{int(safe_ce)} / SELL PE<{int(safe_pe)}",
                'EM': em_straddle
            })
            
        except Exception as e:
            print(f"Error on {file}: {e}")
            traceback.print_exc()
            continue

    if not records:
        print("No records retrieved. Ensure historical files are correct.")
        return

    # Create DF and Export
    rdf = pd.DataFrame(records)
    rdf = rdf.sort_values('Date').reset_index(drop=True)

    print("Fetching precise OHLC history for Spot via Fyers API...")
    try:
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        from FyersAuth import FyersAuthenticator
        
        auth = FyersAuthenticator("QUTT4YYMIG-100", "ZG0WN2NL1B", "http://127.0.0.1:3000/callback")
        fyers = auth.get_fyers_instance()
        
        import datetime
        end_dt = datetime.datetime.now()
        stamp_start = end_dt - datetime.timedelta(days=365)
        
        hist_data = {
            "symbol": "NSE:NIFTY50-INDEX",
            "resolution": "D",
            "date_format": "1",
            "range_from": stamp_start.strftime("%Y-%m-%d"),
            "range_to": end_dt.strftime("%Y-%m-%d"),
            "cont_flag": "1"
        }
        res = fyers.history(data=hist_data)
        
        if res.get('s') == 'ok' and 'candles' in res:
            df_ohlc = pd.DataFrame(res['candles'], columns=['Stamp', 'Open', 'High', 'Low', 'Close', 'Vol'])
            # Convert Fyers stamp (epoch) to pd datetime
            df_ohlc['Date'] = pd.to_datetime(df_ohlc['Stamp'], unit='s').dt.normalize()
            df_ohlc = df_ohlc[['Date', 'Open', 'High', 'Low', 'Close']].drop_duplicates('Date')
            
            # Merge with our sell zones records on 'Date'
            rdf = pd.merge(rdf, df_ohlc, on='Date', how='left')
            # Fallback for missing closing price
            rdf['Close'] = rdf['Close'].fillna(rdf['Spot'])
        else:
            print(f"Failed to retrieve OHLC: {res}")
            rdf['Open'] = np.nan; rdf['High'] = np.nan; rdf['Low'] = np.nan; rdf['Close'] = rdf['Spot']
            
    except Exception as e:
        print(f"Could not fetch Spot OHLC: {e}")
        rdf['Open'] = np.nan; rdf['High'] = np.nan; rdf['Low'] = np.nan; rdf['Close'] = rdf['Spot']

    excel_file = "HistoricalSellZones_6M_OHLC.xlsx"
    writer = pd.ExcelWriter(excel_file, engine='xlsxwriter')
    
    rdf['Date_Str'] = rdf['Date'].dt.strftime('%Y-%m-%d')
    df_out = rdf[['Date_Str', 'Open', 'High', 'Low', 'Close', 'Spot', 'Sell CE Above', 'Sell PE Below', 'Max Pain', 'Action', 'EM']]
    df_out.to_excel(writer, sheet_name='Sell Zones', index=False)

    workbook = writer.book
    worksheet = writer.sheets['Sell Zones']

    worksheet.set_column('A:A', 15)
    worksheet.set_column('B:G', 15)

    max_row = len(rdf)
    chart = workbook.add_chart({'type': 'line'})

    # Spot (Col B=1)
    chart.add_series({
        'name':       ['Sell Zones', 0, 1],
        'categories': ['Sell Zones', 1, 0, max_row, 0],
        'values':     ['Sell Zones', 1, 1, max_row, 1],
        'line':       {'color': 'black', 'width': 2.5}
    })
    # CE
    chart.add_series({
        'name':       ['Sell Zones', 0, 2],
        'categories': ['Sell Zones', 1, 0, max_row, 0],
        'values':     ['Sell Zones', 1, 2, max_row, 2],
        'line':       {'color': '#d62728', 'dash_type': 'dash'}
    })
    # PE
    chart.add_series({
        'name':       ['Sell Zones', 0, 3],
        'categories': ['Sell Zones', 1, 0, max_row, 0],
        'values':     ['Sell Zones', 1, 3, max_row, 3],
        'line':       {'color': '#2ca02c', 'dash_type': 'dash'}
    })

    chart.set_title({'name': 'Spot vs 6-Month Historical Sell Zones (EOD)'})
    chart.set_x_axis({'name': 'Date', 'label_position': 'low'})
    chart.set_y_axis({'name': 'NIFTY Level'})
    chart.set_size({'width': 1000, 'height': 500})
    chart.set_legend({'position': 'bottom'})

    worksheet.insert_chart('H2', chart)
    writer.close()
    
    print(f"\n=========================================================")
    print(f"Successfully generated {excel_file} with {len(rdf)} daily records.")
    print(f"=========================================================")

if __name__ == "__main__":
    main()
