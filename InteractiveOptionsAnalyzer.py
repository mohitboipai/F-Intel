import sys
import time
import pandas as pd
from datetime import datetime
from FyersAuth import FyersAuthenticator
from OptionAnalytics import OptionAnalytics

class InteractiveOptionsAnalyzer:
    def __init__(self):
        self.fyers = self._authenticate()
        self.analytics = OptionAnalytics()
        self.symbol = "NSE:NIFTY50-INDEX"
        self.selected_strikes = []
        self.spot_price = 0
        self.expiry_date = None
        
    def _authenticate(self):
        print("Authenticating with Fyers...")
        APP_ID = "QUTT4YYMIG-100"
        SECRET_ID = "ZG0WN2NL1B"
        REDIRECT_URI = "http://127.0.0.1:3000/callback"
        
        auth = FyersAuthenticator(APP_ID, SECRET_ID, REDIRECT_URI)
        fyers = auth.get_fyers_instance()
        if not fyers:
            print("Authentication Failed!")
            sys.exit(1)
        print("Authentication Successful.")
        return fyers

    def get_spot_price(self, verbose=True):
        data = {"symbols": self.symbol}
        try:
            response = self.fyers.quotes(data=data)
            
            # Handle Token Error
            if response.get('code') == -15 or "token" in response.get('message', '').lower():
                if verbose: print("Token expired or invalid during quote fetch. Re-authenticating...")
                self.fyers = self._authenticate()
                response = self.fyers.quotes(data=data)
                
            if response['s'] == "ok":
                self.spot_price = response['d'][0]['v']['lp']
                if verbose: print(f"Fetched Spot Price: {self.spot_price}")
                return self.spot_price
            else:
                print(f"Error response from quotes: {response}")
        except Exception as e:
            print(f"Error fetching spot: {e}")
        return 0

    def get_option_chain_data(self):
        # Convert expiry to epoch if set
        ts = ""
        if self.expiry_date:
             try:
                 # Fyers might expect simple YYYY-MM-DD or epoch
                 # API doc usually says timestamp (epoch). Let's convert.
                 # Assuming self.expiry_date is YYYY-MM-DD
                 dt = datetime.strptime(self.expiry_date, "%Y-%m-%d")
                 ts = int(dt.timestamp())
             except:
                 ts = ""

        data = {
            "symbol": self.symbol,
            "strikecount": 500, # Increased from 100
            "timestamp": ts
        }
        try:
            response = self.fyers.optionchain(data=data)
            
            # Handle Token Error
            if response.get('code') == -15 or "token" in response.get('message', '').lower():
                print("Token expired or invalid during chain fetch. Re-authenticating...")
                self.fyers = self._authenticate()
                response = self.fyers.optionchain(data=data)

            # Handle Invalid Expiry (Auto-Correction)
            if response.get('s') == 'error' and 'expiryData' in response.get('data', {}):
                # print(f"DEBUG: Timestamp mismatch. Searching valid expiry list...")
                valid_expiries = response['data']['expiryData']
                correct_ts = None
                
                # Check user date against list
                # User format: YYYY-MM-DD (self.expiry_date)
                # API format: DD-MM-YYYY
                if self.expiry_date:
                    try:
                        u_date = datetime.strptime(self.expiry_date, "%Y-%m-%d").date()
                        for item in valid_expiries:
                             api_str = item.get('date')
                             api_ts = item.get('expiry')
                             # Parse API date
                             a_date = datetime.strptime(api_str, "%d-%m-%Y").date()
                             if u_date == a_date:
                                 correct_ts = api_ts
                                 break
                    except Exception as e:
                        print(f"Date matching error: {e}")
                else:
                    # Auto-select nearest if not set
                    if valid_expiries:
                        first = valid_expiries[0]
                        correct_ts = first.get('expiry')
                        print(f"Auto-Selected Nearest Expiry: {first.get('date')}")
                        # Update self.expiry_date for consistency
                        try:
                             d_str = first.get('date')
                             self.expiry_date = datetime.strptime(d_str, "%d-%m-%Y").strftime("%Y-%m-%d")
                        except: pass
                
                if correct_ts:
                    data['timestamp'] = correct_ts
                    response = self.fyers.optionchain(data=data)

            if response.get('s') == "ok":
                return response.get('data', {})
            else:
                print(f"Error fetching option chain: {response}")
                # Optional: specific hint if expiry failed
                if "expiry" in response.get('message', '').lower():
                     print("Hint: Check if the Expiry Date matches a valid Weekly/Monthly expiry.")
        except Exception as e:
            print(f"Error fetching option chain: {e}")
        return None

    def parse_and_filter(self, data):
        if not data:
            return pd.DataFrame()
            
        options = data.get('optionsChain', [])
        records = []
        
        if not options:
             print(f"DEBUG: No optionsChain in data. Keys: {data.keys()}")
             
        for item in options:
            strike = item.get('strike_price')
            
            # Filter by strike if selected
            if self.selected_strikes:
                # Debug first few to check type
                # print(f"Check {strike} ({type(strike)}) against {self.selected_strikes}")
                pass
            
            if self.selected_strikes and strike not in self.selected_strikes:
                 # Try loose float comparison
                 match = False
                 for s in self.selected_strikes:
                     if abs(float(strike) - float(s)) < 0.1:
                         match = True
                         break
                 if not match:
                     continue
                
            # Extract Expiry
            # Fyers 'expiry_date' is usually a timestamp or string. 
            # We need to check the actual key. Based on docs, it might be 'expiry_date'
            # Let's try to find a readable expiry string
            remote_expiry = item.get('expiry_date') # e.g. "28-Nov-2024"?? verify format
            
            # Identify Option Type
            # Fyers: "option_type" -> "CALL" or "PUT"
            otype = item.get('option_type')
            if otype == "CALL":
                option_type = "CE" 
            elif otype == "PUT": 
                option_type = "PE"
            else:
                continue # Skip if unknown

            records.append({
                'symbol': item.get('symbol'),
                'strike': strike,
                'type': option_type,
                'price': item.get('ltp'),
                'expiry': remote_expiry,
                'iv': item.get('iv'),
                'delta': item.get('delta'),
                'gamma': item.get('gamma'),
                'vega': item.get('vega'),
                'theta': item.get('theta'),
                'oi': item.get('oi')
            })
            
        return pd.DataFrame(records)

    def select_expiry_and_filter(self, df):
        if df.empty: return df
        
        # Get unique expiries
        expiries = sorted(df['expiry'].unique())
        
        # If user hasn't set an expiry or it's not in list, ask them
        if not self.expiry_date or self.expiry_date not in expiries:
            print("\nAvailable Expiries:")
            for i, exp in enumerate(expiries):
                print(f"{i+1}. {exp}")
            
            # Auto-select nearest if not interactive or just 1
            if len(expiries) == 1:
                self.expiry_date = expiries[0]
            else:
                try:
                    idx = int(input(f"Select Expiry (1-{len(expiries)}): ")) - 1
                    if 0 <= idx < len(expiries):
                        self.expiry_date = expiries[idx]
                        
                    else:
                        print("Invalid selection. Using nearest.")
                        self.expiry_date = expiries[0]
                except:
                    print("Invalid input. Using nearest.")
                    self.expiry_date = expiries[0]
        
        print(f"Active Expiry: {self.expiry_date}")
        return df[df['expiry'] == self.expiry_date].copy()

    def fetch_realtime_quotes(self, df):
        if df.empty: return df
        
        symbols = df['symbol'].tolist()
        # Batch requests if needed (limit 50 usually)
        # For now, let's assume < 50 items for selected strikes
        if len(symbols) > 50:
             print(f"Warning: Fetching quotes for {len(symbols)} symbols. This might be slow.")
        
        joined_symbols = ",".join(symbols)
        data = {"symbols": joined_symbols}
        
        try:
            # print(f"Fetching quotes for: {joined_symbols}")
            response = self.fyers.quotes(data=data)
            
            # Handle Token Error
            if response.get('code') == -15 or "token" in response.get('message', '').lower():
                 self.fyers = self._authenticate()
                 response = self.fyers.quotes(data=data)
            
            if response.get('s') == "ok":
                # Create a map symbol -> ltp
                quote_map = {}
                for d in response['d']:
                    quote_map[d['n']] = d['v'].get('lp', 0)
                
                # Update DataFrame
                df['price'] = df['symbol'].map(quote_map).fillna(df['price'])
                print("Updated with Real-time Quotes.")
            else:
                print(f"Error fetching quotes: {response}")
                
        except Exception as e:
            print(f"Quote Fetch Exception: {e}")
            
        return df

    def real_time_greek_monitor(self):
        print("\n--- Real-Time Greek Monitor (Call + Put + Straddle) ---")
        print("1. Manual Input (Simulated Greeks)")
        print("2. Fetch from Live Strike")
        
        c = input("Select Initialization: ")
        
        # Structure to hold leg data
        legs = {
            'CE': {'price': 0.0, 'delta': 0.5, 'gamma': 0.002, 'theta': -5.0, 'vega': 5.0, 'iv': 15.0},
            'PE': {'price': 0.0, 'delta': -0.5, 'gamma': 0.002, 'theta': -5.0, 'vega': 5.0, 'iv': 15.0}
        }
        ref_spot = 24000.0
        T = 0.05 # Default T
        
        if c == '2':
            # Fetch logic
            self.get_spot_price()
            ref_spot = self.spot_price
            
            try:
                strike = float(input("Enter Strike (Center): "))
                self.selected_strikes = [strike] # Ensure fetcher includes it
                
                # We need T for calculation
                default_days = 5
                if self.expiry_date:
                    T_years = self.analytics.get_time_to_expiry(str(self.expiry_date))
                    default_days = round(T_years * 365, 1)
                days = float(input(f"Enter Days to Expiry (Default {default_days}): ") or default_days)
                T = days / 365.0
                
                print(f"Fetching data for Strike {strike}...")
                data = self.get_option_chain_data()
                df = self.parse_and_filter(data)
                
                if df.empty:
                    print("No data found. Checking Symbol validity...")
                    # Fallback: Try fetching quotes directly if symbol is known logic
                    # But for now, just return
                    return
                
                # Filter for strike
                row = df[df['strike'] == strike]
                if row.empty:
                    print(f"Strike {strike} not found in chain.")
                    return
                
                # Process Both Legs
                for o_type in ['CE', 'PE']:
                    target = row[row['type'] == o_type]
                    if not target.empty:
                        price = float(target['price'].values[0])
                        iv = float(target['iv'].values[0])
                        if iv <= 0: iv = float(input(f"Enter IV for {o_type}: "))
                        
                        greeks = self.analytics.calculate_greeks(ref_spot, strike, T, 0.10, iv/100.0, o_type)
                        
                        legs[o_type]['price'] = price
                        legs[o_type]['iv'] = iv
                        legs[o_type]['delta'] = greeks['delta']
                        legs[o_type]['gamma'] = greeks['gamma']
                        legs[o_type]['theta'] = greeks['theta']
                        legs[o_type]['vega'] = greeks['vega']
                    else:
                        print(f"Warning: {o_type} not found for strike {strike}")
                        
                print(f"Initialized Straddle at Strike {strike}")

            except Exception as e:
                print(f"Error fetching: {e}")
                return
        else:
            # Manual Input for Straddle? Or just simple single leg? 
            # User asked for Call/Put and Straddle. Let's ask for both.
            print("Enter baseline data (leave blank for defaults):")
            ref_spot = float(input(f"Ref Spot [{ref_spot}]: ") or ref_spot)
            
            for o_type in ['CE', 'PE']:
                print(f"--- {o_type} Leg ---")
                legs[o_type]['price'] = float(input(f"Price: ") or 100)
                legs[o_type]['delta'] = float(input(f"Delta: ") or (0.5 if o_type=='CE' else -0.5))
                legs[o_type]['gamma'] = float(input(f"Gamma: ") or 0.002)

        print("\n--- Monitoring Started (Ctrl+C to stop) ---")
        print(f"Ref Spot: {ref_spot:.2f}")
        print(f"Init: CE={legs['CE']['price']:.2f} (D={legs['CE']['delta']}) | PE={legs['PE']['price']:.2f} (D={legs['PE']['delta']})")
        print("-" * 155)
        # Header
        # Time | Spot | dS | CE Price (dPrice) | PE Price (dPrice) | Straddle (PnL)
        print(f"{'Time':<10} | {'Spot':<10} | {'dS':<8} | {'CE Theo':<10} {'(dCE)':<8} {'CE Delta':<8} | {'PE Theo':<10} {'(dPE)':<8} {'PE Delta':<8} | {'Straddle':<10} {'(PnL)':<8}")
        print("-" * 155)
        
        try:
            while True:
                # 1. Get Live Spot
                live_spot = ref_spot
                try:
                    # Silent quote fetch
                    # If we have Fyers object
                    if self.fyers:
                        data = {"symbols": self.symbol}
                        r = self.fyers.quotes(data=data)
                        if r.get('s') == 'ok':
                            live_spot = r['d'][0]['v']['lp']
                except: pass
                
                dS = live_spot - ref_spot
                
                # Calculate Dynamics for each leg
                # Taylor Series: dP = Delta*dS + 0.5*Gamma*dS^2
                # Update Delta: NewDelta = OldDelta + Gamma*dS
                
                ce_dP = (legs['CE']['delta'] * dS) + (0.5 * legs['CE']['gamma'] * (dS**2))
                pe_dP = (legs['PE']['delta'] * dS) + (0.5 * legs['PE']['gamma'] * (dS**2))
                
                ce_new_price = legs['CE']['price'] + ce_dP
                pe_new_price = legs['PE']['price'] + pe_dP
                
                ce_new_delta = legs['CE']['delta'] + (legs['CE']['gamma'] * dS)
                pe_new_delta = legs['PE']['delta'] + (legs['PE']['gamma'] * dS)
                
                straddle_price = ce_new_price + pe_new_price
                straddle_pnl = ce_dP + pe_dP
                
                now = datetime.now().strftime("%H:%M:%S")
                
                print(f"{now:<10} | {live_spot:<10.2f} | {dS:<8.2f} | {ce_new_price:<10.2f} {ce_dP:<8.2f} {ce_new_delta:<8.2f} | {pe_new_price:<10.2f} {pe_dP:<8.2f} {pe_new_delta:<8.2f} | {straddle_price:<10.2f} {straddle_pnl:<8.2f}", end='\r')
                
                time.sleep(1)
                
        except KeyboardInterrupt:
            print("\nStopped.")

    def what_if_analysis(self):
        print("\n--- What-If Analysis ---")
        if self.spot_price <= 0:
            print("Spot Price is invalid (0). Using default 24000.")
            current_spot = 24000.0
        else:
            current_spot = self.spot_price

        try:
            strike = float(input("Enter Strike Price: "))
            
            # Default to stored expiry T if possible
            default_days = 5
            if self.expiry_date:
                T_years = self.analytics.get_time_to_expiry(str(self.expiry_date))
                default_days = round(T_years * 365, 1)
                
            days_input = input(f"Enter Days to Expiry (Default {default_days}): ")
            days = float(days_input) if days_input.strip() else default_days
            
            # Target Spot Input
            target_spot_input = input(f"Enter Target Spot Price (Default {current_spot}): ")
            target_spot = float(target_spot_input) if target_spot_input.strip() else current_spot

            print(f"\nSimulating for Strike {strike} @ Target Spot {target_spot}...")
            print(f"{'New IV':<10} | {'Call Price':<12} | {'Put Price':<12} | {'Straddle':<12}")
            print("-" * 55)
            
            center_iv = float(input("Enter Center IV (e.g. 15): "))
            
            for iv in range(int(center_iv)-5, int(center_iv)+6):
                if iv <= 0: continue 
                
                T = days / 365.0
                sigma = iv / 100.0
                r = 0.10
                
                ce = self.analytics.black_scholes(target_spot, strike, T, r, sigma, 'CE')
                pe = self.analytics.black_scholes(target_spot, strike, T, r, sigma, 'PE')
                
                print(f"{iv:<10} | {ce:<12.2f} | {pe:<12.2f} | {(ce+pe):<12.2f}")
                
            input("\nPress Enter to continue...")
            
        except ValueError:
            print("Invalid input.")

    def greeks_scenario_analysis(self):
        print("\n--- Greeks Scenario Analysis ---")
        if self.spot_price <= 0:
             # Try fetching if not set
             self.get_spot_price()
             
        current_spot = self.spot_price if self.spot_price > 0 else 24000.0

        try:
            strike = float(input("Enter Strike Price: "))
            
            default_days = 5
            if self.expiry_date:
                T_years = self.analytics.get_time_to_expiry(str(self.expiry_date))
                default_days = round(T_years * 365, 1)
                
            days = float(input(f"Enter Days to Expiry (Default {default_days}): ") or default_days)
            center_iv = float(input("Enter Benchmark IV (e.g. 15): ") or 15)
            
            print(f"\nStarting Real-Time Greek Scenario Monitor (Press Ctrl+C to Stop)...")
            time.sleep(1)
            
            while True:
                # 1. Update Spot
                self.get_spot_price(verbose=False)
                current_spot = self.spot_price if self.spot_price > 0 else 24000.0
                
                # Spot Range: +/- 300 pts, Step 25
                step = 25
                start_spot = current_spot - 300
                steps_count = int(600 / step) + 1
                
                print("\033[H\033[J") # ANSI Clear Screen
                print(f"--- Greeks Scenario Monitor (Live) ---")
                print(f"Symbol: {self.symbol} | Live Spot: {current_spot:.2f} | Strike: {strike} | IV: {center_iv}%")
                
                print("-" * 140)
                print(f"{'Spot':<8} | {'Call':<8} {'Delta':<6} {'Gamma':<8} | {'Put':<8} {'Delta':<6} {'Gamma':<8} | {'Straddle':<10} {'Theta(S)':<8}")
                print("-" * 140)
                
                for i in range(steps_count):
                    sim_spot = start_spot + (i * step)
                    
                    T = days / 365.0
                    sigma = center_iv / 100.0
                    r = 0.10
                    
                    # CALL
                    c_price = self.analytics.black_scholes(sim_spot, strike, T, r, sigma, 'CE')
                    c_g = self.analytics.calculate_greeks(sim_spot, strike, T, r, sigma, 'CE')
                    
                    # PUT
                    p_price = self.analytics.black_scholes(sim_spot, strike, T, r, sigma, 'PE')
                    p_g = self.analytics.calculate_greeks(sim_spot, strike, T, r, sigma, 'PE')
                    
                    # STRADDLE
                    straddle_price = c_price + p_price
                    straddle_theta = c_g['theta'] + p_g['theta']
                    
                    # Highlight current spot range
                    mk = ""
                    if abs(sim_spot - current_spot) < 12.5: mk = " <--- LIVE"
                    
                    c_str = f"{c_price:<8.2f} {c_g['delta']:<6.2f} {c_g['gamma']:<8.4f}"
                    p_str = f"{p_price:<8.2f} {p_g['delta']:<6.2f} {p_g['gamma']:<8.4f}"
                    s_str = f"{straddle_price:<10.2f} {straddle_theta:<8.2f} {mk}"
                    
                    print(f"{sim_spot:<8.2f} | {c_str} | {p_str} | {s_str}")
                    
                print("-" * 140)
                print("Press Ctrl+C to Exit...")
                time.sleep(3)

        except KeyboardInterrupt:
            print("\nMonitor Stopped.")
        except ValueError as e:
            print(f"Invalid Input: {e}")

    def select_symbol_ui(self):
        print("\n--- Select Symbol ---")
        print("1. Index")
        print("2. Stock (Common F&O)")
        print("3. Manual Entry")
        
        c = input("Select Type: ")
        
        if c == '1':
            print("\n--- Indices ---")
            indices = [
                "NSE:NIFTY50-INDEX", 
                "NSE:NIFTYBANK-INDEX", 
                "NSE:FINNIFTY-INDEX", 
                "NSE:MIDCPNIFTY-INDEX"
            ]
            for i, idx in enumerate(indices):
                print(f"{i+1}. {idx}")
            try:
                sel = int(input("Select Index (Number): ")) - 1
                if 0 <= sel < len(indices):
                    self.symbol = indices[sel]
                    print(f"Selected: {self.symbol}")
            except: print("Invalid selection.")
            
        elif c == '2':
            print("\n--- Common Stocks ---")
            stocks = [
                "NSE:RELIANCE-EQ", "NSE:HDFCBANK-EQ", "NSE:INFY-EQ", "NSE:TCS-EQ", 
                "NSE:ICICIBANK-EQ", "NSE:SBIN-EQ", "NSE:TATAMOTORS-EQ", "NSE:MARUTI-EQ"
            ]
            for i, stk in enumerate(stocks):
                print(f"{i+1}. {stk}")
            try:
                sel = int(input("Select Stock (Number): ")) - 1
                if 0 <= sel < len(stocks):
                    self.symbol = stocks[sel]
                    print(f"Selected: {self.symbol}")
            except: print("Invalid selection.")
            
        else:
            self.symbol = input("Enter Symbol (e.g., NSE:NIFTY50-INDEX): ")

    def select_strikes_from_chain_ui(self):
        print("\n--- Select Strikes from Chain ---")
        print("Fetching Chain...")
        data = self.get_option_chain_data()
        df = self.parse_and_filter(data) # This usually filters by selected, need to clear selected first?
        # Actually parse_and_filter filters if selected_strikes is SET.
        # We want to see ALL strikes first.
        
        # Temp clear filters to get full view
        backup_sel = self.selected_strikes
        self.selected_strikes = []
        df = self.parse_and_filter(data)
        
        if df.empty:
            print("No data found for symbol/expiry.")
            self.selected_strikes = backup_sel
            return

        # Prepare UI Table
        # We want to show: ID | Strike | CE LTP | PE LTP
        # DF has mixed types. Pivot?
        # Let's iterate unique strikes
        unique_strikes = sorted(df['strike'].unique())
        
        # Find ATM info for context
        spot = self.spot_price
        if spot == 0: self.get_spot_price(); spot = self.spot_price;
        
        print(f"\nSymbol: {self.symbol} | Spot: {spot}")
        print("-" * 50)
        print(f"{'ID':<4} | {'Strike':<10} | {'CE LTP':<10} | {'PE LTP':<10}")
        print("-" * 50)
        
        display_map = {}
        
        for i, s in enumerate(unique_strikes):
            ce_row = df[(df['strike'] == s) & (df['type'] == 'CE')]
            pe_row = df[(df['strike'] == s) & (df['type'] == 'PE')]
            
            ce_ltp = ce_row.iloc[0]['price'] if not ce_row.empty else 0
            pe_ltp = pe_row.iloc[0]['price'] if not pe_row.empty else 0
            
            mk = ""
            if abs(s - spot) < spot * 0.002: mk = " <-- ATM"
            
            print(f"{i+1:<4} | {s:<10} | {ce_ltp:<10} | {pe_ltp:<10}{mk}")
            display_map[i+1] = s
            
        print("-" * 50)
        print("Enter IDs to select (comma separated, e.g. 10,11,12)")
        print("Or enter Range (e.g. 10-15)")
        
        sel_input = input("Selection: ")
        new_strikes = []
        
        try:
            parts = sel_input.split(',')
            for p in parts:
                if '-' in p:
                    start, end = map(int, p.split('-'))
                    for x in range(start, end + 1):
                        if x in display_map: new_strikes.append(display_map[x])
                else:
                    idx = int(p.strip())
                    if idx in display_map: new_strikes.append(display_map[idx])
            
            if new_strikes:
                self.selected_strikes = sorted(list(set(new_strikes)))
                print(f"Updated Strikes: {self.selected_strikes}")
            else:
                print("No valid selection made.")
                self.selected_strikes = backup_sel # Restore
                
        except Exception as e:
            print(f"Selection Error: {e}")
            self.selected_strikes = backup_sel # Restore

    def run(self):
        while True:
            print("\n1. Set Symbol (Current: {})".format(self.symbol))
            print("2. Set Strikes (Current: {})".format(self.selected_strikes))
            print("3. Real-Time Monitor (Grid-Based)")
            print("4. Price What-If Analysis")
            print("5. Greeks Scenario Analysis")
            print("6. Set Expiry Date (Current: {})".format(self.expiry_date))
            print("7. Exit")
            
            choice = input("Select option: ")
            
            if choice == '1':
                self.select_symbol_ui()
            elif choice == '2':
                # Ask: Manual or Menu?
                print("1. Manual Entry (Comma separated)")
                print("2. Select from Option Chain (Table)")
                sub = input("Select: ")
                if sub == '2':
                    self.select_strikes_from_chain_ui()
                else:
                    s = input("Enter Strikes (comma separated): ")
                    try:
                        self.selected_strikes = [float(x.strip()) for x in s.split(',')]
                    except:
                        print("Invalid format.")
            elif choice == '3':
                self.real_time_greek_monitor()
            elif choice == '4':
                if self.spot_price == 0:
                    self.get_spot_price()
                self.what_if_analysis()
            elif choice == '5':
                self.greeks_scenario_analysis()
            elif choice == '6':
                d = input("Enter Expiry Date (YYYY-MM-DD): ")
                try:
                    # Validate format
                    datetime.strptime(d, "%Y-%m-%d")
                    self.expiry_date = d
                except ValueError:
                    print("Invalid Date Format. Use YYYY-MM-DD")
            elif choice == '7':
                break

if __name__ == "__main__":
    app = InteractiveOptionsAnalyzer()
    app.run()
