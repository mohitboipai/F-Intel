import os
import sys
import datetime
import pandas as pd
import time

# Add root directory to path to find FyersAuth
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

try:
    from FyersAuth import FyersAuthenticator
except ImportError:
    print("Warning: FyersAuth not found in root. Please ensure FyersAuth.py exists.")

class SpotDataManager:
    def __init__(self, symbol="NSE:NIFTY50-INDEX", timeframe="D"):
        self.symbol = symbol
        self.timeframe = timeframe # "D" for Daily, "1" for 1-minute
        self.fyers = None
        self.app_id = "QUTT4YYMIG-100"
        self.secret_id = "ZG0WN2NL1B"
        self.redirect_uri = "http://127.0.0.1:3000/callback"
        self.authenticate()

    def authenticate(self):
        try:
            auth = FyersAuthenticator(self.app_id, self.secret_id, self.redirect_uri)
            self.fyers = auth.get_fyers_instance()
            print("[SpotData] Authentication Successful.")
        except Exception as e:
            print(f"[SpotData] Auth Failed: {e}")

    def fetch_history(self, days=365*2):
        """
        Fetches historical data.
        If timeframe is 'D', fetches Daily candles.
        """
        print(f"[SpotData] Fetching {days} days of history for {self.symbol}...")
        today = datetime.datetime.now()
        start_date = today - datetime.timedelta(days=days)
        current_start = start_date
        all_candles = []

        # Fyers history API allows max 100 days range for some resolutions, safe to loop
        while current_start < today:
            current_end = current_start + datetime.timedelta(days=90)
            if current_end > today: current_end = today
            
            p = {
                "symbol": self.symbol, 
                "resolution": self.timeframe, 
                "date_format": "1",
                "range_from": current_start.strftime("%Y-%m-%d"),
                "range_to": current_end.strftime("%Y-%m-%d"), 
                "cont_flag": "1"
            }
            try:
                r = self.fyers.history(data=p)
                if r.get('s') == 'ok':
                    candles = r['candles']
                    # Check if date is strictly greater to avoid duplicates at boundary
                    if all_candles:
                        last_ts = all_candles[-1][0]
                        candles = [c for c in candles if c[0] > last_ts]
                    
                    all_candles.extend(candles)
                    print(f"  > Got {len(candles)} candles ({current_start.date()} to {current_end.date()})")
                else:
                    print(f"  > Error fetching {current_start.date()}: {r}")
            except Exception as e:
                 print(f"  > Exception: {e}")
            
            current_start = current_end
            time.sleep(0.1)

        if not all_candles:
            return pd.DataFrame()
        
        cols = ['ts', 'open', 'high', 'low', 'close', 'volume']
        df = pd.DataFrame(all_candles, columns=cols)
        
        # Convert Timestamp
        # Fyers Daily timestamp usually 00:00 UTC or IST?
        # Usually it's epoch.
        df['datetime'] = pd.to_datetime(df['ts'], unit='s').dt.tz_localize('UTC').dt.tz_convert('Asia/Kolkata')
        df = df.set_index('datetime').sort_index()
        
        # Clean
        df = df[~df.index.duplicated(keep='first')]
        print(f"[SpotData] Loaded {len(df)} candles.")
        return df[['open', 'high', 'low', 'close', 'volume']]

if __name__ == "__main__":
    dm = SpotDataManager()
    df = dm.fetch_history(days=1000) # Fetch ~3 years
    print(df.tail())
