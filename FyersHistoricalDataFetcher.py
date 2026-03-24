import pandas as pd
from datetime import datetime, timedelta
import os

class FyersHistoricalDataFetcher:
    def __init__(self, fyers):
        self.fyers = fyers
    
    def search_instrument(self, search_term):
        """Search for instruments by name or symbol (Manual Mapping for now)"""
        print(f"\nSearching for: {search_term}")
        
        # Common instruments mapping for Fyers
        instruments = {
            'NIFTY': 'NSE:NIFTY50-INDEX',
            'BANKNIFTY': 'NSE:NIFTYBANK-INDEX',
            'FINNIFTY': 'NSE:FINNIFTY-INDEX',
            'SENSEX': 'BSE:SENSEX-INDEX',
            'RELIANCE': 'NSE:RELIANCE-EQ',
            'TCS': 'NSE:TCS-EQ',
            'INFY': 'NSE:INFY-EQ',
            'HDFC': 'NSE:HDFC-EQ',
            'HDFCBANK': 'NSE:HDFCBANK-EQ',
            'SBIN': 'NSE:SBIN-EQ',
        }
        
        search_upper = search_term.upper()
        if search_upper in instruments:
            return instruments[search_upper]
        else:
            print(f"Instrument '{search_term}' not found in common list.")
            print("Please provide Fyers symbol manually (e.g., NSE:RELIANCE-EQ).")
            return None
    
    def get_historical_data(self, symbol, from_date, to_date, resolution="D"):
        """
        Fetch historical data from Fyers API
        resolution: "D", "1", "5", "15", "60", etc.
        """
        
        data = {
            "symbol": symbol,
            "resolution": resolution,
            "date_format": "1",
            "range_from": from_date,
            "range_to": to_date,
            "cont_flag": "1"
        }

        print(f"Fetching data for {symbol} ({resolution}) from {from_date} to {to_date}...")
        
        try:
            response = self.fyers.history(data=data)
            
            if response['s'] == "ok":
                candles = response['candles']
                return candles
            else:
                print(f"Error: {response['message']}")
                return None
                
        except Exception as e:
            print(f"Exception occurred: {str(e)}")
            return None

    def process_to_dataframe(self, candles):
        """Convert raw candles to pandas DataFrame"""
        if not candles:
            return None
        
        # Fyers returns [timestamp, open, high, low, close, volume]
        df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        
        # Convert timestamp (Fyers uses epoch)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
        
        # Adjust timezone if needed (Fyers is usually IST but returns UTC timestamp in epoch? No, usually local)
        # Let's assume it's correct for now.
        
        return df
    
    def save_to_csv(self, df, filename=None):
        """Save DataFrame to CSV file"""
        if df is None or df.empty:
            print("No data to save")
            return
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"historical_data_{timestamp}.csv"
        
        # Create output directory if it doesn't exist
        os.makedirs('historical_data', exist_ok=True)
        filepath = os.path.join('historical_data', filename)
        
        df.to_csv(filepath, index=False)
        print(f"\n✓ Data saved to: {filepath}")
        print(f"✓ Total rows: {len(df)}")
        return filepath
