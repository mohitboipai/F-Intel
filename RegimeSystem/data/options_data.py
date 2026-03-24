from .spot_data import SpotDataManager

class OptionsDataManager(SpotDataManager):
    def __init__(self):
        # Initialize with India VIX symbol
        super().__init__(symbol="NSE:INDIAVIX-INDEX", timeframe="D")
    
    def fetch_iv_history(self, days=365*2):
        """
        Fetches India VIX history. 
        Returns DataFrame with 'close' which represents the VIX (IV %).
        """
        print("[OptionsData] Fetching India VIX history...")
        df = self.fetch_history(days=days)
        if df.empty:
            return df
        
        # VIX is already annualized vol %, so we can use 'close' as 'iv'
        df = df.rename(columns={'close': 'iv', 'open': 'iv_open', 'high': 'iv_high', 'low': 'iv_low'})
        return df[['iv', 'iv_open', 'iv_high', 'iv_low']]

if __name__ == "__main__":
    odm = OptionsDataManager()
    df = odm.fetch_iv_history(days=100)
    print(df.tail())
