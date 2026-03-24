"""
KeyLevelsEngine.py - Options-Based Key Levels for NIFTY

Provides:
1. Max Pain - Strike where option buyers lose the most
2. PCR (Put-Call Ratio) - Sentiment indicator
3. OI Walls - High OI strikes as Support/Resistance
4. Net GEX - Dealer Gamma Exposure (Long/Short Gamma regime)

Usage:
    python KeyLevelsEngine.py
"""

import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from FyersAuth import FyersAuthenticator
from OptionAnalytics import OptionAnalytics


class KeyLevelsEngine:
    def __init__(self):
        self.fyers = self._authenticate()
        self.analytics = OptionAnalytics()
        self.symbol = "NSE:NIFTY50-INDEX"
        self.spot_price = 0
        self.expiry_date = None
        self.chain_df = None
        
    def _authenticate(self):
        print("Authenticating with Fyers...")
        try:
            auth = FyersAuthenticator("QUTT4YYMIG-100", "ZG0WN2NL1B", "http://127.0.0.1:3000/callback")
            fyers = auth.get_fyers_instance()
            if fyers:
                print("Authentication Successful.")
                return fyers
        except Exception as e:
            print(f"Auth Failed: {e}")
        return None
    
    def get_spot_price(self):
        try:
            r = self.fyers.quotes({"symbols": self.symbol})
            if r.get('s') == 'ok':
                self.spot_price = r['d'][0]['v']['lp']
                return self.spot_price
        except:
            pass
        return 0
    
    def get_option_chain(self, expiry_ts=""):
        """Fetch option chain from Fyers API"""
        try:
            data = {"symbol": self.symbol, "strikecount": 500, "timestamp": expiry_ts}
            r = self.fyers.optionchain(data=data)
            # print(f"Debug: Option Chain Response: {r.get('s')}") # Too verbose to print whole thing
            if r.get('s') == 'ok':
                return r.get('data', {})
            else:
                print(f"Debug: Option Chain API Error: {r}")
        except Exception as e:
            print(f"Error fetching chain: {e}")
        return None
    
    def parse_chain(self, data):
        """Parse option chain into structured DataFrame"""
        if not data:
            return pd.DataFrame()
        
        options = data.get('optionsChain', [])
        if not options:
            return pd.DataFrame()
        
        # Get expiry info
        expiry_list = data.get('expiryData', [])
        if expiry_list:
            expiry_list = sorted(expiry_list, key=lambda x: x.get('expiry', 0))
            self.expiry_date = expiry_list[0].get('date')
        
        records = []
        for item in options:
            strike = item.get('strike_price', 0)
            if strike <= 0: continue
            
            records.append({
                'strike': strike,
                'type': 'CE' if item.get('option_type') == 'CALL' else 'PE',
                'ltp': item.get('ltp', 0),
                'oi': item.get('oi', 0),
                'iv': item.get('iv', 0),
                'delta': item.get('delta', 0),
                'gamma': item.get('gamma', 0)
            })
        
        return pd.DataFrame(records)
    
    # ==================== ANALYSIS METHODS ====================
    
    def calculate_max_pain(self, df):
        """
        Max Pain = Strike where total loss for option buyers is maximized.
        
        For each strike K:
          - Call buyers lose if Spot < K (worthless)
          - Put buyers lose if Spot > K (worthless)
        We calculate total "pain" (loss in premium) at each strike.
        """
        if df.empty:
            return 0
        
        strikes = df['strike'].unique()
        calls = df[df['type'] == 'CE'].set_index('strike')
        puts = df[df['type'] == 'PE'].set_index('strike')
        
        max_pain_strike = 0
        max_pain_value = float('inf')
        
        for K in strikes:
            total_pain = 0
            
            # Call Pain: For each call with strike <= K, it's ITM, buyer gains
            # For each call with strike > K, it's OTM, buyer loses premium
            for call_strike in calls.index:
                oi = calls.loc[call_strike, 'oi'] if call_strike in calls.index else 0
                if isinstance(oi, pd.Series): oi = oi.sum()
                if K > call_strike:
                    # Call is ITM, writer pays (K - call_strike) per contract
                    total_pain += oi * (K - call_strike)
                    
            # Put Pain: For each put with strike >= K, it's ITM, buyer gains
            for put_strike in puts.index:
                oi = puts.loc[put_strike, 'oi'] if put_strike in puts.index else 0
                if isinstance(oi, pd.Series): oi = oi.sum()
                if K < put_strike:
                    # Put is ITM, writer pays (put_strike - K) per contract
                    total_pain += oi * (put_strike - K)
            
            if total_pain < max_pain_value:
                max_pain_value = total_pain
                max_pain_strike = K
                
        return max_pain_strike
    
    def calculate_pcr(self, df):
        """
        Put-Call Ratio based on OI
        PCR > 1 = Bearish sentiment (more puts)
        PCR < 1 = Bullish sentiment (more calls)
        """
        if df.empty:
            return 0
        
        put_oi = df[df['type'] == 'PE']['oi'].sum()
        call_oi = df[df['type'] == 'CE']['oi'].sum()
        
        if call_oi == 0:
            return 0
        
        return put_oi / call_oi
    
    def calculate_oi_walls(self, df, spot, range_pct=0.03):
        """
        Immediate OI Walls = Strikes with highest Open Interest near spot.
        Call Wall (Resistance): Highest OI Call strike ABOVE spot (within range)
        Put Wall (Support): Highest OI Put strike BELOW spot (within range)
        Returns top-2 walls for each side.
        """
        if df.empty:
            return {'call_wall': 0, 'put_wall': 0, 'call_wall_2': 0, 'put_wall_2': 0}

        calls = df[df['type'] == 'CE']
        puts  = df[df['type'] == 'PE']

        # ── Call walls (above spot) ──
        calls_above = calls[(calls['strike'] > spot) & (calls['strike'] <= spot * (1 + range_pct))]
        if calls_above.empty:
            calls_above = calls[calls['strike'] > spot]
        if calls_above.empty:
            calls_above = calls

        call_wall = call_wall_2 = 0
        if not calls_above.empty and calls_above['oi'].sum() > 0:
            # Group by strike to aggregate OI (in case of duplicate rows)
            call_oi = calls_above.groupby('strike')['oi'].sum()
            top_calls = call_oi.nlargest(2)
            call_wall   = int(top_calls.index[0]) if len(top_calls) >= 1 else 0
            call_wall_2 = int(top_calls.index[1]) if len(top_calls) >= 2 else 0

        # ── Put walls (below spot) ──
        puts_below = puts[(puts['strike'] < spot) & (puts['strike'] >= spot * (1 - range_pct))]
        if puts_below.empty:
            puts_below = puts[puts['strike'] < spot]
        if puts_below.empty:
            puts_below = puts

        put_wall = put_wall_2 = 0
        if not puts_below.empty and puts_below['oi'].sum() > 0:
            put_oi = puts_below.groupby('strike')['oi'].sum()
            top_puts = put_oi.nlargest(2)
            put_wall   = int(top_puts.index[0]) if len(top_puts) >= 1 else 0
            put_wall_2 = int(top_puts.index[1]) if len(top_puts) >= 2 else 0

        return {
            'call_wall':   call_wall,
            'call_wall_2': call_wall_2,
            'put_wall':    put_wall,
            'put_wall_2':  put_wall_2,
        }
    
    def calculate_net_gex(self, df, spot):
        """
        Net Gamma Exposure (GEX)
        
        Dealer Position Assumption:
        - Dealers are NET SHORT options (customers buy, dealers sell)
        - Short Call = Negative Gamma (dealer accelerates losses on up-move)
        - Short Put = Negative Gamma (dealer accelerates losses on down-move)
        
        WAIT - This is where the confusion happens. Let me clarify:
        
        Gamma is ALWAYS positive for both calls and puts.
        
        Dealer GEX Sign:
        - If dealers are SHORT Calls: They have NEGATIVE Gamma exposure
        - If dealers are SHORT Puts: They have POSITIVE Gamma exposure
          (Because short put = long stock as price goes down, which is stabilizing)
        
        Standard SpotGamma Convention:
          GEX = Sum[ (Call Gamma * Call OI) - (Put Gamma * Put OI) ] * Spot * 100
        
        If GEX > 0: Dealers need to SELL when price rises, BUY when price falls (Stabilizing)
        If GEX < 0: Dealers need to BUY when price rises, SELL when price falls (Destabilizing)
        """
        if df.empty:
            return 0
        
        T = 0.02  # Approx time to expiry for gamma calc
        if self.expiry_date:
            T = self.analytics.get_time_to_expiry(self.expiry_date)
            if T < 0.001: T = 0.001
        
        net_gex = 0
        
        for _, row in df.iterrows():
            strike = row['strike']
            oi = row['oi']
            o_type = row['type']
            iv = row['iv'] / 100 if row['iv'] > 1 else row['iv']
            if iv < 0.01: iv = 0.15
            
            # Calculate gamma using BSM
            greeks = self.analytics.calculate_greeks(spot, strike, T, 0.10, iv, o_type)
            gamma = greeks.get('gamma', 0)
            
            # GEX contribution (per SpotGamma convention)
            # Multiply by spot to get dollar gamma, by 100 for contract size
            contribution = gamma * oi * spot * 100
            
            if o_type == 'CE':
                net_gex += contribution  # Calls add to GEX
            else:
                net_gex -= contribution  # Puts subtract from GEX
        
        return net_gex
    
    def get_gamma_regime(self, gex):
        """Interpret Net GEX into regime"""
        # Scale is arbitrary, we use sign + magnitude
        if gex > 1e9:
            return "STRONG LONG GAMMA (Very Sticky)"
        elif gex > 0:
            return "LONG GAMMA (Stabilizing)"
        elif gex > -1e9:
            return "SHORT GAMMA (Volatile)"
        else:
            return "STRONG SHORT GAMMA (Very Slippery)"
    
    # ==================== MAIN ANALYSIS ====================
    
    def analyze(self):
        """Run full analysis and return results dict"""
        spot = self.get_spot_price()
        if spot == 0:
            print("Failed to get spot price.")
            return {}
        
        print(f"Spot: {spot}")
        
        data = self.get_option_chain()
        df = self.parse_chain(data)
        
        if df.empty:
            print("Failed to get option chain.")
            return {}
        
        self.chain_df = df
        
        # Calculate all metrics
        max_pain = self.calculate_max_pain(df)
        pcr = self.calculate_pcr(df)
        walls = self.calculate_oi_walls(df, spot)
        net_gex = self.calculate_net_gex(df, spot)
        gamma_regime = self.get_gamma_regime(net_gex)
        
        results = {
            'spot': spot,
            'expiry': self.expiry_date,
            'max_pain': max_pain,
            'max_pain_distance': spot - max_pain,
            'pcr': pcr,
            'call_wall': walls['call_wall'],
            'put_wall': walls['put_wall'],
            'net_gex': net_gex,
            'gamma_regime': gamma_regime
        }
        
        return results
    
    def display_results(self, results):
        """Pretty print analysis results"""
        if not results:
            return
        
        print("\n" + "="*60)
        print(f"  KEY LEVELS ANALYSIS | Expiry: {results['expiry']}")
        print("="*60)
        
        print(f"\n>> SPOT:      {results['spot']:.2f}")
        print(f">> MAX PAIN:  {results['max_pain']} (Distance: {results['max_pain_distance']:+.0f} pts)")
        
        print(f"\n>> PCR (OI):  {results['pcr']:.2f}", end="")
        if results['pcr'] > 1.2:
            print("  [BEARISH SENTIMENT]")
        elif results['pcr'] < 0.8:
            print("  [BULLISH SENTIMENT]")
        else:
            print("  [NEUTRAL]")
        
        print(f"\n>> CALL WALL (Resistance): {results['call_wall']}")
        print(f">> PUT WALL (Support):     {results['put_wall']}")
        
        print(f"\n>> NET GEX:  {results['net_gex']:.2e}")
        print(f">> REGIME:   [{results['gamma_regime']}]")
        
        print("\n" + "="*60)
    
    def run(self):
        """Main entry point"""
        results = self.analyze()
        self.display_results(results)
        return results


if __name__ == "__main__":
    engine = KeyLevelsEngine()
    engine.run()
