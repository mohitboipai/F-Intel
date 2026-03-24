"""
TradingDashboard.py - Unified Analysis Dashboard for NIFTY

Combines:
1. Volatility Analysis (IV/HV, VRP, Vol Regime)
2. Key Levels (Max Pain, PCR, OI Walls, GEX)
3. Price Action (Trend, Key SMAs)

Outputs a consolidated briefing for trading decisions.

Usage:
    python TradingDashboard.py
"""

import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import time

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from FyersAuth import FyersAuthenticator
from OptionAnalytics import OptionAnalytics
from KeyLevelsEngine import KeyLevelsEngine
from AdvancedVolatilityScanner import AdvancedVolatilityScanner


class TradingDashboard:
    def __init__(self):
        self.fyers = self._authenticate()
        self.analytics = OptionAnalytics()
        self.key_levels = KeyLevelsEngine()
        self.key_levels.fyers = self.fyers  # Share auth
        self.scanner = AdvancedVolatilityScanner(self.fyers) # Share auth
        
        self.symbol = "NSE:NIFTY50-INDEX"
        self.spot_price = 0
        
    def _authenticate(self):
        print("Authenticating...")
        try:
            auth = FyersAuthenticator("QUTT4YYMIG-100", "ZG0WN2NL1B", "http://127.0.0.1:3000/callback")
            fyers = auth.get_fyers_instance()
            if fyers:
                print("OK")
                return fyers
        except Exception as e:
            print(f"Auth Failed: {e}")
        return None
    
    # ==================== DATA FETCHING ====================
    
    def get_spot_price(self):
        try:
            r = self.fyers.quotes({"symbols": self.symbol})
            if r.get('s') == 'ok':
                self.spot_price = r['d'][0]['v']['lp']
                print(f"Debug: Spot fetched: {self.spot_price}")
                return self.spot_price
            else:
                print(f"Error fetching spot: {r}")
        except Exception as e:
            print(f"Exception fetching spot: {e}")
        return 0
    
    def get_historical_data(self, days=365):
        """Fetch daily OHLC for volatility calculations"""
        today = datetime.now()
        start = today - timedelta(days=days)
        
        p = {
            "symbol": self.symbol,
            "resolution": "D",
            "date_format": "1",
            "range_from": start.strftime("%Y-%m-%d"),
            "range_to": today.strftime("%Y-%m-%d"),
            "cont_flag": "1"
        }
        
        try:
            r = self.fyers.history(data=p)
            if r.get('s') == 'ok':
                df = pd.DataFrame(r['candles'], columns=['ts', 'o', 'h', 'l', 'c', 'v'])
                df['date'] = pd.to_datetime(df['ts'], unit='s')
                df = df.set_index('date')
                df = df.rename(columns={'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close', 'v': 'volume'})
                return df
        except Exception as e:
            print(f"Error fetching history: {e}")
        return pd.DataFrame()
    
    def get_atm_iv(self):
        """Get ATM IV from option chain"""
        data = self.key_levels.get_option_chain()
        if not data:
            return 0
        
        df = self.key_levels.parse_chain(data)
        if df.empty:
            return 0
        
        # Find ATM strike (closest to spot)
        spot = self.spot_price
        df['dist'] = abs(df['strike'] - spot)
        atm_strike = df.loc[df['dist'].idxmin(), 'strike']
        
        # Average IV of ATM Call and Put (Filter out 0s)
        atm_options = df[df['strike'] == atm_strike]
        print(f"Debug: ATM Strike {atm_strike}. IVs: {atm_options['iv'].tolist()}")
        valid_ivs = atm_options[atm_options['iv'] > 0]['iv']
        
        if valid_ivs.empty:
            print("Debug: All ATM IVs are 0.")
            return 0
            
        atm_iv = valid_ivs.mean()
        print(f"Debug: Calculated ATM IV: {atm_iv}")
        return atm_iv
    
    # ==================== ANALYSIS ====================
    
    def analyze_volatility(self, df):
        """Calculate HV, IV, VRP"""
        if df.empty or len(df) < 30:
            return {}
        
        # Historical Volatility (20-day)
        hv_20 = self.analytics.calculate_historical_volatility(df['close'], window=20)
        
        # HV Statistics (1 year)
        hv_series = self.analytics.calculate_rolling_historical_volatility(df['close'], window=20)
        hv_series = hv_series.dropna()
        
        hv_mean = hv_series.mean() * 100
        hv_min = hv_series.min() * 100
        hv_max = hv_series.max() * 100
        hv_current = hv_20 * 100
        
        # HV Percentile
        hv_percentile = (hv_series < hv_20).sum() / len(hv_series) * 100
        
        # ATM IV
        atm_iv = self.get_atm_iv()
        
        # VRP (Volatility Risk Premium) = IV - HV
        vrp = atm_iv - hv_current if atm_iv > 0 else 0
        
        # IV Z-Score
        iv_zscore = 0
        if hv_series.std() > 0:
            iv_zscore = (atm_iv/100 - hv_series.mean()) / hv_series.std()
        
        return {
            'hv_20': hv_current,
            'hv_mean': hv_mean,
            'hv_min': hv_min,
            'hv_max': hv_max,
            'hv_percentile': hv_percentile,
            'atm_iv': atm_iv,
            'vrp': vrp,
            'iv_zscore': iv_zscore
        }
    
    def analyze_trend(self, df):
        """Simple trend analysis"""
        if df.empty or len(df) < 50:
            return {}
        
        close = df['close'].iloc[-1]
        sma_20 = df['close'].rolling(20).mean().iloc[-1]
        sma_50 = df['close'].rolling(50).mean().iloc[-1]
        
        trend = "NEUTRAL"
        if close > sma_20 and sma_20 > sma_50:
            trend = "BULLISH"
        elif close < sma_20 and sma_20 < sma_50:
            trend = "BEARISH"
        
        # Day change
        prev_close = df['close'].iloc[-2] if len(df) > 1 else close
        day_change = (close - prev_close) / prev_close * 100
        
        return {
            'sma_20': sma_20,
            'sma_50': sma_50,
            'trend': trend,
            'day_change': day_change
        }
    
    # ==================== STRATEGY ENGINE ====================

    def get_expected_move(self, spot, iv):
        """Calculate daily expected move based on IV"""
        if spot == 0 or iv == 0: return 0
        # Daily Vol = Annual Vol / sqrt(252)
        daily_vol = (iv / 100) / np.sqrt(252)
        expected_move = spot * daily_vol
        return expected_move

    def recommend_strategy(self, trend, vol_data, key_levels):
        """
        Rule-based strategy recommender.
        Returns: Strategy Name, Entry, Stop, Target
        """
        strategy = "WAIT / CASH"
        details = "No clear setup. Capital preservation mode."
        
        spot = self.spot_price
        if spot == 0: return strategy, details
        
        iv_percentile = vol_data.get('hv_percentile', 50)
        vrp = vol_data.get('vrp', 0)
        market_trend = trend.get('trend', 'NEUTRAL')
        
        # Determine Vol Regime
        is_high_vol = (iv_percentile > 70) or (vrp > 2)
        is_low_vol = (iv_percentile < 30) or (vrp < -1)
        
        # 1. NEUTRAL MARKET (Rangebound)
        if market_trend == "NEUTRAL":
            if is_high_vol:
                # Sell Premium (Short Strangle / Iron Condor)
                call_strike = int(np.ceil(key_levels['call_wall'] / 50) * 50)
                put_strike = int(np.floor(key_levels['put_wall'] / 50) * 50)
                strategy = "SHORT STRANGLE (Neutral + High Vol)"
                details = f"SELL {call_strike} CE\nSELL {put_strike} PE"
            elif is_low_vol:
                # Buy Premium (Long Straddle / Calendar) - Expect Explosion
                strategy = "LONG STRADDLE (Vol Expansion Setup)"
                details = f"BUY {int(spot/50)*50} CE\nBUY {int(spot/50)*50} PE"
            else:
                strategy = "IRON BUTTERFLY (Range Play)"
                center = int(spot/50)*50
                details = f"SELL {center} CE/PE\nBUY Wings (+/- 200 pts)"
                
        # 2. BULLISH MARKET
        elif market_trend == "BULLISH":
            if is_high_vol:
                # Bull Put Spread (Credit)
                sell_strike = int(np.floor(spot / 50) * 50)
                buy_strike = sell_strike - 100
                strategy = "BULL PUT SPREAD (Credit)"
                details = f"SELL {sell_strike} PE\nBUY {buy_strike} PE"
            else:
                # Bull Call Spread (Debit) or Long Call
                buy_strike = int(np.ceil(spot / 50) * 50)
                sell_strike = buy_strike + 200
                strategy = "BULL CALL SPREAD (Debit)"
                details = f"BUY {buy_strike} CE\nSELL {sell_strike} CE"

        # 3. BEARISH MARKET
        elif market_trend == "BEARISH":
            if is_high_vol:
                # Bear Call Spread (Credit)
                sell_strike = int(np.ceil(spot / 50) * 50)
                buy_strike = sell_strike + 100
                strategy = "BEAR CALL SPREAD (Credit)"
                details = f"SELL {sell_strike} CE\nBUY {buy_strike} CE"
            else:
                # Bear Put Spread (Debit) or Long Put
                buy_strike = int(np.floor(spot / 50) * 50)
                sell_strike = buy_strike - 200
                strategy = "BEAR PUT SPREAD (Debit)"
                details = f"BUY {buy_strike} PE\nSELL {sell_strike} PE"
                
        return strategy, details

    # ==================== DISPLAY ====================
    
    def display_dashboard(self, vol_data, trend_data, key_data):
        """Print consolidated dashboard"""
        # os.system('cls' if os.name == 'nt' else 'clear')
        
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Calculate derived metrics
        em = self.get_expected_move(self.spot_price, vol_data.get('atm_iv', 0))
        strategy, strategy_details = self.recommend_strategy(trend_data, vol_data, key_data)
        
        print("="*70)
        print(f"  NIFTY TRADING DASHBOARD | {now}")
        print("="*70)
        
        # === EXECUTIVE SUMMARY ===
        print(f"\n>> MARKET BIAS:   [{trend_data.get('trend', 'NEUTRAL')}]")
        print(f">> EXPECTED MOVE: +/- {em:.0f} pts  (Range: {self.spot_price - em:.0f} - {self.spot_price + em:.0f})")
        print(f">> GEX REGIME:    [{key_data.get('gamma_regime', 'UNKNOWN')}]")
        
        # === STRATEGY RECOMMENDATION ===
        print(f"\n{'─'*25} RECOMMENDED STRATEGY {'─'*25}")
        print(f"SETUP:  {strategy}")
        print(f"LEGS:   ") 
        for line in strategy_details.split('\n'):
            print(f"  • {line}")
            
        # === EXECUTION ZONES ===
        print(f"\n{'─'*25} EXECUTION ZONES {'─'*25}")
        cw = key_data.get('call_wall', 0)
        pw = key_data.get('put_wall', 0)
        print(f"SELL ZONE (Res):  {cw - 20} - {cw + 20}  (Call Wall)")
        print(f"BUY ZONE (Sup):   {pw - 20} - {pw + 20}  (Put Wall)")
        if key_data:
            print(f"MAX PAIN:         {key_data['max_pain']}  (Spot is {self.spot_price - key_data['max_pain']:+.0f} pts away)")

        # === DATA HEALTH ===
        print(f"\n{'─'*25} MARKET DATA {'─'*25}")
        print(f"Spot: {self.spot_price:.2f} | IV: {vol_data.get('atm_iv', 0):.2f}% | PCR: {key_data.get('pcr', 0):.2f}")
        
        print("\n" + "="*70)

    # ==================== MAIN ====================
    
    def run_once(self):
        """Run analysis once"""
        print("Fetching data...")
        
        self.get_spot_price()
        if self.spot_price == 0:
            print("Failed to get spot price.")
            return
        
        df = self.get_historical_data(days=365)
        vol_data = self.analyze_volatility(df)
        trend_data = self.analyze_trend(df)
        key_data = self.key_levels.analyze()
        
        if not key_data:
            print("Failed to analyze option chain.")
            return

        self.display_dashboard(vol_data, trend_data, key_data)
    
    def run_live(self, refresh_seconds=60):
        """Run in live loop"""
        print("Starting Live Dashboard...")
        while True:
            try:
                self.run_once()
                print(f"\nRefreshing in {refresh_seconds}s... (Ctrl+C to exit)")
                time.sleep(refresh_seconds)
            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                print(f"Error: {e}")
                time.sleep(10)
    
    # ==================== ADVANCED SCANNERS ====================
    
    def run_advanced_scanner(self):
        while True:
            print("\n" + "="*60)
            print("  ADVANCED VOLATILITY SCANNERS")
            print("="*60)
            print("1. VRP Scanner (IV vs HV Regime)")
            print("2. Kink Detector (Smile Mispricing)")
            print("3. Event Volatility Estimator")
            print("4. Direction & Speed (Cheat Sheet)")
            print("5. Back")
            
            c = input("\nSelect: ")
            
            if c == '1':
                print("Fetching data for VRP Scan...")
                self.get_spot_price()
                atm_iv = self.get_atm_iv()
                if atm_iv > 0:
                    res = self.scanner.scan_vrp(atm_iv)
                    print("\n--- VRP REPORT ---")
                    for k, v in res.items():
                        if k == 'regime_data':
                            print("Regime Details:")
                            for rk, rv in v.items():
                                print(f"  - {rk}: {rv}")
                        else:
                            print(f"{k}: {v}")
                else:
                    print("Error: Could not get ATM IV")
                    
            elif c == '2':
                print("Fetching Option Chain for Kink Detection...")
                self.get_spot_price()
                data = self.key_levels.get_option_chain()
                df = self.key_levels.parse_chain(data)
                
                if not df.empty:
                    # Filter for near expiry? parse_chain might include multiple or filtered.
                    # Assuming parse_chain returns a DF with 'strike', 'iv'
                    # We might need to filter for specific expiry if parse_chain returns all.
                    # key_levels.parse_chain usually handles one expiry or raw data.
                    # Let's assume it works or we filter nearby strikes.
                    
                    # Debug columns
                    # print(df.columns)
                    strikes = df['strike'].tolist()
                    ivs = df['iv'].tolist()
                    
                    kinks = self.scanner.detect_kinks(strikes, ivs, self.spot_price)
                    print("\n--- KINK REPORT ---")
                    if kinks:
                        for k in kinks:
                            print(f"STRIKE {k['strike']}: {k['action']} ({k['strategy']}) | Diff: {k['diff']}%")
                    else:
                        print("No significant kinks detected (Smile is smooth).")
                else:
                    print("Error parsing chain.")
            
            elif c == '3':
                days = input("Enter Days to Event: ")
                try:
                    d_val = float(days)
                    self.get_spot_price()
                    atm_iv = self.get_atm_iv()
                    res = self.scanner.get_event_implied_move(self.spot_price, atm_iv, d_val)
                    print("\n--- EVENT MOVE ---")
                    print(f"Implied Move: +/- {res['move_points']} pts ({res['move_pct']}%)")
                    print(f"Range: {res['range_low']} - {res['range_high']}")
                except ValueError:
                    print("Invalid Input")

            elif c == '4':
                print("Analyzing Direction & Speed...")
                # We need Skew and Term Structure.
                # This requires fetching Near and Far data.
                # For now, let's use the 'get_option_chain' which might default to current.
                # To get Term Structure we need explicit Near/Far fetching.
                # TradingDashboard doesn't easily have Near/Far logic built-in w/o modification.
                # We will skip live Term Structure fetch for this snippet or use dummy?
                # Actually, VolatilityAnalyzer has it. TradingDashboard is simpler.
                # Let's just run skew on current chain.
                
                self.get_spot_price()
                data = self.key_levels.get_option_chain()
                df = self.key_levels.parse_chain(data)
                
                if not df.empty:
                    # Calc Skew
                    atm_iv = self.get_atm_iv()
                    # OTM Put ~ 5% OTM
                    target_put = self.spot_price * 0.95
                    # Find closest strike
                    put_df = df[df['type'] == 'PE']
                    if not put_df.empty:
                       closest_idx = (put_df['strike'] - target_put).abs().idxmin()
                       put_iv = put_df.loc[closest_idx, 'iv']
                       skew_ratio = put_iv / atm_iv if atm_iv > 0 else 1.0
                       
                       # Dummy term spread (0) as we don't have far chain loaded here easily
                       res = self.scanner.analyze_direction_speed(skew_ratio, 0.5, atm_iv) 
                       print("\n--- DIRECTION & SPEED ---")
                       print(f"Direction: {res['direction']}")
                       print(f"Speed: {res['speed']} ({res['speed_label']})")
                       print(f"Skew Ratio: {res['skew_ratio']:.2f}")
                       print("(Term spread not available in this view)")
                    else:
                        print("No Puts found.")

            elif c == '5':
                break

    def main_menu(self):
        """Interactive menu"""
        while True:
            # os.system('cls' if os.name == 'nt' else 'clear')
            print("\n=== NIFTY STRATEGY ENGINE ===")
            print("1. Generate Trade Plan (One-Shot)")
            print("2. Run Live Assistant (Auto-Refresh)")
            print("3. Exit")
            print("4. Advanced Volatility Scanners [NEW]")
            
            c = input("\nSelect: ")
            
            if c == '1':
                self.run_once()
                input("\nPress Enter to continue...")
            elif c == '2':
                self.run_live()
            elif c == '3':
                break
            elif c == '4':
                self.run_advanced_scanner()


if __name__ == "__main__":
    app = TradingDashboard()
    app.main_menu()
