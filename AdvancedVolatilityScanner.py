import sys
import os
import pandas as pd
import numpy as np
import time
import csv
from datetime import datetime

# Add current directory to path to find local modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from FyersAuth import FyersAuthenticator
    from OptionAnalytics import OptionAnalytics
except ImportError:
    # Fallback for when running in isolation if modules aren't found
    pass

class AdvancedVolatilityScanner:
    def __init__(self, fyers_instance=None):
        if fyers_instance:
            self.fyers = fyers_instance
        else:
            self.fyers = self._authenticate()
        self.analytics = OptionAnalytics()
        self.symbol = "NSE:NIFTY50-INDEX"
        self.spot_price = 0
        self.log_file = "vol_regime_history.csv"
        self._ensure_log_file()

    def _authenticate(self):
        # Re-using existing auth logic
        APP_ID = "QUTT4YYMIG-100"
        SECRET_ID = "ZG0WN2NL1B"
        REDIRECT_URI = "http://127.0.0.1:3000/callback"
        auth = FyersAuthenticator(APP_ID, SECRET_ID, REDIRECT_URI)
        return auth.get_fyers_instance()

    def _ensure_log_file(self):
        if not os.path.exists(self.log_file):
            with open(self.log_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Timestamp', 'Spot', 'ATM_IV', 'HV_Close', 'HV_Parkinson', 'VRP_Raw', 'Efficiency_Ratio', 'Regime', 'VoV'])

    def get_spot_price(self):
        try:
            r = self.fyers.quotes(data={"symbols": self.symbol})
            if r.get('s') == 'ok':
                self.spot_price = r['d'][0]['v'].get('lp', 0)
                return self.spot_price
        except Exception as e:
            print(f"Error fetching spot: {e}")
        return 0

    def get_historical_data(self, days=365):
        """Fetch daily OHLC for volatility calculations"""
        today = datetime.now()
        start = today - pd.Timedelta(days=days)
        
        data = {
            "symbol": self.symbol,
            "resolution": "D",
            "date_format": "1",
            "range_from": start.strftime("%Y-%m-%d"),
            "range_to": today.strftime("%Y-%m-%d"),
            "cont_flag": "1"
        }
        
        try:
            r = self.fyers.history(data=data)
            if r.get('s') == 'ok':
                candles = r.get('candles', [])
                df = pd.DataFrame(candles, columns=['ts', 'o', 'h', 'l', 'c', 'v'])
                df['date'] = pd.to_datetime(df['ts'], unit='s')
                df = df.set_index('date')
                df = df.rename(columns={'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close', 'v': 'volume'})
                return df
        except Exception as e:
            print(f"Error fetching history: {e}")
        return pd.DataFrame()

    def classify_regime(self, df):
        """
        Classify Volatility Regime using Efficiency Ratio and Vol of Vol.
        """
        if df.empty or len(df) < 30:
            return {}

        # 1. Calculate Volatilities
        # HV Close-to-Close
        # Need to handle calculations manually if Analytics class doesn't fully support Series alignment here
        # But assuming analytics works on Series:
        hv_close_series = self.analytics.calculate_rolling_historical_volatility(df['close'], window=20)
        
        # HV Parkinson (High Low)
        # We need rolling parkinson. Analytics has parkinson but returns a single value or series?
        # Let's check OptionAnalytics.py content... calculate_parkinson_volatility returns a Series (dropna).
        hv_park_series = self.analytics.calculate_parkinson_volatility(df['high'], df['low'], window=20)
        
        # Align
        common = hv_close_series.index.intersection(hv_park_series.index)
        if len(common) < 1: return {}
        
        hv_close = hv_close_series.loc[common]
        hv_park = hv_park_series.loc[common]
        
        current_hv_close = hv_close.iloc[-1]
        current_hv_park = hv_park.iloc[-1]
        
        # 2. Efficiency Ratio (VER)
        # Trend Efficiency: High/Low range vs Close-Close change.
        # Classic Efficiency Ratio (Kaufman) is Change / Sum(Changes).
        # Here we use Volatility Ratio: Park / Close.
        # If Park (Intraday Range) is HIGH but Close (Net Change) is LOW => High Noise / Mean Reversion. Ratio > 1.
        # If Park is LOW/Normal but Close is HIGH => Gap driven?
        # Actually:
        # If Close-to-Close is High and High-Low is Low -> Gap trend.
        # If Close-to-Close is Low and High-Low is High -> Choppy / Mean Reversion.
        
        ver = current_hv_park / current_hv_close if current_hv_close > 0 else 1.0
        
        # 3. Vol of Vol (VoV)
        # Std Dev of the HV series itself
        vov = hv_close.rolling(20).std().iloc[-1]
        
        regime = "NORMAL"
        if ver > 1.1:
            regime = "MEAN REVERTING (Choppy)"
        elif ver < 0.8:
            regime = "TRENDING (Efficient)"
            
        if vov > 5.0:
            regime += " [UNSTABLE/TURBULENT]"
            
        return {
            'hv_close': current_hv_close,
            'hv_park': current_hv_park,
            'efficiency_ratio': ver,
            'vov': vov,
            'regime_label': regime
        }

    def scan_vrp(self, atm_iv):
        """
        Volatility Risk Premium Scanner with Logging
        """
        df = self.get_historical_data()
        regime_data = self.classify_regime(df)
        
        if not regime_data:
            return {'signal': 'INSUFFICIENT DATA'}

        hv = regime_data['hv_close']
        vrp_raw = atm_iv - hv
        vrp_ratio = atm_iv / hv if hv > 0 else 1.0
        
        signal = "NEUTRAL"
        confidence = 0.0
        
        # Logic:
        # IV >> HV: Options Expensive
        if vrp_ratio > 1.25:
            signal = "SELL PREMIUM (Expensive)"
            confidence = 0.8
        elif vrp_ratio < 0.8:
            signal = "BUY PREMIUM (Cheap)"
            confidence = 0.8
        
        # Refine with Efficiency Ratio
        ver = regime_data['efficiency_ratio']
        if "SELL" in signal:
            # Selling is better if market is Mean Reverting (High VER)
            if ver > 1.1: confidence += 0.1
            elif ver < 0.8: confidence -= 0.2 # Dangerous to sell in trending move
        elif "BUY" in signal:
            # Buying is better if market is Trending (Low VER)
            if ver < 0.8: confidence += 0.1
            elif ver > 1.1: confidence -= 0.2
            
        confidence = min(1.0, max(0.0, confidence))
        
        # Log
        self.spot_price = self.get_spot_price() or self.spot_price
        self._log_scan(self.spot_price, atm_iv, hv, regime_data['hv_park'], vrp_raw, ver, regime_data['regime_label'], regime_data['vov'])
        
        return {
            'spot': self.spot_price,
            'atm_iv': atm_iv,
            'hv_close': hv,
            'vrp_ratio': vrp_ratio,
            'signal': signal,
            'confidence': confidence,
            'regime_data': regime_data
        }
        
    def _log_scan(self, spot, iv, hv, pk, vrp, ver, reg, vov):
        try:
            with open(self.log_file, 'a', newline='') as f:
                writer = csv.writer(f)
                ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                writer.writerow([ts, f"{spot:.2f}", f"{iv:.2f}", f"{hv:.2f}", f"{pk:.2f}", f"{vrp:.2f}", f"{ver:.2f}", reg, f"{vov:.2f}"])
        except: pass


    # ============================================================
    # 3. KINK DETECTOR
    # ============================================================
    def detect_kinks(self, strikes, ivs, current_spot):
        """
        Detects 'Kinks' (Mispricing) in the Volatility Smile.
        Uses Cubic Spline smoothing to find outliers.
        """
        if len(strikes) < 5 or len(ivs) < 5:
            return []
            
        # Sort data
        data = sorted(zip(strikes, ivs))
        s_sorted, iv_sorted = zip(*data)
        
        # Fit Cubic Spline
        try:
            cs = CubicSpline(s_sorted, iv_sorted)
            model_ivs = cs(s_sorted)
        except Exception as e:
            print(f"Spline Error: {e}")
            return []
            
        kinks = []
        threshold = 1.5 # 1.5% IV deviation threshold
        
        for i, (strike, actual, model) in enumerate(zip(s_sorted, iv_sorted, model_ivs)):
            if abs(strike - current_spot) > current_spot * 0.10:
                continue
                
            diff = actual - model
            
            if abs(diff) > threshold:
                action = "SELL" if diff > 0 else "BUY" 
                strategy = "Butterfly (Short Body)" if action == "SELL" else "Butterfly (Long Body)"
                
                kinks.append({
                    'strike': strike,
                    'actual_iv': round(actual, 2),
                    'model_iv': round(model, 2),
                    'diff': round(diff, 2),
                    'action': action,
                    'strategy': strategy
                })
                
        return kinks

    # ============================================================
    # 4. EVENT VOLATILITY ESTIMATOR
    # ============================================================
    def get_event_implied_move(self, spot, atm_iv, days_to_expiry):
        """
        Returns the Implied Move for the period.
        """
        if spot == 0 or atm_iv == 0: return {}
        
        # Annual to Daily
        daily_vol = (atm_iv / 100) / np.sqrt(252)
        
        # Move for the period
        t_years = days_to_expiry / 365.0
        move_pct = (atm_iv / 100) * np.sqrt(t_years)
        move_points = spot * move_pct
        
        return {
            'days': days_to_expiry,
            'iv': atm_iv,
            'move_pct': round(move_pct * 100, 2),
            'move_points': round(move_points, 2),
            'range_low': round(spot - move_points, 2),
            'range_high': round(spot + move_points, 2)
        }

    # ============================================================
    # 5. DIRECTION & SPEED (CHEAT SHEET)
    # ============================================================
    def analyze_direction_speed(self, skew_ratio, term_spread, iv_level):
        """
        Determines Market Direction and Speed based on Skew and Structure.
        
        Args:
            skew_ratio (float): Put Skew / Call Skew (or similar metric). 
                                Here: OTM Put IV / ATM IV. 
                                > 1.2 = Bearish (Put Skew). < 0.9 = Bullish (Call Skew).
            term_spread (float): Far IV - Near IV.
                                > 0 = Contango (Slow). < 0 = Backwardation (Fast).
            iv_level (float): Current ATM IV.
            
        Returns:
            dict: { 'direction': ..., 'speed': ..., 'signal': ... }
        """
        # A. DIRECTION
        direction = "NEUTRAL ⚪"
        if skew_ratio < 0.9:
            direction = "BULLISH 🟢 (Call Skew)"
        elif skew_ratio > 1.2:
            direction = "BEARISH 🔴 (Put Skew)"
            
        # B. SPEED
        speed = "NORMAL"
        speed_label = "Cruising"
        
        # High Speed Logic
        if iv_level > 20 or term_spread < -1.0:
            speed = "FAST 🚀"
            if term_spread < -1.0:
                speed_label = "Explosion (Backwardation)"
            else:
                speed_label = "High Velocity"
        # Low Speed Logic
        elif iv_level < 12 and term_spread > 0:
            speed = "SLOW 🐢"
            if term_spread > 1.0:
                 speed_label = "Grind (Contango)"
            else:
                 speed_label = "Drift"
                 
        # Summary Signal
        combined = f"{direction} + {speed}"
        
        return {
            'direction': direction,
            'speed': speed,
            'speed_label': speed_label,
            'summary': combined,
            'skew_ratio': skew_ratio,
            'term_spread': term_spread
        }

if __name__ == "__main__":
    scanner = AdvancedVolatilityScanner()
    print("Fetching History...")
    df = scanner.get_historical_data()
    print(f"Fetched {len(df)} days.")
    
    # Mock
    res = scanner.scan_vrp(atm_iv=14.5)
    print("\nScan Result:")
    for k,v in res.items():
        print(f"{k}: {v}")
    print(f"\nLogged to {scanner.log_file}")
