import numpy as np
from scipy.stats import norm
import pandas as pd
from datetime import datetime

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    import config as _cfg
    _DEFAULT_R = _cfg.get("risk_free_rate", 0.051274)
    _DEFAULT_Q = _cfg.get("dividend_yield", 0.0122)
    _DEFAULT_IV_SEED = _cfg.get("iv_solver_seed", 0.15)
except Exception:
    _DEFAULT_R = 0.051274
    _DEFAULT_Q = 0.0122
    _DEFAULT_IV_SEED = 0.15

class OptionAnalytics:
    def __init__(self):
        pass

    def get_time_to_expiry(self, expiry_date_str):
        """
        Calculate time to expiry in years.
        expiry_date_str: Format 'YYYY-MM-DD' or 'DD-MMM-YYYY' support
        """
        try:
            # Try ISO format first
            try:
                expiry = datetime.strptime(expiry_date_str, "%Y-%m-%d")
            except ValueError:
                # Try 08-Jan-2024 format often used in Indian markets
                expiry = datetime.strptime(expiry_date_str, "%d-%b-%Y")
                
            # Set expiry to 15:30:00 (Indian Market Close)
            expiry = expiry.replace(hour=15, minute=30, second=0)
            
            now = datetime.now()
            delta = expiry - now
            
            # Total seconds / seconds in a year
            seconds_in_year = 365.0 * 24.0 * 3600.0
            years = delta.total_seconds() / seconds_in_year
            
            return max(years, 1e-5) # Prevent divide by zero or negative
        except Exception as e:
            # print(f"Error calculating T: {e}") # Silent error
            return 0.01

    def calculate_iv_regime(self, df, window=20):
        """
        Analyze IV regime based on historical data.
        df: DataFrame containing 'iv' and 'timestamp'
        window: Rolling window size for comparison
        """
        if df is None or df.empty or 'iv' not in df.columns:
            return "Unknown"
            
        # Calculate rolling statistics
        current_iv = df['iv'].iloc[-1]
        rolling_mean = df['iv'].rolling(window=window).mean().iloc[-1]
        rolling_std = df['iv'].rolling(window=window).std().iloc[-1]
        
        if pd.isna(rolling_mean) or pd.isna(rolling_std):
            return "Insufficient Data"
            
        z_score = (current_iv - rolling_mean) / rolling_std if rolling_std != 0 else 0
        
        if z_score > 2:
            return "High / Spiking"
        elif z_score < -2:
            return "Low / Crushing"
        elif z_score > 1:
            return "Elevated"
        elif z_score < -1:
            return "Suppressed"
        else:
            return "Normal"

    def black_scholes(self, S, K, T, r, sigma, option_type='CE', q=0.0):
        """
        S: Spot Price
        K: Strike Price
        T: Time to Expiry (in years)
        r: Risk-free rate (decimal, e.g., 0.0513)
        sigma: Volatility (decimal, e.g., 0.15)
        q: Continuous dividend yield (decimal, e.g. 0.0122, Merton 1973)
        """
        if T <= 0 or sigma <= 0 or K <= 0 or S <= 0:
            return 0.0
            
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        if option_type == 'CE':
            price = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        else:
            price = K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)
            
        return price

    def vega(self, S, K, T, r, sigma, q=0.0):
        """Calculate Vega with optional continuous dividend yield."""
        if T <= 0 or sigma <= 0 or K <= 0 or S <= 0: return 0.0
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        return S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T)

    def implied_volatility(self, price, S, K, T, r, option_type='CE', tol=1e-5, max_iter=100, q=0.0):
        """
        Calculate Implied Volatility using Hybrid Newton-Raphson + Bisection.
        Uses iv_solver_seed from config (default 0.15 = 15%) as initial guess.
        """
        # 1. Bounds check (Intrinsic Value)
        intrinsic = 0
        if option_type == 'CE':
            intrinsic = max(S * np.exp(-q * T) - K * np.exp(-r * T), 0)
        else:
            intrinsic = max(K * np.exp(-r * T) - S * np.exp(-q * T), 0)
            
        if price <= intrinsic + 1e-6:
            return 0.0 # Price is too low, no volatility premium.

        # 2. Hybrid Solver
        # Standard Newton-Raphson with Bisection fallback
        low = 1e-6
        high = 5.0 # Max 500% IV
        sigma = _DEFAULT_IV_SEED # Initial guess from config
        
        for i in range(max_iter):
            # Calculate Price and Vega
            bs_price = self.black_scholes(S, K, T, r, sigma, option_type, q=q)
            diff = bs_price - price
            
            if abs(diff) < tol:
                return sigma * 100
                
            v = self.vega(S, K, T, r, sigma, q=q)
            
            # Newton Step
            if v > 1e-8:
                sigma_new = sigma - diff / v
                if low < sigma_new < high:
                    sigma = sigma_new
                    continue
            
            # Fallback to Bisection (or if Newton failed)
            if diff > 0: # Estimate too high
                high = sigma
            else: # Estimate too low
                low = sigma
                
            sigma = (low + high) / 2.0
            
            if abs(high - low) < 1e-4:
                break
                
        return sigma * 100

    def calculate_greeks(self, S, K, T, r, sigma, option_type='CE', q=0.0):
        """
        Calculate Delta, Gamma, Vega, Theta (Merton 1973 dividend adjusted)
        sigma: Expected as decimal (e.g. 0.20 for 20%)
        """
        if T <= 0 or sigma <= 0 or K <= 0 or S <= 0:
            return {'delta': 0, 'gamma': 0, 'vega': 0, 'theta': 0}

        d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        if option_type == 'CE':
            delta = np.exp(-q * T) * norm.cdf(d1)
            theta = (-S * np.exp(-q * T) * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) 
                     - r * K * np.exp(-r * T) * norm.cdf(d2) 
                     + q * S * np.exp(-q * T) * norm.cdf(d1)) / 365.0
        else:
            delta = np.exp(-q * T) * (norm.cdf(d1) - 1)
            theta = (-S * np.exp(-q * T) * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) 
                     + r * K * np.exp(-r * T) * norm.cdf(-d2) 
                     - q * S * np.exp(-q * T) * norm.cdf(-d1)) / 365.0

        gamma = np.exp(-q * T) * norm.pdf(d1) / (S * sigma * np.sqrt(T))
        vega = (S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T)) / 100.0 # Standard convention to divide by 100
        
        return {
            'delta': round(delta, 3),
            'gamma': round(gamma, 4),
            'vega': round(vega, 3),
            'theta': round(theta, 3)
        }

    def simulate_straddle(self, current_spot, target_spot, current_iv, days_to_expiry, risk_free_rate=0.10):
        """
        Simulate Straddle Price for a target spot price.
        """
        # Find ATM strike for target spot
        target_atm_strike = round(target_spot / 50) * 50
        
        T = days_to_expiry / 365.0
        sigma = current_iv / 100.0 
        
        # Calculate Call and Put prices
        ce_price = self.black_scholes(target_spot, target_atm_strike, T, risk_free_rate, sigma, 'CE')
        pe_price = self.black_scholes(target_spot, target_atm_strike, T, risk_free_rate, sigma, 'PE')
        
        straddle_price = ce_price + pe_price
        
        return {
            'target_spot': target_spot,
            'atm_strike': target_atm_strike,
            'ce_price': round(ce_price, 2),
            'pe_price': round(pe_price, 2),
            'straddle_price': round(straddle_price, 2),
            'iv_used': current_iv
        }

    def calculate_historical_volatility(self, price_series, window=20):
        """
        Calculate annualized Historical Volatility.
        price_series: List or Series of close prices.
        """
        try:
            if len(price_series) < window + 1:
                return 0.0
                
            # Convert to series if list
            if isinstance(price_series, list):    
                s = pd.Series(price_series)
            else:
                s = price_series
                
            # Log Returns: ln(P_t / P_{t-1})
            log_returns = np.log(s / s.shift(1))
            
            # Std Dev of returns
            vol = log_returns.rolling(window=window).std().iloc[-1]
            
            # Annualize (Crypto=365, Stocks=252. Fyers is stocks)
            annualized_vol = vol * np.sqrt(252)
            
            return round(annualized_vol * 100, 2) # Percent
        except Exception as e:
            print(f"HV Calc Error: {e}")
            return 0.0

    def calculate_forward_volatility(self, iv1, t1, iv2, t2):
        """
        Calculate Forward Volatility between time t1 and t2.
        iv1, iv2: Implied Volatility (decimal or percent, must be consistent)
        t1, t2: Time in years
        """
        try:
            # Ensure t2 > t1
            if t2 <= t1: return 0.0
            
            # Convert to variance
            var1 = (iv1 ** 2) * t1
            var2 = (iv2 ** 2) * t2
            
            # Forward Variance
            fwd_var = (var2 - var1) / (t2 - t1)
            
            if fwd_var < 0: return 0.0 # Impossible mathematically if arb-free, but possible in dirty data
            
            return np.sqrt(fwd_var)
        except Exception as e:
            print(f"Fwd Vol Calc Error: {e}")
            return 0.0

    def calculate_rolling_historical_volatility(self, price_series, window=20):
        """
        Calculate rolling annualized Historical Volatility series.
        Returns a pandas Series.
        """
        try:
            if isinstance(price_series, pd.Series):
                s = price_series
            elif isinstance(price_series, pd.DataFrame):
                s = price_series.iloc[:, 0]
            else:
                s = pd.Series(price_series)
                
            log_returns = pd.Series(np.log(s / s.shift(1)), index=s.index)
            rolling_vol = log_returns.rolling(window=window).std() * np.sqrt(252) * 100
            return rolling_vol.dropna()
        except Exception as e:
            print(f"Rolling HV Error: {e}")
            return pd.Series()

    def calculate_z_score(self, value, series):
        """
        Calculate Z-Score of a value relative to a historical series.
        """
        try:
            if series.empty: return 0.0
            mean = series.mean()
            std = series.std()
            if std == 0: return 0.0
            return (value - mean) / std
        except:
            return 0.0

    def calculate_parkinson_volatility(self, highs, lows, window=20):
        """
        Calculate Parkinson Volatility using High/Low range.
        More efficient (requires less data) and precise for intraday vol.
        """
        try:
            if not isinstance(highs, pd.Series): highs = pd.Series(highs)
            if not isinstance(lows, pd.Series): lows = pd.Series(lows)
            
            # Formula: sqrt(1 / (4 * N * ln(2)) * sum(ln(H/L)^2))
            log_hl = np.log(highs / lows) ** 2
            
            # Rolling sum for the window
            rolling_sum = log_hl.rolling(window=window).sum()
            
            factor = 1.0 / (4.0 * window * np.log(2.0))
            vol = np.sqrt(factor * rolling_sum) * np.sqrt(252) # Annualize
            
            return (vol * 100).dropna()
        except Exception as e:
            print(f"Parkinson Vol Error: {e}")
            return pd.Series()
