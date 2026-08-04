import numpy as np
import pandas as pd
from scipy.stats import norm

class GreeksEngine:
    """
    Mathematical, vectorized engine for computing 1st and 2nd order options Greeks.
    Built for institutional analytics; strict separation from visualization.
    """
    def __init__(self, risk_free_rate: float = 0.07, days_in_year: float = 365.0):
        self.r = risk_free_rate
        self.days_in_year = days_in_year

    def calculate_all_greeks(self, 
                             S: float, 
                             K: np.ndarray, 
                             T_days: np.ndarray, 
                             iv: np.ndarray, 
                             option_types: np.ndarray) -> pd.DataFrame:
        """
        Calculates a comprehensive suite of Black-Scholes Greeks.
        
        :param S: Spot price
        :param K: Array of strike prices
        :param T_days: Array of Days to Expiry
        :param iv: Array of Implied Volatilities (decimal)
        :param option_types: Array of 'CE' or 'PE' strings
        :return: DataFrame containing delta, gamma, vega, theta, vanna, charm, vomma
        """
        # Ensure mathematical boundaries to prevent NaN or division by zero
        T = np.maximum(T_days / self.days_in_year, 1e-5)
        iv = np.maximum(iv, 1e-5)
        K = np.maximum(K, 1e-5)
        S = max(S, 1e-5)
        
        # Identify Calls vs Puts — use pd.Series for robust string handling
        # (np.char.upper fails on non-string dtypes; pd.Series.str handles object/categorical)
        opt_series = pd.Series(option_types).astype(str).str.upper()
        is_call = opt_series.values == 'CE'
        is_put = ~is_call

        # Core d1 / d2 calculation
        sqrt_T = np.sqrt(T)
        d1 = (np.log(S / K) + (self.r + 0.5 * iv ** 2) * T) / (iv * sqrt_T)
        d2 = d1 - iv * sqrt_T

        # Precompute Normals
        N_d1 = norm.cdf(d1)
        N_d2 = norm.cdf(d2)
        N_minus_d1 = norm.cdf(-d1)
        N_minus_d2 = norm.cdf(-d2)
        n_d1 = norm.pdf(d1) # Standard normal PDF

        # --- First Order Greeks ---
        
        # Delta
        delta = np.where(is_call, N_d1, N_d1 - 1.0)
        
        # Theta (Decay per 1 calendar day)
        # Call Theta: -(S * n_d1 * iv) / (2 * sqrt_T) - r * K * exp(-rT) * N_d2
        theta_common = -(S * n_d1 * iv) / (2 * sqrt_T)
        theta_call = theta_common - self.r * K * np.exp(-self.r * T) * N_d2
        theta_put = theta_common + self.r * K * np.exp(-self.r * T) * N_minus_d2
        theta = np.where(is_call, theta_call, theta_put) / self.days_in_year
        
        # Vega (Per 1% change in IV)
        vega = (S * n_d1 * sqrt_T) / 100.0
        
        # Rho (Per 1% change in interest rate)
        rho_call = K * T * np.exp(-self.r * T) * N_d2
        rho_put = -K * T * np.exp(-self.r * T) * N_minus_d2
        rho = np.where(is_call, rho_call, rho_put) / 100.0

        # --- Second Order Greeks ---
        
        # Gamma
        gamma = n_d1 / (S * iv * sqrt_T)
        
        # Vanna (Sensitivity of Delta to IV, or Vega to Spot)
        # Vanna = -n_d1 * (d2 / iv)
        # Note: Often scaled per 1% change in IV for easier interpretation.
        vanna = (-n_d1 * d2 / iv) / 100.0
        
        # Charm (Delta decay - Sensitivity of Delta to Time)
        # Represents how Delta changes simply as 1 day passes.
        # Call Charm: n_d1 * (r / (iv * sqrt_T) - d2 / (2 * T))
        charm_common = n_d1 * (self.r / (iv * sqrt_T) - d2 / (2 * T))
        charm_call = charm_common
        charm_put = charm_common - self.r * np.exp(-self.r * T)
        # Scaled to represent a 1-day passage of time (using days_in_year)
        charm = np.where(is_call, charm_call, charm_put) / self.days_in_year
        
        # Vomma (Sensitivity of Vega to IV)
        # Vomma = Vega * (d1 * d2 / iv)
        # Often scaled for 1% IV moves. Vega is already / 100, so we just divide by another 100.
        vomma = vega * (d1 * d2 / iv) / 100.0

        return pd.DataFrame({
            'delta': delta,
            'gamma': gamma,
            'theta': theta,
            'vega': vega,
            'rho': rho,
            'vanna': vanna,
            'charm': charm,
            'vomma': vomma
        })
