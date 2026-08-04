import numpy as np
import pandas as pd
from scipy.stats import norm
from typing import Dict, Any, Literal

class GexEngine:
    """
    Institutional Gamma Exposure (GEX) Calculation Engine.
    Provides mathematically rigorous, vectorized analysis of options chains.
    Strictly isolated from visualization logic.
    """
    
    def __init__(self, 
                 lot_size: int = 75, 
                 risk_free_rate: float = 0.07, 
                 gex_scaling: float = 0.01,
                 positioning_model: Literal['standard', 'inverted', 'flow'] = 'standard',
                 default_iv: float = 0.15):
        """
        Initialize the GexEngine.
        
        :param lot_size: Contract multiplier (e.g., NIFTY = 75, SPX = 100).
        :param risk_free_rate: Risk-free rate for BSM calculation (0.07 = 7%).
        :param gex_scaling: Move magnitude for GEX output (0.01 = 1% Spot move).
        :param positioning_model: Inference model for dealer inventory.
            'standard': Assumes dealers are Long Calls (Overwriting flow) and Short Puts (Protective flow).
                        Call GEX is Positive (+), Put GEX is Negative (-).
            'inverted': Assumes dealers are Short Calls (Speculative flow) and Long Puts.
                        Call GEX is Negative (-), Put GEX is Positive (+).
            'flow': Uses actual trade initiation to infer exact dealer positioning (requires 'initiator' column).
        :param default_iv: Fallback Implied Volatility when unobservable (0.15 = 15%).
        """
        self.lot_size = lot_size
        self.r = risk_free_rate
        self.gex_scaling = gex_scaling
        self.positioning_model = positioning_model
        self.default_iv = default_iv

    def compute_gamma_vectorized(self, S: float, K: np.ndarray, T: np.ndarray, iv: np.ndarray) -> np.ndarray:
        """
        Calculates Black-Scholes Gamma using NumPy vectorization.
        Gracefully handles zero DTE or extreme IV edge cases.
        """
        # Enforce mathematical boundaries to prevent NaN or Inf
        T = np.maximum(T, 1e-5)
        iv = np.maximum(iv, 1e-5)
        K = np.maximum(K, 1e-5)
        S = max(S, 1e-5)
        
        d1 = (np.log(S / K) + (self.r + 0.5 * iv ** 2) * T) / (iv * np.sqrt(T))
        gamma = norm.pdf(d1) / (S * iv * np.sqrt(T))
        return gamma

    def _infer_dealer_sign(self, df: pd.DataFrame) -> np.ndarray:
        """
        Returns a multiplier (+1, -1, or 0) mapping Option Type & Flow to Dealer Gamma.
        Positive Dealer Gamma = Stabilizing Flow (Mean Reversion).
        Negative Dealer Gamma = Destabilizing Flow (Trend Acceleration).
        """
        types = df['type'].str.upper().values
        
        if self.positioning_model == 'standard':
            # Wall Street Standard: Calls = Dealer Long (+), Puts = Dealer Short (-)
            return np.where(types == 'CE', 1.0, -1.0)
            
        elif self.positioning_model == 'inverted':
            # Retail Speculative: Calls = Dealer Short (-), Puts = Dealer Long (+)
            return np.where(types == 'CE', -1.0, 1.0)
            
        elif self.positioning_model == 'flow':
            if 'initiator' not in df.columns:
                # Fallback to standard if explicit flow classification is missing
                return np.where(types == 'CE', 1.0, -1.0)
            
            # Flow model: 
            # If Customer BUYS (initiator = 'buyer'), Dealer SELLS -> Dealer is Short Gamma (-)
            # If Customer SELLS (initiator = 'seller'), Dealer BUYS -> Dealer is Long Gamma (+)
            initiators = df['initiator'].str.lower().values
            return np.where(initiators == 'buyer', -1.0, np.where(initiators == 'seller', 1.0, 0.0))
            
        else:
            raise ValueError(f"Unknown positioning model: {self.positioning_model}")

    def calculate_gex(self, chain_df: pd.DataFrame, spot_price: float) -> Dict[str, Any]:
        """
        Calculates comprehensive institutional GEX metrics from an options chain DataFrame.
        
        Required columns: 'strike', 'type' ('CE'/'PE'), 'oi', 'dte'
        Optional columns: 'iv' (falls back to default_iv), 'volume', 'initiator'
        
        :return: Dict containing structured, presentation-agnostic mathematical outputs.
        """
        if chain_df is None or chain_df.empty:
            return {}

        df = chain_df.copy()
        
        # 1. Clean & Prepare Columns
        df['type'] = df['type'].str.upper()
        
        if 'iv' not in df.columns:
            df['iv'] = self.default_iv
        else:
            # Ensure IV is a decimal and handle missing/NaN
            df['iv'] = df['iv'].fillna(self.default_iv)
            df['iv'] = np.where(df['iv'] > 2.0, df['iv'] / 100.0, df['iv'])
            df['iv'] = np.where(df['iv'] <= 0.001, self.default_iv, df['iv'])
            
        if 'dte' not in df.columns:
            df['dte'] = 1.0  # Safe fallback for 0DTE logic

        if 'volume' not in df.columns:
            df['volume'] = 0.0
            
        if 'oi' not in df.columns:
            df['oi'] = 0.0

        # Convert DTE to Years for BSM
        T_years = np.maximum(df['dte'].values / 365.0, 1e-5)
        
        # 2. Base Mathematical Greek Computation
        df['gamma'] = self.compute_gamma_vectorized(
            spot_price, 
            df['strike'].values, 
            T_years, 
            df['iv'].values
        )
        
        # 3. Apply Institutional Positioning Inference
        dealer_sign = self._infer_dealer_sign(df)
        
        # GEX Scaling: Gamma * Contracts * Spot^2 * 1% * LotSize
        scaling_factor = (spot_price * spot_price * self.gex_scaling) * self.lot_size
        
        df['gex_oi'] = dealer_sign * df['gamma'] * df['oi'] * scaling_factor
        df['gex_vol'] = dealer_sign * df['gamma'] * df['volume'] * scaling_factor
        
        # Time-weighted Gamma (scaled by sqrt(T) for normalization across expiries)
        df['rolling_gex'] = df['gex_oi'] * np.sqrt(T_years)
        
        # 4. Aggregate Analytics
        net_gex = df['gex_oi'].sum()
        profile = df.groupby('strike')['gex_oi'].sum()
        expiry_gex = df.groupby('dte')['gex_oi'].sum()
        
        # Identify Zero Gamma Level (Interpolated Flip Point)
        sorted_strikes = profile.index.sort_values()
        flip_point = 0.0
        
        for i in range(len(sorted_strikes) - 1):
            g1 = profile[sorted_strikes[i]]
            g2 = profile[sorted_strikes[i + 1]]
            if g1 * g2 < 0:
                # Basic linear interpolation for the exact 0 crossing
                w1 = abs(g2) / (abs(g1) + abs(g2) + 1e-9)
                w2 = abs(g1) / (abs(g1) + abs(g2) + 1e-9)
                flip_point = (sorted_strikes[i] * w1) + (sorted_strikes[i+1] * w2)
                break
                
        dist_from_zero = (spot_price - flip_point) if flip_point > 0 else 0.0
        
        # Spot Gamma (GEX at nearest strike)
        try:
            nearest_strike = df['strike'].iloc[(df['strike'] - spot_price).abs().argsort()].values[0]
            spot_gamma = profile.get(nearest_strike, 0.0)
        except Exception:
            spot_gamma = 0.0
            
        # Time-based slices
        gex_0dte = df[df['dte'] <= 1]['gex_oi'].sum()
        forward_gex = df[df['dte'] > 1]['gex_oi'].sum()
        net_rolling_gex = df['rolling_gex'].sum()
        
        # Dealer Positioning Percentages
        pos_gex_total = df[df['gex_oi'] > 0]['gex_oi'].sum()
        neg_gex_total = df[df['gex_oi'] < 0]['gex_oi'].sum()
        total_abs_gex = abs(pos_gex_total) + abs(neg_gex_total)
        
        dealer_long_pct = (pos_gex_total / total_abs_gex * 100) if total_abs_gex > 0 else 0.0
        dealer_short_pct = (abs(neg_gex_total) / total_abs_gex * 100) if total_abs_gex > 0 else 0.0
        
        # Structure Heatmap
        heatmap = df.pivot_table(index='strike', columns='dte', values='gex_oi', aggfunc='sum').fillna(0)
        
        return {
            'net_gex': float(net_gex),
            'spot_gamma': float(spot_gamma),
            'forward_gex': float(forward_gex),
            'rolling_gex': float(net_rolling_gex),
            '0dte_gex': float(gex_0dte),
            'zero_gamma_level': float(flip_point),
            'distance_from_zero': float(dist_from_zero),
            'dealer_long_pct': float(dealer_long_pct),
            'dealer_short_pct': float(dealer_short_pct),
            'profile': profile, # pd.Series
            'expiry_gex': expiry_gex, # pd.Series
            'heatmap': heatmap, # pd.DataFrame
            'gex_vol_total': float(df['gex_vol'].sum()),
            'raw_df': df # Contains base computed Greeks for downstream modules
        }
