import numpy as np
import pandas as pd
from scipy.stats import norm
from typing import Dict, Any, Literal

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    import config as _cfg
    _DEFAULT_LOT_SIZE = _cfg.get("nifty_lot_size", 65)
    _DEFAULT_R = _cfg.get("risk_free_rate", 0.051274)
    _DEFAULT_Q = _cfg.get("dividend_yield", 0.0122)
    _DEFAULT_GEX_SCALING = _cfg.get("gex_move_pct", 0.01)
    _DEFAULT_IV = _cfg.get("iv_fallback_flat", 0.15)
except Exception:
    _DEFAULT_LOT_SIZE = 65
    _DEFAULT_R = 0.051274
    _DEFAULT_Q = 0.0122
    _DEFAULT_GEX_SCALING = 0.01
    _DEFAULT_IV = 0.15

class GexEngine:
    """
    Institutional Gamma Exposure (GEX) Calculation Engine.
    Provides mathematically rigorous, vectorized analysis of options chains.
    Strictly isolated from visualization logic.
    """
    
    def __init__(self, 
                 lot_size: int | None = None, 
                 risk_free_rate: float | None = None,
                 dividend_yield: float | None = None,
                 gex_scaling: float | None = None,
                 positioning_model: Literal['standard', 'inverted', 'flow'] = 'standard',
                 default_iv: float | None = None):
        """
        Initialize the GexEngine.
        
        :param lot_size: Contract multiplier (defaults to config.nifty_lot_size, e.g. 65).
        :param risk_free_rate: Risk-free rate for BSM calculation (defaults to config.risk_free_rate, e.g. ~0.0513).
        :param dividend_yield: Continuous dividend yield for Merton (1973) gamma adjustment (defaults to config.dividend_yield).
        :param gex_scaling: Move magnitude for GEX output (0.01 = 1% Spot move).
        :param positioning_model: Inference model for dealer inventory.
            'standard': Assumes dealers are Long Calls (Overwriting flow) and Short Puts (Protective flow).
                        Call GEX is Positive (+), Put GEX is Negative (-).
            'inverted': Assumes dealers are Short Calls (Speculative flow) and Long Puts.
                        Call GEX is Negative (-), Put GEX is Positive (+).
            'flow': Uses actual trade initiation to infer exact dealer positioning (requires 'initiator' column).
        :param default_iv: Fallback Implied Volatility when unobservable (defaults to config.iv_fallback_flat).
        """
        self.lot_size = lot_size if lot_size is not None else _DEFAULT_LOT_SIZE
        self.r = risk_free_rate if risk_free_rate is not None else _DEFAULT_R
        self.q = dividend_yield if dividend_yield is not None else _DEFAULT_Q
        self.gex_scaling = gex_scaling if gex_scaling is not None else _DEFAULT_GEX_SCALING
        self.positioning_model = positioning_model
        self.default_iv = default_iv if default_iv is not None else _DEFAULT_IV

    def compute_gamma_vectorized(self, S: float, K: np.ndarray, T: np.ndarray, iv: np.ndarray) -> np.ndarray:
        """
        Calculates Merton (1973) dividend-adjusted Gamma using NumPy vectorization.
        Gamma = e^{-qT} * n(d1) / (S * iv * sqrt(T))
        Gracefully handles zero DTE or extreme IV edge cases.
        """
        # Enforce mathematical boundaries to prevent NaN or Inf
        T = np.maximum(T, 1e-5)
        iv = np.maximum(iv, 1e-5)
        K = np.maximum(K, 1e-5)
        S = max(S, 1e-5)

        # Merton (1973): d1 uses (r - q) drift
        d1 = (np.log(S / K) + (self.r - self.q + 0.5 * iv ** 2) * T) / (iv * np.sqrt(T))
        # Apply e^{-qT} dividend discount factor — collapses to 1.0 when q=0
        eq_T = np.exp(-self.q * T)
        gamma = eq_T * norm.pdf(d1) / (S * iv * np.sqrt(T))
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
        
        # GEX Scaling: Gamma * TotalShares * Spot^2 * 1%
        # Note: In NSE / Fyers API, 'oi' and 'volume' are reported in underlying units/shares (e.g. 15M).
        # If 'oi' is in contracts (< 100,000 max), multiply by lot_size to convert to shares.
        is_shares = bool((df['oi'].max() > 100_000)) if not df.empty else False
        total_shares = df['oi'] if is_shares else (df['oi'] * self.lot_size)
        total_vol_shares = df['volume'] if is_shares else (df['volume'] * self.lot_size)

        rupee_scale = (spot_price * spot_price * self.gex_scaling)
        df['gex_oi'] = dealer_sign * df['gamma'] * total_shares * rupee_scale
        df['gex_vol'] = dealer_sign * df['gamma'] * total_vol_shares * rupee_scale

        # Market-Standard Units (Nifty Futures Lots per 50-pt move & ₹ Crores per 100-pt move)
        # Gamma * total_shares gives total delta shares per 1 point move.
        # For a standard 50-pt move, divide by lot_size to get Lots:
        df['gex_lots_50pt'] = dealer_sign * (50.0 * df['gamma'] * total_shares / max(self.lot_size, 1))
        df['gex_crores_100pt'] = dealer_sign * (100.0 * df['gamma'] * total_shares * spot_price / 1e7)
        
        # Time-weighted Gamma (scaled by sqrt(T) for normalization across expiries)
        df['rolling_gex'] = df['gex_oi'] * np.sqrt(T_years)
        
        # 4. Aggregate Analytics
        net_gex = df['gex_oi'].sum()
        net_gex_lots = df['gex_lots_50pt'].sum()
        net_gex_crores = df['gex_crores_100pt'].sum()
        profile = df.groupby('strike')['gex_oi'].sum()
        profile_lots = df.groupby('strike')['gex_lots_50pt'].sum()
        profile_crores = df.groupby('strike')['gex_crores_100pt'].sum()
        expiry_gex = df.groupby('dte')['gex_oi'].sum()
        
        # Identify Call Wall and Put Wall
        ce_df = df[df['type'] == 'CE']
        pe_df = df[df['type'] == 'PE']
        call_wall = float(ce_df.groupby('strike')['oi'].sum().idxmax()) if not ce_df.empty else 0.0
        put_wall = float(pe_df.groupby('strike')['oi'].sum().idxmax()) if not pe_df.empty else 0.0
        dist_call_wall = abs(spot_price - call_wall) if call_wall > 0 else 0.0
        dist_put_wall = abs(spot_price - put_wall) if put_wall > 0 else 0.0

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
            val = profile.get(nearest_strike, 0.0)
            spot_gamma = float(val) if val is not None else 0.0
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
            'net_gex_lots_50pt': float(net_gex_lots),
            'net_gex_crores_100pt': float(net_gex_crores),
            'call_wall': call_wall,
            'put_wall': put_wall,
            'dist_to_call_wall': float(dist_call_wall),
            'dist_to_put_wall': float(dist_put_wall),
            'spot_gamma': spot_gamma,
            'forward_gex': float(forward_gex),
            'rolling_gex': float(net_rolling_gex),
            '0dte_gex': float(gex_0dte),
            'zero_gamma_level': float(flip_point),
            'distance_from_zero': float(dist_from_zero),
            'dealer_long_pct': float(dealer_long_pct),
            'dealer_short_pct': float(dealer_short_pct),
            'profile': profile, # pd.Series
            'profile_lots': profile_lots, # pd.Series (Lots per 50pt move)
            'profile_crores': profile_crores, # pd.Series (₹ Cr per 100pt move)
            'expiry_gex': expiry_gex, # pd.Series
            'heatmap': heatmap, # pd.DataFrame
            'gex_vol_total': float(df['gex_vol'].sum()),
            'raw_df': df # Contains base computed Greeks for downstream modules
        }
