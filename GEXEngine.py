import numpy as np
import pandas as pd
from typing import Dict, Any

class GEXEngine:
    """
    Unified Gamma Exposure (GEX) Engine.
    Provides standard GEX calculations used by multiple F-Intel modules.
    """
    
    @staticmethod
    def compute_gex(df: pd.DataFrame, spot: float, T: float, lot_size: int = 75, r_rate: float = 0.07) -> Dict[str, Any]:
        """
        Computes GEX metrics for a given option chain DataFrame.
        
        Assumptions:
        Dealers are assumed net short calls & short puts (standard retail-driven NIFTY prior).
        Positive net GEX = long gamma = stabilizing.
        
        Formula:
        gex_per_strike = (ce_gamma * ce_oi - pe_gamma * pe_oi) * lot_size * spot * (spot * 0.01)
        """
        profile = {}
        if df.empty or spot <= 0:
            return {
                'profile': profile,
                'net_gex': 0.0,
                'gex_flip_point': 0.0,
                'atm_concentration': 0.0
            }
            
        # Optional: analytics fallback if gamma is missing
        # But we expect 'gamma' to be pre-calculated in df if available.
        
        for strike in df['strike'].unique():
            ce_row = df[(df['strike'] == strike) & (df['type'] == 'CE')]
            pe_row = df[(df['strike'] == strike) & (df['type'] == 'PE')]
            
            ce_gamma = ce_row.iloc[0]['gamma'] if not ce_row.empty else 0.0
            ce_oi = ce_row.iloc[0]['oi'] if not ce_row.empty else 0
            
            pe_gamma = pe_row.iloc[0]['gamma'] if not pe_row.empty else 0.0
            pe_oi = pe_row.iloc[0]['oi'] if not pe_row.empty else 0
            
            # Use unified magnitude scaling
            gex = (ce_gamma * ce_oi - pe_gamma * pe_oi) * lot_size * spot * (spot * 0.01)
            profile[strike] = gex
            
        net_gex = sum(profile.values())
        total_abs = sum(abs(v) for v in profile.values()) or 1
        
        # ATM ±1% concentration
        atm_band_gex = sum(abs(v) for k, v in profile.items() if spot * 0.99 <= k <= spot * 1.01)
        atm_concentration = (atm_band_gex / total_abs) * 100
        
        # GEX flip point
        sorted_strikes = sorted(profile.keys())
        flip_point = 0.0
        for i in range(len(sorted_strikes) - 1):
            g1 = profile[sorted_strikes[i]]
            g2 = profile[sorted_strikes[i + 1]]
            if g1 * g2 < 0:  # sign change
                # Pick the one closer to zero
                flip_point = sorted_strikes[i] if abs(g1) < abs(g2) else sorted_strikes[i + 1]
                break
                
        return {
            'profile': profile,
            'net_gex': net_gex,
            'gex_flip_point': flip_point,
            'atm_concentration': atm_concentration
        }
