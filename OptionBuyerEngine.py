"""
OptionBuyerEngine.py
====================
Translates directional bias (from ConfluenceEngine) into actionable option buying setups.
Calculates Entry Premium, Exact Premium Targets (using Delta/Gamma forecasting), 
and a Dynamic Premium Stop-Loss (derived from Intraday VWAP violation).
"""

from typing import Dict, Optional

class OptionBuyerEngine:
    def __init__(self):
        pass

    def generate_trade_setup(self, confluence_verdict: dict, spot: float, vwap: float, df_chain, gex_acceleration: float, intraday_regime: str) -> Optional[Dict]:
        """
        Generates an Option Buying Trade Setup if the verdict is highly directional.
        """
        verdict = confluence_verdict.get('verdict', 'NEUTRAL')
        if 'BULLISH' not in verdict and 'BEARISH' not in verdict:
            return None # No trade zone

        # 1. Select option type
        opt_type = 'CE' if 'BULLISH' in verdict else 'PE'

        # 2. Select Strike (ATM)
        if df_chain.empty or spot <= 0:
            return None

        # Filter chain for type
        sub_chain = df_chain[df_chain['type'] == opt_type].copy()
        if sub_chain.empty: 
            return None

        # Find closest strike
        sub_chain['dist'] = abs(sub_chain['strike'] - spot)
        atm_row = sub_chain.loc[sub_chain['dist'].idxmin()]
        
        strike = float(atm_row['strike'])
        ltp = float(atm_row['price'])
        if ltp <= 0.5: 
            return None
        
        delta = float(atm_row.get('delta', 0.5))
        if abs(delta) < 0.05: delta = 0.5  # fallback
        gamma = float(atm_row.get('gamma', 0.0))

        # 3. Compute Spot Move Targets
        # Breakout setups look for localized 0.5% to 1.0% pushes
        spot_move_t1 = spot * 0.005
        spot_move_t2 = spot * 0.01

        # Use Taylor expansion for premium change: dP = Delta * dS + 0.5 * Gamma * dS^2
        # ABS(dS) used with ABS(Delta) because the direction is assumed to be favorable.
        dP_target1 = abs(delta) * spot_move_t1 + 0.5 * gamma * (spot_move_t1 ** 2)
        dP_target2 = abs(delta) * spot_move_t2 + 0.5 * gamma * (spot_move_t2 ** 2)

        premium_t1 = max(ltp * 1.05, ltp + dP_target1)
        premium_t2 = max(ltp * 1.10, ltp + dP_target2)

        # 4. Stop Loss via Greeks (Spot crossing VWAP)
        # If BULLISH, spot dropping to VWAP invalidates. 
        # If BEARISH, spot rising to VWAP invalidates.
        if 'BULLISH' in verdict:
            spot_dist_to_vwap = spot - vwap
            if spot_dist_to_vwap <= 0:
                # Already hovering/below VWAP? Very tight mental SL required.
                spot_dist_to_vwap = spot * 0.0025 # 0.25% buffer
        else:
            spot_dist_to_vwap = vwap - spot
            if spot_dist_to_vwap <= 0:
                spot_dist_to_vwap = spot * 0.0025
                
        # How much does premium drop if spot moves AGAINST us by spot_dist_to_vwap?
        # dP = -abs(delta) * dist + 0.5 * gamma * dist^2
        dP_sl = -abs(delta) * spot_dist_to_vwap + 0.5 * gamma * (spot_dist_to_vwap ** 2)
        
        # Hard floor at 40% loss max if VWAP is too far (protects capital from wide gaps)
        premium_sl = max(ltp * 0.60, ltp + dP_sl)
        
        # Safety bound (ensure SL is logically below LTP)
        if premium_sl >= ltp:
            premium_sl = ltp * 0.85

        # 5. Expected Timing Matrix
        timing = "1-2 Hours (Building Support)"
        if intraday_regime == "EXPANSION (LIVE)" or abs(gex_acceleration) > 5e9:
            timing = "IMMINENT (0-30 Mins)"
        elif intraday_regime == "MOMENTUM / TRENDING":
            timing = "30-60 Mins (Trend Running)"

        return {
            'action': f"BUY {strike:.0f} {opt_type}",
            'verdict_base': verdict,
            'entry_limit': round(ltp, 1),
            't1_premium': round(premium_t1, 1),
            't2_premium': round(premium_t2, 1),
            'sl_premium': round(premium_sl, 1),
            'vwap_hard_level': round(vwap, 0),
            'timing': timing,
            'delta_exposure': round(abs(delta), 2)
        }
