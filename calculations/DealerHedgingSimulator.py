import numpy as np
import pandas as pd
from typing import Dict, Any, List
from .DealerPositionEngine import DealerPositionEngine

class DealerHedgingSimulator:
    """
    Simulates dealer hedging flow by fully repricing the options chain 
    under deterministic, multi-dimensional hypothetical scenarios.
    """
    def __init__(self, position_engine: DealerPositionEngine):
        """
        :param position_engine: An initialized DealerPositionEngine to use for evaluation.
        """
        self.engine = position_engine

    def simulate_scenario(self, 
                          chain_df: pd.DataFrame, 
                          base_spot: float, 
                          spot_shift_pct: float = 0.0, 
                          iv_shift_abs: float = 0.0, 
                          days_passed: float = 0.0,
                          oi_drift_pct: float = 0.0) -> Dict[str, Any]:
        """
        Simulates exactly how dealer inventory changes given a multi-factor market shift.
        
        :param spot_shift_pct: Percentage change in Spot Price (e.g., 1.5 = +1.5%).
        :param iv_shift_abs: Absolute change in Implied Volatility (e.g., 0.02 = +2 points of IV).
        :param days_passed: Calendar days elapsed (e.g., 1.0 = 1 day passed).
        :param oi_drift_pct: Uniform percentage expansion/contraction in Open Interest.
        """
        if chain_df is None or chain_df.empty:
            return {}

        # 1. Base State Evaluation
        base_state = self.engine.calculate_dealer_inventory(chain_df, base_spot)
        base_dex = base_state['net_delta_exposure']

        # 2. Mutate the Chain for Simulation
        sim_df = chain_df.copy()
        
        # Spot Shift
        sim_spot = base_spot * (1.0 + (spot_shift_pct / 100.0))
        
        # IV Shift (supports both decimal 0.15 and percentage 15.0 dataframes)
        if 'iv' in sim_df.columns:
            is_pct = (sim_df['iv'].dropna().median() > 2.0) if not sim_df['iv'].dropna().empty else False
            effective_shift = iv_shift_abs * 100.0 if (is_pct and abs(iv_shift_abs) < 1.0) else iv_shift_abs
            min_floor = 1.0 if is_pct else 0.01
            sim_df['iv'] = np.maximum(sim_df['iv'] + effective_shift, min_floor)
            
        # DTE Decay (floored strictly above 0)
        if 'dte' in sim_df.columns:
            sim_df['dte'] = np.maximum(sim_df['dte'] - days_passed, 1e-5)
            
        # OI Drift
        if 'oi' in sim_df.columns and oi_drift_pct != 0.0:
            sim_df['oi'] = sim_df['oi'] * (1.0 + (oi_drift_pct / 100.0))

        # 3. Reprice Simulated State
        sim_state = self.engine.calculate_dealer_inventory(sim_df, sim_spot)
        sim_dex = sim_state['net_delta_exposure']

        # 4. Calculate Inventory Delta & Hedge Requirements
        # If dealer Delta goes from +100 to +150 (a change of +50), 
        # the dealer MUST SELL 50 shares of the underlying to return to neutral.
        # Required Hedge Flow = - (Simulated DEX - Base DEX)
        dex_change = sim_dex - base_dex
        required_hedge_flow = -dex_change
        
        flow_direction = "BUY" if required_hedge_flow > 0 else "SELL" if required_hedge_flow < 0 else "NEUTRAL"
        
        return {
            'scenario': {
                'spot_shift_pct': spot_shift_pct,
                'iv_shift_abs': iv_shift_abs,
                'days_passed': days_passed,
                'oi_drift_pct': oi_drift_pct
            },
            'base_state': {
                'spot': float(base_spot),
                'dex': float(base_dex),
                'gex_shares': base_state['net_gamma_shares'],
                'vanna': base_state['net_vanna_exposure'],
                'charm': base_state['net_charm_exposure']
            },
            'simulated_state': {
                'spot': float(sim_spot),
                'dex': float(sim_dex),
                'gex_shares': sim_state['net_gamma_shares'],
                'vanna': sim_state['net_vanna_exposure'],
                'charm': sim_state['net_charm_exposure']
            },
            'inventory_changes': {
                'delta_change': float(dex_change),
                'gex_change': float(sim_state['net_gamma_shares'] - base_state['net_gamma_shares']),
                'vanna_change': float(sim_state['net_vanna_exposure'] - base_state['net_vanna_exposure']),
                'charm_change': float(sim_state['net_charm_exposure'] - base_state['net_charm_exposure'])
            },
            'required_hedge_flow': {
                'shares_to_trade': float(required_hedge_flow),
                'direction': flow_direction
            }
        }

    def simulate_matrix(self, 
                        chain_df: pd.DataFrame, 
                        base_spot: float, 
                        spot_shifts: List[float], 
                        iv_shifts: List[float], 
                        days_passed: float = 0.0) -> pd.DataFrame:
        """
        Runs a grid of simulations useful for 2D heatmaps.
        Returns a DataFrame where rows = Spot Shifts, cols = IV Shifts, values = Shares to Trade.
        """
        results = []
        for s_shift in spot_shifts:
            row = {}
            for v_shift in iv_shifts:
                res = self.simulate_scenario(
                    chain_df, 
                    base_spot, 
                    spot_shift_pct=s_shift, 
                    iv_shift_abs=v_shift, 
                    days_passed=days_passed
                )
                row[f"IV_{v_shift:+.3f}"] = res['required_hedge_flow']['shares_to_trade']
            row['Spot_Shift_%'] = s_shift
            results.append(row)
            
        return pd.DataFrame(results).set_index('Spot_Shift_%')
