import numpy as np
import pandas as pd
from typing import Dict, Any, Literal
from .GreeksEngine import GreeksEngine

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    import config as _cfg
    _DEFAULT_LOT_SIZE = _cfg.get("nifty_lot_size", 65)
    _DEFAULT_R = _cfg.get("risk_free_rate", 0.051274)
    _DEFAULT_Q = _cfg.get("dividend_yield", 0.0122)
    _DEFAULT_IV = _cfg.get("iv_fallback_flat", 0.15)
except Exception:
    _DEFAULT_LOT_SIZE = 65
    _DEFAULT_R = 0.051274
    _DEFAULT_Q = 0.0122
    _DEFAULT_IV = 0.15

class DealerPositionEngine:
    """
    Estimates institutional dealer inventory and hedging requirements continuously.
    Utilizes configurable inference models to infer dealer positioning from observable data.
    """
    def __init__(self, 
                 lot_size: int | None = None,
                 risk_free_rate: float | None = None,
                 dividend_yield: float | None = None,
                 positioning_model: Literal['standard', 'inverted', 'flow'] = 'standard',
                 default_iv: float | None = None):
        self.lot_size = lot_size if lot_size is not None else _DEFAULT_LOT_SIZE
        self.positioning_model = positioning_model
        self.default_iv = default_iv if default_iv is not None else _DEFAULT_IV
        
        # Instantiate the pure mathematical greeks engine with dividend yield
        self.greeks_engine = GreeksEngine(
            risk_free_rate=risk_free_rate if risk_free_rate is not None else _DEFAULT_R,
            dividend_yield=dividend_yield if dividend_yield is not None else _DEFAULT_Q,
            days_in_year=365.0
        )

    def _infer_dealer_sign(self, df: pd.DataFrame) -> np.ndarray:
        """
        Infers dealer directional position (Long = +1, Short = -1) per contract.
        Matches the GexEngine configurable inference logic.
        """
        types = df['type'].str.upper().values
        
        if self.positioning_model == 'standard':
            # Wall St Standard: Calls = Dealer Long (+1), Puts = Dealer Short (-1)
            # This means dealer buys Calls (Customer Overwriting) and sells Puts (Customer Protection).
            return np.where(types == 'CE', 1.0, -1.0)
            
        elif self.positioning_model == 'inverted':
            # Retail Speculative: Calls = Dealer Short (-1), Puts = Dealer Long (+1)
            return np.where(types == 'CE', -1.0, 1.0)
            
        elif self.positioning_model == 'flow':
            if 'initiator' not in df.columns:
                return np.where(types == 'CE', 1.0, -1.0) # Fallback
            initiators = df['initiator'].str.lower().values
            return np.where(initiators == 'buyer', -1.0, np.where(initiators == 'seller', 1.0, 0.0))
            
        else:
            raise ValueError(f"Unknown positioning model: {self.positioning_model}")

    def calculate_dealer_inventory(self, chain_df: pd.DataFrame, spot_price: float) -> Dict[str, Any]:
        """
        Calculates Dealer Delta, Gamma, Vega, Vanna, and Charm exposures.
        Projects required hedge adjustments (shares/futures to buy/sell).
        """
        if chain_df is None or chain_df.empty:
            return {}

        df = chain_df.copy()
        
        # 1. Clean & Prepare Columns
        df['type'] = df['type'].str.upper()
        
        if 'iv' not in df.columns:
            df['iv'] = self.default_iv
        else:
            df['iv'] = df['iv'].fillna(self.default_iv)
            df['iv'] = np.where(df['iv'] > 2.0, df['iv'] / 100.0, df['iv'])
            df['iv'] = np.where(df['iv'] <= 0.001, self.default_iv, df['iv'])
            
        if 'dte' not in df.columns:
            df['dte'] = 1.0
            
        if 'oi' not in df.columns:
            df['oi'] = 0.0

        # 2. Compute Base Greeks
        greeks_df = self.greeks_engine.calculate_all_greeks(
            S=spot_price,
            K=df['strike'].values,
            T_days=df['dte'].values,
            iv=df['iv'].values,
            option_types=df['type'].values
        )
        
        # Merge greeks into main df for transparency
        for col in greeks_df.columns:
            df[col] = greeks_df[col]

        # 3. Apply Dealer Inference
        dealer_sign = self._infer_dealer_sign(df)
        
        # Calculate exposure per contract side
        is_shares = bool((df['oi'].max() > 100_000)) if not df.empty else False
        total_shares = df['oi'] if is_shares else (df['oi'] * self.lot_size)
        base_multiplier = total_shares * dealer_sign
        
        # Net Delta Exposure (DEX) - Total equivalent underlying shares dealer is holding
        df['dex'] = df['delta'] * base_multiplier
        
        # Net Vega Exposure (VEX) - Dealer P&L change per 1% IV move
        df['vex'] = df['vega'] * base_multiplier
        
        # Net Gamma Exposure (GEX) - Dealer Delta change per 1% Spot move
        # (Gamma gives Delta change per 1 POINT move. Multiply by 1% of Spot to get per 1% move).
        df['gex'] = df['gamma'] * base_multiplier * (spot_price * 0.01) * spot_price 
        # Note: (gamma * spot * 0.01) gives delta change per 1% spot move. 
        # Multiplied by spot again to express in NOTIONAL value, OR keep it in shares.
        # Wall St Convention is usually Notional GEX. If we want shares to hedge: 
        df['gex_shares'] = df['gamma'] * base_multiplier * (spot_price * 0.01)
        
        # Net Vanna Exposure - Dealer Delta change per 1% IV move
        df['vanna_ex'] = df['vanna'] * base_multiplier
        
        # Net Charm Exposure - Dealer Delta change per 1 calendar day passing
        df['charm_ex'] = df['charm'] * base_multiplier
        
        # Net Vomma Exposure - Dealer Vega change per 1% IV move
        df['vomma_ex'] = df['vomma'] * base_multiplier

        # 4. Aggregations & Projected Hedge Adjustments
        net_dex = df['dex'].sum()
        net_vex = df['vex'].sum()
        net_gex_shares = df['gex_shares'].sum()
        net_vanna = df['vanna_ex'].sum()
        net_charm = df['charm_ex'].sum()
        
        # Hedging Projections: 
        # To remain Delta Neutral, a dealer must offset their Delta changes by buying/selling the underlying.
        
        # If Spot moves UP 1%: Dealer delta changes by +GEX_Shares. Dealer must SELL GEX_Shares to flatten.
        hedge_spot_up_1pct = -net_gex_shares
        
        # If IV moves UP 1%: Dealer delta changes by +Vanna. Dealer must SELL Vanna shares to flatten.
        hedge_iv_up_1pct = -net_vanna
        
        # If 1 Day Passes: Dealer delta changes by +Charm. Dealer must SELL Charm shares to flatten.
        hedge_1_day_pass = -net_charm

        return {
            'net_delta_exposure': float(net_dex),
            'net_vega_exposure': float(net_vex),
            'net_gamma_shares': float(net_gex_shares),
            'net_vanna_exposure': float(net_vanna),
            'net_charm_exposure': float(net_charm),
            'projected_hedging': {
                'buy_shares_if_spot_up_1pct': float(hedge_spot_up_1pct),
                'buy_shares_if_iv_up_1pct': float(hedge_iv_up_1pct),
                'buy_shares_if_1_day_passes': float(hedge_1_day_pass)
            },
            'strike_profile': {
                'dex': df.groupby('strike')['dex'].sum(),
                'gex': df.groupby('strike')['gex_shares'].sum(),
                'vanna': df.groupby('strike')['vanna_ex'].sum(),
                'charm': df.groupby('strike')['charm_ex'].sum()
            },
            'raw_df': df
        }
