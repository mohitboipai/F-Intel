"""
calculations/GexRebalanceEngine.py
=============================================================================
Institutional GEX Spot Rebalancing & Option Buyer Radar Engine.

Solves:
1. The Ignition Moment:
   - Wall 2 closer to spot than Wall 1 (Wooden toll gate).
   - Wall 2 15m OI velocity is negative (Call/Put writers capitulating).
   - Spot crosses Wall 2 into the vacuum runway.
2. The Rebalance Horizon:
   - Evaluates mandatory dealer delta-hedging obligations:
     Delta_H(S) = - [DEX(S) - DEX(S0)]
   - Locates the Zero-Gamma Fuel Apex (S_zero): where dealer futures buying peaks
     and shuts off (the natural rebalancing target).
   - Locates the Terminal Wall 1 Fortress Pin (S_wall1).
3. Dual-Strike Selector:
   - Option 1 (Primary ATM): High win-rate, steady delta ~0.50.
   - Option 2 (OTM Gamma Rocket): Cheap OTM strike (₹10-₹25) inside the runway
     that explodes 3x-8x on 0DTE / 1DTE as spot crosses into the money.
4. Direct Premium Converter:
   - Translates spot points directly into option premium ₹ buy zone, ₹ target, and ₹ stop.
=============================================================================
"""

import math
import time
import numpy as np
import pandas as pd
from scipy.stats import norm
from typing import Dict, Any, Optional, List, Tuple

try:
    import config as _cfg
    _DEFAULT_LOT_SIZE = _cfg.get("nifty_lot_size", 65)
    _DEFAULT_R = _cfg.get("risk_free_rate", 0.051274)
    _DEFAULT_Q = _cfg.get("dividend_yield", 0.0122)
except Exception:
    _DEFAULT_LOT_SIZE = 65
    _DEFAULT_R = 0.051274
    _DEFAULT_Q = 0.0122


# ─────────────────────────────────────────────────────────────────────────────
# VECTORIZED BSM MATH HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _bsm_delta_vectorized(S: float, K: np.ndarray, T: np.ndarray, r: float, q: float, iv: np.ndarray, types: np.ndarray) -> np.ndarray:
    """Merton (1973) dividend-adjusted BSM delta."""
    T = np.maximum(T, 1e-5)
    iv = np.maximum(iv, 1e-5)
    K = np.maximum(K, 1e-5)
    S = max(S, 1e-5)
    d1 = (np.log(S / K) + (r - q + 0.5 * iv ** 2) * T) / (iv * np.sqrt(T))
    eq_T = np.exp(-q * T)
    c_delta = eq_T * norm.cdf(d1)
    return np.where(types == 'CE', c_delta, c_delta - eq_T)


def _bsm_gamma_vectorized(S: float, K: np.ndarray, T: np.ndarray, r: float, q: float, iv: np.ndarray) -> np.ndarray:
    """Merton (1973) dividend-adjusted BSM gamma."""
    T = np.maximum(T, 1e-5)
    iv = np.maximum(iv, 1e-5)
    K = np.maximum(K, 1e-5)
    S = max(S, 1e-5)
    d1 = (np.log(S / K) + (r - q + 0.5 * iv ** 2) * T) / (iv * np.sqrt(T))
    eq_T = np.exp(-q * T)
    return eq_T * norm.pdf(d1) / (S * iv * np.sqrt(T))


class GexRebalanceEngine:
    """
    Engine powering the 'One-Glance' Spot Shift & Option Buyer Radar.
    Provides sub-millisecond continuous repricing.
    """

    def __init__(self, lot_size: int = _DEFAULT_LOT_SIZE, risk_free_rate: float = _DEFAULT_R,
                 dividend_yield: float = _DEFAULT_Q):
        self.lot_size = lot_size
        self.r = risk_free_rate
        self.q = dividend_yield
        self.default_iv = 0.145
        self._active_setup = None

    def _prepare_chain_df(self, chain_df: pd.DataFrame, spot: float = 0.0) -> pd.DataFrame:
        if chain_df is None or chain_df.empty:
            return pd.DataFrame()
        df = chain_df.copy()
        df['type'] = df['type'].astype(str).str.upper()
        if 'strike' not in df.columns and 'strike_price' in df.columns:
            df['strike'] = df['strike_price'].astype(float)
        else:
            df['strike'] = df['strike'].astype(float)

        if 'oi' not in df.columns:
            df['oi'] = 0.0
        else:
            df['oi'] = df['oi'].fillna(0).astype(float)

        if 'iv' not in df.columns:
            df['iv'] = self.default_iv
        else:
            df['iv'] = df['iv'].fillna(self.default_iv).astype(float)
            df['iv'] = np.where(df['iv'] > 2.0, df['iv'] / 100.0, df['iv'])
            df['iv'] = np.where(df['iv'] <= 0.001, self.default_iv, df['iv'])

        if 'price' not in df.columns:
            df['price'] = df['ltp'] if 'ltp' in df.columns else 0.0
        df['price'] = df['price'].fillna(0).astype(float)

        if 'dte' not in df.columns:
            df['dte'] = 2.0
        else:
            df['dte'] = df['dte'].fillna(2.0).astype(float)

        # Fallback price to intrinsic + minimum time value if unobservable (0.0)
        if spot > 0:
            intrinsic = np.where(df['type'] == 'CE', np.maximum(0.0, spot - df['strike']), np.maximum(0.0, df['strike'] - spot))
            df['price'] = np.where(df['price'] <= 0.0, np.maximum(1.0, intrinsic), df['price'])

            # Merton (1973) dividend-adjusted Greeks vectorization if missing
            if 'delta' not in df.columns or 'gamma' not in df.columns:
                K_arr = df['strike'].values
                iv_arr = df['iv'].values
                T_arr = np.maximum(df['dte'].values / 365.0, 1e-5)
                types_arr = df['type'].values
                if 'delta' not in df.columns:
                    df['delta'] = _bsm_delta_vectorized(spot, K_arr, T_arr, self.r, self.q, iv_arr, types_arr)
                if 'gamma' not in df.columns:
                    df['gamma'] = _bsm_gamma_vectorized(spot, K_arr, T_arr, self.r, self.q, iv_arr)

        return df

    # ─────────────────────────────────────────────────────────────────────────
    # 1. WALL DETECTION & INVERSION
    # ─────────────────────────────────────────────────────────────────────────

    def detect_walls(self, df: pd.DataFrame, spot: float, range_pct: float = 0.04) -> Dict[str, Any]:
        """
        Identifies Call Wall 1, Call Wall 2, Put Wall 1, Put Wall 2,
        and checks for Wall Inversion (Wall 2 closer to spot than Wall 1).
        """
        if df.empty or spot <= 0:
            return {
                'call_wall_1': 0.0, 'call_wall_2': 0.0,
                'put_wall_1': 0.0, 'put_wall_2': 0.0,
                'ce_inverted': False, 'pe_inverted': False,
                'ce_runway': 0.0, 'pe_runway': 0.0
            }

        calls = df[df['type'] == 'CE']
        puts = df[df['type'] == 'PE']

        # ── Call Walls (Near & Above Spot) ──
        # Include strikes up to 35 pts below spot to track active breach
        calls_above = calls[(calls['strike'] >= spot - 35.0) & (calls['strike'] <= spot * (1 + range_pct))]
        if calls_above.empty:
            calls_above = calls[calls['strike'] >= spot - 35.0]

        cw1, cw2 = 0.0, 0.0
        if not calls_above.empty and calls_above['oi'].sum() > 0:
            top_calls = calls_above.groupby('strike')['oi'].sum().nlargest(2)
            cw1 = float(top_calls.index[0]) if len(top_calls) >= 1 else 0.0
            cw2 = float(top_calls.index[1]) if len(top_calls) >= 2 else 0.0

        # ── Put Walls (Near & Below Spot) ──
        # Include strikes up to 35 pts above spot to track active breach
        puts_below = puts[(puts['strike'] <= spot + 35.0) & (puts['strike'] >= spot * (1 - range_pct))]
        if puts_below.empty:
            puts_below = puts[puts['strike'] <= spot + 35.0]

        pw1, pw2 = 0.0, 0.0
        if not puts_below.empty and puts_below['oi'].sum() > 0:
            top_puts = puts_below.groupby('strike')['oi'].sum().nlargest(2)
            pw1 = float(top_puts.index[0]) if len(top_puts) >= 1 else 0.0
            pw2 = float(top_puts.index[1]) if len(top_puts) >= 2 else 0.0

        # Inversion Check: Wall 2 is closer to spot than Wall 1
        ce_inverted = False
        ce_runway = 0.0
        if cw1 > 0 and cw2 > 0:
            dist_w1 = abs(cw1 - spot)
            dist_w2 = abs(cw2 - spot)
            if dist_w2 < dist_w1:
                ce_inverted = True
                ce_runway = abs(cw1 - cw2)

        pe_inverted = False
        pe_runway = 0.0
        if pw1 > 0 and pw2 > 0:
            dist_pw1 = abs(spot - pw1)
            dist_pw2 = abs(spot - pw2)
            if dist_pw2 < dist_pw1:
                pe_inverted = True
                pe_runway = abs(pw2 - pw1)

        return {
            'call_wall_1': cw1,
            'call_wall_2': cw2,
            'put_wall_1': pw1,
            'put_wall_2': pw2,
            'ce_inverted': ce_inverted,
            'pe_inverted': pe_inverted,
            'ce_runway': round(ce_runway, 1),
            'pe_runway': round(pe_runway, 1)
        }

    # ─────────────────────────────────────────────────────────────────────────
    # 2. REBALANCE HORIZON: ZERO-GAMMA APEX & WALL 1 FORTRESS
    # ─────────────────────────────────────────────────────────────────────────

    def calculate_rebalance_horizon(self, df: pd.DataFrame, spot: float, direction: str = "BULLISH_CE",
                                   gex_res: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Scans spot in the direction of the move and calculates:
        - S_zero: Zero-Gamma Fuel Apex (where net dealer gamma flips from short to long,
          meaning dealer futures buying peaks and shuts off).
        - Terminal Fortress Wall 1 pin.
        """
        if df.empty or spot <= 0:
            return {'rebalance_target': round(spot + 80.0, 1), 'fuel_apex': round(spot + 80.0, 1), 'peak_lots': 15000}

        # Filter strikes within +- 3%
        sub = df[(df['strike'] >= spot * 0.97) & (df['strike'] <= spot * 1.03)].copy()
        if sub.empty:
            sub = df.copy()

        K_arr = sub['strike'].values
        iv_arr = sub['iv'].values
        types_arr = sub['type'].values
        oi_arr = sub['oi'].values
        dte_val = float(sub['dte'].iloc[0]) if 'dte' in sub.columns else 2.0
        T_arr = np.full_like(K_arr, max(dte_val / 365.0, 1e-5))

        # Standard signed dealer mapping: Calls = +1, Puts = -1
        # For Wall 2 (if trapped), signed gamma is negative
        dealer_sign = np.where(types_arr == 'CE', 1.0, -1.0)
        # Check shares
        is_shares = bool((oi_arr.max() > 100_000)) if len(oi_arr) > 0 else False
        total_shares = oi_arr if is_shares else (oi_arr * self.lot_size)

        # Baseline DEX at S0
        delta_0 = _bsm_delta_vectorized(spot, K_arr, T_arr, self.r, self.q, iv_arr, types_arr)
        dex_0 = np.sum(dealer_sign * total_shares * delta_0)

        # Scan grid
        if direction == "BULLISH_CE":
            scan_spots = np.linspace(spot, spot + 250.0, 51)
        else:
            scan_spots = np.linspace(spot, spot - 250.0, 51)

        gex_list = []
        hedge_lots_list = []

        for S_test in scan_spots:
            d_test = _bsm_delta_vectorized(S_test, K_arr, T_arr, self.r, self.q, iv_arr, types_arr)
            g_test = _bsm_gamma_vectorized(S_test, K_arr, T_arr, self.r, self.q, iv_arr)
            dex_test = np.sum(dealer_sign * total_shares * d_test)
            gex_test = np.sum(dealer_sign * total_shares * g_test)

            # Cumulative dealer futures hedge
            shares_to_hedge = -(dex_test - dex_0)
            lots_to_hedge = shares_to_hedge / self.lot_size

            gex_list.append(gex_test)
            hedge_lots_list.append(lots_to_hedge)

        gex_arr = np.array(gex_list)
        lots_arr = np.array(hedge_lots_list)

        # ── Find S_zero (Fuel Apex / Turning Point) ──
        s_zero = None
        peak_lots = 0

        # Direct anchor to GexEngine's zero_gamma_level if aligned with direction
        if gex_res and gex_res.get('zero_gamma_level', 0) > 0:
            zg = float(gex_res['zero_gamma_level'])
            if (direction == "BULLISH_CE" and zg > spot + 10.0) or (direction == "BEARISH_PE" and zg < spot - 10.0):
                s_zero = zg
            if gex_res.get('net_gex_lots_50pt'):
                peak_lots = abs(int(round(gex_res['net_gex_lots_50pt'])))

        if s_zero is None:
            if direction == "BULLISH_CE":
                max_idx = int(np.argmax(lots_arr))
                s_zero = float(scan_spots[max_idx])
                if peak_lots == 0:
                    peak_lots = int(round(abs(lots_arr[max_idx])))
            else:
                min_idx = int(np.argmin(lots_arr))
                s_zero = float(scan_spots[min_idx])
                if peak_lots == 0:
                    peak_lots = int(round(abs(lots_arr[min_idx])))

        # Ensure realistic bounds (at least 45 pts from spot, max 140 pts)
        if direction == "BULLISH_CE":
            s_zero = max(spot + 45.0, min(spot + 140.0, s_zero))
        else:
            s_zero = min(spot - 45.0, max(spot - 140.0, s_zero))

        if peak_lots == 0 and gex_res and gex_res.get('net_gex_lots_50pt'):
            peak_lots = abs(int(round(gex_res['net_gex_lots_50pt'])))
        if peak_lots == 0:
            peak_lots = 15000

        return {
            'rebalance_target': float(round(s_zero, 1)),
            'fuel_apex': float(round(s_zero, 1)),
            'peak_lots': peak_lots
        }

    # ─────────────────────────────────────────────────────────────────────────
    # 3. DUAL-STRIKE SELECTOR (ATM vs OTM GAMMA ROCKET)
    # ─────────────────────────────────────────────────────────────────────────

    def select_strikes(self, df: pd.DataFrame, spot: float, direction: str,
                       rebalance_target: float, dte: float) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Selects:
        1. Primary ATM strike (steady delta, high probability).
        2. OTM Gamma Rocket strike (specifically for DTE <= 1.5):
           Selects the cheap OTM strike inside the runway that explodes as spot rebalances.
        """
        opt_type = 'CE' if direction == "BULLISH_CE" else 'PE'
        sub = df[df['type'] == opt_type].copy()
        if sub.empty:
            return {}, {}

        # ── Primary ATM Strike ──
        sub['dist_atm'] = (sub['strike'] - spot).abs()
        atm_row = sub.loc[sub['dist_atm'].idxmin()]
        atm_strike = float(atm_row['strike'])
        atm_price = max(1.0, float(atm_row['price']))
        atm_delta = float(atm_row.get('delta', 0.50 if opt_type == 'CE' else -0.50))
        atm_gamma = float(atm_row.get('gamma', 0.002))

        # ── OTM Gamma Rocket Strike (0DTE / 1DTE Hero) ──
        # Search strikes strictly between spot and rebalance_target
        otm_strike = 0.0
        otm_price = 0.0
        otm_delta = 0.0
        otm_gamma = 0.0
        otm_active = False

        if direction == "BULLISH_CE":
            target_high = max(spot + 50.0, rebalance_target + 50.0)
            otm_candidates = sub[(sub['strike'] > spot) & (sub['strike'] <= target_high)].copy()
        else:
            target_low = min(spot - 50.0, rebalance_target - 50.0)
            otm_candidates = sub[(sub['strike'] < spot) & (sub['strike'] >= target_low)].copy()

        if not otm_candidates.empty:
            # Score by Gamma Convexity = Gamma / Price (Bang for Buck)
            otm_candidates['convexity'] = np.where(otm_candidates['price'] > 0.5,
                                                   otm_candidates.get('gamma', 0.001) / otm_candidates['price'], 0.0)
            # Filter for realistic sweet spot: price between ₹5 and ₹40
            sweet_spot = otm_candidates[(otm_candidates['price'] >= 5.0) & (otm_candidates['price'] <= 40.0)]
            best_otm = sweet_spot.loc[sweet_spot['convexity'].idxmax()] if not sweet_spot.empty else otm_candidates.iloc[0]

            otm_strike = float(best_otm['strike'])
            otm_price = max(0.5, float(best_otm['price']))
            otm_delta = float(best_otm.get('delta', 0.15 if opt_type == 'CE' else -0.15))
            otm_gamma = float(best_otm.get('gamma', 0.003))
            # Active on Expiry or 1DTE
            otm_active = (dte <= 1.5)

        primary_atm = {
            'strike': atm_strike,
            'type': opt_type,
            'price': atm_price,
            'delta': round(atm_delta, 2),
            'gamma': round(atm_gamma, 4)
        }

        otm_rocket = {
            'strike': otm_strike if otm_strike > 0 else (atm_strike + (50.0 if opt_type == 'CE' else -50.0)),
            'type': opt_type,
            'price': otm_price if otm_price > 0 else max(1.0, atm_price * 0.20),
            'delta': round(otm_delta, 2),
            'gamma': round(otm_gamma, 4),
            'is_active': otm_active
        }

        return primary_atm, otm_rocket

    # ─────────────────────────────────────────────────────────────────────────
    # 4. DIRECT PREMIUM CONVERTER
    # ─────────────────────────────────────────────────────────────────────────

    def convert_to_premium(self, strike_dict: Dict[str, Any], spot_move_pts: float,
                           stop_move_pts: float) -> Dict[str, Any]:
        """
        Uses Taylor expansion (Delta * dS + 0.5 * Gamma * dS^2) to compute
        exact option premium buy zone, target, and stop loss.
        """
        ltp = strike_dict.get('price', 50.0)
        delta = abs(strike_dict.get('delta', 0.50))
        gamma = strike_dict.get('gamma', 0.002)

        # Target 1 Gain: dP = Delta * dS + 0.5 * Gamma * dS^2
        dP_target = delta * spot_move_pts + 0.5 * gamma * (spot_move_pts ** 2)
        target_price = round(ltp + dP_target, 1)
        gain_pct = round(((target_price - ltp) / max(ltp, 0.1)) * 100.0, 1)

        # Runner Target (1.5x spot move)
        runner_move = spot_move_pts * 1.5
        dP_runner = delta * runner_move + 0.5 * gamma * (runner_move ** 2)
        runner_price = round(ltp + dP_runner, 1)
        runner_gain_pct = round(((runner_price - ltp) / max(ltp, 0.1)) * 100.0, 1)

        # Stop Loss Drop: dP_stop = - (Delta * dS_stop)
        dP_stop = delta * stop_move_pts
        stop_price = round(max(ltp * 0.40, ltp - dP_stop), 1)
        if stop_price >= ltp:
            stop_price = round(ltp * 0.85, 1)
        loss_pct = round(((stop_price - ltp) / max(ltp, 0.1)) * 100.0, 1)

        return {
            'strike': strike_dict.get('strike'),
            'type': strike_dict.get('type'),
            'current_price': round(ltp, 1),
            'buy_zone': f"₹{round(ltp, 1)} - ₹{round(ltp * 1.04, 1)}",
            'target_1': target_price,
            'target_gain_pct': gain_pct,
            'runner_target': runner_price,
            'runner_gain_pct': runner_gain_pct,
            'stop_loss': stop_price,
            'stop_loss_pct': loss_pct
        }

    # ─────────────────────────────────────────────────────────────────────────
    # 5. MASTER REAL-TIME EVALUATION
    # ─────────────────────────────────────────────────────────────────────────

    def evaluate(self, chain_df: pd.DataFrame, spot: float,
                 oi_velocity_data: Optional[Dict[str, Any]] = None,
                 dte: float = 2.0,
                 gex_res: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Master method called continuously. Returns the complete, zero-math,
        presentation-ready Option Buyer Radar payload.
        """
        if spot <= 0 or chain_df is None or chain_df.empty:
            return {'ok': False, 'status': 'WAITING_FOR_DATA'}

        df = self._prepare_chain_df(chain_df, spot=spot)
        walls = self.detect_walls(df, spot)

        # Synchronize with GexEngine walls if available
        if gex_res:
            if walls.get('call_wall_1', 0) <= 0 and gex_res.get('call_wall', 0) > 0:
                walls['call_wall_1'] = float(gex_res['call_wall'])
            if walls.get('put_wall_1', 0) <= 0 and gex_res.get('put_wall', 0) > 0:
                walls['put_wall_1'] = float(gex_res['put_wall'])

        # ── Determine Squeeze Direction & Setup ──
        ce_inverted = walls['ce_inverted']
        pe_inverted = walls['pe_inverted']
        is_inverted = ce_inverted or pe_inverted

        direction = "NEUTRAL"
        trigger_strike = 0.0
        fortress_wall_1 = 0.0
        runway_pts = 0.0

        if ce_inverted and not pe_inverted:
            direction = "BULLISH_CE"
            trigger_strike = walls['call_wall_2']
            fortress_wall_1 = walls['call_wall_1']
            runway_pts = walls['ce_runway']
        elif pe_inverted and not ce_inverted:
            direction = "BEARISH_PE"
            trigger_strike = walls['put_wall_2']
            fortress_wall_1 = walls['put_wall_1']
            runway_pts = walls['pe_runway']
        elif ce_inverted and pe_inverted:
            dist_ce = abs(walls['call_wall_2'] - spot)
            dist_pe = abs(spot - walls['put_wall_2'])
            if dist_ce < dist_pe:
                direction = "BULLISH_CE"
                trigger_strike = walls['call_wall_2']
                fortress_wall_1 = walls['call_wall_1']
                runway_pts = walls['ce_runway']
            else:
                direction = "BEARISH_PE"
                trigger_strike = walls['put_wall_2']
                fortress_wall_1 = walls['put_wall_1']
                runway_pts = walls['pe_runway']
        else:
            # Normal market (Wall 1 closer than Wall 2, no inverted vacuum runway)
            # Pick immediate barrier based on proximity or Net GEX regime
            cw1 = walls.get('call_wall_1', 0.0)
            cw2 = walls.get('call_wall_2', 0.0)
            pw1 = walls.get('put_wall_1', 0.0)
            pw2 = walls.get('put_wall_2', 0.0)

            dist_ce = abs(cw1 - spot) if cw1 > 0 else 999.0
            dist_pe = abs(spot - pw1) if pw1 > 0 else 999.0

            # Bias check: if close, bias by net GEX
            if gex_res and abs(dist_ce - dist_pe) < 30.0:
                is_bullish = gex_res.get('net_gex', 0) >= 0
            else:
                is_bullish = dist_ce < dist_pe

            if is_bullish and cw1 > 0:
                direction = "BULLISH_CE"
                trigger_strike = cw1
                fortress_wall_1 = cw2 if cw2 > cw1 else cw1 + 100.0
                runway_pts = abs(fortress_wall_1 - trigger_strike)
            elif pw1 > 0:
                direction = "BEARISH_PE"
                trigger_strike = pw1
                fortress_wall_1 = pw2 if (pw2 > 0 and pw2 < pw1) else max(50.0, pw1 - 100.0)
                runway_pts = abs(trigger_strike - fortress_wall_1)
            else:
                direction = "BULLISH_CE"
                trigger_strike = round((spot + 50.0) / 50.0) * 50.0
                fortress_wall_1 = trigger_strike + 100.0
                runway_pts = 100.0

        if trigger_strike <= 0:
            trigger_strike = round(spot / 50.0) * 50.0
        if fortress_wall_1 <= 0:
            fortress_wall_1 = trigger_strike + (100.0 if direction == "BULLISH_CE" else -100.0)
        if runway_pts <= 0:
            runway_pts = abs(fortress_wall_1 - trigger_strike)

        # ── Compute Rebalance Target (Zero-Gamma Fuel Apex) ──
        horizon = self.calculate_rebalance_horizon(df, spot, direction=direction, gex_res=gex_res)
        rebalance_target = horizon['rebalance_target']
        peak_lots = horizon['peak_lots']

        # Ensure rebalance_target lies inside the runway beyond the trigger strike
        if direction == "BULLISH_CE":
            if rebalance_target <= trigger_strike:
                step = min(abs(fortress_wall_1 - trigger_strike) * 0.5, 80.0) if fortress_wall_1 > trigger_strike else 75.0
                rebalance_target = round(trigger_strike + max(40.0, step), 1)
        else:
            if rebalance_target >= trigger_strike:
                step = min(abs(trigger_strike - fortress_wall_1) * 0.5, 80.0) if fortress_wall_1 < trigger_strike else 75.0
                rebalance_target = round(trigger_strike - max(40.0, step), 1)

        # ── Check OI Velocity & Ignition Trigger ──
        w2_oi_vel = 0.0
        dual_wall_confirmed = False
        velocity_speed = "MODERATE"

        if oi_velocity_data:
            vel_map = oi_velocity_data.get('vel_by_strike', {})
            accel_map = oi_velocity_data.get('accel_by_strike', {})
            opt_type = 'CE' if direction == "BULLISH_CE" else 'PE'
            w2_oi_vel = vel_map.get((trigger_strike, opt_type), 0.0)
            w2_accel = accel_map.get((trigger_strike, opt_type), 0.0)

            opp_type = 'PE' if direction == "BULLISH_CE" else 'CE'
            opp_support_strike = spot - 50.0 if direction == "BULLISH_CE" else spot + 50.0
            opp_vel = vel_map.get((round(opp_support_strike / 50.0) * 50.0, opp_type), 0.0)
            if opp_vel > 20_000:
                dual_wall_confirmed = True

            if abs(w2_oi_vel) > 80_000 or (w2_oi_vel < -40_000 and w2_accel < -5_000):
                velocity_speed = "VIOLENT_CAPITULATION" if abs(w2_oi_vel) > 80_000 else "ACCELERATING_CAPITULATION"
            elif w2_oi_vel < -30_000 and w2_accel > 5_000:
                velocity_speed = "EXHAUSTING_CAPITULATION"
            elif abs(w2_oi_vel) > 30_000:
                velocity_speed = "ACTIVE_SQUEEZE"
            else:
                velocity_speed = "SLOW_DRIFT"

        # Check Price Position vs Trigger Strike
        dist_to_trigger = trigger_strike - spot if direction == "BULLISH_CE" else spot - trigger_strike
        is_coiling = abs(dist_to_trigger) <= 40.0
        is_broken = (spot >= trigger_strike + 2.0) if direction == "BULLISH_CE" else (spot <= trigger_strike - 2.0)

        # Active state persistence
        if self._active_setup and self._active_setup.get('active'):
            saved_dir = self._active_setup.get('direction')
            saved_target = self._active_setup.get('target', rebalance_target)
            saved_trigger = self._active_setup.get('trigger', trigger_strike)

            if saved_dir == direction:
                trigger_strike = saved_trigger
                rebalance_target = saved_target
                direction = saved_dir
                is_broken = True

        # Rebalance progress
        total_move = max(1.0, abs(rebalance_target - trigger_strike))
        current_progress = abs(spot - trigger_strike) if is_broken else 0.0
        progress_pct = max(0.0, min(100.0, round((current_progress / total_move) * 100.0, 1)))

        # Squeeze State Transitions
        if progress_pct >= 90.0:
            status = "TARGET_REACHED"
            status_desc = f"🎯 Rebalance Target {rebalance_target:.0f} reached! Dealer futures buying exhausted. Book profit!"
            self._active_setup = {'active': False}
        elif is_broken and w2_oi_vel < 0:
            status = "IGNITED"
            prefix = "ACTIVE SQUEEZE DETECTED" if is_inverted else "BREAKOUT DETECTED"
            status_desc = f"⚡ {prefix}: {trigger_strike:.0f} toll gate broken with -{abs(w2_oi_vel)/1000:.0f}k unwinding!"
            self._active_setup = {'active': True, 'trigger': trigger_strike, 'target': rebalance_target, 'direction': direction}
        elif is_broken and self._active_setup and self._active_setup.get('active'):
            status = "REBALANCING"
            status_desc = f"Spot rebalancing across runway ({progress_pct:.0f}% complete). Target: {rebalance_target:.0f}."
        elif is_broken and w2_oi_vel >= 0:
            status = "ARMED"
            status_desc = f"Spot crossed {trigger_strike:.0f}, but OI velocity not unwinding yet. Wait for seller capitulation."
        elif is_coiling:
            status = "COILING"
            gate_name = "toll gate (Wall ②)" if is_inverted else "barrier (Wall ①)"
            status_desc = f"Spot coiling {abs(dist_to_trigger):.1f} pts from {trigger_strike:.0f} {gate_name}. Runway open to {fortress_wall_1:.0f}."
        else:
            status = "MONITORING"
            if is_inverted:
                status_desc = f"Wall 2 at {trigger_strike:.0f} is closer than Wall 1 ({fortress_wall_1:.0f}). Watching for approach."
            else:
                status_desc = f"Normal regime: Spot at {spot:.1f}. Monitoring {trigger_strike:.0f} barrier (runway to {fortress_wall_1:.0f})."

        # Invalidation Stop Level (12 pts back behind trigger)
        invalidation_stop = trigger_strike - 12.0 if direction == "BULLISH_CE" else trigger_strike + 12.0
        if is_broken and ((direction == "BULLISH_CE" and spot < invalidation_stop) or
                          (direction == "BEARISH_PE" and spot > invalidation_stop)):
            status = "INVALIDATED"
            status_desc = f"❌ Squeeze failed. Spot fell back below {invalidation_stop:.0f}. Hard exit."
            self._active_setup = {'active': False}

        # ── Dual-Strike Recommendation & Premium ₹ Levels ──
        atm_strike_dict, otm_strike_dict = self.select_strikes(df, spot, direction, rebalance_target, dte)
        expected_spot_pts = max(35.0, abs(rebalance_target - spot) if not is_broken else abs(rebalance_target - trigger_strike))
        stop_pts = 14.0

        primary_premium = self.convert_to_premium(atm_strike_dict, expected_spot_pts, stop_pts)
        otm_premium = self.convert_to_premium(otm_strike_dict, expected_spot_pts, stop_pts)
        otm_premium['is_active'] = otm_strike_dict.get('is_active', False)

        opt_name = "CE" if direction == "BULLISH_CE" else "PE"
        if status == "IGNITED":
            action_summary = f"BUY {primary_premium['strike']:.0f} {opt_name} ON BREAK | TARGET: {rebalance_target:.0f} (+{expected_spot_pts:.0f} pts)"
        elif status == "COILING" or status == "ARMED":
            action_summary = f"WAIT: Prepare to BUY {primary_premium['strike']:.0f} {opt_name} when spot breaks {trigger_strike:.0f}"
        elif status == "TARGET_REACHED":
            action_summary = f"BOOK 75% PROFIT: Rebalance target {rebalance_target:.0f} hit! Trail runners."
        elif status == "REBALANCING":
            action_summary = f"HOLD {opt_name}: Squeeze underway ({progress_pct:.0f}% towards {rebalance_target:.0f})."
        else:
            action_summary = f"MONITORING: Spot {abs(dist_to_trigger):.0f} pts from {trigger_strike:.0f} {opt_name} barrier."

        return {
            'ok': True,
            'status': status,
            'status_desc': status_desc,
            'action_summary': action_summary,
            'direction': direction,
            'spot': round(spot, 1),
            'trigger_strike': trigger_strike,
            'rebalance_target': rebalance_target,
            'terminal_fortress': fortress_wall_1,
            'runway_pts': runway_pts,
            'invalidation_stop': invalidation_stop,
            'progress_pct': progress_pct,
            'dealer_fuel_lots': peak_lots,
            'velocity_speed': velocity_speed,
            'w2_oi_velocity': w2_oi_vel,
            'dual_wall_confirmed': dual_wall_confirmed,
            'primary_option': primary_premium,
            'otm_gamma_rocket': otm_premium,
            'walls': walls,
            'timestamp': time.strftime("%H:%M:%S")
        }
