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
        otm_theta = 0.0
        otm_active = False
        best_otm = None

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
            otm_theta = float(best_otm.get('theta', 0.0))
            # Active on Expiry or 1DTE
            otm_active = (dte <= 1.5)

        primary_atm = {
            'strike': atm_strike,
            'type': opt_type,
            'price': atm_price,
            'delta': round(atm_delta, 2),
            'gamma': round(atm_gamma, 4),
            'theta': round(float(atm_row.get('theta', 0.0)), 2)
        }

        otm_rocket = {
            'strike': otm_strike if otm_strike > 0 else (atm_strike + (50.0 if opt_type == 'CE' else -50.0)),
            'type': opt_type,
            'price': otm_price if otm_price > 0 else max(1.0, atm_price * 0.20),
            'delta': round(otm_delta, 2),
            'gamma': round(otm_gamma, 4),
            'theta': round(otm_theta, 2),
            'is_active': otm_active
        }

        return primary_atm, otm_rocket

    # ─────────────────────────────────────────────────────────────────────────
    # 4. DIRECT PREMIUM CONVERTER WITH TIERED ROI GUARANTEES & THETA BURN
    # ─────────────────────────────────────────────────────────────────────────

    def convert_to_premium(self, strike_dict: Dict[str, Any], spot_move_pts: float,
                           stop_move_pts: float, tier: str = "TIER_1_QUICK_MOMENTUM",
                           is_rocket: bool = False) -> Dict[str, Any]:
        """
        Uses Merton (1973) Taylor expansion (Delta * dS + 0.5 * Gamma * dS^2)
        and calibrates targets to institutional ROI tiers:
        - TIER 1 (Quick Momentum): Target 1 >= +20% to +28% ROI, Target 2 >= +35% to +45% ROI, SL ~12-15%
        - TIER 2 (Runway Squeeze): Target 1 >= +45% to +60% ROI, Target 2 >= +75% to +90% ROI, SL ~18-20%
        - TIER 3 (0DTE Expiry Mega Move): Target 1 = +100% (2x), Target 2 = +250% to +400% (Hero), SL -50%
        """
        ltp = max(0.5, float(strike_dict.get('price', 50.0)))
        delta = max(0.05, abs(float(strike_dict.get('delta', 0.50))))
        gamma = max(0.0001, float(strike_dict.get('gamma', 0.002)))
        raw_theta = abs(float(strike_dict.get('theta', 0.0)))

        # Taylor expansion base spot move
        dP_raw = delta * spot_move_pts + 0.5 * gamma * (spot_move_pts ** 2)

        # 15-Minute Theta Decay Burn Calculation
        # A normal Indian trading session is 375 minutes (25 fifteen-minute blocks)
        # On 0DTE afternoon, decay accelerates significantly (1.35x standard daily theta)
        accel = 1.35 if tier == "TIER_3_EXPIRY_MEGA_MOVE" else 1.0
        effective_theta = raw_theta if raw_theta > 0.1 else max(1.0, ltp * (0.32 if tier == "TIER_3_EXPIRY_MEGA_MOVE" else 0.14))
        theta_15m_pts = round((effective_theta / 25.0) * accel, 1)
        theta_15m_inr = round(theta_15m_pts * float(self.lot_size), 0)
        theta_burn_str = f"-₹{theta_15m_pts:.1f} pts (-₹{theta_15m_inr:,.0f}/lot)"

        if is_rocket and tier == "TIER_3_EXPIRY_MEGA_MOVE":
            # 0DTE Expiry Mega Move (Hero or Zero Gamma Rocket)
            target_1_gain = ltp * 1.00   # +100% (2x double)
            runner_gain = ltp * 2.50     # +250% (3.5x runner)
            target_price = round(ltp + target_1_gain, 1)
            runner_price = round(ltp + runner_gain, 1)
            stop_price = round(max(0.5, ltp * 0.50), 1)  # 50% max capital risk
            gain_pct = 100.0
            runner_gain_pct = 250.0
            loss_pct = -50.0
        elif tier == "TIER_2_RUNWAY_SQUEEZE":
            # Vacuum Runway Squeeze: Target 1 at Zero-Gamma apex (+45% min ROI)
            target_1_gain = max(ltp * 0.45, dP_raw)
            runner_gain = max(ltp * 0.75, target_1_gain * 1.55)
            target_price = round(ltp + target_1_gain, 1)
            runner_price = round(ltp + runner_gain, 1)
            stop_price = round(max(ltp * 0.50, ltp - (delta * stop_move_pts * 1.1)), 1)
            if stop_price >= ltp * 0.90:
                stop_price = round(ltp * 0.82, 1)
            gain_pct = round(((target_price - ltp) / max(ltp, 0.1)) * 100.0, 1)
            runner_gain_pct = round(((runner_price - ltp) / max(ltp, 0.1)) * 100.0, 1)
            loss_pct = round(((stop_price - ltp) / max(ltp, 0.1)) * 100.0, 1)
        else:
            # TIER 1: Intraday Quick Momentum (Guaranteed >= +20% to +25% Target 1 ROI)
            target_1_gain = max(ltp * 0.22, dP_raw)
            runner_gain = max(ltp * 0.38, target_1_gain * 1.6)
            target_price = round(ltp + target_1_gain, 1)
            runner_price = round(ltp + runner_gain, 1)
            # Strict stop loss: 12-15% max drop
            dP_stop = delta * stop_move_pts
            stop_price = round(max(ltp * 0.85, ltp - dP_stop), 1)
            if stop_price >= ltp * 0.92:
                stop_price = round(ltp * 0.86, 1)
            gain_pct = round(((target_price - ltp) / max(ltp, 0.1)) * 100.0, 1)
            runner_gain_pct = round(((runner_price - ltp) / max(ltp, 0.1)) * 100.0, 1)
            loss_pct = round(((stop_price - ltp) / max(ltp, 0.1)) * 100.0, 1)

        rr_ratio = round(abs(gain_pct) / max(abs(loss_pct), 1.0), 2)

        return {
            'strike': strike_dict.get('strike'),
            'type': strike_dict.get('type'),
            'current_price': round(ltp, 1),
            'buy_zone': f"₹{round(ltp, 1)} - ₹{round(ltp * 1.03, 1)}",
            'target_1': target_price,
            'target_gain_pct': gain_pct,
            'runner_target': runner_price,
            'runner_gain_pct': runner_gain_pct,
            'stop_loss': stop_price,
            'stop_loss_pct': loss_pct,
            'rr_ratio': rr_ratio,
            'theta': round(effective_theta, 2),
            'theta_15m_pts': theta_15m_pts,
            'theta_15m_inr': theta_15m_inr,
            'theta_burn_15m_str': theta_burn_str
        }

    # ─────────────────────────────────────────────────────────────────────────
    # 5. MASTER REAL-TIME EVALUATION WITH MULTI-MODULE CONFLUENCE
    # ─────────────────────────────────────────────────────────────────────────

    def evaluate(self, chain_df: pd.DataFrame, spot: float,
                 oi_velocity_data: Optional[Dict[str, Any]] = None,
                 dte: float = 2.0,
                 gex_res: Optional[Dict[str, Any]] = None,
                 intraday_signal_data: Optional[Dict[str, Any]] = None,
                 gamma_explosion_data: Optional[Dict[str, Any]] = None,
                 dealer_data: Optional[Dict[str, Any]] = None,
                 vol_data: Optional[Dict[str, Any]] = None,
                 master_verdict: Optional[Dict[str, Any]] = None,
                 regime_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Master method called continuously. Performs deep multi-module confluence
        evaluation across GEX, OI velocity, microstructural absorption, econometric vol,
        and magnetic pin releases.

        Enforces strict chop/suppression filtering so traders are never fed random,
        low-probability trade ideas.
        """
        if spot <= 0 or chain_df is None or chain_df.empty:
            return {'ok': False, 'status': 'WAITING_FOR_DATA', 'trade_ready': False}

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
        w2_accel = 0.0
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

        # ── MULTI-MODULE CONFLUENCE EVALUATION & SENSORS ──
        confluence_score = 50.0  # Base neutral
        rejection_reasons: List[str] = []
        checklist: Dict[str, Dict[str, Any]] = {
            'regime_swing': {'status': 'NEUTRAL', 'label': 'Regime Swing', 'detail': 'Normal Vol'},
            'dealer_gex':   {'status': 'NEUTRAL', 'label': 'Dealer GEX', 'detail': 'Balanced'},
            'oi_flow':      {'status': 'NEUTRAL', 'label': 'OI Flow', 'detail': 'Stable'},
            'absorption':   {'status': 'NEUTRAL', 'label': 'Absorption', 'detail': 'None'},
            'vol_skew':     {'status': 'NEUTRAL', 'label': 'Vol Asymmetry', 'detail': 'Symmetric'},
            'pin_cascade':  {'status': 'NEUTRAL', 'label': 'Pin Status', 'detail': 'No Active Pin'}
        }

        # 1. Swing Quality & Session Phase (Gatekeeper)
        swing_q = "UNKNOWN"
        adr_pct = 0.0
        phase = ""
        if intraday_signal_data and 'swing_quality' in intraday_signal_data:
            sq = intraday_signal_data['swing_quality']
            if isinstance(sq, dict):
                swing_q = str(sq.get('quality', 'UNKNOWN')).upper()
                adr_pct = float(sq.get('adr_pct', 0.0))
                phase = str(sq.get('session_phase', ''))
            elif isinstance(sq, str):
                swing_q = sq.upper()

            if swing_q == 'CHOPPY':
                confluence_score -= 28.0
                checklist['regime_swing'] = {'status': 'FAIL', 'label': 'Chop Warning', 'detail': f"ADR {adr_pct:.0f}% (Chop)"}
                rejection_reasons.append("Market in Chop/Compression regime (ADR compressed) — high theta decay risk.")
            elif swing_q in ('TRENDING', 'CLEAN_TREND'):
                confluence_score += 16.0
                checklist['regime_swing'] = {'status': 'PASS', 'label': 'Trending Day', 'detail': f"ADR {adr_pct:.0f}% Expansion"}
            elif swing_q == 'COILED':
                confluence_score += 10.0
                checklist['regime_swing'] = {'status': 'PASS', 'label': 'Coiled Spring', 'detail': 'Range Compression'}

            if phase == 'MID_TREND':
                confluence_score += 6.0
            elif phase == 'EXPIRY_HEAT':
                confluence_score += 10.0

        # 2. Dealer Gamma Positioning & Gamma Flip
        net_gex = 0.0
        if gex_res and 'net_gex' in gex_res:
            net_gex = float(gex_res['net_gex'])
        elif dealer_data and 'net_gex_shares' in dealer_data:
            net_gex = float(dealer_data['net_gex_shares']) * spot

        zero_gamma = float(gex_res.get('zero_gamma_level', 0.0)) if gex_res else 0.0

        if net_gex < 0:
            confluence_score += 18.0
            checklist['dealer_gex'] = {'status': 'PASS', 'label': 'Short Gamma', 'detail': 'Dealers Accelerating'}
        elif net_gex > 5e8:
            confluence_score -= 12.0
            checklist['dealer_gex'] = {'status': 'WARN', 'label': 'Long Gamma', 'detail': 'Dealers Suppressing'}
            rejection_reasons.append("Dealers in Long Gamma — market moves are dampened/mean-reverting.")
        else:
            checklist['dealer_gex'] = {'status': 'NEUTRAL', 'label': 'GEX Neutral', 'detail': 'Balanced Gamma'}

        if zero_gamma > 0:
            if direction == "BULLISH_CE" and spot > zero_gamma:
                confluence_score += 6.0
            elif direction == "BEARISH_PE" and spot < zero_gamma:
                confluence_score += 6.0

        # 3. OI Velocity & Writer Capitulation
        if w2_oi_vel < -80_000:
            confluence_score += 24.0
            checklist['oi_flow'] = {'status': 'PASS', 'label': 'Capitulation', 'detail': f"{abs(w2_oi_vel)/1000:.0f}k/m Unwinding"}
        elif w2_oi_vel < -30_000:
            confluence_score += 14.0
            checklist['oi_flow'] = {'status': 'PASS', 'label': 'Unwinding', 'detail': f"{abs(w2_oi_vel)/1000:.0f}k/m Unwinding"}
        elif w2_oi_vel > 35_000:
            confluence_score -= 28.0
            checklist['oi_flow'] = {'status': 'FAIL', 'label': 'Writers Defending', 'detail': f"+{w2_oi_vel/1000:.0f}k/m Defending"}
            rejection_reasons.append(f"Writers actively defending {trigger_strike:.0f} (+{w2_oi_vel/1000:.0f}k/m added) — fakeout danger.")
        elif dual_wall_confirmed:
            confluence_score += 8.0
            checklist['oi_flow'] = {'status': 'PASS', 'label': 'Support Building', 'detail': 'Opposite Wall Fortified'}

        # 4. Microstructural Absorption
        if intraday_signal_data and 'absorption' in intraday_signal_data:
            abs_info = intraday_signal_data['absorption']
            if isinstance(abs_info, dict):
                setup_qual = abs_info.get('setup_quality', 'NONE')
                wick_bars = abs_info.get('wick_bars', 0)
                vol_acc = abs_info.get('vol_accel', 1.0)
                if setup_qual == 'STRONG':
                    confluence_score += 15.0
                    checklist['absorption'] = {'status': 'PASS', 'label': 'Strong Absorption', 'detail': f"{wick_bars} bars · {vol_acc:.1f}x Vol"}
                elif setup_qual == 'MODERATE':
                    confluence_score += 10.0
                    checklist['absorption'] = {'status': 'PASS', 'label': 'Mod Absorption', 'detail': f"{wick_bars} rejection bars"}

        # 5. Econometric Volatility & Jump Dynamics
        if vol_data and isinstance(vol_data, dict):
            semi = vol_data.get('semi_variance')
            if isinstance(semi, dict):
                vai = float(semi.get('vai', 0.0))
                if direction == "BEARISH_PE" and vai > 0.15:
                    confluence_score += 10.0
                    checklist['vol_skew'] = {'status': 'PASS', 'label': 'Toxic Downside', 'detail': f"VAI +{vai:.2f} favors PE"}
                elif direction == "BULLISH_CE" and vai < -0.15:
                    confluence_score += 10.0
                    checklist['vol_skew'] = {'status': 'PASS', 'label': 'Bullish Grind', 'detail': f"VAI {vai:.2f} favors CE"}

            jump_data = vol_data.get('jump_decomposition')
            if isinstance(jump_data, dict):
                if jump_data.get('jump_regime') == 'JUMP_REGIME' or jump_data.get('jump_ratio', 0) > 0.18:
                    confluence_score += 6.0

            fvrp = vol_data.get('forward_vrp')
            if isinstance(fvrp, dict):
                vrp = float(fvrp.get('vrp_5d', 0.0))
                if vrp < -1.0:
                    confluence_score += 5.0
                elif vrp > 3.0:
                    confluence_score -= 5.0
                    rejection_reasons.append(f"High VRP (+{vrp:.1f}) — Implied Vol is expensive and crushing.")

        # 6. Magnetic Pinning Release & Cascade Risk
        top_pin_risk = "STABLE"
        pin_duration_str = ""
        if gamma_explosion_data and isinstance(gamma_explosion_data, dict):
            pins = gamma_explosion_data.get('active_pins')
            if isinstance(pins, list) and pins:
                top_p = pins[0]
                if isinstance(top_p, dict):
                    pin_strike = top_p.get('strike', 0)
                    pin_dur = top_p.get('duration_secs', 0)
                    top_pin_risk = top_p.get('unpinning_risk', 'STABLE')
                    pin_duration_str = top_p.get('duration_str', '')
                    if top_pin_risk in ('IMMINENT', 'HIGH') and pin_dur >= 3600:
                        confluence_score += 14.0
                        checklist['pin_cascade'] = {'status': 'PASS', 'label': f'Unpinning {pin_strike:.0f}', 'detail': f'Release Risk {top_pin_risk}'}
                    elif pin_dur > 0:
                        checklist['pin_cascade'] = {'status': 'NEUTRAL', 'label': f'Pinned {pin_strike:.0f}', 'detail': pin_duration_str}

        # 7. Master Signal Consensus Alignment
        if master_verdict:
            mv = master_verdict.get('verdict', 'NEUTRAL')
            if (direction == "BULLISH_CE" and "BULLISH" in mv) or (direction == "BEARISH_PE" and "BEARISH" in mv):
                confluence_score += 8.0
            elif "AVOID" in mv or "NEUTRAL" in mv:
                confluence_score -= 6.0

        # 8. Realized Volatility Regime & Volatility Cone Fusion
        if regime_data and isinstance(regime_data, dict):
            macro = regime_data.get('macro', {})
            m_stage = macro.get('stage', '')
            rv_dict = regime_data.get('rv', {})
            t_slope = rv_dict.get('term_slope', 0.0)

            if 'COMPRESSION' in m_stage and dte > 6.0:
                confluence_score += 10.0
                checklist['vol_skew'] = {'status': 'PASS', 'label': 'Macro Squeeze', 'detail': 'Vol Cone at Lows'}
            if t_slope > 1.0 and direction == "BEARISH_PE":
                # Inverted term structure (short-term RV > long-term RV) indicates panic / downside rush
                confluence_score += 8.0

        confluence_score = round(max(0.0, min(100.0, confluence_score)), 1)

        # ── MULTI-EXPIRY SETUP ARCHETYPE CLASSIFIER & DYNAMIC TIME HORIZONS ──
        if dte <= 1.2:
            setup_archetype = "0DTE_GAMMA_ROCKET"
            archetype_name = "0DTE GAMMA ROCKET 🔥 (Expiry Squeeze)"
            archetype_badge_color = "#ff7043"
            recommended_horizon = "15 – 35 Mins (Fast Squeeze)"
            max_hold_mins = 35
            hard_time_stop = "EXIT if spot stalls <20 pts after 20 mins — 0DTE theta acceleration will destroy premium."
            active_tier = "TIER_3_EXPIRY_MEGA_MOVE"
            tier_name = "0DTE EXPIRY MEGA MOVE 🔥 (100-300%+ ROI)"

            # On 0DTE, if spot breaks the trigger with dealer negative GEX or unwinding,
            # gamma acceleration overpowers early session ADR compression
            if is_broken and (net_gex < 0 or w2_oi_vel < 0):
                confluence_score = max(confluence_score, 78.0)
        elif dte <= 6.0:
            setup_archetype = "WEEKLY_MOMENTUM_BREAKOUT"
            archetype_name = "WEEKLY MOMENTUM SQUEEZE ⚡ (Absorption Retest)"
            archetype_badge_color = "#00f0ff"
            recommended_horizon = "1 – 3 Hours (Session Trend)"
            max_hold_mins = 150
            hard_time_stop = "EXIT if 5-min candle closes back across VWAP or 60 mins without directional follow-through."
            if is_inverted or runway_pts >= 110.0 or confluence_score >= 75.0:
                active_tier = "TIER_2_RUNWAY_SQUEEZE"
                tier_name = "VACUUM RUNWAY SQUEEZE ⚡ (45-80% ROI)"
            else:
                active_tier = "TIER_1_QUICK_MOMENTUM"
                tier_name = "INTRADAY MOMENTUM (20-35% ROI)"
        else:
            setup_archetype = "MACRO_VOL_EXPANSION"
            archetype_name = "MACRO VOL EXPANSION 🌐 (Multi-Day Swing)"
            archetype_badge_color = "#b388ff"
            recommended_horizon = "1 – 2 Trading Days"
            max_hold_mins = 900
            hard_time_stop = "EXIT on 1D RV trailing stop or daily close below breakout pivot."
            active_tier = "TIER_1_QUICK_MOMENTUM"
            tier_name = "MACRO VOL EXPANSION 🌐 (30-60% ROI)"

        # Strict Chop / Suppression Gate (0DTE exempt from low early morning ADR veto):
        is_vetoed = (w2_oi_vel > 35_000) or (swing_q == 'CHOPPY' and confluence_score < 72.0 and dte > 1.2)
        trade_ready = (confluence_score >= 65.0) and not is_vetoed

        # ── Squeeze State Transitions ──
        invalidation_stop = trigger_strike - 12.0 if direction == "BULLISH_CE" else trigger_strike + 12.0

        if progress_pct >= 90.0:
            status = "TARGET_REACHED"
            status_desc = f"🎯 Rebalance Target {rebalance_target:.0f} reached! Dealer futures buying exhausted. Book profit!"
            self._active_setup = {'active': False}
        elif is_broken and ((direction == "BULLISH_CE" and spot < invalidation_stop) or
                            (direction == "BEARISH_PE" and spot > invalidation_stop)):
            status = "INVALIDATED"
            status_desc = f"❌ Squeeze failed. Spot fell back below {invalidation_stop:.0f}. Hard exit."
            self._active_setup = {'active': False}
        elif not trade_ready:
            if is_coiling and not is_vetoed:
                status = "COILING"
                gate_name = "toll gate (Wall ②)" if is_inverted else "barrier (Wall ①)"
                status_desc = f"Spot coiling {abs(dist_to_trigger):.1f} pts from {trigger_strike:.0f} {gate_name}. Awaiting ignition confluence ({confluence_score:.0f}/100)."
            else:
                status = "STAND_ASIDE"
                if rejection_reasons:
                    status_desc = f"🛡️ STAND ASIDE: {rejection_reasons[0]}"
                elif swing_q == 'CHOPPY':
                    status_desc = f"🛡️ STAND ASIDE: Choppy market regime. Cash is a position. Awaiting trending flow."
                else:
                    status_desc = f"🛡️ STAND ASIDE: Confluence score {confluence_score:.0f}/100 below 65 minimum threshold."
        elif is_broken and w2_oi_vel < 0:
            status = "IGNITED"
            prefix = "ACTIVE SQUEEZE DETECTED" if is_inverted else "BREAKOUT DETECTED"
            status_desc = f"⚡ {prefix}: {trigger_strike:.0f} toll gate broken with -{abs(w2_oi_vel)/1000:.0f}k unwinding! [{archetype_name}]"
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
            status_desc = f"Spot coiling {abs(dist_to_trigger):.1f} pts from {trigger_strike:.0f} {gate_name}. High confluence ({confluence_score:.0f}/100) — prepare for break!"
        else:
            status = "MONITORING"
            status_desc = f"Monitoring {trigger_strike:.0f} barrier. Confluence: {confluence_score:.0f}/100."

        # ── Dual-Strike Recommendation & Premium ₹ Levels ──
        atm_strike_dict, otm_strike_dict = self.select_strikes(df, spot, direction, rebalance_target, dte)
        expected_spot_pts = max(35.0, abs(rebalance_target - spot) if not is_broken else abs(rebalance_target - trigger_strike))
        stop_pts = 14.0

        primary_premium = self.convert_to_premium(
            atm_strike_dict, expected_spot_pts, stop_pts, tier=active_tier, is_rocket=False
        )
        otm_premium = self.convert_to_premium(
            otm_strike_dict, expected_spot_pts, stop_pts, tier=active_tier, is_rocket=True
        )
        otm_premium['is_active'] = otm_strike_dict.get('is_active', False) or (active_tier == "TIER_3_EXPIRY_MEGA_MOVE")

        opt_name = "CE" if direction == "BULLISH_CE" else "PE"

        # ── Structured Exit Triggers ──
        exit_triggers = {
            'target_1': {
                'label': 'Target 1 (Book 70%)',
                'spot': rebalance_target,
                'primary_premium': primary_premium.get('target_1', 0),
                'otm_premium': otm_premium.get('target_1', 0),
                'rationale': f'Zero-Gamma Fuel Apex ({rebalance_target:.0f}) reached'
            },
            'target_2': {
                'label': 'Target 2 (Runner)',
                'spot': fortress_wall_1,
                'primary_premium': primary_premium.get('runner_target', 0),
                'otm_premium': otm_premium.get('runner_target', 0),
                'rationale': f'Terminal Fortress Wall ({fortress_wall_1:.0f}) extension'
            },
            'structural_stop': {
                'label': 'Structural Stop Loss',
                'spot': invalidation_stop,
                'primary_premium': primary_premium.get('stop_loss', 0),
                'otm_premium': otm_premium.get('stop_loss', 0),
                'rationale': f'Spot failure back across {invalidation_stop:.0f}'
            },
            'time_stop': {
                'label': 'Hard Time Stop',
                'max_duration': f"{max_hold_mins} Mins",
                'rule': hard_time_stop,
                'theta_burn_15m': primary_premium.get('theta_burn_15m_str', '--')
            }
        }

        if status == "STAND_ASIDE":
            action_summary = f"STAND ASIDE: {rejection_reasons[0] if rejection_reasons else 'Confluence score ' + str(int(confluence_score)) + '/100 — no high-ROI edge.'}"
        elif status == "IGNITED":
            action_summary = f"BUY {primary_premium['strike']:.0f} {opt_name} NOW | TARGET: ₹{primary_premium['target_1']} (+{primary_premium['target_gain_pct']}%) | SL: ₹{primary_premium['stop_loss']} | HOLD: {recommended_horizon}"
        elif status == "COILING" or status == "ARMED":
            action_summary = f"WAIT FOR TRIGGER: Prepare to BUY {primary_premium['strike']:.0f} {opt_name} when spot breaks {trigger_strike:.0f} | HOLD: {recommended_horizon}"
        elif status == "TARGET_REACHED":
            action_summary = f"BOOK 75% PROFIT: Target {rebalance_target:.0f} hit (+{primary_premium['target_gain_pct']}% ROI)! Trail runners."
        elif status == "REBALANCING":
            action_summary = f"HOLD {opt_name}: {archetype_name} underway ({progress_pct:.0f}% towards {rebalance_target:.0f}) | MAX TIME: {max_hold_mins}m."
        else:
            action_summary = f"MONITORING: Spot {abs(dist_to_trigger):.0f} pts from {trigger_strike:.0f} {opt_name} barrier. Confluence: {confluence_score:.0f}/100."

        return {
            'ok': True,
            'status': status,
            'status_desc': status_desc,
            'action_summary': action_summary,
            'trade_ready': trade_ready,
            'active_tier': active_tier,
            'tier_name': tier_name,
            'setup_archetype': setup_archetype,
            'archetype_name': archetype_name,
            'archetype_badge_color': archetype_badge_color,
            'recommended_horizon': recommended_horizon,
            'max_hold_mins': max_hold_mins,
            'hard_time_stop': hard_time_stop,
            'theta_decay_burn_15m': primary_premium.get('theta_burn_15m_str', '--'),
            'exit_triggers': exit_triggers,
            'confluence_score': confluence_score,
            'confluence_checklist': checklist,
            'rejection_reasons': rejection_reasons,
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

    def compute(self, spot: float, chain_df: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """Convenience alias for evaluate(chain_df=chain_df, spot=spot, ...)."""
        return self.evaluate(chain_df=chain_df, spot=spot, **kwargs)

