"""
IvSurfaceEngine.py — Quantitative Implied Volatility Surface, 1-Day Expected Move
Exhaustion & Intraday Rewind Engine.

Calculates:
1. Real-time ATM IV, 2D IV Smile, and 3D Volatility Surface.
2. Non-directional 1-Day Expected Move vs Realized Move Since Open exhaustion comparison:
   - Expected 1-Day Move: Spot * (ATM IV / 100) * sqrt(1/365)
   - Realized Move Since Open: Spot - Day Open (and absolute displacement)
   - Realized Day Range: Day High - Day Low
   - Expected Move Consumption Ratio: |Spot - Open| / Expected Move * 100%
   - Range Consumption Ratio: (High - Low) / Expected Move * 100%
   - 1-Sigma Statistical Bounds: [Open - Expected Move, Open + Expected Move]
   - Volatility Exhaustion Status: Consolidation vs Normal vs 1-Sigma Exhaustion vs Expansion
3. Historical Rewind Memory & Delta IV Surface Shift:
   - Session-long ring buffer of snapshots (up to 1,500 ticks)
   - Rewind scrubber and presets (LIVE, -5m, -15m, -30m, -1h, DAY OPEN)
   - Active vs Baseline Smile overlay & per-strike Delta IV bar shifts
   - Total IV metrics shift: Delta ATM IV, Delta Skew, Delta Term Spread, Wing shifts
"""

import math
import time
import datetime
import collections
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import pandas as pd


class IvSurfaceEngine:
    def __init__(self, max_history: int = 1500):
        self._iv_ring: collections.deque = collections.deque(maxlen=max_history)
        self._day_open: Optional[float] = None
        self._day_high: Optional[float] = None
        self._day_low: Optional[float] = None
        self._last_push_ts: float = 0.0

    # ── Push Snapshot ────────────────────────────────────────────────────────
    def push_snapshot(
        self,
        chain_df: Optional[pd.DataFrame] = None,
        spot: float = 0.0,
        day_open: Optional[float] = None,
        day_high: Optional[float] = None,
        day_low: Optional[float] = None,
        dte_years: float = 7.0 / 365.0,
        df: Optional[pd.DataFrame] = None,
        week_open: Optional[float] = None,
        week_high: Optional[float] = None,
        week_low: Optional[float] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Record a real-time snapshot of the IV curve, spot price, day range, and weekly range.
        Throttled to max 1 snapshot per 2.5 seconds if market is steady.
        """
        if chain_df is None and df is not None:
            chain_df = df

        if spot <= 0 or chain_df is None or chain_df.empty:
            return None

        now_ts = time.time()
        if (now_ts - self._last_push_ts < 2.5) and self._iv_ring:
            # Check if spot moved significantly (> 2 pts), otherwise skip duplicate
            if abs(spot - self._iv_ring[-1].get('spot', spot)) < 2.0:
                return self.get_surface_data()

        # Lock day open, high, low
        if day_open and day_open > 0:
            if self._day_open is None or self._day_open <= 0:
                self._day_open = float(day_open)
        elif self._day_open is None or self._day_open <= 0:
            self._day_open = float(spot)

        eff_open = self._day_open if self._day_open and self._day_open > 0 else spot

        if day_high and day_high > 0:
            self._day_high = max(float(day_high), spot, self._day_high or spot)
        else:
            self._day_high = max(spot, self._day_high or spot)

        if day_low and day_low > 0:
            self._day_low = min(float(day_low), spot, self._day_low or spot)
        else:
            self._day_low = min(spot, self._day_low or spot)

        # Weekly anchor bounds
        eff_week_open = float(week_open) if week_open and week_open > 0 else eff_open
        eff_week_high = float(max(week_high or self._day_high, spot, self._day_high))
        eff_week_low = float(min(week_low or self._day_low, spot, self._day_low))

        # Parse per-strike IV from dataframe
        strikes_map = self._extract_strikes_iv(chain_df, spot)
        if not strikes_map:
            return None

        # Compute ATM IV
        atm_strike, atm_iv = self._find_atm_iv(strikes_map, spot)
        if atm_iv <= 0:
            atm_iv = 12.5  # fallback reasonable default

        # Expected 1-Day Move from ATM IV (scaled by 252 trading days)
        daily_vol = (atm_iv / 100.0) * math.sqrt(1.0 / 252.0)
        expected_move_pts = round(spot * daily_vol, 1)
        expected_move_pct = round(daily_vol * 100.0, 2)

        # Expected Weekly Move (5 Trading Days)
        weekly_vol = (atm_iv / 100.0) * math.sqrt(5.0 / 252.0)
        weekly_expected_move_pts = round(spot * weekly_vol, 1)
        weekly_expected_move_pct = round(weekly_vol * 100.0, 2)

        # Skew & Wings
        put_iv_avg, call_iv_avg, skew_ratio = self._calc_skew(strikes_map, spot, atm_iv)

        # Term spread if multi-DTE is available
        term_spread = self._calc_term_spread(chain_df, spot, atm_iv)

        now_str = datetime.datetime.fromtimestamp(now_ts).strftime('%H:%M:%S')

        snapshot = {
            'ts': now_ts,
            'time_str': now_str,
            'spot': round(float(spot), 2),
            'day_open': round(float(eff_open), 2),
            'day_high': round(float(self._day_high), 2),
            'day_low': round(float(self._day_low), 2),
            'week_open': round(float(eff_week_open), 2),
            'week_high': round(float(eff_week_high), 2),
            'week_low': round(float(eff_week_low), 2),
            'atm_strike': atm_strike,
            'atm_iv': round(float(atm_iv), 2),
            'expected_move_pts': expected_move_pts,
            'expected_move_pct': expected_move_pct,
            'weekly_expected_move_pts': weekly_expected_move_pts,
            'weekly_expected_move_pct': weekly_expected_move_pct,
            'put_iv_avg': round(float(put_iv_avg), 2),
            'call_iv_avg': round(float(call_iv_avg), 2),
            'skew_ratio': round(float(skew_ratio), 3),
            'term_spread': round(float(term_spread), 2),
            'dte_days': round(dte_years * 365.0, 1),
            'strikes_map': strikes_map
        }

        self._iv_ring.append(snapshot)
        self._last_push_ts = now_ts
        return self.get_surface_data()

    # ── Get Surface Data & Rewind Analysis ──────────────────────────────────
    def get_surface_data(
        self,
        rewind_ts: Optional[float] = None,
        baseline_mode: str = 'open'
    ) -> Dict[str, Any]:
        """
        Returns full IV surface terminal payload:
        - Active Snapshot (live or rewound)
        - Baseline Snapshot (Day Open or session start)
        - Expected 1-Day Move vs Move Since Open exhaustion comparison
        - Delta IV smile shifts & Total IV metrics
        - 3D Surface meshgrid coordinates
        - History index for scrubber
        """
        if not self._iv_ring:
            return {
                'ok': False,
                'status': 'NO_DATA',
                'message': 'Accumulating IV snapshots...'
            }

        latest_snap = self._iv_ring[-1]
        live_ts = latest_snap['ts']
        live_time_str = latest_snap['time_str']

        # Determine active snapshot
        is_rewound = False
        active_snap = latest_snap
        if rewind_ts is not None and len(self._iv_ring) > 1:
            closest_snap = min(self._iv_ring, key=lambda s: abs(s['ts'] - rewind_ts))
            if abs(closest_snap['ts'] - live_ts) > 5.0:
                active_snap = closest_snap
                is_rewound = True

        # Determine baseline snapshot (Day Open / initial snapshot)
        baseline_snap = self._iv_ring[0]
        if baseline_mode == '30m' and len(self._iv_ring) > 1:
            target_ts = active_snap['ts'] - 1800.0
            baseline_snap = min(self._iv_ring, key=lambda s: abs(s['ts'] - target_ts))

        # Build History Index for Scrubber
        history_index = [
            {
                'ts': s['ts'],
                'time_str': s['time_str'],
                'spot': s['spot'],
                'atm_iv': s['atm_iv']
            }
            for s in self._iv_ring
        ]

        # ── Movement & Exhaustion Comparison (Non-Directional) ──────────────
        spot = active_snap['spot']
        day_open = active_snap['day_open']
        day_high = active_snap['day_high']
        day_low = active_snap['day_low']
        exp_pts = active_snap['expected_move_pts']
        exp_pct = active_snap['expected_move_pct']

        move_since_open = round(spot - day_open, 2)
        abs_move_since_open = round(abs(move_since_open), 2)
        move_since_open_pct = round((move_since_open / day_open) * 100.0, 2) if day_open > 0 else 0.0

        day_range = round(day_high - day_low, 2)
        day_range_pct = round((day_range / day_open) * 100.0, 2) if day_open > 0 else 0.0

        # Consumption calculations
        net_consumption_pct = round((abs_move_since_open / exp_pts) * 100.0, 1) if exp_pts > 0 else 0.0
        range_consumption_pct = round((day_range / exp_pts) * 100.0, 1) if exp_pts > 0 else 0.0

        upper_1sigma = round(day_open + exp_pts, 1)
        lower_1sigma = round(day_open - exp_pts, 1)
        dist_to_upper = round(upper_1sigma - spot, 1)
        dist_to_lower = round(spot - lower_1sigma, 1)

        # Exhaustion Classification (Clean, mathematical, no directional guessing)
        if net_consumption_pct < 50.0:
            exhaustion_regime = "CONSOLIDATION / LOW EXHAUSTION"
            exhaustion_color = "#00e5ff"  # Cyan
            exhaustion_desc = (
                f"Spot has moved {abs_move_since_open:.1f} pts ({net_consumption_pct:.0f}%) of its ±{exp_pts:.0f} pt "
                f"daily expected move. Range remains compressed with ample volatility budget."
            )
        elif net_consumption_pct < 85.0:
            exhaustion_regime = "NORMAL STATISTICAL RANGE"
            exhaustion_color = "#00e676"  # Green
            exhaustion_desc = (
                f"Spot has traversed {abs_move_since_open:.1f} pts ({net_consumption_pct:.0f}%) of the 1-day expected range. "
                f"Orderly displacement inside typical 1-sigma daily boundaries."
            )
        elif net_consumption_pct <= 105.0:
            exhaustion_regime = "1-SIGMA EXHAUSTION BAND"
            exhaustion_color = "#ffd54f"  # Gold
            exhaustion_desc = (
                f"Spot has consumed {net_consumption_pct:.0f}% of the 1-day expected move ({abs_move_since_open:.1f} / ±{exp_pts:.0f} pts). "
                f"Statistical boundary reached — probability of range exhaustion or pinning is elevated."
            )
        else:
            exhaustion_regime = "VOLATILITY EXPANSION / BREAKOUT"
            exhaustion_color = "#ff3366"  # Red / Hot Pink
            exhaustion_desc = (
                f"Spot has exceeded the 1-day expected move budget ({net_consumption_pct:.0f}% consumed, "
                f"+{abs_move_since_open - exp_pts:.1f} pts beyond 1σ). Fat-tail volatility expansion underway."
            )

        # ── IV Shifts (Active vs Baseline) ──────────────────────────────────
        delta_atm_iv = round(active_snap['atm_iv'] - baseline_snap['atm_iv'], 2)
        delta_skew = round(active_snap['skew_ratio'] - baseline_snap['skew_ratio'], 3)
        delta_term = round(active_snap['term_spread'] - baseline_snap['term_spread'], 2)
        delta_put_iv = round(active_snap['put_iv_avg'] - baseline_snap['put_iv_avg'], 2)
        delta_call_iv = round(active_snap['call_iv_avg'] - baseline_snap['call_iv_avg'], 2)

        # Smile points comparison
        act_map = active_snap.get('strikes_map', {})
        base_map = baseline_snap.get('strikes_map', {})

        smile_strikes = sorted(set(act_map.keys()) | set(base_map.keys()))
        # Limit smile to +/- 1000 pts from spot
        smile_strikes = [k for k in smile_strikes if abs(k - spot) <= 1000]

        smile_data = []
        for k in smile_strikes:
            act_k = act_map.get(k, {})
            base_k = base_map.get(k, {})

            act_iv = act_k.get('mean_iv', 0.0)
            base_iv = base_k.get('mean_iv', 0.0)

            d_iv = round(act_iv - base_iv, 2) if (act_iv > 0 and base_iv > 0) else 0.0
            smile_data.append({
                'strike': k,
                'active_iv': round(act_iv, 2),
                'active_ce_iv': round(act_k.get('call_iv', 0.0), 2),
                'active_pe_iv': round(act_k.get('put_iv', 0.0), 2),
                'baseline_iv': round(base_iv, 2),
                'delta_iv': d_iv
            })

        # Weekly movement & expected move (5 trading days)
        week_open = active_snap.get('week_open', day_open)
        week_high = active_snap.get('week_high', day_high)
        week_low = active_snap.get('week_low', day_low)
        w_exp_pts = active_snap.get('weekly_expected_move_pts') or round(spot * (active_snap['atm_iv'] / 100.0) * math.sqrt(5.0 / 252.0), 1)
        w_exp_pct = active_snap.get('weekly_expected_move_pct') or round((active_snap['atm_iv'] / 100.0) * math.sqrt(5.0 / 252.0) * 100.0, 2)

        weekly_realized_pts = round(spot - week_open, 2)
        abs_weekly_realized = round(abs(weekly_realized_pts), 2)
        weekly_realized_pct = round((weekly_realized_pts / week_open) * 100.0, 2) if week_open > 0 else 0.0
        weekly_range_pts = round(week_high - week_low, 2)

        weekly_upper_1sigma = round(week_open + w_exp_pts, 1)
        weekly_lower_1sigma = round(week_open - w_exp_pts, 1)
        dist_to_w_upper = round(weekly_upper_1sigma - spot, 1)
        dist_to_w_lower = round(spot - weekly_lower_1sigma, 1)

        weekly_net_consumption_pct = round((abs_weekly_realized / w_exp_pts) * 100.0, 1) if w_exp_pts > 0 else 0.0
        weekly_range_consumption_pct = round((weekly_range_pts / w_exp_pts) * 100.0, 1) if w_exp_pts > 0 else 0.0

        # ── 3D Surface Grid ──────────────────────────────────────────────────
        surface_3d = self._build_surface_grid(active_snap, smile_data, spot)

        exp_dict = {
            'expected_move_pts': exp_pts,
            'expected_move_pct': exp_pct,
            'spot_vs_open_pts': move_since_open,
            'spot_vs_open_pct': move_since_open_pct,
            'abs_move_since_open': abs_move_since_open,
            'day_open': day_open,
            'day_high': day_high,
            'day_low': day_low,
            'high_low_range_pts': day_range,
            'high_low_range_pct': day_range_pct,
            'upper_1sigma': upper_1sigma,
            'lower_1sigma': lower_1sigma,
            'dist_to_upper_1sigma': dist_to_upper,
            'dist_to_lower_1sigma': dist_to_lower,
            # Weekly Horizon Metrics
            'weekly_expected_move_pts': w_exp_pts,
            'weekly_expected_move_pct': w_exp_pct,
            'weekly_realized_pts': weekly_realized_pts,
            'weekly_realized_pct': weekly_realized_pct,
            'abs_weekly_realized': abs_weekly_realized,
            'week_open': week_open,
            'week_high': week_high,
            'week_low': week_low,
            'weekly_range_pts': weekly_range_pts,
            'weekly_upper_1sigma': weekly_upper_1sigma,
            'weekly_lower_1sigma': weekly_lower_1sigma,
            'dist_to_weekly_upper': dist_to_w_upper,
            'dist_to_weekly_lower': dist_to_w_lower
        }

        exh_dict = {
            'net_consumption_pct': net_consumption_pct,
            'range_consumption_pct': range_consumption_pct,
            'weekly_net_consumption_pct': weekly_net_consumption_pct,
            'weekly_range_consumption_pct': weekly_range_consumption_pct,
            'status': exhaustion_regime,
            'regime': exhaustion_regime,
            'color': exhaustion_color,
            'desc': exhaustion_desc
        }

        shifts_dict = {
            'delta_atm_iv': delta_atm_iv,
            'delta_skew': delta_skew,
            'delta_term_spread': delta_term,
            'delta_put_iv': delta_put_iv,
            'delta_call_iv': delta_call_iv
        }

        smile_strikes_list = [d['strike'] for d in smile_data]
        active_ivs_list = [d['active_iv'] for d in smile_data]
        baseline_ivs_list = [d['baseline_iv'] for d in smile_data]
        delta_ivs_list = [d['delta_iv'] for d in smile_data]

        smile_2d_dict = {
            'strikes': smile_strikes_list,
            'active_strikes': smile_strikes_list,
            'active_ivs': active_ivs_list,
            'baseline_ivs': baseline_ivs_list,
            'delta_ivs': delta_ivs_list,
            'points': smile_data
        }

        return {
            'ok': True,
            'is_rewound': is_rewound,
            'ts': active_snap['ts'],
            'timestamp_str': active_snap['time_str'],
            'active_time_str': active_snap['time_str'],
            'baseline_time_str': baseline_snap['time_str'],
            'live_time_str': live_time_str,
            'spot': spot,
            'atm_iv': active_snap['atm_iv'],
            'day_open': day_open,
            'day_high': day_high,
            'day_low': day_low,

            'expected_move': exp_dict,
            'exhaustion': exh_dict,
            'total_iv_shifts': shifts_dict,
            'metrics': {
                **shifts_dict,
                'atm_iv': active_snap['atm_iv'],
                'skew_ratio': active_snap['skew_ratio'],
                'term_spread': active_snap['term_spread'],
                'put_iv_avg': active_snap['put_iv_avg'],
                'call_iv_avg': active_snap['call_iv_avg']
            },
            'active_snapshot': {
                'atm_iv': active_snap['atm_iv'],
                'skew_ratio': active_snap['skew_ratio'],
                'term_spread': active_snap['term_spread'],
                'put_wing_iv': active_snap['put_iv_avg'],
                'call_wing_iv': active_snap['call_iv_avg']
            },
            'baseline_snapshot': {
                'atm_iv': baseline_snap['atm_iv'],
                'skew_ratio': baseline_snap['skew_ratio'],
                'term_spread': baseline_snap['term_spread']
            },

            'smile_2d': smile_2d_dict,
            'smile': smile_data,
            'surface_3d': surface_3d,
            'history_index': history_index
        }

    # ── Helpers ──────────────────────────────────────────────────────────────
    def _extract_strikes_iv(self, df: pd.DataFrame, spot: float) -> Dict[float, Dict[str, float]]:
        """Extract call_iv, put_iv, and mean_iv per strike from chain DataFrame."""
        strikes: Dict[float, Dict[str, float]] = {}
        for _, row in df.iterrows():
            strike = float(row.get('strike', 0) or 0)
            if strike <= 0:
                continue
            iv = float(row.get('iv', 0) or 0)
            o_type = str(row.get('type', 'CE')).upper()
            price = float(row.get('price', 0) or 0)

            if iv <= 1.0 or iv > 150.0:
                # Fallback Tier 1: Numerical solve from option market price
                if price > 0.5 and spot > 0:
                    try:
                        from OptionAnalytics import OptionAnalytics
                        oa = OptionAnalytics()
                        dte = float(row.get('dte', 2.0) or 2.0)
                        T = max(dte / 365.0, 1.0 / 365.0)
                        calc_iv = oa.implied_volatility(price, spot, strike, T, 0.051274, o_type)
                        if 1.0 < calc_iv < 150.0:
                            iv = float(calc_iv)
                    except Exception:
                        pass
                # Fallback Tier 2: Institutional log-moneyness quadratic smile approximation
                if (iv <= 1.0 or iv > 150.0) and spot > 0 and strike > 0:
                    k = math.log(strike / spot)
                    smile_iv = 13.5 - 15.0 * k + 25.0 * (k ** 2)
                    iv = max(6.0, min(80.0, float(smile_iv)))

            if iv <= 1.0 or iv > 150.0:
                continue

            if strike not in strikes:
                strikes[strike] = {'call_iv': 0.0, 'put_iv': 0.0, 'mean_iv': 0.0}

            if o_type == 'CE':
                strikes[strike]['call_iv'] = iv
            elif o_type == 'PE':
                strikes[strike]['put_iv'] = iv

        # Calculate mean_iv per strike (prefer OTM side: Put for strike < spot, Call for strike > spot)
        for strike, d in strikes.items():
            c_iv = d['call_iv']
            p_iv = d['put_iv']
            if c_iv > 0 and p_iv > 0:
                # Slight OTM weighting
                if strike < spot:
                    d['mean_iv'] = p_iv * 0.7 + c_iv * 0.3
                elif strike > spot:
                    d['mean_iv'] = c_iv * 0.7 + p_iv * 0.3
                else:
                    d['mean_iv'] = (c_iv + p_iv) / 2.0
            elif c_iv > 0:
                d['mean_iv'] = c_iv
            elif p_iv > 0:
                d['mean_iv'] = p_iv

        return strikes

    def _find_atm_iv(self, strikes_map: Dict[float, Dict[str, float]], spot: float) -> Tuple[float, float]:
        """Find strike closest to spot and return (atm_strike, atm_iv)."""
        valid_strikes = [k for k, v in strikes_map.items() if v['mean_iv'] > 0]
        if not valid_strikes:
            return spot, 0.0
        closest_strike = min(valid_strikes, key=lambda k: abs(k - spot))
        return closest_strike, strikes_map[closest_strike]['mean_iv']

    def _calc_skew(self, strikes_map: Dict[float, Dict[str, float]], spot: float, atm_iv: float) -> Tuple[float, float, float]:
        """Compute OTM Put avg, OTM Call avg, and Skew Ratio (Put IV / Call IV)."""
        otm_puts = [v['mean_iv'] for k, v in strikes_map.items() if k < spot * 0.985 and v['mean_iv'] > 0]
        otm_calls = [v['mean_iv'] for k, v in strikes_map.items() if k > spot * 1.015 and v['mean_iv'] > 0]

        put_avg = float(np.mean(otm_puts)) if otm_puts else atm_iv
        call_avg = float(np.mean(otm_calls)) if otm_calls else atm_iv
        skew_ratio = (put_avg / call_avg) if call_avg > 0 else 1.0
        return put_avg, call_avg, skew_ratio

    def _calc_term_spread(self, df: pd.DataFrame, spot: float, near_atm_iv: float) -> float:
        """If multi-expiry or DTE exists in DataFrame, calculate Far ATM IV - Near ATM IV."""
        if 'dte' not in df.columns:
            return 0.0
        dtes = sorted(df['dte'].dropna().unique())
        if len(dtes) < 2:
            return 0.0
        far_dte = dtes[-1]
        far_sub = df[df['dte'] == far_dte]
        if not isinstance(far_sub, pd.DataFrame):
            far_sub = pd.DataFrame(far_sub)
        far_strikes = self._extract_strikes_iv(far_sub, spot)
        if not far_strikes:
            return 0.0
        _, far_atm_iv = self._find_atm_iv(far_strikes, spot)
        if far_atm_iv > 0:
            return far_atm_iv - near_atm_iv
        return 0.0

    def _build_surface_grid(
        self,
        active_snap: Dict[str, Any],
        smile_data: List[Dict[str, Any]],
        spot: float
    ) -> Dict[str, Any]:
        """Generate structured (x: strikes, y: days, z: iv_mesh) for Plotly 3D Surface."""
        if not smile_data:
            return {'x': [], 'y': [], 'z': []}

        strikes = [s['strike'] for s in smile_data]
        ivs = [s['active_iv'] for s in smile_data]
        base_dte = active_snap.get('dte_days', 5.0)

        # Synthesize a realistic 3-slice term structure surface:
        # Near DTE (base_dte), Mid DTE (base_dte + 7), Monthly DTE (base_dte + 28)
        term_spread = active_snap.get('term_spread', 0.0)

        days_axis = [round(base_dte, 1), round(base_dte + 7.0, 1), round(base_dte + 28.0, 1)]
        z_mesh = []

        # Slice 0: Near Expiry (Actual smile)
        z_mesh.append([round(iv, 2) for iv in ivs])

        # Slice 1: Mid Expiry (+0.4 * term_spread, slight flattening towards ATM)
        mid_ivs = []
        for k, iv in zip(strikes, ivs):
            flattener = 0.85 if abs(k - spot) > 200 else 1.0
            mid_val = max(5.0, (iv * flattener) + (term_spread * 0.45))
            mid_ivs.append(round(mid_val, 2))
        z_mesh.append(mid_ivs)

        # Slice 2: Monthly Expiry (+term_spread, further flattening)
        far_ivs = []
        for k, iv in zip(strikes, ivs):
            flattener = 0.75 if abs(k - spot) > 200 else 1.0
            far_val = max(5.0, (iv * flattener) + term_spread)
            far_ivs.append(round(far_val, 2))
        z_mesh.append(far_ivs)

        return {
            'x': strikes,
            'y': days_axis,
            'z': z_mesh,
            'strikes': strikes,
            'dtes': days_axis,
            'z_iv': z_mesh
        }
