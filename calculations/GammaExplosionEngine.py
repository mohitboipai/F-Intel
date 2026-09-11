"""
calculations/GammaExplosionEngine.py
=============================================================================
Institutional Market Maker Gamma Pinning & Order Flow Absorption Engine.
Directly replicates the quantitative methodologies demonstrated in:
  1. Reel 1 (quantedoptions): Strike-level signed dealer gamma switch-ons,
     magnetic pinning duration timer (e.g. 4h 40m), price corridor lock,
     and position unwinding/release cascades to the next major strike.
  2. Reel 2 (aleksrosme): Retest of major GEX level (Put Wall / Call Wall / Gamma Flip)
     combined with microstructural order flow absorption (rejection wicks + volume)
     yielding asymmetric 100-point explosion setups.
=============================================================================
"""

import sys
import os
import time
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd

from .GexEngine import GexEngine
from .DealerPositionEngine import DealerPositionEngine
from .DealerHedgingSimulator import DealerHedgingSimulator

try:
    import config as _cfg
    _DEFAULT_LOT_SIZE = _cfg.get("nifty_lot_size", 65)
except Exception:
    _DEFAULT_LOT_SIZE = 65


class GammaExplosionEngine:
    """
    Quantitative engine replicating quantedoptions (pinning duration & unwind)
    and aleksrosme (GEX level retest + absorption = 100-pt explosion).
    """

    CORRIDOR_PTS = 50.0  # Strike pinning tolerance corridor (+-50 pts on NIFTY)

    def __init__(self,
                 lot_size: int = _DEFAULT_LOT_SIZE,
                 history_len: int = 60,
                 gex_engine: Optional[GexEngine] = None):
        self.lot_size = lot_size
        self.gex_engine = gex_engine or GexEngine(lot_size=lot_size)
        self.pos_engine = DealerPositionEngine(lot_size=lot_size)
        self.hedging_sim = DealerHedgingSimulator(self.pos_engine)

        # Pin state: {strike: {'first_trapped_ts': float, 'last_seen_ts': float, 'peak_gex': float, 'max_high': float, 'min_low': float}}
        self._pin_state: Dict[float, Dict[str, Any]] = {}
        # Recent confirmed absorption events
        self._recent_absorptions: List[Dict[str, Any]] = []

    # ──────────────────────────────────────────────────────────────────────────
    # 1. REEL 1: DOMINANT STRIKE PINNING & DURATION TIMER (quantedoptions)
    # ──────────────────────────────────────────────────────────────────────────

    def detect_dominant_pins(self, profile: pd.Series, spot_price: float,
                             now_ts: float) -> List[Dict[str, Any]]:
        """
        Identifies the dominant positive GEX strike (Long Gamma Pin) interacting with spot,
        tracks pinning duration, magnetic pull force, and position unwinding/release risk.
        In Reel 1: "One strike ran the whole session — ES, Sept 8. $5.2B MM gamma switched on at 7700.
        Price never printed a high above it for 4 hours and 40 minutes."
        """
        if profile.empty or spot_price <= 0:
            return []

        # Positive GEX strikes where dealers are Long Gamma (stabilizing mean-reverting pin)
        pos_strikes = profile[profile > 0].sort_values(ascending=False)
        total_pos_gex = max(pos_strikes.sum(), 1.0)

        # Select candidate: dominant positive strike closest to spot (within 300 pts) or highest overall
        dominant_candidate = None
        for strike, raw_val in pos_strikes.items():
            strike = float(strike)
            dist = abs(spot_price - strike)
            gex_cr = raw_val / 1e7
            share_pct = (raw_val / total_pos_gex) * 100
            if dist <= 160:  # Immediate active pin near spot
                dominant_candidate = (strike, gex_cr, share_pct)
                break
            elif dominant_candidate is None and dist <= 350:
                dominant_candidate = (strike, gex_cr, share_pct)

        if not dominant_candidate and not pos_strikes.empty:
            top_s = float(pos_strikes.index[0])
            dominant_candidate = (top_s, pos_strikes.iloc[0] / 1e7, (pos_strikes.iloc[0] / total_pos_gex) * 100)

        if not dominant_candidate:
            return []

        pin_strike, pin_gex_cr, share_pct = dominant_candidate
        dist_pts = spot_price - pin_strike
        is_inside_corridor = abs(dist_pts) <= self.CORRIDOR_PTS

        if pin_strike not in self._pin_state:
            self._pin_state[pin_strike] = {
                'first_seen_ts': now_ts,
                'first_trapped_ts': now_ts if is_inside_corridor else None,
                'last_seen_ts': now_ts,
                'peak_gex': pin_gex_cr,
                'max_high_at_pin': spot_price,
                'min_low_at_pin': spot_price
            }

        p_state = self._pin_state[pin_strike]
        duration_sec = now_ts - p_state['first_seen_ts']

        if is_inside_corridor:
            if p_state['first_trapped_ts'] is None:
                p_state['first_trapped_ts'] = now_ts
            p_state['max_high_at_pin'] = max(p_state['max_high_at_pin'], spot_price)
            p_state['min_low_at_pin'] = min(p_state['min_low_at_pin'], spot_price)
        else:
            if p_state['first_trapped_ts'] and (now_ts - p_state['last_seen_ts']) > 900:
                p_state['first_trapped_ts'] = None

        p_state['last_seen_ts'] = now_ts
        if pin_gex_cr > p_state['peak_gex']:
            p_state['peak_gex'] = pin_gex_cr

        duration_min = round(duration_sec / 60.0, 1)

        # Format duration string: e.g. "4h 40m" or "45m"
        hrs = int(duration_min // 60)
        mins = int(duration_min % 60)
        dur_str = f"{hrs}h {mins}m" if hrs > 0 else f"{mins}m"

        # Magnetic Pull Score: 100 when spot is at strike, decays exponentially outwards
        pull_score = max(5.0, min(100.0, 100.0 * np.exp(-abs(dist_pts) / 45.0)))

        # Unpinning Risk / Release Cascade Analysis (Reel 1: "when the position closed out after 14:25 the pull faded and price fell out")
        decay_ratio = pin_gex_cr / max(p_state['peak_gex'], 1.0)
        if decay_ratio < 0.70:
            unpin_risk = "IMMINENT"
            status_desc = f"Dealer gamma fading (-{(1 - decay_ratio) * 100:.0f}% from peak). Unwind imminent."
        elif duration_min >= 180:
            unpin_risk = "ELEVATED"
            status_desc = f"Pinned for {dur_str}. Testing corridor boundaries."
        elif not is_inside_corridor and abs(dist_pts) > (self.CORRIDOR_PTS * 2.5):
            unpin_risk = "RELEASED_BREAKOUT"
            status_desc = f"Spot broke out of {pin_strike:.0f} pin corridor ({dist_pts:+.1f} pts). Pin release cascade active."
        else:
            unpin_risk = "LOW"
            status_desc = f"Active magnetic lock. Price pinned within {pin_strike - self.CORRIDOR_PTS:.0f} - {pin_strike + self.CORRIDOR_PTS:.0f}."

        # Cascade Target if unpinned (next major strike below or above)
        put_wall = float(profile.idxmin())
        cascade_next_strike = put_wall if dist_pts <= 0 else float(profile.idxmax())

        return [{
            'strike': pin_strike,
            'gex_cr': round(pin_gex_cr, 1),
            'peak_gex_cr': round(p_state['peak_gex'], 1),
            'pin_type': 'LONG_GAMMA_PIN',
            'duration_min': duration_min,
            'duration_str': dur_str,
            'is_inside_corridor': is_inside_corridor,
            'corridor_low': pin_strike - self.CORRIDOR_PTS,
            'corridor_high': pin_strike + self.CORRIDOR_PTS,
            'dist_pts': round(dist_pts, 1),
            'magnetic_force_score': round(pull_score, 1),
            'unpinning_risk': unpin_risk,
            'cascade_next_strike': cascade_next_strike,
            'status_desc': status_desc
        }]

    def update_and_detect_pins(self, chain_df: pd.DataFrame, spot_price: float,
                               now_ts: Optional[float] = None) -> List[Dict[str, Any]]:
        """Backwards-compatible wrapper for unit tests."""
        ts = now_ts or time.time()
        gex_res = self.gex_engine.calculate_gex(chain_df, spot_price)
        profile = gex_res.get('profile', pd.Series()).sort_index()
        return self.detect_dominant_pins(profile, spot_price, ts)

    # ──────────────────────────────────────────────────────────────────────────
    # 2. REEL 2: RETEST OF GEX LEVEL + ABSORPTION (aleksrosme)
    # ──────────────────────────────────────────────────────────────────────────

    def detect_gex_absorption(self,
                              profile: Optional[pd.Series] = None,
                              gamma_flip: float = 0.0,
                              spot_price: float = 0.0,
                              candles: Optional[List[List[Any]]] = None,
                              now_ts: Optional[float] = None,
                              chain_df: Optional[pd.DataFrame] = None,
                              candle_high: Optional[float] = None,
                              candle_low: Optional[float] = None,
                              candle_close: Optional[float] = None,
                              buyer_vol: float = 0.0,
                              seller_vol: float = 0.0) -> List[Dict[str, Any]]:
        """
        Monitors whether Spot is retesting Put Wall, Call Wall, or Gamma Flip,
        analyzing real candle wicks and volume to detect institutional absorption.
        In Reel 2: "09/08 - Retest of 720GEX + Absorption = 100points".
        """
        if profile is None and chain_df is not None:
            gex_res = self.gex_engine.calculate_gex(chain_df, spot_price)
            profile = gex_res.get('profile', pd.Series()).sort_index()
            gamma_flip = float(gex_res.get('zero_gamma_level', 0.0))

        if profile is None or profile.empty or spot_price <= 0:
            return []

        now_dt_str = datetime.fromtimestamp(now_ts or time.time()).strftime("%H:%M:%S")

        call_wall = float(profile.idxmax())
        put_wall  = float(profile.idxmin())
        call_wall_gex_cr = round(profile[call_wall] / 1e7, 1)
        put_wall_gex_cr  = round(profile[put_wall] / 1e7, 1)

        levels_to_check = [
            {'name': 'PUT WALL', 'level': put_wall, 'gex_cr': put_wall_gex_cr, 'barrier_type': 'SUPPORT'},
            {'name': 'CALL WALL', 'level': call_wall, 'gex_cr': call_wall_gex_cr, 'barrier_type': 'RESISTANCE'},
        ]
        if gamma_flip > 0:
            levels_to_check.append({
                'name': 'GAMMA FLIP', 'level': gamma_flip, 'gex_cr': 0.0, 'barrier_type': 'INFLECTION'
            })

        latest_c = candles[-1] if (candles and len(candles) > 0) else None
        med_vol = 1.0
        if candles and len(candles) > 0:
            vols = [c[5] for c in candles if len(c) > 5 and c[5] > 0]
            if vols:
                med_vol = max(float(np.median(vols)), 1.0)

        # Allow manual single-candle parameters from tests
        if candle_high is not None and candle_low is not None and candle_close is not None:
            c_open = (candle_high + candle_low) / 2.0
            c_vol = buyer_vol + seller_vol
            latest_c = [now_ts or time.time(), c_open, candle_high, candle_low, candle_close, c_vol]
            med_vol = max(1.0, c_vol / 2.0)

        absorptions = []
        for item in levels_to_check:
            lvl_name = item['name']
            lvl = float(item['level'])
            b_type = item['barrier_type']
            dist_pts = spot_price - lvl
            abs_dist = abs(dist_pts)

            is_retest = abs_dist <= 45.0
            wick_score = 30.0
            vol_mult = 1.0
            absorption_type = "NONE"
            wick_pct = 0.0
            delta_imbalance = 0.0

            if latest_c and len(latest_c) >= 6:
                c_open  = float(latest_c[1])
                c_high  = float(latest_c[2])
                c_low   = float(latest_c[3])
                c_close = float(latest_c[4])
                c_vol   = float(latest_c[5])
                c_range = max(c_high - c_low, 1.0)

                vol_mult = round(c_vol / med_vol, 2) if med_vol > 0 else 1.0

                if seller_vol + buyer_vol > 0:
                    delta_imbalance = round((buyer_vol - seller_vol) / (buyer_vol + seller_vol), 2)

                if b_type == 'SUPPORT':
                    # Put Wall retest: Low dipped to support, rejected upwards
                    lower_wick = max(0.0, min(c_open, c_close) - c_low)
                    wick_pct = round((lower_wick / c_range) * 100, 1)
                    if (c_low <= (lvl + 20.0) and c_close >= (lvl - 5.0)) or (is_retest and wick_pct >= 35.0):
                        absorption_type = "SELLERS_ABSORBED"
                        wick_score = min(100.0, 50.0 + (wick_pct * 0.7))
                    elif is_retest:
                        wick_score = 55.0
                elif b_type == 'RESISTANCE':
                    # Call Wall retest: High probed resistance, rejected downwards
                    upper_wick = max(0.0, c_high - max(c_open, c_close))
                    wick_pct = round((upper_wick / c_range) * 100, 1)
                    if (c_high >= (lvl - 20.0) and c_close <= (lvl + 5.0)) or (is_retest and wick_pct >= 35.0):
                        absorption_type = "BUYERS_ABSORBED"
                        wick_score = min(100.0, 50.0 + (wick_pct * 0.7))
                    elif is_retest:
                        wick_score = 55.0
                else:  # Gamma Flip
                    lower_wick = max(0.0, min(c_open, c_close) - c_low)
                    upper_wick = max(0.0, c_high - max(c_open, c_close))
                    if lower_wick > upper_wick and is_retest:
                        absorption_type = "SELLERS_ABSORBED"
                        wick_score = 65.0
                    elif upper_wick > lower_wick and is_retest:
                        absorption_type = "BUYERS_ABSORBED"
                        wick_score = 65.0
                    elif is_retest:
                        wick_score = 55.0
            else:
                if is_retest:
                    wick_score = 55.0

            proximity_score = max(0.0, 100.0 - (abs_dist * 2.0))
            vol_score = min(100.0, vol_mult * 40.0)
            composite_score = round((wick_score * 0.50) + (proximity_score * 0.30) + (vol_score * 0.20), 1)

            if composite_score >= 70.0 or (is_retest and absorption_type != "NONE" and wick_pct >= 30.0):
                status = "CONFIRMED"
            elif composite_score >= 50.0 or is_retest:
                status = "ABSORBING"
            else:
                status = "MONITORING"

            if absorption_type == "SELLERS_ABSORBED":
                desc = f"Retest of {lvl:.0f} {lvl_name} + Microstructural Absorption = 100-Point Bullish Squeeze Setup."
            elif absorption_type == "BUYERS_ABSORBED":
                desc = f"Retest of {lvl:.0f} {lvl_name} + Buyer Absorption = 100-Point Bearish Cascade Setup."
            else:
                desc = f"Monitoring {lvl_name} at {lvl:.0f} (dist: {dist_pts:+.1f} pts). Waiting for test and volume absorption."

            event = {
                'level_name': lvl_name,
                'level_strike': lvl,
                'gex_cr': item['gex_cr'],
                'barrier_type': b_type,
                'dist_pts': round(dist_pts, 1),
                'is_retesting': is_retest,
                'absorption_type': absorption_type,
                'absorption_score': composite_score,
                'rejection_wick_pct': wick_pct,
                'volume_multiplier': vol_mult,
                'delta_imbalance': delta_imbalance,
                'status': status,
                'setup_description': desc,
                'timestamp': now_dt_str
            }
            absorptions.append(event)

            if status == "CONFIRMED":
                self._recent_absorptions.append(event)
                if len(self._recent_absorptions) > 10:
                    self._recent_absorptions.pop(0)

        absorptions.sort(key=lambda x: abs(x['dist_pts']))
        return absorptions

    # ──────────────────────────────────────────────────────────────────────────
    # 3. REEL 2: 100-POINT EXPLOSION TARGET ENGINE (aleksrosme model)
    # ──────────────────────────────────────────────────────────────────────────

    def project_explosion_targets(self,
                                  chain_df: pd.DataFrame,
                                  spot_price: float,
                                  profile: Optional[pd.Series] = None,
                                  top_absorption: Optional[Dict[str, Any]] = None,
                                  confirmed_event: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Projects the 50-point primary and 100-point squeeze targets from GEX barrier absorption.
        In Reel 2: "Retest of 720GEX + Absorption = 100points".
        """
        if profile is None:
            gex_res = self.gex_engine.calculate_gex(chain_df, spot_price)
            profile = gex_res.get('profile', pd.Series()).sort_index()

        top_abs = confirmed_event or top_absorption
        call_wall = float(profile.idxmax()) if not profile.empty else spot_price + 200.0
        put_wall  = float(profile.idxmin()) if not profile.empty else spot_price - 200.0

        direction = "BULLISH"
        anchor_barrier = put_wall

        if top_abs:
            abs_type = top_abs.get('absorption_type')
            bar_type = top_abs.get('barrier_type')
            if abs_type == 'BUYERS_ABSORBED' or bar_type == 'RESISTANCE':
                direction = "BEARISH"
                anchor_barrier = call_wall
            elif abs_type == 'SELLERS_ABSORBED' or bar_type == 'SUPPORT':
                direction = "BULLISH"
                anchor_barrier = put_wall
            else:
                direction = "BULLISH" if spot_price >= put_wall else "BEARISH"
                anchor_barrier = put_wall if direction == "BULLISH" else call_wall
        else:
            if abs(spot_price - put_wall) < abs(spot_price - call_wall):
                direction = "BULLISH"
                anchor_barrier = put_wall
            else:
                direction = "BEARISH"
                anchor_barrier = call_wall

        if direction == "BULLISH":
            trigger_price = round(spot_price + 3.0, 1)
            invalidation_stop = round(trigger_price - 18.0, 1)  # 18 pts tight institutional risk
            target_1 = round(trigger_price + 50.0, 1)
            target_2 = round(trigger_price + 100.0, 1)  # Aleks Rosme 100-point explosion target
        else:
            trigger_price = round(spot_price - 3.0, 1)
            invalidation_stop = round(trigger_price + 18.0, 1)  # 18 pts tight institutional risk
            target_1 = round(trigger_price - 50.0, 1)
            target_2 = round(trigger_price - 100.0, 1)

        risk_pts = max(1.0, round(abs(trigger_price - invalidation_stop), 1))
        reward_pts = round(abs(target_2 - trigger_price), 1)
        rr_ratio = round(reward_pts / risk_pts, 2)

        # Dealer Hedging Acceleration per 25-point move (focused around barrier/spot)
        shift_pct = (25.0 / spot_price) * 100.0 * (1.0 if direction == "BULLISH" else -1.0)
        sim = self.hedging_sim.simulate_scenario(chain_df, spot_price, spot_shift_pct=shift_pct)
        req_shares = sim.get('required_hedge_flow', {}).get('shares_to_trade', 0.0)
        req_lots = int(round(abs(req_shares) / self.lot_size))

        return {
            'setup_type': f"RETEST_{direction}_EXPLOSION",
            'direction': direction,
            'anchor_barrier': float(round(anchor_barrier, 1)),
            'trigger_price': float(trigger_price),
            'invalidation_stop': float(invalidation_stop),
            'target_1': float(target_1),
            'target_2': float(target_2),
            'expected_points': float(reward_pts),
            'risk_points': float(risk_pts),
            'risk_reward_ratio': float(rr_ratio),
            'hedge_acceleration': {
                'shares_to_hedge_25pts': int(abs(req_shares)),
                'lots_to_hedge_25pts': int(req_lots),
                'flow_action': 'BUYING_PRESSURE' if direction == 'BULLISH' else 'SELLING_PRESSURE'
            }
        }

    # ──────────────────────────────────────────────────────────────────────────
    # 4. MASTER PAYLOAD
    # ──────────────────────────────────────────────────────────────────────────

    def get_full_status_payload(self, chain_df: pd.DataFrame, spot_price: float,
                                recent_candles: Optional[List[List[Any]]] = None) -> Dict[str, Any]:
        """
        Consolidates active pins, retest absorption signals, and cascade target forecast into
        one complete institutional analytics payload.
        """
        now_ts = time.time()
        gex_res = self.gex_engine.calculate_gex(chain_df, spot_price)
        profile = gex_res.get('profile', pd.Series()).sort_index()
        gamma_flip = float(gex_res.get('zero_gamma_level', 0.0))

        if profile.empty:
            return {'ok': False}

        call_wall = float(profile.idxmax())
        put_wall  = float(profile.idxmin())
        call_wall_gex_cr = float(round(profile[call_wall] / 1e7, 1))
        put_wall_gex_cr  = float(round(profile[put_wall] / 1e7, 1))

        pins = self.detect_dominant_pins(profile, spot_price, now_ts)
        absorptions = self.detect_gex_absorption(
            profile=profile,
            gamma_flip=gamma_flip,
            spot_price=spot_price,
            candles=recent_candles,
            now_ts=now_ts
        )

        top_abs = absorptions[0] if absorptions else None
        targets = self.project_explosion_targets(
            chain_df=chain_df,
            spot_price=spot_price,
            profile=profile,
            top_absorption=top_abs
        )

        # Structure candle summary for charting
        chart_candles = []
        if recent_candles:
            for c in recent_candles[-60:]:
                if len(c) >= 6:
                    chart_candles.append({
                        'time': datetime.fromtimestamp(c[0]).strftime("%H:%M"),
                        'open': float(c[1]),
                        'high': float(c[2]),
                        'low': float(c[3]),
                        'close': float(c[4]),
                        'volume': float(c[5])
                    })

        return {
            'ok': True,
            'spot': spot_price,
            'timestamp': datetime.fromtimestamp(now_ts).strftime("%Y-%m-%d %H:%M:%S"),
            'call_wall': {'strike': call_wall, 'gex_cr': call_wall_gex_cr},
            'put_wall': {'strike': put_wall, 'gex_cr': put_wall_gex_cr},
            'gamma_flip': round(gamma_flip, 1),
            'active_pins': pins,
            'retest_absorptions': absorptions,
            'explosion_targets': targets,
            'chart_candles': chart_candles,
            'recent_history': self._recent_absorptions[-5:]
        }
