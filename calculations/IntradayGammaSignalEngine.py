"""
calculations/IntradayGammaSignalEngine.py
=============================================================================
Intraday Gamma Signal Fusion Engine.

Fuses the three reel methodologies into a single real-time, intraday-aware
decision layer:

  Reel 1 (quantedoptions): Pin timer + OI velocity unwind -> cascade release
  Reel 2 (aleksrosme):     GEX level retest + multi-candle absorption -> explosion
  GexRebalanceEngine:      Wall inversion + live OI velocity -> IGNITED status

Key additions over existing engines:
  1. compute_oi_velocity()        -- per-strike OI delta/min from snapshot ring
  2. score_candle_absorption()    -- 3-5 bar wick confirmation (not single-candle)
  3. classify_swing_quality()     -- ADR expansion + RSI momentum + session phase
  4. update()                     -- master call: feeds all child engines + emits
                                     a single unified IntradaySignal dict
=============================================================================
"""

from __future__ import annotations

import time
import math
from collections import deque
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .GammaExplosionEngine import GammaExplosionEngine
from .GexRebalanceEngine import GexRebalanceEngine
from .GexEngine import GexEngine

try:
    import config as _cfg
    _DEFAULT_LOT_SIZE = _cfg.get("nifty_lot_size", 65)
    _DEFAULT_R = _cfg.get("risk_free_rate", 0.051274)
    _DEFAULT_Q = _cfg.get("dividend_yield", 0.0122)
except Exception:
    _DEFAULT_LOT_SIZE = 65
    _DEFAULT_R = 0.051274
    _DEFAULT_Q = 0.0122


# Session phase boundaries (IST 24h)
_OPENING_RANGE_END   = (11, 0)   # 09:15 - 11:00
_MID_TREND_END       = (14, 0)   # 11:00 - 14:00
# 14:00+ = EXPIRY_HEAT on Wed/Thu, else LATE

# ADR expansion thresholds
_ADR_TRENDING_PCT    = 0.55      # today range > 55% of 10d ADR -> trending
_ADR_COILED_PCT      = 0.30      # today range < 30% of 10d ADR -> coiled

# Multi-candle absorption
_MIN_ABSORPTION_BARS = 2         # of last 5 bars must show rejection wick
_WICK_THRESHOLD_PCT  = 30.0      # wick >= 30% of bar range qualifies
_VOL_ACCEL_MULT      = 1.5       # volume on wick bars >= 1.5x 5-bar avg

# OI velocity
_OI_VIOLENT_DELTA    = -100_000  # OI delta/min below this = violent unwind

# Signal scores
_SCORE_ACTIONABLE    = 65.0
_SCORE_WATCH         = 45.0


class IntradayGammaSignalEngine:
    """
    Fusion layer: calls GammaExplosionEngine + GexRebalanceEngine with live
    intraday OI velocity and multi-candle absorption data, classifies swing
    quality, and emits a single unified actionable signal.
    """

    def __init__(self,
                 lot_size: int = _DEFAULT_LOT_SIZE,
                 gex_engine: Optional[GexEngine] = None):
        self.lot_size       = lot_size
        self.gex_engine     = gex_engine or GexEngine(lot_size=lot_size)
        self.pin_engine     = GammaExplosionEngine(lot_size=lot_size, gex_engine=self.gex_engine)
        self.rebalance_eng  = GexRebalanceEngine(lot_size=lot_size)
        self._spot_history: deque = deque(maxlen=30)

    # -------------------------------------------------------------------------
    # 1. OI VELOCITY ENRICHMENT
    # -------------------------------------------------------------------------

    def compute_oi_velocity(self, oi_velocity_data: dict) -> dict:
        """
        Interprets pre-computed OI velocity dict from SharedDataCache.get_oi_velocity_data().
        Labels each strike as BUILDING / STABLE / UNWINDING / VIOLENT_CAPITULATION / ACCELERATING_UNWIND / EXHAUSTING_UNWIND.

        Args:
            oi_velocity_data: dict from SharedDataCache.get_oi_velocity_data()

        Returns:
            Enriched dict with 'labels', 'has_capitulation', 'accelerating_unwind',
            'exhausting_unwind', 'fortress_building', 'available'.
        """
        if not oi_velocity_data or "vel_by_strike" not in oi_velocity_data:
            return {"available": False}

        vel = oi_velocity_data["vel_by_strike"]
        accel = oi_velocity_data.get("accel_by_strike", {})
        labels = {}
        for key, v in vel.items():
            a = accel.get(key, 0.0)
            if v < _OI_VIOLENT_DELTA:
                labels[key] = "VIOLENT_CAPITULATION"
            elif v < -40_000 and a < -5_000:
                labels[key] = "ACCELERATING_UNWIND"
            elif v < -40_000 and a > 5_000:
                labels[key] = "EXHAUSTING_UNWIND"
            elif v < -30_000:
                labels[key] = "UNWINDING"
            elif v > 50_000 and a > 5_000:
                labels[key] = "FORTRESS_DEFENSE"
            elif v > 30_000:
                labels[key] = "BUILDING"
            else:
                labels[key] = "STABLE"

        violent = oi_velocity_data.get("violently_unwinding", [])
        accel_unwind = oi_velocity_data.get("accelerating_unwind", [])
        exhaust_unwind = oi_velocity_data.get("exhausting_unwind", [])
        fortress = oi_velocity_data.get("fortress_building", [])

        return {
            "available":            True,
            "vel_by_strike":        vel,
            "fast_vel_by_strike":   oi_velocity_data.get("fast_vel_by_strike", {}),
            "accel_by_strike":      accel,
            "pct_vel_by_strike":    oi_velocity_data.get("pct_vel_by_strike", {}),
            "labels":               labels,
            "violently_unwinding":  violent,
            "accelerating_unwind":  accel_unwind,
            "exhausting_unwind":    exhaust_unwind,
            "fortress_building":    fortress,
            "has_capitulation":     len(violent) > 0 or len(accel_unwind) > 0,
            "is_warmed_up":         oi_velocity_data.get("is_warmed_up", False),
            "elapsed_min":          oi_velocity_data.get("elapsed_min", 0.0),
            "snapshot_count":       oi_velocity_data.get("snapshot_count", 0)
        }

    def evaluate_oi_velocity_triggers(self,
                                      spot: float,
                                      call_wall: float,
                                      put_wall: float,
                                      oi_velocity_data: dict) -> dict:
        """
        Evaluates institutional 15m OI velocity vectors and acceleration to generate
        concrete, high-conviction Entry & Exit triggers for both Option Buyers and Option Sellers.

        Returns:
            {
                'buyer_trigger': {'signal': str, 'strike': float, 'urgency': str, 'rationale': str},
                'seller_trigger': {'signal': str, 'strike': float, 'urgency': str, 'rationale': str},
                'exit_warning': {'signal': str, 'strike': float, 'urgency': str, 'rationale': str} or None
            }
        """
        vel_map = oi_velocity_data.get("vel_by_strike", {})
        accel_map = oi_velocity_data.get("accel_by_strike", {})

        cw_vel = vel_map.get((call_wall, 'CE'), 0.0)
        cw_accel = accel_map.get((call_wall, 'CE'), 0.0)
        pw_vel = vel_map.get((put_wall, 'PE'), 0.0)
        pw_accel = accel_map.get((put_wall, 'PE'), 0.0)

        buyer_trigger = {
            "signal": "MONITORING",
            "strike": 0.0,
            "urgency": "LOW",
            "rationale": "OI velocity within normal boundaries."
        }
        seller_trigger = {
            "signal": "SAFE_HOLD",
            "strike": 0.0,
            "urgency": "LOW",
            "rationale": "No aggressive writer defense or capitulation detected."
        }
        exit_warning = None

        dist_cw = call_wall - spot
        dist_pw = spot - put_wall

        # ── 1. Option Buyer Breakout Triggers ──
        if 0 <= dist_cw <= 35.0 and (cw_vel < -35_000 or cw_accel < -4_000):
            buyer_trigger = {
                "signal": "BUY_CALL_MOMENTUM",
                "strike": call_wall,
                "urgency": "HIGH" if cw_accel < -8_000 else "MEDIUM",
                "rationale": f"Call Wall {call_wall:.0f} writers unwinding: Vel={cw_vel:+,.0f}/m, Accel={cw_accel:+,.0f}/m²."
            }
        elif 0 <= dist_pw <= 35.0 and (pw_vel < -35_000 or pw_accel < -4_000):
            buyer_trigger = {
                "signal": "BUY_PUT_BREAKDOWN",
                "strike": put_wall,
                "urgency": "HIGH" if pw_accel < -8_000 else "MEDIUM",
                "rationale": f"Put Wall {put_wall:.0f} writers unwinding: Vel={pw_vel:+,.0f}/m, Accel={pw_accel:+,.0f}/m²."
            }

        # ── 2. Option Buyer Momentum Exhaustion (Exit) ──
        if dist_cw < 0 and cw_accel > 6_000:
            exit_warning = {
                "signal": "EXIT_CALL_EXHAUSTION",
                "strike": call_wall,
                "urgency": "MODERATE",
                "rationale": f"Call squeeze momentum decelerating (+{cw_accel:+,.0f}/m² curvature). Lock profits."
            }
        elif dist_pw < 0 and pw_accel > 6_000:
            exit_warning = {
                "signal": "EXIT_PUT_EXHAUSTION",
                "strike": put_wall,
                "urgency": "MODERATE",
                "rationale": f"Put cascade momentum decelerating (+{pw_accel:+,.0f}/m² curvature). Lock profits."
            }

        # ── 3. Option Seller Resistance / Support Entry Triggers ──
        if abs(dist_cw) <= 40.0 and cw_vel > 40_000:
            seller_trigger = {
                "signal": "SELL_CE_RESISTANCE",
                "strike": call_wall,
                "urgency": "MEDIUM",
                "rationale": f"Call writers aggressively defending ceiling (+{cw_vel:+,.0f}/m addition)."
            }
        elif abs(dist_pw) <= 40.0 and pw_vel > 40_000:
            seller_trigger = {
                "signal": "SELL_PE_SUPPORT",
                "strike": put_wall,
                "urgency": "MEDIUM",
                "rationale": f"Put writers aggressively defending floor (+{pw_vel:+,.0f}/m addition)."
            }

        # ── 4. Option Seller Emergency Stop-Loss Triggers ──
        for (strike, otype), v in vel_map.items():
            if abs(strike - spot) <= 45.0 and v < -45_000:
                acc = accel_map.get((strike, otype), 0.0)
                if acc < 0:
                    exit_warning = {
                        "signal": f"EMERGENCY_STOP_{otype}",
                        "strike": strike,
                        "urgency": "CRITICAL",
                        "rationale": f"Severe writer capitulation on {otype} {strike:.0f} (Vel={v:+,.0f}/m). Cut short position immediately!"
                    }
                    break

        return {
            "buyer_trigger": buyer_trigger,
            "seller_trigger": seller_trigger,
            "exit_warning": exit_warning
        }

    # -------------------------------------------------------------------------
    # 2. MULTI-CANDLE ABSORPTION SCORER (Reel 2 verbatim: 3-5 bar confirmation)
    # -------------------------------------------------------------------------

    def score_candle_absorption(self,
                                candles: List[List[Any]],
                                level: float,
                                barrier_type: str = "SUPPORT") -> dict:
        """
        Scores GEX level absorption over the last 5 x 1-min candles.

        Requires >=2 of 5 bars showing:
          - Rejection wick >= 30% of bar range at the level
          - Volume >= 1.5x 5-bar trailing average on those wick bars
          - Delta alignment (buyer/seller flow direction matches barrier type)

        Args:
            candles:      List of [ts, open, high, low, close, volume] 1-min bars
            level:        GEX level being retested (Put Wall / Call Wall / Gamma Flip)
            barrier_type: 'SUPPORT' or 'RESISTANCE'

        Returns:
            score (0-100), confirmed (bool), setup_quality, wick_bars, vol_accel
        """
        empty = {"score": 0.0, "confirmed": False, "setup_quality": "NONE",
                 "wick_bars": 0, "vol_accel": 0.0, "delta_alignment": False}

        if not candles or len(candles) < 3 or level <= 0:
            return empty

        window = list(candles)[-5:]
        if len(window) < 3:
            return empty

        # Proximity check: at least one bar must have touched the level vicinity
        proximity_ok = False
        for c in window:
            if barrier_type == "SUPPORT" and float(c[3]) <= (level + 10.0):
                proximity_ok = True
                break
            elif barrier_type == "RESISTANCE" and float(c[2]) >= (level - 10.0):
                proximity_ok = True
                break

        if not proximity_ok:
            return empty

        wick_bars_list   = []
        non_wick_vols    = []
        wick_bar_vols    = []
        total_buyer_vol  = 0.0
        total_seller_vol = 0.0

        for c in window:
            c_open  = float(c[1])
            c_high  = float(c[2])
            c_low   = float(c[3])
            c_close = float(c[4])
            c_vol   = float(c[5])
            c_range = max(c_high - c_low, 1.0)

            if barrier_type == "SUPPORT":
                lower_wick = max(0.0, min(c_open, c_close) - c_low)
                wick_pct   = (lower_wick / c_range) * 100.0
                is_wick    = (wick_pct >= _WICK_THRESHOLD_PCT and
                              c_low <= (level + 10.0) and
                              c_close > c_low)
                # Approximate flow from candle direction
                if c_close >= c_open:
                    total_buyer_vol  += c_vol * 0.7
                    total_seller_vol += c_vol * 0.3
                else:
                    total_buyer_vol  += c_vol * 0.3
                    total_seller_vol += c_vol * 0.7
            else:  # RESISTANCE
                upper_wick = max(0.0, c_high - max(c_open, c_close))
                wick_pct   = (upper_wick / c_range) * 100.0
                is_wick    = (wick_pct >= _WICK_THRESHOLD_PCT and
                              c_high >= (level - 10.0) and
                              c_close < c_high)
                if c_close <= c_open:
                    total_buyer_vol  += c_vol * 0.3
                    total_seller_vol += c_vol * 0.7
                else:
                    total_buyer_vol  += c_vol * 0.7
                    total_seller_vol += c_vol * 0.3

            if is_wick:
                wick_bar_vols.append(c_vol)
            else:
                if c_vol > 0:
                    non_wick_vols.append(c_vol)

        wick_bar_count = len(wick_bar_vols)
        if non_wick_vols:
            baseline_vol = float(np.mean(non_wick_vols))
        else:
            all_vols = [float(c[5]) for c in window if float(c[5]) > 0]
            baseline_vol = float(np.mean(all_vols[:-1])) if len(all_vols) > 1 else max(float(all_vols[0]), 1.0) if all_vols else 1.0
        baseline_vol = max(baseline_vol, 1.0)

        avg_wick_vol = float(np.mean(wick_bar_vols)) if wick_bar_vols else 0.0
        vol_accel    = round(avg_wick_vol / baseline_vol, 2)

        total_flow = total_buyer_vol + total_seller_vol
        if total_flow > 0 and barrier_type == "SUPPORT":
            delta_alignment = (total_buyer_vol / total_flow) >= 0.55
        elif total_flow > 0:
            delta_alignment = (total_seller_vol / total_flow) >= 0.55
        else:
            delta_alignment = False

        # Composite score: wick(40%) + vol_accel(35%) + delta(25%)
        wick_score  = min(100.0, (wick_bar_count / max(_MIN_ABSORPTION_BARS, 1)) * 100.0) * 0.40
        vol_score   = min(100.0, (vol_accel / _VOL_ACCEL_MULT) * 100.0) * 0.35
        delta_score = 25.0 if delta_alignment else 0.0
        score       = round(wick_score + vol_score + delta_score, 1)

        confirmed = (wick_bar_count >= _MIN_ABSORPTION_BARS and vol_accel >= _VOL_ACCEL_MULT)

        if confirmed and score >= 75.0:
            quality = "STRONG"
        elif confirmed and score >= 55.0:
            quality = "MODERATE"
        elif wick_bar_count >= 1 and score >= 35.0:
            quality = "WEAK"
        else:
            quality = "NONE"

        return {
            "score":           score,
            "confirmed":       confirmed,
            "wick_bars":       wick_bar_count,
            "vol_accel":       vol_accel,
            "delta_alignment": delta_alignment,
            "setup_quality":   quality
        }

    # -------------------------------------------------------------------------
    # 3. SWING QUALITY CLASSIFIER
    # -------------------------------------------------------------------------

    def classify_swing_quality(self,
                               candles: List[List[Any]],
                               ohlc_df: Optional[pd.DataFrame] = None) -> dict:
        """
        Determines whether today is a trending/coiled/choppy session.

        Checks:
          A. ADR Expansion: today range vs 10-day Average Daily Range
          B. RSI on 5-min closes (aggregated from 1-min bars)
          C. Session Phase: Opening Range / Mid Trend / Expiry Heat / Late
          D. Directional consistency of last 10 bars

        Returns quality, session_phase, adr_pct, rsi_5min, directional_bars.
        """
        now    = datetime.now()
        hour   = now.hour
        minute = now.minute

        # Session phase
        if (hour, minute) < _OPENING_RANGE_END:
            phase = "OPENING_RANGE"
        elif (hour, minute) < _MID_TREND_END:
            phase = "MID_TREND"
        elif now.weekday() in (2, 3):
            phase = "EXPIRY_HEAT"
        else:
            phase = "LATE"

        # ADR expansion
        adr_pct = 0.0
        if candles and len(candles) >= 5:
            today_high  = max(float(c[2]) for c in candles)
            today_low   = min(float(c[3]) for c in candles)
            today_range = today_high - today_low

            if ohlc_df is not None and not ohlc_df.empty and len(ohlc_df) >= 10:
                ranges  = ohlc_df["high"].tail(10) - ohlc_df["low"].tail(10)
                adr_10d = float(ranges.mean())
                if adr_10d > 0:
                    adr_pct = round((today_range / adr_10d) * 100.0, 1)

        # RSI on 5-min closes
        rsi_5min = 50.0
        if candles and len(candles) >= 15:
            closes_1min = [float(c[4]) for c in candles]
            closes_5min = [closes_1min[i + 4] for i in range(0, len(closes_1min) - 4, 5)]
            if len(closes_5min) >= 5:
                rsi_5min = self._rsi(closes_5min, period=min(14, len(closes_5min) - 1))

        # Directional consistency
        directional_bars = 0
        if candles and len(candles) >= 5:
            last10    = list(candles)[-10:]
            up_bars   = sum(1 for c in last10 if float(c[4]) > float(c[1]))
            down_bars = sum(1 for c in last10 if float(c[4]) < float(c[1]))
            directional_bars = max(up_bars, down_bars)

        is_trending = (adr_pct >= _ADR_TRENDING_PCT * 100 or
                       (rsi_5min >= 65.0 or rsi_5min <= 35.0) or
                       directional_bars >= 7)
        is_coiled   = (adr_pct <= _ADR_COILED_PCT * 100 and
                       40.0 < rsi_5min < 60.0 and
                       directional_bars <= 4)

        if is_trending:
            quality = "TRENDING"
        elif is_coiled:
            quality = "COILED"
        else:
            quality = "CHOPPY"

        return {
            "quality":          quality,
            "session_phase":    phase,
            "adr_pct":          adr_pct,
            "rsi_5min":         round(rsi_5min, 1),
            "directional_bars": directional_bars,
            "is_trending":      is_trending,
            "is_coiled":        is_coiled
        }

    # -------------------------------------------------------------------------
    # 4. MASTER UPDATE
    # -------------------------------------------------------------------------

    def update(self,
               spot: float,
               chain_df: pd.DataFrame,
               oi_velocity_data: Optional[dict] = None,
               candles: Optional[List[List[Any]]] = None,
               ohlc_df: Optional[pd.DataFrame] = None,
               dte: float = 1.0) -> dict:
        """
        Master fusion call. Feeds all child engines with live intraday data.

        Args:
            spot:             Current spot price
            chain_df:         Live option chain DataFrame
            oi_velocity_data: From SharedDataCache.get_oi_velocity_data()
            candles:          Recent 1-min OHLCV candles (list of lists)
            ohlc_df:          Daily OHLC from SharedDataCache (for ADR)
            dte:              Days to expiry

        Returns:
            Unified IntradaySignal dict with actionable entry_signal when score >= 65.
        """
        if spot <= 0 or chain_df is None or chain_df.empty:
            return {"ok": False, "reason": "Insufficient data"}

        self._spot_history.append(spot)
        candles  = candles or []
        now_str  = datetime.now().strftime("%H:%M:%S")

        # Step 1: Swing Quality
        swing = self.classify_swing_quality(candles, ohlc_df)

        # Step 2: OI Velocity
        oi_vel = self.compute_oi_velocity(oi_velocity_data or {})

        # Step 3: GEX Profile
        gex_res    = self.gex_engine.calculate_gex(chain_df, spot)
        profile    = gex_res.get("profile", pd.Series()).sort_index()
        gamma_flip = float(gex_res.get("zero_gamma_level", 0.0))
        net_gex    = float(gex_res.get("net_gex", 0.0))
        call_wall  = float(profile.idxmax()) if not profile.empty else spot + 200.0
        put_wall   = float(profile.idxmin()) if not profile.empty else spot - 200.0

        # Step 4: Pin Detection (Reel 1)
        pin_result  = self.pin_engine.get_full_status_payload(
            chain_df=chain_df,
            spot_price=spot,
            recent_candles=candles[-60:] if candles else None
        )
        active_pins = pin_result.get("active_pins", [])
        top_pin     = active_pins[0] if active_pins else {}

        # Step 5: Multi-Candle Absorption — find closest GEX level
        dist_put  = abs(spot - put_wall)
        dist_call = abs(spot - call_wall)
        dist_flip = abs(spot - gamma_flip) if gamma_flip > 0 else 9999.0

        closest_level, closest_barrier = put_wall, "SUPPORT"
        if dist_call < dist_put and dist_call < dist_flip:
            closest_level, closest_barrier = call_wall, "RESISTANCE"
        elif dist_flip < dist_put:
            closest_level = gamma_flip
            closest_barrier = "SUPPORT" if spot > gamma_flip else "RESISTANCE"

        absorption = self.score_candle_absorption(candles, closest_level, closest_barrier)

        # Step 6: GexRebalance with live OI velocity (the critical missing wire)
        vel_dict = oi_vel.get("vel_by_strike", {}) if oi_vel.get("available") else {}
        oi_vel_for_engine = {"vel_by_strike": vel_dict} if vel_dict else None

        rebalance        = self.rebalance_eng.evaluate(
            chain_df=chain_df, spot=spot,
            oi_velocity_data=oi_vel_for_engine,
            dte=dte
        )
        rebalance_status = rebalance.get("status", "MONITORING")

        # Step 7: Momentum
        momentum_status = self._compute_momentum_status()

        # Step 8: Composite Score
        score     = 0.0
        rationale = []

        # A. Swing quality gate
        if swing["quality"] == "CHOPPY":
            score -= 20.0
            rationale.append("WARNING: Choppy day — reduce conviction.")
        elif swing["quality"] == "TRENDING":
            score += 15.0
            rationale.append(f"Trending day ({swing['adr_pct']:.0f}% ADR reached).")
        elif swing["quality"] == "COILED":
            score += 10.0
            rationale.append("Coiled structure — breakout potential.")

        # B. Multi-candle absorption (Reel 2 enhanced)
        abs_quality = absorption.get("setup_quality", "NONE")
        if abs_quality == "STRONG":
            score += 30.0
            rationale.append(
                f"STRONG absorption at {closest_level:.0f} "
                f"({absorption['wick_bars']} wick bars, {absorption['vol_accel']:.1f}x vol)."
            )
        elif abs_quality == "MODERATE":
            score += 18.0
            rationale.append(f"Moderate absorption at {closest_level:.0f}.")
        elif abs_quality == "WEAK":
            score += 8.0
            rationale.append(f"Weak absorption signal at {closest_level:.0f}.")

        # C. OI velocity / Reel 1 unwind
        if oi_vel.get("has_capitulation"):
            score += 25.0
            violent     = oi_vel.get("violently_unwinding", [])
            strikes_str = ", ".join(f"{k[0]:.0f}{k[1]}" for k in violent[:3])
            rationale.append(
                f"VIOLENT OI CAPITULATION at {strikes_str} — pin release imminent."
            )
        elif oi_vel.get("available"):
            wall_strikes = {(put_wall, "PE"), (call_wall, "CE")}
            labels       = oi_vel.get("labels", {})
            for ws in wall_strikes:
                lbl = labels.get(ws, "STABLE")
                if lbl in ("UNWINDING", "VIOLENT_CAPITULATION"):
                    score += 12.0
                    rationale.append(f"{ws[1]} wall {ws[0]:.0f} OI unwinding.")

        # D. GexRebalance ignition
        if rebalance_status == "IGNITED":
            score += 20.0
            rt = rebalance.get("rebalance_target", 0.0)
            rationale.append(f"IGNITED: Wall 2 breached + OI capitulating -> {rt:.0f}.")
        elif rebalance_status == "COILING":
            score += 5.0
            rationale.append("COILING: Wall inversion in place, awaiting ignition.")

        # E. Session phase
        if swing["session_phase"] == "MID_TREND":
            score += 8.0
            rationale.append("Mid-trend window (11:00-14:00) — highest swing reliability.")
        elif swing["session_phase"] == "OPENING_RANGE":
            score -= 5.0
            rationale.append("Opening range — wait for direction confirmation.")

        # F. Pin unwind risk (Reel 1)
        if top_pin and top_pin.get("unpinning_risk") == "IMMINENT":
            score += 15.0
            rationale.append(
                f"Pin release at {top_pin.get('strike', 0):.0f} IMMINENT "
                f"({top_pin.get('duration_str', '')})."
            )

        score = round(max(0.0, min(100.0, score)), 1)

        # Step 9: Entry Signal
        actionable   = score >= _SCORE_ACTIONABLE
        entry_signal = None
        if actionable:
            entry_signal = self._build_entry_signal(
                spot=spot,
                rebalance=rebalance,
                absorption=absorption,
                closest_level=closest_level,
                closest_barrier=closest_barrier,
                momentum_status=momentum_status,
                oi_vel=oi_vel
            )

        # Step 10: OI Velocity Tactical Triggers (Buyer & Seller Entry / Exit)
        oi_triggers = self.evaluate_oi_velocity_triggers(
            spot=spot,
            call_wall=call_wall,
            put_wall=put_wall,
            oi_velocity_data=oi_velocity_data or {}
        )

        return {
            "ok":                   True,
            "timestamp":            now_str,
            "spot":                 spot,
            "score":                score,
            "actionable":           actionable,
            "swing_quality":        swing,
            "oi_velocity":          oi_vel if oi_vel.get("available") else {"available": False},
            "oi_velocity_triggers": oi_triggers,
            "absorption":           absorption,
            "pin_status":           top_pin,
            "rebalance_radar":      rebalance,
            "gex_summary": {
                "net_gex":    round(net_gex / 1e7, 2),
                "call_wall":  call_wall,
                "put_wall":   put_wall,
                "gamma_flip": gamma_flip
            },
            "momentum_status": momentum_status,
            "rationale":       rationale,
            "entry_signal":    entry_signal
        }

    # -------------------------------------------------------------------------
    # HELPERS
    # -------------------------------------------------------------------------

    def _compute_momentum_status(self) -> str:
        """Momentum from rolling spot history. 7+ of 10 same direction = LONG/SHORT."""
        history = list(self._spot_history)
        if len(history) < 6:
            return "NEUTRAL"
        last10 = history[-10:]
        ups    = sum(1 for i in range(1, len(last10)) if last10[i] > last10[i - 1])
        downs  = len(last10) - 1 - ups
        if ups >= 7:
            return "LONG"
        if downs >= 7:
            return "SHORT"
        return "NEUTRAL"

    def _rsi(self, closes: List[float], period: int = 14) -> float:
        """Wilder RSI."""
        if len(closes) < period + 1:
            return 50.0
        deltas   = [closes[i] - closes[i - 1] for i in range(1, len(closes))]
        gains    = [max(d, 0.0) for d in deltas]
        losses   = [max(-d, 0.0) for d in deltas]
        avg_gain = float(np.mean(gains[:period]))
        avg_loss = float(np.mean(losses[:period]))
        for i in range(period, len(gains)):
            avg_gain = (avg_gain * (period - 1) + gains[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        if avg_loss == 0:
            return 100.0
        return round(100.0 - 100.0 / (1.0 + avg_gain / avg_loss), 1)

    def _build_entry_signal(self,
                            spot: float,
                            rebalance: dict,
                            absorption: dict,
                            closest_level: float,
                            closest_barrier: str,
                            momentum_status: str,
                            oi_vel: dict) -> dict:
        """Build actionable entry dict. Uses GexRebalanceEngine dual-strikes when ignited."""
        direction = "BULLISH"
        if closest_barrier == "RESISTANCE" or momentum_status == "SHORT":
            direction = "BEARISH"

        primary    = rebalance.get("primary_option", {})
        otm_rocket = rebalance.get("otm_gamma_rocket", {})
        reb_target = rebalance.get("rebalance_target", 0.0)

        if rebalance.get("status") == "IGNITED" and primary:
            cur_price  = primary.get("current_price", spot)
            entry_zone = [round(cur_price * 0.98, 1), round(cur_price * 1.02, 1)]
            trig       = rebalance.get("trigger_strike", spot)
            sl_spot    = round(trig - 30.0, 1) if direction == "BULLISH" else round(trig + 30.0, 1)
            t1_spot    = primary.get("target_1", reb_target)
            t2_spot    = primary.get("target_2", reb_target + 50.0)
            opt_type   = primary.get("type", "CE" if direction == "BULLISH" else "PE")
            opt_strike = primary.get("strike", round(spot / 50) * 50)
            opt_price  = primary.get("current_price", 0.0)
        else:
            sl_off     = 20.0
            tgt_off    = 100.0 if absorption.get("setup_quality") == "STRONG" else 50.0
            if direction == "BULLISH":
                entry_zone = [round(closest_level + 5.0, 1), round(closest_level + 20.0, 1)]
                sl_spot    = round(closest_level - sl_off, 1)
                t1_spot    = round(closest_level + 50.0, 1)
                t2_spot    = round(closest_level + tgt_off, 1)
                opt_type   = "CE"
                opt_strike = int(round((closest_level + 50.0) / 50.0) * 50)
            else:
                entry_zone = [round(closest_level - 20.0, 1), round(closest_level - 5.0, 1)]
                sl_spot    = round(closest_level + sl_off, 1)
                t1_spot    = round(closest_level - 50.0, 1)
                t2_spot    = round(closest_level - tgt_off, 1)
                opt_type   = "PE"
                opt_strike = int(round((closest_level - 50.0) / 50.0) * 50)
            opt_price = 0.0

        risk_pts   = round(abs(entry_zone[0] - sl_spot), 1)
        reward_pts = round(abs(t2_spot - entry_zone[0]), 1)
        rr         = round(reward_pts / max(risk_pts, 1.0), 2)

        rationale = [
            f"{'Bullish' if direction == 'BULLISH' else 'Bearish'} setup from "
            f"{closest_level:.0f} {closest_barrier.lower()} retest.",
            f"Absorption: {absorption.get('setup_quality', 'N/A')} "
            f"({absorption.get('wick_bars', 0)} rejection bars, "
            f"{absorption.get('vol_accel', 0.0):.1f}x vol).",
            f"Target {t2_spot:.0f} ({reward_pts:.0f}pts) | SL {sl_spot:.0f} "
            f"({risk_pts:.0f}pts) | R:R {rr:.1f}."
        ]

        return {
            "direction":     direction,
            "entry_zone":    entry_zone,
            "sl_spot":       sl_spot,
            "t1":            t1_spot,
            "t2":            t2_spot,
            "option_strike": opt_strike,
            "option_type":   opt_type,
            "option_price":  opt_price,
            "risk_pts":      risk_pts,
            "reward_pts":    reward_pts,
            "rr_ratio":      rr,
            "otm_rocket":    otm_rocket if otm_rocket and otm_rocket.get("is_active") else None,
            "rationale":     rationale
        }
