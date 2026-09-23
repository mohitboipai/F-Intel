"""
calculations/IgnitionScannerEngine.py
=============================================================================
0DTE Gamma Ignition Scanner — Cross-Strike Compression → Ignition Detector.

Scans ATM ± N strikes (both CE and PE) every refresh cycle and detects the
pattern: premium sits in a cheap, compressed base → spot makes a fast
directional move → option premium explodes 2-4× via gamma acceleration + IV pop.

Stages:
  1. Chain Scan — rolling per-strike window maintained each cycle
  2. Compression Filter — range_pct below 20th-percentile + underlying ATR gate
  3. Ignition Trigger — spot ROC z-score + volume surge + OI rising + IV spike
  4. Confluence Scoring — composite across compression/ignition/GEX/OI/IV
  5. Entry/Exit Signal — Greeks snapshot, SL, tiered targets
=============================================================================
"""

import math
import time
import numpy as np
import collections
from typing import Dict, Any, Optional, List, Tuple
from scipy.stats import norm

try:
    import config as _cfg
    _DEFAULT_LOT_SIZE = _cfg.get("nifty_lot_size", 65)
    _DEFAULT_R = _cfg.get("risk_free_rate", 0.051274)
    _DEFAULT_Q = _cfg.get("dividend_yield", 0.0122)
except Exception:
    _cfg = None
    _DEFAULT_LOT_SIZE = 65
    _DEFAULT_R = 0.051274
    _DEFAULT_Q = 0.0122


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG DEFAULTS (overridden by config.py)
# ─────────────────────────────────────────────────────────────────────────────

def _c(key, default):
    if _cfg is not None:
        try:
            return _cfg.get(key, default)
        except Exception:
            return default
    return default


# ─────────────────────────────────────────────────────────────────────────────
# BSM HELPERS (reuse from GexRebalanceEngine pattern)
# ─────────────────────────────────────────────────────────────────────────────

def _bsm_greeks(S: float, K: float, T: float, r: float, q: float, iv: float,
                opt_type: str) -> Dict[str, float]:
    """Single-strike BSM Greeks (Merton 1973 dividend-adjusted)."""
    T = max(T, 1e-6)
    iv = max(iv, 0.01)
    sqrt_T = math.sqrt(T)
    d1 = (math.log(S / K) + (r - q + 0.5 * iv * iv) * T) / (iv * sqrt_T)
    d2 = d1 - iv * sqrt_T

    if opt_type == 'CE':
        delta = math.exp(-q * T) * norm.cdf(d1)
    else:
        delta = -math.exp(-q * T) * norm.cdf(-d1)

    gamma = math.exp(-q * T) * norm.pdf(d1) / (S * iv * sqrt_T)
    theta_day = -(S * iv * math.exp(-q * T) * norm.pdf(d1)) / (2.0 * sqrt_T * 365.0)
    vega = S * math.exp(-q * T) * norm.pdf(d1) * sqrt_T / 100.0

    return {
        'delta': round(delta, 4),
        'gamma': round(gamma, 6),
        'theta': round(theta_day, 4),
        'vega': round(vega, 4),
        'iv': round(iv * 100, 2)
    }


# ─────────────────────────────────────────────────────────────────────────────
# PER-STRIKE ROLLING STATE
# ─────────────────────────────────────────────────────────────────────────────

class _StrikeState:
    """Rolling window of per-strike observations for compression/ignition."""
    __slots__ = ('strike', 'opt_type', 'prices', 'oi_history', 'volumes',
                 'iv_history', 'range_pcts', 'last_ts')

    MAX_HISTORY = 60  # ~60 cycles × 10s ≈ 10 minutes of history

    def __init__(self, strike: float, opt_type: str):
        self.strike = strike
        self.opt_type = opt_type
        self.prices: collections.deque = collections.deque(maxlen=self.MAX_HISTORY)
        self.oi_history: collections.deque = collections.deque(maxlen=self.MAX_HISTORY)
        self.volumes: collections.deque = collections.deque(maxlen=self.MAX_HISTORY)
        self.iv_history: collections.deque = collections.deque(maxlen=self.MAX_HISTORY)
        self.range_pcts: collections.deque = collections.deque(maxlen=self.MAX_HISTORY)
        self.last_ts: float = 0.0

    def push(self, price: float, oi: int, volume: float, iv: float, ts: float):
        self.prices.append(price)
        self.oi_history.append(oi)
        self.volumes.append(volume)
        self.iv_history.append(iv)
        self.last_ts = ts

        # Compute rolling range_pct over last N observations
        if len(self.prices) >= 3:
            window = list(self.prices)[-8:] if len(self.prices) >= 8 else list(self.prices)
            if window and window[-1] > 0.5:
                rp = (max(window) - min(window)) / window[-1]
                self.range_pcts.append(rp)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN ENGINE
# ─────────────────────────────────────────────────────────────────────────────

class IgnitionScannerEngine:
    """
    Cross-strike scanner for compression → ignition patterns.
    Call `scan()` every refresh cycle with the live chain + spot data.
    Returns a ranked list of ignition candidates.
    """

    def __init__(self, lot_size: int = _DEFAULT_LOT_SIZE):
        self.lot_size = lot_size
        self.r = _DEFAULT_R
        self.q = _DEFAULT_Q

        # Per-strike rolling state: key = (strike, type)
        self._states: Dict[Tuple[float, str], _StrikeState] = {}

        # Underlying spot history for ATR and ROC
        self._spot_history: collections.deque = collections.deque(maxlen=120)
        self._spot_roc_history: collections.deque = collections.deque(maxlen=60)

    # ─────────────────────────────────────────────────────────────────────
    # STAGE 1: CHAIN SCAN — update rolling per-strike state
    # ─────────────────────────────────────────────────────────────────────

    def _update_strike_states(self, chain_df, spot: float, ts: float):
        """Feed fresh chain snapshot into per-strike rolling windows."""
        scan_range = int(_c("ignition_scan_range_strikes", 6))
        strike_step = float(_c("nifty_strike_step", 50))
        atm_strike = round(spot / strike_step) * strike_step

        # ATM ± scan_range strikes
        low_bound = atm_strike - scan_range * strike_step
        high_bound = atm_strike + scan_range * strike_step

        for _, row in chain_df.iterrows():
            strike = float(row.get('strike', 0))
            opt_type = str(row.get('type', '')).upper()
            if strike < low_bound or strike > high_bound:
                continue
            if opt_type not in ('CE', 'PE'):
                continue

            price = float(row.get('price', 0) or row.get('ltp', 0) or 0)
            oi = int(row.get('oi', 0) or 0)
            volume = float(row.get('volume', 0) or row.get('vol', 0) or 0)
            iv = float(row.get('iv', 0) or 0)
            # Normalize IV (some feeds send percentage, some decimal)
            if iv > 2.0:
                iv = iv / 100.0

            key = (strike, opt_type)
            if key not in self._states:
                self._states[key] = _StrikeState(strike, opt_type)

            # Throttle: don't push duplicate if < 5s since last push
            state = self._states[key]
            if ts - state.last_ts < 5.0:
                continue

            state.push(price, oi, volume, iv, ts)

        # Update spot history
        self._spot_history.append((ts, spot))

    # ─────────────────────────────────────────────────────────────────────
    # STAGE 2: COMPRESSION FILTER
    # ─────────────────────────────────────────────────────────────────────

    def _is_compressed(self, state: _StrikeState) -> Tuple[bool, float, float]:
        """
        Check if this strike's premium is in a compressed base.
        Returns (is_compressed, range_pct, compression_rank).
        """
        lookback = int(_c("ignition_compression_lookback", 8))
        pct_threshold = float(_c("ignition_compression_percentile", 20))
        premium_ceil = float(_c("ignition_premium_ceiling", 40.0))

        prices = list(state.prices)
        if len(prices) < lookback:
            return False, 0.0, 0.0

        recent = prices[-lookback:]
        current_price = recent[-1]

        # Must be below premium ceiling
        if current_price > premium_ceil or current_price < 0.5:
            return False, 0.0, 0.0

        # Range percentage
        high = max(recent)
        low = min(recent)
        if current_price < 0.5:
            return False, 0.0, 0.0
        range_pct = (high - low) / current_price

        # Compare against rolling distribution
        all_range_pcts = list(state.range_pcts)
        if len(all_range_pcts) < 5:
            # Not enough history — use absolute threshold
            is_comp = range_pct < 0.15  # 15% range = tight
            rank = max(0.0, 1.0 - range_pct / 0.30) if is_comp else 0.0
            return is_comp, range_pct, rank

        threshold = float(np.percentile(all_range_pcts, pct_threshold))
        # Compressed if below own percentile OR under tight absolute range (<= 10%)
        is_comp = bool((range_pct <= threshold) or (range_pct <= 0.10))

        # Rank: how compressed (0 = not compressed, 1 = extremely tight)
        if is_comp:
            eff_thresh = max(threshold, 0.10)
            rank = float(max(0.0, min(1.0, 1.0 - (range_pct / max(eff_thresh * 2, 0.01)))))
        else:
            rank = 0.0

        return is_comp, float(range_pct), float(rank)

    def _is_spot_compressed(self) -> bool:
        """Check if the underlying spot is also range-compressed."""
        atr_pct = float(_c("ignition_spot_atr_percentile", 25))
        spots = list(self._spot_history)
        if len(spots) < 10:
            return True  # Assume compressed if warming up (permissive)

        # Compute 8-bar ATR from spot history
        close_prices = [s[1] for s in spots]
        lookback = min(8, len(close_prices) - 1)
        recent_ranges = []
        for i in range(-lookback, 0):
            tr = abs(close_prices[i] - close_prices[i - 1])
            recent_ranges.append(tr)

        if not recent_ranges:
            return True

        current_atr = np.mean(recent_ranges)

        # Compare against full history
        all_ranges = []
        for i in range(1, len(close_prices)):
            all_ranges.append(abs(close_prices[i] - close_prices[i - 1]))

        if len(all_ranges) < 10:
            return True

        threshold = float(np.percentile(all_ranges, atr_pct))
        return bool(current_atr <= threshold)

    # ─────────────────────────────────────────────────────────────────────
    # STAGE 3: IGNITION TRIGGER
    # ─────────────────────────────────────────────────────────────────────

    def _check_ignition(self, state: _StrikeState, spot: float
                        ) -> Tuple[bool, float, Dict[str, Any]]:
        """
        Check if this strike is igniting. Computed on underlying, confirmed on option.
        Returns (is_igniting, ignition_rank, details_dict).
        """
        spot_z_thresh = float(_c("ignition_spot_zscore_threshold", 2.0))
        vol_z_thresh = float(_c("ignition_volume_zscore_threshold", 1.5))
        iv_spike_pct = float(_c("ignition_iv_spike_pct", 0.05))

        details: Dict[str, Any] = {
            'spot_roc_zscore': 0.0, 'vol_zscore': 0.0,
            'oi_rising': False, 'iv_spiking': False,
            'direction_match': False
        }

        spots = list(self._spot_history)
        if len(spots) < 3:
            return False, 0.0, details

        # ── Spot ROC window ──
        roc_step = min(6, max(2, (len(spots) - 1) // 2)) if len(spots) < 18 else 12
        if len(spots) <= roc_step:
            return False, 0.0, details

        spot_now = spots[-1][1]
        spot_ago = spots[-roc_step][1]
        spot_roc = (spot_now - spot_ago) / max(spot_ago, 1.0) * 100.0

        roc_dist = []
        for i in range(roc_step, len(spots)):
            r = (spots[i][1] - spots[i - roc_step][1]) / max(spots[i - roc_step][1], 1.0) * 100.0
            roc_dist.append(r)

        if len(roc_dist) < 5:
            # Fallback during initial warmup: absolute ROC threshold (~0.15% in 1-2m)
            is_spot_igniting = abs(spot_roc) >= 0.15
            spot_roc_z = (spot_roc / 0.10)
            details['spot_roc_zscore'] = round(spot_roc_z, 2)
        else:
            roc_mean = float(np.mean(roc_dist))
            roc_std = float(np.std(roc_dist))
            if roc_std < 0.001:
                roc_std = 0.001
            spot_roc_z = (spot_roc - roc_mean) / roc_std
            details['spot_roc_zscore'] = round(spot_roc_z, 2)
            is_spot_igniting = abs(spot_roc_z) > spot_z_thresh

        # ── Direction Match ──
        direction_match = (
            (spot_roc > 0 and state.opt_type == 'CE') or
            (spot_roc < 0 and state.opt_type == 'PE')
        )
        details['direction_match'] = direction_match

        if not direction_match:
            return False, 0.0, details

        # ── Volume Surge ──
        vols = list(state.volumes)
        vol_surging = False
        vol_zscore = 0.0
        if len(vols) >= 5:
            recent_vol = vols[-1] if vols[-1] > 0 else max(1, sum(vols[-3:]) / 3)
            vol_mean = float(np.mean(vols[:-1])) if len(vols) > 1 else 1.0
            vol_std = float(np.std(vols[:-1])) if len(vols) > 1 else 1.0
            if vol_std < 1.0:
                vol_std = 1.0
            vol_zscore = (recent_vol - vol_mean) / vol_std
            vol_surging = vol_zscore > vol_z_thresh
        details['vol_zscore'] = round(vol_zscore, 2)
        details['vol_surge'] = vol_surging

        # ── OI Rising (fresh buying, not short covering) ──
        ois = list(state.oi_history)
        oi_rising = False
        if len(ois) >= 3:
            oi_delta = ois[-1] - ois[-3]
            oi_rising = oi_delta > 0
        details['oi_rising'] = oi_rising

        # ── IV Spike ──
        ivs = list(state.iv_history)
        iv_spiking = False
        if len(ivs) >= 3:
            iv_now = ivs[-1]
            iv_ago = ivs[-3]
            if iv_ago > 0.01:
                iv_change = (iv_now - iv_ago) / iv_ago
                iv_spiking = iv_change > iv_spike_pct
        details['iv_spiking'] = iv_spiking

        # ── Composite Ignition Check ──
        if not is_spot_igniting:
            return False, 0.0, details

        # Rank: how strong the ignition signal is (0.0 - 1.0)
        rank = 0.0
        rank += min(0.4, abs(spot_roc_z) / (spot_z_thresh * 2.0) * 0.4)
        if vol_surging:
            rank += 0.25
        if oi_rising:
            rank += 0.20
        if iv_spiking:
            rank += 0.15

        is_igniting = rank >= 0.3

        return is_igniting, round(rank, 3), details

    # ─────────────────────────────────────────────────────────────────────
    # STAGE 4: CONFLUENCE SCORING
    # ─────────────────────────────────────────────────────────────────────

    def _score_candidate(self, compression_rank: float, ignition_rank: float,
                         state: _StrikeState, gex_data: Optional[Dict],
                         oi_vel_data: Optional[Dict]) -> float:
        """
        Composite score across all modules. Returns 0-100 score.
        """
        score = 0.0

        # 1. Compression (0-30)
        score += 30.0 * compression_rank

        # 2. Ignition (0-25)
        score += 25.0 * ignition_rank

        # 3. Dealer Gamma Weight (0-20)
        if gex_data:
            net_gex = float(gex_data.get('net_gex', 0))
            if net_gex < 0:
                gex_score = min(1.0, abs(net_gex) / 1e9)
                score += 20.0 * gex_score
            elif net_gex > 5e8:
                score += 5.0

        # 4. OI Flow (0-15)
        if oi_vel_data and oi_vel_data.get('vel_by_strike'):
            key = (state.strike, state.opt_type)
            vel = oi_vel_data['vel_by_strike'].get(key, 0)
            if vel > 0:
                oi_score = min(1.0, vel / 50_000.0)
                score += 15.0 * oi_score
            else:
                score += 5.0

        # 5. IV Momentum (0-10)
        ivs = list(state.iv_history)
        if len(ivs) >= 5:
            iv_recent = np.mean(ivs[-3:])
            iv_older = np.mean(ivs[-6:-3]) if len(ivs) >= 6 else ivs[0]
            if iv_older > 0.01:
                iv_accel = (iv_recent - iv_older) / iv_older
                if iv_accel > 0:
                    iv_score = min(1.0, iv_accel / 0.10)
                    score += 10.0 * iv_score

        return round(max(0.0, min(100.0, score)), 1)

    # ─────────────────────────────────────────────────────────────────────
    # STAGE 5: ENTRY/EXIT SIGNAL WITH GREEKS
    # ─────────────────────────────────────────────────────────────────────

    def _build_entry_signal(self, state: _StrikeState, spot: float,
                            score: float, compression_rank: float,
                            ignition_details: Dict, dte: float
                            ) -> Dict[str, Any]:
        """Build the actionable entry signal for a candidate."""
        prices = list(state.prices)
        current_price = prices[-1] if prices else 0.0

        # Greeks at entry
        T = max(dte / 365.0, 1e-6)
        iv_val = list(state.iv_history)[-1] if state.iv_history else 0.15
        greeks = _bsm_greeks(spot, state.strike, T, self.r, self.q, iv_val, state.opt_type)

        # SL: Greeks-derived
        spot_data = list(self._spot_history)
        if len(spot_data) >= 10:
            recent_moves = [abs(spot_data[i][1] - spot_data[i-1][1]) for i in range(-8, 0)]
            atr = float(np.mean(recent_moves))
        else:
            atr = spot * 0.003

        delta = abs(greeks['delta'])
        sl_drop = delta * atr
        sl_premium = round(max(current_price * 0.50, current_price - sl_drop), 1)
        sl_premium = max(0.5, sl_premium)

        # Targets
        target_1 = round(current_price * 2.0, 1)
        target_2 = round(current_price * 3.0, 1)
        trail_pct = int(_c("ignition_trail_stop_pct", 25))

        # R:R
        risk = max(0.1, current_price - sl_premium)
        reward = target_1 - current_price
        rr = round(reward / risk, 2)

        comp_bars = min(len(prices), int(_c("ignition_compression_lookback", 8)))

        # Ignition source description
        z = ignition_details.get('spot_roc_zscore', 0)
        vol_z = ignition_details.get('vol_zscore', 0)
        sources = []
        spot_dir = "+" if state.opt_type == 'CE' else "-"
        sources.append(f"Spot {spot_dir}{abs(z):.1f}\u03c3 velocity")
        if vol_z > 1.0:
            sources.append(f"Volume {vol_z:.1f}\u03c3 surge")
        if ignition_details.get('oi_rising'):
            sources.append("OI rising (fresh buying)")
        if ignition_details.get('iv_spiking'):
            sources.append("IV accelerating")
        ignition_source = " + ".join(sources)

        return {
            'strike': state.strike,
            'type': state.opt_type,
            'entry_premium': round(current_price, 1),
            'sl_premium': sl_premium,
            'target_1': target_1,
            'target_1_pct': round(((target_1 - current_price) / max(current_price, 0.1)) * 100, 0),
            'target_2': target_2,
            'target_2_pct': round(((target_2 - current_price) / max(current_price, 0.1)) * 100, 0),
            'trail_stop_pct': trail_pct,
            'rr_ratio': rr,
            'greeks_at_entry': greeks,
            'confluence_score': score,
            'compression_rank': round(compression_rank, 2),
            'compression_bars': comp_bars,
            'ignition_source': ignition_source,
            'ignition_details': ignition_details,
            'status': 'IGNITING' if score >= float(_c("ignition_entry_threshold", 65)) else 'COILING',
            'timestamp': time.strftime("%H:%M:%S")
        }

    # ─────────────────────────────────────────────────────────────────────
    # PUBLIC API: SCAN
    # ─────────────────────────────────────────────────────────────────────

    def scan(self, chain_df, spot: float, dte: float = 1.0,
             gex_data: Optional[Dict] = None,
             oi_vel_data: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Master scan method. Call every refresh cycle (~10-15s).

        Args:
            chain_df: Live option chain DataFrame (strike, type, price, oi, iv, volume)
            spot: Current underlying spot price
            dte: Days to expiry (calendar)
            gex_data: Output from GexEngine.calculate_gex() (optional)
            oi_vel_data: Output from SharedDataCache.get_oi_velocity_data() (optional)

        Returns:
            {
                'ok': True,
                'candidates': [...],
                'scan_count': int,
                'compressed_count': int,
                'spot_compressed': bool,
                'timestamp': str
            }
        """
        if chain_df is None or chain_df.empty or spot <= 0:
            return {'ok': False, 'candidates': [], 'scan_count': 0}

        ts = time.time()
        max_candidates = int(_c("ignition_max_candidates", 6))

        # Stage 1: Update rolling per-strike state
        self._update_strike_states(chain_df, spot, ts)

        # Stage 2: Check if spot itself is compressed
        spot_compressed = self._is_spot_compressed()

        # Stage 3+4: Scan all tracked strikes
        candidates = []
        compressed_count = 0

        for key, state in self._states.items():
            if len(state.prices) < 5:
                continue

            current_price = list(state.prices)[-1]
            if current_price < 0.5:
                continue

            # Stage 2: Compression check
            is_comp, range_pct, comp_rank = self._is_compressed(state)
            if is_comp:
                compressed_count += 1

            # Stage 3: Ignition check
            is_igniting, ign_rank, ign_details = self._check_ignition(state, spot)

            if not is_comp and not is_igniting:
                continue

            effective_comp_rank = comp_rank if is_comp else 0.0
            effective_ign_rank = ign_rank if is_igniting else 0.0

            if effective_comp_rank == 0 and effective_ign_rank == 0:
                continue

            # Stage 4: Score
            score = self._score_candidate(
                effective_comp_rank, effective_ign_rank,
                state, gex_data, oi_vel_data
            )

            # Build entry signal for anything with score > 30
            if score >= 30.0:
                entry = self._build_entry_signal(
                    state, spot, score, effective_comp_rank,
                    ign_details, dte
                )
                candidates.append(entry)

        # Sort by score descending
        candidates.sort(key=lambda c: c['confluence_score'], reverse=True)
        candidates = candidates[:max_candidates]

        return {
            'ok': True,
            'candidates': candidates,
            'scan_count': len(self._states),
            'compressed_count': compressed_count,
            'spot_compressed': spot_compressed,
            'timestamp': time.strftime("%H:%M:%S")
        }

    def get_state_summary(self) -> Dict[str, Any]:
        """Debug: return summary of tracked strike states."""
        return {
            'tracked_strikes': len(self._states),
            'spot_history_len': len(self._spot_history),
            'strikes': [
                {
                    'key': f"{s.strike} {s.opt_type}",
                    'observations': len(s.prices),
                    'last_price': round(list(s.prices)[-1], 1) if s.prices else 0,
                    'range_pcts_count': len(s.range_pcts)
                }
                for s in list(self._states.values())[:10]
            ]
        }
