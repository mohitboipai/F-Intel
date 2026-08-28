"""
GammaExplosionModel.py — Gamma Explosion & Cascade Detector
=============================================================
Detects pre-conditions for dealer-hedging-driven price explosions:

  Signal 1: Per-strike GEX profile + ATM concentration (dealer hedging force)
  Signal 2: OI surge rate on ATM strikes (option accumulation)
  Signal 3: IV-RV divergence (cheap/expensive gamma)
  Signal 4: Skew velocity (directional bias of explosion)

Composite score 0-100 with alert levels:
  70+  → EXPLOSION IMMINENT
  50+  → ELEVATED RISK
  30+  → BUILDING
  <30  → NORMAL

Standalone module; reuses FyersAuth, OptionAnalytics.
Called from VolatilityAnalyzer menu option 6.
"""

import sys
import os
import time
import numpy as np
import pandas as pd
from datetime import datetime

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from FyersAuth import FyersAuthenticator
from OptionAnalytics import OptionAnalytics


class GammaExplosionModel:

    LOT_SIZE = 75  # NIFTY lot size

    def __init__(self, fyers_instance=None):
        if fyers_instance:
            self.fyers = fyers_instance
        else:
            self.fyers = self._authenticate()

        self.analytics   = OptionAnalytics()
        self.symbol      = "NSE:NIFTY50-INDEX"
        self.spot_price  = 0
        self.expiry_date = None

        # Rolling memory for OI surge detection
        self._prev_oi_snapshot = {}   # {strike_type: OI}
        self._prev_skew        = None
        self._gex_history      = []   # [(timestamp, net_gex, concentration)]

        # Historical RV baseline (fetched once)
        self._rv_5d  = 0.0
        self._rv_20d = 0.0
        self._hv_20d = 0.0

        # Shared signal memory (injected by VolatilityAnalyzer, optional)
        self.memory  = None

    # ──────────────────────────────────────────────────────────────────────────
    # AUTH / FETCH HELPERS
    # ──────────────────────────────────────────────────────────────────────────

    def _authenticate(self):


        from fyers_auth_manager import get_fyers_instance


        return get_fyers_instance()

    def _get_spot(self):
        if not self.fyers:
            return self.spot_price
        try:
            r = self.fyers.quotes(data={"symbols": self.symbol})
            if r.get('code') == -15 or "token" in r.get('message', '').lower():
                self.fyers = self._authenticate()
                if self.fyers:
                    r = self.fyers.quotes(data={"symbols": self.symbol})
                else:
                    return self.spot_price
            if r.get('s') == 'ok':
                self.spot_price = r['d'][0]['v'].get('lp', 0)
        except Exception as e:
            print(f"  Spot fetch error: {e}")
        return self.spot_price

    def _get_chain(self):
        """Fetch and parse full option chain. Returns DataFrame."""
        if not self.expiry_date or not self.fyers:
            return pd.DataFrame()
        try:
            dt = datetime.strptime(self.expiry_date, "%Y-%m-%d")
            ts = int(dt.timestamp())
        except Exception:
            dt = None
            ts = ""

        try:
            r = self.fyers.optionchain(data={
                "symbol": self.symbol,
                "strikecount": 500,
                "timestamp": ts
            })
            if r.get('code') == -15 or "token" in r.get('message', '').lower():
                self.fyers = self._authenticate()
                if self.fyers:
                    r = self.fyers.optionchain(data={
                        "symbol": self.symbol, "strikecount": 500, "timestamp": ts
                    })
                else:
                    return pd.DataFrame()

            # Handle expiry mismatch
            if r.get('s') == 'error' and 'expiryData' in r.get('data', {}):
                for item in r['data']['expiryData']:
                    try:
                        a_date = datetime.strptime(item['date'], "%d-%m-%Y").date()
                        if dt and a_date == dt.date():
                            ts = item['expiry']
                            r = self.fyers.optionchain(data={
                                "symbol": self.symbol, "strikecount": 500, "timestamp": ts
                            })
                            break
                    except Exception:
                        continue

            if r.get('s') == 'ok':
                return self.parse_chain(r['data'])
        except Exception as e:
            print(f"  Chain fetch error: {e}")
        return pd.DataFrame()

    def parse_chain(self, data):
        """Parse option chain into structured DataFrame"""
        if not data:
            return pd.DataFrame()
        
        options = data.get('optionsChain', [])
        if not options:
            return pd.DataFrame()
        
        records = []
        for item in options:
            strike = float(item.get('strike_price', 0))
            if strike <= 0: continue
            
            records.append({
                'strike': strike,
                'type':   'CE' if item.get('option_type', '') in ('CE', 'CALL') else 'PE',
                'ltp':    float(item.get('ltp', 0) or 0),
                'price':  float(item.get('ltp', 0) or 0), # alias
                'iv':     float(item.get('iv', 0) or 0),
                'oi':     int(item.get('oi', 0) or 0),
                'delta':  float(item.get('delta', 0) or 0),
                'gamma':  float(item.get('gamma', 0) or 0),
            })
        return pd.DataFrame(records)

    def _fetch_rv_baseline(self):
        """Fetch 1-year daily data and compute RV baseline (once)."""
        if not self.fyers:
            return
        print("  Fetching 60-day history for IV-RV baseline...")
        try:
            today = datetime.now()
            start = today - pd.Timedelta(days=365)
            r = self.fyers.history(data={
                "symbol": self.symbol, "resolution": "D", "date_format": "1",
                "range_from": start.strftime("%Y-%m-%d"),
                "range_to":   today.strftime("%Y-%m-%d"),
                "cont_flag":  "1"
            })
            if r.get('s') == 'ok':
                closes = pd.Series([c[4] for c in r['candles']])
                log_rets = np.log(closes / closes.shift(1)).dropna()

                rv5  = log_rets.tail(5).std()  * np.sqrt(252) * 100
                rv20 = log_rets.tail(20).std() * np.sqrt(252) * 100
                hv20 = log_rets.tail(20).std() * np.sqrt(252) * 100  # same as rv20 using log-rets

                self._rv_5d  = round(rv5,  2)
                self._rv_20d = round(rv20, 2)
                self._hv_20d = round(hv20, 2)
                print(f"  RV-5d: {self._rv_5d:.2f}%  RV-20d: {self._rv_20d:.2f}%")
        except Exception as e:
            print(f"  RV baseline error: {e}")

    # ──────────────────────────────────────────────────────────────────────────
    # SIGNAL 1 — PER-STRIKE GEX PROFILE & CONCENTRATION
    # ──────────────────────────────────────────────────────────────────────────

    def _compute_gex_profile(self, df, spot, T):
        """
        Build per-strike GEX using unified GEXEngine and return:
          - profile dict {strike: gex}
          - net_gex (aggregate)
          - gex_flip_point (zero crossing strike)
          - concentration (fraction of |GEX| within ±1% of spot)
          - gex_acceleration (local historical delta)
        """
        try:
            from GEXEngine import GEXEngine
            res = GEXEngine.compute_gex(df, spot, T, lot_size=self.LOT_SIZE)
            profile = res['profile']
            net_gex = res['net_gex']
            concentration = res['atm_concentration']
            flip_point = res['gex_flip_point']
        except Exception as e:
            print(f"Error computing GEX: {e}")
            return {}, 0, 0, 0, 0

        # GEX Acceleration Calculation (Ddelta / Dtime)
        now_epoch = time.time()
        self._gex_history.append((now_epoch, net_gex, concentration))
        # Keep 15m window (900 seconds)
        self._gex_history = [x for x in self._gex_history if now_epoch - x[0] <= 900]
        
        gex_acceleration = 0
        if len(self._gex_history) > 1:
            oldest_gex = self._gex_history[0][1]
            gex_acceleration = net_gex - oldest_gex

        return profile, net_gex, flip_point, concentration, gex_acceleration

    # ──────────────────────────────────────────────────────────────────────────
    # SIGNAL 2 — OI SURGE DETECTOR
    # ──────────────────────────────────────────────────────────────────────────

    def _compute_oi_surge(self, df, spot):
        """
        Compare current ATM OI to previous snapshot.
        Returns dict with surge stats and call/put build bias.
        """
        # Build current snapshot (ATM ± 5 strikes)
        atm_strikes = sorted(df['strike'].unique(),
                             key=lambda x: abs(x - spot))[:10]

        current_snap = {}
        for _, row in df[df['strike'].isin(atm_strikes)].iterrows():
            key = f"{row['strike']}_{row['type']}"
            current_snap[key] = row['oi']

        if not self._prev_oi_snapshot:
            self._prev_oi_snapshot = current_snap
            return {
                'max_surge_pct': 0, 'call_build': 0, 'put_build': 0,
                'bias': 0, 'surge_strike': 0, 'first_cycle': True
            }

        # Compute changes
        surges = {}
        call_delta = put_delta = 0

        for key, cur_oi in current_snap.items():
            prev_oi = self._prev_oi_snapshot.get(key, cur_oi)
            if prev_oi is not None and prev_oi > 0:
                pct = (cur_oi - prev_oi) / prev_oi * 100
                surges[key] = pct
                if '_CE' in key:
                    call_delta += max(0, cur_oi - prev_oi)
                else:
                    put_delta += max(0, cur_oi - prev_oi)

        self._prev_oi_snapshot = current_snap

        max_surge_key = max(surges, key=lambda x: abs(surges[x])) if surges else ''
        max_surge_pct = surges.get(max_surge_key, 0)
        surge_strike  = float(max_surge_key.split('_')[0]) if max_surge_key else 0

        total_build = call_delta + put_delta or 1
        bias = (call_delta - put_delta) / total_build  # +1 = all calls, -1 = all puts

        return {
            'max_surge_pct': max_surge_pct,
            'call_build': call_delta,
            'put_build':  put_delta,
            'bias': bias,
            'surge_strike': surge_strike,
            'first_cycle': False
        }

    # ──────────────────────────────────────────────────────────────────────────
    # SIGNAL 3 — IV-RV DIVERGENCE
    # ──────────────────────────────────────────────────────────────────────────

    def _compute_iv_rv(self, df, spot, T):
        """Compute ATM IV and compare to RV baseline."""
        df = df.copy()
        df['dist'] = abs(df['strike'] - spot)
        atm_row = df.loc[df['dist'].idxmin()]
        iv = atm_row['iv']
        if 0 < iv < 1:  # decimal
            iv *= 100
        if iv <= 0 or iv > 200:
            # Recalculate via BSM
            try:
                iv = self.analytics.implied_volatility(
                    atm_row['price'], spot, atm_row['strike'], T, 0.07,
                    atm_row['type']
                )
            except Exception:
                iv = 15.0

        rv5  = self._rv_5d
        rv20 = self._rv_20d
        consensus_rv = (rv5 * 0.4 + rv20 * 0.6) if rv5 > 0 and rv20 > 0 else max(rv5, rv20)
        rv_ratio = rv5 / rv20 if rv20 > 0 else 1.0  # acceleration
        iv_rv_ratio = iv / consensus_rv if consensus_rv > 0 else 1.0
        vrp = iv - consensus_rv

        return {
            'atm_iv': round(iv, 2),
            'rv_5d': rv5,
            'rv_20d': rv20,
            'consensus_rv': round(consensus_rv, 2),
            'rv_acceleration': round(rv_ratio, 2),
            'iv_rv_ratio': round(iv_rv_ratio, 3),
            'vrp': round(vrp, 2)
        }

    # ──────────────────────────────────────────────────────────────────────────
    # SIGNAL 4 — SKEW VELOCITY
    # ──────────────────────────────────────────────────────────────────────────

    def _compute_skew_velocity(self, df, spot):
        """Compute put-call skew and its change from last cycle."""
        try:
            pe_df = df[df['type'] == 'PE']
            ce_df = df[df['type'] == 'CE']
            p5_iv = pe_df.iloc[(pe_df['strike'] - spot * 0.95).abs().argmin()]['iv']
            c5_iv = ce_df.iloc[(ce_df['strike'] - spot * 1.05).abs().argmin()]['iv']
            skew = float(p5_iv - c5_iv)
        except Exception:
            skew = 0.0

        velocity = 0.0
        if self._prev_skew is not None:
            velocity = skew - self._prev_skew
        self._prev_skew = skew

        # Positive velocity = skew rising = puts getting pricier = downside bias
        # Negative velocity = skew falling = puts cheaper / calls rising = upside bias
        return {'skew': round(skew, 2), 'velocity': round(velocity, 3)}

    # ──────────────────────────────────────────────────────────────────────────
    # COMPOSITE SCORE
    # ──────────────────────────────────────────────────────────────────────────

    def _compute_score(self, concentration, net_gex, iv_rv, oi_surge):
        """
        0-100 composite explosion score.
        """
        # 1. Gamma concentration (0-100): >40% = max
        conc_score = min(100, concentration * 2.5)  # 40% → 100

        # 2. Short gamma score: negative GEX = destabilizing = high score
        # Scale adjusted for new `Spot * Spot * 0.01` calculation (values are ~200x larger)
        if net_gex < -300e9:
            sg_score = 100
        elif net_gex < 0:
            sg_score = min(100, abs(net_gex) / 300e9 * 100)
        else:
            sg_score = 0  # long gamma = stabilizing = no explosion risk

        # 3. IV-RV divergence: IV < RV = cheap gamma = explosion likely
        ratio = iv_rv['iv_rv_ratio']
        if ratio < 0.70:
            ivr_score = 100
        elif ratio < 0.80:
            ivr_score = 75
        elif ratio < 0.90:
            ivr_score = 50
        elif ratio < 1.00:
            ivr_score = 25
        elif ratio < 1.15:
            ivr_score = 10
        else:
            ivr_score = 0  # overpriced = low explosion risk

        # Boost if RV is accelerating (5d >> 20d)
        if iv_rv['rv_acceleration'] > 1.3:
            ivr_score = min(100, ivr_score * 1.3)

        # 4. OI surge score: >2% change = high
        surge_pct = abs(oi_surge['max_surge_pct'])
        oi_score = min(100, surge_pct * 20)  # 5% surge → 100

        composite = (
            conc_score * 0.30 +
            sg_score   * 0.25 +
            ivr_score  * 0.25 +
            oi_score   * 0.20
        )

        if composite >= 70:
            status = "🔥 EXPLOSION IMMINENT"
            alert  = "CRITICAL"
        elif composite >= 50:
            status = "⚡ ELEVATED RISK"
            alert  = "HIGH"
        elif composite >= 30:
            status = "⚠  BUILDING"
            alert  = "MEDIUM"
        else:
            status = "✓  NORMAL"
            alert  = "LOW"

        return {
            'composite': round(composite, 1),
            'status':    status,
            'alert':     alert,
            'breakdown': {
                'concentration': round(conc_score, 1),
                'short_gamma':   round(sg_score,   1),
                'iv_rv_cheap':   round(ivr_score,  1),
                'oi_surge':      round(oi_score,   1)
            }
        }

    # ──────────────────────────────────────────────────────────────────────────
    # DIRECTION BIAS
    # ──────────────────────────────────────────────────────────────────────────

    def _explosion_direction(self, oi_surge, skew_vel, net_gex, spot, flip_point):
        """Estimate likely direction of the explosion."""
        score = 0.0

        # OI bias: positive = call-heavy = upside
        score += oi_surge['bias'] * 30

        # Skew velocity: negative = put skew falling = upside
        score -= skew_vel['velocity'] * 5

        # GEX flip point: if spot < flip → squeeze UP likely; if spot > flip → squeeze DOWN
        if flip_point > 0:
            if spot < flip_point:
                score += 15
            else:
                score -= 15

        if score > 5:
            direction = "UPSIDE"
        elif score < -5:
            direction = "DOWNSIDE"
        else:
            direction = "UNCLEAR"

        return direction, round(score, 1)

    # ──────────────────────────────────────────────────────────────────────────
    # PRINT HELPERS
    # ──────────────────────────────────────────────────────────────────────────

    def _print_gex_bar(self, profile, spot):
        """Print a horizontal bar chart of GEX by strike (±5 strikes)."""
        if not profile:
            return

        sorted_strikes = sorted(profile.keys())
        atm_idx = min(range(len(sorted_strikes)),
                      key=lambda i: abs(sorted_strikes[i] - spot))
        lo = max(0, atm_idx - 7)
        hi = min(len(sorted_strikes), atm_idx + 8)
        nearby = sorted_strikes[lo:hi]

        values = [profile[k] for k in nearby]
        max_abs = max(abs(v) for v in values) if values else 1

        BAR_W = 20
        print("\n  ┌─────── GEX PROFILE (near ATM strikes) ──────────────────────────┐")

        for strike in nearby:
            gex = profile[strike]
            label = f"{strike:>6.0f}"
            magnitude = int(abs(gex) / max_abs * BAR_W)
            sign = "▶ " if gex >= 0 else "◀ "

            if gex >= 0:
                bar = sign + "█" * magnitude + "░" * (BAR_W - magnitude)
                gex_str = f"+{gex/1e9:.2f}B"
            else:
                bar = sign + "█" * magnitude + "░" * (BAR_W - magnitude)
                gex_str = f"{gex/1e9:.2f}B"

            atm_marker = " ← ATM" if abs(strike - spot) < 30 else ""
            print(f"  │ {label}  {bar}  {gex_str:>8}{atm_marker}")

        print("  └──────────────────────────────────────────────────────────────────┘")

    def _print_score_breakdown(self, score_result):
        """Visual score breakdown bar."""
        bd = score_result['breakdown']
        print("\n  Score Breakdown (0 = no risk, 100 = max):")
        items = [
            ("Gamma Conc",   bd['concentration']),
            ("Short Gamma",  bd['short_gamma']),
            ("IV/RV Cheap",  bd['iv_rv_cheap']),
            ("OI Surge",     bd['oi_surge']),
        ]
        for label, val in items:
            filled = int(val / 5)
            bar = "█" * filled + "░" * (20 - filled)
            print(f"    {label:<14} {bar}  {val:.0f}")

    # ──────────────────────────────────────────────────────────────────────────
    # MAIN ENTRY
    # ──────────────────────────────────────────────────────────────────────────

    def analyze(self, df: pd.DataFrame, spot: float, T: float) -> dict:
        """Analyze chain data to compute composite GEX signals and return structured output."""
        profile, net_gex, flip_point, concentration, gex_acceleration = \
            self._compute_gex_profile(df, spot, T)

        oi_surge  = self._compute_oi_surge(df, spot)
        iv_rv     = self._compute_iv_rv(df, spot, T)
        skew_vel  = self._compute_skew_velocity(df, spot)
        score_res = self._compute_score(concentration, net_gex, iv_rv, oi_surge)
        direction, dir_score = self._explosion_direction(
            oi_surge, skew_vel, net_gex, spot, flip_point)

        # Trigger zone
        if direction == "UPSIDE" and flip_point > spot:
            trigger = f"Break above {flip_point:.0f} → cascade UP"
        elif direction == "DOWNSIDE" and flip_point > 0 and flip_point < spot:
            trigger = f"Break below {flip_point:.0f} → cascade DOWN"
        elif flip_point > 0:
            trigger = f"GEX flip at {flip_point:.0f}"
        else:
            trigger = "No clear flip point"

        profile_list = [{"strike": float(k), "gex": float(v)} for k, v in profile.items()]

        return {
            "composite_score": score_res['composite'],
            "status": score_res['status'],
            "alert": score_res['alert'],
            "net_gex": net_gex,
            "gex_flip_point": flip_point,
            "concentration": concentration,
            "gex_acceleration": gex_acceleration,
            "direction": direction,
            "dir_score": dir_score,
            "trigger": trigger,
            "iv_rv": iv_rv,
            "oi_surge": oi_surge,
            "skew_vel": skew_vel,
            "profile": profile_list,
            "timestamp": datetime.now().isoformat(),
            "score_breakdown": score_res.get('breakdown', {})
        }

    def run(self):
        print("\n" + "═" * 70)
        print("  GAMMA EXPLOSION MONITOR")
        print("═" * 70)

        # ── Context from previous signals ──────────────────────────
        if self.memory is not None:
            brief = self.memory.get_brief_text()
            if brief:
                print(brief)

        # Setup
        print("\nFetching Spot Price...")
        self._get_spot()
        spot = self.spot_price
        if spot <= 0:
            print("Error: Could not fetch spot."); return
        print(f"Spot: {spot:,.2f}")

        exp = input("Enter Expiry (YYYY-MM-DD): ").strip()
        if not exp:
            if not self.fyers:
                print("Error: No Fyers instance.")
                return
            try:
                probe = self.fyers.optionchain(data={
                    "symbol": self.symbol, "strikecount": 1, "timestamp": ""})
                expiry_list = []
                if 'data' in probe and 'expiryData' in probe.get('data', {}):
                    expiry_list = probe['data']['expiryData']
                elif probe.get('s') == 'error' and 'expiryData' in probe.get('data', {}):
                    expiry_list = probe['data']['expiryData']
                if expiry_list:
                    print("\n  Available Expiries:")
                    for idx, item in enumerate(expiry_list[:8], 1):
                        print(f"    {idx}. {item['date']}")
                    pick = input("  Select (1-8): ").strip()
                    if pick.isdigit() and 1 <= int(pick) <= len(expiry_list[:8]):
                        selected = expiry_list[int(pick) - 1]
                        exp = datetime.strptime(selected['date'], "%d-%m-%Y").strftime("%Y-%m-%d")
                    else:
                        print("Invalid selection."); return
            except Exception as e:
                print(f"Could not fetch expiries: {e}"); return

        self.expiry_date = exp
        T = self.analytics.get_time_to_expiry(exp)
        if T <= 0:
            print("Error: Expiry in the past."); return
        DTE = max(1, int(T * 365))
        print(f"DTE: {DTE} days")

        # Fetch RV baseline once
        self._fetch_rv_baseline()

        print("\nStarting Gamma Explosion Monitor... (Ctrl+C to Stop)")
        print("First cycle: OI Surge will show N/A (no previous snapshot)\n")

        try:
            cycle = 0
            while True:
                cycle += 1
                self._get_spot()
                spot = self.spot_price

                # Refresh chain
                df = self._get_chain()
                if df.empty:
                    print("No chain data. Retrying in 10s...")
                    time.sleep(10)
                    continue

                # Recompute T (DTE shrinks)
                T = self.analytics.get_time_to_expiry(exp)
                if T < 0.001: T = 0.001

                # ── Compute all signals ────────────────────────────────────
                res = self.analyze(df, spot, T)

                net_gex = res['net_gex']
                flip_point = res['gex_flip_point']
                concentration = res['concentration']
                gex_acceleration = res['gex_acceleration']
                direction = res['direction']
                dir_score = res['dir_score']
                trigger = res['trigger']
                iv_rv = res['iv_rv']
                oi_surge = res['oi_surge']
                skew_vel = res['skew_vel']
                profile = {p['strike']: p['gex'] for p in res['profile']}
                
                score_res = {
                    'composite': res['composite_score'],
                    'status': res['status'],
                    'alert': res['alert'],
                    'breakdown': res.get('score_breakdown', {})
                }

                # ── Print Dashboard ────────────────────────────────────────
                now = datetime.now().strftime('%H:%M:%S')
                print("\n" + "═" * 70)
                print(f"  GAMMA EXPLOSION MONITOR  |  {now}  |  Spot: {spot:,.2f}")
                print("═" * 70)

                gex_sign = "SHORT GAMMA ⚡ (Destabilizing)" if net_gex < 0 else "LONG GAMMA ✓ (Stabilizing)"
                print(f"\n  Net GEX:          {net_gex:>+20,.0f}")
                print(f"  GEX Regime:       {gex_sign}")
                
                accel_prefix = "Expanding Long" if gex_acceleration > 0 else "Expanding Short" if gex_acceleration < 0 else "Stable"
                print(f"  GEX Acceleration: {gex_acceleration:>+20,.0f} (15m: {accel_prefix})")
                
                print(f"  GEX Flip Point:   {flip_point:.0f}" if flip_point else "  GEX Flip Point:   Not found")
                print(f"  ATM Concentration:{concentration:>6.1f}%  ({'⚠ HIGH' if concentration > 40 else 'normal'})")

                # IV-RV section
                ir = iv_rv
                print(f"\n  ── IV-RV Analysis ──────────────────────────────────────")
                print(f"  ATM IV:           {ir['atm_iv']:.2f}%")
                print(f"  5d RV:            {ir['rv_5d']:.2f}%   (RV accel: {ir['rv_acceleration']:.2f}×)")
                print(f"  20d RV:           {ir['rv_20d']:.2f}%")
                print(f"  Consensus RV:     {ir['consensus_rv']:.2f}%")

                ratio_label = ""
                if ir['iv_rv_ratio'] < 0.80:
                    ratio_label = " ← CHEAP GAMMA ⚠"
                elif ir['iv_rv_ratio'] > 1.20:
                    ratio_label = " ← EXPENSIVE GAMMA"
                print(f"  IV/RV Ratio:      {ir['iv_rv_ratio']:.3f}{ratio_label}")
                print(f"  VRP (IV - RV):    {ir['vrp']:+.2f}%")

                # OI Surge section
                oi = oi_surge
                print(f"\n  ── OI Surge (last 30s) ─────────────────────────────────")
                if oi['first_cycle']:
                    print("  First cycle — establishing baseline...")
                else:
                    print(f"  Call OI Δ:        {oi['call_build']:>+10,.0f}  (ATM strikes)")
                    print(f"  Put  OI Δ:        {oi['put_build']:>+10,.0f}")
                    print(f"  Max Strike Δ:     {oi['max_surge_pct']:>+6.1f}%  (at {oi['surge_strike']:.0f})")
                    bias_str = (f"+{oi['bias']:.2f} CALL-HEAVY → bullish accumulation" if oi['bias'] > 0.1
                                else f"{oi['bias']:.2f} PUT-HEAVY → bearish accumulation" if oi['bias'] < -0.1
                                else f"{oi['bias']:.2f} BALANCED")
                    print(f"  Bias:             {bias_str}")

                # Skew section
                sv = skew_vel
                skew_bias = ""
                if sv['velocity'] > 0.3:
                    skew_bias = " → puts getting pricier → DOWNSIDE pressure"
                elif sv['velocity'] < -0.3:
                    skew_bias = " → puts cheapening → UPSIDE pressure"
                print(f"\n  Skew:             {sv['skew']:+.2f}  (velocity: {sv['velocity']:+.3f}{skew_bias})")

                # GEX Profile Bar
                self._print_gex_bar(profile, spot)

                # Score
                print(f"\n  ┌─── EXPLOSION SCORE ──────────────────────────────────┐")
                print(f"  │  Score : {score_res['composite']:>5.1f} / 100                               │")
                print(f"  │  Status: {score_res['status']:<45}│")
                print(f"  │  Direction: {direction:<20}  (bias score: {dir_score:+.1f})  │")
                print(f"  │  Trigger: {trigger:<50}│")
                print(f"  └──────────────────────────────────────────────────────┘")

                self._print_score_breakdown(score_res)

                # ── Alert dispatch ──────────────────────────────────────────────
                try:
                    from AlertDispatcher import fire as _ad_fire
                    if score_res['composite'] >= 70:
                        _ad_fire(
                            "GammaExplosionModel", "CRITICAL",
                            f"Gamma Explosion Imminent — score {score_res['composite']:.0f}",
                            f"Direction: {direction} | Trigger: {trigger}",
                            data={"score": score_res['composite'], "direction": direction}
                        )
                    elif score_res['composite'] >= 50:
                        _ad_fire(
                            "GammaExplosionModel", "WARNING",
                            f"Elevated explosion risk — score {score_res['composite']:.0f}",
                            f"Direction: {direction}",
                            data={"score": score_res['composite']}
                        )
                except Exception:
                    pass

                # ── Persist to SignalMemory ─────────────────────────────────
                if self.memory is not None:
                    try:
                        gex_label = "SHORT GAMMA" if net_gex < 0 else "LONG GAMMA"
                        self.memory.update_context({
                            'gex_regime':         gex_label,
                            'explosion_score':    score_res['composite'],
                            'explosion_direction':direction,
                            'atm_iv':             iv_rv['atm_iv'],
                            'rv_5d':              iv_rv['rv_5d'],
                            'rv_20d':             iv_rv['rv_20d'],
                            'consensus_rv':       iv_rv['consensus_rv'],
                            'vrp':                iv_rv['vrp'],
                        }, spot=spot)
                        # Only log a new signal entry when score crosses ELEVATED
                        if score_res['composite'] >= 50 and cycle == 1:
                            self.memory.log_signal('GammaExplosionModel', {
                                'direction':          direction,
                                'explosion_direction':direction,
                                'action':             f"Monitor {trigger}",
                                'score':              score_res['composite'],
                                'gex_regime':         gex_label,
                                'flip_point':         float(flip_point),
                                'atm_iv':             iv_rv['atm_iv'],
                                'vrp':                iv_rv['vrp'],
                                'alert':              score_res['alert'],
                            }, spot=spot, expiry=exp)
                    except Exception:
                        pass

                # Actionable summary
                print(f"\n  ─── What to Do ──────────────────────────────────────────")
                s = score_res['composite']
                if s >= 70:
                    print(f"  ACTION: Consider BUYING options — {direction} breakout imminent")
                    print(f"  Hedge: Wide straddle / directional debit spread")
                elif s >= 50:
                    print(f"  ACTION: Watch for breakout trigger — {trigger}")
                    print(f"  Prepare entry, wait for volume confirmation")
                elif s >= 30:
                    print(f"  ACTION: Monitor — conditions building but not yet critical")
                else:
                    print(f"  ACTION: No trade — gamma structure stable")

                print(f"\n  Waiting 30s for next update... [Cycle {cycle}] (Ctrl+C to Exit)")
                time.sleep(30)

        except KeyboardInterrupt:
            print("\nExiting Gamma Explosion Monitor...")

if __name__ == "__main__":
    gm = GammaExplosionModel()
    gm.run()
