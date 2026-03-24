"""
SignalMemory.py — Shared Signal Memory & Context Layer
=======================================================
Persists all trading signals to market_memory.json so every
module can see what other modules have recommended, detect
clashes, and track whether those predictions came true.

Usage:
    from SignalMemory import SignalMemory
    mem = SignalMemory()

    # Log a recommendation
    sig_id = mem.log_signal("OptionBuyerAdvisor", {
        "direction": "BULLISH",
        "action": "BUY 25700 CE",
        "entry": 88.50,
        "sl_premium": 44.25,
        "sl_spot": 25415,
        "t1_spot": 25600,
        "t2_spot": 25745,
        "score": 73.0,
        "atm_iv": 11.9,
        "consensus_rv": 15.2,
        "vrp": -3.3,
    }, spot=25466, expiry="2026-02-27")

    # Auto-check outcomes
    mem.check_signal_outcomes(current_spot)

    # Show full brief in console
    mem.print_brief(current_spot)

    # Update rolling market context (called by each module)
    mem.update_context({"regime": "UNDERPRICED", "vrp": -3.3, ...})
"""

import os
import json
import uuid
from datetime import datetime
from typing import Optional


MEMORY_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "market_memory.json")
MAX_HISTORY = 50   # Keep last 50 resolved signals


class SignalMemory:

    # ─────────────────────────────────────────────────────────────────
    # INIT / LOAD / SAVE
    # ─────────────────────────────────────────────────────────────────

    def __init__(self, memory_file: str = MEMORY_FILE):
        self.memory_file = memory_file
        self._data = self._load()

    def _load(self) -> dict:
        if os.path.exists(self.memory_file):
            try:
                with open(self.memory_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception:
                pass
        return self._empty_store()

    def _save(self):
        try:
            with open(self.memory_file, 'w', encoding='utf-8') as f:
                json.dump(self._data, f, indent=2, default=str)
        except Exception as e:
            print(f"  [SignalMemory] Save error: {e}")

    @staticmethod
    def _empty_store() -> dict:
        return {
            "schema_version": 2,
            "active_signals": [],
            "resolved_signals": [],
            "context": {
                "last_updated": None,
                "spot": 0,
                "regime": None,
                "vrp": None,
                "gex_regime": None,
                "explosion_score": None,
                "explosion_direction": None,
                "atm_iv": None,
                "rv_5d": None,
                "rv_20d": None,
                "consensus_rv": None,
            }
        }

    # ─────────────────────────────────────────────────────────────────
    # SIGNAL LOGGING
    # ─────────────────────────────────────────────────────────────────

    def log_signal(self, source: str, signal: dict,
                   spot: float = 0, expiry: str = "") -> str:
        """
        Persist a new signal recommendation.
        Returns the signal_id (UUID) for later resolution.
        """
        sig_id = str(uuid.uuid4())[:8]
        entry = {
            "id":          sig_id,
            "timestamp":   datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "source":      source,
            "spot_at_log": spot,
            "expiry":      expiry,
            "signal":      signal,
            "outcome": {
                "resolved":       False,
                "spot_at_exit":   None,
                "hit_t1":         False,
                "hit_t2":         False,
                "hit_sl":         False,
                "pnl_pts":        None,
                "resolved_at":    None
            }
        }
        self._data["active_signals"].append(entry)
        self._save()
        return sig_id

    def resolve_signal(self, sig_id: str, outcome: dict):
        """Mark a signal as resolved with its outcome."""
        for sig in self._data["active_signals"]:
            if sig["id"] == sig_id:
                sig["outcome"].update(outcome)
                sig["outcome"]["resolved"]    = True
                sig["outcome"]["resolved_at"] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                self._data["resolved_signals"].append(sig)
                self._data["active_signals"].remove(sig)
                # Trim history
                if len(self._data["resolved_signals"]) > MAX_HISTORY:
                    self._data["resolved_signals"] = self._data["resolved_signals"][-MAX_HISTORY:]
                self._save()
                return

    def update_signal(self, sig_id: str, updates: dict):
        """Update an existing active signal in-place (for dedup)."""
        for sig in self._data["active_signals"]:
            if sig["id"] == sig_id:
                sig["signal"].update(updates)
                sig["timestamp"] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                self._save()
                return True
        return False

    def get_last_signal_by_source(self, source: str):
        """Return the most recent active signal from a given source, or None."""
        for sig in reversed(self._data["active_signals"]):
            if sig["source"] == source:
                return sig
        return None

    def get_synthesized_recommendation(self, current_spot: float = 0) -> dict:
        """
        Synthesize all active signals into one coherent recommendation.
        Returns: {bias, confidence, play, risk, details}
        """
        active = self._data["active_signals"]
        ctx    = self.get_context()

        if not active and not ctx.get("regime"):
            return {
                'bias': 'NO DATA',
                'confidence': 'LOW',
                'play': 'Run an analysis module first to generate signals',
                'risk': '─',
                'details': []
            }

        # Count directional votes
        DIRECTION_MAP = {
            "BULLISH": 1, "UPSIDE": 1,
            "BEARISH": -1, "DOWNSIDE": -1,
            "NEUTRAL": 0, "UNCLEAR": 0,
        }
        votes = []
        details = []
        sl_levels = []

        for sig in active:
            s = sig["signal"]
            d_raw = (s.get("direction") or s.get("explosion_direction") or "").upper()
            vote = DIRECTION_MAP.get(d_raw, 0)
            votes.append(vote)
            details.append(f"{sig['source']}: {s.get('action', d_raw)}")

            if s.get("sl_spot"):
                sl_levels.append(float(s["sl_spot"]))

        # Context regime vote
        regime = (ctx.get("regime") or "").upper()
        if regime in ("UNDERPRICED", "SQUEEZE"):
            votes.append(0.5)  # mild buy-vol bias
            details.append(f"Regime: {regime} → BUY VOL edge")
        elif regime in ("OVERPRICED",):
            votes.append(-0.3)
            details.append(f"Regime: {regime} → SELL VOL edge")

        # GEX regime vote
        gex = (ctx.get("gex_regime") or "").upper()
        expl_score = ctx.get("explosion_score") or 0
        if "SHORT" in gex and expl_score >= 50:
            expl_dir = (ctx.get("explosion_direction") or "").upper()
            v = 0.5 if expl_dir == "UPSIDE" else -0.5 if expl_dir == "DOWNSIDE" else 0
            votes.append(v)
            details.append(f"GEX: {gex}, explosion {expl_score}/100 → {expl_dir}")
            
        # VWAP trend vote (added from RealizedVolEngine)
        vwap_dist = ctx.get("vwap_dist")
        if vwap_dist is not None:
            v_vote = 0.4 if vwap_dist > 0 else -0.4
            votes.append(v_vote)
            dir_txt = "BULLISH" if vwap_dist > 0 else "BEARISH"
            details.append(f"Intraday VWAP: Spot is {abs(vwap_dist):.1f} pts {'above' if vwap_dist > 0 else 'below'} ({dir_txt})")

        total = sum(votes)
        n = len(votes) or 1
        
        # Check alignment across technical (VWAP) and Gamma (GEX)
        is_bull_aligned = (total > 1.0 and (vwap_dist is None or vwap_dist > 0) and ("LONG" in gex or ("SHORT" in gex and expl_dir == "UPSIDE")))
        is_bear_aligned = (total < -1.0 and (vwap_dist is None or vwap_dist < 0) and ("SHORT" in gex and expl_dir == "DOWNSIDE"))

        if is_bull_aligned:
            bias = "STRONG BULLISH"
            confidence = "HIGH"
        elif is_bear_aligned:
            bias = "STRONG BEARISH"
            confidence = "HIGH"
        elif total > 0.3:
            bias = "BULLISH"
            confidence = "MEDIUM" if abs(total)/n > 0.3 else "LOW"
        elif total < -0.3:
            bias = "BEARISH"
            confidence = "MEDIUM" if abs(total)/n > 0.3 else "LOW"
        elif clashes:
            bias = "TURBULENCE"
            confidence = "LOW"
        else:
            bias = "NEUTRAL"
            confidence = "LOW"

        # Build play suggestion
        if "BULLISH" in bias:
            play = "BUY CE directional or debit call spread"
            if "STRONG" in bias: play = "HIGH CONVICTION: " + play
            if sl_levels:
                play += f" — SL at {min(sl_levels):.0f}"
        elif "BEARISH" in bias:
            play = "BUY PE directional or debit put spread"
            if "STRONG" in bias: play = "HIGH CONVICTION: " + play
            if sl_levels:
                play += f" — SL at {max(sl_levels):.0f}"
        elif bias == "TURBULENCE":
            play = "Conflicting signals detected. Do not deploy directional trades."
        else:
            play = "Wait for clearer signal alignment or sell premium (iron condor)"

        # Risk
        clashes = self.detect_clashes()
        if clashes:
            risk = f"⚠ {len(clashes)} signal clash(es) — reduce size"
        elif confidence == "HIGH":
            risk = "Controlled — signals aligned"
        else:
            risk = "Moderate — mixed signals"

        return {
            'bias':       bias,
            'confidence': confidence,
            'play':       play,
            'risk':       risk,
            'details':    details,
            'vote_total': round(total, 2),
            'vote_count': n,
        }

    def check_signal_outcomes(self, current_spot: float):
        """
        Auto-resolve active signals whose SL or T1/T2 levels
        have been crossed by the current spot price.
        Called by each module whenever spot is refreshed.
        """
        to_resolve = []
        for sig in self._data["active_signals"]:
            s = sig["signal"]
            spot_at_log = sig.get("spot_at_log", 0)
            direction   = s.get("direction", "").upper()

            sl_spot = s.get("sl_spot")
            t1_spot = s.get("t1_spot")
            t2_spot = s.get("t2_spot")

            outcome = {}

            if direction == "BULLISH":
                if sl_spot and current_spot <= sl_spot:
                    outcome = {
                        "hit_sl": True,
                        "spot_at_exit": current_spot,
                        "pnl_pts": round(current_spot - spot_at_log, 1)
                    }
                elif t2_spot and current_spot >= t2_spot:
                    outcome = {
                        "hit_t2": True, "hit_t1": True,
                        "spot_at_exit": current_spot,
                        "pnl_pts": round(current_spot - spot_at_log, 1)
                    }
                elif t1_spot and current_spot >= t1_spot:
                    outcome = {
                        "hit_t1": True,
                        "spot_at_exit": current_spot,
                        "pnl_pts": round(current_spot - spot_at_log, 1)
                    }

            elif direction == "BEARISH":
                if sl_spot and current_spot >= sl_spot:
                    outcome = {
                        "hit_sl": True,
                        "spot_at_exit": current_spot,
                        "pnl_pts": round(spot_at_log - current_spot, 1)
                    }
                elif t2_spot and current_spot <= t2_spot:
                    outcome = {
                        "hit_t2": True, "hit_t1": True,
                        "spot_at_exit": current_spot,
                        "pnl_pts": round(spot_at_log - current_spot, 1)
                    }
                elif t1_spot and current_spot <= t1_spot:
                    outcome = {
                        "hit_t1": True,
                        "spot_at_exit": current_spot,
                        "pnl_pts": round(spot_at_log - current_spot, 1)
                    }

            if outcome:
                to_resolve.append((sig["id"], outcome))

        for sig_id, outcome in to_resolve:
            self.resolve_signal(sig_id, outcome)

    # ─────────────────────────────────────────────────────────────────
    # CONTEXT UPDATE
    # ─────────────────────────────────────────────────────────────────

    def update_context(self, updates: dict, spot: float = 0):
        """
        Update the rolling market context dict.
        Called by each module after computing regime/VRP/GEX etc.
        """
        ctx = self._data["context"]
        ctx.update(updates)
        ctx["last_updated"] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        if spot > 0:
            ctx["spot"] = spot
        self._save()

    def get_context(self) -> dict:
        return self._data.get("context", {})

    # ─────────────────────────────────────────────────────────────────
    # CLASH DETECTION
    # ─────────────────────────────────────────────────────────────────

    def detect_clashes(self) -> list:
        """
        Find pairs of active signals with contradictory directions.
        Returns list of clash dicts: {sig_a, sig_b, reason, severity}
        """
        active = self._data["active_signals"]
        clashes = []

        for i in range(len(active)):
            for j in range(i + 1, len(active)):
                a = active[i]
                b = active[j]
                d_a = a["signal"].get("direction", "").upper()
                d_b = b["signal"].get("direction", "").upper()

                # Direct contradiction
                if ((d_a == "BULLISH" and d_b == "BEARISH") or
                        (d_a == "BEARISH" and d_b == "BULLISH")):
                    clashes.append({
                        "sig_a":    a,
                        "sig_b":    b,
                        "reason":   f"{a['source']} says {d_a} ↔ {b['source']} says {d_b}",
                        "severity": "HIGH",
                        "type":     "DIRECTION_CLASH"
                    })

                # Gamma regime vs buyer direction clash
                a_is_buyer = a["source"] == "OptionBuyerAdvisor"
                b_is_buyer = b["source"] == "OptionBuyerAdvisor"
                a_is_gamma = a["source"] == "GammaExplosionModel"
                b_is_gamma = b["source"] == "GammaExplosionModel"

                if a_is_gamma or b_is_gamma:
                    gamma_sig  = a if a_is_gamma else b
                    other_sig  = b if a_is_gamma else a
                    expl_dir   = gamma_sig["signal"].get("explosion_direction", "")
                    other_dir  = other_sig["signal"].get("direction", "").upper()

                    if expl_dir == "UPSIDE" and other_dir == "BEARISH":
                        clashes.append({
                            "sig_a":    gamma_sig,
                            "sig_b":    other_sig,
                            "reason":   f"GEX says UPSIDE explosion ↔ {other_sig['source']} says BEARISH",
                            "severity": "MEDIUM",
                            "type":     "GEX_DIRECTION_CLASH"
                        })
                    elif expl_dir == "DOWNSIDE" and other_dir == "BULLISH":
                        clashes.append({
                            "sig_a":    gamma_sig,
                            "sig_b":    other_sig,
                            "reason":   f"GEX says DOWNSIDE explosion ↔ {other_sig['source']} says BULLISH",
                            "severity": "MEDIUM",
                            "type":     "GEX_DIRECTION_CLASH"
                        })

        return clashes

    def detect_alignments(self) -> list:
        """
        Find pairs of active signals that point the same direction.
        Returns list of alignment dicts.
        """
        active = self._data["active_signals"]
        alignments = []

        DIRECTION_MAP = {
            "UPSIDE": "BULLISH", "DOWNSIDE": "BEARISH",
            "BULLISH": "BULLISH", "BEARISH":  "BEARISH"
        }

        for i in range(len(active)):
            for j in range(i + 1, len(active)):
                a = active[i]
                b = active[j]

                d_a_raw = a["signal"].get("direction") or \
                          a["signal"].get("explosion_direction", "")
                d_b_raw = b["signal"].get("direction") or \
                          b["signal"].get("explosion_direction", "")

                d_a = DIRECTION_MAP.get(d_a_raw.upper(), "")
                d_b = DIRECTION_MAP.get(d_b_raw.upper(), "")

                if d_a and d_b and d_a == d_b:
                    alignments.append({
                        "sig_a":  a,
                        "sig_b":  b,
                        "reason": f"{a['source']} + {b['source']} both say {d_a}",
                        "type":   "ALIGNMENT"
                    })

        return alignments

    # ─────────────────────────────────────────────────────────────────
    # STARTUP BRIEF  (one-line text for module startup)
    # ─────────────────────────────────────────────────────────────────

    def get_brief_text(self, current_spot: float = 0) -> str:
        """
        One-paragraph context brief printed at startup of each module.
        """
        ctx = self.get_context()
        active = self._data["active_signals"]
        clashes = self.detect_clashes()
        alignments = self.detect_alignments()

        lines = ["  ── Signal Memory Context ─────────────────────────────────"]

        if ctx.get("last_updated"):
            lines.append(f"  Last update: {ctx['last_updated']}  |  Spot: {ctx.get('spot', '─')}")

        if ctx.get("regime"):
            lines.append(f"  Regime: {ctx['regime']:15s}  VRP: {ctx.get('vrp', '─'):+.2f}%  "
                         f"IV: {ctx.get('atm_iv', '─')}")

        if ctx.get("gex_regime"):
            expl = ctx.get("explosion_score")
            lines.append(f"  GEX: {ctx['gex_regime']:25s}  "
                         f"Explosion Score: {expl if expl else '─'}")

        if active:
            lines.append(f"\n  Active signals: {len(active)}")
            for sig in active[-3:]:   # Show last 3
                s = sig["signal"]
                spot_at_log = sig.get("spot_at_log", 0)
                delta_str = ""
                if current_spot > 0 and spot_at_log > 0:
                    delta = current_spot - spot_at_log
                    delta_str = f"  (spot Δ: {delta:+.0f} since signal)"
                lines.append(
                    f"  [{sig['id']}] {sig['timestamp'][11:16]}  "
                    f"{sig['source']}: {s.get('action', s.get('direction','?'))}"
                    f"{delta_str}"
                )
        else:
            lines.append("  No active signals in memory.")

        if clashes:
            lines.append(f"\n  ⚠ SIGNAL CLASHES ({len(clashes)}):")
            for c in clashes:
                lines.append(f"    [{c['severity']}] {c['reason']}")
        elif alignments:
            lines.append(f"\n  ✓ SIGNALS ALIGNED ({len(alignments)}):")
            for al in alignments[:2]:
                lines.append(f"    ✓ {al['reason']}")

        lines.append("  " + "─" * 60)
        return "\n".join(lines)

    # ─────────────────────────────────────────────────────────────────
    # FULL CONSOLE BRIEFING
    # ─────────────────────────────────────────────────────────────────

    def print_brief(self, current_spot: float = 0):
        """Full rich console briefing — called by menu option 8."""
        self.check_signal_outcomes(current_spot)
        ctx     = self.get_context()
        active  = self._data["active_signals"]
        resolved = self._data["resolved_signals"]
        clashes  = self.detect_clashes()
        aligns   = self.detect_alignments()

        now = datetime.now().strftime('%Y-%m-%d %H:%M')
        print("\n" + "═" * 60)
        print(f"  SIGNAL MEMORY BRIEFING  |  {now}")
        print("═" * 60)

        # ── Context ────────────────────────────────────────────────
        print("\n  Current Market Context:")
        print(f"  {'─' * 56}")
        if ctx.get("last_updated"):
            print(f"  Last Updated   : {ctx['last_updated']}")
            print(f"  Spot (context) : {ctx.get('spot', '─')}")
            print(f"  Now Spot       : {current_spot:,.2f}" if current_spot else "")
        if ctx.get("regime"):
            vrp = ctx.get("vrp")
            vrp_str = f"{vrp:+.2f}%" if vrp is not None else "─"
            print(f"  Regime         : {ctx['regime']}")
            print(f"  VRP            : {vrp_str}")
            iv = ctx.get("atm_iv")
            rv = ctx.get("consensus_rv")
            print(f"  ATM IV         : {f'{iv:.2f}%' if iv else '─':10s}  "
                  f"Consensus RV: {f'{rv:.2f}%' if rv else '─'}")
        if ctx.get("gex_regime"):
            expl = ctx.get("explosion_score")
            expl_dir = ctx.get("explosion_direction", "─")
            print(f"  GEX Regime     : {ctx['gex_regime']}")
            print(f"  Explosion Score: {expl if expl else '─'}    "
                  f"Direction: {expl_dir}")

        # ── Active Signals ─────────────────────────────────────────
        print(f"\n  Active Signals ({len(active)}):")
        print(f"  {'─' * 56}")

        if not active:
            print("  No active signals.")
        else:
            for sig in active:
                s = sig["signal"]
                spot_at_log  = sig.get("spot_at_log", 0)
                delta_spot   = current_spot - spot_at_log if current_spot and spot_at_log else 0
                direction    = s.get("direction", "─").upper()
                pnl_indicator = ""
                if direction == "BULLISH" and delta_spot != 0:
                    pnl_indicator = f"  ({'▲' if delta_spot > 0 else '▼'} {abs(delta_spot):.0f} pts)"
                elif direction == "BEARISH" and delta_spot != 0:
                    pnl_indicator = f"  ({'▲' if delta_spot < 0 else '▼'} {abs(delta_spot):.0f} pts {'toward' if direction == 'BEARISH' and delta_spot < 0 else 'away'})"

                print(f"\n  ┌── [{sig['id']}] {sig['timestamp']}  ─────────────────────")
                print(f"  │  Source   : {sig['source']}")
                print(f"  │  Action   : {s.get('action', s.get('direction', '─'))}")
                if s.get("entry"):
                    print(f"  │  Entry    : {s['entry']:.2f}  "
                          f"SL: {s.get('sl_premium', '─')}  "
                          f"T1: {s.get('t1_prem', '─')}  T2: {s.get('t2_prem', '─')}")
                if s.get("sl_spot"):
                    print(f"  │  Spot SL  : {s['sl_spot']:.0f}  "
                          f"T1 Spot: {s.get('t1_spot', '─')}  "
                          f"T2 Spot: {s.get('t2_spot', '─')}")
                print(f"  │  Spot @ Log: {spot_at_log:.0f}  Now: {current_spot:.0f}{pnl_indicator}")
                if s.get("score"):
                    print(f"  │  Score    : {s['score']:.1f}")
                print(f"  └{'─' * 54}")

        # ── Clash / Alignment ──────────────────────────────────────
        if clashes:
            print(f"\n  ⚠ SIGNAL CLASHES DETECTED ({len(clashes)}):")
            print(f"  {'─' * 56}")
            for c in clashes:
                print(f"  [{c['severity']}] {c['reason']}")
                print(f"          → Consider reducing position size or waiting for resolution")
        elif active and aligns:
            print(f"\n  ✓ SIGNAL ALIGNMENT ({len(aligns)}):")
            print(f"  {'─' * 56}")
            for al in aligns:
                print(f"  ✓ {al['reason']} → Confidence BOOSTED")

        # ── Recent History ─────────────────────────────────────────
        recent = resolved[-7:] if resolved else []
        if recent:
            print(f"\n  Signal History (last {len(recent)}):")
            print(f"  {'─' * 56}")
            print(f"  {'Time':<17} {'Source':<22} {'Result':<12} {'P&L'}")
            print(f"  {'─'*17}  {'─'*22}  {'─'*12}  {'─'*8}")
            for sig in reversed(recent):
                o = sig["outcome"]
                result = ("T2 HIT" if o.get("hit_t2")
                          else "T1 HIT" if o.get("hit_t1")
                          else "SL HIT" if o.get("hit_sl")
                          else "MANUAL")
                pnl = f"{o['pnl_pts']:+.1f}" if o.get("pnl_pts") is not None else "─"
                icon = "✓" if not o.get("hit_sl") else "✗"
                print(f"  {icon} {sig['timestamp'][5:16]:<16} {sig['source']:<22} "
                      f"{result:<12} {pnl}")

        print("\n" + "═" * 60)
