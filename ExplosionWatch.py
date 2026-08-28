"""
ExplosionWatch.py — Expiry-Day Gamma Squeeze Watch
====================================================
Monitors for a forming gamma-squeeze / dealer-hedging-cascade setup
in the last 120 minutes before NIFTY weekly expiry.

Conditions evaluated each 60s cycle:
  A. TIME PRESSURE   — minutes_to_expiry < 120 (arms the watch)
  B. PROXIMITY       — spot within 0.5% of GEX flip / OI wall, and closing
  C. ACCELERATION    — near-strike volume > 2x trailing average (leading tell)
  D. DIRECTION       — spot has moved toward the level in the same window
  E. CONCENTRATION   — target strike OI meaningfully above neighbours

Severity:
  >=4 conditions -> ALERT
  2-3 conditions -> WATCH
  <2 or no time  -> silent

Events broadcast on /stream websocket as {"type": "explosion_alert", ...}
and written to alerts.jsonl via AlertDispatcher (audit log).

De-duplication via _last_severity/_last_strike (in-process) +
SignalMemory context key "explosion_watch_state".
"""

from __future__ import annotations

import time
import threading
from collections import deque
from datetime import datetime
from typing import Deque, Optional


# ── tiny snapshot struct ─────────────────────────────────────────────────────
class _Snap:
    __slots__ = ("ts", "spot", "near_volume", "near_oi", "distance_pct")

    def __init__(self, ts: float, spot: float, near_volume: float,
                 near_oi: float, distance_pct: float):
        self.ts           = ts
        self.spot         = spot
        self.near_volume  = near_volume
        self.near_oi      = near_oi
        self.distance_pct = distance_pct


class ExplosionWatch:
    """
    Standalone watch-dog for expiry-day gamma squeezes.

    Usage (DataServer.py):
        watch = ExplosionWatch(
            broadcast_fn = hub.broadcast,
            gex_snapshot_ref = _gex_snapshot,   # the shared dict
            spot_fn  = lambda: hub.latest_data["spot"],
            chain_df_fn = lambda: _parse_chain_to_df(hub.latest_data.get("chain", {})),
            signal_memory = SignalMemory(),
        )
        watch.start()
    """

    # ── tunables ─────────────────────────────────────────────────────────────
    CYCLE_SECONDS        = 60
    RISK_WINDOW_MIN      = 120      # arm when minutes_to_expiry < this
    PROXIMITY_PCT        = 0.005    # 0.5% of spot
    VOLUME_ACCEL_MULT    = 2.0      # current / trailing avg
    BUFFER_SIZE          = 6        # snapshots kept (~6 min history)
    MIN_CONDITIONS_WATCH = 2
    MIN_CONDITIONS_ALERT = 4
    CONCENTRATION_MULT   = 1.5      # target OI vs neighbour mean

    def __init__(self, broadcast_fn, gex_snapshot_ref: dict,
                 spot_fn, chain_df_fn, signal_memory=None):
        self._broadcast   = broadcast_fn
        self._gex_snap    = gex_snapshot_ref
        self._spot_fn     = spot_fn
        self._chain_df_fn = chain_df_fn
        self._mem         = signal_memory
        self._stop        = threading.Event()

        self._buf: Deque[_Snap] = deque(maxlen=self.BUFFER_SIZE)

        # In-process dedup state
        self._last_severity: Optional[str]   = None
        self._last_strike:   Optional[float] = None
        self._last_fired_at: Optional[str]   = None

    # ── public ───────────────────────────────────────────────────────────────

    def start(self):
        threading.Thread(target=self._run, daemon=True,
                         name="ExplosionWatch").start()
        print("[ExplosionWatch] Background watch started (60s cycle).")

    def stop(self):
        self._stop.set()

    # ── main loop ────────────────────────────────────────────────────────────

    def _run(self):
        while not self._stop.is_set():
            try:
                self._cycle()
            except Exception as exc:
                print(f"[ExplosionWatch] Cycle error (non-fatal): {exc}")
            self._stop.wait(timeout=self.CYCLE_SECONDS)

    def _cycle(self):
        # ── time gate ────────────────────────────────────────────────────────
        minutes_left = self._minutes_to_expiry()
        if minutes_left is None or minutes_left > self.RISK_WINDOW_MIN:
            if self._last_severity is not None:
                self._resolve_and_clear()
            return

        spot = float(self._spot_fn() or 0)
        if spot <= 0:
            return

        # ── find target level ─────────────────────────────────────────────────
        flip_point = float(
            self._gex_snap.get("gex_flip_point", 0) or
            self._gex_snap.get("zero_gamma_level", 0) or 0
        )
        if flip_point <= 0:
            flip_point = self._best_oi_wall(spot)
        if flip_point <= 0:
            return

        direction   = "UPSIDE" if flip_point > spot else "DOWNSIDE"
        dist_pct    = abs(spot - flip_point) / spot

        # ── near-strike metrics ───────────────────────────────────────────────
        near_vol, near_oi = self._near_strike_metrics(spot)

        # ── push snapshot ─────────────────────────────────────────────────────
        self._buf.append(_Snap(time.time(), spot, near_vol, near_oi, dist_pct))

        # ── evaluate ─────────────────────────────────────────────────────────
        score, met = self._evaluate(spot, flip_point, dist_pct,
                                    near_vol, minutes_left)

        if score >= self.MIN_CONDITIONS_ALERT:
            severity = "ALERT"
        elif score >= self.MIN_CONDITIONS_WATCH:
            severity = "WATCH"
        else:
            self._resolve_and_clear()
            return

        # ── dedup ─────────────────────────────────────────────────────────────
        same_strike = (self._last_strike is not None and
                       abs(flip_point - self._last_strike) < 25)
        if same_strike and self._last_severity == severity:
            return   # no change → stay silent

        # ── fire ──────────────────────────────────────────────────────────────
        now_iso = datetime.now().isoformat(timespec="seconds")
        headline, detail, inv = self._make_text(
            spot, flip_point, direction, near_vol, dist_pct, minutes_left, met
        )

        event = {
            "type":         "explosion_alert",
            "severity":     severity,
            "strike":       flip_point,
            "direction":    direction,
            "headline":     headline,
            "detail":       detail,
            "minutes_left": round(minutes_left, 1),
            "invalidation": inv,
            "ts":           now_iso,
        }

        self._broadcast(event)
        self._write_audit(event)
        self._write_memory(flip_point, severity, now_iso)

        self._last_severity = severity
        self._last_strike   = flip_point
        self._last_fired_at = now_iso

    # ── condition evaluation ─────────────────────────────────────────────────

    def _evaluate(self, spot, flip, dist_pct, near_vol, minutes_left):
        met = {}

        # A: time gate (prerequisite, not a scored point)
        met["time"] = minutes_left < self.RISK_WINDOW_MIN

        # Loosen thresholds when very close to expiry
        late_session = minutes_left < 30
        prox_thr  = self.PROXIMITY_PCT * (1.5 if not late_session else 1.0)
        accel_thr = self.VOLUME_ACCEL_MULT * (0.7 if late_session else 1.0)

        # B: proximity AND closing in
        met["proximity"] = (dist_pct < prox_thr) and self._is_closing(spot, flip)

        # C: volume acceleration
        avg_vol = self._trailing_avg_volume()
        met["acceleration"] = bool(avg_vol > 0 and near_vol / avg_vol >= accel_thr)

        # D: directional move toward level
        met["direction"] = self._is_moving_toward(spot, flip)

        # E: OI concentration
        met["concentration"] = self._has_concentration(flip)

        score = sum(1 for k, v in met.items() if v and k != "time")
        if not met["time"]:
            score = 0

        return score, met

    # ── resolution ───────────────────────────────────────────────────────────

    def _resolve_and_clear(self):
        if self._last_severity is None:
            return
        event = {
            "type":         "explosion_alert",
            "severity":     "RESOLVED",
            "strike":       self._last_strike or 0,
            "direction":    "",
            "headline":     "Gamma squeeze setup resolved",
            "detail":       ("Volume and proximity conditions no longer met. "
                             "Setup has dissipated."),
            "minutes_left": 0,
            "invalidation": "",
            "ts":           datetime.now().isoformat(timespec="seconds"),
        }
        self._broadcast(event)
        self._write_audit(event)
        self._write_memory(0, "RESOLVED", event["ts"])
        self._last_severity = None
        self._last_strike   = None
        self._last_fired_at = None

    # ── buffer helpers ───────────────────────────────────────────────────────

    def _is_closing(self, spot: float, flip: float) -> bool:
        snaps = list(self._buf)
        if len(snaps) < 2:
            return False
        return abs(snaps[-1].spot - flip) < abs(snaps[-2].spot - flip)

    def _is_moving_toward(self, spot: float, flip: float) -> bool:
        snaps = list(self._buf)
        if len(snaps) < 2:
            return False
        old = snaps[0].spot
        return (flip > old and spot > old) or (flip < old and spot < old)

    def _trailing_avg_volume(self) -> float:
        snaps = list(self._buf)
        if len(snaps) < 2:
            return 0.0
        return sum(s.near_volume for s in snaps[:-1]) / (len(snaps) - 1)

    # ── chain helpers ────────────────────────────────────────────────────────

    def _near_strike_metrics(self, spot: float):
        try:
            df = self._chain_df_fn()
            if df is None or df.empty:
                return 0.0, 0.0
            band = spot * 0.01
            near = df[(df["strike"] - spot).abs() <= band]
            return float(near["volume"].sum()), float(near["oi"].sum())
        except Exception:
            return 0.0, 0.0

    def _has_concentration(self, flip: float) -> bool:
        try:
            df = self._chain_df_fn()
            if df is None or df.empty:
                return False
            strikes = sorted(df["strike"].unique())
            closest = min(strikes, key=lambda s: abs(s - flip))
            tgt_oi  = float(df[df["strike"] == closest]["oi"].sum())
            idx     = strikes.index(closest)
            nbrs    = [strikes[i] for i in range(max(0, idx-2),
                        min(len(strikes), idx+3)) if strikes[i] != closest]
            if not nbrs:
                return False
            nbr_oi = df[df["strike"].isin(nbrs)]["oi"].sum() / len(nbrs)
            return tgt_oi >= self.CONCENTRATION_MULT * nbr_oi
        except Exception:
            return False

    def _best_oi_wall(self, spot: float) -> float:
        try:
            df = self._chain_df_fn()
            if df is None or df.empty:
                return 0.0
            band = spot * 0.02
            near = df[(df["strike"] - spot).abs() <= band]
            if near.empty:
                return 0.0
            return float(near.groupby("strike")["oi"].sum().idxmax())
        except Exception:
            return 0.0

    # ── expiry calendar ──────────────────────────────────────────────────────

    def _minutes_to_expiry(self) -> Optional[float]:
        """
        Returns minutes until 15:30 IST if today is an expiry day, else None.
        Uses Wednesday (post-2024 switch) and Thursday (legacy) heuristic.
        No BhavCopy dependency — lightweight and self-contained.
        """
        now    = datetime.now()
        today  = now.weekday()          # 2=Wed, 3=Thu
        if today not in (2, 3):
            return None
        close  = now.replace(hour=15, minute=30, second=0, microsecond=0)
        delta  = (close - now).total_seconds()
        if delta < 0:
            return None
        return delta / 60

    # ── text helpers ─────────────────────────────────────────────────────────

    def _make_text(self, spot, flip, direction, near_vol, dist_pct,
                   minutes_left, met):
        dist_pts  = abs(spot - flip)
        conds_str = ", ".join(k for k, v in met.items() if v and k != "time") or "none"

        headline = (
            f"Spot {dist_pts:.0f} pts from {flip:.0f} {direction.lower()}side "
            f"flip — {minutes_left:.0f} min to expiry"
        )

        detail = (
            f"Target level: {flip:.0f} ({direction.lower()}side). "
            f"Distance: {dist_pts:.0f} pts ({dist_pct*100:.2f}%). "
            f"Near-strike volume: {near_vol:.0f} lots (5-min window). "
            f"Conditions met: {conds_str}. "
            f"Time remaining: {minutes_left:.0f} min."
        )

        if direction == "UPSIDE":
            inv = (f"Loses relevance if spot fails to hold above "
                   f"{spot*(1-0.003):.0f} in next 10 minutes "
                   f"or if volume decelerates.")
        else:
            inv = (f"Loses relevance if spot fails to hold below "
                   f"{spot*(1+0.003):.0f} in next 10 minutes "
                   f"or if volume decelerates.")

        return headline, detail, inv

    # ── audit + memory ───────────────────────────────────────────────────────

    def _write_audit(self, event: dict):
        try:
            from AlertDispatcher import fire
            level = {"ALERT": "CRITICAL", "WATCH": "WARNING",
                     "RESOLVED": "INFO"}.get(event["severity"], "INFO")
            fire(
                source="ExplosionWatch",
                level=level,
                title=event["headline"],
                body=event["detail"],
                data={k: v for k, v in event.items()
                      if k not in ("headline", "detail", "type")},
            )
        except Exception as _e:
            print(f"[ExplosionWatch] Audit write error (non-fatal): {_e}")

    def _write_memory(self, strike: float, severity: str, ts: str):
        if self._mem is None:
            return
        try:
            self._mem.update_context({
                "explosion_watch_state": {
                    "strike":         strike,
                    "severity":       severity,
                    "first_fired_at": self._last_fired_at or ts,
                }
            })
        except Exception:
            pass
