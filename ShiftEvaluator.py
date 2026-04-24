"""
ShiftEvaluator.py
=================
Evaluates whether an open strangle (or other strategy) should be shifted
on a given day. Returns a ShiftDecision dataclass.

Two trigger categories:
  1. SPOT-BASED  -- short strike is within 0.8% of current spot
  2. SIGNAL-BASED -- VRP ratio dropped below 1.0 (selling no longer favourable)

Guards against shifting:
  - Last 2 calendar days before expiry -> no shift
  - Already shifted today (tracked by caller)
  - Net running credit is already negative -> hold, do not compound

Public API
----------
    evaluator = ShiftEvaluator(threshold_pct=0.8)
    decision  = evaluator.evaluate(
        spot, ce_strike, pe_strike, expiry_date, trade_date,
        signal_row, running_net_per_unit
    )  -> ShiftDecision
"""

import os
import sys
from dataclasses import dataclass, field
from datetime import date as date_type, datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

RISK_FREE = 0.07


def _to_date(d) -> date_type:
    if isinstance(d, date_type) and not isinstance(d, datetime):
        return d
    if isinstance(d, datetime):
        return d.date()
    return datetime.strptime(str(d)[:10], '%Y-%m-%d').date()


def _round50(x: float) -> float:
    return round(x / 50) * 50


@dataclass
class ShiftDecision:
    should_shift:   bool
    shift_type:     str | None      # "SPOT_CE" | "SPOT_PE" | "SPOT_BOTH" | "SIGNAL" | None
    ce_action:      str | None      # "CLOSE_AND_REOPEN" | "KEEP" | None
    pe_action:      str | None      # "CLOSE_AND_REOPEN" | "KEEP" | None
    new_ce_strike:  float | None
    new_pe_strike:  float | None
    reason:         str


class ShiftEvaluator:
    """
    Evaluates daily whether an open strangle position needs adjustment.

    Parameters
    ----------
    threshold_pct : float
        Minimum % distance from spot to short strike before triggering
        a spot-based shift. Default 0.8%.
    vrp_neutral_threshold : float
        VRP ratio below which selling premium is no longer favourable.
        Default 1.0 (IV < HV means IV is cheap).
    """

    def __init__(
        self,
        threshold_pct: float = 0.8,
        vrp_neutral_threshold: float = 1.0,
    ):
        self.threshold_pct         = threshold_pct
        self.vrp_neutral_threshold = vrp_neutral_threshold

    # ── main evaluate ────────────────────────────────────────────────────────

    def evaluate(
        self,
        spot:                float,
        ce_strike:           float,
        pe_strike:           float,
        expiry_date,
        trade_date,
        signal_row:          dict,
        running_net_per_unit: float,
        already_shifted_today: bool = False,
        strangle_width_pct:  float = 2.0,   # % from spot when opening new legs
    ) -> ShiftDecision:
        """
        Decide whether to shift the position today.

        Parameters
        ----------
        spot                  : current spot price (closing bar)
        ce_strike             : currently open short CE strike
        pe_strike             : currently open short PE strike
        expiry_date           : expiry date of this position
        trade_date            : today's date
        signal_row            : dict from BacktestSignalExtractor.get_signal()
        running_net_per_unit  : cumulative net credit per unit so far
        already_shifted_today : prevent double-shift in one day
        strangle_width_pct    : % from spot for new strikes when repositioning
        """
        expiry = _to_date(expiry_date)
        today  = _to_date(trade_date)

        # ── Guard 1: too close to expiry ─────────────────────────────────────
        days_to_expiry = (expiry - today).days
        if days_to_expiry <= 1:
            return ShiftDecision(
                should_shift=False, shift_type=None,
                ce_action=None, pe_action=None,
                new_ce_strike=None, new_pe_strike=None,
                reason='Last 2 calendar days before expiry -- no shift'
            )

        # ── Guard 2: already shifted today ───────────────────────────────────
        if already_shifted_today:
            return ShiftDecision(
                should_shift=False, shift_type=None,
                ce_action=None, pe_action=None,
                new_ce_strike=None, new_pe_strike=None,
                reason='Already shifted once today -- max 1 shift/day'
            )

        # ── Guard 3: net position already a net debit ────────────────────────
        if running_net_per_unit < 0:
            return ShiftDecision(
                should_shift=False, shift_type=None,
                ce_action=None, pe_action=None,
                new_ce_strike=None, new_pe_strike=None,
                reason=f'Net debit position (running net={running_net_per_unit:.2f}) -- hold until expiry'
            )

        # ── Distance calculations ─────────────────────────────────────────────
        ce_dist_pct = (ce_strike - spot) / spot * 100.0
        pe_dist_pct = (spot - pe_strike) / spot * 100.0

        ce_breached = ce_dist_pct < self.threshold_pct
        pe_breached = pe_dist_pct < self.threshold_pct

        # ── Signal-based check ────────────────────────────────────────────────
        vrp_ratio = signal_row.get('vrp_ratio')
        signal_flip = (
            vrp_ratio is not None
            and vrp_ratio < self.vrp_neutral_threshold
        )

        # ── Decision logic ────────────────────────────────────────────────────

        # Signal-based: takes precedence if VRP flipped -- reposition both legs
        if signal_flip and not ce_breached and not pe_breached:
            new_ce = _round50(spot * (1 + strangle_width_pct / 100))
            new_pe = _round50(spot * (1 - strangle_width_pct / 100))
            return ShiftDecision(
                should_shift=True,
                shift_type='SIGNAL',
                ce_action='CLOSE_AND_REOPEN',
                pe_action='CLOSE_AND_REOPEN',
                new_ce_strike=new_ce,
                new_pe_strike=new_pe,
                reason=(
                    f'VRP ratio {vrp_ratio:.2f} < {self.vrp_neutral_threshold} -- '
                    f'IV cheap, signal flipped; repositioning strangle at {new_pe}/{new_ce}'
                )
            )

        # Both legs breached simultaneously (spot inside the strangle)
        if ce_breached and pe_breached:
            new_ce = _round50(spot * (1 + strangle_width_pct / 100))
            new_pe = _round50(spot * (1 - strangle_width_pct / 100))
            return ShiftDecision(
                should_shift=True,
                shift_type='SPOT_BOTH',
                ce_action='CLOSE_AND_REOPEN',
                pe_action='CLOSE_AND_REOPEN',
                new_ce_strike=new_ce,
                new_pe_strike=new_pe,
                reason=(
                    f'Spot {spot:.0f} is inside strangle ({pe_strike}/{ce_strike}); '
                    f'CE dist={ce_dist_pct:.2f}%, PE dist={pe_dist_pct:.2f}% -- both legs shifted'
                )
            )

        # Only CE breached
        if ce_breached:
            new_ce = _round50(spot * (1 + strangle_width_pct / 100))
            return ShiftDecision(
                should_shift=True,
                shift_type='SPOT_CE',
                ce_action='CLOSE_AND_REOPEN',
                pe_action='KEEP',
                new_ce_strike=new_ce,
                new_pe_strike=pe_strike,
                reason=(
                    f'CE {ce_strike} within {ce_dist_pct:.2f}% of spot {spot:.0f} '
                    f'(threshold {self.threshold_pct}%) -- CE shifted to {new_ce}'
                )
            )

        # Only PE breached
        if pe_breached:
            new_pe = _round50(spot * (1 - strangle_width_pct / 100))
            return ShiftDecision(
                should_shift=True,
                shift_type='SPOT_PE',
                ce_action='KEEP',
                pe_action='CLOSE_AND_REOPEN',
                new_ce_strike=ce_strike,
                new_pe_strike=new_pe,
                reason=(
                    f'PE {pe_strike} within {pe_dist_pct:.2f}% of spot {spot:.0f} '
                    f'(threshold {self.threshold_pct}%) -- PE shifted to {new_pe}'
                )
            )

        # No shift required
        return ShiftDecision(
            should_shift=False, shift_type=None,
            ce_action='KEEP', pe_action='KEEP',
            new_ce_strike=None, new_pe_strike=None,
            reason=(
                f'No shift: CE dist={ce_dist_pct:.2f}%, PE dist={pe_dist_pct:.2f}%, '
                f'VRP={vrp_ratio if vrp_ratio is not None else "N/A"}'
            )
        )

    # ── Straddle variant ─────────────────────────────────────────────────────

    def evaluate_straddle(
        self,
        spot:                float,
        atm_strike:          float,
        expiry_date,
        trade_date,
        signal_row:          dict,
        running_net_per_unit: float,
        already_shifted_today: bool = False,
        straddle_shift_pct:  float = 1.5,
    ) -> ShiftDecision:
        """
        Shift logic for short straddle: trigger if spot moves > 1.5% from ATM.
        When triggered, close both legs and recentre at new ATM.
        """
        expiry = _to_date(expiry_date)
        today  = _to_date(trade_date)

        if (expiry - today).days <= 1:
            return ShiftDecision(False, None, None, None, None, None,
                                 'Last 2 days before expiry -- no shift')
        if already_shifted_today:
            return ShiftDecision(False, None, None, None, None, None,
                                 'Already shifted today')
        if running_net_per_unit < 0:
            return ShiftDecision(False, None, None, None, None, None,
                                 'Net debit -- hold until expiry')

        dist_pct = abs(spot - atm_strike) / atm_strike * 100.0

        if dist_pct > straddle_shift_pct:
            new_atm = _round50(spot)
            return ShiftDecision(
                should_shift=True, shift_type='SPOT_BOTH',
                ce_action='CLOSE_AND_REOPEN', pe_action='CLOSE_AND_REOPEN',
                new_ce_strike=new_atm, new_pe_strike=new_atm,
                reason=(
                    f'Straddle: spot {spot:.0f} moved {dist_pct:.2f}% from ATM {atm_strike} '
                    f'(threshold {straddle_shift_pct}%) -- recentre at {new_atm}'
                )
            )

        return ShiftDecision(False, None, 'KEEP', 'KEEP', None, None,
                             f'Straddle HOLD: dist {dist_pct:.2f}%')

    # ── Iron Condor variant ──────────────────────────────────────────────────

    def evaluate_condor(
        self,
        spot:                float,
        short_ce:            float,
        short_pe:            float,
        long_ce:             float,
        long_pe:             float,
        expiry_date,
        trade_date,
        running_net_per_unit: float,
        already_shifted_today: bool = False,
        threshold_pct:       float = 0.8,
        width_pct:           float = 2.0,
        wing_pct:            float = 1.0,
    ) -> ShiftDecision:
        """Shift logic for Iron Condor. Only shifts if net credit > 0 after rebuild."""
        expiry = _to_date(expiry_date)
        today  = _to_date(trade_date)

        if (expiry - today).days <= 1:
            return ShiftDecision(False, None, None, None, None, None,
                                 'Last 2 days before expiry')
        if already_shifted_today or running_net_per_unit < 0:
            return ShiftDecision(False, None, None, None, None, None,
                                 'Condor: guard prevented shift')

        ce_dist = (short_ce - spot) / spot * 100
        pe_dist = (spot - short_pe) / spot * 100

        if ce_dist < threshold_pct or pe_dist < threshold_pct:
            new_short_ce = _round50(spot * (1 + width_pct / 100))
            new_long_ce  = _round50(spot * (1 + width_pct / 100 + wing_pct / 100))
            new_short_pe = _round50(spot * (1 - width_pct / 100))
            new_long_pe  = _round50(spot * (1 - width_pct / 100 - wing_pct / 100))
            breached = []
            if ce_dist < threshold_pct: breached.append('CE')
            if pe_dist < threshold_pct: breached.append('PE')
            return ShiftDecision(
                should_shift=True, shift_type='SPOT_BOTH',
                ce_action='CLOSE_AND_REOPEN', pe_action='CLOSE_AND_REOPEN',
                new_ce_strike=new_short_ce, new_pe_strike=new_short_pe,
                reason=f'Condor {"/".join(breached)} breach -- rebuild around {spot:.0f}'
            )

        return ShiftDecision(False, None, 'KEEP', 'KEEP', None, None,
                             f'Condor HOLD: CE dist={ce_dist:.2f}%, PE dist={pe_dist:.2f}%')
