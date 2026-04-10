"""
ScenarioEngine.py
=================
Computes strategy P&L across a 3D grid of scenarios for interactive
"what-if" sliders — equivalent to Sensibull's scenario analysis panel.

The three dimensions are:
  - Spot change   (default ±10%)
  - DTE remaining (0 = today's BSM price, 1.0 = at expiry)
  - IV change     (default ±5 volatility points)

Also provides a 1D spot_ladder() for a clean payoff table with
per-level probability.

Usage:
    engine = ScenarioEngine(strategy, spot, T, sigma)
    grid   = engine.compute_grid()
    ladder = engine.spot_ladder()
"""

import numpy as np
from typing import List, Optional

from StrategyEngine import (
    bsm_price, bsm_prob_above, bsm_prob_below,
    RISK_FREE, NIFTY_LOT, Strategy
)

LOT_SIZE = NIFTY_LOT   # 75


class ScenarioEngine:
    """
    Computes strategy P&L across a 3D grid of scenarios.
    Used by the /api/scenario endpoint for interactive what-if sliders.
    """

    def __init__(self, strategy: Strategy, spot: float, T: float,
                 sigma: float, r: float = RISK_FREE, pricing_router=None):
        """
        Parameters
        ----------
        strategy       : Strategy object with populated legs
        spot           : Current NIFTY spot price
        T              : Time to expiry in years
        sigma          : ATM implied volatility (decimal, e.g. 0.13)
        r              : Risk-free rate (default 0.07)
        pricing_router : Optional PricingRouter for Heston pricing
                         (currently reserved for future use; BSM used for speed)
        """
        self.strategy = strategy
        self.spot     = spot
        self.T        = T
        self.sigma    = sigma
        self.r        = r
        self.router   = pricing_router

    def compute_grid(self,
                     spot_pct_range: tuple = (-0.10, 0.10),
                     dte_range:      tuple = (0.0,   1.0),
                     iv_bump_range:  tuple = (-5,    5),
                     n_spot: int = 21,
                     n_dte:  int = 5,
                     n_iv:   int = 5) -> dict:
        """
        Compute P&L across a 3D grid.

        Parameters
        ----------
        spot_pct_range : (min_pct, max_pct) — spot move as fraction of current spot
        dte_range      : (0.0, 1.0) — 0=now (BSM), 1=at expiry (intrinsic)
        iv_bump_range  : (low, high) — IV shift in volatility points
        n_spot, n_dte, n_iv : grid resolution

        Returns
        -------
        dict with keys:
          spot_pcts        list[float]   spot move fractions
          dte_fracs        list[float]   DTE fractions
          iv_bumps         list[float]   IV bumps in vol points
          pnl              list (3D, [n_iv][n_dte][n_spot]) P&L in INR per lot
          breakeven_spots  list[float]   approximate breakeven spot levels
        """
        spot_pcts = np.linspace(spot_pct_range[0], spot_pct_range[1], n_spot)
        dte_fracs = np.linspace(dte_range[0],       dte_range[1],       n_dte)
        iv_bumps  = np.linspace(iv_bump_range[0],   iv_bump_range[1],   n_iv)

        pnl = np.zeros((n_iv, n_dte, n_spot))

        for i, iv_bump in enumerate(iv_bumps):
            adj_sigma = max(0.01, self.sigma + iv_bump / 100.0)
            for j, dte_frac in enumerate(dte_fracs):
                T_remaining = max(0.0, self.T * (1.0 - dte_frac))
                for k, sp in enumerate(spot_pcts):
                    test_spot = self.spot * (1.0 + sp)
                    if T_remaining <= 0:
                        # At expiry: use intrinsic pnl
                        leg_pnl = sum(
                            leg.pnl_at_expiry(test_spot)
                            for leg in self.strategy.legs
                        )
                    else:
                        # Before expiry: BSM mark-to-market
                        leg_pnl = sum(
                            leg.direction * (
                                bsm_price(test_spot, leg.strike, T_remaining,
                                          self.r, adj_sigma, leg.opt_type)
                                - leg.entry_price
                            )
                            for leg in self.strategy.legs
                        )
                    pnl[i, j, k] = leg_pnl * LOT_SIZE

        # Breakeven spots at mid-IV, at expiry (row: iv_bump=0, dte=1.0)
        mid_iv_idx  = n_iv  // 2
        expiry_idx  = n_dte - 1
        expiry_row  = pnl[mid_iv_idx, expiry_idx, :]
        breakeven_spots = []
        for k in range(1, n_spot):
            if expiry_row[k - 1] * expiry_row[k] < 0:
                # Linear interpolation
                frac = -expiry_row[k - 1] / (expiry_row[k] - expiry_row[k - 1])
                be = self.spot * (1 + spot_pcts[k - 1] +
                                  frac * (spot_pcts[k] - spot_pcts[k - 1]))
                breakeven_spots.append(round(be, 0))

        return {
            'spot_pcts':       spot_pcts.tolist(),
            'dte_fracs':       dte_fracs.tolist(),
            'iv_bumps':        iv_bumps.tolist(),
            'pnl':             pnl.tolist(),
            'breakeven_spots': breakeven_spots,
        }

    def spot_ladder(self, spot_pct_range: tuple = (-0.10, 0.10),
                    n: int = 21) -> List[dict]:
        """
        1D payoff table: for each hypothetical spot level produce
        pnl_expiry, pnl_today (BSM), and probability of reaching that level.

        This is the Sensibull "payoff summary table" equivalent.

        Returns
        -------
        List of dicts with keys:
          spot       float   spot price level
          move_pct   float   % change from current spot
          pnl_expiry float   intrinsic P&L at expiry (INR, lot-adjusted)
          pnl_today  float   BSM mark-to-market P&L today (INR, lot-adjusted)
          prob       float   probability of reaching this level (%)
        """
        spot_levels = np.linspace(
            self.spot * (1.0 + spot_pct_range[0]),
            self.spot * (1.0 + spot_pct_range[1]),
            n
        )
        results = []
        for s in spot_levels:
            pnl_exp   = sum(l.pnl_at_expiry(s) for l in self.strategy.legs) * LOT_SIZE
            pnl_today = sum(l.pnl_now_bsm(s, self.T, self.r)
                            for l in self.strategy.legs) * LOT_SIZE

            # BSM risk-neutral probability of reaching this level
            if self.T > 0 and self.sigma > 0:
                if s > self.spot:
                    prob = bsm_prob_above(self.spot, s, self.T, self.sigma, self.r)
                else:
                    prob = bsm_prob_below(self.spot, s, self.T, self.sigma, self.r)
            else:
                prob = 1.0 if abs(s - self.spot) < 1 else 0.0

            results.append({
                'spot':       round(float(s), 0),
                'move_pct':   round((s / self.spot - 1.0) * 100, 2),
                'pnl_expiry': round(float(pnl_exp), 0),
                'pnl_today':  round(float(pnl_today), 0),
                'prob':       round(float(prob * 100), 1),
            })
        return results
