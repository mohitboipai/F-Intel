"""
StrategyWizard.py
=================
View-driven strategy selection layer on top of SmartStrategyGenerator.

Sensibull's key UX insight: strategy selection should be driven by the
trader's stated MARKET VIEW and CAPITAL, not just quantitative signals.
This module adds that view-driven layer without replacing the existing
signal-scoring system — it re-ranks and filters, then provides a
`lots_possible` capital check.

Usage:
    wizard = StrategyWizard(spot, df_chain, market_context, expiry)
    recommendations = wizard.recommend(
        view='NEUTRAL', risk='MODERATE', capital_inr=200000, conviction='HIGH'
    )
    # Returns list of up to 3 dicts, each with Strategy + metadata.
"""

import numpy as np
from typing import List

from StrategyEngine import SmartStrategyGenerator, Strategy

LOT_SIZE  = 75     # NIFTY lot size — must match StrategyEngine.NIFTY_LOT
RISK_FREE = 0.07   # 7% Indian risk-free


class StrategyWizard:
    """
    View-driven strategy selection layer on top of SmartStrategyGenerator.
    Filters and re-ranks strategies based on trader's stated view, risk
    appetite, available capital, and conviction level.
    """

    # Strategy names that are appropriate for each market view
    VIEW_STRATEGY_MAP = {
        'BULLISH':      ['Bull Put Spread', 'Short Straddle', 'Short Strangle'],
        'BEARISH':      ['Bear Call Spread', 'Short Straddle', 'Short Strangle'],
        'NEUTRAL':      ['Iron Condor', 'Short Straddle', 'Short Strangle',
                         'Bull Put Spread', 'Bear Call Spread'],
        'VOLATILE':     ['Long Straddle', 'Long Strangle'],
        'NON-VOLATILE': ['Iron Condor', 'Short Straddle', 'Short Strangle',
                         'Bull Put Spread', 'Bear Call Spread'],
    }

    # Strategy names allowed for each risk appetite
    RISK_STRATEGY_MAP = {
        'CONSERVATIVE': ['Iron Condor', 'Bull Put Spread', 'Bear Call Spread'],
        'MODERATE':     ['Iron Condor', 'Short Strangle', 'Bull Put Spread',
                         'Bear Call Spread', 'Short Straddle'],
        'AGGRESSIVE':   ['Short Straddle', 'Short Strangle', 'Long Straddle',
                         'Bull Put Spread', 'Bear Call Spread', 'Iron Condor'],
    }

    def __init__(self, spot: float, df_chain, market_context: dict, expiry: str = ''):
        self.spot   = spot
        self.chain  = df_chain
        self.ctx    = market_context
        self.expiry = expiry
        self._gen   = SmartStrategyGenerator(spot, df_chain, market_context, expiry)

    def recommend(self, view: str, risk: str, capital_inr: float,
                  conviction: str = 'MODERATE') -> List[dict]:
        """
        Generate and rank strategies given trader's stated view and capital.

        Parameters
        ----------
        view        : 'BULLISH' | 'BEARISH' | 'NEUTRAL' | 'VOLATILE' | 'NON-VOLATILE'
        risk        : 'CONSERVATIVE' | 'MODERATE' | 'AGGRESSIVE'
        capital_inr : Available capital in Indian Rupees
        conviction  : 'LOW' | 'MODERATE' | 'HIGH'  — amplifies view bonus

        Returns
        -------
        List of up to 3 dicts, sorted by blended_score descending,
        view-matched strategies first. Each dict contains:
          strategy      Strategy object
          score         int 0-100 (blended signal + view score)
          view_match    bool
          max_loss_inr  float (per lot, INR)
          lots_possible int
          reasoning     str
        """
        # Step 1: generate all strategies (signal-scored by SmartStrategyGenerator)
        all_strats = self._gen.generate()

        # Step 2: allowed sets from view and risk maps
        view_key = view.upper().replace('-', '-')   # preserve NON-VOLATILE
        risk_key = risk.upper()
        allowed_by_view = set(self.VIEW_STRATEGY_MAP.get(view_key, []))
        allowed_by_risk = set(self.RISK_STRATEGY_MAP.get(risk_key, []))
        allowed         = allowed_by_view & allowed_by_risk

        # Conviction multiplier — HIGH conviction amplifies the view bonus
        conv_mult = 1.3 if conviction.upper() == 'HIGH' else 1.0

        # Step 3: score each strategy
        price_range = np.linspace(self.spot * 0.85, self.spot * 1.15, 500)
        results = []

        for strat in all_strats:
            view_match   = strat.name in allowed
            signal_score = strat.score          # 0-100 from SmartStrategyGenerator

            # View bonus: matching strategies get +20, mismatches get -15
            view_bonus    = 20 if view_match else -15
            blended_score = min(100, int(signal_score + view_bonus * conv_mult))

            # Capital check: determine max_loss_per_lot and lots_possible
            max_loss_per_lot = abs(strat.max_loss(price_range)) * LOT_SIZE
            lots_possible    = 0

            if max_loss_per_lot > 0 and max_loss_per_lot < 1e6:
                # Defined-risk strategy — straightforward capital division
                lots_possible = max(0, int(capital_inr / max_loss_per_lot))
            elif strat.strategy_type == 'CREDIT':
                # Unlimited-risk strategy (naked) — use 3× net premium as margin estimate
                margin_est    = abs(strat.net_premium) * LOT_SIZE * 3
                lots_possible = max(1, int(capital_inr / margin_est)) if margin_est > 0 else 1

            # Build human-readable reasoning
            reason_parts = [
                f"{'Matches' if view_match else 'Does not match'} {view_key} view",
                f"Signal score: {signal_score}/100",
            ]
            if strat.net_premium > 0:
                reason_parts.append(f"Credit: +Rs.{strat.net_premium * LOT_SIZE:.0f}/lot")
            elif strat.net_premium < 0:
                reason_parts.append(f"Debit: -Rs.{abs(strat.net_premium * LOT_SIZE):.0f}/lot")
            if max_loss_per_lot >= 1e6:
                reason_parts.append("Max loss: Unlimited (margin strategy)")
            else:
                reason_parts.append(f"Max loss: Rs.{max_loss_per_lot:,.0f}/lot")

            results.append({
                'strategy':      strat,
                'score':         blended_score,
                'view_match':    view_match,
                'max_loss_inr':  max_loss_per_lot,
                'lots_possible': lots_possible,
                'reasoning':     ' | '.join(reason_parts),
            })

        # Sort: view matches first, then by blended score descending
        results.sort(key=lambda x: (x['view_match'], x['score']), reverse=True)
        return results[:3]
