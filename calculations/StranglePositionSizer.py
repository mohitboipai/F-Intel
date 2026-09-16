"""
StranglePositionSizer.py — Institutional Dynamic Strangle Position Sizer
================================================================================
Calculates mathematically optimal capital allocation and quantity scaling for
short strangles / straddles in Indian Index Options (NIFTY 50).

Combines 4 quantitative gears:
1. Base Margin Capacity (Accounts for exchange initial margin & 30% cash buffer)
2. Predictive Volatility Edge Scaler (Forward VRP: IV - E[RV_forward])
3. Jump & Gamma Hazard Haircut (Bipower jump ratio & Net GEX flip boundary)
4. Asymmetric Strangle Leg Allocation (Semi-variance VAI for CE vs PE lot split)
5. 2-Sigma Overnight Gap Stress Test
"""

import math
from typing import Dict, Any, Optional


class StranglePositionSizer:
    """
    Computes optimal lot size and CE/PE leg distribution for option writing strategies.
    """

    DEFAULT_MARGIN_PER_LOT = 120_000.0  # ₹1.2 Lakhs approx margin for 1 NIFTY strangle lot (65 qty)
    DEFAULT_LOT_SIZE = 65                # Current NIFTY lot size
    DEFAULT_UTILIZATION = 0.70           # Max 70% capital utilization (30% buffer for MTM)

    def __init__(
        self,
        margin_per_lot: float = DEFAULT_MARGIN_PER_LOT,
        lot_size: int = DEFAULT_LOT_SIZE,
        max_utilization: float = DEFAULT_UTILIZATION
    ):
        self.margin_per_lot = margin_per_lot
        self.lot_size = lot_size
        self.max_utilization = max_utilization

    def calculate_sizing(
        self,
        capital: float,
        atm_iv: float = 14.0,
        forward_vrp: float = 2.0,
        jump_ratio: float = 0.10,
        vai: float = 0.0,
        spot: float = 23300.0,
        gamma_flip: Optional[float] = None,
        is_pinned: bool = False
    ) -> Dict[str, Any]:
        """
        Calculate complete strangle positioning recommendation.
        :param capital: Account capital in ₹ (e.g. 2,000,000)
        :param atm_iv: Live ATM Implied Volatility (e.g. 13.5)
        :param forward_vrp: Predictive Forward VRP = IV - Forecasted RV (e.g. +3.2)
        :param jump_ratio: Bipower Jump Ratio R_J (e.g. 0.12 = 12%)
        :param vai: Volatility Asymmetry Index [-1, 1] from Realized Semi-Variance
        :param spot: Current underlying spot price
        :param gamma_flip: Optional Zero Gamma level
        :param is_pinned: True if spot is locked in dominant pin corridor
        :return: Comprehensive sizing dictionary with lots, multipliers, and execution plan
        """
        capital = max(float(capital), 50_000.0)

        # ── 1. Base Capacity ──────────────────────────────────────────────────
        usable_capital = capital * self.max_utilization
        base_lots = max(1, math.floor(usable_capital / self.margin_per_lot))

        # ── 2. Edge Multiplier (Forward VRP) ──────────────────────────────────
        if forward_vrp >= 4.0:
            m_edge = 1.40
            edge_desc = f"Heavy VRP Edge (+{forward_vrp:.1f}%). Implied volatility is substantially richer than forecasted moves."
        elif forward_vrp >= 2.0:
            m_edge = 1.25
            edge_desc = f"Solid VRP Edge (+{forward_vrp:.1f}%). Options offer attractive statistical decay harvest."
        elif forward_vrp >= 0.5:
            m_edge = 1.05
            edge_desc = f"Moderate VRP Edge (+{forward_vrp:.1f}%). Standard decay harvest conditions."
        elif forward_vrp >= -0.5:
            m_edge = 0.90
            edge_desc = "Neutral VRP. Premiums are in parity with forecasted movement."
        elif forward_vrp >= -2.0:
            m_edge = 0.50
            edge_desc = f"Negative VRP ({forward_vrp:.1f}%). Options are cheap; selling premium has thin to negative edge."
        else:
            m_edge = 0.0
            edge_desc = f"Severe Negative VRP ({forward_vrp:.1f}%). Realized movement expected to outpace options premium. Avoid writing."

        # ── 3. Hazard Multiplier (Jumps & Gamma Flip) ─────────────────────────
        m_hazard = 1.0
        hazard_reasons = []

        # Jump penalty
        if jump_ratio >= 0.35:
            m_hazard *= 0.60
            hazard_reasons.append(f"High jump risk ({jump_ratio * 100:.0f}% transitory variance) — overnight gap hazard elevated")
        elif jump_ratio >= 0.20:
            m_hazard *= 0.80
            hazard_reasons.append(f"Moderate jump activity ({jump_ratio * 100:.0f}%)")
        else:
            hazard_reasons.append("Low jump hazard (continuous volatility flow)")

        # Gamma Flip boundary check
        if gamma_flip and gamma_flip > 0 and spot > 0:
            if spot < gamma_flip:
                m_hazard *= 0.75
                hazard_reasons.append(f"Spot ({spot:.0f}) below Gamma Flip ({gamma_flip:.0f}) — Negative Dealer Gamma volatility acceleration")
            else:
                m_hazard *= 1.05
                hazard_reasons.append(f"Spot above Gamma Flip ({gamma_flip:.0f}) — Positive Dealer Gamma stabilizing buffer")

        if is_pinned:
            m_hazard *= 1.10
            hazard_reasons.append("Spot anchored in Session Dominant Pin Corridor (+10% stability bonus)")

        # Cap hazard multiplier
        m_hazard = round(min(max(m_hazard, 0.40), 1.25), 2)

        # ── 4. Total Multiplier & Lots ────────────────────────────────────────
        if m_edge == 0.0:
            total_multiplier = 0.0
            optimal_lots = 0
        else:
            total_multiplier = round(m_edge * m_hazard, 2)
            optimal_lots = max(1, round(base_lots * total_multiplier))

        # Deployment %
        deployment_pct = round(total_multiplier * 100.0, 1)

        # ── 5. Strangle Leg Asymmetry (CE vs PE Distribution) ─────────────────
        if vai > 0.20:
            # Toxic downside hazard — reduce Put selling exposure
            ce_ratio = 0.62
            pe_ratio = 0.38
            skew_bias = "CALL_HEAVY_SKEW"
            skew_desc = f"Downside semi-variance is high (VAI +{vai:.2f}). Selling more Calls than Puts to hedge against Put gamma spike."
        elif vai < -0.20:
            # Bullish grind — low put panic
            ce_ratio = 0.45
            pe_ratio = 0.55
            skew_bias = "PUT_HEAVY_SKEW"
            skew_desc = f"Upside grind momentum dominant (VAI {vai:.2f}). Low crash hazard; Put writing offers higher safe harvest."
        else:
            ce_ratio = 0.50
            pe_ratio = 0.50
            skew_bias = "SYMMETRIC_NEUTRAL"
            skew_desc = "Balanced volatility between upside and downside. 1:1 symmetric strangle recommended."

        if optimal_lots > 0:
            ce_lots = max(1, round(optimal_lots * ce_ratio))
            pe_lots = max(0, optimal_lots - ce_lots)
            # Always rebalance to ensure ce_lots + pe_lots == optimal_lots exactly
            if ce_lots + pe_lots != optimal_lots:
                pe_lots = max(0, optimal_lots - ce_lots)
        else:
            ce_lots = 0
            pe_lots = 0

        # ── 6. 2-Sigma Overnight Gap Stress Test ──────────────────────────────
        daily_sigma = (atm_iv / 100.0) / math.sqrt(252.0)
        gap_2sigma_pts = round(2.0 * daily_sigma * spot, 1)
        # Assuming ~150-200 pt OTM strangle buffer:
        otm_buffer_pts = 180.0
        excess_pts = max(0.0, gap_2sigma_pts - otm_buffer_pts)
        estimated_stress_loss_per_lot = excess_pts * self.lot_size
        total_stress_loss = round(estimated_stress_loss_per_lot * optimal_lots, 0)
        stress_loss_pct = round((total_stress_loss / capital) * 100.0, 2)

        # ── 7. Sizing Verdict & Execution Blueprint ───────────────────────────
        if total_multiplier >= 1.25:
            verdict = "INCREASE_SIZE"
            verdict_badge = f"⚡ INCREASE SIZE ({deployment_pct:.0f}%)"
            verdict_color = "var(--color-bullish, #10b981)"
        elif total_multiplier >= 0.85:
            verdict = "STANDARD_SIZE"
            verdict_badge = f"🟢 STANDARD SIZE ({deployment_pct:.0f}%)"
            verdict_color = "#38bdf8"
        elif total_multiplier > 0.0:
            verdict = "DEFENSIVE_CUT"
            verdict_badge = f"🛡 DEFENSIVE CUT ({deployment_pct:.0f}%)"
            verdict_color = "var(--color-warning, #f59e0b)"
        else:
            verdict = "AVOID_STRANGLES"
            verdict_badge = "⛔ AVOID SHORT STRANGLES (0%)"
            verdict_color = "var(--color-bearish, #ef4444)"

        return {
            "capital": capital,
            "base_lots": base_lots,
            "optimal_lots": optimal_lots,
            "ce_lots": ce_lots,
            "pe_lots": pe_lots,
            "total_multiplier": total_multiplier,
            "deployment_pct": deployment_pct,
            "verdict": verdict,
            "verdict_badge": verdict_badge,
            "verdict_color": verdict_color,
            "edge": {
                "forward_vrp": round(forward_vrp, 2),
                "multiplier": round(m_edge, 2),
                "description": edge_desc
            },
            "hazard": {
                "jump_ratio_pct": round(jump_ratio * 100.0, 1),
                "multiplier": round(m_hazard, 2),
                "reasons": hazard_reasons
            },
            "skew": {
                "vai": round(vai, 3),
                "bias": skew_bias,
                "ce_ratio": round(ce_ratio, 2),
                "pe_ratio": round(pe_ratio, 2),
                "description": skew_desc
            },
            "stress_test": {
                "gap_2sigma_pts": gap_2sigma_pts,
                "estimated_loss_inr": total_stress_loss,
                "risk_pct_of_capital": stress_loss_pct,
                "is_within_budget": stress_loss_pct <= 2.5
            }
        }
