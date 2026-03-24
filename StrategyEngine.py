"""
StrategyEngine.py
=================
Core options strategy engine for the VolatilityAnalyzer dashboard.

Provides:
  - OptionLeg       : Single option contract representation
  - Strategy        : Multi-leg strategy with payoff / POP / greeks
  - StrategyBuilder : Factory for standard NIFTY option strategies
  - PayoffEngine    : Plotly-based payoff chart + greeks table builder
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ──────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────
NIFTY_LOT = 75          # NIFTY lot size
RISK_FREE  = 0.07       # 7% risk-free rate (India)

DARK_BG  = '#0f0f19'
CARD_BG  = '#1a1a2e'
ACCENT   = '#4fc3f7'
GREEN    = '#66bb6a'
RED      = '#ff4444'
YELLOW   = '#ffd54f'
WHITE    = '#e0e0e0'
MUTED    = '#888888'
ORANGE   = '#ff7043'


# ──────────────────────────────────────────────────────────────
# Black-Scholes helpers
# ──────────────────────────────────────────────────────────────
def _d1d2(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return 0.0, 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return d1, d2


def bsm_price(S, K, T, r, sigma, opt_type='CE'):
    if T <= 0:
        if opt_type == 'CE':
            return max(0.0, S - K)
        return max(0.0, K - S)
    d1, d2 = _d1d2(S, K, T, r, sigma)
    if opt_type == 'CE':
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def bsm_greeks(S, K, T, r, sigma, opt_type='CE'):
    if T <= 0 or sigma <= 0:
        return {'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0}
    d1, d2 = _d1d2(S, K, T, r, sigma)
    pdf_d1 = norm.pdf(d1)
    gamma  = pdf_d1 / (S * sigma * np.sqrt(T))
    vega   = S * pdf_d1 * np.sqrt(T) / 100          # per 1% IV move
    if opt_type == 'CE':
        delta = norm.cdf(d1)
        theta = (-(S * pdf_d1 * sigma) / (2 * np.sqrt(T))
                 - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
    else:
        delta = norm.cdf(d1) - 1
        theta = (-(S * pdf_d1 * sigma) / (2 * np.sqrt(T))
                 + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365
    return {'delta': delta, 'gamma': gamma, 'theta': theta, 'vega': vega}


def bsm_prob_above(S, K, T, sigma, r=RISK_FREE):
    """P(spot > K at expiry) under risk-neutral measure."""
    if T <= 0 or sigma <= 0:
        return (1.0 if S > K else 0.0)
    _, d2 = _d1d2(S, K, T, r, sigma)
    return norm.cdf(d2)


def bsm_prob_below(S, K, T, sigma, r=RISK_FREE):
    return 1.0 - bsm_prob_above(S, K, T, sigma, r)


# ──────────────────────────────────────────────────────────────
# OptionLeg
# ──────────────────────────────────────────────────────────────
@dataclass
class OptionLeg:
    """Single option leg in a strategy."""
    opt_type:    str    # 'CE' or 'PE'
    action:      str    # 'BUY' or 'SELL'
    strike:      float
    entry_price: float  # per unit (not lot)
    iv:          float  # decimal e.g. 0.13
    lots:        int    = 1
    expiry:      str    = ''   # 'YYYY-MM-DD'

    @property
    def direction(self) -> int:
        """+1 for BUY, -1 for SELL."""
        return 1 if self.action == 'BUY' else -1

    @property
    def premium_received(self) -> float:
        """Net cash received (+) or paid (-) per unit."""
        return -self.direction * self.entry_price

    @property
    def lot_premium(self) -> float:
        return self.premium_received * NIFTY_LOT * self.lots

    def pnl_at_expiry(self, spot: float) -> float:
        """P&L per unit at expiry (not lot-adjusted)."""
        if self.opt_type == 'CE':
            intrinsic = max(0.0, spot - self.strike)
        else:
            intrinsic = max(0.0, self.strike - spot)
        return self.direction * intrinsic - self.direction * self.entry_price

    def pnl_now_bsm(self, spot: float, T: float, r: float = RISK_FREE) -> float:
        """Mark-to-market P&L using BSM (theoretical)."""
        current_bsm = bsm_price(spot, self.strike, T, r, self.iv, self.opt_type)
        return self.direction * (current_bsm - self.entry_price)

    def greeks(self, spot: float, T: float, r: float = RISK_FREE) -> Dict:
        g = bsm_greeks(spot, self.strike, T, r, self.iv, self.opt_type)
        # Sign by direction (SELL flips all greeks)
        factor = self.direction * self.lots * NIFTY_LOT
        return {k: v * factor for k, v in g.items()}

    def to_dict(self) -> dict:
        return {
            'type': self.opt_type, 'action': self.action,
            'strike': self.strike, 'price': self.entry_price,
            'iv': round(self.iv * 100, 2), 'lots': self.lots, 'expiry': self.expiry
        }


# ──────────────────────────────────────────────────────────────
# Strategy
# ──────────────────────────────────────────────────────────────
class Strategy:
    """
    A named multi-leg options strategy.
    Works on per-unit (1 lot) quantities unless lots > 1.
    """

    def __init__(self, name: str, legs: List[OptionLeg], strategy_type: str = 'CREDIT'):
        self.name          = name
        self.legs          = legs
        self.strategy_type = strategy_type   # 'CREDIT' or 'DEBIT'
        self.score         = 0               # set externally by scoring logic

    # ── Premium ──────────────────────────────────────────
    @property
    def net_premium(self) -> float:
        """Net premium per unit (+ = credit received, - = debit paid)."""
        return sum(l.premium_received for l in self.legs)

    @property
    def net_premium_lots(self) -> float:
        return sum(l.lot_premium for l in self.legs)

    # ── Payoff vectors ───────────────────────────────────
    def payoff_at_expiry(self, spot_range: np.ndarray) -> np.ndarray:
        return np.array([sum(l.pnl_at_expiry(s) for l in self.legs) for s in spot_range])

    def payoff_now_bsm(self, spot_range: np.ndarray, T: float, r: float = RISK_FREE) -> np.ndarray:
        return np.array([sum(l.pnl_now_bsm(s, T, r) for l in self.legs) for s in spot_range])

    # ── Greeks ───────────────────────────────────────────
    def net_greeks(self, spot: float, T: float, r: float = RISK_FREE) -> Dict:
        totals = {'delta': 0.0, 'gamma': 0.0, 'theta': 0.0, 'vega': 0.0}
        for leg in self.legs:
            g = leg.greeks(spot, T, r)
            for k in totals:
                totals[k] += g[k]
        return totals

    def per_leg_greeks(self, spot: float, T: float, r: float = RISK_FREE) -> List[Dict]:
        result = []
        for leg in self.legs:
            g = leg.greeks(spot, T, r)
            g.update({'strike': leg.strike, 'type': leg.opt_type,
                      'action': leg.action, 'entry_price': leg.entry_price})
            result.append(g)
        return result

    # ── Risk metrics ─────────────────────────────────────
    def breakevens(self, spot_range: np.ndarray) -> List[float]:
        """Returns spot levels where expiry P&L crosses zero."""
        pnl = self.payoff_at_expiry(spot_range)
        bes = []
        for i in range(1, len(pnl)):
            if pnl[i - 1] * pnl[i] < 0:
                # linear interpolation
                be = spot_range[i - 1] - pnl[i - 1] * (spot_range[i] - spot_range[i - 1]) / (pnl[i] - pnl[i - 1])
                bes.append(round(be, 0))
        return bes

    def max_profit(self, spot_range: np.ndarray) -> float:
        pnl = self.payoff_at_expiry(spot_range)
        return float(np.max(pnl))

    def max_loss(self, spot_range: np.ndarray) -> float:
        pnl = self.payoff_at_expiry(spot_range)
        return float(np.min(pnl))

    def pop(self, spot: float, T: float, sigma: float, r: float = RISK_FREE) -> float:
        """
        Probability of Profit at expiry using BSM log-normal CDF.
        Samples 2000 price points for accuracy.
        """
        lo = spot * 0.70
        hi = spot * 1.30
        prices = np.linspace(lo, hi, 2000)
        pnl    = self.payoff_at_expiry(prices)

        # BSM log-normal probability density
        if T <= 0 or sigma <= 0:
            return float(np.mean(pnl > 0))
        mu      = np.log(spot) + (r - 0.5 * sigma ** 2) * T
        std_dev = sigma * np.sqrt(T)
        log_prices = np.log(prices)
        dp     = log_prices[1] - log_prices[0]
        pdf    = np.exp(-0.5 * ((log_prices - mu) / std_dev) ** 2) / (std_dev * np.sqrt(2 * np.pi))
        weights = pdf / (pdf.sum() + 1e-12)
        return float(np.sum(weights * (pnl > 0)))

    def to_dict(self) -> dict:
        return {
            'name': self.name, 'type': self.strategy_type,
            'premium': self.net_premium, 'score': self.score,
            'reasoning': '',
            'legs': [l.to_dict() for l in self.legs]
        }


# ──────────────────────────────────────────────────────────────
# StrategyBuilder  — live-context-aware strategy factory
# ──────────────────────────────────────────────────────────────
class StrategyBuilder:
    """
    Generates standard NIFTY option strategies calibrated to live market data.
    All strikes are rounded to nearest 50 (NIFTY strike spacing).
    """

    @staticmethod
    def _round50(x: float) -> float:
        return round(x / 50) * 50

    @classmethod
    def short_straddle(cls, spot, atm_strike, T, iv_dec, ce_price, pe_price,
                       expiry='', lots=1) -> Strategy:
        legs = [
            OptionLeg('CE', 'SELL', atm_strike, ce_price, iv_dec, lots, expiry),
            OptionLeg('PE', 'SELL', atm_strike, pe_price, iv_dec, lots, expiry),
        ]
        return Strategy('Short Straddle', legs, 'CREDIT')

    @classmethod
    def short_strangle(cls, spot, T, iv_dec, df_chain, expiry='', lots=1,
                       otm_pct=0.03) -> Strategy:
        """OTM strangle — sell CE ~3% above spot, PE ~3% below."""
        ce_strike = cls._round50(spot * (1 + otm_pct))
        pe_strike = cls._round50(spot * (1 - otm_pct))
        ce_price  = cls._chain_price(df_chain, ce_strike, 'CE')
        pe_price  = cls._chain_price(df_chain, pe_strike, 'PE')
        ce_iv     = cls._chain_iv(df_chain, ce_strike, 'CE', iv_dec)
        pe_iv     = cls._chain_iv(df_chain, pe_strike, 'PE', iv_dec)
        legs = [
            OptionLeg('CE', 'SELL', ce_strike, ce_price, ce_iv, lots, expiry),
            OptionLeg('PE', 'SELL', pe_strike, pe_price, pe_iv, lots, expiry),
        ]
        return Strategy('Short Strangle', legs, 'CREDIT')

    @classmethod
    def iron_condor(cls, spot, T, iv_dec, df_chain, call_wall=0, put_wall=0,
                    expiry='', lots=1, wing_width=100) -> Strategy:
        """
        Iron Condor contained within OI walls (if available).
        Short strikes: EM boundary or wall (whichever is tighter).
        Long strikes: wing_width beyond the short strikes.
        """
        em = spot * iv_dec * np.sqrt(T / (1 / 52)) if T > 0 else spot * 0.03
        short_ce = cls._round50(min(call_wall if call_wall > 0 else spot * 1.04,
                                     spot + em))
        short_pe = cls._round50(max(put_wall if put_wall > 0 else spot * 0.96,
                                     spot - em))
        long_ce  = cls._round50(short_ce + wing_width)
        long_pe  = cls._round50(short_pe - wing_width)

        legs = [
            OptionLeg('CE', 'SELL', short_ce, cls._chain_price(df_chain, short_ce, 'CE'),
                      cls._chain_iv(df_chain, short_ce, 'CE', iv_dec), lots, expiry),
            OptionLeg('CE', 'BUY',  long_ce,  cls._chain_price(df_chain, long_ce,  'CE'),
                      cls._chain_iv(df_chain, long_ce,  'CE', iv_dec), lots, expiry),
            OptionLeg('PE', 'SELL', short_pe, cls._chain_price(df_chain, short_pe, 'PE'),
                      cls._chain_iv(df_chain, short_pe, 'PE', iv_dec), lots, expiry),
            OptionLeg('PE', 'BUY',  long_pe,  cls._chain_price(df_chain, long_pe,  'PE'),
                      cls._chain_iv(df_chain, long_pe,  'PE', iv_dec), lots, expiry),
        ]
        return Strategy('Iron Condor', legs, 'CREDIT')

    @classmethod
    def bull_put_spread(cls, spot, T, iv_dec, df_chain, put_wall=0,
                        expiry='', lots=1, spread=100) -> Strategy:
        """Credit spread: sell higher PE, buy lower PE."""
        sell_pe = cls._round50(max(put_wall * 0.98 if put_wall > 0 else spot * 0.97,
                                    spot - spot * 0.03))
        buy_pe  = cls._round50(sell_pe - spread)
        legs = [
            OptionLeg('PE', 'SELL', sell_pe, cls._chain_price(df_chain, sell_pe, 'PE'),
                      cls._chain_iv(df_chain, sell_pe, 'PE', iv_dec), lots, expiry),
            OptionLeg('PE', 'BUY',  buy_pe,  cls._chain_price(df_chain, buy_pe,  'PE'),
                      cls._chain_iv(df_chain, buy_pe,  'PE', iv_dec), lots, expiry),
        ]
        return Strategy('Bull Put Spread', legs, 'CREDIT')

    @classmethod
    def bear_call_spread(cls, spot, T, iv_dec, df_chain, call_wall=0,
                         expiry='', lots=1, spread=100) -> Strategy:
        """Credit spread: sell lower CE, buy higher CE."""
        sell_ce = cls._round50(min(call_wall * 1.02 if call_wall > 0 else spot * 1.03,
                                    spot + spot * 0.03))
        buy_ce  = cls._round50(sell_ce + spread)
        legs = [
            OptionLeg('CE', 'SELL', sell_ce, cls._chain_price(df_chain, sell_ce, 'CE'),
                      cls._chain_iv(df_chain, sell_ce, 'CE', iv_dec), lots, expiry),
            OptionLeg('CE', 'BUY',  buy_ce,  cls._chain_price(df_chain, buy_ce,  'CE'),
                      cls._chain_iv(df_chain, buy_ce,  'CE', iv_dec), lots, expiry),
        ]
        return Strategy('Bear Call Spread', legs, 'CREDIT')

    @classmethod
    def long_straddle(cls, spot, atm_strike, T, iv_dec, ce_price, pe_price,
                      expiry='', lots=1) -> Strategy:
        legs = [
            OptionLeg('CE', 'BUY', atm_strike, ce_price, iv_dec, lots, expiry),
            OptionLeg('PE', 'BUY', atm_strike, pe_price, iv_dec, lots, expiry),
        ]
        return Strategy('Long Straddle', legs, 'DEBIT')

    @classmethod
    def calendar_spread(cls, atm_strike, near_price, far_price, near_iv, far_iv,
                        near_expiry='', far_expiry='', lots=1) -> Strategy:
        """Buy far expiry, sell near expiry (same strike)."""
        legs = [
            OptionLeg('CE', 'SELL', atm_strike, near_price, near_iv, lots, near_expiry),
            OptionLeg('CE', 'BUY',  atm_strike, far_price,  far_iv,  lots, far_expiry),
        ]
        return Strategy('Calendar Spread (CE)', legs, 'DEBIT')

    # ── Chain lookup helpers ──────────────────────────────
    @staticmethod
    def _chain_price(df_chain, strike: float, opt_type: str,
                     fallback: float = 1.0) -> float:
        if df_chain is None or df_chain.empty:
            return fallback
        row = df_chain[(df_chain['strike'] == strike) & (df_chain['type'] == opt_type)]
        if row.empty:
            # Nearest available strike
            sub = df_chain[df_chain['type'] == opt_type].copy()
            if sub.empty:
                return fallback
            idx = (sub['strike'] - strike).abs().idxmin()
            return float(sub.loc[idx, 'price'])
        return float(row.iloc[0]['price'])

    @staticmethod
    def _chain_iv(df_chain, strike: float, opt_type: str,
                  fallback_dec: float = 0.15) -> float:
        if df_chain is None or df_chain.empty:
            return fallback_dec
        row = df_chain[(df_chain['strike'] == strike) & (df_chain['type'] == opt_type)]
        if row.empty:
            return fallback_dec
        iv = float(row.iloc[0].get('iv', 0))
        return (iv / 100.0) if iv > 1 else (iv if iv > 0 else fallback_dec)


# ──────────────────────────────────────────────────────────────
# Smart Strategy Generator  — context-aware selection + scoring
# ──────────────────────────────────────────────────────────────
class SmartStrategyGenerator:
    """
    Generates and scores strategies given live market context.
    Compatible with the existing VolatilityAnalyzer integration pattern.
    """

    def __init__(self, spot: float, df_chain, market_context: dict, expiry: str = ''):
        self.spot    = spot
        self.chain   = df_chain
        self.ctx     = market_context
        self.expiry  = expiry

        self.T       = market_context.get('T', 7 / 365)
        self.iv_dec  = market_context.get('iv', 15) / 100
        self.vrp     = market_context.get('vrp', 0)
        self.regime  = market_context.get('regime', 'NORMAL')
        self.call_wall = market_context.get('call_wall', 0)
        self.put_wall  = market_context.get('put_wall', 0)
        self.atm_iv  = market_context.get('atm_iv', market_context.get('iv', 15))
        self.straddle = market_context.get('em', spot * self.iv_dec * np.sqrt(self.T))

        # ATM strike + prices
        self.atm_strike = self._find_atm()
        self.atm_ce_price = StrategyBuilder._chain_price(df_chain, self.atm_strike, 'CE', self.straddle / 2)
        self.atm_pe_price = StrategyBuilder._chain_price(df_chain, self.atm_strike, 'PE', self.straddle / 2)

    def _find_atm(self) -> float:
        if self.chain is None or self.chain.empty:
            return StrategyBuilder._round50(self.spot)
        strikes = self.chain['strike'].unique()
        return float(min(strikes, key=lambda x: abs(x - self.spot)))

    def generate(self) -> List[Strategy]:
        strats = []
        B = StrategyBuilder

        # 1. Short Straddle — best when regime = COMPRESSION + high VRP
        s = B.short_straddle(self.spot, self.atm_strike, self.T, self.iv_dec,
                              self.atm_ce_price, self.atm_pe_price, self.expiry)
        s.score = self._score_straddle(s)
        s.to_dict()['reasoning'] = self._straddle_reason()
        strats.append(s)

        # 2. Short Strangle
        s2 = B.short_strangle(self.spot, self.T, self.iv_dec, self.chain,
                               self.expiry)
        s2.score = self._score_strangle(s2)
        strats.append(s2)

        # 3. Iron Condor (always generated — safest structure)
        ic = B.iron_condor(self.spot, self.T, self.iv_dec, self.chain,
                            self.call_wall, self.put_wall, self.expiry)
        ic.score = self._score_ic(ic)
        strats.append(ic)

        # 4. Bull Put Spread (bullish / OI pressure bullish)
        bps = B.bull_put_spread(self.spot, self.T, self.iv_dec, self.chain,
                                 self.put_wall, self.expiry)
        bps.score = self._score_directional(bps, 'BULLISH')
        strats.append(bps)

        # 5. Bear Call Spread (bearish / OI pressure bearish)
        bcs = B.bear_call_spread(self.spot, self.T, self.iv_dec, self.chain,
                                  self.call_wall, self.expiry)
        bcs.score = self._score_directional(bcs, 'BEARISH')
        strats.append(bcs)

        # Sort by score descending
        strats.sort(key=lambda x: x.score, reverse=True)
        return strats

    # ── Scoring ──────────────────────────────────────────────
    def _base_score(self) -> int:
        score = 50
        if self.vrp > 3:     score += 20
        elif self.vrp < -2:  score -= 25
        if self.regime in ('COMPRESSION', 'MEAN_REVERSION'): score += 15
        elif self.regime == 'EXPANSION':                      score -= 20
        return score

    def _score_straddle(self, s: Strategy) -> int:
        base = self._base_score()
        # Straddle riskier — penalize in expansion
        if self.regime == 'EXPANSION': base -= 15
        if s.net_premium > 0: base += min(10, int(s.net_premium / 10))
        return max(5, min(100, base))

    def _score_strangle(self, s: Strategy) -> int:
        base = self._base_score() + 5   # slightly safer than straddle
        if self.call_wall > 0 and self.put_wall > 0: base += 10
        return max(5, min(100, base))

    def _score_ic(self, ic: Strategy) -> int:
        base = self._base_score() + 10  # capped risk — always a bonus
        if self.call_wall > 0 and self.put_wall > 0: base += 15
        return max(5, min(100, base))

    def _score_directional(self, s: Strategy, bias: str) -> int:
        oi_pressure = self.ctx.get('oi_pressure', 'NEUTRAL')
        base = 45
        if oi_pressure == bias: base += 25
        if self.vrp > 2: base += 10
        return max(5, min(100, base))

    def _straddle_reason(self) -> str:
        parts = []
        if self.vrp > 0:
            parts.append(f'VRP={self.vrp:+.1f}% (IV rich)')
        if self.regime in ('COMPRESSION', 'MEAN_REVERSION'):
            parts.append(f'Regime: {self.regime}')
        parts.append(f'ATM IV: {self.atm_iv:.1f}%')
        return ' | '.join(parts) if parts else 'Standard market conditions'


# ──────────────────────────────────────────────────────────────
# PayoffEngine  — Plotly visualization
# ──────────────────────────────────────────────────────────────
class PayoffEngine:
    """
    Builds Plotly payoff diagrams and HTML greeks tables for Strategy objects.
    """

    def __init__(self, strategy: Strategy, spot: float, T: float,
                 sigma: float, r: float = RISK_FREE,
                 call_wall: float = 0, put_wall: float = 0):
        self.strategy  = strategy
        self.spot      = spot
        self.T         = T
        self.sigma     = sigma
        self.r         = r
        self.call_wall = call_wall
        self.put_wall  = put_wall

        # Price range: ±15% around spot
        lo = spot * 0.85
        hi = spot * 1.15
        self.price_range = np.linspace(lo, hi, 1000)

    def build_payoff_chart(self) -> go.Figure:
        strat = self.strategy
        prices = self.price_range

        pnl_expiry = strat.payoff_at_expiry(prices) * NIFTY_LOT
        pnl_today  = strat.payoff_now_bsm(prices, self.T, self.r) * NIFTY_LOT

        bes = strat.breakevens(prices)
        max_p = strat.max_profit(prices) * NIFTY_LOT
        max_l = strat.max_loss(prices) * NIFTY_LOT

        fig = go.Figure()

        # OI Wall shading (safe zone between walls)
        if self.put_wall > 0 and self.call_wall > 0:
            fig.add_vrect(
                x0=self.put_wall, x1=self.call_wall,
                fillcolor='rgba(102,187,106,0.07)',
                layer='below', line_width=0,
                annotation_text='OI Safe Zone',
                annotation_position='top right',
                annotation_font=dict(color=MUTED, size=9)
            )

        # Zero line
        fig.add_hline(y=0, line=dict(color=MUTED, width=1, dash='dot'))

        # Profit/loss fill zones for expiry P&L
        prof_mask = pnl_expiry >= 0
        loss_mask = pnl_expiry < 0
        for mask, fill_color in [(prof_mask, 'rgba(102,187,106,0.12)'),
                                  (loss_mask, 'rgba(255,68,68,0.10)')]:
            x_seg = np.where(mask, prices, np.nan)
            y_seg = np.where(mask, pnl_expiry, np.nan)
            fig.add_trace(go.Scatter(
                x=x_seg, y=y_seg, fill='tozeroy',
                fillcolor=fill_color, line=dict(width=0),
                hoverinfo='skip', showlegend=False
            ))

        # Expiry P&L line
        fig.add_trace(go.Scatter(
            x=prices, y=pnl_expiry, name='Expiry P&L',
            line=dict(color=GREEN if strat.strategy_type == 'CREDIT' else ACCENT, width=2.5),
            hovertemplate='Spot: %{x:.0f}<br>P&L: ₹%{y:,.0f}<extra></extra>'
        ))

        # Today's BSM line
        fig.add_trace(go.Scatter(
            x=prices, y=pnl_today, name="Today's P&L (BSM)",
            line=dict(color=YELLOW, width=1.5, dash='dash'),
            hovertemplate='Spot: %{x:.0f}<br>P&L: ₹%{y:,.0f}<extra></extra>'
        ))

        # Current spot marker
        spot_idx = np.argmin(np.abs(prices - self.spot))
        fig.add_vline(
            x=self.spot,
            line=dict(color=ACCENT, width=1.5, dash='dash'),
            annotation_text=f'Spot {self.spot:,.0f}',
            annotation_position='top',
            annotation_font=dict(color=ACCENT, size=10)
        )

        # Breakeven markers
        for be in bes:
            fig.add_vline(
                x=be,
                line=dict(color=ORANGE, width=1, dash='dot'),
                annotation_text=f'BE {be:,.0f}',
                annotation_position='bottom',
                annotation_font=dict(color=ORANGE, size=9)
            )

        # Max profit / max loss annotation
        fig.add_annotation(
            x=prices[np.argmax(pnl_expiry)], y=max_p,
            text=f'Max Profit ₹{max_p:,.0f}',
            showarrow=True, arrowhead=2,
            font=dict(color=GREEN, size=10), bgcolor='rgba(0,0,0,0.5)',
            bordercolor=GREEN
        )
        if max_l < -10:
            fig.add_annotation(
                x=prices[np.argmin(pnl_expiry)], y=max_l,
                text=f'Max Loss ₹{abs(max_l):,.0f}',
                showarrow=True, arrowhead=2,
                font=dict(color=RED, size=10), bgcolor='rgba(0,0,0,0.5)',
                bordercolor=RED
            )

        fig.update_layout(
            title=dict(text=f'{strat.name} — Payoff Diagram', font=dict(color=WHITE, size=15)),
            paper_bgcolor=DARK_BG, plot_bgcolor='rgba(20,20,35,0.9)',
            font=dict(color=WHITE, family='Inter, sans-serif', size=11),
            legend=dict(bgcolor='rgba(30,30,50,0.85)', font=dict(size=11),
                        x=0.01, y=0.99, bordercolor='#333'),
            hovermode='x unified',
            margin=dict(l=60, r=30, t=55, b=50),
            height=380,
            xaxis=dict(title='NIFTY Spot', gridcolor='rgba(100,100,100,0.12)',
                       tickformat=',.0f'),
            yaxis=dict(title='P&L (₹ per lot)', gridcolor='rgba(100,100,100,0.12)',
                       zeroline=True, zerolinecolor='#444', tickformat=',.0f'),
        )

        return fig

    def build_greeks_html(self) -> str:
        """Returns an HTML string of a per-leg + net greeks table."""
        per_leg = self.strategy.per_leg_greeks(self.spot, self.T, self.r)
        net     = self.strategy.net_greeks(self.spot, self.T, self.r)

        def _rc(v, good_neg=False):
            if abs(v) < 0.005: return MUTED
            if good_neg:
                return GREEN if v > 0 else RED
            return RED if v > 0 else GREEN

        rows = ''
        for lg in per_leg:
            rows += (
                f'<tr>'
                f'<td style="color:{"#4fc3f7" if lg["type"]=="CE" else "#ef9a9a"};">'
                f'{lg["action"]} {lg["type"]} {lg["strike"]:.0f}</td>'
                f'<td>₹{lg["entry_price"]:.1f}</td>'
                f'<td style="color:{_rc(lg["delta"])};">{lg["delta"]:+.3f}</td>'
                f'<td style="color:{MUTED};">{lg["gamma"]:.4f}</td>'
                f'<td style="color:{GREEN if lg["theta"] > 0 else RED};">{lg["theta"]:+.2f}</td>'
                f'<td style="color:{_rc(lg["vega"], good_neg=True)};">{lg["vega"]:+.2f}</td>'
                f'</tr>'
            )

        net_row = (
            f'<tr style="border-top:2px solid #444;font-weight:700;">'
            f'<td style="color:{ACCENT};">NET POSITION</td>'
            f'<td style="color:{GREEN if self.strategy.net_premium > 0 else RED};">'
            f'{"+" if self.strategy.net_premium > 0 else ""}₹{self.strategy.net_premium * NIFTY_LOT:.0f}</td>'
            f'<td style="color:{_rc(net["delta"])};">{net["delta"]:+.3f}</td>'
            f'<td style="color:{MUTED};">{net["gamma"]:.4f}</td>'
            f'<td style="color:{GREEN if net["theta"] > 0 else RED};">{net["theta"]:+.2f}</td>'
            f'<td style="color:{_rc(net["vega"], good_neg=True)};">{net["vega"]:+.2f}</td>'
            f'</tr>'
        )

        return f'''
        <table style="width:100%;border-collapse:collapse;font-size:12px;">
            <thead>
                <tr style="color:{MUTED};border-bottom:1px solid #333;">
                    <th style="text-align:left;padding:5px 8px;">Leg</th>
                    <th style="padding:5px 8px;">Premium</th>
                    <th style="padding:5px 8px;">Δ Delta</th>
                    <th style="padding:5px 8px;">Γ Gamma</th>
                    <th style="padding:5px 8px;">θ Theta/d</th>
                    <th style="padding:5px 8px;">ν Vega/%</th>
                </tr>
            </thead>
            <tbody>{rows}{net_row}</tbody>
        </table>'''

    def risk_reward_html(self) -> str:
        """Returns a small HTML snippet with max profit/loss, POP, breakevens."""
        prices  = self.price_range
        max_p   = self.strategy.max_profit(prices) * NIFTY_LOT
        max_l   = abs(self.strategy.max_loss(prices)) * NIFTY_LOT
        bes     = self.strategy.breakevens(prices)
        pop     = self.strategy.pop(self.spot, self.T, self.sigma) * 100
        net_prem = self.strategy.net_premium * NIFTY_LOT
        rr       = (max_p / max_l) if max_l > 0 else float('inf')

        pop_color = GREEN if pop >= 65 else YELLOW if pop >= 50 else RED
        prem_color = GREEN if net_prem > 0 else RED
        bes_str = ' / '.join(f'{b:,.0f}' for b in bes) if bes else 'N/A'

        rr_str  = f'1 : {rr:.2f}' if rr < 99 else '∞ (capped)'

        return f'''
        <div style="display:flex;flex-wrap:wrap;gap:10px;margin-top:10px;">
            <div style="background:{pop_color}1a;border:1px solid {pop_color}55;
                        border-radius:8px;padding:10px 18px;text-align:center;flex:1;min-width:120px;">
                <div style="color:{MUTED};font-size:10px;font-weight:700;text-transform:uppercase;margin-bottom:4px;">
                    P(Profit) at Expiry</div>
                <div style="font-size:26px;font-weight:900;color:{pop_color};">{pop:.1f}%</div>
            </div>
            <div class="metric-box" style="flex:1;min-width:110px;">
                <div class="metric-label">Net Premium</div>
                <div style="font-size:18px;font-weight:700;color:{prem_color};">
                    {"+" if net_prem > 0 else ""}₹{net_prem:.0f}</div>
                <div class="metric-sub">per lot</div>
            </div>
            <div class="metric-box" style="flex:1;min-width:110px;">
                <div class="metric-label">Max Profit</div>
                <div style="font-size:18px;font-weight:700;color:{GREEN};">
                    {"∞" if max_p > 1e6 else f"₹{max_p:,.0f}"}</div>
            </div>
            <div class="metric-box" style="flex:1;min-width:110px;">
                <div class="metric-label">Max Loss</div>
                <div style="font-size:18px;font-weight:700;color:{RED};">
                    {"∞" if max_l > 1e6 else f"₹{max_l:,.0f}"}</div>
            </div>
            <div class="metric-box" style="flex:1;min-width:110px;">
                <div class="metric-label">Risk : Reward</div>
                <div style="font-size:18px;font-weight:700;color:{WHITE};">{rr_str}</div>
            </div>
            <div class="metric-box" style="flex:1;min-width:130px;">
                <div class="metric-label">Breakeven(s)</div>
                <div style="font-size:14px;font-weight:700;color:{ORANGE};">{bes_str}</div>
            </div>
        </div>'''


# ──────────────────────────────────────────────────────────────
# Convenience — build_strategy_html (called from TAB 5)
# ──────────────────────────────────────────────────────────────
def build_strategy_card_html(strategy: Strategy, spot: float, T: float,
                              sigma: float, call_wall: float = 0,
                              put_wall: float = 0, payload_json: str = '') -> str:
    """
    Returns the full HTML card for one strategy:
    payoff chart + risk-reward row + greeks table.
    Used by VolatilityAnalyzer TAB 5.
    """
    engine = PayoffEngine(strategy, spot, T, sigma, call_wall=call_wall, put_wall=put_wall)
    chart_html = engine.build_payoff_chart().to_html(include_plotlyjs=False, full_html=False)
    rr_html    = engine.risk_reward_html()
    greek_html = engine.build_greeks_html()

    score_color = GREEN if strategy.score >= 70 else YELLOW if strategy.score >= 45 else RED
    prem_txt    = (f'+₹{strategy.net_premium * NIFTY_LOT:.0f} credit'
                   if strategy.net_premium > 0 else
                   f'-₹{abs(strategy.net_premium * NIFTY_LOT):.0f} debit')

    return f'''
    <div style="background:#12122a;border:1px solid #2a2a4a;border-radius:10px;
                padding:14px;margin-bottom:14px;" id="strat-{hash(strategy.name) % 10000}">
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;">
            <div>
                <span style="font-size:16px;font-weight:800;color:{ACCENT};">{strategy.name}</span>
                <span style="font-size:12px;color:{score_color};margin-left:10px;
                             background:{score_color}18;padding:2px 10px;border-radius:10px;
                             border:1px solid {score_color}44;">Score {strategy.score}/100</span>
            </div>
            <div style="display:flex;gap:8px;align-items:center;">
                <span style="font-size:13px;color:{"#66bb6a" if "credit" in prem_txt else "#ff7043"};font-weight:700;">{prem_txt}</span>
                <button onclick="trackStrategy('{payload_json}')"
                        style="background:{GREEN};color:#000;border:none;padding:5px 14px;
                               border-radius:6px;font-weight:700;cursor:pointer;font-size:11px;">
                    Track Live
                </button>
            </div>
        </div>
        {rr_html}
        <div style="margin-top:12px;">{chart_html}</div>
        <details style="margin-top:8px;">
            <summary style="color:{MUTED};font-size:11px;cursor:pointer;padding:4px 0;">
                ▶ Greeks Breakdown (per lot)
            </summary>
            <div style="margin-top:8px;">{greek_html}</div>
        </details>
    </div>'''
