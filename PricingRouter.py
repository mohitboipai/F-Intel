"""
PricingRouter.py
================
Intelligent model selection layer that routes each option leg to either
Black-Scholes (BSM) or the Heston semi-analytical pricer, depending on
market conditions.

Decision tree (see _select_model for rationale):
  - Far OTM (|moneyness| > 3%)    → HESTON  (smile)
  - High vol regime (score > 60)  → HESTON  (fat tails)
  - Strong term structure (> 2.5) → HESTON  (calendar spread)
  - Steep skew (put/atm_iv > 1.2) → HESTON  (vol surface non-flat)
  - Everything else               → BSM     (speed)

Heston pricing uses HestonMath.price_vanilla_call() (Fourier transform, fast).
Heston POP uses heston_paths() MC (slow but accurate).
BSM fallback is always available — calibration failure never propagates.

Constants: LOT_SIZE=75, RISK_FREE=0.07
"""

import time
import numpy as np
from typing import Tuple, Optional

# ── Existing BSM utilities ────────────────────────────────────────────────────
from StrategyEngine import bsm_price, RISK_FREE, NIFTY_LOT

# ── Heston math / MC  ────────────────────────────────────────────────────────
from NiftyHestonMC import HestonMath, NiftyHestonMC

LOT_SIZE = NIFTY_LOT   # 75 — canonical reference


# ──────────────────────────────────────────────────────────────────────────────
# HestonParamCache  — 5-minute TTL wrapper around calibrated params
# ──────────────────────────────────────────────────────────────────────────────
class HestonParamCache:
    """
    Caches calibrated Heston parameters for 5 minutes so per-tick pricing
    does not re-trigger the 10-20s calibration optimisation.
    """
    TTL_SECONDS = 300

    def __init__(self):
        self._params: dict | None = None
        self._ts: float = 0.0

    def get(self) -> dict | None:
        """Return cached params, or None if TTL expired."""
        if self._params is None:
            return None
        return self._params if (time.time() - self._ts) < self.TTL_SECONDS else None

    def set(self, params: dict):
        """Store new params and reset TTL."""
        self._params = params
        self._ts = time.time()


# ──────────────────────────────────────────────────────────────────────────────
# PricingRouter
# ──────────────────────────────────────────────────────────────────────────────
class PricingRouter:
    """
    Drop-in replacement / extension for bsm_price() / Strategy.pop() that
    routes to HESTON when market conditions warrant it.

    Usage:
        router = PricingRouter()
        p = router.price(S, K, T, r, iv, 'CE', context=ctx)
        pop = router.pop(strategy, spot, T, sigma, context=ctx)
    """

    def __init__(self):
        self._cache = HestonParamCache()
        # Lazy NiftyHestonMC instance — only created when MC paths are needed.
        # We instantiate with fyers=None to skip OAuth entirely; heston_paths()
        # is a pure numerical method that doesn't need an API connection.
        self._mc: NiftyHestonMC | None = None

    # ── Public interface ──────────────────────────────────────────────────────

    def price(self, S: float, K: float, T: float, r: float,
              iv: float, opt_type: str, context: dict | None = None) -> float:
        """
        Price a single European option.
        Uses HestonMath.price_vanilla_call() when HESTON is selected,
        bsm_price() otherwise. PE uses put-call parity when HESTON.
        Falls back to BSM silently on any Heston error.
        """
        model = self._select_model(context)
        if model == 'BSM' or T <= 0:
            return bsm_price(S, K, T, r, iv, opt_type)

        params = self._resolve_heston_params(context, S, T, r)
        if params is None:
            return bsm_price(S, K, T, r, iv, opt_type)

        try:
            ce_price = HestonMath.price_vanilla_call(
                S, K, T, r,
                params['kappa'], params['theta'], params['v0'],
                params['rho'],   params['xi']
            )
            ce_price = max(0.0, ce_price)
            if opt_type == 'CE':
                return ce_price
            # PE via put-call parity: PE = CE - S + K*exp(-rT)
            pe_price = ce_price - S + K * np.exp(-r * T)
            return max(0.0, pe_price)
        except Exception:
            return bsm_price(S, K, T, r, iv, opt_type)

    def pop(self, strategy, spot: float, T: float, sigma: float,
            r: float = RISK_FREE, context: dict | None = None) -> float:
        """
        Probability of Profit using Heston terminal distribution when context
        warrants it; otherwise delegates to strategy's built-in BSM pop().
        """
        model = self._select_model(context)
        if model == 'BSM' or context is None:
            return strategy.pop(spot, T, sigma, r)

        params = self._resolve_heston_params(context, spot, T, r)
        if params is None:
            return strategy.pop(spot, T, sigma, r)

        try:
            prices, weights = self._heston_terminal_pdf(spot, T, params)
            pnl = strategy.payoff_at_expiry(prices)
            return float(np.sum(weights * (pnl > 0)))
        except Exception:
            return strategy.pop(spot, T, sigma, r)

    def terminal_pdf(self, spot: float, T: float, sigma: float,
                     context: dict | None = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return (prices, weights) for plotting terminal distribution.
        Uses Heston MC when context warrants it, BSM log-normal otherwise.
        """
        model = self._select_model(context)
        if model == 'HESTON' and context is not None:
            params = self._resolve_heston_params(context, spot, T, RISK_FREE)
            if params is not None:
                try:
                    return self._heston_terminal_pdf(spot, T, params)
                except Exception:
                    pass
        return self._bsm_terminal_pdf(spot, T, sigma)

    # ── Model selection ───────────────────────────────────────────────────────

    def _select_model(self, context: dict | None) -> str:
        """
        Returns 'BSM' or 'HESTON' based on market conditions.

        Rule 1: Far OTM — smile matters (BSM underprices far OTM puts)
        Rule 2: High vol regime — fat tails that BSM normal misses
        Rule 3: Strong term structure — calendar spread misprice
        Rule 4: Steep skew — vol surface is non-flat
        """
        if context is None:
            return 'BSM'

        mono  = abs(context.get('moneyness', 0.0))
        expl  = context.get('explosion_score', 0.0)
        term  = abs(context.get('term_spread', 0.0))
        skew  = context.get('skew_ratio', 1.0)

        if mono > 0.03:     return 'HESTON'   # Rule 1: Far OTM
        if expl > 60:       return 'HESTON'   # Rule 2: High vol regime
        if term > 2.5:      return 'HESTON'   # Rule 3: Calendar play
        if skew > 1.20:     return 'HESTON'   # Rule 4: Steep skew

        return 'BSM'

    # ── Heston helpers ────────────────────────────────────────────────────────

    def _resolve_heston_params(self, context: dict, spot: float,
                                T: float, r: float) -> dict | None:
        """
        Return Heston params from (in priority order):
          1. context['heston_params'] — pre-calibrated params passed in
          2. local HestonParamCache  — cached from a previous call
          3. SharedDataCache         — params from background calibrator
          4. Attempt live calibration (rare, only when all caches miss)
        Falls back to None (BSM) if calibration fails for any reason.
        """
        # 1. Context-supplied params (highest priority)
        if context and context.get('heston_params'):
            return context['heston_params']

        # 2. Local cache
        cached = self._cache.get()
        if cached:
            return cached

        # 3. SharedDataCache (written by background HestonCalibrator)
        try:
            from SharedDataCache import SharedDataCache
            # SharedDataCache is a singleton-ish — get global instance if available
            sc = _get_shared_cache()
            if sc:
                p = sc.get_heston_params()
                if p:
                    self._cache.set(p)   # promote to local cache
                    return p
        except Exception:
            pass

        # 4. Last resort: run calibration inline (slow, best-effort)
        try:
            return self._calibrate_inline(spot, T, r)
        except Exception:
            return None

    def _calibrate_inline(self, spot: float, T: float, r: float) -> dict | None:
        """Attempt quick calibration from SharedDataCache chain data."""
        try:
            import pandas as pd
            sc = _get_shared_cache()
            if sc is None:
                return None
            raw = sc.get_raw_chain()
            if not raw:
                return None

            mc = self._get_mc()
            df = mc.parse_chain(raw)
            if df.empty:
                return None

            df['dist'] = abs(df['strike'] - spot)
            subset = (df[(df['type'] == 'CE') & (df['dist'] < spot * 0.02)]
                      .sort_values('dist').head(8))
            if len(subset) < 3:
                return None

            params = mc.calibrate_parameters(subset, spot, T, r)
            self._cache.set(params)
            return params
        except Exception:
            return None

    def _get_mc(self) -> NiftyHestonMC:
        """Lazy-create NiftyHestonMC without Fyers auth (pure-math usage)."""
        if self._mc is None:
            # Bypass __init__ authentication by monkey-patching fyers
            mc = object.__new__(NiftyHestonMC)
            mc.fyers = None
            from OptionAnalytics import OptionAnalytics
            mc.analytics = OptionAnalytics()
            mc.symbol = "NSE:NIFTY50-INDEX"
            mc.spot_price = 0
            mc.expiry_date = None
            mc.regimes = {}
            self._mc = mc
        return self._mc

    def _heston_terminal_pdf(self, spot: float, T: float,
                              params: dict, N: int = 4000) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run N Heston MC paths and return (bin_centers, normalised_weights)
        representing the terminal price distribution at expiry.
        """
        mc = self._get_mc()
        steps = max(50, int(T * 252))
        S, _ = mc.heston_paths(spot, T, RISK_FREE, params, steps, N)
        terminal = S[:, -1]      # shape (N,)

        lo = spot * 0.60
        hi = spot * 1.40
        counts, edges = np.histogram(terminal, bins=200,
                                     range=(lo, hi), density=True)
        bin_centers = 0.5 * (edges[:-1] + edges[1:])
        weights = counts / (counts.sum() + 1e-12)
        return bin_centers, weights

    def _bsm_terminal_pdf(self, spot: float, T: float,
                           sigma: float) -> Tuple[np.ndarray, np.ndarray]:
        """BSM log-normal terminal distribution (fast fallback)."""
        prices = np.linspace(spot * 0.60, spot * 1.40, 1000)
        if T <= 0 or sigma <= 0:
            uniform = np.ones(len(prices)) / len(prices)
            return prices, uniform
        mu = np.log(spot) + (RISK_FREE - 0.5 * sigma ** 2) * T
        std = sigma * np.sqrt(T)
        log_p = np.log(prices)
        pdf = np.exp(-0.5 * ((log_p - mu) / std) ** 2) / (std * np.sqrt(2 * np.pi))
        weights = pdf / (pdf.sum() + 1e-12)
        return prices, weights


# ──────────────────────────────────────────────────────────────────────────────
# Module-level SharedDataCache accessor
# ──────────────────────────────────────────────────────────────────────────────
_shared_cache_ref = None   # set by DataServer after hub starts


def register_shared_cache(cache_obj):
    """Called by DataServer to register the shared cache singleton."""
    global _shared_cache_ref
    _shared_cache_ref = cache_obj


def _get_shared_cache():
    """Return the registered SharedDataCache, or None."""
    return _shared_cache_ref
