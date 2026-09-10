import numpy as np
import pandas as pd
from scipy.stats import norm

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    import config as _cfg
    _DEFAULT_R = _cfg.get("risk_free_rate")
    _DEFAULT_Q = _cfg.get("dividend_yield")
    _DEFAULT_DTE_THRESH = _cfg.get("dividend_dte_threshold")
except Exception:
    _DEFAULT_R = 0.051274   # RBI 91-day T-bill Sep 2026, continuously compounded
    _DEFAULT_Q = 0.0122     # NIFTY50 trailing dividend yield Jul 2026
    _DEFAULT_DTE_THRESH = 7

class GreeksEngine:
    """
    Mathematical, vectorized engine for computing 1st and 2nd order options Greeks.
    Built for institutional analytics; strict separation from visualization.

    Dividend yield (q) is modelled via the Merton (1973) continuous-dividend
    BSM extension:
        d1 = [ln(S/K) + (r - q + 0.5·σ²)·T] / (σ·√T)
        Delta_CE = e^{-qT} · N(d1)
        Gamma    = e^{-qT} · n(d1) / (S·σ·√T)
        Vega     = e^{-qT} · S·n(d1)·√T / 100

    For DTE ≤ dividend_dte_threshold (default 7 days), q is forced to 0
    because the dividend impact is negligible for short-dated options and
    introduces unnecessary complexity.
    """
    def __init__(self,
                 risk_free_rate:          float | None = None,
                 days_in_year:            float = 365.0,
                 dividend_yield:          float | None = None,
                 dividend_dte_threshold:  int | None   = None):
        self.r = risk_free_rate if risk_free_rate is not None else _DEFAULT_R
        self.q = dividend_yield if dividend_yield is not None else _DEFAULT_Q
        self.days_in_year = days_in_year
        self.div_dte_threshold = (
            dividend_dte_threshold
            if dividend_dte_threshold is not None
            else _DEFAULT_DTE_THRESH
        )

    def calculate_all_greeks(self,
                             S: float,
                             K: np.ndarray,
                             T_days: np.ndarray,
                             iv: np.ndarray,
                             option_types: np.ndarray) -> pd.DataFrame:
        """
        Calculates a comprehensive suite of Black-Scholes Greeks with Merton (1973)
        continuous dividend yield.

        :param S: Spot price
        :param K: Array of strike prices
        :param T_days: Array of Days to Expiry (calendar days)
        :param iv: Array of Implied Volatilities (decimal)
        :param option_types: Array of 'CE' or 'PE' strings
        :return: DataFrame containing delta, gamma, vega, theta, vanna, charm, vomma
        """
        # Ensure mathematical boundaries to prevent NaN or division by zero
        T = np.maximum(T_days / self.days_in_year, 1e-5)
        iv = np.maximum(iv, 1e-5)
        K = np.maximum(K, 1e-5)
        S = max(S, 1e-5)

        # Gate dividend: set q=0 for short-dated options (DTE ≤ threshold)
        # Use a per-row mask so mixed-expiry batches are handled correctly
        q_vec = np.where(T_days > self.div_dte_threshold, self.q, 0.0)

        # Identify Calls vs Puts — use pd.Series for robust string handling
        # (np.char.upper fails on non-string dtypes; pd.Series.str handles object/categorical)
        opt_series = pd.Series(option_types).astype(str).str.upper()
        is_call = opt_series.values == 'CE'

        # Core d1 / d2 — Merton (1973) dividend-adjusted
        sqrt_T = np.sqrt(T)
        d1 = (np.log(S / K) + (self.r - q_vec + 0.5 * iv ** 2) * T) / (iv * sqrt_T)
        d2 = d1 - iv * sqrt_T

        # e^{-qT} factor — collapses to 1 when q_vec=0 (short-dated)
        eq_T = np.exp(-q_vec * T)

        # Precompute Normals
        N_d1       = norm.cdf(d1)
        N_d2       = norm.cdf(d2)
        N_minus_d1 = norm.cdf(-d1)
        N_minus_d2 = norm.cdf(-d2)
        n_d1       = norm.pdf(d1)   # Standard normal PDF

        # ── First Order Greeks ─────────────────────────────────────────────────

        # Delta (Merton: multiply by e^{-qT})
        delta = np.where(is_call, eq_T * N_d1, eq_T * (N_d1 - 1.0))

        # Theta (decay per 1 calendar day)
        # CE: -(S·e^{-qT}·n(d1)·σ)/(2√T) + q·S·e^{-qT}·N(d1) - r·K·e^{-rT}·N(d2)
        # PE: -(S·e^{-qT}·n(d1)·σ)/(2√T) - q·S·e^{-qT}·N(-d1) + r·K·e^{-rT}·N(-d2)
        er_T = np.exp(-self.r * T)
        theta_common = -(S * eq_T * n_d1 * iv) / (2 * sqrt_T)
        theta_call = (theta_common
                      + q_vec * S * eq_T * N_d1
                      - self.r * K * er_T * N_d2)
        theta_put  = (theta_common
                      - q_vec * S * eq_T * N_minus_d1
                      + self.r * K * er_T * N_minus_d2)
        theta = np.where(is_call, theta_call, theta_put) / self.days_in_year

        # Vega (per 1% change in IV) — scaled by e^{-qT}
        vega = (S * eq_T * n_d1 * sqrt_T) / 100.0

        # Rho (per 1% change in interest rate)
        rho_call = K * T * er_T * N_d2
        rho_put  = -K * T * er_T * N_minus_d2
        rho = np.where(is_call, rho_call, rho_put) / 100.0

        # ── Second Order Greeks ────────────────────────────────────────────────

        # Gamma — scaled by e^{-qT}
        gamma = eq_T * n_d1 / (S * iv * sqrt_T)

        # Vanna (sensitivity of Delta to IV, or Vega to Spot)
        # Vanna = -e^{-qT} · n(d1) · (d2 / σ) / 100
        vanna = (-eq_T * n_d1 * d2 / iv) / 100.0

        # Charm (Delta decay — sensitivity of Delta to time)
        # CE Charm: e^{-qT} · [q·N(d1) - n(d1)·(r−q)/(σ√T) + n(d1)·d2/(2T)]  (per day)
        # PE Charm: e^{-qT} · [-q·N(-d1) - n(d1)·(r−q)/(σ√T) + n(d1)·d2/(2T)] (per day)
        charm_inner = n_d1 * (2 * (self.r - q_vec) * T - d2 * iv * sqrt_T) / (2 * T * iv * sqrt_T)
        charm_call = eq_T * (-q_vec * N_d1       + charm_inner)
        charm_put  = eq_T * ( q_vec * N_minus_d1 + charm_inner)
        charm = np.where(is_call, charm_call, charm_put) / self.days_in_year

        # Vomma (sensitivity of Vega to IV)
        # Vomma = Vega · (d1 · d2 / σ) / 100
        vomma = vega * (d1 * d2 / iv) / 100.0

        return pd.DataFrame({
            'delta': delta,
            'gamma': gamma,
            'theta': theta,
            'vega':  vega,
            'rho':   rho,
            'vanna': vanna,
            'charm': charm,
            'vomma': vomma
        })
