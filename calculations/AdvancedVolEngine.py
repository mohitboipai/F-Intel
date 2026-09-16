"""
AdvancedVolEngine.py — Advanced Institutional Econometric Volatility Engine
================================================================================
Implements state-of-the-art quantitative volatility modeling from first principles:
1. Realized Semi-Variance (Downside Risk RV- vs Upside Potential RV+)
   - Barndorff-Nielsen, Kinnebrock & Shephard (2010)
   - Volatility Asymmetry Index (VAI) for directional crash hazard vs grind
2. Corsi (2009) HAR-RV Multi-Horizon Volatility Forecasting
   - Heterogeneous Autoregressive model across Daily (1d), Weekly (5d), and Monthly (22d) cascade
   - Out-of-sample forward volatility projections with 95% confidence intervals
3. Bipower Variation (BV) Jump vs Continuous Volatility Decomposition
   - Barndorff-Nielsen & Shephard (2004, 2006)
   - Separates continuous structural volatility from transitory one-off gap/news jumps
4. Higher Return Moments (Realized Skewness & Realized Kurtosis)
5. Predictive Forward Volatility Risk Premium (Forward VRP)
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple


class RealizedSemiVariance:
    """
    Decomposes realized return variance into upside (good vol) and downside (bad vol).
    """

    @staticmethod
    def calculate(returns: np.ndarray, annualize: bool = True, periods_per_year: int = 252) -> Dict[str, float]:
        """
        Calculate RV+, RV-, and Volatility Asymmetry Index (VAI).
        :param returns: Array of periodic log returns
        :param annualize: Whether to annualize the output standard deviations
        :param periods_per_year: Annualization factor (252 for daily, 75*252 for 5m intraday)
        :return: Dict containing total_rv, rv_plus, rv_minus, vai, and regime_tag
        """
        clean_rets = returns[~np.isnan(returns)]
        if len(clean_rets) < 2:
            return {
                "total_rv": 0.0,
                "rv_plus": 0.0,
                "rv_minus": 0.0,
                "rv_plus_pct": 50.0,
                "rv_minus_pct": 50.0,
                "vai": 0.0,
                "bias": "BALANCED",
                "interpretation": "Insufficient return data"
            }

        ann_factor = np.sqrt(periods_per_year) * 100.0 if annualize else 100.0

        pos_rets = np.maximum(clean_rets, 0.0)
        neg_rets = np.minimum(clean_rets, 0.0)

        # Variances
        var_total = np.mean(clean_rets ** 2)
        var_plus = np.mean(pos_rets ** 2)
        var_minus = np.mean(neg_rets ** 2)

        tot_semi = var_plus + var_minus
        if tot_semi > 1e-12:
            rv_plus_pct = float(var_plus / tot_semi * 100.0)
            rv_minus_pct = float(var_minus / tot_semi * 100.0)
            vai = float((var_minus - var_plus) / tot_semi)
        else:
            rv_plus_pct = 50.0
            rv_minus_pct = 50.0
            vai = 0.0

        rv_total = float(np.sqrt(max(var_total, 0.0)) * ann_factor)
        rv_plus = float(np.sqrt(max(var_plus * 2.0, 0.0)) * ann_factor)   # normalized by 2 for semi-sd
        rv_minus = float(np.sqrt(max(var_minus * 2.0, 0.0)) * ann_factor)

        if vai > 0.20:
            bias = "TOXIC_DOWNSIDE"
            interp = f"Downside crash risk dominates (+{rv_minus_pct:.1f}%). High Put hazard; OTM Put IV bid."
        elif vai < -0.20:
            bias = "BULLISH_GRIND"
            interp = f"Upside momentum dominates (+{rv_plus_pct:.1f}%). Low panic; Put writing risk low."
        else:
            bias = "BALANCED"
            interp = "Symmetric volatility dispersion between rallies and pullbacks."

        return {
            "total_rv": round(rv_total, 2),
            "rv_plus": round(rv_plus, 2),
            "rv_minus": round(rv_minus, 2),
            "rv_plus_pct": round(rv_plus_pct, 1),
            "rv_minus_pct": round(rv_minus_pct, 1),
            "vai": round(vai, 3),
            "bias": bias,
            "interpretation": interp
        }


class CorsiHARModel:
    """
    Heterogeneous Autoregressive model of Realized Volatility (Corsi, 2009).
    Models volatility as an additive cascade of Daily, Weekly, and Monthly components.
    """

    def __init__(self, min_obs: int = 40):
        self.min_obs = min_obs
        self.weights = {"intercept": 0.0, "beta_d": 0.35, "beta_w": 0.40, "beta_m": 0.25}
        self.r_squared = 0.0
        self.residual_se = 2.0
        self.is_fitted = False

    def fit(self, daily_rv: np.ndarray) -> bool:
        """
        Fit HAR-RV model on an array of daily realized volatility observations.
        :param daily_rv: 1D array of daily annualized RV values (e.g. from Parkinson or Close-to-Close)
        :return: True if successfully fitted, False otherwise
        """
        rv_series = pd.Series(daily_rv).dropna()
        if len(rv_series) < self.min_obs:
            return False

        # Build predictors: RV_daily (1d lag), RV_weekly (rolling 5d mean), RV_monthly (rolling 22d mean)
        rv_d = rv_series.shift(1)
        rv_w = rv_series.shift(1).rolling(5).mean()
        rv_m = rv_series.shift(1).rolling(22).mean()

        df = pd.DataFrame({"y": rv_series, "rv_d": rv_d, "rv_w": rv_w, "rv_m": rv_m}).dropna()
        if len(df) < 20:
            return False

        y = df["y"].values
        X = np.column_stack([np.ones(len(df)), df["rv_d"].values, df["rv_w"].values, df["rv_m"].values])

        try:
            # OLS via pseudo-inverse
            beta, residuals, rank, s = np.linalg.lstsq(X, y, rcond=None)
            self.weights = {
                "intercept": max(float(beta[0]), 0.0),
                "beta_d": max(float(beta[1]), 0.05),
                "beta_w": max(float(beta[2]), 0.05),
                "beta_m": max(float(beta[3]), 0.05),
            }
            # Normalize sum of betas if explosive
            beta_sum = self.weights["beta_d"] + self.weights["beta_w"] + self.weights["beta_m"]
            if beta_sum > 0.98:
                scale = 0.95 / beta_sum
                self.weights["beta_d"] *= scale
                self.weights["beta_w"] *= scale
                self.weights["beta_m"] *= scale

            y_pred = X @ np.array([self.weights["intercept"], self.weights["beta_d"], self.weights["beta_w"], self.weights["beta_m"]])
            res = y - y_pred
            ss_res = np.sum(res ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            self.r_squared = float(1.0 - (ss_res / (ss_tot + 1e-8)))
            self.residual_se = float(np.std(res))
            self.is_fitted = True
            return True
        except Exception:
            return False

    def forecast(self, current_rv_d: float, current_rv_w: float, current_rv_m: float) -> Dict[str, Any]:
        """
        Forecast forward realized volatility for 1-day and 5-day (weekly expiry) horizons.
        """
        c = self.weights["intercept"]
        bd = self.weights["beta_d"]
        bw = self.weights["beta_w"]
        bm = self.weights["beta_m"]

        # 1-day ahead forward forecast
        forecast_1d = c + bd * current_rv_d + bw * current_rv_w + bm * current_rv_m
        forecast_1d = max(forecast_1d, 5.0)

        # 5-day ahead forward forecast (iterative mean reversion to monthly)
        forecast_5d = 0.60 * forecast_1d + 0.40 * current_rv_m
        forecast_5d = max(forecast_5d, 5.0)

        ci_margin = 1.96 * self.residual_se

        return {
            "forecast_1d": round(float(forecast_1d), 2),
            "forecast_1d_low": round(float(max(forecast_1d - ci_margin, 3.0)), 2),
            "forecast_1d_high": round(float(forecast_1d + ci_margin), 2),
            "forecast_5d": round(float(forecast_5d), 2),
            "forecast_5d_low": round(float(max(forecast_5d - ci_margin * 1.2, 3.0)), 2),
            "forecast_5d_high": round(float(forecast_5d + ci_margin * 1.2), 2),
            "r_squared": round(float(self.r_squared), 3),
            "residual_se": round(float(self.residual_se), 2),
            "weights": {k: round(v, 3) for k, v in self.weights.items()},
            "is_fitted": self.is_fitted
        }


class BipowerVariation:
    """
    Barndorff-Nielsen & Shephard (2004, 2006) Bipower Variation and Jump Decomposition.
    Separates continuous persistent volatility from discontinuous sudden jumps.
    """

    @staticmethod
    def decompose(returns: np.ndarray, annualize: bool = True, periods_per_year: int = 252) -> Dict[str, Any]:
        """
        Calculate Realized Variance (RV), Bipower Variation (BV), Continuous (C), and Jump (J) components.
        :param returns: High-frequency or daily log returns
        :param annualize: Whether to return values in annualized standard deviation %
        :param periods_per_year: Period scaling
        """
        r = returns[~np.isnan(returns)]
        N = len(r)
        if N < 4:
            return {
                "rv_ann": 0.0,
                "continuous_rv": 0.0,
                "jump_rv": 0.0,
                "jump_ratio": 0.0,
                "jump_regime": "CONTINUOUS_FLOW",
                "action_badge": "SAFE_FLOW",
                "description": "Insufficient return observations"
            }

        ann_factor = np.sqrt(periods_per_year) * 100.0 if annualize else 100.0

        # Realized Variance
        rv_stat = np.sum(r ** 2)

        # Bipower Variation: (pi/2) * sum(|r_i| * |r_{i-1}|)
        mu1 = np.sqrt(2.0 / np.pi)
        abs_r = np.abs(r)
        bv_stat = (1.0 / (mu1 ** 2)) * np.sum(abs_r[1:] * abs_r[:-1])

        # Jump variance J = max(RV - BV, 0)
        # Continuous variance C = min(RV, BV)
        c_var = min(rv_stat, bv_stat)
        j_var = max(rv_stat - bv_stat, 0.0)

        jump_ratio = float(j_var / rv_stat) if rv_stat > 1e-12 else 0.0
        jump_ratio = min(max(jump_ratio, 0.0), 1.0)

        # Convert to annualized volatility
        rv_ann = float(np.sqrt(rv_stat / N) * ann_factor)
        c_ann = float(np.sqrt(c_var / N) * ann_factor)
        j_ann = float(np.sqrt(j_var / N) * ann_factor)

        if jump_ratio >= 0.35:
            regime = "TRANSITORY_JUMP"
            badge = "FADE_CANDIDATE"
            desc = f"Heavy discrete jump activity ({jump_ratio * 100:.1f}% jump variance). Do not chase breakouts; fade extreme IV spikes."
        elif jump_ratio >= 0.15:
            regime = "MODERATE_JUMP"
            badge = "MONITOR_JUMP"
            desc = f"Moderate jump presence ({jump_ratio * 100:.1f}%). Partial news headline impact."
        else:
            regime = "CONTINUOUS_FLOW"
            badge = "STRUCTURAL_FLOW"
            desc = "Volatility is smooth and continuous. High persistence, suitable for trend following or systematic strangles."

        return {
            "rv_ann": round(rv_ann, 2),
            "continuous_rv": round(c_ann, 2),
            "jump_rv": round(j_ann, 2),
            "jump_ratio": round(jump_ratio, 3),
            "jump_ratio_pct": round(jump_ratio * 100.0, 1),
            "jump_regime": regime,
            "action_badge": badge,
            "description": desc
        }


class HigherMoments:
    """
    Computes Realized Skewness and Realized Kurtosis from return series.
    """

    @staticmethod
    def calculate(returns: np.ndarray) -> Dict[str, float]:
        clean_rets = returns[~np.isnan(returns)]
        N = len(clean_rets)
        if N < 8:
            return {"realized_skew": 0.0, "realized_kurtosis": 0.0, "tail_risk": "NORMAL"}

        m2 = np.mean((clean_rets - np.mean(clean_rets)) ** 2)
        if m2 < 1e-12:
            return {"realized_skew": 0.0, "realized_kurtosis": 0.0, "tail_risk": "NORMAL"}

        m3 = np.mean((clean_rets - np.mean(clean_rets)) ** 3)
        m4 = np.mean((clean_rets - np.mean(clean_rets)) ** 4)

        skew = float(m3 / (m2 ** 1.5))
        kurt = float(m4 / (m2 ** 2) - 3.0)  # excess kurtosis

        if kurt > 2.0:
            tail = "FAT_TAIL_HAZARD"
        elif kurt < -1.0:
            tail = "THIN_TAILS"
        else:
            tail = "MESOKURTIC_NORMAL"

        return {
            "realized_skew": round(skew, 3),
            "realized_kurtosis": round(kurt, 3),
            "tail_risk": tail
        }


class AdvancedVolEngine:
    """
    Master econometric facade combining Semi-Variance, HAR-RV, Bipower Jumps, and Forward VRP.
    """

    def __init__(self):
        self.semi_variance = RealizedSemiVariance()
        self.har_model = CorsiHARModel()
        self.bipower = BipowerVariation()
        self.higher_moments = HigherMoments()

    def analyze(self, df_daily: pd.DataFrame, atm_iv: float, df_intraday: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Perform complete econometric volatility analysis.
        :param df_daily: DataFrame containing at least 'close', 'high', 'low', 'open'
        :param atm_iv: Live ATM Implied Volatility in percentage (e.g. 14.5)
        :param df_intraday: Optional 5-min intraday DataFrame
        :return: Full dictionary of econometric metrics and sizing inputs
        """
        if df_daily.empty or len(df_daily) < 10:
            return {}

        closes = df_daily["close"].values if "close" in df_daily else df_daily["closes"].values
        log_rets_daily = np.diff(np.log(closes))

        # 1. Semi-Variance on daily returns (last 20 days)
        recent_daily_rets = log_rets_daily[-20:] if len(log_rets_daily) >= 20 else log_rets_daily
        daily_semi = self.semi_variance.calculate(recent_daily_rets, annualize=True, periods_per_year=252)

        # 2. Semi-Variance on intraday returns if available
        if df_intraday is not None and not df_intraday.empty and len(df_intraday) > 10:
            intra_closes = df_intraday["close"].values
            intra_rets = np.diff(np.log(intra_closes))
            intra_semi = self.semi_variance.calculate(intra_rets, annualize=True, periods_per_year=75 * 252)
            active_semi = intra_semi
        else:
            active_semi = daily_semi

        # 3. Fit HAR-RV model on rolling 20d Parkinson or C2C volatility
        rolling_20d_rv = pd.Series(log_rets_daily).rolling(20).std().dropna().values * np.sqrt(252) * 100.0
        har_fitted = False
        if len(rolling_20d_rv) >= 30:
            har_fitted = self.har_model.fit(rolling_20d_rv)

        # Formulate predictors
        # Daily RV: use 5-day rolling std (consistent with Corsi 2009 HAR-RV convention)
        # Using abs(single return) is too noisy — near-zero on quiet days, huge on event days.
        cur_rv_d = float(np.std(log_rets_daily[-5:]) * np.sqrt(252) * 100.0) if len(log_rets_daily) >= 5 else \
                   float(abs(log_rets_daily[-1]) * np.sqrt(252) * 100.0) if len(log_rets_daily) > 0 else 12.0
        cur_rv_w = float(np.std(log_rets_daily[-5:]) * np.sqrt(252) * 100.0) if len(log_rets_daily) >= 5 else cur_rv_d
        cur_rv_m = float(np.std(log_rets_daily[-22:]) * np.sqrt(252) * 100.0) if len(log_rets_daily) >= 22 else cur_rv_w

        har_forecast = self.har_model.forecast(cur_rv_d, cur_rv_w, cur_rv_m)

        # 4. Bipower Variation Jump Decomposition
        # Best on intraday if available; fallback to daily
        if df_intraday is not None and not df_intraday.empty and len(df_intraday) > 15:
            decomp_rets = np.diff(np.log(df_intraday["close"].values))
            jump_decomp = self.bipower.decompose(decomp_rets, annualize=True, periods_per_year=75 * 252)
        else:
            jump_decomp = self.bipower.decompose(recent_daily_rets, annualize=True, periods_per_year=252)

        # 5. Higher Moments
        moments = self.higher_moments.calculate(recent_daily_rets)

        # 6. Predictive Forward VRP
        forecast_rv_5d = har_forecast["forecast_5d"]
        forward_vrp = float(atm_iv - forecast_rv_5d)
        forward_vrp_1d = float(atm_iv - har_forecast["forecast_1d"])

        if forward_vrp > 3.5:
            vrp_verdict = "HIGHLY_OVERPRICED"
            vrp_action = "AGGRESSIVE_THETA_HARVEST"
            vrp_desc = f"Options overpriced by +{forward_vrp:.1f}% vs forecasted move. Sellers have high statistical edge."
        elif forward_vrp >= 0.5:
            vrp_verdict = "FAVORABLE_PREMIUM"
            vrp_action = "NORMAL_STRANGLE_WRITING"
            vrp_desc = f"Options offer positive forward harvest (+{forward_vrp:.1f}% VRP). Safe for standard deployment."
        elif forward_vrp >= -1.5:
            vrp_verdict = "FAIRLY_PRICED"
            vrp_action = "NEUTRAL_CAUTION"
            vrp_desc = "Option pricing in parity with forecasted volatility. Selective deployment."
        else:
            vrp_verdict = "UNDERPRICED"
            vrp_action = "AVOID_SHORT_STRANGLES"
            vrp_desc = f"Options underpriced by {abs(forward_vrp):.1f}%. Realized movement expected to exceed premium."

        return {
            "atm_iv": round(atm_iv, 2),
            "semi_variance": active_semi,
            "har_forecast": har_forecast,
            "jump_decomposition": jump_decomp,
            "higher_moments": moments,
            "forward_vrp": {
                "vrp_5d": round(forward_vrp, 2),
                "vrp_1d": round(forward_vrp_1d, 2),
                "verdict": vrp_verdict,
                "action": vrp_action,
                "description": vrp_desc
            }
        }
