import numpy as np
import pandas as pd

class OptionsFeatures:
    """
    Calculates Options/IV based features for both daily and intraday regime detection.
    Designed to work alongside VolatilityFeatures for multi-timeframe regime signals.
    """

    @staticmethod
    def add_features(iv_df):
        """
        Expects DF with ['iv'] (Index IV / VIX)
        Returns enriched DataFrame with IV momentum, regime, and intraday-capable signals.
        """
        d = iv_df.copy()

        # 1. IV Change (velocity)
        d['iv_chg']     = d['iv'].diff()
        d['iv_pct_chg'] = d['iv'].pct_change()

        # 2. IV velocity over 1 period (most recent day-over-day move — key for intraday)
        d['iv_velocity_1d'] = d['iv'].diff(1)

        # 3. IV Rank & Percentile (1 Year Rolling = 252 Days)
        roll_min = d['iv'].rolling(252).min()
        roll_max = d['iv'].rolling(252).max()
        d['iv_rank'] = (d['iv'] - roll_min) / (roll_max - roll_min + 1e-9)

        # 4. IV relative to slow MA (200d) — regime anchor
        d['iv_rel_ma200'] = d['iv'] / (d['iv'].rolling(200).mean() + 1e-9)

        # 5. IV Momentum (IV / IV_MA20) — medium-term momentum
        d['iv_mom']     = d['iv'] / (d['iv'].rolling(20).mean() + 1e-9)

        # 6. IV z-score (20-day) — shows how extreme current IV is
        iv_mean_20 = d['iv'].rolling(20).mean()
        iv_std_20  = d['iv'].rolling(20).std()
        d['iv_zscore_20'] = (d['iv'] - iv_mean_20) / (iv_std_20 + 1e-9)

        # 7. IV Bollinger breakout flag (IV > upper band = potential expansion)
        upper_band  = iv_mean_20 + 2 * iv_std_20
        lower_band  = iv_mean_20 - 2 * iv_std_20
        d['iv_above_upper_bb'] = (d['iv'] > upper_band).astype(int)
        d['iv_below_lower_bb'] = (d['iv'] < lower_band).astype(int)

        # 8. IV acceleration (2nd derivative) — detects change-of-regime moments
        d['iv_accel'] = d['iv_velocity_1d'].diff()

        return d[[
            'iv', 'iv_chg', 'iv_pct_chg',
            'iv_velocity_1d', 'iv_accel',
            'iv_rank', 'iv_rel_ma200', 'iv_mom',
            'iv_zscore_20',
            'iv_above_upper_bb', 'iv_below_lower_bb',
        ]]

    @staticmethod
    def compute_live_iv_features(current_iv: float, iv_history: list) -> dict:
        """
        Compute real-time IV feature snapshot for live inference.

        Args:
            current_iv: Current ATM IV (e.g. 13.5)
            iv_history: List of recent daily IV values (oldest first), min 20 needed

        Returns:
            Dict with computed features matching the training feature set
        """
        if len(iv_history) < 5:
            return {}

        hist = np.array(iv_history)
        prev_iv = hist[-1] if len(hist) >= 1 else current_iv

        iv_chg         = current_iv - prev_iv
        iv_pct_chg     = iv_chg / (prev_iv + 1e-9)
        iv_velocity_1d = iv_chg
        iv_accel       = (current_iv - 2 * prev_iv + hist[-2]) if len(hist) >= 2 else 0.0

        window_252 = hist[-252:] if len(hist) >= 252 else hist
        iv_rank = (current_iv - window_252.min()) / (window_252.max() - window_252.min() + 1e-9)

        window_200 = hist[-200:] if len(hist) >= 200 else hist
        iv_rel_ma200 = current_iv / (window_200.mean() + 1e-9)

        window_20 = hist[-20:] if len(hist) >= 20 else hist
        iv_mom       = current_iv / (window_20.mean() + 1e-9)
        iv_mean_20   = window_20.mean()
        iv_std_20    = window_20.std() + 1e-9
        iv_zscore_20 = (current_iv - iv_mean_20) / iv_std_20
        upper_bb     = iv_mean_20 + 2 * iv_std_20
        lower_bb     = iv_mean_20 - 2 * iv_std_20

        return {
            'iv':                  current_iv,
            'iv_chg':              iv_chg,
            'iv_pct_chg':          iv_pct_chg,
            'iv_velocity_1d':      iv_velocity_1d,
            'iv_accel':            iv_accel,
            'iv_rank':             iv_rank,
            'iv_rel_ma200':        iv_rel_ma200,
            'iv_mom':              iv_mom,
            'iv_zscore_20':        iv_zscore_20,
            'iv_above_upper_bb':   int(current_iv > upper_bb),
            'iv_below_lower_bb':   int(current_iv < lower_bb),
        }
