import numpy as np
import pandas as pd

class VolatilityFeatures:
    """
    Calculates Realized Volatility Metrics on Spot Data.
    Designed for both daily regime classification and intraday session bias.
    """

    @staticmethod
    def add_features(df):
        """
        Expects DF with ['open', 'high', 'low', 'close']
        Returns enriched DataFrame with multi-period vol metrics.
        """
        d = df.copy()

        # 1. Log Returns
        d['log_ret']    = np.log(d['close'] / d['close'].shift(1))
        d['log_ret_5d'] = d['log_ret'].rolling(5).sum()

        # 2. Historical Volatility — 20D Annualized (primary)
        d['hv_20'] = d['log_ret'].rolling(20).std() * np.sqrt(252) * 100

        # 3. 5-Day HV — captures recent intraday vol spikes faster
        d['hv_5'] = d['log_ret'].rolling(5).std() * np.sqrt(252) * 100

        # 4. HV ratio (5d / 20d): > 1 means vol is rising faster than average
        d['hv_ratio_5_20'] = d['hv_5'] / (d['hv_20'] + 1e-9)

        # 5. Parkinson Volatility (High-Low proxy, variance efficient)
        const = 1.0 / (4.0 * np.log(2.0))
        d['log_hl']       = np.log(d['high'] / d['low'])
        d['parkinson_var'] = const * (d['log_hl'] ** 2)
        d['parkinson_vol'] = np.sqrt(d['parkinson_var'].rolling(20).mean()) * np.sqrt(252) * 100

        # 6. ATR (Normalized)
        d['tr1']  = d['high'] - d['low']
        d['tr2']  = (d['high'] - d['close'].shift(1)).abs()
        d['tr3']  = (d['low']  - d['close'].shift(1)).abs()
        d['tr']   = d[['tr1', 'tr2', 'tr3']].max(axis=1)
        d['atr']  = d['tr'].rolling(14).mean()
        d['natr'] = (d['atr'] / d['close']) * 100  # Normalized ATR %

        # 7. Gap Size (Open / PrevClose - 1)
        d['gap'] = (d['open'] / d['close'].shift(1)) - 1

        # 8. Intraday range as % of open (OHLC proxy for session vol)
        d['range_pct'] = (d['high'] - d['low']) / (d['open'] + 1e-9) * 100

        # 9. HV z-score (20d window of 5d HV)
        hv5_mean_20 = d['hv_5'].rolling(20).mean()
        hv5_std_20  = d['hv_5'].rolling(20).std()
        d['hv5_zscore'] = (d['hv_5'] - hv5_mean_20) / (hv5_std_20 + 1e-9)

        return d[[
            'log_ret', 'log_ret_5d',
            'hv_20', 'hv_5', 'hv_ratio_5_20',
            'parkinson_vol', 'natr', 'gap',
            'range_pct', 'hv5_zscore',
        ]]
