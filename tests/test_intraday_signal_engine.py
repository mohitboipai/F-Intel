"""
tests/test_intraday_signal_engine.py
Unit tests for calculations/IntradayGammaSignalEngine
"""
import time
import pytest
import numpy as np
import pandas as pd
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from calculations.IntradayGammaSignalEngine import IntradayGammaSignalEngine


# ─────────────────────────────────────────────────────────────────────────────
# FIXTURES
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def engine():
    return IntradayGammaSignalEngine(lot_size=65)


def _make_chain(spot=24000.0, n_strikes=10, lot_size=65):
    """Synthetic chain around spot."""
    strikes = [spot - 200 + 50 * i for i in range(n_strikes)]
    rows = []
    for s in strikes:
        for t in ('CE', 'PE'):
            oi = int(50_000 + abs(s - spot) * 10)
            rows.append({
                'strike': s, 'type': t, 'oi': oi,
                'iv': 0.14, 'price': max(1.0, 100 - abs(s - spot) * 0.5),
                'delta': 0.5 if t == 'CE' else -0.5,
                'gamma': 0.002, 'theta': -1.0, 'vega': 10.0,
                'volume': 1000, 'dte': 3.0
            })
    return pd.DataFrame(rows)


def _make_candles(n=10, base=24000.0, direction='up'):
    """Make synthetic 1-min candles as lists [ts, o, h, l, c, vol]."""
    candles = []
    ts = time.time() - n * 60
    price = base
    for i in range(n):
        step  = 5.0 if direction == 'up' else -5.0
        o     = price
        c     = price + step
        h     = max(o, c) + 3.0
        l     = min(o, c) - 3.0
        candles.append([ts + i * 60, o, h, l, c, 1200.0])
        price = c
    return candles


# ─────────────────────────────────────────────────────────────────────────────
# TEST: OI VELOCITY LABELLING
# ─────────────────────────────────────────────────────────────────────────────

class TestComputeOIVelocity:

    def test_no_data_returns_unavailable(self, engine):
        r = engine.compute_oi_velocity({})
        assert r['available'] is False

    def test_violent_capitulation_detected(self, engine):
        vel = {(24000.0, 'PE'): -200_000.0, (24050.0, 'CE'): 5000.0}
        r   = engine.compute_oi_velocity({
            'vel_by_strike': vel,
            'violently_unwinding': [(24000.0, 'PE')],
            'snapshot_count': 5
        })
        assert r['available'] is True
        assert r['has_capitulation'] is True
        assert r['labels'][(24000.0, 'PE')] == 'VIOLENT_CAPITULATION'

    def test_building_label(self, engine):
        vel = {(24500.0, 'CE'): 80_000.0}
        r   = engine.compute_oi_velocity({
            'vel_by_strike': vel, 'violently_unwinding': [], 'snapshot_count': 3
        })
        assert r['labels'][(24500.0, 'CE')] == 'BUILDING'

    def test_stable_label(self, engine):
        vel = {(24000.0, 'PE'): 1_000.0}
        r   = engine.compute_oi_velocity({
            'vel_by_strike': vel, 'violently_unwinding': [], 'snapshot_count': 3
        })
        assert r['labels'][(24000.0, 'PE')] == 'STABLE'


# ─────────────────────────────────────────────────────────────────────────────
# TEST: MULTI-CANDLE ABSORPTION SCORER
# ─────────────────────────────────────────────────────────────────────────────

class TestScoreCandleAbsorption:

    def _make_rejection_candles(self, level, barrier_type, n=5):
        """Candles that show clear rejection at the given level with volume surge."""
        candles = []
        ts = time.time() - n * 60
        for i in range(n):
            if barrier_type == 'SUPPORT':
                if i < 2:
                    # Normal approach bars (baseline vol 1000)
                    candles.append([ts + i * 60,
                                     level + 30,
                                     level + 35,
                                     level + 20,
                                     level + 22,
                                     1000.0])
                else:
                    # Bullish rejection: dips below level, closes above with volume surge
                    candles.append([ts + i * 60,
                                     level + 10,         # open
                                     level + 40,         # high (big upper body)
                                     level - 5,          # low (dips to level)
                                     level + 35,         # close (bullish bar)
                                     2500.0])            # vol (2.5x surge)
            else:
                if i < 2:
                    # Normal approach bars (baseline vol 1000)
                    candles.append([ts + i * 60,
                                     level - 30,
                                     level - 20,
                                     level - 35,
                                     level - 22,
                                     1000.0])
                else:
                    # Bearish rejection: probes above level, closes below with volume surge
                    candles.append([ts + i * 60,
                                     level - 10,         # open
                                     level + 5,          # high (probes level)
                                     level - 40,         # low (big lower body)
                                     level - 35,         # close (bearish bar)
                                     2500.0])            # vol (2.5x surge)
        return candles

    def test_empty_candles_returns_zero(self, engine):
        r = engine.score_candle_absorption([], 24000.0, 'SUPPORT')
        assert r['score'] == 0.0
        assert r['confirmed'] is False
        assert r['setup_quality'] == 'NONE'

    def test_no_proximity_returns_zero(self, engine):
        # Level is far from candles
        candles = _make_candles(5, base=25000.0)
        r = engine.score_candle_absorption(candles, 24000.0, 'SUPPORT')
        assert r['score'] == 0.0

    def test_strong_bullish_absorption(self, engine):
        level   = 24000.0
        candles = self._make_rejection_candles(level, 'SUPPORT', n=5)
        r       = engine.score_candle_absorption(candles, level, 'SUPPORT')
        assert r['wick_bars'] >= 2
        assert r['setup_quality'] in ('STRONG', 'MODERATE')
        assert r['score'] > 40.0

    def test_strong_bearish_absorption(self, engine):
        level   = 24500.0
        candles = self._make_rejection_candles(level, 'RESISTANCE', n=5)
        r       = engine.score_candle_absorption(candles, level, 'RESISTANCE')
        assert r['wick_bars'] >= 2
        assert r['score'] > 40.0


# ─────────────────────────────────────────────────────────────────────────────
# TEST: SWING QUALITY CLASSIFIER
# ─────────────────────────────────────────────────────────────────────────────

class TestClassifySwingQuality:

    def _make_ohlc(self, n=15, high=24200.0, low=23800.0):
        rows = [{'open': 24000, 'high': high, 'low': low, 'close': 24100, 'volume': 500000}
                for _ in range(n)]
        return pd.DataFrame(rows)

    def test_trending_day_large_range(self, engine):
        # Today range 400pts, ADR = 200 -> 200% -> TRENDING
        ohlc  = self._make_ohlc(15, high=24200, low=24000)   # ADR ~200
        cands = []
        ts    = time.time() - 60 * 30
        # 30 bars with a 400pt range
        for i in range(30):
            cands.append([ts + i*60, 24000, 24400, 23800, 24300, 1500.0])
        r = engine.classify_swing_quality(cands, ohlc)
        assert r['quality'] == 'TRENDING'
        assert r['is_trending'] is True

    def test_session_phase_mid_trend(self, engine, monkeypatch):
        from datetime import datetime
        mock_dt = datetime(2026, 9, 15, 12, 30, 0)   # 12:30 = MID_TREND
        import calculations.IntradayGammaSignalEngine as mod
        monkeypatch.setattr(mod, 'datetime', type('_DT', (), {
            'now': staticmethod(lambda: mock_dt),
            'fromtimestamp': datetime.fromtimestamp
        }))
        r = engine.classify_swing_quality([])
        assert r['session_phase'] == 'MID_TREND'

    def test_no_candles_returns_choppy_or_coiled(self, engine):
        r = engine.classify_swing_quality([])
        assert r['quality'] in ('CHOPPY', 'COILED')


# ─────────────────────────────────────────────────────────────────────────────
# TEST: MASTER UPDATE
# ─────────────────────────────────────────────────────────────────────────────

class TestUpdate:

    def test_update_no_data_returns_error(self, engine):
        r = engine.update(spot=0, chain_df=pd.DataFrame())
        assert r['ok'] is False

    def test_update_returns_valid_schema(self, engine):
        chain = _make_chain(spot=24000.0)
        r     = engine.update(
            spot=24000.0,
            chain_df=chain,
            oi_velocity_data=None,
            candles=_make_candles(15, base=24000.0),
            ohlc_df=None,
            dte=3.0
        )
        assert r['ok'] is True
        assert 'score' in r
        assert 'swing_quality' in r
        assert 'absorption' in r
        assert 'gex_summary' in r
        assert 'momentum_status' in r
        assert 'rationale' in r
        assert isinstance(r['rationale'], list)
        assert 0.0 <= r['score'] <= 100.0

    def test_update_with_oi_velocity_boosts_score(self, engine):
        chain    = _make_chain(spot=24000.0)
        oi_vel   = {
            'vel_by_strike':      {(23800.0, 'PE'): -250_000.0},
            'violently_unwinding': [(23800.0, 'PE')],
            'snapshot_count':     5
        }
        r = engine.update(
            spot=24000.0,
            chain_df=chain,
            oi_velocity_data=oi_vel,
            candles=_make_candles(15, base=24000.0, direction='up'),
            dte=3.0
        )
        assert r['ok'] is True
        # Violent capitulation should push score up significantly
        assert r['score'] > 20.0

    def test_actionable_signal_has_entry(self, engine, monkeypatch):
        """If score crosses 65, entry_signal must be populated."""
        chain = _make_chain(spot=24000.0)
        # Force high score by patching _SCORE_ACTIONABLE threshold
        import calculations.IntradayGammaSignalEngine as mod
        monkeypatch.setattr(mod, '_SCORE_ACTIONABLE', 0.0)
        r = engine.update(
            spot=24000.0,
            chain_df=chain,
            candles=_make_candles(20, base=24000.0, direction='up'),
            dte=3.0
        )
        assert r['actionable'] is True
        assert r['entry_signal'] is not None
        es = r['entry_signal']
        assert 'direction' in es
        assert 'entry_zone' in es
        assert 'sl_spot' in es
        assert 't2' in es
        assert es['rr_ratio'] > 0


# ─────────────────────────────────────────────────────────────────────────────
# TEST: MOMENTUM STATUS
# ─────────────────────────────────────────────────────────────────────────────

class TestMomentumStatus:

    def test_neutral_insufficient_history(self, engine):
        assert engine._compute_momentum_status() == 'NEUTRAL'

    def test_long_momentum(self, engine):
        for p in range(10):
            engine._spot_history.append(24000.0 + p * 10)  # strongly rising
        assert engine._compute_momentum_status() == 'LONG'

    def test_short_momentum(self, engine):
        for p in range(10):
            engine._spot_history.append(24000.0 - p * 10)  # strongly falling
        assert engine._compute_momentum_status() == 'SHORT'
