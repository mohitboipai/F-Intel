import time
import pytest
import pandas as pd
from unittest.mock import MagicMock
from SharedDataCache import SharedDataCache
from calculations.IntradayGammaSignalEngine import IntradayGammaSignalEngine


class TestOIVelocityCalculation:
    """Tests for multi-order 15-min and 5-min OI velocity and acceleration in SharedDataCache."""

    def test_multi_order_velocity_and_acceleration(self):
        fyers = MagicMock()
        cache = SharedDataCache(fyers, symbol="NSE:NIFTY50-INDEX")

        now = time.time()

        # Snapshot 1: 15 minutes ago (t - 900s)
        snap_15m = {
            'ts': now - 900,
            'oi_map': {
                (23200.0, 'CE'): 10_000_000,
                (23000.0, 'PE'): 8_000_000
            }
        }
        # Snapshot 2: 5 minutes ago (t - 300s)
        # 23200 CE unwinding slowly (-50k over 10 mins = -5k/min)
        # 23000 PE building (+100k over 10 mins = +10k/min)
        snap_5m = {
            'ts': now - 300,
            'oi_map': {
                (23200.0, 'CE'): 9_950_000,
                (23000.0, 'PE'): 8_100_000
            }
        }
        # Snapshot 3: Now (t = now)
        # 23200 CE capitulation accelerates: drops 500k in last 5 min (-100k/min)
        # 23000 PE fortress building: adds 400k in last 5 min (+80k/min)
        snap_now = {
            'ts': now,
            'oi_map': {
                (23200.0, 'CE'): 9_450_000,
                (23000.0, 'PE'): 8_500_000
            }
        }

        cache._oi_ring.extend([snap_15m, snap_5m, snap_now])

        res = cache.get_oi_velocity_data(window_secs=900, fast_window_secs=300)

        assert 'vel_by_strike' in res
        assert 'fast_vel_by_strike' in res
        assert 'accel_by_strike' in res
        assert 'pct_vel_by_strike' in res

        # 23200 CE 15m vel: (9.45M - 10M) / 15 min = -550k / 15 = ~ -36,667/min
        ce_v15 = res['vel_by_strike'][(23200.0, 'CE')]
        assert ce_v15 < -30_000

        # 23200 CE 5m fast vel: (9.45M - 9.95M) / 5 min = -500k / 5 = -100,000/min
        ce_v5 = res['fast_vel_by_strike'][(23200.0, 'CE')]
        assert ce_v5 <= -90_000

        # 23200 CE acceleration is negative (unwinding is speeding up)
        ce_accel = res['accel_by_strike'][(23200.0, 'CE')]
        assert ce_accel < -5_000

        # 23000 PE is building fortress
        pe_v15 = res['vel_by_strike'][(23000.0, 'PE')]
        assert pe_v15 > 30_000
        assert (23000.0, 'PE') in res['fortress_building']


class TestIntradayVelocityTriggers:
    """Tests for Buyer and Seller entry and exit triggers in IntradayGammaSignalEngine."""

    def test_buyer_breakout_trigger_on_accelerating_unwind(self):
        engine = IntradayGammaSignalEngine()
        spot = 23190.0
        call_wall = 23200.0
        put_wall = 23000.0

        oi_velocity_data = {
            'vel_by_strike': {
                (23200.0, 'CE'): -48_000.0,
                (23000.0, 'PE'): 10_000.0
            },
            'accel_by_strike': {
                (23200.0, 'CE'): -9_500.0,
                (23000.0, 'PE'): 500.0
            }
        }

        triggers = engine.evaluate_oi_velocity_triggers(
            spot=spot,
            call_wall=call_wall,
            put_wall=put_wall,
            oi_velocity_data=oi_velocity_data
        )

        buyer = triggers['buyer_trigger']
        assert buyer['signal'] == "BUY_CALL_MOMENTUM"
        assert buyer['strike'] == 23200.0
        assert buyer['urgency'] == "HIGH"
        assert "unwinding" in buyer['rationale']

    def test_buyer_exit_trigger_on_momentum_exhaustion(self):
        engine = IntradayGammaSignalEngine()
        spot = 23215.0  # Spot has crossed 23200
        call_wall = 23200.0
        put_wall = 23000.0

        oi_velocity_data = {
            'vel_by_strike': {
                (23200.0, 'CE'): -15_000.0,
            },
            'accel_by_strike': {
                (23200.0, 'CE'): +8_000.0  # Curvature flipped positive (unwind decelerating)
            }
        }

        triggers = engine.evaluate_oi_velocity_triggers(
            spot=spot,
            call_wall=call_wall,
            put_wall=put_wall,
            oi_velocity_data=oi_velocity_data
        )

        exit_w = triggers['exit_warning']
        assert exit_w is not None
        assert exit_w['signal'] == "EXIT_CALL_EXHAUSTION"
        assert "decelerating" in exit_w['rationale']

    def test_seller_resistance_entry_trigger(self):
        engine = IntradayGammaSignalEngine()
        spot = 23185.0
        call_wall = 23200.0
        put_wall = 23000.0

        oi_velocity_data = {
            'vel_by_strike': {
                (23200.0, 'CE'): +65_000.0,  # Heavy writer addition
                (23000.0, 'PE'): +12_000.0
            },
            'accel_by_strike': {
                (23200.0, 'CE'): +2_000.0,
                (23000.0, 'PE'): 0.0
            }
        }

        triggers = engine.evaluate_oi_velocity_triggers(
            spot=spot,
            call_wall=call_wall,
            put_wall=put_wall,
            oi_velocity_data=oi_velocity_data
        )

        seller = triggers['seller_trigger']
        assert seller['signal'] == "SELL_CE_RESISTANCE"
        assert seller['strike'] == 23200.0
        assert "defending ceiling" in seller['rationale']

    def test_seller_emergency_stop_loss_trigger(self):
        engine = IntradayGammaSignalEngine()
        spot = 23190.0
        call_wall = 23250.0
        put_wall = 23000.0

        # Short strike 23200 is being overrun with massive capitulation
        oi_velocity_data = {
            'vel_by_strike': {
                (23200.0, 'CE'): -60_000.0,
            },
            'accel_by_strike': {
                (23200.0, 'CE'): -10_000.0
            }
        }

        triggers = engine.evaluate_oi_velocity_triggers(
            spot=spot,
            call_wall=call_wall,
            put_wall=put_wall,
            oi_velocity_data=oi_velocity_data
        )

        exit_w = triggers['exit_warning']
        assert exit_w is not None
        assert exit_w['signal'] == "EMERGENCY_STOP_CE"
        assert exit_w['urgency'] == "CRITICAL"
        assert exit_w['strike'] == 23200.0


class TestMergedOptionChain:
    """Tests for merged Option Chain containing Seller Greek & P(OTM) metrics."""

    def test_chain_rows_have_seller_metrics(self):
        from VolatilityAnalyzer import VolatilityAnalyzer

        va = VolatilityAnalyzer()
        spot = 23150.0

        # Create synthetic option chain dataframe around spot
        records = []
        strikes = [23000, 23050, 23100, 23150, 23200, 23250, 23300]
        for s in strikes:
            records.append({
                'strike': float(s),
                'type': 'CE',
                'price': max(0.5, 23150 - s if s < 23150 else 30.0),
                'iv': 15.0,
                'oi': 1_000_000,
                'delta': 0.5,
                'gamma': 0.001,
                'theta': -10.0,
                'vega': 5.0
            })
            records.append({
                'strike': float(s),
                'type': 'PE',
                'price': max(0.5, s - 23150 if s > 23150 else 30.0),
                'iv': 15.0,
                'oi': 1_000_000,
                'delta': -0.5,
                'gamma': 0.001,
                'theta': -10.0,
                'vega': 5.0
            })

        df_chain = pd.DataFrame(records)

        res = va._compute_seller_data(df_chain, spot=spot, T=7/365, DTE=7, near_exp="2026-09-25")

        assert res is not None
        assert 'chain_rows' in res
        assert len(res['chain_rows']) > 0

        first_row = res['chain_rows'][0]
        # Verify strike selection metrics are merged directly on each row
        assert 'ce_prob_otm' in first_row
        assert 'ce_theta' in first_row
        assert 'ce_signal' in first_row
        assert 'pe_prob_otm' in first_row
        assert 'pe_theta' in first_row
        assert 'pe_signal' in first_row

        # Verify OTM call has valid probability
        otm_call_row = next(r for r in res['chain_rows'] if r['strike'] >= 23250)
        assert otm_call_row['ce_prob_otm'] > 50.0
        assert otm_call_row['ce_signal'] in ["★ SAFE", "✓ GOOD", "~ OK", "✗ RISKY"]

