"""
StrategyBacktestEngine.py
=========================
Modular, high-performance backtesting engine for options strategies.
Supports Option Buyer Radar (ATM & OTM), Gamma Ignition, and Option Seller strategies
across 1+ year of indexed historical NSE Bhavcopy data.
Outputs comprehensive institutional metrics and a granular trade-by-trade log.
"""

import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from abc import ABC, abstractmethod

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from HistoricalDataManager import HistoricalDataManager
from calculations.GexRebalanceEngine import GexRebalanceEngine
from calculations.GexEngine import GexEngine

try:
    import config as _cfg
    NIFTY_LOT_SIZE = int(_cfg.get("nifty_lot_size", 65))
except Exception:
    NIFTY_LOT_SIZE = 65


# ─────────────────────────────────────────────────────────────────────────────
# TRADE DATA STRUCTURES
# ─────────────────────────────────────────────────────────────────────────────

class TradeLeg:
    def __init__(self, strike: float, opt_type: str, action: str,
                 entry_price: float, target_1: float, stop_loss: float,
                 target_2: Optional[float] = None):
        self.strike = float(strike)
        self.opt_type = opt_type.upper()
        self.action = action.upper()  # 'BUY' or 'SELL'
        self.entry_price = float(entry_price)
        self.target_1 = float(target_1)
        self.stop_loss = float(stop_loss)
        self.target_2 = float(target_2) if target_2 is not None else float(target_1 * 1.5)


class TradeSignal:
    def __init__(self, strategy_name: str, signal_date: str, expiry_date: str,
                 direction: str, legs: List[TradeLeg], max_hold_days: int = 3,
                 confluence_score: float = 60.0):
        self.strategy_name = strategy_name
        self.signal_date = signal_date
        self.expiry_date = expiry_date
        self.direction = direction
        self.legs = legs
        self.max_hold_days = max_hold_days
        self.confluence_score = confluence_score


class TradeResult:
    def __init__(self, trade_id: int, signal: TradeSignal, entry_date: str,
                 entry_price: float, exit_date: str, exit_price: float,
                 exit_reason: str, lot_size: int = NIFTY_LOT_SIZE, lots: int = 1):
        self.trade_id = trade_id
        self.strategy_name = signal.strategy_name
        self.signal_date = signal.signal_date
        self.entry_date = entry_date
        self.expiry_date = signal.expiry_date
        self.direction = signal.direction
        self.legs = signal.legs
        
        # Primary leg summary
        p_leg = signal.legs[0] if signal.legs else None
        self.strike = p_leg.strike if p_leg else 0.0
        self.opt_type = p_leg.opt_type if p_leg else 'CE'
        self.action = p_leg.action if p_leg else 'BUY'

        self.entry_price = round(entry_price, 2)
        self.exit_date = exit_date
        self.exit_price = round(exit_price, 2)
        self.exit_reason = exit_reason  # 'TARGET_1', 'TARGET_2', 'STOP_LOSS', 'EXPIRY', 'MAX_HOLD'
        self.lot_size = lot_size
        self.lots = lots

        # Multipliers
        mult = 1 if self.action == 'BUY' else -1
        self.points = round(mult * (self.exit_price - self.entry_price), 2)
        self.lot_pnl = round(self.points * lot_size * lots, 2)
        
        base = max(0.5, self.entry_price)
        self.roi_pct = round((self.points / base) * 100.0, 1)

        try:
            d_in = datetime.strptime(self.entry_date, "%Y-%m-%d")
            d_out = datetime.strptime(self.exit_date, "%Y-%m-%d")
            self.duration_days = max(1, (d_out - d_in).days)
        except Exception:
            self.duration_days = 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            'trade_id': self.trade_id,
            'strategy': self.strategy_name,
            'signal_date': self.signal_date,
            'entry_date': self.entry_date,
            'expiry_date': self.expiry_date,
            'strike': self.strike,
            'opt_type': self.opt_type,
            'contract': f"{int(self.strike)} {self.opt_type}",
            'action': self.action,
            'entry_price': self.entry_price,
            'exit_date': self.exit_date,
            'exit_price': self.exit_price,
            'exit_reason': self.exit_reason,
            'points': self.points,
            'lot_pnl': self.lot_pnl,
            'roi_pct': self.roi_pct,
            'duration_days': self.duration_days
        }


# ─────────────────────────────────────────────────────────────────────────────
# STRATEGY PLUGINS
# ─────────────────────────────────────────────────────────────────────────────

class BaseStrategy(ABC):
    name: str = "BaseStrategy"
    strategy_type: str = "BUYER"  # 'BUYER' or 'SELLER'

    @abstractmethod
    def generate_signal(self, date: str, spot: float, chain_df: pd.DataFrame,
                        hdm: HistoricalDataManager) -> Optional[TradeSignal]:
        pass


class OptionBuyerRadarStrategy(BaseStrategy):
    """
    Backtests the Option Buyer Radar (GexRebalanceEngine) breakout trades.
    Can trade either 'ATM' (Primary balanced delta) or 'OTM' (Momentum rocket).
    """

    def __init__(self, mode: str = "ATM", min_confluence: float = 50.0,
                 stop_loss_pct: float = 0.40, target_pct: float = 0.80):
        self.mode = mode.upper()  # 'ATM' or 'OTM'
        self.name = f"RADAR_{self.mode}"
        self.strategy_type = "BUYER"
        self.min_confluence = min_confluence
        self.stop_loss_pct = stop_loss_pct
        self.target_pct = target_pct
        self.engine = GexRebalanceEngine()

    def generate_signal(self, date: str, spot: float, chain_df: pd.DataFrame,
                        hdm: HistoricalDataManager) -> Optional[TradeSignal]:
        if chain_df.empty or spot <= 0:
            return None

        # Evaluate GEX Rebalance Radar
        radar = self.engine.evaluate(spot=spot, chain_df=chain_df)
        status = radar.get('status', '')
        direction = radar.get('direction', '')
        confluence = float(radar.get('confluence_score', 0.0) or 0.0)

        # Only trade on actionable breakout/momentum setups
        if status not in ('IGNITED', 'ARMED', 'COILING'):
            return None
        if direction not in ('BULLISH_CE', 'BEARISH_PE'):
            return None
        if confluence < self.min_confluence:
            return None

        opt_info = radar.get('primary_option') if self.mode == 'ATM' else radar.get('otm_gamma_rocket')
        if not opt_info or not opt_info.get('strike'):
            return None

        strike = float(opt_info['strike'])
        opt_type = str(opt_info.get('type', 'CE')).upper()
        entry_est = float(opt_info.get('current_price') or opt_info.get('ltp') or 0.0)

        if entry_est <= 1.0:
            return None

        # Set targets & stop loss
        t1 = entry_est * (1.0 + self.target_pct)
        sl = max(0.5, entry_est * (1.0 - self.stop_loss_pct))
        t2 = entry_est * (1.0 + self.target_pct * 1.5)

        nearest_exp = hdm.get_nearest_expiry(date)
        if not nearest_exp:
            return None

        leg = TradeLeg(strike=strike, opt_type=opt_type, action="BUY",
                       entry_price=entry_est, target_1=t1, stop_loss=sl, target_2=t2)

        return TradeSignal(
            strategy_name=self.name,
            signal_date=date,
            expiry_date=nearest_exp,
            direction=direction,
            legs=[leg],
            max_hold_days=3,
            confluence_score=confluence
        )


class ShortStraddleStrategy(BaseStrategy):
    """
    Delta-neutral Option Seller Strategy: Sells ATM CE and ATM PE on Monday/weekly start.
    """
    def __init__(self, stop_loss_mult: float = 1.5):
        self.name = "SHORT_STRADDLE"
        self.strategy_type = "SELLER"
        self.stop_loss_mult = stop_loss_mult

    def generate_signal(self, date: str, spot: float, chain_df: pd.DataFrame,
                        hdm: HistoricalDataManager) -> Optional[TradeSignal]:
        if chain_df.empty or spot <= 0:
            return None

        # Enter on Mondays or weekly start
        d_obj = datetime.strptime(date, "%Y-%m-%d")
        if d_obj.weekday() != 0:  # 0 = Monday
            return None

        nearest_exp = hdm.get_nearest_expiry(date)
        if not nearest_exp:
            return None

        atm_strike = round(spot / 50.0) * 50.0

        ce_row = chain_df[(chain_df['strike'] == atm_strike) & (chain_df['type'] == 'CE')]
        pe_row = chain_df[(chain_df['strike'] == atm_strike) & (chain_df['type'] == 'PE')]
        if ce_row.empty or pe_row.empty:
            return None

        ce_price = float(ce_row.iloc[0]['close'])
        pe_price = float(pe_row.iloc[0]['close'])
        if ce_price <= 1.0 or pe_price <= 1.0:
            return None

        tot_prem = ce_price + pe_price
        sl_price = tot_prem * self.stop_loss_mult

        leg1 = TradeLeg(strike=atm_strike, opt_type='CE', action='SELL',
                        entry_price=ce_price, target_1=0.0, stop_loss=sl_price)
        leg2 = TradeLeg(strike=atm_strike, opt_type='PE', action='SELL',
                        entry_price=pe_price, target_1=0.0, stop_loss=sl_price)

        return TradeSignal(
            strategy_name=self.name,
            signal_date=date,
            expiry_date=nearest_exp,
            direction="NEUTRAL",
            legs=[leg1, leg2],
            max_hold_days=4
        )


# ─────────────────────────────────────────────────────────────────────────────
# TRADE SIMULATOR & BACKTEST RUNNER
# ─────────────────────────────────────────────────────────────────────────────

class StrategyBacktestEngine:
    def __init__(self, hdm: Optional[HistoricalDataManager] = None):
        self.hdm = hdm if hdm is not None else HistoricalDataManager()

    def run(self, strategy: BaseStrategy, days: int = 365, start_date: str = "",
            end_date: str = "", lots: int = 1) -> Dict[str, Any]:
        """
        Executes a walk-forward backtest for the provided strategy.
        Returns aggregate stats, equity curve, drawdown array, and full trade log.
        """
        all_dates = self.hdm.get_available_dates(start_date, end_date)
        if not all_dates:
            return {'ok': False, 'error': 'No historical dates found.'}

        if days > 0 and len(all_dates) > days:
            all_dates = all_dates[-days:]

        spot_series = self.hdm.get_spot_series()
        trades: List[TradeResult] = []
        trade_counter = 1

        print(f"[StrategyBacktestEngine] Running {strategy.name} across {len(all_dates)} days...")

        i = 0
        while i < len(all_dates) - 1:
            date_str = all_dates[i]
            spot_bar = spot_series.get(date_str)
            if not spot_bar or spot_bar['close'] <= 0:
                i += 1
                continue

            spot = spot_bar['close']
            nearest_exp = self.hdm.get_nearest_expiry(date_str)
            if not nearest_exp:
                i += 1
                continue

            chain_df = self.hdm.get_daily_chain(date_str, expiry=nearest_exp)
            if chain_df.empty:
                i += 1
                continue

            # 1. Generate Signal
            signal = strategy.generate_signal(date_str, spot, chain_df, self.hdm)
            if not signal or not signal.legs:
                i += 1
                continue

            # 2. Execute on Next Trading Day
            next_date = all_dates[i + 1]
            p_leg = signal.legs[0]
            
            # Retrieve exact Entry Bar for the option
            entry_bar = self.hdm.get_option_bar(next_date, signal.expiry_date, p_leg.strike, p_leg.opt_type)
            if not entry_bar or entry_bar['open'] <= 0.5:
                # Fallback to leg entry estimate
                actual_entry = p_leg.entry_price
            else:
                actual_entry = entry_bar['open']

            # Recalculate target and SL based on actual entry
            t1 = actual_entry * (1.0 + getattr(strategy, 'target_pct', 0.80))
            sl = max(0.5, actual_entry * (1.0 - getattr(strategy, 'stop_loss_pct', 0.40)))

            # 3. Simulate Forward Holding Days
            exit_date = next_date
            exit_price = actual_entry
            exit_reason = "EXPIRY"

            # Check from entry date forward until expiry or max hold days
            hold_indices = [idx for idx in range(i + 1, len(all_dates)) 
                            if all_dates[idx] <= signal.expiry_date and (idx - (i + 1)) < signal.max_hold_days]

            for h_idx in hold_indices:
                curr_date = all_dates[h_idx]
                curr_bar = self.hdm.get_option_bar(curr_date, signal.expiry_date, p_leg.strike, p_leg.opt_type)
                
                if not curr_bar:
                    continue

                c_high = curr_bar['high']
                c_low = curr_bar['low']
                c_close = curr_bar['close']

                # Check Target 1 Hit
                if c_high >= t1:
                    exit_date = curr_date
                    exit_price = t1
                    exit_reason = "TARGET_1"
                    break

                # Check Stop Loss Hit
                if c_low <= sl:
                    exit_date = curr_date
                    exit_price = sl
                    exit_reason = "STOP_LOSS"
                    break

                # Expiry Settlement
                if curr_date == signal.expiry_date:
                    exit_date = curr_date
                    # Settle at intrinsic
                    curr_spot = spot_series.get(curr_date, {}).get('close', spot)
                    if p_leg.opt_type == 'CE':
                        intrinsic = max(0.0, curr_spot - p_leg.strike)
                    else:
                        intrinsic = max(0.0, p_leg.strike - curr_spot)
                    exit_price = intrinsic if intrinsic > 0 else c_close
                    exit_reason = "EXPIRY"
                    break

                exit_date = curr_date
                exit_price = c_close
                exit_reason = "MAX_HOLD"

            # Create trade record
            trade = TradeResult(
                trade_id=trade_counter,
                signal=signal,
                entry_date=next_date,
                entry_price=actual_entry,
                exit_date=exit_date,
                exit_price=exit_price,
                exit_reason=exit_reason,
                lot_size=NIFTY_LOT_SIZE,
                lots=lots
            )
            trades.append(trade)
            trade_counter += 1

            # Advance by at least 1 day
            i += 1

        if not trades:
            return {'ok': False, 'error': 'No trades executed under current parameters.'}

        # 4. Compute Comprehensive Statistics
        pnl_arr = np.array([t.lot_pnl for t in trades])
        wins = pnl_arr[pnl_arr > 0]
        losses = pnl_arr[pnl_arr <= 0]

        cum_pnl = np.cumsum(pnl_arr)
        running_max = np.maximum.accumulate(cum_pnl)
        drawdowns = cum_pnl - running_max
        max_dd = float(np.min(drawdowns)) if len(drawdowns) else 0.0

        win_rate = float(len(wins) / len(pnl_arr) * 100.0)
        tot_win = float(np.sum(wins)) if len(wins) else 0.0
        tot_loss = abs(float(np.sum(losses))) if len(losses) else 0.0
        profit_factor = float(round(tot_win / (tot_loss + 1e-9), 2))

        # Annualized Sharpe
        returns_pct = np.array([t.roi_pct for t in trades])
        sharpe = float(round((np.mean(returns_pct) / (np.std(returns_pct) + 1e-9)) * np.sqrt(52), 2)) if len(returns_pct) > 1 else 0.0

        equity_curve = [
            {'date': t.exit_date, 'trade_id': t.trade_id, 'pnl': float(t.lot_pnl), 'cum_pnl': float(round(cum, 2))}
            for t, cum in zip(trades, cum_pnl)
        ]

        summary = {
            'strategy': strategy.name,
            'total_trades': len(trades),
            'total_net_pnl': round(float(np.sum(pnl_arr)), 2),
            'win_rate_pct': round(win_rate, 1),
            'profit_factor': profit_factor,
            'max_drawdown': round(max_dd, 2),
            'sharpe_ratio': sharpe,
            'avg_trade_pnl': round(float(np.mean(pnl_arr)), 2),
            'avg_win': round(float(np.mean(wins)), 2) if len(wins) else 0.0,
            'avg_loss': round(float(np.mean(losses)), 2) if len(losses) else 0.0,
            'max_profit': round(float(np.max(pnl_arr)), 2) if len(pnl_arr) else 0.0,
            'max_loss': round(float(np.min(pnl_arr)), 2) if len(pnl_arr) else 0.0,
            'avg_duration_days': round(float(np.mean([t.duration_days for t in trades])), 1),
            'target_1_hits': sum(1 for t in trades if t.exit_reason == 'TARGET_1'),
            'stop_loss_hits': sum(1 for t in trades if t.exit_reason == 'STOP_LOSS'),
            'expiry_settlements': sum(1 for t in trades if t.exit_reason in ('EXPIRY', 'MAX_HOLD')),
        }

        return {
            'ok': True,
            'strategy_name': strategy.name,
            'start_date': trades[0].entry_date,
            'end_date': trades[-1].exit_date,
            'summary': summary,
            'equity_curve': equity_curve,
            'trades': [t.to_dict() for t in trades]
        }
