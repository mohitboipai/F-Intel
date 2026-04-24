"""
PositionLedger.py
=================
Maintains the full transaction history for a single weekly position.
Every leg open/close/settle is recorded as an individual transaction
so the entire adjustment history is auditable.

Usage
-----
    ledger = PositionLedger('2024-W42', entry_date='2024-10-14',
                             expiry_date='2024-10-17')
    ledger.add_transaction(date='2024-10-14', time_type='OPEN',
                           option_type='CE', strike=24500, action='SELL',
                           price_per_unit=85.0, shift_number=0,
                           reason='original entry')
    ledger.add_transaction(...)
    print(ledger.summary())
"""

import os
import sys
from dataclasses import dataclass, field
from datetime import date as date_type, datetime
from typing import List, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

LOT_SIZE  = 75
RISK_FREE = 0.07


def _to_date(d) -> date_type:
    if isinstance(d, date_type) and not isinstance(d, datetime):
        return d
    if isinstance(d, datetime):
        return d.date()
    return datetime.strptime(str(d)[:10], '%Y-%m-%d').date()


@dataclass
class LegTransaction:
    position_id:       str
    date:              str          # YYYY-MM-DD
    time_type:         str          # "OPEN" | "CLOSE" | "SETTLE"
    option_type:       str          # "CE" | "PE"
    strike:            float
    action:            str          # "SELL" | "BUY"
    price_per_unit:    float
    quantity_lots:     int          # always 1 in this version
    cashflow_per_unit: float        # + for credits (SELL), - for debits (BUY)
    cashflow_rupees:   float        # cashflow_per_unit * LOT_SIZE
    shift_number:      int          # 0 = original, 1 = first shift, ...
    reason:            str

    def to_dict(self) -> dict:
        return {
            'position_id':       self.position_id,
            'date':              self.date,
            'time_type':         self.time_type,
            'option_type':       self.option_type,
            'strike':            self.strike,
            'action':            self.action,
            'price_per_unit':    round(self.price_per_unit, 2),
            'quantity_lots':     self.quantity_lots,
            'cashflow_per_unit': round(self.cashflow_per_unit, 2),
            'cashflow_rupees':   round(self.cashflow_rupees, 2),
            'shift_number':      self.shift_number,
            'reason':            self.reason,
        }


class PositionLedger:
    """
    Tracks all leg transactions for a single weekly position.
    Provides running accounting and final P&L.
    """

    def __init__(
        self,
        position_id:  str,
        entry_date:   str,
        expiry_date:  str,
        entry_spot:   float = 0.0,
    ):
        self.position_id  = position_id
        self.entry_date   = entry_date
        self.expiry_date  = expiry_date
        self.entry_spot   = entry_spot
        self.exit_date:   Optional[str] = None
        self.exit_reason: Optional[str] = None

        self._transactions: List[LegTransaction] = []
        self._closed = False
        self._shift_count = 0

        # Entry credit is fixed at trade open for SL reference
        self._entry_credit_per_unit: Optional[float] = None

    # ── add transaction ───────────────────────────────────────────────────────

    def add_transaction(
        self,
        date:           str,
        time_type:      str,    # "OPEN" | "CLOSE" | "SETTLE"
        option_type:    str,    # "CE" | "PE"
        strike:         float,
        action:         str,    # "SELL" | "BUY"
        price_per_unit: float,
        shift_number:   int,
        reason:         str,
        quantity_lots:  int = 1,
    ):
        cashflow_pu = +price_per_unit if action == 'SELL' else -price_per_unit
        tx = LegTransaction(
            position_id=self.position_id,
            date=str(date)[:10],
            time_type=time_type,
            option_type=option_type,
            strike=float(strike),
            action=action,
            price_per_unit=float(price_per_unit),
            quantity_lots=quantity_lots,
            cashflow_per_unit=cashflow_pu,
            cashflow_rupees=cashflow_pu * LOT_SIZE,
            shift_number=shift_number,
            reason=reason,
        )
        self._transactions.append(tx)

        # Capture entry credit from first two OPEN/SELL transactions (legs)
        if self._entry_credit_per_unit is None:
            opens = [t for t in self._transactions if t.time_type == 'OPEN' and t.action == 'SELL']
            if len(opens) >= 2:
                self._entry_credit_per_unit = sum(t.cashflow_per_unit for t in opens)

    def mark_shift(self):
        """Call once per shift event to increment the shift counter."""
        self._shift_count += 1

    def close(self, exit_date: str, exit_reason: str):
        """Mark this position as closed."""
        self.exit_date   = exit_date
        self.exit_reason = exit_reason
        self._closed     = True

    # ── properties ────────────────────────────────────────────────────────────

    @property
    def current_ce_strike(self) -> Optional[float]:
        """Most recent open CE short that has not been closed."""
        return self._current_strike('CE')

    @property
    def current_pe_strike(self) -> Optional[float]:
        """Most recent open PE short that has not been closed."""
        return self._current_strike('PE')

    def _current_strike(self, opt_type: str) -> Optional[float]:
        """
        Find the last SELL of the given type and check it hasn't been
        matched by a BUY of the same shift_number.
        """
        sells = [t for t in self._transactions
                 if t.option_type == opt_type and t.action == 'SELL']
        if not sells:
            return None
        last_sell = sells[-1]
        # Check if it has been closed
        buys_after = [t for t in self._transactions
                      if t.option_type == opt_type
                      and t.action in ('BUY',)
                      and t.shift_number == last_sell.shift_number]
        if buys_after:
            return None   # leg was closed
        return last_sell.strike

    @property
    def running_net_per_unit(self) -> float:
        return sum(t.cashflow_per_unit for t in self._transactions)

    @property
    def running_net_rupees(self) -> float:
        return self.running_net_per_unit * LOT_SIZE

    @property
    def shift_count(self) -> int:
        return self._shift_count

    @property
    def is_active(self) -> bool:
        return not self._closed

    @property
    def entry_credit_per_unit(self) -> float:
        return self._entry_credit_per_unit or 0.0

    @property
    def total_credit_received(self) -> float:
        return sum(t.cashflow_per_unit for t in self._transactions if t.cashflow_per_unit > 0)

    @property
    def total_debit_paid(self) -> float:
        return sum(abs(t.cashflow_per_unit) for t in self._transactions if t.cashflow_per_unit < 0)

    # ── output ────────────────────────────────────────────────────────────────

    def get_transactions(self) -> List[dict]:
        return [t.to_dict() for t in self._transactions]

    def summary(self) -> dict:
        return {
            'position_id':          self.position_id,
            'entry_date':           self.entry_date,
            'expiry_date':          self.expiry_date,
            'exit_date':            self.exit_date,
            'exit_reason':          self.exit_reason,
            'entry_spot':           round(self.entry_spot, 2),
            'shift_count':          self.shift_count,
            'total_credit_received': round(self.total_credit_received, 2),
            'total_debit_paid':      round(self.total_debit_paid, 2),
            'final_pnl_per_unit':   round(self.running_net_per_unit, 2),
            'final_pnl_rupees':     round(self.running_net_rupees, 2),
        }

    def to_dict(self) -> dict:
        """Flat dict for a DataFrame row (position-level). """
        s = self.summary()
        s['entry_credit_per_unit'] = round(self.entry_credit_per_unit, 2)
        return s
