import math
import logging

logger = logging.getLogger(__name__)

class PositionSizer:
    """
    Dynamically calculates trade size based on Account Equity, Volatility, and Signal Confidence.
    """
    def __init__(self, account_equity: float, risk_per_trade_pct: float = 0.02):
        self.account_equity = account_equity
        self.risk_per_trade_pct = risk_per_trade_pct # Max % of account willing to lose on one trade

    def calculate_size(self, confidence: float, entry_price: float, stop_loss: float, lot_size: int = 25) -> int:
        """
        Calculate the number of lots to trade.
        :param confidence: 0.0 to 1.0 (from MasterSignalEngine)
        :param entry_price: Premium of the option (e.g. 150)
        :param stop_loss: Absolute stop loss price (e.g. 120)
        :param lot_size: Broker lot size (NIFTY = 25)
        :return: Integer number of lots to buy/sell
        """
        if entry_price <= stop_loss or confidence <= 0:
            return 0
            
        risk_per_unit = entry_price - stop_loss
        max_monetary_risk = self.account_equity * self.risk_per_trade_pct
        
        # Scale risk down linearly if confidence is low (e.g., 0.5 conf = half risk)
        adjusted_monetary_risk = max_monetary_risk * confidence
        
        # Calculate raw units
        total_units = adjusted_monetary_risk / risk_per_unit
        
        # Floor to nearest whole lot
        lots = math.floor(total_units / lot_size)
        
        logger.info(f"Position Sizer: Conf={confidence:.2f}, MaxRisk={max_monetary_risk}, AdjRisk={adjusted_monetary_risk:.2f}")
        logger.info(f"Position Sizer: Risk/Unit={risk_per_unit:.2f} -> Allocated {lots} lots.")
        
        return max(lots, 0)
