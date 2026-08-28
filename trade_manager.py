import logging

logger = logging.getLogger(__name__)

class ActiveTrade:
    def __init__(self, setup: dict, lots: int):
        self.strategy_name = setup.get("strategy")
        self.instrument = setup.get("instrument") or {}
        self.entry_price = self.instrument.get('ltp', 0.0)
        self.lots = lots
        
        risk = setup.get("risk", {})
        self.stop_loss = risk.get("stop_loss", self.entry_price * 0.85) # Default 15% stop
        self.take_profit_1 = risk.get("take_profit_1", self.entry_price * 1.20)
        
        self.peak_price = self.entry_price
        self.is_active = True
        self.scaled_out = False

    def update_ltp(self, current_ltp: float) -> str:
        """
        Evaluate current market price against risk parameters.
        Returns an action string: 'HOLD', 'STOPPED_OUT', 'TAKE_PROFIT_1', or 'TRAILING_STOP'
        """
        if not self.is_active:
            return 'CLOSED'
            
        # Update high watermark for trailing stop
        if current_ltp > self.peak_price:
            self.peak_price = current_ltp
            
        # 1. Hard Stop Loss / Trailing Stop condition
        if current_ltp <= self.stop_loss:
            self.is_active = False
            return 'STOPPED_OUT'
            
        # 2. Scale Out (Take Profit 1)
        if not self.scaled_out and current_ltp >= self.take_profit_1:
            self.scaled_out = True
            # Move stop to breakeven after scaling out
            self.stop_loss = self.entry_price
            return 'TAKE_PROFIT_1'
            
        # 3. Dynamic Trailing Stop (e.g., trail by 10% from peak if in significant profit)
        # To be implemented based on specific strategy rules.
            
        return 'HOLD'


class TradeManager:
    """
    Tracks and manages live active trades, handling exits, trailing stops, and scaling out.
    """
    def __init__(self):
        self.active_trades = []
        
    def register_trade(self, setup: dict, lots: int):
        if lots > 0:
            trade = ActiveTrade(setup, lots)
            self.active_trades.append(trade)
            logger.info(f"Registered Trade: {trade.strategy_name} | Entry: {trade.entry_price} | SL: {trade.stop_loss} | Lots: {lots}")
            
    def update_market_prices(self, ltp_dict: dict):
        """
        :param ltp_dict: dict mapping instrument symbols to their current LTP
        """
        for trade in self.active_trades:
            if not trade.is_active:
                continue
                
            symbol = trade.instrument.get('symbol')
            current_ltp = ltp_dict.get(symbol)
            
            if current_ltp:
                action = trade.update_ltp(current_ltp)
                if action == 'STOPPED_OUT':
                    logger.warning(f"🚨 TRADE STOPPED OUT: {symbol} at {current_ltp}")
                    # Broadcast exit signal to Minion here
                elif action == 'TAKE_PROFIT_1':
                    logger.info(f"💰 PARTIAL TAKE PROFIT HIT: {symbol} at {current_ltp}")
                    # Broadcast scale-out signal to Minion here
