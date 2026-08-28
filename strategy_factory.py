import logging

logger = logging.getLogger(__name__)

class BaseStrategy:
    """
    Abstract base class for all Trading Strategies.
    Inherit from this to define custom entry, sizing, and exit rules.
    """
    def __init__(self):
        self.name = "Base Strategy"

    def evaluate_entry(self, context: dict, verdict: dict) -> bool:
        """
        Evaluate structural conditions to trigger a trade.
        :param context: Dictionary containing GEX, VRP, Regime, Momentum etc.
        :param verdict: Dictionary from MasterSignalEngine (verdict, score, confidence)
        :return: True if entry conditions are met, False otherwise.
        """
        raise NotImplementedError("evaluate_entry() must be implemented by the strategy")

    def select_instrument(self, context: dict, df_chain) -> dict:
        """
        Determine exactly what options to trade (Buy/Sell, Call/Put, Spread).
        :return: dict detailing the specific instrument setup (e.g. {'action': 'BUY', 'type': 'CE', 'strike': 24000})
        """
        raise NotImplementedError("select_instrument() must be implemented by the strategy")

    def calculate_risk(self, entry_price: float) -> dict:
        """
        Calculate Stop Loss and Take Profit levels based on option premium.
        :return: dict with 'stop_loss' and 'take_profit' absolute values.
        """
        raise NotImplementedError("calculate_risk() must be implemented by the strategy")


class StrategyFactory:
    """
    Manages active strategy profiles and runs them against the current market context.
    """
    def __init__(self):
        self.active_strategies = []

    def register_strategy(self, strategy: BaseStrategy):
        self.active_strategies.append(strategy)
        logger.info(f"Registered Strategy: {strategy.name}")

    def evaluate_all(self, context: dict, verdict: dict, df_chain) -> list:
        """
        Loop through all active strategies and return a list of triggered trade setups.
        """
        triggered_setups = []
        for strategy in self.active_strategies:
            if strategy.evaluate_entry(context, verdict):
                logger.info(f"Strategy Triggered: {strategy.name}")
                instrument = strategy.select_instrument(context, df_chain)
                if instrument:
                    risk = strategy.calculate_risk(instrument.get('ltp', 0.0))
                    
                    # Construct full trade payload
                    setup = {
                        "strategy": strategy.name,
                        "instrument": instrument,
                        "risk": risk,
                        "context_score": verdict.get('score', 0),
                        "confidence": verdict.get('confidence', 0)
                    }
                    triggered_setups.append(setup)
                    
        return triggered_setups
