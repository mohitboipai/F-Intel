import json
import os
import uuid
import threading
from datetime import datetime
from StrategyEngine import OptionLeg, Strategy

PORTFOLIO_FILE = "portfolio.json"

class PortfolioManager:
    def __init__(self, filepath=PORTFOLIO_FILE):
        self.filepath = filepath
        self._lock = threading.Lock()
        self.data = {"active": [], "history": []}
        self.load()

    def load(self):
        with self._lock:
            if os.path.exists(self.filepath):
                try:
                    with open(self.filepath, "r") as f:
                        self.data = json.load(f)
                except Exception as e:
                    print(f"Error loading portfolio: {e}")
                    self.data = {"active": [], "history": []}

    def save(self):
        with self._lock:
            with open(self.filepath, "w") as f:
                json.dump(self.data, f, indent=4)

    def _estimate_margin(self, legs, spot) -> float:
        """
        Heuristic margin estimator.
        If max loss is limited, use max loss. Otherwise, 1,00,000 per naked short lot.
        """
        try:
            parsed_legs = []
            short_lots = 0
            for l in legs:
                parsed_legs.append(OptionLeg(
                    opt_type=l['type'],
                    action=l['action'],
                    strike=l['strike'],
                    entry_price=l['price'],
                    iv=l['iv'],
                    lots=l['lots']
                ))
                if l['action'] == 'SELL':
                    short_lots += l['lots']
                    
            strategy = Strategy("tmp", parsed_legs)
            # Check max loss between +/- 20%
            import numpy as np
            spot_range = np.linspace(spot * 0.8, spot * 1.2, 50)
            max_loss = strategy.max_loss(spot_range)
            
            if max_loss < -999999: # Unlimited loss
                return short_lots * 100000.0
            else:
                net_prem = sum(l.lot_premium for l in parsed_legs)
                # If credit spread, margin is max loss
                # If debit spread, margin is just premium paid (net_prem will be positive cost, so we use max_loss as well)
                return abs(max_loss)
        except Exception:
            return 100000.0

    def deploy(self, strategy_name, legs, spot):
        pos = {
            "id": str(uuid.uuid4()),
            "name": strategy_name or "Custom Strategy",
            "entry_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "entry_spot": spot,
            "legs": legs, # list of dicts with {action, type, strike, price, lots, iv}
            "estimated_margin": self._estimate_margin(legs, spot)
        }
        self.data["active"].append(pos)
        self.save()
        return pos

    def get_status(self, live_chain, spot, T_now, atm_iv):
        """
        live_chain: dict with { 'CE': {strike: ltp}, 'PE': {strike: ltp} }
        Calculate live P&L and Live POP for all active positions.
        """
        for pos in self.data["active"]:
            live_pnl = 0.0
            parsed_legs = []
            
            for leg in pos["legs"]:
                opt_type = leg["type"]
                strike = leg["strike"]
                entry_price = leg["price"]
                lots = leg["lots"]
                action = leg["action"]
                
                live_price = live_chain.get(opt_type, {}).get(strike, entry_price)
                diff = live_price - entry_price if action == 'BUY' else entry_price - live_price
                live_pnl += diff * lots * 65 # Lot size
                
                leg["live_price"] = live_price
                leg["live_pnl"] = diff * lots * 65
                
                # For POP, we construct the strategy at the current live price, NOT the entry price!
                # Wait, POP is probability of profit of the REMAINING position, so we should consider entry price to see if the overall trade will be profitable?
                # Yes, we pass the original entry prices to the Strategy object so breakevens are relative to entry.
                parsed_legs.append(OptionLeg(
                    opt_type=opt_type,
                    action=action,
                    strike=strike,
                    entry_price=entry_price, # original entry
                    iv=leg.get('iv', atm_iv),
                    lots=lots
                ))
                
            pos["live_pnl"] = live_pnl
            
            try:
                strategy = Strategy(pos["name"], parsed_legs)
                pop = strategy.pop(spot, T_now, atm_iv) * 100
                pos["live_pop"] = pop
            except Exception:
                pos["live_pop"] = 0.0
            
        return self.data

    def exit_position(self, pos_id, live_chain):
        active = self.data["active"]
        pos = next((p for p in active if p["id"] == pos_id), None)
        if not pos:
            return None
            
        # calculate final realized pnl
        realized_pnl = 0.0
        for leg in pos["legs"]:
            opt_type = leg["type"]
            strike = leg["strike"]
            entry_price = leg["price"]
            lots = leg["lots"]
            action = leg["action"]
            exit_price = live_chain.get(opt_type, {}).get(strike, entry_price)
            diff = exit_price - entry_price if action == 'BUY' else entry_price - exit_price
            realized_pnl += diff * lots * 65
            leg["exit_price"] = exit_price
            
        pos["exit_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        pos["realized_pnl"] = realized_pnl
        
        self.data["active"] = [p for p in active if p["id"] != pos_id]
        self.data["history"].append(pos)
        self.save()
        return pos
