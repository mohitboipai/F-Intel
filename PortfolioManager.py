import json
import os
import uuid
import threading
from datetime import datetime
from StrategyEngine import OptionLeg, Strategy

PORTFOLIO_FILE = "portfolio.json"

import math
from scipy.stats import norm

try:
    import config as _cfg
    _LOT_SIZE = _cfg.get("nifty_lot_size", 65)
    _RISK_FREE = _cfg.get("risk_free_rate", 0.051274)
    _DIVIDEND_YIELD = _cfg.get("dividend_yield", 0.0122)
except Exception:
    _LOT_SIZE = 65
    _RISK_FREE = 0.051274
    _DIVIDEND_YIELD = 0.0122


def _calc_greeks(S: float, K: float, T: float, iv: float, opt_type: str,
                 r: float = _RISK_FREE, q: float = _DIVIDEND_YIELD):
    """BSM Greeks for position tracking."""
    try:
        T = max(T, 1e-6)
        iv = max(iv, 0.01)
        sqrt_T = math.sqrt(T)
        d1 = (math.log(S / K) + (r - q + 0.5 * iv * iv) * T) / (iv * sqrt_T)
        if opt_type == 'CE':
            delta = math.exp(-q * T) * norm.cdf(d1)
        else:
            delta = -math.exp(-q * T) * norm.cdf(-d1)
        gamma = math.exp(-q * T) * norm.pdf(d1) / (S * iv * sqrt_T)
        theta_day = -(S * iv * math.exp(-q * T) * norm.pdf(d1)) / (2.0 * sqrt_T * 365.0)
        vega = S * math.exp(-q * T) * norm.pdf(d1) * sqrt_T / 100.0
        return {
            'delta': round(float(delta), 4),
            'gamma': round(float(gamma), 6),
            'theta': round(float(theta_day), 4),
            'vega': round(float(vega), 4),
            'iv': round(float(iv * 100), 2)
        }
    except Exception:
        return {'delta': 0.0, 'gamma': 0.0, 'theta': 0.0, 'vega': 0.0, 'iv': round(float(iv * 100), 2)}


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
                opt_t = l.get('opt_type') or l.get('type', 'CE')
                act = l.get('action', 'BUY')
                k = float(l.get('strike', 0))
                p = float(l.get('price', l.get('entry_price', 0)))
                lots = int(l.get('lots', 1))
                iv_val = float(l.get('iv', 0.15) or 0.15)
                parsed_legs.append(OptionLeg(
                    opt_type=opt_t,
                    action=act,
                    strike=k,
                    entry_price=p,
                    iv=iv_val,
                    lots=lots
                ))
                if act == 'SELL':
                    short_lots += lots
                    
            strategy = Strategy("tmp", parsed_legs)
            import numpy as np
            spot_range = np.linspace(spot * 0.8, spot * 1.2, 50)
            max_loss = strategy.max_loss(spot_range)
            
            if max_loss < -999999: # Unlimited loss
                return short_lots * 100000.0
            else:
                return abs(max_loss)
        except Exception:
            return 100000.0

    def deploy(self, strategy_name, legs, spot, source='MANUAL',
               sl_premium=None, target_premiums=None, greeks_at_entry=None):
        pos = {
            "id": str(uuid.uuid4()),
            "name": strategy_name or "Custom Strategy",
            "source": source,
            "entry_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "entry_spot": spot,
            "legs": legs, # list of dicts with {action, type, strike, price, lots, iv}
            "estimated_margin": self._estimate_margin(legs, spot),
            "sl_premium": float(sl_premium) if sl_premium is not None else None,
            "target_premiums": [float(t) for t in target_premiums] if target_premiums else [],
            "greeks_at_entry": greeks_at_entry or {},
            "sl_hit": False,
            "target_1_hit": False,
            "target_2_hit": False
        }
        self.data["active"].append(pos)
        self.save()
        return pos

    def get_status(self, live_chain, spot, T_now, atm_iv):
        """
        live_chain: dict with { 'CE': {strike: ltp}, 'PE': {strike: ltp} }
        Calculate live P&L, Greeks, and live POP for all active positions.
        """
        lot_size = _LOT_SIZE
        for pos in self.data["active"]:
            live_pnl = 0.0
            total_entry_cost = 0.0
            parsed_legs = []
            
            # Net position Greeks
            net_delta = 0.0
            net_gamma = 0.0
            net_theta = 0.0
            net_vega = 0.0

            for leg in pos["legs"]:
                opt_type = leg.get("opt_type") or leg.get("type", "CE")
                strike = float(leg.get("strike", 0))
                entry_price = float(leg.get("price", leg.get("entry_price", 0)))
                lots = int(leg.get("lots", 1))
                action = leg.get("action", "BUY")
                
                live_price = live_chain.get(opt_type, {}).get(strike, entry_price)
                diff = live_price - entry_price if action == 'BUY' else entry_price - live_price
                leg_pnl = diff * lots * lot_size
                live_pnl += leg_pnl

                if action == 'BUY':
                    total_entry_cost += entry_price * lots * lot_size
                
                leg["type"] = opt_type
                leg["opt_type"] = opt_type
                leg["live_price"] = live_price
                leg["live_pnl"] = leg_pnl

                # Compute leg greeks
                leg_iv = leg.get('iv', atm_iv) or atm_iv
                leg_greeks = _calc_greeks(spot, strike, T_now, leg_iv, opt_type)
                leg["live_greeks"] = leg_greeks

                sign = 1.0 if action == 'BUY' else -1.0
                net_delta += sign * leg_greeks['delta'] * lots
                net_gamma += sign * leg_greeks['gamma'] * lots
                net_theta += sign * leg_greeks['theta'] * lots
                net_vega += sign * leg_greeks['vega'] * lots
                
                parsed_legs.append(OptionLeg(
                    opt_type=opt_type,
                    action=action,
                    strike=strike,
                    entry_price=entry_price,
                    iv=leg_iv,
                    lots=lots
                ))
                
            pos["live_pnl"] = round(live_pnl, 2)
            pos["pnl_pct"] = round((live_pnl / total_entry_cost * 100), 1) if total_entry_cost > 0 else 0.0

            pos["live_greeks"] = {
                'delta': round(float(net_delta), 4),
                'gamma': round(float(net_gamma), 6),
                'theta': round(float(net_theta), 2),
                'vega': round(float(net_vega), 2)
            }

            # Compare against Greeks at entry
            g_entry = pos.get("greeks_at_entry") or {}
            if g_entry and 'delta' in g_entry:
                try:
                    e_delta = float(g_entry['delta'])
                    # For a single-leg position, compare per-unit delta
                    pos_unit_delta = net_delta / max(pos['legs'][0]['lots'], 1) if pos.get('legs') else net_delta
                    delta_diff = pos_unit_delta - e_delta
                    pos["delta_change_pts"] = round(delta_diff, 4)
                    pos["delta_change_pct"] = round((delta_diff / abs(e_delta)) * 100, 1) if abs(e_delta) > 1e-4 else 0.0
                except Exception:
                    pass

            # SL and Target checks
            sl_prem = pos.get("sl_premium")
            if sl_prem is not None and pos.get("legs"):
                # Check primary leg price against SL
                first_leg = pos["legs"][0]
                cur_lp = first_leg.get("live_price", first_leg.get("price", 0))
                if first_leg.get("action") == "BUY":
                    pos["sl_hit"] = cur_lp <= sl_prem
                else:
                    pos["sl_hit"] = cur_lp >= sl_prem

            tgts = pos.get("target_premiums") or []
            if tgts and pos.get("legs"):
                first_leg = pos["legs"][0]
                cur_lp = first_leg.get("live_price", first_leg.get("price", 0))
                if len(tgts) >= 1:
                    pos["target_1_hit"] = cur_lp >= tgts[0]
                if len(tgts) >= 2:
                    pos["target_2_hit"] = cur_lp >= tgts[1]
            
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
            opt_type = leg.get("opt_type") or leg.get("type", "CE")
            strike = float(leg.get("strike", 0))
            entry_price = float(leg.get("price", leg.get("entry_price", 0)))
            lots = int(leg.get("lots", 1))
            action = leg.get("action", "BUY")
            exit_price = live_chain.get(opt_type, {}).get(strike, entry_price)
            diff = exit_price - entry_price if action == 'BUY' else entry_price - exit_price
            realized_pnl += diff * lots * _LOT_SIZE
            leg["exit_price"] = exit_price
            
        pos["exit_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        pos["realized_pnl"] = realized_pnl
        
        self.data["active"] = [p for p in active if p["id"] != pos_id]
        self.data["history"].append(pos)
        self.save()
        return pos
