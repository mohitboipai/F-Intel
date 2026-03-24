import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import time
from scipy.optimize import minimize
from scipy.integrate import quad

# Local Imports
from FyersAuth import FyersAuthenticator
from OptionAnalytics import OptionAnalytics

class HestonMath:
    """
    Separated Math Logic for Heston Model (Pricing & Characteristic Functions)
    """
    @staticmethod
    def heston_char_func(u, S0, K, T, r, kappa, theta, v0, rho, xi):
        """
        Heston Characteristic Function (Heston 1993)
        """
        # Albrecher (2007) "Little Heston Trap" formulation which is stable
        j = complex(0, 1)
        
        # d parameter
        d = np.sqrt((j*rho*xi*u - kappa)**2 + (xi**2)*(u**2 + j*u))
        
        # g parameter
        g = (kappa - j*rho*xi*u - d) / (kappa - j*rho*xi*u + d)
        
        # C & D terms
        # D = (kappa - j*rho*xi*u - d) / xi**2 * ((1 - exp(-d*T)) / (1 - g*exp(-d*T)))
        # But we compute exp(-d*T) separately
        exp_dT = np.exp(-d*T)
        
        D = (kappa - complex(0,1)*rho*xi*u - d) / (xi**2) * ((1 - exp_dT) / (1 - g*exp_dT))
        
        C = (kappa * theta / xi**2) * (
            (kappa - complex(0,1)*rho*xi*u - d) * T - 2 * np.log((1 - g*exp_dT) / (1 - g))
        )
        
        # Final Characteristic Function Value
        # phi = exp(C + D*v0 + j*u*log(S0))
        return np.exp(C + D*v0 + complex(0,1)*u*np.log(S0))

    @staticmethod
    def price_vanilla_call(S0, K, T, r, kappa, theta, v0, rho, xi):
        """
        Price European Call using Heston Semi-Analytical Formula (Fourier Transform)
        Much faster than MC for calibration.
        """
        # Heston Call Price = S0 * P1 - K * e^(-rT) * P2
        
        def integrand(phi, u, j_num):
            # Integrands for P1 (j_num=1) and P2 (j_num=2)
            # Based on Gil-Pelaez formula
            F = HestonMath.heston_char_func(u - complex(0,1) if j_num==1 else u, S0, K, T, r, kappa, theta, v0, rho, xi)
            if j_num == 1:
                # Adjust for change of measure if needed, usually implemented as:
                # phi(u-i) / (S0 * exp(rT)) for P1
                # But calculating direct probabilities Pj:
                # Pj = 0.5 + 1/pi * integral( Real( exp(-i*u*lnK) * phi_j(u) / (i*u) ) du )
                pass 
                
            # Alternative implementation (Standard Heston 93):
            # P1: u shifted by -i. Denom term differences.
            pass
            return 0 # Placeholder for complex logic, using easier library approach if available? 
            # Actually implementing full integration from scratch is error prone in one shot.
            # Simplified approach: Use existing library or robustness check.
            pass

        # Let's use a simpler implementation: "Lewis (2001)" formulation is often cleaner.
        # Price = S0 - K*exp(-rT)/pi * Integral[0, inf] ( Real( exp(-iu*k) * phi(u - i/2) ... ) )
        # To avoid bugs in integration integration implementation during "One Shot", 
        # I will use a simplified objective for calibration if integration fails:
        # Just use MC with low N for calibration? Too slow.
        # I will implement the standard Heston Price formula carefully.
        
        limit = 100 # Integration limit
        
        def P(u, k, j_num):
             # Characteristic functions for P1 and P2
             # phi_1(u) = phi_hst(u - i, ...) / (S0 * exp(rT))
             # phi_2(u) = phi_hst(u, ...)
             
             # But let's use the explicit probability formula
             i = complex(0, 1)
             
             if j_num == 1:
                 # Calculate phi(u - i)
                 val = HestonMath.heston_char_func(u - i, S0, K, T, r, kappa, theta, v0, rho, xi)
                 # Normalize
                 val = val / (S0 * np.exp(r*T))
             else:
                 val = HestonMath.heston_char_func(u, S0, K, T, r, kappa, theta, v0, rho, xi)
                 
             return np.real(np.exp(-i * u * np.log(K)) * val / (i * u))

        # Integrate
        p1_int = quad(lambda u: P(u, K, 1), 0, limit)[0]
        p2_int = quad(lambda u: P(u, K, 2), 0, limit)[0]
        
        P1 = 0.5 + (1/np.pi) * p1_int
        P2 = 0.5 + (1/np.pi) * p2_int
        
        return S0 * P1 - K * np.exp(-r*T) * P2


class NiftyHestonMC:
    def __init__(self):
        self.fyers = self._authenticate()
        self.analytics = OptionAnalytics()
        self.symbol = "NSE:NIFTY50-INDEX"
        self.spot_price = 0
        self.expiry_date = None
        self.regimes = {
            "Normal (Contango)":     {"kappa": 2.0, "theta": 0.04, "xi": 0.3, "rho": -0.7, "v0": 0.02}, # Vol ~14%
            "High Vol (Fear)":       {"kappa": 4.0, "theta": 0.09, "xi": 0.8, "rho": -0.9, "v0": 0.09}, # Vol ~30%
            "Low Vol (Complacency)": {"kappa": 1.0, "theta": 0.015,"xi": 0.1, "rho": -0.5, "v0": 0.015},# Vol ~12%
            "Earnings / Event":      {"kappa": 5.0, "theta": 0.04, "xi": 2.0, "rho": -0.6, "v0": 0.05}  # High Vol of Vol
        }
        
    def _authenticate(self):
        print("Authenticating with Fyers...")
        try:
            from FyersAuth import FyersAuthenticator
            auth = FyersAuthenticator("QUTT4YYMIG-100", "ZG0WN2NL1B", "http://127.0.0.1:3000/callback")
            fyers = auth.get_fyers_instance()
            if not fyers: sys.exit(1)
            print("Authentication Successful.")
            return fyers
        except:
             # Fallback for dry runs / testing without auth file nearby
             print("Warning: Auth Module not found or failed.")
             return None

    def get_spot_price(self):
        if not self.fyers: return 24000.0
        data = {"symbols": self.symbol}
        try:
            response = self.fyers.quotes(data=data)
            if response['s'] == "ok":
                self.spot_price = response['d'][0]['v']['lp']
                print(f"Fetched Spot Price: {self.spot_price}")
                return self.spot_price
        except: pass
        return 0

    def get_option_chain(self):
        if not self.fyers: return {}
        ts = ""
        if self.expiry_date:
            try:
                dt = datetime.strptime(self.expiry_date, "%Y-%m-%d")
                ts = int(dt.timestamp())
            except: pass

        data = {"symbol": self.symbol, "strikecount": 500, "timestamp": ts} # 500 strikes
        try:
            res = self.fyers.optionchain(data=data)
            # Auto-Correction Logic
            if res.get('s') == 'error' and 'expiryData' in res.get('data', {}):
                 print("Correcting Expiry...")
                 valid = res['data']['expiryData']
                 if valid:
                     first = valid[0]
                     data['timestamp'] = first['expiry']
                     try:
                         self.expiry_date = datetime.strptime(first['date'], "%d-%m-%Y").strftime("%Y-%m-%d")
                         print(f"Auto-selected Expiry: {self.expiry_date}")
                     except: pass
                     res = self.fyers.optionchain(data=data)
            if res.get('s') == 'ok': return res.get('data', {})
        except Exception as e:
            print(f"Error: {e}")
        return {}
        
    def parse_chain(self, data):
        if not data: return pd.DataFrame()
        opts = data.get('optionsChain', [])
        recs = []
        for x in opts:
            recs.append({
                'strike': x.get('strike_price'),
                'type': 'CE' if x.get('option_type') == 'CALL' else 'PE',
                'price': x.get('ltp'),
                'iv': x.get('iv'),
            })
        return pd.DataFrame(recs)

    def calibrate_parameters(self, market_data_df, S0, T, r):
        """
        Calibrate Heston Parameters to Market Data.
        market_data_df: Columns [strike, price, type]
        """
        print("\n--- Calibrating Heston Model to Market ---")
        print(f"Data Points: {len(market_data_df)}")
        
        # Initial Guess: [kappa, theta, v0, rho, xi]
        # v0 ~ Current ATM IV^2
        current_iv = market_data_df['iv'].mean() / 100.0
        v0_guess = current_iv ** 2 if current_iv > 0 else 0.04
        
        initial_guess = [2.0, v0_guess, v0_guess, -0.7, 0.3]
        bounds = [
            (0.1, 10.0),  # kappa
            (0.001, 0.5), # theta (variance)
            (0.001, 0.5), # v0
            (-0.99, 0.99),# rho
            (0.01, 5.0)   # xi
        ]
        
        def objective(params):
            k, th, v, rh, x = params
            error_sum = 0.0
            
            # Feller Condition Penalty (2*k*th > x^2) to ensure stable variance
            # penalty = 0
            # if 2*k*th <= x**2: penalty = 1000.0
            
            for _, row in market_data_df.iterrows():
                K = row['strike']
                mkt_price = row['price']
                
                # Only calibrate Calls for now (Put Parity holds)
                if row['type'] == 'CE':
                    try:
                        model_price = HestonMath.price_vanilla_call(S0, K, T, r, k, th, v, rh, x)
                        # Weighted squared error (Giving more weight to ATM?)
                        # Use relative error
                        if mkt_price > 0:
                            error_sum += ((model_price - mkt_price) / mkt_price) ** 2
                        else:
                            error_sum += (model_price - mkt_price) ** 2
                    except:
                        error_sum += 1e6
                        
            return error_sum
        
        # Run Optimization
        print("Optimizing... (This may take 10-20s)")
        result = minimize(objective, initial_guess, bounds=bounds, method='L-BFGS-B')
        
        p = result.x
        calibrated = {"kappa": p[0], "theta": p[1], "v0": p[2], "rho": p[3], "xi": p[4]}
        print(f"Calibration Success: {result.success}")
        print(f"Params: {calibrated}")
        return calibrated

    def heston_paths(self, S0, T, r, p, steps, N_paths):
        # Unpack
        kappa, theta, v0, rho, xi = p['kappa'], p['theta'], p['v0'], p['rho'], p['xi']
        dt = T / steps
        
        Z1 = np.random.normal(size=(N_paths, steps))
        Z3 = np.random.normal(size=(N_paths, steps))
        Z2 = rho * Z1 + np.sqrt(1 - rho**2) * Z3
        
        S = np.zeros((N_paths, steps + 1))
        v = np.zeros((N_paths, steps + 1))
        S[:, 0] = S0
        v[:, 0] = v0
        
        for t in range(steps):
            v_curr = v[:, t]
            v_pos = np.maximum(v_curr, 0)
            
            # Milstein for Variance? No, stick to Euler Full Truncation
            dS = (r - 0.5*v_pos)*dt + np.sqrt(v_pos)*np.sqrt(dt)*Z1[:, t]
            S[:, t+1] = S[:, t] * np.exp(dS)
            
            dv = kappa*(theta - v_pos)*dt + xi*np.sqrt(v_pos)*np.sqrt(dt)*Z2[:, t]
            v[:, t+1] = v_curr + dv
            
        return S, v

    def run_ui(self):
        while True:
            print("\n--- Heston NIFTY Simulator v2.0 ---")
            print("1. Run Simulation (Manual Params)")
            print("2. Run Simulation (Auto-Calibration)")
            print("3. Run Simulation (Preset Regime)")
            print("4. Exit")
            
            mode = input("Select Mode: ")
            if mode == '4': break
            
            # Common Setup
            if self.spot_price == 0: self.get_spot_price()
            if self.spot_price == 0:
                 try: self.spot_price = float(input("Enter Spot: "))
                 except: self.spot_price = 24000.0
                 
            atm = round(self.spot_price/50)*50
            if not self.expiry_date:
                ex = input("Enter Expiry [Auto]: ")
                if ex: self.expiry_date = ex
            
            # Chain Fetch (needed for all really, especially calibration)
            chain_data = self.get_option_chain()
            if chain_data and self.expiry_date:
                T = self.analytics.get_time_to_expiry(self.expiry_date)
            else:
                T = 0.05
                print("Using default T=0.05 (No Expiry Data)")

            # Mode Logic
            params = {}
            if mode == '1':
                print("Enter Params (Press Enter for Default):")
                v0    = float(input("v0 [0.02]: ") or 0.02)
                theta = float(input("theta [0.02]: ") or 0.02)
                kappa = float(input("kappa [2.0]: ") or 2.0)
                xi    = float(input("xi [0.3]: ") or 0.3)
                rho   = float(input("rho [-0.7]: ") or -0.7)
                params = {"v0":v0, "theta":theta, "kappa":kappa, "xi":xi, "rho":rho}
                
            elif mode == '2':
                # Calibration
                df = self.parse_chain(chain_data)
                if df.empty:
                    print("Cannot Calibrate: No Chain Data.")
                    continue
                
                # Filter for useful calibration points
                # 1 ATM, 2 OTM Calls, 2 OTM Puts (via Calls? Nifty liquid usually)
                # Let's just grab 5-6 strikes around ATM
                df['dist'] = abs(df['strike'] - self.spot_price)
                subset = df[ (df['type']=='CE') & (df['dist'] < self.spot_price*0.02) ].sort_values('dist').head(10)
                
                params = self.calibrate_parameters(subset, self.spot_price, T, 0.08)
                
            elif mode == '3':
                print("\nRegimes:")
                keys = list(self.regimes.keys())
                for i, k in enumerate(keys): print(f"{i+1}. {k}")
                idx = int(input("Select: ") or 1) - 1
                params = self.regimes[keys[idx]]
                print(f"loaded: {keys[idx]}")

            # Run MC
            s_in = input(f"Target Strike [{atm}]: ")
            strike = float(s_in) if s_in else atm
            
            print(f"Simulating {strike} | T={T:.3f} | Params={params}...") 
            S, v = self.heston_paths(self.spot_price, T, 0.08, params, 100, 5000)
            
            # Results
            payoffs = np.maximum(S[:,-1] - strike, 0)
            price = np.mean(payoffs) * np.exp(-0.08 * T)
            
            print(f"\n>> Heston Price: {price:.2f}")
            
            # Compare with Market if available
            df = self.parse_chain(chain_data)
            if not df.empty:
                matches = df[ (df['strike'] == strike) & (df['type']=='CE') ]
                if not matches.empty:
                    mp = matches.iloc[0]['price']
                    print(f">> Market Price: {mp:.2f}")
                    diff = price - mp
                    signal = "FAIR"
                    if diff > mp*0.1: signal = "MODEL HIGHER (Market Cheap?)"
                    if diff < -mp*0.1: signal = "MODEL LOWER (Market Expensive?)"
                    print(f">> Signal: {signal} (Diff {diff:.2f})")
            
            # Plot
            try:
                plt.figure(figsize=(10,6))
                plt.subplot(2,1,1)
                plt.plot(S[:50].T, alpha=0.3)
                plt.title("Spot Paths")
                plt.axhline(strike, color='r', linestyle='--')
                plt.subplot(2,1,2)
                plt.hist(S[:,-1], bins=100)
                plt.axvline(strike, color='r')
                plt.title("Payoff Dist")
                plt.tight_layout()
                plt.show() # Note: This will block execution until closed
            except: pass

if __name__ == "__main__":
    app = NiftyHestonMC()
    app.run_ui()
