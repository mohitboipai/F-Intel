import os
import time
import sys
import datetime
import numpy as np
import pandas as pd
import traceback
import math

# Fyers API
from fyers_apiv3 import fyersModel

# Local Auth
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
    from FyersAuth import FyersAuthenticator
except ImportError:
    print("Warning: FyersAuth not found.")

from OptionAnalytics import OptionAnalytics
from DataClient import DataHubClient

# TensorFlow / Keras
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
    from tensorflow.keras.optimizers import Adam
    TF_AVAILABLE = True
except ImportError:
    print("CRITICAL: TensorFlow not found. Install it to run this Neural Network.")
    TF_AVAILABLE = False

try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
except ImportError:
    print("CRITICAL: Sklearn not found.")

# --- Config ---
SYMBOL = "NSE:NIFTY50-INDEX"
TIMEFRAME = "1" # 1 Minute
LOOKBACK_DAYS = 300
SEQ_LEN = 60 # Look back 60 minutes
MODEL_PATH = "models/nifty_lstm_15m_opt_v1.keras"

LOT_SIZE = 75   # NIFTY lot size — must match GammaExplosionModel.LOT_SIZE

class NiftyRangePredictor:
    def __init__(self):
        self.fyers = None
        self.model = None
        self.analytics = OptionAnalytics()
        self.gex_cache = None
        self.gex_last_update = None
        self.scaler = StandardScaler()
        self.data = pd.DataFrame()
        self.data_hub = DataHubClient()

        # Cached option-chain scalars (updated each live cycle)
        self._last_atm_iv          = 15.0   # ATM IV % (default 15)
        self._last_call_wall_dist  = 0.0    # (call_wall - spot) / spot
        self._last_put_wall_dist   = 0.0    # (spot - put_wall)  / spot
        
        # Targets: [H_15, L_15]
        self.target_cols = ['H_15', 'L_15']
        self.feature_cols = [
            'log_ret', 'rvol', 'atr', 'rsi', 'bb_width',
            'dist_ma20', 'dist_vwap',
            'min_sin', 'hour_sin', 'hour_cos',
            'atm_iv', 'call_wall_dist', 'put_wall_dist'
        ]
        
        # Auth
        self.app_id = "QUTT4YYMIG-100"
        self.secret_id = "ZG0WN2NL1B"
        self.redirect_uri = "http://127.0.0.1:3000/callback"
        self.authenticate()

    def authenticate(self):
        try:
            auth = FyersAuthenticator(self.app_id, self.secret_id, self.redirect_uri)
            self.fyers = auth.get_fyers_instance()
            print("Authentication Successful.")
        except Exception as e:
            print(f"Auth Failed: {e}")

    # --- Data & Logic ---

    def fetch_data(self):
        print(f"Fetching {LOOKBACK_DAYS} days of history...")
        today = datetime.datetime.now()
        start_date = today - datetime.timedelta(days=LOOKBACK_DAYS)
        current_start = start_date
        all_candles = []

        while current_start < today:
            current_end = current_start + datetime.timedelta(days=90)
            if current_end > today: current_end = today
            
            p = {
                "symbol": SYMBOL, "resolution": TIMEFRAME, "date_format": "1",
                "range_from": current_start.strftime("%Y-%m-%d"),
                "range_to": current_end.strftime("%Y-%m-%d"), "cont_flag": "1"
            }
            try:
                r = self.fyers.history(data=p)
                if r.get('s') == 'ok':
                    all_candles.extend(r['candles'])
                    print(f"  > Got {len(r['candles'])} candles ({current_start.date()})")
            except: pass
            
            current_start = current_end + datetime.timedelta(days=1)
            time.sleep(0.2)
        
        if not all_candles: return False
        
        df = pd.DataFrame(all_candles, columns=['ts', 'o', 'h', 'l', 'c', 'v'])
        df['datetime'] = pd.to_datetime(df['ts'], unit='s').dt.tz_localize('UTC').dt.tz_convert('Asia/Kolkata')
        df = df.set_index('datetime').sort_index()
        df = df[~df.index.duplicated(keep='first')]
        
        rename = {'o':'open', 'h':'high', 'l':'low', 'c':'close', 'v':'volume'}
        self.data = df.rename(columns=rename)[['open','high','low','close','volume']]
        print(f"Loaded {len(self.data)} candles.")
        return True

    def get_option_chain_data(self):
        # 1. Try Data Hub first
        hub_data = self.data_hub.get_latest_data()
        if hub_data and hub_data.get('chain'):
             return hub_data['chain']

        # 2. Fallback
        try:
            data = {"symbol": SYMBOL, "strikecount": 500, "timestamp": ""}
            r = self.fyers.optionchain(data=data)
            if r.get('s') == 'ok': return r.get('data', {})
        except: pass
        return None

    def fetch_gex_profile(self, spot):
        # Cache check (update every 5 mins)
        if self.gex_cache and self.gex_last_update and (datetime.datetime.now() - self.gex_last_update).seconds < 300:
            return self.gex_cache

        data = self.get_option_chain_data()
        if not data: return None

        options = data.get('optionsChain', [])
        if not options: return None
        
        expiry_list = data.get('expiryData', [])
        if not expiry_list: return None
        expiry_list = sorted(expiry_list, key=lambda x: x.get('expiry', 0))
        nearest_exp = expiry_list[0]
        expiry_date = nearest_exp.get('date') 
        
        T = self.analytics.get_time_to_expiry(expiry_date)
        if T < 0.001: T = 0.001
        
        # --- GEX Logic ---
        # Net GEX = sum(Gamma * OI * Spot * 100)
        # Sign: Calls are SOLD by Dealers (Short Call = Negative Gamma)
        #       Puts are SOLD by Dealers (Short Put = Positive Gamma)
        # Result: Total Net Gamma Exposure ($ per 1% move)
        
        net_gamma = 0
        call_oi_map = {}
        put_oi_map = {}
        
        for item in options:
            strike = item['strike_price']
            if abs(strike - spot) > 2000: continue
            
            oi = item.get('oi', 0)
            o_type = 'CE' if item['option_type'] == 'CALL' else 'PE'
            iv = item.get('iv', 0)
            if iv <= 0.01: iv = 15.0
            
            greeks = self.analytics.calculate_greeks(spot, strike, T, 0.10, iv/100.0, o_type)
            gamma = greeks.get('gamma', 0)
            
            # Record OI for Walls
            if o_type == 'CE': call_oi_map[strike] = call_oi_map.get(strike, 0) + oi
            else: put_oi_map[strike] = put_oi_map.get(strike, 0) + oi
            
            # Calculate GEX contribution — multiply by LOT_SIZE to match GammaExplosionModel scale
            gex_val = gamma * oi * spot * LOT_SIZE
            
            if o_type == 'CE': 
                net_gamma -= gex_val # Dealer Short Call
            else:
                net_gamma += gex_val # Dealer Short Put
                
        # Find Walls (Max OI)
        call_wall = max(call_oi_map, key=call_oi_map.get) if call_oi_map else 0
        put_wall = max(put_oi_map, key=put_oi_map.get) if put_oi_map else 0
        
        res = {
            'call_wall': call_wall,
            'put_wall': put_wall,
            'net_gamma': net_gamma,
            'expiry': expiry_date
        }
        self.gex_cache = res
        self.gex_last_update = datetime.datetime.now()
        return res

    def prepare_data(self, df, training=True):
        d = df.copy()
        # Features
        d['log_ret'] = np.log(d['close'] / d['close'].shift(1))
        d['vol_ma'] = d['volume'].rolling(20).mean()
        d['rvol'] = d['volume'] / d['vol_ma'].replace(0, 1)
        d['tr'] = d['high'] - d['low']
        d['atr'] = d['tr'].rolling(14).mean() / d['close']
        
        # RSI
        delta = d['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss.replace(0, 1)
        d['rsi'] = 100 - (100 / (1 + rs))
        d['rsi'] = d['rsi'] / 100.0 
        
        # BB
        d['ma20'] = d['close'].rolling(20).mean()
        d['std20'] = d['close'].rolling(20).std()
        d['bb_width'] = ((d['ma20'] + 2*d['std20']) - (d['ma20'] - 2*d['std20'])) / d['ma20']
        d['dist_ma20'] = (d['close'] - d['ma20']) / d['ma20']

        # VWAP
        d['tp'] = (d['high'] + d['low'] + d['close']) / 3
        d['vwap'] = (d['tp'] * d['volume']).rolling(300, min_periods=1).sum() / (d['volume'].rolling(300, min_periods=1).sum() + 1e-5)
        d['dist_vwap'] = (d['close'] - d['vwap']) / d['vwap']
        
        d['min_sin'] = np.sin(2 * np.pi * d.index.minute / 60)

        # Hour-of-day encoding (captures 9:15 open and 15:15 close dynamics)
        d['hour_sin'] = np.sin(2 * np.pi * d.index.hour / 24)
        d['hour_cos'] = np.cos(2 * np.pi * d.index.hour / 24)

        # Option-chain scalars (forward-filled from latest cached values)
        d['atm_iv']         = getattr(self, '_last_atm_iv', 15.0) / 100.0
        d['call_wall_dist'] = getattr(self, '_last_call_wall_dist', 0.0)
        d['put_wall_dist']  = getattr(self, '_last_put_wall_dist', 0.0)
        
        # Replace inf with NaN, then drop all NaNs
        d = d.replace([np.inf, -np.inf], np.nan)
        d = d.dropna(subset=self.feature_cols) 
        
        if training:
            # Targets (15m Look Ahead)
            indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=15)
            d['H_15_abs'] = d['high'].rolling(window=indexer).max().shift(-1)
            d['L_15_abs'] = d['low'].rolling(window=indexer).min().shift(-1)
            
            d['H_15'] = (d['H_15_abs'] / d['close']) - 1
            d['L_15'] = (d['L_15_abs'] / d['close']) - 1
            
            d = d.dropna(subset=['H_15', 'L_15'])
            
        return d

    def build_sequences(self, df):
        feature_cols = self.feature_cols
        
        valid_df = df.dropna(subset=self.target_cols)
        if len(valid_df) < SEQ_LEN: return None, None, None
        
        data_x = valid_df[feature_cols].values
        data_y = valid_df[self.target_cols].values
        
        data_x_scaled = self.scaler.fit_transform(data_x)
        
        X, y = [], []
        for i in range(len(data_x_scaled) - SEQ_LEN):
            X.append(data_x_scaled[i : i+SEQ_LEN])
            y.append(data_y[i+SEQ_LEN-1]) 
            
        return np.array(X), np.array(y), feature_cols

    # --- Training ---
    def train_model(self):
        print("Preprocessing Data...")
        df_processed = self.prepare_data(self.data)
        X, y, feats = self.build_sequences(df_processed)
        
        if X is None or len(X) < 100:
            print("Not enough data to train.")
            return

        print(f"Training LSTM on {len(X)} samples.")
        print("Predicting 2 Targets: [H_15, L_15]")

        model = Sequential([
            Input(shape=(SEQ_LEN, len(feats))),
            LSTM(64, return_sequences=True),
            Dropout(0.2),
            LSTM(32, return_sequences=False),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dense(2)
        ])
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mae')
        
        model.fit(X, y, epochs=10, batch_size=32, validation_split=0.1, verbose=1)
        model.save(MODEL_PATH)
        self.model = model
        print(f"Training Complete. Model saved to {MODEL_PATH}")
        input("Press Enter to continue...")

    # --- Analysis Engine ---
    def detect_regime(self, df, gex):
        # 1. Trend (Price vs SMA20)
        curr_price = df['close'].iloc[-1]
        sma20 = df['close'].rolling(20).mean().iloc[-1]
        vwap = df['vwap'].iloc[-1]
        
        trend = "NEUTRAL"
        if curr_price > sma20 and curr_price > vwap: trend = "BULLISH"
        elif curr_price < sma20 and curr_price < vwap: trend = "BEARISH"
        
        # 2. Volatility (ATR check)
        atr_pct = df['atr'].iloc[-1] * 100
        vol_state = "NORMAL"
        if atr_pct < 0.15: vol_state = "COMPRESSED"
        elif atr_pct > 0.4: vol_state = "VOLATILE"
        
        # 3. Gamma Regime
        gamma_state = "UNKNOWN"
        if gex:
            ng = gex['net_gamma']
            # Heuristic scale: > +1B is significant Long Gamma, < -1B signif Short Gamma.
            # But values depend on OI scale. Just sign for now.
            if ng > 0: gamma_state = "LONG GAMMA (Stabilizing)"
            else: gamma_state = "SHORT GAMMA (Accelerating)"
            
        return f"{trend} | {vol_state} | {gamma_state}"

    # --- Live Dashboard Logic ---
    def dashboard_ui(self, spot, h15, l15, gex, regime, advice):
        os.system('cls' if os.name == 'nt' else 'clear')
        
        h15 = int(h15)
        l15 = int(l15)
        
        print("="*70)
        print(f"  SMART TRADING ASSISTANT (NIFTY 50)  |  {datetime.datetime.now().strftime('%H:%M:%S')}")
        print("="*70)
        
        print(f"\n>> SPOT:  {spot:.2f}    [{regime}]")
        print(f">> RANGE: {l15} --( {spot-l15:.1f} pts )-- SPOT --( {h15-spot:.1f} pts )-- {h15}")
        
        print("\n" + "-"*30 + " ANALYSIS " + "-"*30)
        print(f"SIGNAL: {advice}")
        
        if gex:
            print("\n" + "-"*30 + " GEX STRUCTURAL LEVELS " + "-"*30)
            print(f"Call Wall (Res): {gex['call_wall']}  (Max Call OI)")
            print(f"Put Wall (Sup):  {gex['put_wall']}  (Max Put OI)")
            print(f"Net Gamma:       {gex['net_gamma']:+,.0f}  (Positive=Sticky, Negative=Slippery)")
            print(f"  [sanity: net_gamma above includes LOT_SIZE=75 multiplier — same scale as GammaExplosionModel]")
        
        print("\n" + "="*70)
        print("Press Ctrl+C to Exit")

    def run_live_session(self):
        if not os.path.exists(MODEL_PATH):
            print("Model not found. Please Train first.")
            return

        try:
            self.model = load_model(MODEL_PATH)
        except Exception:
            print("Error loading model.")
            return

        # ── Model input-shape guard ─────────────────────────────────────────
        expected_features = len(self.feature_cols)
        if self.model is not None:
            model_input_shape = self.model.input_shape[-1]
            if model_input_shape != expected_features:
                print(
                    f"[NiftyPredictor] Model input shape {model_input_shape} "
                    f"!= feature count {expected_features}. "
                    "Feature set has changed — please retrain (option 2)."
                )
                return

        print("Initializing Live Feed...")
        current_prediction = None
        
        while True:
            try:
                # 1. Fetch Latest Data
                now = datetime.datetime.now()
                p = {
                    "symbol": SYMBOL, "resolution": TIMEFRAME, "date_format": "1",
                    "range_from": (now - datetime.timedelta(days=20)).strftime("%Y-%m-%d"),
                    "range_to": now.strftime("%Y-%m-%d"), "cont_flag": "1"
                }
                r = self.fyers.history(data=p)
                if r.get('s') != 'ok':
                    time.sleep(5)
                    continue
                    
                candles = r['candles']
                df = pd.DataFrame(candles, columns=['ts','o','h','l','c','v'])
                df['datetime'] = pd.to_datetime(df['ts'], unit='s').dt.tz_localize('UTC').dt.tz_convert('Asia/Kolkata')
                df = df.set_index('datetime').sort_index()
                df['close'] = df['c']; df['high'] = df['h']; df['low'] = df['l']; df['volume'] = df['v']
                
                spot_price = df['close'].iloc[-1]
                
                # 2. Compute Features & GEX
                df_features = self.prepare_data(df, training=False)
                gex = self.fetch_gex_profile(spot_price)

                # Update cached GEX scalars used as LSTM features
                if gex:
                    self._last_call_wall_dist = (gex['call_wall'] - spot_price) / spot_price
                    self._last_put_wall_dist  = (spot_price - gex['put_wall'])  / spot_price

                # Update cached ATM IV from option chain
                chain_data = self.get_option_chain_data()
                if chain_data:
                    options = chain_data.get('optionsChain', [])
                    atm_opts = [o for o in options
                                if abs(o.get('strike_price', 0) - spot_price) < 60]
                    if atm_opts:
                        self._last_atm_iv = float(atm_opts[0].get('iv', 15.0) or 15.0)
                
                # 3. Detect Regime
                regime = self.detect_regime(df_features, gex)
                
                # 4. Predict Range (If new block)
                block_minute = (now.minute // 15) * 15
                block_start = now.replace(minute=block_minute, second=0, microsecond=0)
                
                if current_prediction is None or current_prediction['start'] != block_start:
                    if len(df_features) >= SEQ_LEN:
                        all_feats = df_features[self.feature_cols].values
                        self.scaler.fit(all_feats)
                        
                        last_seq = df_features.iloc[-SEQ_LEN:][self.feature_cols].values
                        last_seq_scaled = self.scaler.transform(last_seq)
                        
                        X_live = last_seq_scaled.reshape(1, SEQ_LEN, -1)
                        preds = self.model.predict(X_live, verbose=0)[0]
                        
                        h15 = int(spot_price * (1 + preds[0]))
                        l15 = int(spot_price * (1 + preds[1]))
                        
                        current_prediction = {'start': block_start, 'h': h15, 'l': l15}
                
                # 5. Generate Advice
                advice = "WAIT"
                cp = current_prediction
                if cp:
                    # Logic
                    range_h = cp['h']
                    range_l = cp['l']
                    
                    if spot_price >= range_h - 5:
                        advice = "RESISTANCE TEST: Watch for Rejection or Breakout."
                    elif spot_price <= range_l + 5:
                        advice = "SUPPORT TEST: Watch for Bounce or Breakdown."
                    else:
                        advice = "IN RANGE: Monitor Levels."
                        
                    if gex:
                        if spot_price > gex['call_wall']:
                            advice += " [GEX SQUEEZE UP]"
                            from AlertDispatcher import fire as _ad_fire
                            _ad_fire("NiftyPredictor", "WARNING",
                                     f"GEX Squeeze Up — spot {spot_price:.0f} above Call Wall {gex['call_wall']}",
                                     "")
                        if spot_price < gex['put_wall']:
                            advice += " [GEX CRASH ALERT]"
                            from AlertDispatcher import fire as _ad_fire
                            _ad_fire("NiftyPredictor", "CRITICAL",
                                     f"GEX Crash Alert — spot {spot_price:.0f} below Put Wall {gex['put_wall']}",
                                     "")

                # 6. Display
                if current_prediction:
                    self.dashboard_ui(spot_price, current_prediction['h'], current_prediction['l'], gex, regime, advice)
                    
                time.sleep(5)
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                # print(f"Error: {e}")
                time.sleep(5)

    def main_menu(self):
        while True:
            os.system('cls' if os.name == 'nt' else 'clear')
            print("=== NIFTY AI ASSISTANT (GEX v2.0) ===")
            print("1. Run Live Assistant")
            print("2. Retrain Model (Full History)")
            print("3. Exit")
            
            c = input("Select: ")
            
            if c == '1':
                self.run_live_session()
            elif c == '2':
                if self.fetch_data():
                    self.train_model()
            elif c == '3':
                sys.exit(0)

if __name__ == "__main__":
    app = NiftyRangePredictor()
    app.main_menu()
