import os
import sys
import time
import datetime
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from RegimeSystem.model.train import prepare_dataset, SEQ_LEN, MODEL_PATH

def clear_console():
    os.system('cls' if os.name == 'nt' else 'clear')

def predict_live_loop():
    print("=== NIFTY Volatility Regime Predictor (Live Dashboard) ===")
    
    if not os.path.exists(MODEL_PATH):
        print(f"Model not found at {MODEL_PATH}. Please run model/train.py first.")
        return

    print("Loading Model...")
    try:
        model = load_model(MODEL_PATH)
        print("Model Loaded Successfully.")
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    print("Starting Live Loop (Refresh every 60s)...")
    
    while True:
        try:
            # Refresh Data
            # For this MVP, we re-run the full fetch pipeline to ensure we get the latest intraday candle
            # In a highly optimized system, we would just fetch the latest candle and append to buffer.
            # But fetching 2000 daily candles is cheap (<1 sec).
            
            X, y, scaler = prepare_dataset()
            
            if X is None or len(X) == 0:
                print("Warning: Failed to fetch data. Retrying in 60s...")
                time.sleep(60)
                continue

            # Take the last available sequence (Most recent market state)
            last_sequence = X[-1].reshape(1, SEQ_LEN, -1)
            
            # Predict
            prob = model.predict(last_sequence, verbose=0)[0][0]
            
            # Strategy Logic
            regime = "Neutral"
            strategy = "Calendars / Balanced"
            
            if prob < 0.30:
                regime = "Short Volatility"
                strategy = "Sell Strangles / Iron Condors"
            elif prob > 0.55:
                regime = "Long Gamma"
                strategy = "Buy Straddles / Backspreads"
            
            # Dashboard Display
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            clear_console()
            print("==================================================")
            print(f"   NIFTY REGIME PREDICTOR (LSTM) | {timestamp}")
            print("==================================================")
            print(f"Probability of high Volatility: {prob:.4f} ({prob*100:.1f}%)")
            print("--------------------------------------------------")
            print(f"REGIME   : [{regime.upper()}]")
            print(f"STRATEGY : {strategy}")
            print("--------------------------------------------------")
            print("Updates every 60 seconds... (Ctrl+C to Stop)")
            
        except KeyboardInterrupt:
            print("\nStopping...")
            break
        except Exception as e:
            print(f"Error in loop: {e}")
            
        time.sleep(60)

if __name__ == "__main__":
    predict_live_loop()
