import sys
import os
import time
import traceback

try:
    from FyersAuth import FyersAuthenticator
    from SharedDataCache import SharedDataCache

    APP_ID = "QUTT4YYMIG-100"
    SECRET_ID = "ZG0WN2NL1B"
    REDIRECT_URI = "http://127.0.0.1:3000/callback"
    
    print("Initializing auth...")
    auth = FyersAuthenticator(APP_ID, SECRET_ID, REDIRECT_URI)
    fyers_client = auth.get_fyers_instance()
    
    print("Initializing cache...")
    cache = SharedDataCache(fyers=fyers_client, symbol="NSE:NIFTY50-INDEX")
    
    print("Getting LSTM prediction for the first time (force=True)...")
    t0 = time.time()
    res = cache.get_lstm_prediction(spot=22000, iv=15, force=True)
    t1 = time.time()
    
    print(f"Prediction result: {res}")
    print(f"Took: {t1-t0:.2f} seconds")
    
except Exception as e:
    traceback.print_exc()
