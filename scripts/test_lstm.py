import time
import traceback

def main():
    """Run a manual, live LSTM-cache smoke check."""
    from FyersAuth import FyersAuthenticator
    from SharedDataCache import SharedDataCache
    from fyers_auth_manager import get_fyers_instance

    try:
        print("Initializing auth...")
        fyers_client = get_fyers_instance()

        print("Initializing cache...")
        cache = SharedDataCache(fyers=fyers_client, symbol="NSE:NIFTY50-INDEX")

        print("Getting LSTM prediction for the first time (force=True)...")
        t0 = time.time()
        res = cache.get_lstm_prediction(spot=22000, iv=15, force=True)
        t1 = time.time()

        print(f"Prediction result: {res}")
        print(f"Took: {t1 - t0:.2f} seconds")
    except Exception:
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
