import os
from dotenv import load_dotenv
from FyersAuth import FyersAuthenticator

load_dotenv()

_fyers_instance = None

def get_fyers_instance():
    global _fyers_instance
    if _fyers_instance is not None:
        return _fyers_instance

    app_id = os.getenv("FYERS_APP_ID")
    secret_id = os.getenv("FYERS_SECRET_ID")
    redirect_uri = os.getenv("FYERS_REDIRECT_URI", "http://127.0.0.1:3000/callback")
    
    if os.getenv("PYTEST_CURRENT_TEST") or os.getenv("CI"):
        return None

    if not app_id or not secret_id:
        print("Error: Missing FYERS_APP_ID or FYERS_SECRET_ID in .env file.")
        return None

    try:
        auth = FyersAuthenticator(app_id, secret_id, redirect_uri)
        _fyers_instance = auth.get_fyers_instance()
        if not _fyers_instance:
            print("Authentication Failed: Could not get Fyers instance.")
        return _fyers_instance
    except Exception as e:
        print(f"Error initializing FyersAuthenticator: {e}")
        return None

def get_access_token():
    if os.path.exists("access_token.txt"):
        with open("access_token.txt", "r") as f:
            return f.read().strip()
    return None
