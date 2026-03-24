import os
import time
import webbrowser
from fyers_apiv3 import fyersModel

import os
import time
import webbrowser
from fyers_apiv3 import fyersModel
from http.server import BaseHTTPRequestHandler, HTTPServer
import urllib.parse
import sys

class FyersAuthenticator:
    def __init__(self, client_id, secret_id, redirect_uri, token_file="access_token.txt"):
        self.client_id = client_id
        self.secret_id = secret_id
        self.redirect_uri = redirect_uri
        self.token_file = token_file
        self.access_token = None
        self.auth_code = None

    def get_access_token(self):
        """
        Get access token. If saved token exists and is valid, use it.
        Otherwise, perform OAuth flow.
        """
        # Try to load saved token
        if os.path.exists(self.token_file):
            with open(self.token_file, "r") as f:
                self.access_token = f.read().strip()
                print("Loaded saved access token.")
                return self.access_token
        
        # If no token, perform login
        return self.authenticate()

    def authenticate(self):
        """
        Perform OAuth2 authentication flow to get a new access token.
        """
        # 1. Generate Session Object
        session = fyersModel.SessionModel(
            client_id=self.client_id,
            secret_key=self.secret_id,
            redirect_uri=self.redirect_uri,
            response_type="code",
            grant_type="authorization_code"
        )

        # 2. Generate Auth URL
        auth_url = session.generate_authcode()
        print(f"\nOpening browser for authentication...")
        print(f"URL: {auth_url}")
        webbrowser.open(auth_url)

        # 3. Start Local Server to catch Callback
        print("\nWaiting for callback on http://127.0.0.1:3000/callback ...")
        
        # Define handler to capture code
        authenticator = self
        
        class CallbackHandler(BaseHTTPRequestHandler):
            def do_GET(self):
                parsed_path = urllib.parse.urlparse(self.path)
                query_params = urllib.parse.parse_qs(parsed_path.query)
                
                if 'auth_code' in query_params:
                    authenticator.auth_code = query_params['auth_code'][0]
                    
                    # Send response to browser
                    self.send_response(200)
                    self.send_header('Content-type', 'text/html')
                    self.end_headers()
                    self.wfile.write(b"<html><body><h1>Authentication Successful!</h1><p>You can close this window and return to the terminal.</p></body></html>")
                else:
                    self.send_response(400)
                    self.end_headers()
                    self.wfile.write(b"Authentication Failed: No auth_code found.")
                
            def log_message(self, format, *args):
                return # Suppress server logs

        # Start server and handle one request
        try:
            server = HTTPServer(('127.0.0.1', 3000), CallbackHandler)
            server.handle_request() # Blocks until request is handled
            server.server_close()
        except Exception as e:
            print(f"Error starting local server: {e}")
            print("Please enter auth_code manually.")
            self.auth_code = input("Enter the auth_code here: ").strip()

        if not self.auth_code:
            print("Failed to capture auth code.")
            return None

        print(f"\nAuth Code captured: {self.auth_code[:10]}...")

        # 4. Generate Access Token
        session.set_token(self.auth_code)
        response = session.generate_token()

        if response and "access_token" in response:
            self.access_token = response["access_token"]
            
            # Save token
            with open(self.token_file, "w") as f:
                f.write(self.access_token)
            
            print("\n✓ Authentication successful! Token saved.")
            return self.access_token
        else:
            print(f"\n✗ Authentication failed: {response}")
            return None

    def get_fyers_instance(self):
        """
        Return an authenticated fyersModel instance.
        """
        token = self.get_access_token()
        if not token:
            return None
        
        fyers = fyersModel.FyersModel(client_id=self.client_id, token=token, log_path=".")
        
        # Validate token
        print("Validating access token...")
        try:
            response = fyers.get_profile()
            # logic to check if token is valid
            # Fyers API usually returns 's': 'error', 'code': -15 for invalid token
            # But get_profile structure might be different. 
            # If invalid, response might be {'s': 'error', 'code': -15, ...}
            
            if isinstance(response, dict) and response.get('s') == 'error':
                if response.get('code') == -15 or "token" in response.get('message', '').lower():
                    print("Saved token is invalid or expired. Re-authenticating...")
                    
                    # Remove invalid token file if it exists
                    if os.path.exists(self.token_file):
                        os.remove(self.token_file)
                    
                    # Re-authenticate
                    token = self.authenticate()
                    if not token:
                        return None
                    
                    # Return new instance
                    fyers = fyersModel.FyersModel(client_id=self.client_id, token=token, log_path=".")
                    return fyers
            
            print("Token is valid.")
            return fyers
            
        except Exception as e:
            print(f"Error validating token: {e}")
            # If we can't validate, we return the instance anyway or maybe re-authenticate?
            # Safe to assume if validation crashes, something else is wrong.
            # But if it's just a network error, we shouldn't force re-login.
            # Let's return the instance and let the user deal with it if it persists.
            return fyers

if __name__ == "__main__":
    # Test Authentication
    APP_ID = "QUTT4YYMIG-100"
    SECRET_ID = "ZG0WN2NL1B"
    REDIRECT_URI = "http://127.0.0.1:3000/callback"
    
    auth = FyersAuthenticator(APP_ID, SECRET_ID, REDIRECT_URI)
    fyers = auth.get_fyers_instance()
    
    if fyers:
        print("\nVerifying connection...")
        # Try a simple profile call or history call to verify
        try:
            profile = fyers.get_profile()
            print("Profile Response:", profile)
        except Exception as e:
            print("Error verifying connection:", e)
