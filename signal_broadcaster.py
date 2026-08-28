import requests
import logging

logger = logging.getLogger(__name__)

class SignalBroadcaster:
    # A simple secret token to ensure only our Master can send trades
    SECRET_TOKEN = "fintel_master_secret_2026"
    
    # List of all client Minion node IPs. 
    # For now, it's just our local test minion.
    CLIENT_ENDPOINTS = [
        "http://localhost:5000/execute_trade"
    ]

    @classmethod
    def broadcast_trade(cls, setup_data):
        """
        Broadcasts the trade setup to all connected Minion nodes.
        """
        payload = {
            "secret_token": cls.SECRET_TOKEN,
            "setup": setup_data
        }
        
        logger.info(f"Broadcasting trade setup to {len(cls.CLIENT_ENDPOINTS)} clients...")
        
        for endpoint in cls.CLIENT_ENDPOINTS:
            try:
                # Use a short timeout so we don't block the Master node if a client is down
                response = requests.post(endpoint, json=payload, timeout=2.0)
                
                if response.status_code == 200:
                    logger.info(f"✅ Successfully sent to {endpoint}")
                else:
                    logger.warning(f"❌ Failed to send to {endpoint} - Status {response.status_code}")
                    
            except requests.exceptions.RequestException as e:
                logger.error(f"❌ Network error sending to {endpoint}: {e}")
