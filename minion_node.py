from flask import Flask, request, jsonify
import logging
import json
from datetime import datetime

# Configure basic logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | MINION NODE | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# A simple secret token to ensure only our Master can send trades
SECRET_TOKEN = "fintel_master_secret_2026"

@app.route('/', methods=['GET'])
def index():
    return "<h2>Minion Node is running and waiting for Master signals!</h2><p>Endpoint: <code>POST /execute_trade</code></p>"

@app.route('/execute_trade', methods=['POST'])
def execute_trade():
    data = request.json
    if not data:
        return jsonify({"status": "error", "message": "No data provided"}), 400
        
    token = data.get("secret_token")
    if token != SECRET_TOKEN:
        logger.warning(f"Unauthorized trade attempt from {request.remote_addr}")
        return jsonify({"status": "error", "message": "Unauthorized"}), 401
        
    setup = data.get("setup", {})
    if not setup:
        return jsonify({"status": "error", "message": "No trade setup provided"}), 400
        
    # Pretty print the received trade signal
    logger.info("="*50)
    logger.info(f"🚨 RECEIVED TRADE SIGNAL: {setup.get('symbol', 'UNKNOWN')} 🚨")
    logger.info(f"Direction: {setup.get('type', 'N/A').upper()}")
    logger.info(f"Strike:    {setup.get('strike', 'N/A')}")
    logger.info(f"LTP:       {setup.get('ltp', 'N/A')}")
    logger.info(f"Target 1:  {setup.get('target1', 'N/A')}")
    logger.info(f"Target 2:  {setup.get('target2', 'N/A')}")
    logger.info("="*50)
    
    # FUTURE: Actually execute the trade on Dhan/Fyers here!
    
    return jsonify({"status": "success", "message": "Trade signal received and queued"}), 200

if __name__ == '__main__':
    logger.info("Minion Node started. Waiting for Master signals on port 5000...")
    # Run on 0.0.0.0 to allow network connections (if Master is on another machine later)
    app.run(host='0.0.0.0', port=5000)
