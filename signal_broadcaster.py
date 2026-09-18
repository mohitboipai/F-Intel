import time
import threading
import logging
import requests

try:
    import config
except ImportError:
    config = None

logger = logging.getLogger("SignalBroadcaster")

class SignalBroadcaster:
    """
    Broadcasts high-conviction trade setups to execution minion nodes (e.g. minion_node.py).
    Dispatches asynchronously in a background daemon thread to avoid blocking the main analyzer loop.
    Gracefully handles offline minion endpoints with throttled warnings.
    """
    SECRET_TOKEN = "fintel_master_secret_2026"
    
    # Use 127.0.0.1 to avoid Windows IPv6 (::1) DNS resolution delays
    DEFAULT_CLIENT_ENDPOINTS = [
        "http://127.0.0.1:5000/execute_trade"
    ]
    
    # Last offline warning timestamp per endpoint to prevent log flooding
    _last_warn_time: dict[str, float] = {}
    _WARN_COOLDOWN_SECS = 60.0

    @classmethod
    def get_endpoints(cls) -> list[str]:
        if config:
            custom = config.get("minion_endpoints")
            if custom and isinstance(custom, list):
                return custom
        return cls.DEFAULT_CLIENT_ENDPOINTS

    @classmethod
    def is_enabled(cls) -> bool:
        if config:
            return bool(config.get("enable_trade_broadcast", True))
        return True

    @classmethod
    def broadcast_trade(cls, setup_data: dict) -> None:
        """
        Broadcasts the trade setup to all connected Minion nodes asynchronously.
        Never blocks the caller.
        """
        if not cls.is_enabled():
            return

        # Run dispatch in a background thread to guarantee 0 ms blocking on main loop
        threading.Thread(
            target=cls._dispatch_payload,
            args=(setup_data,),
            daemon=True,
            name="TradeBroadcastWorker"
        ).start()

    @classmethod
    def _dispatch_payload(cls, setup_data: dict) -> None:
        endpoints = cls.get_endpoints()
        if not endpoints:
            return

        payload = {
            "secret_token": cls.SECRET_TOKEN,
            "setup": setup_data
        }

        for endpoint in endpoints:
            try:
                # 0.8s connect timeout, 2.0s read timeout
                response = requests.post(endpoint, json=payload, timeout=(0.8, 2.0))
                
                if response.status_code == 200:
                    logger.info(f"✅ Trade setup dispatched to {endpoint}")
                    # Clear offline warn timestamp on success
                    cls._last_warn_time.pop(endpoint, None)
                else:
                    logger.warning(f"⚠️ Failed to send trade to {endpoint} (HTTP {response.status_code}): {response.text[:100]}")

            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
                now = time.time()
                last_warn = cls._last_warn_time.get(endpoint, 0.0)
                if now - last_warn >= cls._WARN_COOLDOWN_SECS:
                    cls._last_warn_time[endpoint] = now
                    logger.warning(
                        f"⚠️ Minion node offline at {endpoint} — "
                        f"run 'python minion_node.py' if you want automated trade execution."
                    )
            except Exception as e:
                logger.error(f"❌ Error broadcasting to {endpoint}: {e}")
