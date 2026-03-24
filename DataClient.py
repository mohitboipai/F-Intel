import requests
import time

class DataHubClient:
    def __init__(self, port=8082):
        self._base_url = f"http://127.0.0.1:{port}"
        self._last_data = None

    def get_latest_data(self, timeout=5):
        """Fetch the latest snapshot from the DataHub server."""
        try:
            res = requests.get(f"{self._base_url}/get_data", timeout=timeout)
            if res.status_code == 200:
                data = res.json()
                self._last_data = data
                return data
            else:
                return None
        except Exception as e:
            # print(f"DataHubClient Error: {e}")
            return None

    def is_alive(self):
        try:
            res = requests.get(f"{self._base_url}/status", timeout=2)
            return res.status_code == 200
        except:
            return False

    def wait_for_data(self, max_retries=10, interval=2):
        """Wait for the server to have live data."""
        for _ in range(max_retries):
            data = self.get_latest_data()
            if data and data.get("spot", 0) > 0:
                return data
            time.sleep(interval)
        return None
