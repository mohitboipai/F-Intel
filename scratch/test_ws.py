import websocket
import json

try:
    ws = websocket.create_connection("ws://127.0.0.1:8082/stream", timeout=5)
    print("Connected to WebSocket successfully!")
    msg = ws.recv()
    data = json.loads(msg)
    print("Received initial WS message:", data.get("type"), "keys:", list(data.get("data", {}).keys()))
    ws.close()
except Exception as e:
    print("WebSocket error:", e)
