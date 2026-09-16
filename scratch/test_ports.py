import urllib.request

for port in [8082, 62923, 50976, 52457]:
    try:
        url = f"http://127.0.0.1:{port}/fragment?t=123"
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=2) as resp:
            print(f"Port {port} /fragment -> Status {resp.status}, length {len(resp.read())}")
    except Exception as e:
        print(f"Port {port} /fragment -> Error: {e}")
