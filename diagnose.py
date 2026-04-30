import urllib.request
import json

tests = [
    ('GET', '/'),
    ('GET', '/get_data'),
    ('GET', '/fragment'),
    ('GET', '/api/strategies'),
    ('GET', '/api/regime'),
    ('GET', '/api/gex'),
    ('GET', '/api/signals'),
    ('GET', '/bt_SHORT_STRADDLE.html'),
]

for method, url in tests:
    try:
        req = urllib.request.Request(f'http://localhost:8082{url}', method=method)
        r = urllib.request.urlopen(req, timeout=5)
        print(f'{method} {url} -> {r.status}')
    except Exception as e:
        print(f'{method} {url} -> ERROR: {e}')
