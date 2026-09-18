import os
import sys

# Search in current directory (non-venv)
for root, dirs, files in os.walk('.'):
    if '.venv' in root or '.git' in root or '__pycache__' in root:
        continue
    for file in files:
        if file.endswith(('.py', '.html', '.js')):
            path = os.path.join(root, file)
            if 'VolatilityAnalyzer.py' in path and not path.endswith('.bak'):
                try:
                    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                        for idx, line in enumerate(f, 1):
                            if 'oi_vel' in line or 'velocity' in line.lower():
                                print(f"{path}:{idx}: {line.strip()[:120]}".encode('ascii', 'replace').decode())
                except Exception as e:
                    pass
                pass
