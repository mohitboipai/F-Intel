import os
import re

pat = re.compile(r'Call\s+unwinding|short-covering|bid\s+support|Put\s+liquidation', re.IGNORECASE)

for root, dirs, files in os.walk('.'):
    if '.venv' in root or '.git' in root or '__pycache__' in root:
        continue
    for f in files:
        p = os.path.join(root, f)
        try:
            with open(p, 'r', encoding='utf-8', errors='ignore') as fh:
                txt = fh.read()
                matches = pat.findall(txt)
                if len(matches) > 1:
                    print(f"Matched in {p}: {set(matches)}")
        except Exception as e:
            pass
print("Done regex scan.")
