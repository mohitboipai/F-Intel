with open("unified_dashboard.html", "r", encoding="utf-8") as f:
    for i, line in enumerate(f, 1):
        if 'id="tab-' in line:
            print(f"{i}: {line.strip()[:80]}")
