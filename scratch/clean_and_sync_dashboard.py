import os
import re

def clean_file(filepath):
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return
    with open(filepath, 'r', encoding='utf-8') as f:
        html = f.read()

    # 1. Remove Institutional from titles and headers
    html = html.replace('INSTITUTIONAL GEX DISTRIBUTION &amp; VOLATILITY CONTOURS', 'GEX DISTRIBUTION &amp; VOLATILITY CONTOURS')
    html = html.replace('INSTITUTIONAL GEX DISTRIBUTION & VOLATILITY CONTOURS', 'GEX DISTRIBUTION & VOLATILITY CONTOURS')
    html = html.replace('INSTITUTIONAL GEX DISTRIBUTION', 'GEX DISTRIBUTION')
    html = html.replace('SPOT MOVEMENT TRAJECTORY &amp; INSTITUTIONAL BOUNDARIES', 'SPOT MOVEMENT TRAJECTORY &amp; REAL-TIME INTRADAY PROJECTION')
    html = html.replace('SPOT MOVEMENT TRAJECTORY & INSTITUTIONAL BOUNDARIES', 'SPOT MOVEMENT TRAJECTORY & REAL-TIME INTRADAY PROJECTION')
    html = html.replace('Institutional Gamma Exposure (GEX)', 'GEX DISTRIBUTION & Spot Movement Trajectory')
    html = html.replace('INSTITUTIONAL GEX &amp; DEALER POSITIONING', 'GEX &amp; DEALER POSITIONING')
    html = html.replace('INSTITUTIONAL GEX & DEALER POSITIONING', 'GEX & DEALER POSITIONING')
    html = html.replace('STRIKE GEX LADDER &amp; INSTITUTIONAL DISTRIBUTION', 'STRIKE GEX LADDER &amp; GEX DISTRIBUTION')
    html = html.replace('STRIKE GEX LADDER & INSTITUTIONAL DISTRIBUTION', 'STRIKE GEX LADDER & GEX DISTRIBUTION')
    html = html.replace('PER-STRIKE INSTITUTIONAL INVENTORY &amp; DEALER EXPOSURE', 'PER-STRIKE INVENTORY &amp; DEALER EXPOSURE')
    html = html.replace('PER-STRIKE INSTITUTIONAL INVENTORY & DEALER EXPOSURE', 'PER-STRIKE INVENTORY & DEALER EXPOSURE')
    html = html.replace('Institutional Role', 'Dealer Role / Wall')
    html = html.replace('<!-- Institutional Gamma Explosion Quick-Launch Banner -->', '<!-- Gamma Explosion Quick-Launch Banner -->')

    # 2. Add DOM IDs to Tab 5 cards if missing
    html = re.sub(r'<div class="card" style="border-top:4px solid ([^>]+);">\s*<div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:6px;">\s*<div style="color:[^>]+>DEALER GAMMA REGIME</div>\s*<span style="([^"]+)">([^<]+)</span>\s*</div>\s*<div style="font-size:20px;font-weight:800;color:[^>]+>([^<]+)</div>\s*<div style="font-size:11px;color:[^>]+>Net GEX per 100-pt move</div>\s*<div style="font-size:11px;color:#aaa;line-height:1.4;">([^<]+)</div>',
                  r'<div id="dealer-regime-card" class="card" style="border-top:4px solid \1;">\n<div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:6px;">\n<div style="color:#4fc3f7;font-size:11px;font-weight:800;letter-spacing:1px;">DEALER GAMMA REGIME</div>\n<span id="dealer-regime-badge" style="\2">\3</span>\n</div>\n<div id="dealer-gex-val" style="font-size:20px;font-weight:800;color:\1;margin-bottom:2px;">\4</div>\n<div style="font-size:11px;color:#888;margin-bottom:8px;">Net GEX per 100-pt move</div>\n<div id="dealer-regime-desc" style="font-size:11px;color:#aaa;line-height:1.4;">\5</div>',
                  html)

    # 3. Add live net gex badge and spot-move IDs if missing
    if 'id="live-net-gex-badge"' not in html:
        html = html.replace(
            '<span style="font-size:10px;font-weight:700;color:#ff4444;background:#ff444418;padding:3px 10px;border-radius:4px;border:1px solid #ff444444;">ACCELERATING ↕ (Breakout Risk)</span>',
            '<div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap;">\n<span id="spot-move-regime-badge" style="font-size:10px;font-weight:800;color:#ff4444;background:#ff444418;padding:3px 10px;border-radius:4px;border:1px solid #ff444444;">ACCELERATING ↕ (Breakout Risk)</span>\n<span id="live-net-gex-badge" style="font-size:10px;font-weight:800;color:#ff4444;background:rgba(255,255,255,0.06);padding:3px 10px;border-radius:4px;border:1px solid #ff444444;">LIVE NET GEX</span>\n</div>'
        )

    # 4. Add IDs to Put Wall, Call Wall, Zero-Gamma Flip, Expected Move, and Outlook Text
    html = re.sub(r'(🟢 PUT WALL ① [^<]*</div>\s*<div )style="font-size:15px;font-weight:800;', r'\1id="spot-move-put-wall" style="font-size:15px;font-weight:800;', html)
    html = re.sub(r'(🔴 CALL WALL ① [^<]*</div>\s*<div )style="font-size:15px;font-weight:800;', r'\1id="spot-move-call-wall" style="font-size:15px;font-weight:800;', html)
    html = re.sub(r'(⚡ ZERO-GAMMA FLIP LEVEL</div>\s*<div )style="font-size:15px;font-weight:800;', r'\1id="spot-move-flip-strike" style="font-size:15px;font-weight:800;', html)
    html = re.sub(r'(🎯 1-SIGMA EXPECTED MOVE</div>\s*<div )style="font-size:15px;font-weight:800;', r'\1id="spot-move-expected-move" style="font-size:15px;font-weight:800;', html)
    html = re.sub(r'(<span style="color:#00e5ff;font-weight:800;margin-right:4px;">PREDICTIVE OUTLOOK:</span>\s*)([^<]+)(</div>)', r'\1<span id="spot-move-outlook-text">\2</span>\3', html)

    # Write back
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"Successfully cleaned: {filepath}")

if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    clean_file(os.path.join(base_dir, 'templates', 'unified_dashboard.html'))
    clean_file(os.path.join(base_dir, 'unified_dashboard.html'))
