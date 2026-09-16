import re
import os
import sys

def sync_html_files():
    targets = ['templates/unified_dashboard.html', 'unified_dashboard.html', 'unified_dashboard_fragment.html']

    for path in targets:
        if not os.path.exists(path):
            print(f"File not found: {path}")
            continue

        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()

        orig_len = len(content)

        # 1. Update Section 3 Header & View Switchers
        # Look for MASTER OPTION CHAIN header
        old_chain_header_pattern = r'(<div style="font-size:14px;font-weight:900;letter-spacing:1\.5px;color:#00e5ff;">MASTER OPTION CHAIN &amp; GEX WALL MATRIX</div>\s*<span[^>]*>[^<]*</span>\s*</div>)'
        view_switcher_html = '''\\1
                                <div style="display:flex;gap:6px;align-items:center;flex-wrap:wrap;">
                                    <span style="font-size:10px;color:#64748b;font-weight:700;margin-right:4px;">VIEW:</span>
                                    <button type="button" id="btn-chain-all" class="chain-mode-btn active" onclick="setChainMode('all')" style="padding:4px 10px;font-size:10px;font-weight:800;border-radius:4px;background:rgba(0,229,255,0.15);color:#00e5ff;border:1px solid #00e5ff;cursor:pointer;">COMPREHENSIVE</button>
                                    <button type="button" id="btn-chain-seller" class="chain-mode-btn" onclick="setChainMode('seller')" style="padding:4px 10px;font-size:10px;font-weight:800;border-radius:4px;background:rgba(255,255,255,0.05);color:#94a3b8;border:1px solid #333a60;cursor:pointer;">PRO SELLER</button>
                                    <button type="button" id="btn-chain-gex" class="chain-mode-btn" onclick="setChainMode('gex')" style="padding:4px 10px;font-size:10px;font-weight:800;border-radius:4px;background:rgba(255,255,255,0.05);color:#94a3b8;border:1px solid #333a60;cursor:pointer;">CLASSIC GEX</button>
                                </div>'''

        if 'id="btn-chain-all"' not in content:
            content = re.sub(old_chain_header_pattern, view_switcher_html, content, count=1)

        # Update table tag and thead if not already updated
        old_thead_pattern = r'<table class="data-table" style="width:100%;margin:0;border-collapse:collapse;font-size:12px;">\s*<thead[^>]*>.*?</thead>'
        new_thead_html = '''<table id="master-chain-table" class="data-table" style="width:100%;margin:0;border-collapse:collapse;font-size:12px;">
                                    <thead style="position:sticky;top:0;z-index:10;background:#0d1124;box-shadow:0 2px 8px rgba(0,0,0,0.7);">
                                        <tr style="border-bottom:1px solid #222744;">
                                            <th id="th-calls-header" colspan="8" style="text-align:center;color:#ff5252;background:rgba(255,51,102,0.1);letter-spacing:1px;font-size:11px;padding:6px;">CALLS (RESISTANCE / PREMIUM SELLERS)</th>
                                            <th style="text-align:center;color:#00e5ff;background:rgba(0,229,255,0.12);letter-spacing:1px;font-size:11px;padding:6px;">STRIKE</th>
                                            <th id="th-puts-header" colspan="8" style="text-align:center;color:#00e676;background:rgba(0,230,118,0.1);letter-spacing:1px;font-size:11px;padding:6px;">PUTS (SUPPORT / PREMIUM SELLERS)</th>
                                        </tr>
                                        <tr style="border-bottom:1px solid #333a60;font-size:11px;color:#94a3b8;">
                                            <th class="col-seller" style="text-align:center;padding:7px 6px;">Signal</th>
                                            <th class="col-seller" style="text-align:right;padding:7px 6px;">P(OTM)</th>
                                            <th class="col-seller" style="text-align:right;padding:7px 6px;">Theta</th>
                                            <th class="col-gex" style="text-align:right;padding:7px 8px;">Call GEX</th>
                                            <th class="col-gex" style="text-align:right;padding:7px 8px;">Call OI</th>
                                            <th class="col-base" style="text-align:right;padding:7px 8px;">15m Vel</th>
                                            <th class="col-base" style="text-align:right;padding:7px 8px;">Call LTP</th>
                                            <th class="col-base" style="text-align:right;padding:7px 8px;">IV</th>
                                            <th class="col-strike" style="text-align:center;padding:7px 12px;color:#fff;font-weight:800;">Strike Price</th>
                                            <th class="col-base" style="text-align:left;padding:7px 8px;">IV</th>
                                            <th class="col-base" style="text-align:left;padding:7px 8px;">Put LTP</th>
                                            <th class="col-base" style="text-align:right;padding:7px 8px;">15m Vel</th>
                                            <th class="col-gex" style="text-align:right;padding:7px 8px;">Put OI</th>
                                            <th class="col-gex" style="text-align:right;padding:7px 8px;">Put GEX</th>
                                            <th class="col-seller" style="text-align:right;padding:7px 6px;">Theta</th>
                                            <th class="col-seller" style="text-align:right;padding:7px 6px;">P(OTM)</th>
                                            <th class="col-seller" style="text-align:center;padding:7px 6px;">Signal</th>
                                        </tr>
                                    </thead>'''
        if 'id="master-chain-table"' not in content:
            content = re.sub(old_thead_pattern, new_thead_html, content, flags=re.DOTALL, count=1)

        # 2. Update Section 4: Replace Split Grid with Full-Width GEX Distribution Card
        # Look for section 4 grid
        old_sec4_pattern = r'<!-- 4\. GEX DISTRIBUTION BAR CHART & STRIKE SELECTION TABLE \(Grid\) -->\s*<div style="display:grid;grid-template-columns:1\.2fr 1fr;gap:12px;margin-bottom:14px;">\s*<div class="card"[^>]*>(.*?)</div>\s*<div class="card" style="max-height:560px;overflow-y:auto;padding:12px 14px;">\s*<div[^>]*>STRIKE SELECTION TABLE.*?</div>\s*<table.*?</table>\s*</div>\s*</div>'

        def sec4_replacement(m):
            inner_chart = m.group(1).strip()
            return f'''<!-- 4. INSTITUTIONAL GEX DISTRIBUTION BAR CHART & VOLATILITY CONTOURS -->
                        <div class="card" style="margin-bottom:14px;padding:16px 20px;border:1px solid rgba(0,229,255,0.25);border-radius:10px;box-shadow:0 6px 24px rgba(0,0,0,0.4);">
                            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px;flex-wrap:wrap;gap:8px;">
                                <div style="display:flex;align-items:center;gap:8px;">
                                    <div style="width:8px;height:8px;border-radius:50%;background:#00e5ff;box-shadow:0 0 8px #00e5ff;"></div>
                                    <div style="font-size:13px;font-weight:900;letter-spacing:1.5px;color:#00e5ff;">
                                        INSTITUTIONAL GEX DISTRIBUTION &amp; VOLATILITY CONTOURS
                                    </div>
                                </div>
                                <span style="font-size:11px;color:#94a3b8;font-family:'JetBrains Mono',monospace;">
                                    Full Width Contours · Strike Selection Merged in Master Chain
                                </span>
                            </div>
                            {inner_chart}
                        </div>'''

        content = re.sub(old_sec4_pattern, sec4_replacement, content, flags=re.DOTALL, count=1)

        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)

        print(f"Updated {path} (length: {orig_len} -> {len(content)})")

if __name__ == '__main__':
    sync_html_files()
