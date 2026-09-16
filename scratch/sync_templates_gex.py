import re

def update_html(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        text = f.read()

    orig_len = len(text)
    print(f"Processing {filepath} (len: {orig_len})...")

    # 1. Remove Corridor Runway card: <!-- 2. VISUAL GEX CORRIDOR RUNWAY BAR --> ... <!-- 3. MASTER OPTION CHAIN
    corridor_pattern = re.compile(
        r'\s*<!-- 2\. VISUAL GEX CORRIDOR RUNWAY BAR -->\s*<div class="card".*?</div>\s*</div>(?=\s*<!-- 3\. MASTER OPTION CHAIN)',
        re.DOTALL
    )
    if corridor_pattern.search(text):
        text = corridor_pattern.sub('\n\n                        <!-- 2. MASTER OPTION CHAIN', text)
        print("  Removed corridor card block.")
    else:
        # Try alternate pattern
        c_start = text.find('<!-- 2. VISUAL GEX CORRIDOR RUNWAY BAR -->')
        c_next = text.find('<!-- 3. MASTER OPTION CHAIN', c_start)
        if c_start != -1 and c_next != -1:
            text = text[:c_start] + '<!-- 2. MASTER OPTION CHAIN' + text[c_next + len('<!-- 3. MASTER OPTION CHAIN'):]
            print("  Removed corridor card block via index slicing.")

    # 2. Renumber remaining section comments if needed
    text = text.replace('<!-- 4. INSTITUTIONAL GEX DISTRIBUTION', '<!-- 3. INSTITUTIONAL GEX DISTRIBUTION')

    # 3. Master option chain headers
    # Update colspan="8" to colspan="7"
    text = re.sub(r'(id="th-calls-header"\s+colspan=)"8"', r'\1"7"', text)
    text = re.sub(r'(id="th-puts-header"\s+colspan=)"8"', r'\1"7"', text)

    # Remove P(OTM) headers from table
    text = re.sub(r'\s*<th class="col-seller" style="[^"]*">P\(OTM\)</th>', '', text)

    # 4. Remove P(OTM) cells from rows
    # In row, format is:
    # <tr>
    # <td class="col-seller" ...>Signal</td>
    # <td class="col-seller" ...>POTM</td>  <-- remove
    # <td class="col-seller" ...>Theta</td>
    # ...
    # <td class="col-seller" ...>Theta</td>
    # <td class="col-seller" ...>POTM</td>  <-- remove
    # <td class="col-seller" ...>Signal</td>
    # </tr>
    def clean_row(m):
        row_str = m.group(0)
        # Find all td.col-seller in this row
        seller_tds = list(re.finditer(r'<td class="col-seller"[^>]*>.*?</td>', row_str))
        if len(seller_tds) == 6:
            # 3 call seller cells, 3 put seller cells
            # remove index 1 (ce potm) and index 4 (pe potm)
            ce_potm = seller_tds[1].group(0)
            pe_potm = seller_tds[4].group(0)
            # Remove them from row
            # To avoid replacing wrong td if strings match, slice carefully
            # Slice from end to preserve indices:
            row_str = (
                row_str[:seller_tds[4].start()] +
                row_str[seller_tds[4].end():]
            )
            # Adjust index for ce_potm
            row_str = (
                row_str[:seller_tds[1].start()] +
                row_str[seller_tds[1].end():]
            )
        return row_str

    text = re.sub(r'<tr style="[^"]*">.*?</tr>', clean_row, text, flags=re.DOTALL)
    print(f"  Cleaned master table rows.")

    # 5. Wrap GEX distribution chart container with min-height 520px
    # Check if gex-chart-container already exists
    if '<div id="gex-chart-container"' not in text:
        # Find where the plotly chart or div is
        # Usually inside:
        # <!-- 3. INSTITUTIONAL GEX DISTRIBUTION BAR CHART & VOLATILITY CONTOURS -->
        # ...
        # </span>
        # </div>
        # <div style="height:520px... or <div id="gex-distribution-chart" or plotly div
        # <div style="margin-top:12px;padding:12px 14px... (gex_analysis_html)
        pattern_gex = re.compile(
            r'(<div style="font-size:13px;font-weight:900;letter-spacing:1\.5px;color:#00e5ff;">\s*INSTITUTIONAL GEX DISTRIBUTION &amp; VOLATILITY CONTOURS\s*</div>\s*</div>\s*<span style="font-size:11px;color:#94a3b8;font-family:\'JetBrains Mono\',monospace;">\s*Full Width Contours[^\n]*\s*</span>\s*</div>\s*)(.*?)(?=<div style="margin-top:1[02]px;)',
            re.DOTALL
        )
        m_gex = pattern_gex.search(text)
        if m_gex:
            header_part = m_gex.group(1)
            chart_body = m_gex.group(2).strip()
            wrapped_chart = f'\n                            <div id="gex-chart-container" style="min-height:520px; height:520px; width:100%; position:relative; overflow:hidden;">\n                                {chart_body}\n                            </div>\n                            '
            text = text[:m_gex.start()] + header_part + wrapped_chart + text[m_gex.end():]
            print("  Wrapped GEX chart with #gex-chart-container.")

    # Ensure plotly div has id="gex-distribution-chart" if not present
    if 'id="gex-distribution-chart"' not in text:
        # Find class="plotly-graph-div" inside gex section
        text = re.sub(r'class="plotly-graph-div"', 'id="gex-distribution-chart" class="plotly-graph-div"', text, count=1)
        print("  Added id='gex-distribution-chart' to plotly graph div.")

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(text)
    print(f"Saved {filepath} (new len: {len(text)})\n")

for target in ['templates/unified_dashboard.html', 'unified_dashboard.html']:
    update_html(target)
