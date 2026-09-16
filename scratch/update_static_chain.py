import re

def update_files():
    # Full exact definition of OPTION BUYER RADAR card
    gr_html = '''<!-- 5. OPTION BUYER RADAR: GEX SPOT REBALANCE & VACUUM RUNWAY ENGINE -->
<div id="gex-rebalance-card" class="card" style="margin-bottom:14px;border:1px solid rgba(0,240,255,0.3);background:linear-gradient(135deg, rgba(13,17,38,0.95), rgba(18,24,54,0.95));box-shadow:0 4px 24px rgba(0,0,0,0.4);border-radius:10px;padding:16px;">
    <!-- Top Header: Title, Status Badge, State Description -->
    <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:10px;margin-bottom:14px;border-bottom:1px solid rgba(255,255,255,0.08);padding-bottom:12px;">
        <div style="display:flex;align-items:center;gap:10px;">
            <div style="width:10px;height:10px;border-radius:50%;background:#00f0ff;box-shadow:0 0 10px #00f0ff;animation:pulseBadgeCyan 1.5s infinite;"></div>
            <div>
                <div style="font-size:14px;font-weight:900;color:#00f0ff;letter-spacing:1.5px;display:flex;align-items:center;gap:8px;">
                    <span>⚡ OPTION BUYER RADAR</span>
                    <span style="font-size:10px;color:#cbd5e1;background:rgba(255,255,255,0.08);padding:2px 8px;border-radius:4px;font-weight:600;">GEX VACUUM REBALANCE</span>
                </div>
                <div id="gr-desc" style="font-size:12px;color:#94a3b8;margin-top:2px;">
                    Monitoring Wall ② vs Wall ① distance and dealer delta-hedging vacuum runway...
                </div>
            </div>
        </div>
        <div style="display:flex;align-items:center;gap:8px;">
            <span id="gr-status-badge" style="font-size:12px;font-weight:900;padding:5px 14px;border-radius:6px;background:rgba(255,213,79,0.15);color:#ffd54f;border:1px solid #ffd54f;letter-spacing:0.5px;text-transform:uppercase;">
                COILING AT WALL ②
            </span>
            <span id="gr-direction-badge" style="font-size:11px;font-weight:800;padding:5px 12px;border-radius:6px;background:rgba(0,230,118,0.15);color:#00e676;border:1px solid rgba(0,230,118,0.3);">
                CALL BUY
            </span>
        </div>
    </div>

    <!-- The 4 Core Levels Grid (Zero Math - Direct Clean Numbers) -->
    <div style="display:grid;grid-template-columns:repeat(auto-fit, minmax(180px, 1fr));gap:10px;margin-bottom:16px;">
        <!-- Level 1: Current Spot -->
        <div class="metric-box" style="padding:10px 12px;background:rgba(255,255,255,0.03);border:1px solid #222744;border-radius:8px;">
            <div class="metric-label" style="font-size:10px;color:#94a3b8;">CURRENT SPOT</div>
            <div id="gr-spot-val" style="font-size:22px;font-weight:900;color:#ffffff;font-family:'JetBrains Mono',monospace;">--</div>
            <div id="gr-spot-sub" class="metric-sub" style="color:#64748b;">Live NIFTY Index</div>
        </div>

        <!-- Level 2: Ignition Toll Gate (Wall 2) -->
        <div class="metric-box" style="padding:10px 12px;background:rgba(255,213,79,0.06);border:1px solid rgba(255,213,79,0.3);border-radius:8px;">
            <div class="metric-label" style="font-size:10px;color:#ffd54f;">IGNITION TRIGGER (WALL ②)</div>
            <div id="gr-trigger-val" style="font-size:22px;font-weight:900;color:#ffd54f;font-family:'JetBrains Mono',monospace;">--</div>
            <div id="gr-trigger-sub" class="metric-sub" style="color:#ffd54f;">Breakout Toll Gate</div>
        </div>

        <!-- Level 3: Rebalance Target (Zero-Gamma Fuel Apex) -->
        <div class="metric-box" style="padding:10px 12px;background:rgba(0,240,255,0.06);border:1px solid rgba(0,240,255,0.35);border-radius:8px;">
            <div class="metric-label" style="font-size:10px;color:#00f0ff;">REBALANCE TARGET (FUEL APEX)</div>
            <div id="gr-target-val" style="font-size:22px;font-weight:900;color:#00f0ff;font-family:'JetBrains Mono',monospace;">--</div>
            <div id="gr-target-sub" class="metric-sub" style="color:#38bdf8;">Where Dealer Buying Peaks</div>
        </div>

        <!-- Level 4: Terminal Wall 1 Pin (Fortress) -->
        <div class="metric-box" style="padding:10px 12px;background:rgba(255,68,68,0.06);border:1px solid rgba(255,68,68,0.3);border-radius:8px;">
            <div class="metric-label" style="font-size:10px;color:#ff7043;">FORTRESS PIN (WALL ①)</div>
            <div id="gr-fortress-val" style="font-size:22px;font-weight:900;color:#ff7043;font-family:'JetBrains Mono',monospace;">--</div>
            <div id="gr-fortress-sub" class="metric-sub" style="color:#ff8a65;">Hard Terminal Barrier</div>
        </div>
    </div>

    <!-- Runway Progress Visual Gauge -->
    <div style="background:rgba(10,14,30,0.8);border:1px solid #1e2442;border-radius:8px;padding:12px 16px;margin-bottom:16px;">
        <div style="display:flex;justify-content:space-between;align-items:center;font-size:11px;color:#94a3b8;margin-bottom:8px;">
            <span style="display:flex;align-items:center;gap:6px;">
                <span style="color:#ffd54f;font-weight:700;">Wall ②: <span id="gr-bar-start">--</span></span>
                <span>&rarr;</span>
                <span style="color:#00f0ff;font-weight:700;">Runway: <span id="gr-bar-runway">-- pts</span></span>
            </span>
            <span style="font-weight:800;color:#00e676;">
                Rebalance Progress: <span id="gr-progress-pct">0%</span>
            </span>
            <span style="color:#ff7043;font-weight:700;">Target: <span id="gr-bar-target">--</span></span>
        </div>
        <div style="height:10px;background:#0d1124;border-radius:5px;overflow:hidden;position:relative;border:1px solid #222744;">
            <div id="gr-progress-fill" style="width:0%;height:100%;background:linear-gradient(90deg, #ffd54f, #00f0ff, #00e676);box-shadow:0 0 10px #00f0ff;transition:width 0.5s ease-in-out;"></div>
        </div>
        <div style="display:flex;justify-content:space-between;font-size:10px;color:#64748b;margin-top:5px;">
            <span>Trigger Level</span>
            <span id="gr-fuel-indicator" style="color:#38bdf8;">Dealer Futures Fuel: -- Lots</span>
            <span>Zero-Gamma Apex</span>
        </div>
    </div>

    <!-- Dual Strike Recommendation Grid (Primary ATM + 0DTE OTM Gamma Rocket) -->
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-bottom:14px;">
        <!-- Option 1: Primary ATM Strike -->
        <div style="background:rgba(18,22,46,0.9);border:1px solid rgba(0,240,255,0.25);border-radius:8px;padding:14px;">
            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;">
                <div>
                    <span style="font-size:10px;font-weight:800;color:#38bdf8;background:rgba(0,240,255,0.12);padding:2px 6px;border-radius:4px;">OPTION ① · PRIMARY ATM</span>
                    <div id="gr-p-strike-name" style="font-size:18px;font-weight:900;color:#ffffff;margin-top:4px;">-- CE</div>
                </div>
                <div style="text-align:right;">
                    <div style="font-size:10px;color:#94a3b8;">LTP (Current)</div>
                    <div id="gr-p-ltp" style="font-size:18px;font-weight:900;color:#00f0ff;">₹--</div>
                </div>
            </div>
            <!-- Pricing Grid -->
            <div style="display:grid;grid-template-columns:repeat(4, 1fr);gap:6px;text-align:center;">
                <div style="background:rgba(255,255,255,0.03);padding:6px;border-radius:4px;border:1px solid #1e2442;">
                    <div style="font-size:9px;color:#94a3b8;">BUY ZONE</div>
                    <div id="gr-p-buy" style="font-size:13px;font-weight:800;color:#ffffff;">₹--</div>
                </div>
                <div style="background:rgba(0,230,118,0.06);padding:6px;border-radius:4px;border:1px solid rgba(0,230,118,0.2);">
                    <div style="font-size:9px;color:#00e676;">TARGET ①</div>
                    <div id="gr-p-t1" style="font-size:13px;font-weight:800;color:#00e676;">₹--</div>
                    <div id="gr-p-t1-pct" style="font-size:9px;color:#00e676;">+--%</div>
                </div>
                <div style="background:rgba(0,240,255,0.06);padding:6px;border-radius:4px;border:1px solid rgba(0,240,255,0.2);">
                    <div style="font-size:9px;color:#00f0ff;">RUNNER T2</div>
                    <div id="gr-p-t2" style="font-size:13px;font-weight:800;color:#00f0ff;">₹--</div>
                    <div id="gr-p-t2-pct" style="font-size:9px;color:#00f0ff;">+--%</div>
                </div>
                <div style="background:rgba(255,68,68,0.06);padding:6px;border-radius:4px;border:1px solid rgba(255,68,68,0.2);">
                    <div style="font-size:9px;color:#ef5350;">STOP LOSS</div>
                    <div id="gr-p-sl" style="font-size:13px;font-weight:800;color:#ef5350;">₹--</div>
                    <div id="gr-p-sl-pct" style="font-size:9px;color:#ef5350;">---%</div>
                </div>
            </div>
            <div style="font-size:10px;color:#64748b;margin-top:8px;display:flex;justify-content:space-between;">
                <span>Steady Delta ~0.50</span>
                <span>Low Decay Risk</span>
            </div>
        </div>

        <!-- Option 2: 0DTE OTM Gamma Rocket Strike -->
        <div style="background:rgba(26,18,36,0.9);border:1px solid rgba(255,112,67,0.3);border-radius:8px;padding:14px;">
            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;">
                <div>
                    <span style="font-size:10px;font-weight:800;color:#ff7043;background:rgba(255,112,67,0.15);padding:2px 6px;border-radius:4px;">OPTION ② · 0DTE GAMMA ROCKET 🔥</span>
                    <div id="gr-o-strike-name" style="font-size:18px;font-weight:900;color:#ffffff;margin-top:4px;">-- CE</div>
                </div>
                <div style="text-align:right;">
                    <div style="font-size:10px;color:#94a3b8;">LTP (Cheap Entry)</div>
                    <div id="gr-o-ltp" style="font-size:18px;font-weight:900;color:#ff7043;">₹--</div>
                </div>
            </div>
            <!-- Pricing Grid -->
            <div style="display:grid;grid-template-columns:repeat(4, 1fr);gap:6px;text-align:center;">
                <div style="background:rgba(255,255,255,0.03);padding:6px;border-radius:4px;border:1px solid #1e2442;">
                    <div style="font-size:9px;color:#94a3b8;">BUY ZONE</div>
                    <div id="gr-o-buy" style="font-size:13px;font-weight:800;color:#ffffff;">₹--</div>
                </div>
                <div style="background:rgba(0,230,118,0.06);padding:6px;border-radius:4px;border:1px solid rgba(0,230,118,0.2);">
                    <div style="font-size:9px;color:#00e676;">EXPLOSION T1</div>
                    <div id="gr-o-t1" style="font-size:13px;font-weight:800;color:#00e676;">₹--</div>
                    <div id="gr-o-t1-pct" style="font-size:9px;color:#00e676;">+--%</div>
                </div>
                <div style="background:rgba(0,240,255,0.06);padding:6px;border-radius:4px;border:1px solid rgba(0,240,255,0.2);">
                    <div style="font-size:9px;color:#00f0ff;">JACKPOT T2</div>
                    <div id="gr-o-t2" style="font-size:13px;font-weight:800;color:#00f0ff;">₹--</div>
                    <div id="gr-o-t2-pct" style="font-size:9px;color:#00f0ff;">+--%</div>
                </div>
                <div style="background:rgba(255,68,68,0.06);padding:6px;border-radius:4px;border:1px solid rgba(255,68,68,0.2);">
                    <div style="font-size:9px;color:#ef5350;">STOP LOSS</div>
                    <div id="gr-o-sl" style="font-size:13px;font-weight:800;color:#ef5350;">₹--</div>
                    <div id="gr-o-sl-pct" style="font-size:9px;color:#ef5350;">---%</div>
                </div>
            </div>
            <div style="font-size:10px;color:#ff8a65;margin-top:8px;display:flex;justify-content:space-between;">
                <span>Convexity Hero · 3x-6x Potential</span>
                <span id="gr-o-status-tag" style="color:#00e676;">Runway Inside Vacuum</span>
            </div>
        </div>
    </div>

    <!-- Bottom Action Summary Strip (Executive One-Sentence Instruction) -->
    <div style="background:rgba(0,0,0,0.4);border:1px solid rgba(255,255,255,0.08);border-radius:6px;padding:10px 14px;display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:8px;">
        <div style="display:flex;align-items:center;gap:8px;">
            <span style="font-size:11px;font-weight:800;color:#ffd54f;">⚡ ACTION INSTRUCTION:</span>
            <span id="gr-action-text" style="font-size:12px;font-weight:700;color:#ffffff;">Monitoring option chain for Wall ② proximity...</span>
        </div>
        <div style="font-size:10px;color:#64748b;" id="gr-update-ts">
            Updated: --:--:--
        </div>
    </div>
</div>'''

    raw_chain_data = [
        (22950, 53755, "+1,820", "+92,820", 4910230),
        (23000, 508365, "+2,340", "-70,785", 16386890),
        (23050, 145600, "+6,110", "+153,595", 6383000),
        (23100, 533455, "+33,605", "+314,665", 14064245),
        (23150, 485615, "+37,960", "+107,835", 10347740),
        (23200, 3331120, "+278,785", "+1,085,695", 18720260),
        (23250, 3759665, "+50,050", "+1,315,145", 16569345),
        (23300, 16985540, "+992,290", "+839,995", 27039610),
        (23350, 26355940, "+2,267,980", "+188,695", 15860325),
        (23400, 39326430, "+1,016,015", "-1,164,020", 11096995),
        (23450, 29284385, "-671,125", "-783,510", 4566965),
        (23500, 47836490, "+327,080", "-95,810", 4853420),
        (23550, 20307495, "+70,070", "-36,790", 1429090),
        (23600, 21093995, "-557,050", "-42,640", 1908595),
        (23650, 14714765, "+225,355", "-16,185", 592215),
        (23700, 15506335, "+264,485", "-8,060", 1336985),
        (23750, 10421450, "+505,245", "-8,125", 396760),
    ]

    LOT_SIZE = 65
    parsed_rows = []
    for sk, ce_sh, ce_v, pe_v, pe_sh in raw_chain_data:
        parsed_rows.append({
            'strike': sk,
            'ce_oi_shares': ce_sh,
            'ce_oi_lots': round(ce_sh / LOT_SIZE),
            'ce_vel': ce_v,
            'pe_vel': pe_v,
            'pe_oi_shares': pe_sh,
            'pe_oi_lots': round(pe_sh / LOT_SIZE),
        })

    gex_map = {
        22950: 53901,
        23000: 254147,
        23050: 143577,
        23100: 474835,
        23150: 546119,
        23200: 1286145,
        23250: 1494180,
        23300: 1608026,
        23350: -1220100,
        23400: -3452298,
        23450: -2537258,
        23500: -3366812,
        23550: -1064215,
        23600: -797828,
        23650: -378538,
        23700: -280157,
        23750: -140051,
    }

    ce_price_map = {
        22950: 405.00,
        23000: 356.00,
        23050: 308.00,
        23100: 260.00,
        23150: 214.00,
        23200: 169.00,
        23250: 126.50,
        23300: 88.20,
        23350: 55.55,
        23400: 35.10,
        23450: 21.70,
        23500: 13.70,
        23550: 8.00,
        23600: 4.95,
        23650: 3.05,
        23700: 2.15,
        23750: 1.70,
    }
    pe_price_map = {
        22950: 1.15,
        23000: 1.65,
        23050: 2.20,
        23100: 3.40,
        23150: 6.05,
        23200: 11.05,
        23250: 20.00,
        23300: 34.65,
        23350: 56.40,
        23400: 84.10,
        23450: 118.00,
        23500: 157.00,
        23550: 200.00,
        23600: 246.00,
        23650: 293.00,
        23700: 342.00,
        23750: 391.00
    }

    iv_map = {
        22950: (14.2, 13.8),
        23000: (14.0, 13.6),
        23050: (13.8, 13.4),
        23100: (13.6, 13.2),
        23150: (13.5, 13.0),
        23200: (13.3, 12.8),
        23250: (13.2, 12.6),
        23300: (13.0, 12.5),
        23350: (12.9, 12.4),
        23400: (12.8, 12.5),
        23450: (12.9, 12.7),
        23500: (13.1, 13.0),
        23550: (13.3, 13.3),
        23600: (13.5, 13.6),
        23650: (13.7, 13.9),
        23700: (14.0, 14.2),
        23750: (14.3, 14.5)
    }

    spot = 23344.0
    call_wall_1 = 23500
    put_wall_1 = 23300
    call_wall_2 = 23400
    put_wall_2 = 23200
    max_pain = 23350

    max_ce_oi = max([r['ce_oi_lots'] for r in parsed_rows] + [1])
    max_pe_oi = max([r['pe_oi_lots'] for r in parsed_rows] + [1])

    # Build Master Option Chain Table Rows
    oi_rows_html = ''
    for r in parsed_rows:
        sk = r['strike']
        is_cw1 = (sk == call_wall_1)
        is_pw1 = (sk == put_wall_1)
        is_cw2 = (sk == call_wall_2)
        is_pw2 = (sk == put_wall_2)
        is_mp  = (sk == max_pain)
        is_atm = abs(sk - spot) < 30

        if is_cw1:
            row_style = "background:rgba(255,51,102,0.18);box-shadow:inset 0 0 12px rgba(255,51,102,0.3);border-top:1.5px solid #ff3366;border-bottom:1.5px solid #ff3366;"
        elif is_pw1:
            row_style = "background:rgba(0,230,118,0.18);box-shadow:inset 0 0 12px rgba(0,230,118,0.3);border-top:1.5px solid #00e676;border-bottom:1.5px solid #00e676;"
        elif is_atm:
            row_style = "background:rgba(0,229,255,0.12);box-shadow:inset 0 0 8px rgba(0,229,255,0.25);border-top:1.5px solid #00e5ff;border-bottom:1.5px solid #00e5ff;"
        elif is_cw2:
            row_style = "background:rgba(255,51,102,0.08);border-top:1px solid rgba(255,51,102,0.25);border-bottom:1px solid rgba(255,51,102,0.25);"
        elif is_pw2:
            row_style = "background:rgba(0,230,118,0.08);border-top:1px solid rgba(0,230,118,0.25);border-bottom:1px solid rgba(0,230,118,0.25);"
        elif is_mp:
            row_style = "background:rgba(255,214,0,0.08);"
        else:
            row_style = "background:transparent;border-bottom:1px solid rgba(255,255,255,0.04);"

        ce_itm = (sk < spot)
        pe_itm = (sk > spot)
        ce_cell_bg = "rgba(255,255,255,0.025)" if ce_itm else "transparent"
        pe_cell_bg = "rgba(255,255,255,0.025)" if pe_itm else "transparent"

        ce_oi_pct = min(100, int((r['ce_oi_lots'] / max_ce_oi) * 100))
        pe_oi_pct = min(100, int((r['pe_oi_lots'] / max_pe_oi) * 100))
        ce_oi_bg = f"background:linear-gradient(to left, rgba(255,82,82,0.25) {ce_oi_pct}%, {ce_cell_bg} {ce_oi_pct}%);"
        pe_oi_bg = f"background:linear-gradient(to right, rgba(0,230,118,0.25) {pe_oi_pct}%, {pe_cell_bg} {pe_oi_pct}%);"

        ce_v = r['ce_vel']
        pe_v = r['pe_vel']
        ce_v_col = "#66bb6a" if '+' in ce_v else "#ff4444" if '-' in ce_v else "#888888"
        pe_v_col = "#66bb6a" if '+' in pe_v else "#ff4444" if '-' in pe_v else "#888888"

        net_g = gex_map.get(sk, 0)
        ce_gex_val = -abs(net_g) if sk >= spot else 0
        pe_gex_val = abs(net_g) if sk < spot else 0
        ce_gex_txt = f"{ce_gex_val:+,.0f}" if ce_gex_val != 0 else "·"
        pe_gex_txt = f"{pe_gex_val:+,.0f}" if pe_gex_val != 0 else "·"
        ce_gex_col = "#ff4444" if ce_gex_val < 0 else "#66bb6a" if ce_gex_val > 0 else "#888888"
        pe_gex_col = "#66bb6a" if pe_gex_val > 0 else "#ff4444" if pe_gex_val < 0 else "#888888"

        ce_p = ce_price_map.get(sk, 0.0)
        pe_p = pe_price_map.get(sk, 0.0)
        ce_price_txt = f"₹{ce_p:.2f}" if ce_p > 0 else "·"
        pe_price_txt = f"₹{pe_p:.2f}" if pe_p > 0 else "·"

        ce_iv, pe_iv = iv_map.get(sk, (13.0, 13.0))
        ce_iv_txt = f"{ce_iv:.1f}%"
        pe_iv_txt = f"{pe_iv:.1f}%"

        dist_from_spot = sk - spot
        dist_label = f"+{dist_from_spot:.0f}" if dist_from_spot > 0 else f"{dist_from_spot:.0f}"

        strike_badges = []
        if is_atm:
            strike_badges.append('<span style="background:rgba(0,229,255,0.25);color:#00e5ff;font-size:9px;font-weight:900;padding:2px 6px;border-radius:3px;border:1px solid #00e5ff;">◄ ATM</span>')
        if is_cw1:
            strike_badges.append('<span style="background:rgba(255,51,102,0.3);color:#ff3366;font-size:9px;font-weight:900;padding:2px 6px;border-radius:3px;border:1px solid #ff3366;box-shadow:0 0 8px #ff3366;">🔴 CALL WALL ①</span>')
        elif is_cw2:
            strike_badges.append('<span style="background:rgba(255,51,102,0.15);color:#ff6688;font-size:9px;font-weight:700;padding:1px 5px;border-radius:3px;border:1px solid rgba(255,51,102,0.4);">CALL WALL ②</span>')
        if is_pw1:
            strike_badges.append('<span style="background:rgba(0,230,118,0.3);color:#00e676;font-size:9px;font-weight:900;padding:2px 6px;border-radius:3px;border:1px solid #00e676;box-shadow:0 0 8px #00e676;">🟢 PUT WALL ①</span>')
        elif is_pw2:
            strike_badges.append('<span style="background:rgba(0,230,118,0.15);color:#55e088;font-size:9px;font-weight:700;padding:1px 5px;border-radius:3px;border:1px solid rgba(0,230,118,0.4);">PUT WALL ②</span>')
        if is_mp and not is_atm:
            strike_badges.append('<span style="background:rgba(255,214,0,0.2);color:#ffd54f;font-size:9px;font-weight:800;padding:2px 6px;border-radius:3px;border:1px solid #ffd54f;">🟡 MAX PAIN</span>')

        strike_badges_html = " ".join(strike_badges)
        strike_main_col = "#00e5ff" if is_atm else "#ff4444" if (is_cw1 or is_cw2) else "#66bb6a" if (is_pw1 or is_pw2) else "#ffd54f" if is_mp else "#ffffff"

        oi_rows_html += f'''<tr style="{row_style}">
<td style="text-align:right;padding:7px 8px;font-family:monospace;color:{ce_gex_col};background:{ce_cell_bg};font-weight:700;">{ce_gex_txt}</td>
<td style="text-align:right;padding:7px 8px;font-family:monospace;color:#ffffff;{ce_oi_bg};font-weight:600;">{r["ce_oi_lots"]:,}</td>
<td style="text-align:right;padding:7px 8px;font-family:monospace;color:{ce_v_col};background:{ce_cell_bg};font-weight:600;">{ce_v}</td>
<td style="text-align:right;padding:7px 8px;font-family:monospace;color:#ffffff;background:{ce_cell_bg};font-weight:700;">{ce_price_txt}</td>
<td style="text-align:right;padding:7px 8px;font-family:monospace;color:#ffd54f;background:{ce_cell_bg};font-size:11px;">{ce_iv_txt}</td>
<td style="text-align:center;padding:7px 12px;font-weight:900;background:rgba(18,22,46,0.9);border-left:1px solid rgba(255,255,255,0.08);border-right:1px solid rgba(255,255,255,0.08);">
  <div style="display:flex;align-items:center;justify-content:center;gap:6px;flex-wrap:wrap;">
    <span style="font-size:14px;color:{strike_main_col};">{sk}</span>
    <span style="font-size:10px;color:#64748b;">({dist_label})</span>
    {strike_badges_html}
  </div>
</td>
<td style="text-align:left;padding:7px 8px;font-family:monospace;color:#ffd54f;background:{pe_cell_bg};font-size:11px;">{pe_iv_txt}</td>
<td style="text-align:left;padding:7px 8px;font-family:monospace;color:#ffffff;background:{pe_cell_bg};font-weight:700;">{pe_price_txt}</td>
<td style="text-align:right;padding:7px 8px;font-family:monospace;color:{pe_v_col};background:{pe_cell_bg};font-weight:600;">{pe_v}</td>
<td style="text-align:right;padding:7px 8px;font-family:monospace;color:#ffffff;{pe_oi_bg};font-weight:600;">{r["pe_oi_lots"]:,}</td>
<td style="text-align:right;padding:7px 8px;font-family:monospace;color:{pe_gex_col};background:{pe_cell_bg};font-weight:700;">{pe_gex_txt}</td>
</tr>'''

    # Visual GEX Corridor Runway Bar
    corridor_pct = ((spot - put_wall_1) / (call_wall_1 - put_wall_1)) * 100.0
    corridor_html = f'''<!-- 2. VISUAL GEX CORRIDOR RUNWAY BAR -->
<div class="card" style="margin-bottom:14px;background:linear-gradient(135deg, rgba(13,17,38,0.95), rgba(18,24,54,0.95));border:1px solid rgba(0,229,255,0.25);border-radius:10px;padding:14px 18px;">
    <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px;margin-bottom:12px;">
        <div style="display:flex;align-items:center;gap:8px;">
            <div style="width:8px;height:8px;border-radius:50%;background:#00e5ff;box-shadow:0 0 8px #00e5ff;"></div>
            <span style="color:#00e5ff;font-size:12px;font-weight:900;letter-spacing:1.5px;">INSTITUTIONAL GEX TRADING CORRIDOR &amp; VOLATILITY RUNWAY</span>
        </div>
        <span style="font-size:11px;font-weight:800;color:#4fc3f7;background:#4fc3f722;padding:3px 10px;border-radius:12px;border:1px solid #4fc3f755;">
            🔒 RANGE BOUND PINNING ({corridor_pct:.0f}% from Floor)
        </span>
    </div>

    <div style="display:flex;justify-content:space-between;align-items:flex-end;margin-bottom:6px;font-size:12px;font-family:'JetBrains Mono',monospace;">
        <div>
            <span style="color:#00e676;font-weight:900;font-size:15px;">🟢 PUT WALL ①: {put_wall_1}</span>
            <span style="color:#94a3b8;font-size:11px;margin-left:6px;">(-{spot - put_wall_1:.0f} pts)</span>
            <div style="font-size:10px;color:#00e676;">Breakdown Trigger: &lt; {put_wall_1-25}</div>
        </div>
        <div style="text-align:center;">
            <span style="color:#ffffff;font-weight:900;font-size:16px;">◄ SPOT {spot:.0f} ►</span>
            <div style="font-size:10px;color:#38bdf8;">Corridor Position: {corridor_pct:.0f}%</div>
        </div>
        <div style="text-align:right;">
            <span style="color:#ff3366;font-weight:900;font-size:15px;">🔴 CALL WALL ①: {call_wall_1}</span>
            <span style="color:#94a3b8;font-size:11px;margin-left:6px;">(+{call_wall_1 - spot:.0f} pts)</span>
            <div style="font-size:10px;color:#ff3366;">Squeeze Trigger: &gt; {call_wall_1+25}</div>
        </div>
    </div>

    <div style="height:12px;background:#0d1124;border-radius:6px;position:relative;border:1px solid #222744;overflow:hidden;margin-bottom:10px;">
        <div style="position:absolute;left:0;top:0;bottom:0;width:{corridor_pct:.0f}%;background:linear-gradient(90deg, rgba(0,230,118,0.7), rgba(0,229,255,0.85));"></div>
        <div style="position:absolute;left:{corridor_pct:.0f}%;top:-2px;bottom:-2px;width:4px;background:#ffffff;box-shadow:0 0 10px #ffffff;border-radius:2px;transform:translateX(-50%);"></div>
    </div>

    <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px;font-size:11px;color:#94a3b8;padding-top:6px;border-top:1px solid rgba(255,255,255,0.06);">
        <span>Corridor Width: <strong style="color:#fff;">{call_wall_1 - put_wall_1} pts</strong></span>
        <span>Max Pain Magnet: <strong style="color:#ffd54f;">{max_pain}</strong></span>
        <span>Call Wall ②: <strong style="color:#ff7043;">{call_wall_2}</strong></span>
        <span>Put Wall ②: <strong style="color:#81c784;">{put_wall_2}</strong></span>
        <span>Gamma Regime: <strong style="color:#ff4444;">ACCELERATING ↕ (Breakout Risk)</strong></span>
        <span>15m OI Flow: <strong style="color:#66bb6a;">BULLISH (+20)</strong></span>
    </div>
</div>'''

    # Master Option Chain Matrix
    master_table_html = f'''<!-- 3. MASTER OPTION CHAIN & GEX WALL MATRIX -->
<div class="card" style="margin-bottom:14px;padding:0;overflow:hidden;border:1px solid rgba(0,229,255,0.2);border-radius:10px;box-shadow:0 6px 24px rgba(0,0,0,0.4);">
    <div style="display:flex;justify-content:space-between;align-items:center;padding:12px 18px;background:rgba(18,22,46,0.95);border-bottom:1px solid rgba(255,255,255,0.08);flex-wrap:wrap;gap:8px;">
        <div style="display:flex;align-items:center;gap:10px;">
            <div style="font-size:14px;font-weight:900;letter-spacing:1.5px;color:#00e5ff;">MASTER OPTION CHAIN &amp; GEX WALL MATRIX</div>
            <span style="font-size:10px;background:rgba(0,229,255,0.12);color:#00e5ff;padding:2px 8px;border-radius:4px;font-weight:700;">LIVE NIFTY · EXPIRY 2026-09-15</span>
        </div>
        <div style="display:flex;gap:12px;font-size:11px;font-family:'JetBrains Mono',monospace;flex-wrap:wrap;">
            <span style="color:#ff3366;">🔴 CALL WALL / RESISTANCE</span>
            <span style="color:#00e676;">🟢 PUT WALL / SUPPORT</span>
            <span style="color:#00e5ff;">🔵 ATM PIVOT</span>
            <span style="color:#ffd54f;">🟡 MAX PAIN PIN</span>
        </div>
    </div>
    <div style="max-height:560px;overflow-y:auto;">
        <table class="data-table" style="width:100%;margin:0;border-collapse:collapse;font-size:12px;">
            <thead style="position:sticky;top:0;z-index:10;background:#0d1124;box-shadow:0 2px 8px rgba(0,0,0,0.7);">
                <tr style="border-bottom:1px solid #222744;">
                    <th colspan="5" style="text-align:center;color:#ff5252;background:rgba(255,51,102,0.1);letter-spacing:1px;font-size:11px;padding:6px;">CALLS (RESISTANCE / DEALER OVERHEAD)</th>
                    <th style="text-align:center;color:#00e5ff;background:rgba(0,229,255,0.12);letter-spacing:1px;font-size:11px;padding:6px;">STRIKE</th>
                    <th colspan="5" style="text-align:center;color:#00e676;background:rgba(0,230,118,0.1);letter-spacing:1px;font-size:11px;padding:6px;">PUTS (SUPPORT / DEALER FLOOR)</th>
                </tr>
                <tr style="border-bottom:1px solid #333a60;font-size:11px;color:#94a3b8;">
                    <th style="text-align:right;padding:7px 8px;">Call GEX (Lots)</th>
                    <th style="text-align:right;padding:7px 8px;">Call OI (Lots)</th>
                    <th style="text-align:right;padding:7px 8px;">15m Vel</th>
                    <th style="text-align:right;padding:7px 8px;">Call LTP</th>
                    <th style="text-align:right;padding:7px 8px;">IV</th>
                    <th style="text-align:center;padding:7px 12px;color:#fff;font-weight:800;">Strike Price</th>
                    <th style="text-align:left;padding:7px 8px;">IV</th>
                    <th style="text-align:left;padding:7px 8px;">Put LTP</th>
                    <th style="text-align:right;padding:7px 8px;">15m Vel</th>
                    <th style="text-align:right;padding:7px 8px;">Put OI (Lots)</th>
                    <th style="text-align:right;padding:7px 8px;">Put GEX (Lots)</th>
                </tr>
            </thead>
            <tbody>
                {oi_rows_html}
            </tbody>
        </table>
    </div>
</div>'''

    # Extract clean GEX Bar Chart and Strike Table
    with open('templates/unified_dashboard.html', 'r', encoding='utf-8') as f:
        src_content = f.read()

    # Extract Plotly div from src_content
    plotly_match = re.search(r'(<div class="card" style="margin-top:12px;padding:8px 16px;">.*?INSTITUTIONAL GEX MARKET GUIDANCE.*?</div>\s*</div>\s*</div>)', src_content, re.DOTALL)
    if not plotly_match:
        plotly_match = re.search(r'(<div class="card"[^>]*>\s*<div[^>]*class="plotly-graph-div".*?INSTITUTIONAL GEX MARKET GUIDANCE.*?</div>\s*</div>\s*</div>)', src_content, re.DOTALL)
    
    # Strip any trailing </section> from plotly card if captured
    plotly_card = plotly_match.group(1) if plotly_match else ''
    plotly_card = re.sub(r'</section>.*', '', plotly_card, flags=re.DOTALL).strip()

    # Strike selection table
    st_match = re.search(r'(<div class="card" style="max-height:\d+px;overflow-y:auto;[^"]*">\s*<div[^>]*>STRIKE SELECTION TABLE.*?</table>\s*</div>)', src_content, re.DOTALL)
    st_card = st_match.group(1) if st_match else ''

    gex_and_strike_row = f'''<!-- 4. GEX DISTRIBUTION BAR CHART & STRIKE SELECTION TABLE (Grid) -->
<div style="display:grid;grid-template-columns:1.2fr 1fr;gap:12px;margin-bottom:14px;">
    {plotly_card}
    {st_card}
</div>'''

    km_html = f'''<!-- 1. KEY METRICS & SELL ZONES -->
<div class="card" style="margin-bottom:12px;">
    <div style="color:#4fc3f7;font-size:13px;font-weight:700;letter-spacing:2px;margin-bottom:14px;">KEY METRICS &amp; SELL ZONES — DTE 1 | 2026-09-15</div>
    <div style="display:flex;gap:10px;flex-wrap:wrap;margin-bottom:14px;">
        <div class="metric-box"><div class="metric-label">ATM IV</div><div style="font-size:20px;font-weight:700;color:#e0e0e0;">29.92%</div></div>
        <div class="metric-box"><div class="metric-label">Straddle</div><div style="font-size:20px;font-weight:700;color:#e0e0e0;">90</div></div>
        <div class="metric-box"><div class="metric-label">Exp. Move</div><div style="font-size:20px;font-weight:700;color:#e0e0e0;">±90 (0.4%)</div></div>
        <div class="metric-box"><div class="metric-label">Max Pain</div><div style="font-size:20px;font-weight:700;color:#ffd54f;">23350</div></div>
        <div class="metric-box" style="border-left:2px solid #66bb6a;">
            <div class="metric-label">Put Wall ①</div>
            <div style="font-size:20px;font-weight:700;color:#66bb6a;">23300</div>
            <div class="metric-sub" style="color:#66bb6a88;">② 23200</div>
        </div>
        <div class="metric-box" style="border-left:2px solid #ff4444;">
            <div class="metric-label">Call Wall ①</div>
            <div style="font-size:20px;font-weight:700;color:#ff4444;">23500</div>
            <div class="metric-sub" style="color:#ff525288;">② 23400</div>
        </div>
        <div class="metric-box"><div class="metric-label">15m OI VELOCITY</div><div style="font-size:20px;font-weight:700;color:#66bb6a;">BULLISH</div><div class="metric-sub">Score: +20</div></div>
        <div class="metric-box"><div class="metric-label">PCR</div><div style="font-size:20px;font-weight:700;color:#ff4444;">0.58</div><div class="metric-sub">Bearish (Call Writing)</div></div>
    </div>
    <div class="action-bar">
        <span style="color:#888888;font-size:12px;margin-right:8px;">⚡ SELL ZONES:</span>
        <span style="color:#66bb6a;font-size:16px;font-weight:700;">SELL CE &gt; 23435</span>
        <span style="margin:0 20px;color:#888888;">|</span>
        <span style="color:#ff4444;font-size:16px;font-weight:700;">SELL PE &lt; 23300</span>
    </div>
</div>'''

    new_tab_chain_body = f'''
{km_html}

{corridor_html}

{master_table_html}

{gex_and_strike_row}

{gr_html}
'''

    for target_path in ['templates/unified_dashboard.html', 'unified_dashboard.html']:
        with open(target_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Replace tab-chain section accurately
        chain_start = content.find('<section id="tab-chain"')
        theta_start = content.find('<section id="tab-theta"')
        if chain_start != -1 and theta_start != -1:
            new_content = content[:chain_start] + f'<section id="tab-chain" class="tab-content">{new_tab_chain_body}</section>\n        ' + content[theta_start:]
            with open(target_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
            print(f"Updated {target_path}")
        else:
            print(f"Could not find section boundaries in {target_path}")

    # Update unified_dashboard_fragment.html
    with open('unified_dashboard_fragment.html', 'r', encoding='utf-8') as f:
        frag_content = f.read()
    f_chain_start = frag_content.find('<div id="frag-chain">')
    f_theta_start = frag_content.find('<div id="frag-theta">')
    if f_chain_start != -1 and f_theta_start != -1:
        new_frag_content = frag_content[:f_chain_start] + f'<div id="frag-chain">{new_tab_chain_body}</div>\n' + frag_content[f_theta_start:]
        with open('unified_dashboard_fragment.html', 'w', encoding='utf-8') as f:
            f.write(new_frag_content)
        print("Updated unified_dashboard_fragment.html")
    else:
        print("Could not find fragment boundaries in unified_dashboard_fragment.html")

if __name__ == '__main__':
    update_files()
