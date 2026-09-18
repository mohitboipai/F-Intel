import os

def update_iv_surface_tab(filepath):
    if not os.path.exists(filepath):
        return
    with open(filepath, 'r', encoding='utf-8') as f:
        html = f.read()

    # 1. Rename Tab Navigation
    html = html.replace(
        '<button class="tab-btn" data-tab="iv"><span>IV SURFACE &amp; SMILE</span></button>',
        '<button class="tab-btn" data-tab="iv"><span>IV SURFACE</span></button>'
    )
    html = html.replace(
        '<button class="tab-btn" data-tab="iv"><span>IV SURFACE & SMILE</span></button>',
        '<button class="tab-btn" data-tab="iv"><span>IV SURFACE</span></button>'
    )

    # 2. Dual Horizon Exhaustion Panel
    old_panel_start = '<!-- Mathematical Movement & Exhaustion Panel -->'
    old_panel_end = '<!-- Rewind Scrubber & Preset Controls -->'

    new_panel = '''<!-- Mathematical Movement & Exhaustion Panel -->
    <div class="iv-exhaustion-panel">
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:12px;">
            <div style="display:flex; align-items:center; gap:10px;">
                <div style="font-family:var(--font-mono); font-size:12px; font-weight:700; color:#00e5ff; letter-spacing:1px;">
                    QUANTITATIVE VOLATILITY BUDGET &amp; CONSUMPTION GAUGES
                </div>
                <span style="font-family:var(--font-mono); font-size:10px; color:#868993; background:rgba(255,255,255,0.06); padding:2px 8px; border-radius:3px;">1-DAY &amp; WEEKLY (1σ)</span>
            </div>
            <span id="iv-regime-tag" class="iv-regime-tag consolidation">CALCULATING</span>
        </div>

        <div class="iv-exhaustion-grid">
            <!-- 1-Day Expected Move -->
            <div class="iv-metric-tile">
                <div class="iv-metric-label">EXPECTED 1-DAY MOVE (1σ)</div>
                <div id="iv-exp-move-pts" class="iv-metric-value">±-- pts</div>
                <div id="iv-exp-move-pct" class="iv-metric-sub">±--%</div>
            </div>
            <!-- Intraday Range -->
            <div class="iv-metric-tile">
                <div class="iv-metric-label">INTRADAY RANGE (H - L)</div>
                <div id="iv-range-pts" class="iv-metric-value">-- pts</div>
                <div id="iv-range-sub" class="iv-metric-sub">H: -- | L: --</div>
            </div>
            <!-- 1-Day 1σ Boundaries -->
            <div class="iv-metric-tile">
                <div class="iv-metric-label">1-DAY 1σ BOUNDARIES</div>
                <div style="display:flex; gap:12px; margin-top:2px;">
                    <div>
                        <span style="font-size:10px; color:#868993;">UPPER:</span>
                        <div id="iv-boundary-upper" style="font-family:var(--font-mono); font-size:13px; font-weight:700; color:#ffa726;">--</div>
                    </div>
                    <div>
                        <span style="font-size:10px; color:#868993;">LOWER:</span>
                        <div id="iv-boundary-lower" style="font-family:var(--font-mono); font-size:13px; font-weight:700; color:#ffa726;">--</div>
                    </div>
                </div>
                <div class="iv-metric-sub" style="margin-top:4px;">Daily standard deviation band</div>
            </div>
            <!-- Weekly Expected Move -->
            <div class="iv-metric-tile" style="border-left: 2px solid rgba(124, 77, 255, 0.4);">
                <div class="iv-metric-label" style="color:#b388ff;">WEEKLY EXPECTED MOVE (1σ)</div>
                <div id="iv-weekly-exp-pts" class="iv-metric-value" style="color:#e0e0ff;">±-- pts</div>
                <div id="iv-weekly-exp-pct" class="iv-metric-sub">±--% (5-Day Horizon)</div>
            </div>
            <!-- Weekly Realized & 5D Range -->
            <div class="iv-metric-tile" style="border-left: 2px solid rgba(124, 77, 255, 0.4);">
                <div class="iv-metric-label" style="color:#b388ff;">WEEKLY REALIZED MOVE &amp; RANGE</div>
                <div id="iv-weekly-realized-pts" class="iv-metric-value">-- pts</div>
                <div id="iv-weekly-range-sub" class="iv-metric-sub">5D Range: -- pts (H: -- | L: --)</div>
            </div>
            <!-- Weekly 1σ Boundaries -->
            <div class="iv-metric-tile" style="border-left: 2px solid rgba(124, 77, 255, 0.4);">
                <div class="iv-metric-label" style="color:#b388ff;">WEEKLY 1σ BOUNDARIES</div>
                <div style="display:flex; gap:12px; margin-top:2px;">
                    <div>
                        <span style="font-size:10px; color:#868993;">UPPER:</span>
                        <div id="iv-weekly-boundary-upper" style="font-family:var(--font-mono); font-size:13px; font-weight:700; color:#b388ff;">--</div>
                    </div>
                    <div>
                        <span style="font-size:10px; color:#868993;">LOWER:</span>
                        <div id="iv-weekly-boundary-lower" style="font-family:var(--font-mono); font-size:13px; font-weight:700; color:#b388ff;">--</div>
                    </div>
                </div>
                <div class="iv-metric-sub" style="margin-top:4px;">Expiry / Weekly volatility band</div>
            </div>
        </div>

        <!-- 1-Day Gauge Bar -->
        <div class="iv-gauge-wrap" style="margin-bottom:12px;">
            <div class="iv-gauge-labels">
                <span style="display:flex; align-items:center; gap:6px;">
                    <span style="color:#00e5ff; font-weight:700;">1-DAY BUDGET:</span>
                    <span>Spot Net Move: <b id="iv-consumption-pct-label" style="color:#00e5ff;">0%</b></span>
                    <span style="color:var(--text-muted);">|</span>
                    <span>Intraday Range Traversed: <b id="iv-range-consumption-label" style="color:#94a3b8;">0%</b></span>
                </span>
                <span style="color:#ffd54f;">100% (1σ Barrier)</span>
            </div>
            <div class="iv-gauge-track">
                <div id="iv-gauge-bar-range" class="iv-gauge-bar-range" title="Intraday High-Low Range Traversed"></div>
                <div id="iv-gauge-bar-fill" class="iv-gauge-bar" title="Current Spot Net Displacement"></div>
                <div class="iv-gauge-marker-1sigma" title="1-Sigma Statistical Barrier (80% scale)"></div>
            </div>
        </div>

        <!-- Weekly Gauge Bar -->
        <div class="iv-gauge-wrap">
            <div class="iv-gauge-labels">
                <span style="display:flex; align-items:center; gap:6px;">
                    <span style="color:#b388ff; font-weight:700;">WEEKLY BUDGET:</span>
                    <span>Weekly Realized Move: <b id="iv-weekly-consumption-label" style="color:#b388ff;">0%</b></span>
                    <span style="color:var(--text-muted);">|</span>
                    <span>5-Day Range Traversed: <b id="iv-weekly-range-consumption-label" style="color:#94a3b8;">0%</b></span>
                </span>
                <span style="color:#ffd54f;">100% (Weekly 1σ Barrier)</span>
            </div>
            <div class="iv-gauge-track">
                <div id="iv-weekly-bar-range" class="iv-gauge-bar-range" title="Weekly High-Low Range Traversed"></div>
                <div id="iv-weekly-bar-fill" class="iv-gauge-bar" title="Weekly Realized Net Displacement"></div>
                <div class="iv-gauge-marker-1sigma" title="Weekly 1-Sigma Statistical Barrier (80% scale)"></div>
            </div>
            <div class="iv-gauge-legend">
                <div class="iv-gauge-legend-items">
                    <span><span class="legend-box solid" style="background:#00e5ff;"></span> Solid Fill: Current Spot Net Move</span>
                    <span><span class="legend-box" style="background:rgba(0, 229, 255, 0.25); border:1px solid #00e5ff;"></span> Shaded Area: Total Range Traversed</span>
                </div>
                <span>Markers: Gold line indicates 1.0σ full volatility budget</span>
            </div>
        </div>
    </div>

    '''

    if old_panel_start in html and old_panel_end in html:
        p_start = html.index(old_panel_start)
        p_end = html.index(old_panel_end)
        html = html[:p_start] + new_panel + html[p_end:]

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"Updated {filepath}")

if __name__ == '__main__':
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    update_iv_surface_tab(os.path.join(base, 'templates', 'unified_dashboard.html'))
    update_iv_surface_tab(os.path.join(base, 'unified_dashboard.html'))
