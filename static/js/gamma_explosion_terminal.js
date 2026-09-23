/**
 * static/js/gamma_explosion_terminal.js
 * Institutional Market Maker Gamma Pinning & Order Flow Absorption Terminal
 * Replicates the quantitative models from quantedoptions (strike-level gamma pins)
 * and aleksrosme (GEX level retest + order flow absorption = 100-pt explosion).
 */

(function () {
    'use strict';

    let _lastPayload = null;
    let _pollTimer = null;
    let _chartInitialized = false;

    async function fetchExplosionData() {
        try {
            const res = await fetch('/api/gamma/explosion?t=' + Date.now());
            if (!res.ok) return null;
            const data = await res.json();
            return data;
        } catch (e) {
            console.warn('[GammaExplosion] Fetch error:', e);
            return null;
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // 1. PLOTLY CANDLESTICK & GEX INTERACTION CHART (aleksrosme & quantedoptions)
    // ─────────────────────────────────────────────────────────────────────────
    function renderPlotlyChart(payload) {
        const chartEl = document.getElementById('ge-interactive-chart');
        if (!chartEl || typeof Plotly === 'undefined') return;

        const candles = payload.chart_candles || [];
        if (candles.length === 0) return;

        const times  = candles.map(c => c.time);
        const opens  = candles.map(c => c.open);
        const highs  = candles.map(c => c.high);
        const lows   = candles.map(c => c.low);
        const closes = candles.map(c => c.close);

        const candleTrace = {
            x: times,
            open: opens,
            high: highs,
            low: lows,
            close: closes,
            type: 'candlestick',
            name: 'NIFTY 1m',
            increasing: { line: { color: '#00e676', width: 1.5 }, fillcolor: '#00e676' },
            decreasing: { line: { color: '#ff2a6d', width: 1.5 }, fillcolor: '#ff2a6d' },
            showlegend: false
        };

        const shapes = [];
        const annotations = [];

        // 1. Call Wall (Resistance)
        if (payload.call_wall && payload.call_wall.strike) {
            const cw = payload.call_wall.strike;
            shapes.push({
                type: 'line', xref: 'paper', x0: 0, x1: 1,
                yref: 'y', y0: cw, y1: cw,
                line: { color: '#00e676', width: 1.8, dash: 'dash' }
            });
            annotations.push({
                xref: 'paper', x: 0.98, yref: 'y', y: cw,
                text: `CALL WALL: ${cw} (+₹${payload.call_wall.gex_cr || 0} Cr)`,
                showarrow: false,
                font: { family: 'JetBrains Mono, monospace', size: 10, color: '#00e676' },
                bgcolor: 'rgba(0, 230, 118, 0.15)',
                bordercolor: '#00e676', borderpad: 3
            });
        }

        // 2. Put Wall (Support)
        if (payload.put_wall && payload.put_wall.strike) {
            const pw = payload.put_wall.strike;
            shapes.push({
                type: 'line', xref: 'paper', x0: 0, x1: 1,
                yref: 'y', y0: pw, y1: pw,
                line: { color: '#ff3366', width: 1.8, dash: 'dash' }
            });
            annotations.push({
                xref: 'paper', x: 0.98, yref: 'y', y: pw,
                text: `PUT WALL: ${pw} (-₹${Math.abs(payload.put_wall.gex_cr || 0)} Cr)`,
                showarrow: false,
                font: { family: 'JetBrains Mono, monospace', size: 10, color: '#ff3366' },
                bgcolor: 'rgba(255, 51, 102, 0.15)',
                bordercolor: '#ff3366', borderpad: 3
            });
        }

        // 3. Gamma Flip Point
        if (payload.gamma_flip && payload.gamma_flip > 0) {
            const gf = payload.gamma_flip;
            shapes.push({
                type: 'line', xref: 'paper', x0: 0, x1: 1,
                yref: 'y', y0: gf, y1: gf,
                line: { color: '#00e5ff', width: 1.2, dash: 'dot' }
            });
            annotations.push({
                xref: 'paper', x: 0.02, yref: 'y', y: gf,
                text: `GAMMA FLIP: ${gf.toFixed(0)}`,
                showarrow: false,
                font: { family: 'JetBrains Mono, monospace', size: 10, color: '#00e5ff' },
                bgcolor: 'rgba(0, 229, 255, 0.12)',
                borderpad: 2
            });
        }

        // 4. Dominant Pin Corridor (Reel 1)
        const activePins = payload.active_pins || [];
        if (activePins.length > 0) {
            const topPin = activePins[0];
            const pLow = topPin.corridor_low;
            const pHigh = topPin.corridor_high;
            if (pLow && pHigh) {
                shapes.push({
                    type: 'rect', xref: 'paper', x0: 0, x1: 1,
                    yref: 'y', y0: pLow, y1: pHigh,
                    fillcolor: 'rgba(255, 213, 79, 0.06)',
                    line: { color: 'rgba(255, 213, 79, 0.4)', width: 1, dash: 'dot' }
                });
                annotations.push({
                    xref: 'paper', x: 0.5, yref: 'y', y: (pLow + pHigh) / 2,
                    text: `MAGNETIC PIN CORRIDOR: ${topPin.strike} (${topPin.duration_str})`,
                    showarrow: false,
                    font: { family: 'JetBrains Mono, monospace', size: 10, color: '#ffd54f' }
                });
            }
        }

        // 5. Aleks Rosme Target Corridor (Reel 2: 100-Point Target)
        const targets = payload.explosion_targets;
        if (targets && targets.target_2) {
            const t2 = targets.target_2;
            const trig = targets.trigger_price;
            const stop = targets.invalidation_stop;

            // Target line
            shapes.push({
                type: 'line', xref: 'paper', x0: 0.7, x1: 1,
                yref: 'y', y0: t2, y1: t2,
                line: { color: '#00e676', width: 2, dash: 'solid' }
            });
            annotations.push({
                xref: 'paper', x: 0.85, yref: 'y', y: t2,
                text: `🎯 100-PT TARGET: ${t2}`,
                showarrow: false,
                font: { family: 'JetBrains Mono, monospace', size: 10, color: '#00e676' },
                bgcolor: 'rgba(0, 230, 118, 0.2)'
            });

            // Stop line
            if (stop) {
                shapes.push({
                    type: 'line', xref: 'paper', x0: 0.7, x1: 1,
                    yref: 'y', y0: stop, y1: stop,
                    line: { color: '#ff2a6d', width: 1.5, dash: 'dot' }
                });
                annotations.push({
                    xref: 'paper', x: 0.85, yref: 'y', y: stop,
                    text: `STOP: ${stop}`,
                    showarrow: false,
                    font: { family: 'JetBrains Mono, monospace', size: 9, color: '#ff2a6d' }
                });
            }
        }

        const layout = {
            dragmode: 'pan',
            autosize: true,
            height: 380,
            margin: { l: 40, r: 50, t: 25, b: 35 },
            paper_bgcolor: 'transparent',
            plot_bgcolor: 'transparent',
            shapes: shapes,
            annotations: annotations,
            xaxis: {
                rangeslider: { visible: false },
                type: 'category',
                tickfont: { family: 'JetBrains Mono, monospace', size: 10, color: '#888' },
                gridcolor: 'rgba(255, 255, 255, 0.04)',
                showgrid: true,
                nticks: 12
            },
            yaxis: {
                tickfont: { family: 'JetBrains Mono, monospace', size: 10, color: '#aaa' },
                gridcolor: 'rgba(255, 255, 255, 0.05)',
                side: 'right',
                autorange: true
            }
        };

        const config = {
            responsive: true,
            displayModeBar: false,
            scrollZoom: false
        };

        Plotly.react(chartEl, [candleTrace], layout, config);
        _chartInitialized = true;
    }

    // ─────────────────────────────────────────────────────────────────────────
    // 2. REEL 1 SPOTLIGHT CARD: DOMINANT STRIKE PINNING & UNWINDING (quantedoptions)
    // ─────────────────────────────────────────────────────────────────────────
    function renderReel1Spotlight(pins, spot) {
        const r1Strike   = document.getElementById('ge-r1-strike');
        const r1Dur      = document.getElementById('ge-r1-duration');
        const r1Gamma    = document.getElementById('ge-r1-gamma');
        const r1Pull     = document.getElementById('ge-r1-pull');
        const r1Corridor = document.getElementById('ge-r1-corridor');
        const r1Alert    = document.getElementById('ge-r1-unwind-alert');
        const r1Pill     = document.getElementById('ge-reel1-status-pill');

        if (!pins || pins.length === 0) {
            if (r1Strike) r1Strike.textContent = 'None Active';
            if (r1Dur)    r1Dur.textContent    = '--';
            if (r1Gamma)  r1Gamma.textContent  = '--';
            if (r1Pull)   r1Pull.textContent   = '--';
            if (r1Corridor) r1Corridor.textContent = '--';
            if (r1Alert)  r1Alert.innerHTML = 'Institutional dealer gamma is dispersed across multiple strikes. No dominant magnetic trap detected.';
            if (r1Pill) {
                r1Pill.className = 'ge-pin-badge low';
                r1Pill.textContent = 'DISPERSED';
            }
            return;
        }

        const pin = pins[0];
        if (r1Strike)   r1Strike.textContent   = `${pin.strike} PIN`;
        if (r1Dur)      r1Dur.textContent      = pin.duration_str || `${pin.duration_min}m`;
        if (r1Gamma)    r1Gamma.textContent    = `+₹${Math.abs(pin.gex_cr).toLocaleString()} Cr`;
        if (r1Pull)     r1Pull.textContent     = `${pin.magnetic_force_score || 0}/100`;
        if (r1Corridor) r1Corridor.textContent = `${pin.corridor_low} - ${pin.corridor_high}`;

        if (r1Pill) {
            const rk = (pin.unpinning_risk || 'LOW').toLowerCase();
            r1Pill.className = `ge-pin-badge ${rk}`;
            r1Pill.textContent = pin.unpinning_risk === 'RELEASED_BREAKOUT' ? 'UNWIND CASCADE' : `RISK: ${pin.unpinning_risk}`;
        }

        if (r1Alert) {
            if (pin.unpinning_risk === 'RELEASED_BREAKOUT') {
                r1Alert.style.borderColor = '#ff0055';
                r1Alert.style.background = 'rgba(255, 0, 85, 0.12)';
                r1Alert.innerHTML = `
                    <span style="color:#ff3366; font-weight:700;">⚠ PIN UNWIND CASCADE ACTIVE:</span>
                    Spot broke out of the ${pin.strike} corridor (${pin.dist_pts > 0 ? '+' : ''}${pin.dist_pts} pts).
                    Dealer gamma faded. Downside cascade forecast targeting next major strike: 
                    <strong style="color:#fff;">${pin.cascade_next_strike || 23200} Put Wall</strong>.
                `;
            } else if (pin.unpinning_risk === 'IMMINENT') {
                r1Alert.style.borderColor = '#ffd54f';
                r1Alert.style.background = 'rgba(255, 213, 79, 0.12)';
                r1Alert.innerHTML = `
                    <span style="color:#ffd54f; font-weight:700;">⚠ UNPINNING IMMINENT:</span>
                    Dealer gamma decayed from peak. Magnetic lock loosening. Watch for breakout cascade.
                `;
            } else {
                r1Alert.style.borderColor = 'rgba(0, 229, 255, 0.25)';
                r1Alert.style.background = 'rgba(0, 229, 255, 0.05)';
                r1Alert.innerHTML = `
                    <span style="color:#00e5ff; font-weight:700;">🔒 ACTIVE MAGNETIC PIN:</span>
                    ${pin.status_desc} Price has maintained session high inside this corridor for <strong>${pin.duration_str}</strong>.
                `;
            }
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // 3. REEL 2 SPOTLIGHT CARD: GEX RETEST & 100-POINT SQUEEZE (aleksrosme)
    // ─────────────────────────────────────────────────────────────────────────
    function renderReel2Spotlight(absorptions, targets, spot) {
        const headlineEl = document.getElementById('ge-r2-headline');
        const statusPill = document.getElementById('ge-radar-status');
        const trigEl     = document.getElementById('ge-target-trigger');
        const stopEl     = document.getElementById('ge-target-stop');
        const t1El       = document.getElementById('ge-target-t1');
        const t2El       = document.getElementById('ge-target-t2');
        const rrEl       = document.getElementById('ge-target-rr');
        const hedgeEl    = document.getElementById('ge-hedge-flow');

        const top = (absorptions && absorptions.length > 0) ? absorptions[0] : null;

        if (statusPill) {
            const st = top ? (top.status || 'MONITORING').toLowerCase() : 'monitoring';
            statusPill.className = `ge-status-pill ${st}`;
            statusPill.textContent = top ? `${top.status}: ${top.level_name}` : 'MONITORING';
        }

        if (headlineEl) {
            if (top && top.absorption_type === 'SELLERS_ABSORBED') {
                headlineEl.innerHTML = `
                    <span style="color:#00e676;">Retest of ${top.level_strike} ${top.level_name} + Microstructural Absorption = 100-Point Squeeze Setup</span>
                    <div style="font-size:11px; font-weight:500; color:#aaa; margin-top:4px;">
                        Rejection Wick: <strong style="color:#fff;">${top.rejection_wick_pct}%</strong> | Vol Multiplier: <strong style="color:#fff;">${top.volume_multiplier}x</strong> | Score: <strong style="color:#00e676;">${top.absorption_score}/100</strong>
                    </div>
                `;
            } else if (top && top.absorption_type === 'BUYERS_ABSORBED') {
                headlineEl.innerHTML = `
                    <span style="color:#ff3366;">Retest of ${top.level_strike} ${top.level_name} + Buyer Absorption = 100-Point Cascade Setup</span>
                    <div style="font-size:11px; font-weight:500; color:#aaa; margin-top:4px;">
                        Upper Wick: <strong style="color:#fff;">${top.rejection_wick_pct}%</strong> | Vol Multiplier: <strong style="color:#fff;">${top.volume_multiplier}x</strong> | Score: <strong style="color:#ff3366;">${top.absorption_score}/100</strong>
                    </div>
                `;
            } else if (top) {
                headlineEl.innerHTML = `
                    <span>Monitoring ${top.level_name} at ${top.level_strike} (dist: ${top.dist_pts > 0 ? '+' : ''}${top.dist_pts} pts)</span>
                    <div style="font-size:11px; font-weight:500; color:#888; margin-top:4px;">
                        Waiting for retest and volume absorption confirmation...
                    </div>
                `;
            } else {
                headlineEl.textContent = `Spot at ${spot.toFixed(0)} trading between major GEX walls.`;
            }
        }

        if (targets) {
            if (trigEl) trigEl.textContent = targets.trigger_price || '--';
            if (stopEl) stopEl.textContent = targets.invalidation_stop ? `${targets.invalidation_stop} (${targets.risk_points} pts)` : '--';
            if (t1El)   t1El.textContent   = targets.target_1 ? `${targets.target_1} (+50 pts)` : '--';
            if (t2El)   t2El.textContent   = targets.target_2 ? `${targets.target_2} (+100 pts)` : '--';
            if (rrEl)   rrEl.textContent   = targets.risk_reward_ratio ? `${targets.risk_reward_ratio} : 1` : '--';

            if (hedgeEl && targets.hedge_acceleration) {
                const ha = targets.hedge_acceleration;
                const isBuy = ha.flow_action === 'BUYING_PRESSURE';
                const col = isBuy ? '#00e676' : '#ff3366';
                hedgeEl.innerHTML = `
                    Dealer Hedging Flow: <span style="color:${col}; font-weight:700;">${isBuy ? '▲' : '▼'} ${ha.lots_to_hedge_25pts.toLocaleString()} Lots / 25pts</span>
                `;
            }
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // 4. STRIKE GEX LADDER
    // ─────────────────────────────────────────────────────────────────────────
    function renderStrikeLadder(gexData, pins, spot) {
        const ladderWrap = document.getElementById('ge-ladder-container');
        if (!ladderWrap) return;

        const strikes = (gexData && gexData.strikes) ? gexData.strikes : [];
        if (!strikes || strikes.length === 0) {
            ladderWrap.innerHTML = '<div style="color:var(--text-muted); font-size:12px; padding:10px;">Awaiting options strike data...</div>';
            return;
        }

        // Filter around spot (+- 350 pts)
        const nearby = strikes.filter(s => Math.abs(s.strike - spot) <= 350);
        if (nearby.length === 0) return;

        let maxAbs = 1.0;
        nearby.forEach(s => {
            if (Math.abs(s.gex) > maxAbs) maxAbs = Math.abs(s.gex);
        });

        const pinStrikes = new Set((pins || []).map(p => p.strike));

        let html = '';
        nearby.forEach(s => {
            const isPin = pinStrikes.has(s.strike);
            const isNearSpot = Math.abs(s.strike - spot) <= 25;
            const gexCr = (s.gex / 1e7).toFixed(1);
            const isPos = s.gex >= 0;
            const widthPct = Math.min(100, Math.max(4, (Math.abs(s.gex) / maxAbs) * 100)).toFixed(1);

            let rowCls = 'ge-ladder-row';
            if (isPin) rowCls += ' is-pin glow-max-pain';
            if (isNearSpot) rowCls += ' is-spot glow-atm';

            html += `
                <div class="${rowCls}">
                    <div class="ge-strike-lbl">
                        ${s.strike}
                        ${isPin ? '<span class="wall-badge-pain">★ PIN</span>' : ''}
                        ${isNearSpot ? '<span class="wall-badge-atm">● ATM</span>' : ''}
                    </div>
                    <div class="ge-bar-track">
                        <div class="ge-bar-fill ${isPos ? 'pos' : 'neg'}" style="width: ${widthPct}%;"></div>
                    </div>
                    <div class="ge-val-mono ${isPos ? 'dealer-glow-green' : 'dealer-glow-red'}">
                        ${isPos ? '+' : ''}${gexCr} Cr
                    </div>
                </div>
            `;
        });

        ladderWrap.innerHTML = html;
    }

    // ─────────────────────────────────────────────────────────────────────────
    // 5. DOM CLEANUP & DEALER POSITIONING GLOWS
    // ─────────────────────────────────────────────────────────────────────────
    function cleanupMMDOM() {
        // No-op: Obsolete elements removed from template
    }

    function enhanceDealerInventoryGlows(payload, spot) {
        // Server-side VolatilityAnalyzer already injects optimized badges and classes
    }

    // ─────────────────────────────────────────────────────────────────────────
    // 6. MASTER UPDATE LOOP
    // ─────────────────────────────────────────────────────────────────────────
    let _isUpdating = false;
    async function updateTerminal() {
        if (_isUpdating) return;
        _isUpdating = true;
        try {
            const payload = await fetchExplosionData();
            if (!payload || !payload.ok) return;

            _lastPayload = payload;
            const spot = payload.spot || 23290;

            // Fetch strike GEX from /api/gex
            let gexData = null;
            try {
                const gexRes = await fetch('/api/gex?t=' + Date.now());
                if (gexRes.ok) gexData = await gexRes.json();
            } catch (e) {
                // ignore
            }

            // Render Strike Ladder
            renderStrikeLadder(gexData, payload.active_pins, spot);

            // Update home quick-launch banner if present
            const bannerPin = document.getElementById('ge-banner-pin');
            if (bannerPin) {
                if (payload.active_pins && payload.active_pins.length > 0) {
                    const topPin = payload.active_pins[0];
                    bannerPin.innerHTML = `<span style="color:#ffd54f;">${topPin.strike} PIN</span> (${topPin.duration_str} | +₹${Math.abs(topPin.gex_cr)} Cr)`;
                } else {
                    bannerPin.textContent = 'None Active (Dispersed)';
                }
            }
        } finally {
            _isUpdating = false;
        }
    }

    // Expose globally
    window.GammaExplosionTerminal = {
        init: function () {
            updateTerminal();
        },
        refresh: updateTerminal,
        handleWsMessage: function (msg) {
            if (msg.type === 'gamma_explosion_update' && msg.payload) {
                _lastPayload = msg.payload;
                updateTerminal();
            }
        }
    };

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', () => {
            if (document.getElementById('gamma-explosion-root')) {
                window.GammaExplosionTerminal.init();
            }
        });
    } else {
        if (document.getElementById('gamma-explosion-root')) {
            window.GammaExplosionTerminal.init();
        }
    }
})();
