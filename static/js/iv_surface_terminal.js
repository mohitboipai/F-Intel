/**
 * static/js/iv_surface_terminal.js
 * ═══════════════════════════════════════════════════════════════════════════
 * F-INTEL: REAL-TIME IV SURFACE & EXHAUSTION TERMINAL
 * ═══════════════════════════════════════════════════════════════════════════
 * Key Features:
 * 1. Expected Move vs Day Open Displacement & Range Consumption:
 *    - 1-Day Expected Move (±Pts, ±%) derived mathematically from ATM IV
 *    - Displacement Since Open (Spot - Day Open) & Intraday Range (High - Low)
 *    - Exhaustion Ratio Gauge (0% - 100%+ of 1σ budget)
 *    - Quantitative boundaries (±1σ Upper & Lower Levels) without directional guesswork
 * 2. Session Memory Rewind Scrubber:
 *    - Interactive scrubber timeline across the entire trading day
 *    - Presets: [LIVE], [-5m], [-15m], [-30m], [-1h], [DAY OPEN]
 *    - Dual comparison: Active Smile vs Baseline Smile (Day Open or prior snapshot)
 * 3. Reactive Plotly Visualizations:
 *    - 2D Volatility Smile with per-strike ΔIV bar overlay
 *    - 3D Volatility Surface with Strike, DTE, IV mesh preserving 3D camera angles
 * 4. Real-time updates via WebSocket (iv_surface_update) & REST fallback
 */

(function () {
    'use strict';

    // Internal state
    var _isRewound = false;
    var _isScrubbing = false;
    var _rewindTs = null;
    var _baselineMode = 'open'; // 'open' or 'prev'
    var _historyIndex = [];
    var _latestPayload = null;
    var _pollTimer = null;

    // ── Fetch Helper ────────────────────────────────────────────────────────
    async function fetchSurfaceData(rewindTs, baselineMode) {
        try {
            var url = '/api/iv-surface?baseline=' + encodeURIComponent(baselineMode || _baselineMode);
            if (rewindTs !== null && rewindTs !== undefined) {
                url += '&rewind_ts=' + encodeURIComponent(rewindTs);
            }
            url += '&_t=' + Date.now();

            var res = await fetch(url);
            if (!res.ok) return null;
            return await res.json();
        } catch (e) {
            console.warn('[IvSurfaceTerminal] Fetch error:', e);
            return null;
        }
    }

    // ── WebSocket Handler ───────────────────────────────────────────────────
    function handleWsMessage(msg) {
        if (!msg || msg.type !== 'iv_surface_update' || !msg.payload) return;
        var payload = msg.payload;

        // Keep history index fresh
        if (payload.history_index && payload.history_index.length) {
            _historyIndex = payload.history_index;
            updateScrubberBounds();
        }

        // If user is currently scrubbing or in historical rewind mode, do not snap back
        if (_isRewound || _isScrubbing) {
            return;
        }

        _latestPayload = payload;
        renderAll(payload);
    }

    // ── Scrubber & Presets Controller ───────────────────────────────────────
    function updateScrubberBounds() {
        var slider = document.getElementById('iv-rewind-slider');
        if (!slider) return;

        if (_historyIndex && _historyIndex.length > 1) {
            slider.min = 0;
            slider.max = _historyIndex.length - 1;
            slider.disabled = false;

            if (!_isRewound && !_isScrubbing) {
                slider.value = _historyIndex.length - 1;
                var latestSnap = _historyIndex[_historyIndex.length - 1];
                updateTimeLabel(latestSnap.time_str || 'LIVE', false);
            }
        } else {
            slider.min = 0;
            slider.max = 1;
            slider.value = 1;
            slider.disabled = true;
            updateTimeLabel('LIVE', false);
        }
    }

    function updateTimeLabel(timeStr, isRewound) {
        var label = document.getElementById('iv-rewind-time-label');
        var badge = document.getElementById('iv-rewind-status-badge');
        var btnLive = document.getElementById('btn-iv-return-live');

        if (label) {
            label.textContent = timeStr;
            label.style.color = isRewound ? '#ffd54f' : '#00e5ff';
        }

        if (badge) {
            if (isRewound) {
                badge.textContent = 'REWOUND MEMORY (' + timeStr + ')';
                badge.className = 'iv-status-badge rewound';
            } else {
                badge.textContent = 'LIVE STREAMING';
                badge.className = 'iv-status-badge';
            }
        }

        if (btnLive) {
            btnLive.style.display = isRewound ? 'inline-flex' : 'none';
        }
    }

    async function applyRewindIndex(idx) {
        if (!_historyIndex || !_historyIndex[idx]) return;
        var isLast = (idx === _historyIndex.length - 1);
        if (isLast) {
            returnToLive();
            return;
        }

        var snap = _historyIndex[idx];
        _isRewound = true;
        _rewindTs = snap.ts;
        updateTimeLabel(snap.time_str || 'HISTORY', true);

        // Highlight active preset button if applicable
        updatePresetButtonsHighlight(null);

        var data = await fetchSurfaceData(_rewindTs, _baselineMode);
        if (data && data.ok) {
            renderAll(data);
        }
    }

    async function jumpToPreset(presetKey) {
        if (!_historyIndex || _historyIndex.length < 2) return;

        if (presetKey === 'live') {
            returnToLive();
            return;
        }

        var targetTs = null;
        if (presetKey === 'open') {
            // Jump to the very first recorded snapshot of the session
            targetTs = _historyIndex[0].ts;
        } else {
            // Preset like 5m, 15m, 30m, 60m
            var mins = parseInt(presetKey, 10);
            if (isNaN(mins)) return;
            var nowTs = _historyIndex[_historyIndex.length - 1].ts;
            targetTs = nowTs - (mins * 60);
        }

        // Find closest snapshot in history
        var bestIdx = 0;
        var bestDiff = Infinity;
        for (var i = 0; i < _historyIndex.length; i++) {
            var diff = Math.abs(_historyIndex[i].ts - targetTs);
            if (diff < bestDiff) {
                bestDiff = diff;
                bestIdx = i;
            }
        }

        var slider = document.getElementById('iv-rewind-slider');
        if (slider) slider.value = bestIdx;

        updatePresetButtonsHighlight(presetKey);
        applyRewindIndex(bestIdx);
    }

    function updatePresetButtonsHighlight(activePreset) {
        document.querySelectorAll('.iv-preset-btn').forEach(function (btn) {
            var pId = btn.getAttribute('data-preset');
            if (pId === activePreset) {
                btn.classList.add('active');
            } else {
                btn.classList.remove('active');
            }
        });
    }

    async function returnToLive() {
        _isRewound = false;
        _rewindTs = null;
        updateScrubberBounds();
        updateTimeLabel('LIVE', false);
        updatePresetButtonsHighlight('live');

        var data = await fetchSurfaceData(null, _baselineMode);
        if (data && data.ok) {
            _latestPayload = data;
            renderAll(data);
        }
    }

    function setBaselineMode(mode) {
        if (_baselineMode === mode) return;
        _baselineMode = mode;

        document.querySelectorAll('.iv-baseline-btn').forEach(function (btn) {
            var bMode = btn.getAttribute('data-baseline');
            if (bMode === mode) btn.classList.add('active');
            else btn.classList.remove('active');
        });

        // Re-fetch current view with new baseline
        fetchSurfaceData(_rewindTs, _baselineMode).then(function (data) {
            if (data && data.ok) renderAll(data);
        });
    }

    // ── Render All Sections ─────────────────────────────────────────────────
    function renderAll(data) {
        if (!data) return;

        renderHeaderAndSpot(data);
        renderExhaustionPanel(data);
        renderTotalShifts(data);
        render2dSmileChart(data);
        render3dSurfaceChart(data);
    }

    // 1. Header & Spot Strip
    function renderHeaderAndSpot(data) {
        var spotEl = document.getElementById('iv-spot-val');
        var openEl = document.getElementById('iv-open-val');
        var timeEl = document.getElementById('iv-snap-time');

        if (spotEl && data.spot) {
            spotEl.textContent = Number(data.spot).toLocaleString('en-IN', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
        }
        if (openEl && data.day_open) {
            openEl.textContent = Number(data.day_open).toLocaleString('en-IN', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
        }
        if (timeEl && data.timestamp_str) {
            timeEl.textContent = data.timestamp_str;
        }
    }

    // 2. Mathematical Movement & Exhaustion Panel (1-Day & Weekly Horizons)
    function renderExhaustionPanel(data) {
        var em = data.expected_move || {};
        var ex = data.exhaustion || {};

        // ── 1-Day Horizon ──────────────────────────────────────────────────
        var expPtsEl = document.getElementById('iv-exp-move-pts');
        var expPctEl = document.getElementById('iv-exp-move-pct');
        if (expPtsEl) expPtsEl.textContent = '±' + (em.expected_move_pts ? em.expected_move_pts.toFixed(1) : '--') + ' pts';
        if (expPctEl) expPctEl.textContent = '±' + (em.expected_move_pct ? em.expected_move_pct.toFixed(2) : '--') + '%';

        // Intraday Range (H - L)
        var rangePtsEl = document.getElementById('iv-range-pts');
        var rangeSubEl = document.getElementById('iv-range-sub');
        if (rangePtsEl) {
            rangePtsEl.textContent = (em.high_low_range_pts ? em.high_low_range_pts.toFixed(1) : '--') + ' pts';
        }
        if (rangeSubEl && em.day_high && em.day_low) {
            rangeSubEl.textContent = 'H: ' + em.day_high.toFixed(0) + ' | L: ' + em.day_low.toFixed(0);
        }

        // 1-Day 1σ Boundary Levels
        var upperEl = document.getElementById('iv-boundary-upper');
        var lowerEl = document.getElementById('iv-boundary-lower');
        if (upperEl && em.upper_1sigma) {
            upperEl.textContent = em.upper_1sigma.toFixed(0) + ' (+' + (em.dist_to_upper_1sigma || 0).toFixed(0) + ')';
        }
        if (lowerEl && em.lower_1sigma) {
            lowerEl.textContent = em.lower_1sigma.toFixed(0) + ' (' + (em.dist_to_lower_1sigma || 0).toFixed(0) + ')';
        }

        // ── Weekly Horizon (5 Trading Days / Expiry) ───────────────────────
        var wExpPtsEl = document.getElementById('iv-weekly-exp-pts');
        var wExpPctEl = document.getElementById('iv-weekly-exp-pct');
        if (wExpPtsEl) wExpPtsEl.textContent = '±' + (em.weekly_expected_move_pts ? em.weekly_expected_move_pts.toFixed(1) : '--') + ' pts';
        if (wExpPctEl) wExpPctEl.textContent = '±' + (em.weekly_expected_move_pct ? em.weekly_expected_move_pct.toFixed(2) : '--') + '%';

        // Weekly Realized Move & Range
        var wRealizedEl = document.getElementById('iv-weekly-realized-pts');
        var wRangeSubEl = document.getElementById('iv-weekly-range-sub');
        if (wRealizedEl) {
            var wPts = em.weekly_realized_pts || 0;
            var wSign = wPts > 0 ? '+' : '';
            var wPct = em.weekly_realized_pct || 0;
            wRealizedEl.textContent = wSign + wPts.toFixed(1) + ' pts (' + (wPct >= 0 ? '+' : '') + wPct.toFixed(2) + '%)';
            wRealizedEl.style.color = wPts >= 0 ? '#26a69a' : '#ef5350';
        }
        if (wRangeSubEl && em.week_high && em.week_low) {
            wRangeSubEl.textContent = '5D Range: ' + (em.weekly_range_pts ? em.weekly_range_pts.toFixed(1) : '--') + ' pts (H: ' + em.week_high.toFixed(0) + ' | L: ' + em.week_low.toFixed(0) + ')';
        }

        // Weekly 1σ Boundary Levels
        var wUpperEl = document.getElementById('iv-weekly-boundary-upper');
        var wLowerEl = document.getElementById('iv-weekly-boundary-lower');
        if (wUpperEl && em.weekly_upper_1sigma) {
            wUpperEl.textContent = em.weekly_upper_1sigma.toFixed(0) + ' (+' + (em.dist_to_weekly_upper || 0).toFixed(0) + ')';
        }
        if (wLowerEl && em.weekly_lower_1sigma) {
            wLowerEl.textContent = em.weekly_lower_1sigma.toFixed(0) + ' (' + (em.dist_to_weekly_lower || 0).toFixed(0) + ')';
        }

        // ── 1-Day Gauge Bar (Shaded Intraday Move vs Solid Current Spot) ───
        var consPct = ex.net_consumption_pct || 0;
        var rangeConsPct = ex.range_consumption_pct || 0;

        var consLabelEl = document.getElementById('iv-consumption-pct-label');
        var rangeConsLabelEl = document.getElementById('iv-range-consumption-label');
        var gaugeBarFill = document.getElementById('iv-gauge-bar-fill');
        var gaugeBarRange = document.getElementById('iv-gauge-bar-range');

        if (consLabelEl) consLabelEl.textContent = consPct.toFixed(1) + '%';
        if (rangeConsLabelEl) rangeConsLabelEl.textContent = rangeConsPct.toFixed(1) + '%';

        var curWidth = Math.min(100, Math.max(0, consPct * 0.8));
        var rangeWidth = Math.min(100, Math.max(0, rangeConsPct * 0.8));

        if (gaugeBarRange) {
            gaugeBarRange.style.width = rangeWidth + '%';
            if (rangeConsPct > 105) {
                gaugeBarRange.style.background = 'rgba(255, 51, 102, 0.22)';
                gaugeBarRange.style.borderRightColor = '#ff3366';
            } else if (rangeConsPct > 85) {
                gaugeBarRange.style.background = 'rgba(255, 213, 79, 0.22)';
                gaugeBarRange.style.borderRightColor = '#ffd54f';
            } else {
                gaugeBarRange.style.background = 'rgba(0, 229, 255, 0.2)';
                gaugeBarRange.style.borderRightColor = 'rgba(0, 229, 255, 0.8)';
            }
        }

        if (gaugeBarFill) {
            gaugeBarFill.style.width = curWidth + '%';
            if (consPct < 50) {
                gaugeBarFill.style.background = 'linear-gradient(90deg, #00e5ff, #00b0ff)';
            } else if (consPct < 85) {
                gaugeBarFill.style.background = 'linear-gradient(90deg, #00b0ff, #26a69a)';
            } else if (consPct <= 105) {
                gaugeBarFill.style.background = 'linear-gradient(90deg, #ffa726, #ff7043)';
            } else {
                gaugeBarFill.style.background = 'linear-gradient(90deg, #ff5252, #e040fb)';
            }
        }

        // ── Weekly Gauge Bar (Shaded Weekly Range vs Solid Weekly Move) ────
        var wNetConsPct = ex.weekly_net_consumption_pct || em.weekly_net_consumption_pct || 0;
        var wRangeConsPct = ex.weekly_range_consumption_pct || em.weekly_range_consumption_pct || 0;

        var wConsLabelEl = document.getElementById('iv-weekly-consumption-label');
        var wRangeConsLabelEl = document.getElementById('iv-weekly-range-consumption-label');
        var wGaugeBarFill = document.getElementById('iv-weekly-bar-fill');
        var wGaugeBarRange = document.getElementById('iv-weekly-bar-range');

        if (wConsLabelEl) wConsLabelEl.textContent = wNetConsPct.toFixed(1) + '%';
        if (wRangeConsLabelEl) wRangeConsLabelEl.textContent = wRangeConsPct.toFixed(1) + '%';

        var wCurWidth = Math.min(100, Math.max(0, wNetConsPct * 0.8));
        var wRangeWidth = Math.min(100, Math.max(0, wRangeConsPct * 0.8));

        if (wGaugeBarRange) {
            wGaugeBarRange.style.width = wRangeWidth + '%';
            if (wRangeConsPct > 105) {
                wGaugeBarRange.style.background = 'rgba(255, 51, 102, 0.22)';
                wGaugeBarRange.style.borderRightColor = '#ff3366';
            } else if (wRangeConsPct > 85) {
                wGaugeBarRange.style.background = 'rgba(255, 213, 79, 0.22)';
                wGaugeBarRange.style.borderRightColor = '#ffd54f';
            } else {
                wGaugeBarRange.style.background = 'rgba(124, 77, 255, 0.22)';
                wGaugeBarRange.style.borderRightColor = 'rgba(124, 77, 255, 0.8)';
            }
        }

        if (wGaugeBarFill) {
            wGaugeBarFill.style.width = wCurWidth + '%';
            if (wNetConsPct < 50) {
                wGaugeBarFill.style.background = 'linear-gradient(90deg, #7c4dff, #00e5ff)';
            } else if (wNetConsPct < 85) {
                wGaugeBarFill.style.background = 'linear-gradient(90deg, #00b0ff, #26a69a)';
            } else if (wNetConsPct <= 105) {
                wGaugeBarFill.style.background = 'linear-gradient(90deg, #ffa726, #ff7043)';
            } else {
                wGaugeBarFill.style.background = 'linear-gradient(90deg, #ff5252, #e040fb)';
            }
        }

        // Status Tag
        var regimeTag = document.getElementById('iv-regime-tag');
        if (regimeTag) {
            var rText = ex.status || 'NORMAL RANGE';
            regimeTag.textContent = rText;
            regimeTag.className = 'iv-regime-tag ' + (
                consPct < 50 ? 'consolidation' :
                consPct < 85 ? 'normal' :
                consPct <= 105 ? 'exhaustion' : 'expansion'
            );
        }
    }

    // 3. Total Shifts Strip
    function renderTotalShifts(data) {
        var shifts = data.total_iv_shifts || {};
        var active = data.active_snapshot || {};

        var atmValEl = document.getElementById('iv-shift-atm-val');
        var skewValEl = document.getElementById('iv-shift-skew-val');
        var tsValEl = document.getElementById('iv-shift-ts-val');
        var wingsValEl = document.getElementById('iv-shift-wings-val');

        if (atmValEl) {
            var atm = active.atm_iv || 0;
            var dAtm = shifts.delta_atm_iv || 0;
            var sign = dAtm > 0 ? '+' : '';
            var col = dAtm > 0 ? '#00e676' : dAtm < 0 ? '#ff5252' : '#d1d4dc';
            atmValEl.innerHTML = atm.toFixed(2) + '% <span style="font-size:11px;color:' + col + ';">(' + sign + dAtm.toFixed(2) + '%)</span>';
        }

        if (skewValEl) {
            var skew = active.skew_ratio || 0;
            var dSkew = shifts.delta_skew || 0;
            var sSign = dSkew > 0 ? '+' : '';
            skewValEl.innerHTML = skew.toFixed(3) + ' <span style="font-size:11px;color:#868993;">(' + sSign + dSkew.toFixed(3) + ')</span>';
        }

        if (tsValEl) {
            var ts = active.term_spread || 0;
            var dTs = shifts.delta_term_spread || 0;
            var tSign = dTs > 0 ? '+' : '';
            tsValEl.innerHTML = ts.toFixed(2) + '% <span style="font-size:11px;color:#868993;">(' + tSign + dTs.toFixed(2) + '%)</span>';
        }

        if (wingsValEl) {
            var putWing = active.put_wing_iv || 0;
            var callWing = active.call_wing_iv || 0;
            wingsValEl.innerHTML = '<span style="color:#ef5350;">P:' + putWing.toFixed(1) + '%</span> | <span style="color:#26a69a;">C:' + callWing.toFixed(1) + '%</span>';
        }
    }

    // 4. 2D Volatility Smile & Delta IV Overlay Chart
    function render2dSmileChart(data) {
        var chartDiv = document.getElementById('iv-smile-plot');
        if (!chartDiv || typeof Plotly === 'undefined') return;

        var smileData = data.smile_2d || {};
        var activeStrikes = smileData.active_strikes || [];
        var activeIvs = smileData.active_ivs || [];
        var baselineIvs = smileData.baseline_ivs || [];
        var deltaIvs = smileData.delta_ivs || [];

        if (!activeStrikes.length) return;

        var spot = data.spot || 0;
        var em = data.expected_move || {};
        var upper1s = em.upper_1sigma || (spot * 1.01);
        var lower1s = em.lower_1sigma || (spot * 0.99);

        // Bar colors for Delta IV
        var barColors = deltaIvs.map(function (v) {
            return v >= 0 ? 'rgba(0, 230, 118, 0.45)' : 'rgba(255, 82, 82, 0.45)';
        });

        var traces = [
            // Trace 0: Bar chart of Delta IV per strike
            {
                x: activeStrikes,
                y: deltaIvs,
                type: 'bar',
                name: 'Δ IV (% pts)',
                yaxis: 'y2',
                marker: { color: barColors },
                hoverinfo: 'x+y+name'
            },
            // Trace 1: Baseline Smile
            {
                x: activeStrikes,
                y: baselineIvs,
                mode: 'lines',
                name: 'Baseline (' + (_baselineMode === 'open' ? 'Day Open' : 'Prior') + ')',
                line: { color: '#ffd54f', width: 2, dash: 'dot' },
                hoverinfo: 'x+y+name'
            },
            // Trace 2: Active Smile
            {
                x: activeStrikes,
                y: activeIvs,
                mode: 'lines+markers',
                name: 'Active Smile (' + (data.timestamp_str || 'Live') + ')',
                line: { color: '#00e5ff', width: 2.5 },
                marker: { size: 5, color: '#00e5ff' },
                hoverinfo: 'x+y+name'
            }
        ];

        // Shapes & annotations for Spot and 1σ boundaries
        var shapes = [
            // Spot line
            {
                type: 'line',
                x0: spot,
                x1: spot,
                y0: 0,
                y1: 1,
                yref: 'paper',
                line: { color: '#64b5f6', width: 1.5, dash: 'dash' }
            },
            // Upper 1-Sigma line
            {
                type: 'line',
                x0: upper1s,
                x1: upper1s,
                y0: 0,
                y1: 1,
                yref: 'paper',
                line: { color: '#ffa726', width: 1.2, dash: 'dot' }
            },
            // Lower 1-Sigma line
            {
                type: 'line',
                x0: lower1s,
                x1: lower1s,
                y0: 0,
                y1: 1,
                yref: 'paper',
                line: { color: '#ffa726', width: 1.2, dash: 'dot' }
            }
        ];

        var annotations = [
            {
                x: spot,
                y: 1.03,
                yref: 'paper',
                text: 'SPOT ' + spot.toFixed(0),
                showarrow: false,
                font: { color: '#64b5f6', size: 10, family: 'JetBrains Mono, monospace', weight: 'bold' }
            },
            {
                x: upper1s,
                y: 0.95,
                yref: 'paper',
                text: '+1σ',
                showarrow: false,
                font: { color: '#ffa726', size: 9, family: 'JetBrains Mono, monospace' }
            },
            {
                x: lower1s,
                y: 0.95,
                yref: 'paper',
                text: '-1σ',
                showarrow: false,
                font: { color: '#ffa726', size: 9, family: 'JetBrains Mono, monospace' }
            }
        ];

        var minIv = Math.min.apply(null, activeIvs.concat(baselineIvs));
        var maxIv = Math.max.apply(null, activeIvs.concat(baselineIvs));
        var ivPadding = Math.max(2, (maxIv - minIv) * 0.15);

        var maxDelta = Math.max.apply(null, deltaIvs.map(Math.abs));
        var deltaRange = Math.max(3, maxDelta * 2);

        var layout = {
            paper_bgcolor: 'transparent',
            plot_bgcolor: 'transparent',
            margin: { t: 30, r: 40, b: 40, l: 45 },
            legend: {
                orientation: 'h',
                y: -0.15,
                x: 0.05,
                font: { color: '#94a3b8', size: 10, family: 'Inter, sans-serif' }
            },
            xaxis: {
                title: { text: 'Strike', font: { color: '#64748b', size: 11 } },
                gridcolor: 'rgba(255, 255, 255, 0.05)',
                tickfont: { color: '#f1f5f9', size: 10, family: 'JetBrains Mono, monospace' }
            },
            yaxis: {
                title: { text: 'IV (%)', font: { color: '#00e5ff', size: 11 } },
                range: [Math.max(0, minIv - ivPadding), maxIv + ivPadding],
                gridcolor: 'rgba(255, 255, 255, 0.05)',
                tickfont: { color: '#00e5ff', size: 10, family: 'JetBrains Mono, monospace' }
            },
            yaxis2: {
                title: { text: 'Δ IV (% pts)', font: { color: '#94a3b8', size: 10 } },
                range: [-deltaRange, deltaRange],
                overlaying: 'y',
                side: 'right',
                showgrid: false,
                tickfont: { color: '#94a3b8', size: 9, family: 'JetBrains Mono, monospace' }
            },
            shapes: shapes,
            annotations: annotations,
            hovermode: 'x unified'
        };

        var config = {
            responsive: true,
            displayModeBar: false
        };

        Plotly.react(chartDiv, traces, layout, config);
    }

    // 5. 3D Volatility Surface Mesh Chart
    function render3dSurfaceChart(data) {
        var chartDiv = document.getElementById('iv-surface-3d-plot');
        if (!chartDiv || typeof Plotly === 'undefined') return;

        var mesh = data.surface_3d || {};
        var strikes = mesh.strikes || [];
        var dtes = mesh.dtes || [];
        var zIv = mesh.z_iv || [];

        if (!strikes.length || !dtes.length || !zIv.length) return;

        // Preserve user camera rotation if already interacting
        var userCamera = null;
        if (chartDiv._fullLayout && chartDiv._fullLayout.scene && chartDiv._fullLayout.scene.camera) {
            userCamera = JSON.parse(JSON.stringify(chartDiv._fullLayout.scene.camera));
        }

        var trace3d = {
            type: 'surface',
            x: strikes,
            y: dtes,
            z: zIv,
            colorscale: [
                [0.0, '#00e5ff'],
                [0.35, '#2962ff'],
                [0.7, '#7c4dff'],
                [1.0, '#ff1744']
            ],
            contours: {
                z: { show: true, usecolormap: true, highlightcolor: '#fff', project: { z: true } }
            },
            colorbar: {
                title: 'IV%',
                titleside: 'top',
                tickfont: { color: '#94a3b8', size: 9, family: 'JetBrains Mono, monospace' },
                titlefont: { color: '#00e5ff', size: 10 },
                len: 0.7,
                thickness: 12
            }
        };

        var defaultCamera = {
            eye: { x: 1.6, y: -1.7, z: 0.85 },
            center: { x: 0, y: 0, z: -0.1 },
            projection: { type: 'perspective' }
        };

        var layout3d = {
            uirevision: 'iv_surface_3d_constant', // Keeps user zoom, orbit, and pan states across live updates!
            autosize: true,
            paper_bgcolor: 'transparent',
            margin: { t: 25, r: 15, b: 25, l: 15 },
            scene: {
                aspectmode: 'manual',
                aspectratio: { x: 1.25, y: 1.0, z: 0.55 },
                camera: userCamera || defaultCamera,
                xaxis: {
                    title: 'Strike',
                    backgroundcolor: 'rgba(0,0,0,0)',
                    gridcolor: 'rgba(255,255,255,0.08)',
                    tickfont: { color: '#f1f5f9', size: 9, family: 'JetBrains Mono, monospace' },
                    titlefont: { color: '#64748b', size: 10 }
                },
                yaxis: {
                    title: 'DTE',
                    backgroundcolor: 'rgba(0,0,0,0)',
                    gridcolor: 'rgba(255,255,255,0.08)',
                    tickfont: { color: '#f1f5f9', size: 9, family: 'JetBrains Mono, monospace' },
                    titlefont: { color: '#64748b', size: 10 }
                },
                zaxis: {
                    title: 'IV (%)',
                    backgroundcolor: 'rgba(0,0,0,0)',
                    gridcolor: 'rgba(255,255,255,0.08)',
                    tickfont: { color: '#00e5ff', size: 9, family: 'JetBrains Mono, monospace' },
                    titlefont: { color: '#00e5ff', size: 10 }
                }
            }
        };

        var config3d = {
            responsive: true,
            displayModeBar: true,
            displaylogo: false,
            modeBarButtonsToRemove: ['sendDataToCloud', 'hoverClosest3d'],
            scrollZoom: true
        };

        Plotly.react(chartDiv, [trace3d], layout3d, config3d);
    }

    // ── Setup Event Listeners (Document-Level Delegation) ────────────────────
    function initListeners() {
        document.addEventListener('click', function (e) {
            // 1. Presets
            var presetBtn = e.target.closest('.iv-preset-btn');
            if (presetBtn) {
                var pId = presetBtn.getAttribute('data-preset');
                jumpToPreset(pId);
                return;
            }

            // 2. Return to Live button
            var liveBtn = e.target.closest('#btn-iv-return-live');
            if (liveBtn) {
                returnToLive();
                return;
            }

            // 3. Baseline toggle buttons
            var baselineBtn = e.target.closest('.iv-baseline-btn');
            if (baselineBtn) {
                var bMode = baselineBtn.getAttribute('data-baseline');
                setBaselineMode(bMode);
                return;
            }
        });

        // Rewind Slider delegation
        document.addEventListener('input', function (e) {
            if (e.target && e.target.id === 'iv-rewind-slider') {
                var slider = e.target;
                _isScrubbing = true;
                var idx = parseInt(slider.value, 10);
                if (_historyIndex && _historyIndex[idx]) {
                    var isLast = (idx === _historyIndex.length - 1);
                    updateTimeLabel(_historyIndex[idx].time_str || 'HISTORY', !isLast);
                }
            }
        });

        document.addEventListener('change', function (e) {
            if (e.target && e.target.id === 'iv-rewind-slider') {
                var slider = e.target;
                _isScrubbing = false;
                var idx = parseInt(slider.value, 10);
                applyRewindIndex(idx);
            }
        });
    }

    // ── Initialization ──────────────────────────────────────────────────────
    function init() {
        initListeners();
        returnToLive();

        // Fallback polling every 4s if WebSocket is quiet
        if (_pollTimer) clearInterval(_pollTimer);
        _pollTimer = setInterval(function () {
            if (!_isRewound && !_isScrubbing) {
                fetchSurfaceData(null, _baselineMode).then(function (data) {
                    if (data && data.ok) renderAll(data);
                });
            }
        }, 4000);
    }

    // Expose on window for WebSocket dispatch from dashboard_core.js
    window.IvSurfaceTerminal = {
        handleWsMessage: handleWsMessage,
        returnToLive: returnToLive,
        jumpToPreset: jumpToPreset,
        setBaselineMode: setBaselineMode
    };

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }
})();
