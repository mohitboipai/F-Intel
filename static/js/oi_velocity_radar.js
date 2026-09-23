/**
 * static/js/oi_velocity_radar.js
 * ═══════════════════════════════════════════════════════════════════════════
 * F-INTEL: REAL-TIME OI DYNAMICS & ORDER FLOW RADAR
 * ═══════════════════════════════════════════════════════════════════════════
 * Features:
 * 1. Bidirectional Plotly Tornado Chart:
 *    - Left side: Call Flow (Rose/Red for Writing, Sky/Cyan for Unwinding)
 *    - Right side: Put Flow (Green for Writing, Amber for Panic Unwinding)
 *    - Toggle between:
 *        * [Δ OI VOLUME (Contracts / Lots)] (Default: How much OI changed)
 *        * [RATE / MIN] (Speed of change)
 *    - Strike Field of View Range Selector:
 *        * [±10], [±20 (1k)], [±30], [ALL]
 *        * Dynamically calculates chart height to prevent squishing even on ALL strikes
 *    - Live Spot horizontal reference line with label
 * 2. Historical Memory Rewind Scrubber:
 *    - Time slider + quick jump presets (LIVE, -1m, -3m, -5m, -10m, -15m, -30m)
 *    - Dual Ghost comparison bars rendering live vs past on the exact same graph
 *    - Instant snap-back "↺ RETURN TO LIVE" button
 * 3. Multi-Timeframe Trend Matrix:
 *    - 1m, 3m, 5m, 15m rate of change & institutional bias cards
 *    - Unblocks stuck "CALCULATING" status across all timeframes
 * 4. Focused Advisory: Major Structural Levels & Outlier Hotspots:
 *    - Major Levels: Call Wall (Ceiling), Put Wall (Floor), ATM Pin (status: strengthening vs crumbling)
 *    - Outlier Hotspots: Top 2-3 strikes where major volume shifts occurred
 *    - Executive Macro Narrative & Net Flow Pressure Pill
 */

(function () {
    'use strict';

    // State
    var _currentTimeframe = '5m';
    var _currentMetricMode = 'delta'; // 'total' (Whole-Day Cumulative), 'delta' (Contracts/Lots), or 'rate' (Rate/min)
    var _currentStrikeRange = 1000;    // 500, 1000, 1500, or 'all'
    var _rewindTs = null;
    var _historyIndex = [];
    var _isRewound = false;
    var _isScrubbing = false;
    var _lastData = null;
    var _pollTimer = null;

    try {
        var sM = localStorage.getItem('fintel_oi_metric_mode');
        if (sM && (sM === 'total' || sM === 'delta' || sM === 'rate')) _currentMetricMode = sM;
        var sT = localStorage.getItem('fintel_oi_timeframe');
        if (sT) _currentTimeframe = sT;
        var sR = localStorage.getItem('fintel_oi_range');
        if (sR) _currentStrikeRange = (sR === 'all' ? 'all' : parseInt(sR, 10));
    } catch (e) {}

    // ── Fetch Helper ────────────────────────────────────────────────────────
    async function fetchVelocityData(tf, rewindTs, strikeRange) {
        try {
            var rangeVal = (strikeRange !== undefined && strikeRange !== null) ? strikeRange : _currentStrikeRange;
            var url = '/api/oi-velocity?timeframe=' + encodeURIComponent(tf || _currentTimeframe);
            url += '&strike_range=' + encodeURIComponent(rangeVal);
            if (rewindTs !== null && rewindTs !== undefined) {
                url += '&rewind_ts=' + encodeURIComponent(rewindTs);
            }
            url += '&t=' + Date.now();

            var res = await fetch(url);
            if (!res.ok) return null;
            var data = await res.json();
            return data;
        } catch (e) {
            console.warn('[OiVelocityRadar] Fetch error:', e);
            return null;
        }
    }

    // ── WebSocket Message Handler ───────────────────────────────────────────
    function handleWsMessage(msg) {
        if (!msg || msg.type !== 'oi_velocity_update' || !msg.payload) return;
        var payload = msg.payload;

        // Keep history index fresh for the rewind slider
        if (payload.history_index && payload.history_index.length) {
            _historyIndex = payload.history_index;
            updateScrubberBounds();
        }

        // Always update Multi-Timeframe Matrix cards regardless of selected timeframe
        if (payload.analysis) {
            var tfMatrix = payload.analysis.tf_matrix || payload.analysis.timeframe_matrix;
            if (tfMatrix) {
                renderTfMatrix(tfMatrix);
            }
        }

        // If user is currently scrubbing or rewound, do not snap the graph back
        if (_isRewound || _isScrubbing) {
            return;
        }

        // Live mode: update graph and advisory if timeframe matches
        if (payload.timeframe === _currentTimeframe) {
            renderAll(payload);
        }
    }

    // ── Scrubber & Presets Controller ───────────────────────────────────────
    function updateScrubberBounds() {
        var slider = document.getElementById('oi-rewind-slider');
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
        var label = document.getElementById('oi-rewind-time-label');
        var badge = document.getElementById('oi-rewind-status-badge');
        var btnLive = document.getElementById('btn-return-live');

        if (label) {
            label.textContent = timeStr;
            label.style.color = isRewound ? '#ffd54f' : '#00e5ff';
        }

        if (badge) {
            if (isRewound) {
                badge.textContent = 'REWOUND HISTORY';
                badge.style.background = 'rgba(255, 213, 79, 0.18)';
                badge.style.color = '#ffd54f';
                badge.style.borderColor = '#ffd54f';
            } else {
                badge.textContent = 'LIVE STREAMING';
                badge.style.background = 'rgba(0, 230, 118, 0.15)';
                badge.style.color = '#00e676';
                badge.style.borderColor = 'rgba(0, 230, 118, 0.4)';
            }
        }

        if (btnLive) {
            btnLive.style.display = isRewound ? 'inline-flex' : 'none';
        }
    }

    function jumpToPreset(minutesAgo) {
        if (!_historyIndex || _historyIndex.length < 2) return;
        var nowTs = _historyIndex[_historyIndex.length - 1].ts;
        var targetTs = nowTs - (minutesAgo * 60);

        // Find closest snapshot
        var closestIdx = 0;
        var minDiff = Infinity;
        for (var i = 0; i < _historyIndex.length; i++) {
            var diff = Math.abs(_historyIndex[i].ts - targetTs);
            if (diff < minDiff) {
                minDiff = diff;
                closestIdx = i;
            }
        }

        var slider = document.getElementById('oi-rewind-slider');
        if (slider) slider.value = closestIdx;

        applyRewindIndex(closestIdx);
    }

    async function applyRewindIndex(idx) {
        if (!_historyIndex || !_historyIndex[idx]) return;
        var targetSnap = _historyIndex[idx];
        var isLast = (idx === _historyIndex.length - 1);

        if (isLast) {
            _rewindTs = null;
            _isRewound = false;
        } else {
            _rewindTs = targetSnap.ts;
            _isRewound = true;
        }

        updateTimeLabel(targetSnap.time_str || 'HISTORY', _isRewound);
        updatePresetButtonsUI();

        var data = await fetchVelocityData(_currentTimeframe, _rewindTs, _currentStrikeRange);
        if (data && data.ok) {
            renderAll(data);
        }
    }

    function returnToLive() {
        _rewindTs = null;
        _isRewound = false;
        var slider = document.getElementById('oi-rewind-slider');
        if (slider && _historyIndex.length > 0) {
            slider.value = _historyIndex.length - 1;
            var latest = _historyIndex[_historyIndex.length - 1];
            updateTimeLabel(latest.time_str || 'LIVE', false);
        }
        updatePresetButtonsUI();
        refreshLive();
    }

    function updatePresetButtonsUI() {
        var presets = ['live', '1m', '3m', '5m', '10m', '15m', '30m'];
        presets.forEach(function (p) {
            var btn = document.getElementById('btn-preset-' + p);
            if (btn) {
                if (p === 'live' && !_isRewound) {
                    btn.classList.add('active');
                    btn.style.background = 'rgba(0, 230, 118, 0.15)';
                    btn.style.color = '#00e676';
                    btn.style.borderColor = 'rgba(0, 230, 118, 0.4)';
                } else {
                    btn.classList.remove('active');
                    btn.style.background = 'rgba(255, 255, 255, 0.04)';
                    btn.style.color = '#94a3b8';
                    btn.style.borderColor = 'rgba(255, 255, 255, 0.1)';
                }
            }
        });
    }

    // ── Metric Mode & Strike Range Switchers ─────────────────────────────────
    function setMetricMode(mode) {
        if (!mode) mode = 'delta';
        _currentMetricMode = mode;
        try { localStorage.setItem('fintel_oi_metric_mode', mode); } catch (e) {}

        updateMetricButtonsUI();

        // Re-render chart with current data in new metric mode
        if (_lastData) {
            renderPlotlyChart(_lastData);
        }
    }

    function updateMetricButtonsUI() {
        var modes = ['total', 'delta', 'rate'];
        modes.forEach(function (m) {
            var btn = document.getElementById('btn-metric-' + m);
            if (btn) {
                if (m === _currentMetricMode) {
                    btn.classList.add('active');
                    btn.style.background = 'rgba(0, 229, 255, 0.18)';
                    btn.style.color = '#00e5ff';
                    btn.style.borderColor = '#00e5ff';
                } else {
                    btn.classList.remove('active');
                    btn.style.background = 'transparent';
                    btn.style.color = '#94a3b8';
                    btn.style.borderColor = 'transparent';
                }
            }
        });
    }

    function setStrikeRange(rangeVal) {
        _currentStrikeRange = rangeVal;
        try { localStorage.setItem('fintel_oi_range', rangeVal); } catch (e) {}

        updateStrikeRangeButtonsUI();
        refreshLive();
    }

    function updateStrikeRangeButtonsUI() {
        var ranges = [500, 1000, 1500, 'all'];
        ranges.forEach(function (r) {
            var btn = document.getElementById('btn-range-' + r);
            if (btn) {
                if (String(r) === String(_currentStrikeRange)) {
                    btn.classList.add('active');
                    btn.style.background = 'rgba(0, 229, 255, 0.15)';
                    btn.style.color = '#00e5ff';
                    btn.style.borderColor = 'rgba(0, 229, 255, 0.3)';
                    btn.style.borderWidth = '1px';
                    btn.style.borderStyle = 'solid';
                } else {
                    btn.classList.remove('active');
                    btn.style.background = 'transparent';
                    btn.style.color = '#94a3b8';
                    btn.style.border = 'none';
                }
            }
        });
    }

    function setTimeframe(tf) {
        if (!tf) tf = '5m';
        _currentTimeframe = tf;
        try { localStorage.setItem('fintel_oi_timeframe', tf); } catch (e) {}

        updateTimeframeButtonsUI();
        refreshLive();
    }

    function updateTimeframeButtonsUI() {
        ['1m', '3m', '5m', '15m'].forEach(function (t) {
            var btn = document.getElementById('btn-tf-' + t);
            if (btn) {
                if (t === _currentTimeframe) {
                    btn.classList.add('active');
                    btn.style.borderColor = '#00e5ff';
                    btn.style.color = '#00e5ff';
                    btn.style.background = 'rgba(0, 229, 255, 0.18)';
                } else {
                    btn.classList.remove('active');
                    btn.style.borderColor = 'rgba(255, 255, 255, 0.1)';
                    btn.style.color = '#94a3b8';
                    btn.style.background = 'rgba(255, 255, 255, 0.04)';
                }
            }
        });
    }

    async function refreshLive() {
        if (_pendingChartData) {
            var chartDiv = document.getElementById('oi-velocity-chart');
            if (chartDiv && chartDiv.offsetParent !== null) {
                renderPlotlyChart(_pendingChartData);
            }
        }
        var data = await fetchVelocityData(_currentTimeframe, _rewindTs, _currentStrikeRange);
        if (data && data.ok) {
            renderAll(data);
        }
    }

    // ── Macro Day Summary Banner ─────────────────────────────────────────────
    function renderMacroBanner(data) {
        if (!data) return;
        var ceValEl = document.getElementById('oi-macro-ce-val');
        var peValEl = document.getElementById('oi-macro-pe-val');
        var pcrValEl = document.getElementById('oi-macro-pcr-val');
        var biasBadgeEl = document.getElementById('oi-macro-bias-badge');

        var totCe = data.total_call_oi || (data.analysis && data.analysis.total_call_oi) || 0;
        var totPe = data.total_put_oi || (data.analysis && data.analysis.total_put_oi) || 0;
        var pcr = data.day_pcr || (data.analysis && data.analysis.day_pcr) || 1.0;

        if (ceValEl && totCe > 0) {
            ceValEl.textContent = (totCe >= 100000) ? (totCe / 100000).toFixed(2) + 'L' : totCe.toLocaleString();
        }
        if (peValEl && totPe > 0) {
            peValEl.textContent = (totPe >= 100000) ? (totPe / 100000).toFixed(2) + 'L' : totPe.toLocaleString();
        }
        if (pcrValEl) {
            var pcrStr = Number(pcr).toFixed(2);
            var pcrColor = pcr >= 1.2 ? '#00e676' : (pcr <= 0.8 ? '#ff3366' : '#ffd54f');
            pcrValEl.textContent = pcrStr;
            pcrValEl.style.color = pcrColor;
        }
        if (biasBadgeEl) {
            var bias = (data.analysis && data.analysis.bias) || 'BALANCED';
            var biasColor = (data.analysis && data.analysis.bias_color) || '#cbd5e1';
            biasBadgeEl.textContent = 'WHOLE-DAY MACRO: ' + bias;
            biasBadgeEl.style.color = biasColor;
            biasBadgeEl.style.borderColor = biasColor;
            biasBadgeEl.style.background = biasColor + '18';
        }
    }

    // ── Main UI Rendering ───────────────────────────────────────────────────
    function renderAll(data) {
        if (!data) return;
        _lastData = data;

        if (data.history_index && data.history_index.length) {
            _historyIndex = data.history_index;
            updateScrubberBounds();
        }

        _isRewound = !!data.is_rewound;
        var displayTime = _isRewound ? (data.rewind_time_str || 'HISTORY') : (data.live_time_str || 'LIVE');
        updateTimeLabel(displayTime, _isRewound);

        // Keep button active states locked to user's selections
        updateMetricButtonsUI();
        updateStrikeRangeButtonsUI();
        updateTimeframeButtonsUI();

        // 0. Render Macro Whole-Day Context Banner
        renderMacroBanner(data);

        // 1. Render Multi-Timeframe Trend Matrix Cards
        var tfMatrix = data.analysis ? (data.analysis.tf_matrix || data.analysis.timeframe_matrix) : null;
        renderTfMatrix(tfMatrix);

        // 2. Render Focused Advisory: Major Structural Levels
        renderMajorLevels(data.major_levels, data.spot);

        // 3. Render Focused Advisory: Outlier Hotspots & Call/Put Flow
        renderHotspots(data.hotspots, data.analysis, data);

        // 4. Render Bidirectional Plotly Tornado Chart
        renderPlotlyChart(data);
    }

    // ── Multi-Timeframe Trend Matrix ─────────────────────────────────────────
    function renderTfMatrix(tfMatrix) {
        if (!tfMatrix) return;

        ['1m', '3m', '5m', '15m'].forEach(function (tf) {
            var item = tfMatrix[tf];
            var valEl = document.getElementById('tf-val-' + tf);
            var biasEl = document.getElementById('tf-bias-' + tf);
            var cardEl = document.getElementById('tf-card-' + tf);

            if (item) {
                var netVel = item.net_vel || item.net_velocity || 0;
                var sign = netVel > 0 ? '+' : '';
                var kStr = (Math.abs(netVel) >= 1000) ? (sign + (netVel / 1000).toFixed(1) + 'k/m') : (sign + netVel + '/m');

                if (valEl) {
                    valEl.textContent = kStr;
                    valEl.style.color = item.bias_color || '#94a3b8';
                }

                if (biasEl) {
                    biasEl.textContent = item.bias || 'BALANCED';
                    biasEl.style.color = item.bias_color || '#94a3b8';
                }
            }

            if (cardEl) {
                if (tf === _currentTimeframe) {
                    cardEl.style.borderColor = '#00e5ff';
                    cardEl.style.background = 'rgba(0, 229, 255, 0.08)';
                } else {
                    cardEl.style.borderColor = 'rgba(255, 255, 255, 0.08)';
                    cardEl.style.background = 'rgba(15, 23, 42, 0.6)';
                }
            }
        });
    }

    // ── Major Structural Levels Rendering ───────────────────────────────────
    function renderMajorLevels(majorLevels, spot) {
        if (!majorLevels) return;

        var cw = majorLevels.call_wall;
        var pw = majorLevels.put_wall;

        var cwStrikeEl = document.getElementById('major-call-wall-strike');
        var cwDeltaEl = document.getElementById('major-call-wall-delta');
        var pwStrikeEl = document.getElementById('major-put-wall-strike');
        var pwDeltaEl = document.getElementById('major-put-wall-delta');
        var statusTextEl = document.getElementById('major-levels-status-text');

        if (cw && cwStrikeEl) {
            var cwTotalK = cw.total_oi ? (cw.total_oi / 1000).toFixed(1) + 'k' : '';
            cwStrikeEl.textContent = cw.strike ? (cw.strike + ' CE' + (cwTotalK ? ' (' + cwTotalK + ')' : '')) : '--';

            if (cwDeltaEl) {
                var sign = cw.delta >= 0 ? '+' : '';
                var kDelta = (Math.abs(cw.delta) >= 1000) ? (sign + (cw.delta / 1000).toFixed(1) + 'k') : (sign + cw.delta);
                var pctStr = cw.pct_change !== undefined ? ' (' + (cw.pct_change >= 0 ? '+' : '') + cw.pct_change + '%)' : '';
                var lotsStr = cw.lots ? ' [' + Math.abs(cw.lots) + ' lots]' : '';
                cwDeltaEl.textContent = 'Δ ' + kDelta + pctStr + lotsStr;
                cwDeltaEl.style.color = cw.status_color || (cw.delta >= 0 ? '#ff3366' : '#00e5ff');
            }
        }

        if (pw && pwStrikeEl) {
            var pwTotalK = pw.total_oi ? (pw.total_oi / 1000).toFixed(1) + 'k' : '';
            pwStrikeEl.textContent = pw.strike ? (pw.strike + ' PE' + (pwTotalK ? ' (' + pwTotalK + ')' : '')) : '--';

            if (pwDeltaEl) {
                var signP = pw.delta >= 0 ? '+' : '';
                var kDeltaP = (Math.abs(pw.delta) >= 1000) ? (signP + (pw.delta / 1000).toFixed(1) + 'k') : (signP + pw.delta);
                var pctStrP = pw.pct_change !== undefined ? ' (' + (pw.pct_change >= 0 ? '+' : '') + pw.pct_change + '%)' : '';
                var lotsStrP = pw.lots ? ' [' + Math.abs(pw.lots) + ' lots]' : '';
                pwDeltaEl.textContent = 'Δ ' + kDeltaP + pctStrP + lotsStrP;
                pwDeltaEl.style.color = pw.status_color || (pw.delta >= 0 ? '#00e676' : '#ff9100');
            }
        }

        if (statusTextEl) {
            var cwDesc = (cw && cw.status) ? ('Call Wall (' + cw.strike + '): ' + cw.status) : 'Call Wall steady';
            var pwDesc = (pw && pw.status) ? ('Put Wall (' + pw.strike + '): ' + pw.status) : 'Put Wall steady';
            statusTextEl.innerHTML = '<span style="color:#f1f5f9;font-weight:700;">' + cwDesc + '</span> · <span style="color:#f1f5f9;font-weight:700;">' + pwDesc + '</span>';
        }
    }

    // ── Outlier Hotspots Rendering ──────────────────────────────────────────
    function renderHotspots(hotspots, analysis, data) {
        var container = document.getElementById('oi-hotspots-container');
        var netPillEl = document.getElementById('oi-net-pressure-pill');
        var narrativeEl = document.getElementById('oi-advisory-narrative');

        // Net Pressure Pill
        if (netPillEl && analysis) {
            var netVel = analysis.net_velocity || analysis.net_delta || 0;
            var isDelta = (_currentMetricMode === 'delta');
            var unit = isDelta ? ' contracts' : ' / min';
            var sign = netVel >= 0 ? '+' : '';
            var formatted = (Math.abs(netVel) >= 1000) ? (sign + (netVel / 1000).toFixed(1) + 'k' + unit) : (sign + netVel + unit);

            netPillEl.textContent = 'NET: ' + formatted;
            var color = netVel > 1500 ? '#00e676' : (netVel < -1500 ? '#ff3366' : '#ffd54f');
            netPillEl.style.color = color;
            netPillEl.style.borderColor = color;
            netPillEl.style.background = (netVel > 1500 ? 'rgba(0,230,118,0.12)' : (netVel < -1500 ? 'rgba(255,51,102,0.12)' : 'rgba(255,213,79,0.12)'));
        }

        // Narrative
        if (narrativeEl && analysis) {
            var text = analysis.narrative || analysis.advisory_narrative || 'Monitoring institutional positioning changes...';
            narrativeEl.textContent = text;
        }

        // Call vs Put Flow Concentration Breakdown Cards
        var callFlow = (analysis && analysis.call_flow) || (data && data.call_flow) || null;
        var putFlow = (analysis && analysis.put_flow) || (data && data.put_flow) || null;

        var callWriteEl = document.getElementById('oi-call-write-hotspot');
        var callUnwindEl = document.getElementById('oi-call-unwind-hotspot');
        var putWriteEl = document.getElementById('oi-put-write-hotspot');
        var putUnwindEl = document.getElementById('oi-put-unwind-hotspot');

        if (callFlow && callWriteEl && callUnwindEl) {
            if (callFlow.top_writing_strike) {
                var ckW = (Math.abs(callFlow.top_writing_delta) >= 1000) ? ('+' + (callFlow.top_writing_delta / 1000).toFixed(1) + 'k') : ('+' + callFlow.top_writing_delta);
                callWriteEl.innerHTML = 'Writing: <strong style="color:#ff3366;">' + callFlow.top_writing_strike + ' CE</strong> (' + ckW + ')';
            } else {
                callWriteEl.innerHTML = 'Writing: <span style="color:#64748b;">Orderly</span>';
            }

            if (callFlow.top_unwinding_strike) {
                var ckU = (Math.abs(callFlow.top_unwinding_delta) >= 1000) ? ((callFlow.top_unwinding_delta / 1000).toFixed(1) + 'k') : callFlow.top_unwinding_delta;
                callUnwindEl.innerHTML = 'Covering: <strong style="color:#00e5ff;">' + callFlow.top_unwinding_strike + ' CE</strong> (' + ckU + ')';
            } else {
                callUnwindEl.innerHTML = 'Covering: <span style="color:#64748b;">None</span>';
            }
        }

        if (putFlow && putWriteEl && putUnwindEl) {
            if (putFlow.top_writing_strike) {
                var pkW = (Math.abs(putFlow.top_writing_delta) >= 1000) ? ('+' + (putFlow.top_writing_delta / 1000).toFixed(1) + 'k') : ('+' + putFlow.top_writing_delta);
                putWriteEl.innerHTML = 'Writing: <strong style="color:#00e676;">' + putFlow.top_writing_strike + ' PE</strong> (' + pkW + ')';
            } else {
                putWriteEl.innerHTML = 'Writing: <span style="color:#64748b;">Orderly</span>';
            }

            if (putFlow.top_unwinding_strike) {
                var pkU = (Math.abs(putFlow.top_unwinding_delta) >= 1000) ? ((putFlow.top_unwinding_delta / 1000).toFixed(1) + 'k') : putFlow.top_unwinding_delta;
                putUnwindEl.innerHTML = 'Dumping: <strong style="color:#ff9100;">' + putFlow.top_unwinding_strike + ' PE</strong> (' + pkU + ')';
            } else {
                putUnwindEl.innerHTML = 'Dumping: <span style="color:#64748b;">None</span>';
            }
        }

        // Hotspots List
        if (container) {
            if (!hotspots || !hotspots.length) {
                container.innerHTML = '<div style="font-size:11px;color:#64748b;font-style:italic;">No extreme volume surge hotspots detected in this timeframe. Flow is orderly.</div>';
                return;
            }

            var html = '';
            hotspots.forEach(function (h) {
                var isStraddle = (h.type === 'STRADDLE_UNWINDING' || h.type === 'STRADDLE_PINNING');
                var badgeBg = isStraddle ? 'rgba(192, 132, 252, 0.18)' : (h.color ? (h.color + '22') : 'rgba(0,229,255,0.12)');
                var badgeBorder = isStraddle ? '#c084fc' : (h.color || '#00e5ff');
                var deltaSign = h.delta >= 0 ? '+' : '';
                var deltaStr = (Math.abs(h.delta) >= 1000) ? (deltaSign + (h.delta / 1000).toFixed(1) + 'k') : (deltaSign + h.delta);
                var lotsStr = h.lots ? (' (' + Math.abs(h.lots) + ' lots)') : '';
                var pctStr = h.pct_change ? (' [' + (h.pct_change >= 0 ? '+' : '') + h.pct_change + '%]') : '';

                var optLabel = isStraddle ? (h.strike + ' STRADDLE (DUAL FLOW)') : (h.strike + ' ' + (h.option_type || ''));

                html += '<div style="display:flex;justify-content:space-between;align-items:center;padding:5px 8px;background:rgba(15,23,42,0.6);border:1px solid ' + (isStraddle ? 'rgba(192,132,252,0.35)' : 'rgba(255,255,255,0.06)') + ';border-radius:4px;font-family:\'JetBrains Mono\',monospace;font-size:11px;">' +
                    '<div style="display:flex;align-items:center;gap:8px;">' +
                        '<span style="font-weight:900;color:' + badgeBorder + ';background:' + badgeBg + ';border:1px solid ' + badgeBorder + ';padding:1px 6px;border-radius:3px;font-size:9px;letter-spacing:0.5px;">' + (isStraddle ? '⚡ ' + h.type : h.type) + '</span>' +
                        '<span style="font-weight:800;color:#f8fafc;">' + optLabel + '</span>' +
                    '</div>' +
                    '<div style="display:flex;align-items:center;gap:6px;">' +
                        '<span style="font-weight:800;color:' + badgeBorder + ';">Δ ' + deltaStr + pctStr + '</span>' +
                        '<span style="color:#64748b;font-size:10px;">' + lotsStr + '</span>' +
                    '</div>' +
                '</div>';
            });
            container.innerHTML = html;
        }
    }

    var _pendingChartData = null;

    // ── Bidirectional Plotly Tornado Chart ──────────────────────────────────
    function renderPlotlyChart(data) {
        var chartDiv = document.getElementById('oi-velocity-chart');
        if (!chartDiv || !window.Plotly) return;

        var strikesData = (data && data.strikes) ? data.strikes : [];
        var liveStrikesData = (data && data.comparison_live_strikes) ? data.comparison_live_strikes : null;
        var spot = parseFloat((data && data.spot) || 0);

        if (!strikesData.length) {
            Plotly.purge(chartDiv);
            chartDiv.innerHTML = '<div style="display:flex;align-items:center;justify-content:center;height:100%;min-height:300px;color:#64748b;font-size:13px;">Accumulating OI ticks for rate of change calculation...</div>';
            return;
        }

        // Defer rendering if container is hidden to prevent 0x0 collapsed layout
        if (chartDiv.offsetParent === null) {
            _pendingChartData = data;
            return;
        }
        _pendingChartData = null;

        // Sort strikes ascending
        strikesData.sort(function (a, b) { return a.strike - b.strike; });

        // Dynamic Height Calculation so bars never get vertically squished!
        var dynamicHeight = Math.max(540, strikesData.length * 28 + 120);
        chartDiv.style.height = dynamicHeight + 'px';

        var isTotalMode = (_currentMetricMode === 'total');
        var isDeltaMode = (_currentMetricMode === 'delta');

        var strikes = [];
        var ceVals = [];
        var peVals = [];
        var ceColors = [];
        var peColors = [];
        var ceHover = [];
        var peHover = [];

        strikesData.forEach(function (s) {
            strikes.push(s.strike);

            // Calls on Left (negative X)
            var ceDelta = s.call_delta || 0;
            var ceVel = s.call_vel || 0;
            var ceTotalOi = s.call_curr_oi || s.call_oi || 0;
            var ceMetricVal = isTotalMode ? ceTotalOi : (isDeltaMode ? ceDelta : ceVel);

            ceVals.push(-Math.abs(ceMetricVal));

            // Color
            if (isTotalMode) {
                ceColors.push('#38bdf8'); // Sky blue for Call Total OI
            } else {
                var ceIsWriting = (isDeltaMode ? ceDelta : ceVel) >= 0;
                ceColors.push(ceIsWriting ? '#ff3366' : '#00e5ff');
            }

            var ceSign = ceDelta >= 0 ? '+' : '';
            var ceBase = (s.call_base_oi !== undefined) ? s.call_base_oi.toLocaleString() : '--';
            var ceCurr = ceTotalOi.toLocaleString();
            var cePct = (s.call_pct !== undefined) ? (s.call_pct + '%') : '--';
            var ceLots = (s.call_lots !== undefined) ? (s.call_lots + ' lots') : Math.round(ceTotalOi / 65).toLocaleString() + ' lots';

            if (isTotalMode) {
                ceHover.push(
                    '<b>Strike ' + s.strike + ' CALL (TOTAL OI)</b><br>' +
                    'Total Open Interest: ' + ceCurr + ' contracts (' + ceLots + ')<br>' +
                    'Intraday Shift (' + _currentTimeframe + '): ' + ceSign + ceDelta.toLocaleString() + ' (' + cePct + ')<br>' +
                    'Flow Velocity: ' + (ceVel >= 0 ? '+' : '') + ceVel.toLocaleString() + ' / min<br>' +
                    'Signal: <i>' + (s.signal || 'BALANCED') + '</i>'
                );
            } else {
                var ceFlowStr = ((isDeltaMode ? ceDelta : ceVel) >= 0) ? 'Call Writing (Ceiling Defense)' : 'Short Covering (Capitulation)';
                ceHover.push(
                    '<b>Strike ' + s.strike + ' CALL</b><br>' +
                    'Shift: ' + ceSign + ceDelta.toLocaleString() + ' contracts (' + ceLots + ', ' + cePct + ')<br>' +
                    'Base OI ➔ Curr OI: ' + ceBase + ' ➔ ' + ceCurr + '<br>' +
                    'Velocity: ' + (ceVel >= 0 ? '+' : '') + ceVel.toLocaleString() + ' / min<br>' +
                    'Flow: ' + ceFlowStr + '<br>' +
                    'Signal: <i>' + (s.signal || 'BALANCED') + '</i>'
                );
            }

            // Puts on Right (positive X)
            var peDelta = s.put_delta || 0;
            var peVel = s.put_vel || 0;
            var peTotalOi = s.put_curr_oi || s.put_oi || 0;
            var peMetricVal = isTotalMode ? peTotalOi : (isDeltaMode ? peDelta : peVel);

            peVals.push(Math.abs(peMetricVal));

            // Color
            if (isTotalMode) {
                peColors.push('#10b981'); // Emerald green for Put Total OI
            } else {
                var peIsWriting = (isDeltaMode ? peDelta : peVel) >= 0;
                peColors.push(peIsWriting ? '#00e676' : '#ff9100');
            }

            var peSign = peDelta >= 0 ? '+' : '';
            var peBase = (s.put_base_oi !== undefined) ? s.put_base_oi.toLocaleString() : '--';
            var peCurr = peTotalOi.toLocaleString();
            var pePct = (s.put_pct !== undefined) ? (s.put_pct + '%') : '--';
            var peLots = (s.put_lots !== undefined) ? (s.put_lots + ' lots') : Math.round(peTotalOi / 65).toLocaleString() + ' lots';

            if (isTotalMode) {
                peHover.push(
                    '<b>Strike ' + s.strike + ' PUT (TOTAL OI)</b><br>' +
                    'Total Open Interest: ' + peCurr + ' contracts (' + peLots + ')<br>' +
                    'Intraday Shift (' + _currentTimeframe + '): ' + peSign + peDelta.toLocaleString() + ' (' + pePct + ')<br>' +
                    'Flow Velocity: ' + (peVel >= 0 ? '+' : '') + peVel.toLocaleString() + ' / min<br>' +
                    'Signal: <i>' + (s.signal || 'BALANCED') + '</i>'
                );
            } else {
                var peFlowStr = ((isDeltaMode ? peDelta : peVel) >= 0) ? 'Put Writing (Floor Defense)' : 'Put Capitulation (Panic Exit)';
                peHover.push(
                    '<b>Strike ' + s.strike + ' PUT</b><br>' +
                    'Shift: ' + peSign + peDelta.toLocaleString() + ' contracts (' + peLots + ', ' + pePct + ')<br>' +
                    'Base OI ➔ Curr OI: ' + peBase + ' ➔ ' + peCurr + '<br>' +
                    'Velocity: ' + (peVel >= 0 ? '+' : '') + peVel.toLocaleString() + ' / min<br>' +
                    'Flow: ' + peFlowStr + '<br>' +
                    'Signal: <i>' + (s.signal || 'BALANCED') + '</i>'
                );
            }
        });

        // Dynamic symmetric X-axis range
        var maxVal = 1000;
        strikesData.forEach(function (s) {
            var ceV = Math.abs(isTotalMode ? (s.call_curr_oi || s.call_oi || 0) : (isDeltaMode ? (s.call_delta || 0) : (s.call_vel || 0)));
            var peV = Math.abs(isTotalMode ? (s.put_curr_oi || s.put_oi || 0) : (isDeltaMode ? (s.put_delta || 0) : (s.put_vel || 0)));
            maxVal = Math.max(maxVal, ceV, peV);
        });
        var step = maxVal > 500000 ? 100000 : (maxVal > 100000 ? 50000 : (maxVal > 50000 ? 25000 : (maxVal > 10000 ? 5000 : 2000)));
        maxVal = Math.ceil((maxVal * 1.18) / step) * step;

        var barWidth = Math.max(16, Math.min(32, Math.floor(620 / Math.max(strikes.length, 10))));

        var traces = [
            // Trace 0: Calls (Left side)
            {
                type: 'bar',
                orientation: 'h',
                x: ceVals,
                y: strikes,
                name: isTotalMode ? 'Call Total OI' : ('Call Flow (' + (_isRewound ? 'Rewound' : 'Live') + ')'),
                marker: {
                    color: ceColors,
                    opacity: 0.9,
                    line: { color: 'rgba(255, 255, 255, 0.15)', width: 1 }
                },
                hoverinfo: 'text',
                hovertext: ceHover,
                width: barWidth
            },
            // Trace 1: Puts (Right side)
            {
                type: 'bar',
                orientation: 'h',
                x: peVals,
                y: strikes,
                name: isTotalMode ? 'Put Total OI' : ('Put Flow (' + (_isRewound ? 'Rewound' : 'Live') + ')'),
                marker: {
                    color: peColors,
                    opacity: 0.9,
                    line: { color: 'rgba(255, 255, 255, 0.15)', width: 1 }
                },
                hoverinfo: 'text',
                hovertext: peHover,
                width: barWidth
            }
        ];

        // If Rewound, add Dual Ghost comparison bars from liveStrikesData
        if (_isRewound && liveStrikesData && liveStrikesData.length && !isTotalMode) {
            var liveMap = {};
            liveStrikesData.forEach(function (ls) { liveMap[ls.strike] = ls; });

            var liveCeVals = [];
            var livePeVals = [];
            var liveCeHover = [];
            var livePeHover = [];

            strikes.forEach(function (stk) {
                var ls = liveMap[stk];
                if (ls) {
                    var lCeVal = isDeltaMode ? (ls.call_delta || 0) : (ls.call_vel || 0);
                    var lPeVal = isDeltaMode ? (ls.put_delta || 0) : (ls.put_vel || 0);
                    liveCeVals.push(-Math.abs(lCeVal));
                    livePeVals.push(Math.abs(lPeVal));
                    liveCeHover.push('<b>CURRENT LIVE CALL FLOW</b><br>' + (isDeltaMode ? 'Live Δ: ' : 'Live Rate: ') + lCeVal.toLocaleString());
                    livePeHover.push('<b>CURRENT LIVE PUT FLOW</b><br>' + (isDeltaMode ? 'Live Δ: ' : 'Live Rate: ') + lPeVal.toLocaleString());
                } else {
                    liveCeVals.push(0);
                    livePeVals.push(0);
                    liveCeHover.push('');
                    livePeHover.push('');
                }
            });

            traces.push({
                type: 'bar',
                orientation: 'h',
                x: liveCeVals,
                y: strikes,
                name: 'Current Live Calls (Ghost)',
                marker: {
                    color: 'rgba(255, 51, 102, 0.12)',
                    line: { color: '#ff3366', width: 1.5, dash: 'dot' }
                },
                hoverinfo: 'text',
                hovertext: liveCeHover,
                width: barWidth + 6
            });

            traces.push({
                type: 'bar',
                orientation: 'h',
                x: livePeVals,
                y: strikes,
                name: 'Current Live Puts (Ghost)',
                marker: {
                    color: 'rgba(0, 230, 118, 0.12)',
                    line: { color: '#00e676', width: 1.5, dash: 'dot' }
                },
                hoverinfo: 'text',
                hovertext: livePeHover,
                width: barWidth + 6
            });
        }

        // Layout
        var yMin = strikes[0] - 25;
        var yMax = strikes[strikes.length - 1] + 25;

        var shapes = [];
        var annotations = [];

        // Vertical Center Axis
        shapes.push({
            type: 'line',
            x0: 0,
            x1: 0,
            y0: yMin,
            y1: yMax,
            line: { color: 'rgba(0, 229, 255, 0.4)', width: 1.5, dash: 'solid' }
        });

        // Spot Horizontal Line
        if (spot > 0 && spot >= yMin && spot <= yMax) {
            shapes.push({
                name: 'oi_spot_line',
                type: 'line',
                x0: -maxVal,
                x1: maxVal,
                y0: spot,
                y1: spot,
                line: { color: '#ffd54f', width: 2, dash: 'dash' }
            });

            annotations.push({
                x: 0,
                y: spot,
                text: '🎯 LIVE SPOT: ' + spot.toFixed(1),
                showarrow: false,
                font: { color: '#ffd54f', size: 11, family: 'JetBrains Mono, monospace', weight: 'bold' },
                bgcolor: 'rgba(18, 22, 46, 0.92)',
                bordercolor: '#ffd54f',
                borderwidth: 1,
                borderpad: 4
            });
        }

        // Title annotations on graph header
        var ceHeaderTxt = isTotalMode ? '◀ CALL TOTAL OPEN INTEREST (WHOLE DAY)' : (isDeltaMode ? '◀ CALL Δ SHIFT (UNWINDING vs WRITING)' : '◀ CALL VELOCITY (UNWINDING vs WRITING)');
        var peHeaderTxt = isTotalMode ? 'PUT TOTAL OPEN INTEREST (WHOLE DAY) ▶' : (isDeltaMode ? 'PUT Δ SHIFT (WRITING vs UNWINDING) ▶' : 'PUT VELOCITY (WRITING vs UNWINDING) ▶');

        annotations.push({
            xref: 'paper',
            yref: 'paper',
            x: 0.18,
            y: 1.04,
            text: ceHeaderTxt,
            showarrow: false,
            font: { color: isTotalMode ? '#38bdf8' : '#ff3366', size: 11, weight: 'bold' }
        });

        annotations.push({
            xref: 'paper',
            yref: 'paper',
            x: 0.82,
            y: 1.04,
            text: peHeaderTxt,
            showarrow: false,
            font: { color: isTotalMode ? '#10b981' : '#00e676', size: 11, weight: 'bold' }
        });

        var xAxisLabel = isTotalMode ? 'Total Cumulative Open Interest (Contracts)' : (isDeltaMode ? 'Net Δ OI Shift (Contracts)' : 'Rate of Change (ΔOI / min)');

        var layout = {
            barmode: _isRewound ? 'overlay' : 'relative',
            bargap: 0.22,
            margin: { l: 65, r: 35, t: 45, b: 40 },
            paper_bgcolor: 'transparent',
            plot_bgcolor: 'rgba(10, 14, 30, 0.55)',
            showlegend: false,
            xaxis: {
                title: {
                    text: xAxisLabel,
                    font: { color: '#64748b', size: 11 }
                },
                range: [-maxVal, maxVal],
                zeroline: false,
                gridcolor: 'rgba(255, 255, 255, 0.05)',
                tickfont: { color: '#94a3b8', size: 10, family: 'JetBrains Mono, monospace' },
                tickformat: '+d'
            },
            yaxis: {
                title: {
                    text: 'Strike',
                    font: { color: '#64748b', size: 11 }
                },
                range: [yMin, yMax],
                autorange: false,
                fixedrange: true,
                dtick: 50,
                gridcolor: 'rgba(255, 255, 255, 0.05)',
                tickfont: { color: '#f1f5f9', size: 11, family: 'JetBrains Mono, monospace', weight: 'bold' }
            },
            shapes: shapes,
            annotations: annotations,
            hovermode: 'closest'
        };

        var config = {
            responsive: true,
            displayModeBar: false
        };

        Plotly.react(chartDiv, traces, layout, config);
    }

    // ── Setup Event Listeners (Document-Level Delegation) ────────────────────
    function initListeners() {
        // Document-level event delegation guarantees buttons NEVER fail or unbind!
        document.addEventListener('click', function (e) {
            // 1. Metric Mode buttons
            var metricBtn = e.target.closest('.oi-metric-btn');
            if (metricBtn) {
                var mId = metricBtn.id.replace('btn-metric-', '');
                setMetricMode(mId);
                return;
            }

            // 2. Strike Range buttons
            var rangeBtn = e.target.closest('.oi-range-btn');
            if (rangeBtn) {
                var rId = rangeBtn.id.replace('btn-range-', '');
                var rVal = (rId === 'all') ? 'all' : parseInt(rId, 10);
                setStrikeRange(rVal);
                return;
            }

            // 3. Timeframe buttons
            var tfBtn = e.target.closest('.oi-tf-btn');
            if (tfBtn) {
                var tf = tfBtn.id.replace('btn-tf-', '');
                setTimeframe(tf);
                return;
            }

            // 4. Timeframe cards in Multi-Timeframe Matrix
            var tfCard = e.target.closest('[id^="tf-card-"]');
            if (tfCard) {
                var cardTf = tfCard.id.replace('tf-card-', '');
                setTimeframe(cardTf);
                return;
            }

            // 5. Presets
            var presetBtn = e.target.closest('.oi-preset-btn');
            if (presetBtn) {
                var pId = presetBtn.id.replace('btn-preset-', '');
                if (pId === 'live') returnToLive();
                else {
                    var mins = parseInt(pId.replace('m', ''), 10);
                    if (!isNaN(mins)) jumpToPreset(mins);
                }
                return;
            }

            // 6. Return to LIVE button
            var liveBtn = e.target.closest('#btn-return-live');
            if (liveBtn) {
                returnToLive();
                return;
            }
        });

        // Rewind Slider delegation
        document.addEventListener('input', function (e) {
            if (e.target && e.target.id === 'oi-rewind-slider') {
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
            if (e.target && e.target.id === 'oi-rewind-slider') {
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
        refreshLive();

        // Fallback polling every 5s if WebSocket is idle
        if (_pollTimer) clearInterval(_pollTimer);
        _pollTimer = setInterval(function () {
            if (!_isRewound && !_isScrubbing) {
                refreshLive();
            }
        }, 5000);
    }

    // Expose on window
    window.OiVelocityRadar = {
        handleWsMessage: handleWsMessage,
        refresh: refreshLive,
        setTimeframe: setTimeframe,
        setMetricMode: setMetricMode,
        setStrikeRange: setStrikeRange,
        jumpToPreset: jumpToPreset,
        returnToLive: returnToLive
    };

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }
})();
