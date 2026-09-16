/**
 * ═══════════════════════════════════════════════════════════════════════════
 * F-INTEL QUANTITATIVE TERMINAL — CORE DASHBOARD CONTROLLER
 * ═══════════════════════════════════════════════════════════════════════════
 * Manages tab switching, WebSocket streaming, fragment live-updates,
 * and institutional API integration.
 */

(function () {
    'use strict';

    // ── Day / Night Theme Toggle ──────────────────────────────────────────────
    var _currentTheme = 'dark';

    function applyTheme(theme) {
        _currentTheme = theme;
        document.documentElement.setAttribute('data-theme', theme);

        var isDark = (theme === 'dark');
        var icon  = document.getElementById('theme-icon');
        var label = document.getElementById('theme-label');
        var input = document.getElementById('theme-toggle-input');
        var meta  = document.getElementById('theme-color-meta');

        if (icon)  icon.textContent  = isDark ? '\uD83C\uDF19' : '\u2600\uFE0F';
        if (label) label.textContent = isDark ? 'NIGHT' : 'DAY';
        if (input) input.checked     = !isDark;  // checked = light mode
        if (meta)  meta.content      = isDark ? '#131722' : '#f0f3fa';

        try { localStorage.setItem('fintel_theme', theme); } catch (e) {}
    }

    // Expose globally so the HTML onclick can call it
    window.toggleTheme = function () {
        applyTheme(_currentTheme === 'dark' ? 'light' : 'dark');
    };

    // Initialize from saved preference & inject toggle widget
    (function initTheme() {
        var saved = 'dark';
        try { saved = localStorage.getItem('fintel_theme') || 'dark'; } catch (e) {}

        // Apply immediately before DOM ready (sets CSS variables on <html>)
        document.documentElement.setAttribute('data-theme', saved);
        _currentTheme = saved;

        document.addEventListener('DOMContentLoaded', function () {
            // ── Hide legacy canvas + FX toggle (server keeps re-injecting them) ──
            var canvas = document.getElementById('quant-bg-canvas');
            if (canvas) { canvas.style.display = 'none'; canvas.style.animation = 'none'; }

            var fxWrap = document.querySelector('.fx-toggle-wrap');
            if (fxWrap) fxWrap.style.display = 'none';

            // ── Inject Day/Night toggle widget if not present ──
            if (!document.getElementById('theme-toggle-input')) {
                var marketStrip = document.querySelector('.market-strip');
                if (marketStrip) {
                    var toggleDiv = document.createElement('div');
                    toggleDiv.className = 'theme-toggle-wrap';
                    toggleDiv.title = 'Toggle Day / Night mode';
                    toggleDiv.innerHTML =
                        '<span class="theme-toggle-icon" id="theme-icon">\uD83C\uDF19</span>' +
                        '<label class="theme-toggle">' +
                            '<input type="checkbox" id="theme-toggle-input">' +
                            '<span class="theme-slider"></span>' +
                        '</label>' +
                        '<span style="font-size:11px;color:var(--text-muted);font-weight:600;" id="theme-label">NIGHT</span>';
                    marketStrip.appendChild(toggleDiv);
                }
            }

            // ── Wire up checkbox & apply saved theme ──
            applyTheme(saved);
            var input = document.getElementById('theme-toggle-input');
            if (input) {
                input.addEventListener('change', function () {
                    applyTheme(input.checked ? 'light' : 'dark');
                });
            }

            // ── Hide time emoji (cleanup) ──
            var timeEl = document.getElementById('time-display');
            if (timeEl && timeEl.textContent.includes('\u23F1')) {
                timeEl.textContent = timeEl.textContent.replace(/[^\d:]/g, '').trim();
            }
        });
    })();

    var activeTab = 'regime';
    var ws = null;
    var wsReconnectTimer = null;

    // ── Tab Management ──
    function switchTab(name) {
        activeTab = name;

        // Update Tab Buttons
        var btns = document.querySelectorAll('.tab-btn');
        btns.forEach(function (btn) {
            var tabName = btn.getAttribute('data-tab');
            if (tabName === name) {
                btn.classList.add('active');
            } else {
                btn.classList.remove('active');
            }
        });

        // Update Tab Content Panels
        var tabs = document.querySelectorAll('.tab-content');
        tabs.forEach(function (tab) {
            var tabId = tab.id.replace('tab-', '');
            if (tabId === name) {
                tab.classList.add('active');
            } else {
                tab.classList.remove('active');
            }
        });

        // Save preference and update URL hash
        try {
            localStorage.setItem('fintel_active_tab', name);
            if (window.location.hash !== '#' + name) {
                window.history.replaceState(null, '', '#' + name);
            }
        } catch (e) {}

        // Trigger Plotly chart resize for visible layout
        setTimeout(function () {
            window.dispatchEvent(new Event('resize'));
            if (name === 'theta' && typeof window.updateThetaSim === 'function') {
                window.updateThetaSim();
            }
            if (name === 'mm') {
                cleanupMMTab();
                if (window.GammaExplosionTerminal && typeof window.GammaExplosionTerminal.refresh === 'function') {
                    window.GammaExplosionTerminal.refresh();
                }
            }
            if (name === 'chain') {
                optimizeOptionChainAndWalls();
                var gexChart = document.getElementById('gex-distribution-chart');
                if (gexChart && window.Plotly && typeof window.Plotly.Plots.resize === 'function') {
                    window.Plotly.Plots.resize(gexChart);
                }
                if (window.GexRebalanceRadar && typeof window.GexRebalanceRadar.refresh === 'function') {
                    window.GexRebalanceRadar.refresh();
                }
            }
        }, 50);
    }

    function executeScripts(container) {
        if (!container) return;
        container.querySelectorAll('script').forEach(function (old) {
            var s = document.createElement('script');
            Array.from(old.attributes).forEach(function (a) { s.setAttribute(a.name, a.value); });
            s.textContent = old.textContent;
            old.parentNode.replaceChild(s, old);
        });
    }

    // ── MM Tab Cleanup (Remove Candlesticks, Reel 1, Reel 2) ──
    function cleanupMMTab() {
        var mmTab = document.getElementById('tab-mm');
        if (!mmTab) return;
        mmTab.querySelectorAll('.ge-chart-card, .ge-grid, #ge-interactive-chart, #ge-reel1-spotlight, #ge-reel2-spotlight').forEach(function (el) {
            el.remove();
        });
    }

    // ── Option Chain & GEX Wall Enhancements ──
    function optimizeOptionChainAndWalls() {
        var chainTab = document.getElementById('tab-chain');
        if (!chainTab) return;

        // 1. Move #gex-rebalance-card to the very bottom so chain is immediately visible
        var rebCard = chainTab.querySelector('#gex-rebalance-card');
        if (rebCard) {
            chainTab.appendChild(rebCard);
        }

        // 2. Extract Key Metrics for Corridor Runway
        var putWall1 = 0, callWall1 = 0, putWall2 = 0, callWall2 = 0, maxPain = 0, pcr = '--', flow = '--';
        var mBoxes = chainTab.querySelectorAll('.metric-box');
        mBoxes.forEach(function (mb) {
            var lbl = (mb.querySelector('.metric-label') || {}).textContent || '';
            var valEl = mb.querySelector('div:not(.metric-label):not(.metric-sub)');
            var val = valEl ? valEl.textContent.trim() : '';
            var sub = (mb.querySelector('.metric-sub') || {}).textContent || '';

            if (lbl.includes('Put Wall ①')) {
                putWall1 = parseFloat(val.replace(/,/g, '')) || 0;
                var m = sub.match(/②\s*(\d+)/);
                if (m) putWall2 = parseFloat(m[1]) || 0;
            } else if (lbl.includes('Call Wall ①')) {
                callWall1 = parseFloat(val.replace(/,/g, '')) || 0;
                var m = sub.match(/②\s*(\d+)/);
                if (m) callWall2 = parseFloat(m[1]) || 0;
            } else if (lbl.includes('Max Pain')) {
                maxPain = parseFloat(val.replace(/,/g, '')) || 0;
            } else if (lbl.includes('PCR')) {
                pcr = val;
            } else if (lbl.includes('15m OI VELOCITY')) {
                flow = val;
            }
        });

        // Get live spot
        var spotEl = document.getElementById('spot-display');
        var spot = 0;
        if (spotEl) {
            var sv = spotEl.querySelector('.spot-val');
            if (sv) spot = parseFloat(sv.textContent.replace(/,/g, '')) || 0;
        }
        if (!spot && putWall1 && callWall1) spot = Math.round((putWall1 + callWall1) / 2);

        // 3. Ensure removed GEX Corridor Runway Bar is stripped if present
        var existingRunway = chainTab.querySelector('.gex-corridor-runway-bar');
        if (existingRunway) {
            existingRunway.remove();
        }

        // 4. Enhance Option Chain Table with Glowing Markers & OI Depth Bars
        var tables = chainTab.querySelectorAll('table.data-table');
        tables.forEach(function (table) {
            var rows = table.querySelectorAll('tbody tr');
            if (!rows || rows.length === 0) return;

            // Compute Max OI for relative depth bars
            var maxCeOi = 1, maxPeOi = 1;
            rows.forEach(function (r) {
                var tds = r.querySelectorAll('td');
                if (tds.length >= 5) {
                    var ceVal = parseFloat(tds[0].textContent.replace(/,/g, '')) || 0;
                    var peVal = parseFloat(tds[tds.length - 1].textContent.replace(/,/g, '')) || 0;
                    if (ceVal > maxCeOi) maxCeOi = ceVal;
                    if (peVal > maxPeOi) maxPeOi = peVal;
                }
            });

            rows.forEach(function (r) {
                var tds = r.querySelectorAll('td');
                if (tds.length < 3) return;

                // Find strike cell (supports .strike-cell, data-strike, or fallback to index)
                var strikeCell = r.querySelector('.strike-cell') || (tds.length === 5 ? tds[2] : (tds.length >= 11 ? tds[5] : null));
                if (!strikeCell) {
                    for (var i = 0; i < tds.length; i++) {
                        var num = parseFloat(tds[i].textContent.replace(/,/g, ''));
                        if (num >= 10000 && num <= 60000) { strikeCell = tds[i]; break; }
                    }
                }
                if (!strikeCell) return;

                var sVal = parseFloat(strikeCell.getAttribute('data-strike')) || parseFloat(strikeCell.textContent.replace(/[^0-9.]/g, '')) || 0;
                if (!sVal) return;

                // Glowing Wall Badges & Row Highlights
                if (callWall1 && sVal === callWall1) {
                    r.classList.add('glow-call-wall');
                    if (!strikeCell.querySelector('.wall-badge-cw1')) {
                        strikeCell.innerHTML += ' <span class="wall-badge-cw1">CW ①</span>';
                    }
                } else if (callWall2 && sVal === callWall2) {
                    r.classList.add('glow-call-wall-2');
                    if (!strikeCell.querySelector('.wall-badge-cw2')) {
                        strikeCell.innerHTML += ' <span class="wall-badge-cw2">CW ②</span>';
                    }
                } else if (putWall1 && sVal === putWall1) {
                    r.classList.add('glow-put-wall');
                    if (!strikeCell.querySelector('.wall-badge-pw1')) {
                        strikeCell.innerHTML += ' <span class="wall-badge-pw1">PW ①</span>';
                    }
                } else if (putWall2 && sVal === putWall2) {
                    r.classList.add('glow-put-wall-2');
                    if (!strikeCell.querySelector('.wall-badge-pw2')) {
                        strikeCell.innerHTML += ' <span class="wall-badge-pw2">PW ②</span>';
                    }
                } else if (Math.abs(sVal - spot) <= 25) {
                    r.classList.add('glow-atm');
                    if (!strikeCell.querySelector('.wall-badge-atm')) {
                        strikeCell.innerHTML += ' <span class="wall-badge-atm">ATM</span>';
                    }
                }

                if (maxPain && sVal === maxPain && !strikeCell.querySelector('.wall-badge-pain')) {
                    strikeCell.innerHTML += ' <span class="wall-badge-pain">PAIN</span>';
                }

                // Mini OI Depth Bars behind CE and PE OI numbers
                var ceTd = tds[0];
                var peTd = tds[tds.length - 1];
                if (ceTd && maxCeOi > 0) {
                    var ceOi = parseFloat(ceTd.textContent.replace(/,/g, '')) || 0;
                    var cePct = Math.min(100, Math.round((ceOi / maxCeOi) * 100));
                    ceTd.style.background = 'linear-gradient(270deg, rgba(255, 51, 102, 0.22) ' + cePct + '%, transparent ' + cePct + '%)';
                }
                if (peTd && maxPeOi > 0) {
                    var peOi = parseFloat(peTd.textContent.replace(/,/g, '')) || 0;
                    var pePct = Math.min(100, Math.round((peOi / maxPeOi) * 100));
                    peTd.style.background = 'linear-gradient(90deg, rgba(0, 230, 118, 0.22) ' + pePct + '%, transparent ' + pePct + '%)';
                }
            });
        });
    }

    // ── Live Fragment Polling (HTTP Hot-Reload) ──
    function refreshContent() {
        fetch('/fragment?t=' + Date.now())
            .then(function (res) {
                if (!res.ok) throw new Error('Fragment fetch failed with status ' + res.status);
                return res.text();
            })
            .then(function (html) {
                if (!html || html.length < 50) return;
                var parser = new DOMParser();
                var doc = parser.parseFromString(html, 'text/html');

                var tabNames = ['regime', 'iv', 'vol', 'chain', 'theta', 'prob', 'mm'];
                tabNames.forEach(function (t) {
                    try {
                        var frag = doc.getElementById('frag-' + t);
                        var dest = document.getElementById('tab-' + t);
                        if (frag && dest) {
                            // Preserve active capital input if user is interacting with it
                            var oldCap = dest.querySelector('#sz-capital-input');
                            var activeCapVal = (oldCap && document.activeElement === oldCap) ? oldCap.value : null;

                            dest.innerHTML = frag.innerHTML;

                            if (activeCapVal !== null) {
                                var newCap = dest.querySelector('#sz-capital-input');
                                if (newCap) {
                                    newCap.value = activeCapVal;
                                    newCap.focus();
                                }
                            }

                            executeScripts(dest);
                        }
                    } catch (tabErr) {
                        console.warn('Error updating tab ' + t + ':', tabErr);
                    }
                });

                // Enforce instant visual optimizations on chain and mm tabs
                optimizeOptionChainAndWalls();
                cleanupMMTab();

                // Update Spot Pill & Verdict
                var spotFrag = doc.getElementById('frag-spot');
                if (spotFrag) {
                    var spotVal = spotFrag.getAttribute('data-spot');
                    var timeVal = spotFrag.getAttribute('data-time');

                    var spotEl = document.getElementById('spot-display');
                    if (spotEl && spotVal) {
                        spotEl.innerHTML = 'SPOT: <span class="spot-val">' + Number(spotVal).toLocaleString('en-IN', { minimumFractionDigits: 2 }) + '</span>';
                        updateGexChartSpot(spotVal);
                    }

                    // Always update time display with latest refresh timestamp
                    var timeEl = document.getElementById('time-display');
                    if (timeEl && timeVal) {
                        var isWs = (ws && ws.readyState === WebSocket.OPEN);
                        timeEl.innerHTML = '&#128339; ' + timeVal + (isWs ? ' &nbsp;|&nbsp; Live' : ' &nbsp;|&nbsp; Auto-Refreshed');
                    }

                    var verdTransfer = doc.getElementById('frag-verdict-transfer');
                    var verdDest = document.getElementById('top-verdict-pill');
                    if (verdTransfer && verdDest) {
                        verdDest.innerHTML = verdTransfer.innerHTML;
                    }
                }

                // Visual flash on status badge to confirm live update
                var badge = document.getElementById('ws-status-badge');
                if (badge) {
                    badge.style.opacity = '0.35';
                    setTimeout(function () { badge.style.opacity = '1'; }, 300);
                }

                // Re-apply filters and simulation after DOM injection
                try {
                    if (typeof window.applyThetaFilters === 'function') window.applyThetaFilters();
                    if (typeof window.updateThetaSim === 'function') window.updateThetaSim();
                    if (window.GammaExplosionTerminal && typeof window.GammaExplosionTerminal.refresh === 'function') {
                        window.GammaExplosionTerminal.refresh();
                    }
                    if (window.GexRebalanceRadar && typeof window.GexRebalanceRadar.refresh === 'function') {
                        window.GexRebalanceRadar.refresh();
                    }
                } catch (e) {
                    console.warn('Post-update hook error:', e);
                }
            })
            .catch(function (err) {
                console.warn('Fragment refresh error (retrying):', err);
            });
    }

    // ── WebSocket Real-Time Ticker ──
    function initWebSocket() {
        if (wsReconnectTimer) clearTimeout(wsReconnectTimer);

        var wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        var wsUrl = wsProtocol + '//' + window.location.host + '/stream';

        var badge = document.getElementById('ws-status-badge');

        try {
            ws = new WebSocket(wsUrl);

            ws.onopen = function () {
                if (badge) {
                    badge.textContent = 'LIVE WS';
                    badge.style.background = 'var(--color-bullish-dim, rgba(16, 185, 129, 0.15))';
                    badge.style.color = 'var(--color-bullish, #10b981)';
                    badge.style.borderColor = 'var(--color-bullish-glow, #10b981)';
                }
            };

            ws.onmessage = function (event) {
                try {
                    var msg = JSON.parse(event.data);

                    // Initial state on connect
                    if (msg.type === 'init' && msg.data) {
                        if (msg.data.spot) {
                            var spotEl = document.getElementById('spot-display');
                            if (spotEl) {
                                spotEl.innerHTML = 'SPOT: <span class="spot-val">' + Number(msg.data.spot).toLocaleString('en-IN', { minimumFractionDigits: 2 }) + '</span>';
                            }
                        }
                        var initTime = msg.data.last_update || msg.data.time;
                        if (initTime) {
                            var timeEl = document.getElementById('time-display');
                            if (timeEl) {
                                timeEl.innerHTML = '&#128339; ' + initTime + ' &nbsp;|&nbsp; Live';
                            }
                        }
                    }

                    // Real-time ticks: match BOTH msg.type === 'tick' and 'spot_tick'
                    if ((msg.type === 'tick' || msg.type === 'spot_tick') && msg.spot) {
                        var spotEl = document.getElementById('spot-display');
                        if (spotEl) {
                            spotEl.innerHTML = 'SPOT: <span class="spot-val">' + Number(msg.spot).toLocaleString('en-IN', { minimumFractionDigits: 2 }) + '</span>';
                        }
                        updateGexChartSpot(msg.spot);
                        var tickTime = msg.time || msg.server_time;
                        if (tickTime) {
                            var timeEl = document.getElementById('time-display');
                            if (timeEl) {
                                timeEl.innerHTML = '&#128339; ' + tickTime + ' &nbsp;|&nbsp; Live';
                            }
                        }
                    }

                    if (msg.server_time || msg.time) {
                        var timeEl = document.getElementById('time-display');
                        if (timeEl) {
                            timeEl.innerHTML = '&#128339; ' + (msg.server_time || msg.time) + ' &nbsp;|&nbsp; Live';
                        }
                    }

                    if (msg.type === 'gamma_explosion_update' && window.GammaExplosionTerminal) {
                        window.GammaExplosionTerminal.handleWsMessage(msg);
                    }

                    if (msg.type === 'gex_rebalance_update' && window.GexRebalanceRadar) {
                        window.GexRebalanceRadar.handleWsMessage(msg);
                    }
                } catch (e) {}
            };

            ws.onerror = function () {
                if (badge) {
                    badge.textContent = 'POLLING';
                    badge.style.background = 'var(--color-warning-dim, rgba(245, 158, 11, 0.15))';
                    badge.style.color = 'var(--color-warning, #f59e0b)';
                    badge.style.borderColor = 'rgba(245, 158, 11, 0.3)';
                }
            };

            ws.onclose = function () {
                if (badge) {
                    badge.textContent = 'RECONNECTING';
                    badge.style.background = 'var(--color-warning-dim, rgba(245, 158, 11, 0.15))';
                    badge.style.color = 'var(--color-warning, #f59e0b)';
                }
                wsReconnectTimer = setTimeout(initWebSocket, 4000);
            };
        } catch (e) {
            wsReconnectTimer = setTimeout(initWebSocket, 5000);
        }
    }

    // ── Dynamic GEX Chart Spot Slide ──
    function updateGexChartSpot(newSpot) {
        var chartEl = document.getElementById('gex-distribution-chart');
        if (!chartEl || !chartEl.layout || !window.Plotly) return;
        var spotVal = parseFloat(newSpot);
        if (!spotVal || isNaN(spotVal)) return;

        var shapes = chartEl.layout.shapes || [];
        var annotations = chartEl.layout.annotations || [];
        var update = {};

        for (var i = 0; i < shapes.length; i++) {
            if (shapes[i].name === 'spot_line') {
                update['shapes[' + i + '].y0'] = spotVal;
                update['shapes[' + i + '].y1'] = spotVal;
                break;
            }
        }
        for (var j = 0; j < annotations.length; j++) {
            if (annotations[j].name === 'spot_annotation') {
                update['annotations[' + j + '].y'] = spotVal;
                update['annotations[' + j + '].text'] = '◄ SPOT ' + spotVal.toFixed(0);
                break;
            }
        }
        if (Object.keys(update).length > 0) {
            try {
                window.Plotly.relayout(chartEl, update);
            } catch (e) {}
        }
    }
    window.updateGexChartSpot = updateGexChartSpot;

    // ── Institutional Market Maker & GEX API ──
    function pollGexAndDealer() {
        Promise.all([
            fetch('/api/gex').then(function (r) { return r.ok ? r.json() : null; }),
            fetch('/api/dealer').then(function (r) { return r.ok ? r.json() : null; })
        ]).then(function (results) {
            var gexData = results[0];
            var dealerData = results[1];
            if (!gexData || !dealerData) return;

            var gexVal = gexData.net_gex || 0;
            var vannaVal = dealerData.net_vanna || 0;
            var charmVal = dealerData.net_charm || 0;

            var gColor = gexVal >= 0 ? 'var(--color-bullish, #10b981)' : 'var(--color-bearish, #ef4444)';
            var gexEl = document.getElementById('gex-content');
            if (gexEl) {
                gexEl.innerHTML =
                    '<div style="display:flex; flex-wrap:wrap; gap:12px;">' +
                        '<div class="metric-card glass-panel" style="flex:1; min-width:180px;">' +
                            '<div class="card-label">NET GEX</div>' +
                            '<div class="card-value num-mono" style="color:' + gColor + '">' + (gexVal / 1e7).toFixed(2) + ' Cr</div>' +
                        '</div>' +
                        '<div class="metric-card glass-panel" style="flex:1; min-width:180px;">' +
                            '<div class="card-label">ZERO GAMMA FLIP</div>' +
                            '<div class="card-value num-mono">' + (gexData.zero_gamma_level ? gexData.zero_gamma_level.toFixed(2) : '--') + '</div>' +
                        '</div>' +
                    '</div>';
            }

            var vColor = vannaVal >= 0 ? 'var(--color-bullish, #10b981)' : 'var(--color-bearish, #ef4444)';
            var cColor = charmVal >= 0 ? 'var(--color-bullish, #10b981)' : 'var(--color-bearish, #ef4444)';
            var dealerEl = document.getElementById('dealer-content');
            if (dealerEl) {
                dealerEl.innerHTML =
                    '<div style="display:flex; flex-wrap:wrap; gap:12px;">' +
                        '<div class="metric-card glass-panel" style="flex:1; min-width:180px;">' +
                            '<div class="card-label">NET VANNA EXPOSURE</div>' +
                            '<div class="card-value num-mono" style="color:' + vColor + '">' + (vannaVal / 1e7).toFixed(2) + ' Cr</div>' +
                        '</div>' +
                        '<div class="metric-card glass-panel" style="flex:1; min-width:180px;">' +
                            '<div class="card-label">NET CHARM EXPOSURE</div>' +
                            '<div class="card-value num-mono" style="color:' + cColor + '">' + (charmVal / 1e7).toFixed(2) + ' Cr</div>' +
                        '</div>' +
                    '</div>';
            }
        }).catch(function () {});
    }

    // ── Expose globally & Initialize ──
    window.switchTab = switchTab;

    document.addEventListener('DOMContentLoaded', function () {
        // Restore initial tab from URL hash or localStorage
        var initialTab = 'regime';
        var hash = window.location.hash.replace('#', '');
        if (hash && document.getElementById('tab-' + hash)) {
            initialTab = hash;
        } else {
            try {
                var savedTab = localStorage.getItem('fintel_active_tab');
                if (savedTab && document.getElementById('tab-' + savedTab)) {
                    initialTab = savedTab;
                }
            } catch (e) {}
        }
        switchTab(initialTab);
        optimizeOptionChainAndWalls();
        cleanupMMTab();

        // Bind tab buttons
        var tabBtns = document.querySelectorAll('.tab-btn');
        tabBtns.forEach(function (btn) {
            btn.addEventListener('click', function () {
                var t = btn.getAttribute('data-tab');
                if (t) switchTab(t);
            });
        });

        // Bind Ambient FX switch
        var fxInput = document.getElementById('fx-toggle-input');
        if (fxInput) {
            try {
                var fxPref = localStorage.getItem('fintel_ambient_fx');
                fxInput.checked = fxPref !== '0';
            } catch (e) {
                fxInput.checked = true;
            }
            fxInput.addEventListener('change', function () {
                if (typeof window.toggleAmbientFX === 'function') {
                    window.toggleAmbientFX(fxInput.checked);
                }
            });
        }

        // Start WebSocket and Periodic Pollers
        initWebSocket();
        refreshContent();
        setInterval(refreshContent, 15000);
        setInterval(pollGexAndDealer, 15000);
        pollGexAndDealer();
    });

    // ── Master Option Chain View Switcher ──
    window.setChainMode = function (mode) {
        var table = document.getElementById('master-chain-table');
        if (!table) return;

        document.querySelectorAll('.chain-mode-btn').forEach(function (b) {
            b.style.background = 'rgba(255,255,255,0.05)';
            b.style.color = '#94a3b8';
            b.style.border = '1px solid #333a60';
        });

        var activeBtn = document.getElementById('btn-chain-' + mode);
        if (activeBtn) {
            activeBtn.style.background = 'rgba(0,229,255,0.15)';
            activeBtn.style.color = '#00e5ff';
            activeBtn.style.border = '1px solid #00e5ff';
        }

        var showSeller = (mode === 'all' || mode === 'seller');
        var showGex = (mode === 'all' || mode === 'gex');

        table.querySelectorAll('.col-seller').forEach(function (el) {
            el.style.display = showSeller ? '' : 'none';
        });
        table.querySelectorAll('.col-gex').forEach(function (el) {
            el.style.display = showGex ? '' : 'none';
        });

        var callCols = (showSeller ? 2 : 0) + (showGex ? 2 : 0) + 3;
        var putCols = (showSeller ? 2 : 0) + (showGex ? 2 : 0) + 3;
        var thC = document.getElementById('th-calls-header');
        var thP = document.getElementById('th-puts-header');
        if (thC) thC.setAttribute('colspan', callCols);
        if (thP) thP.setAttribute('colspan', putCols);
    };

    // Auto-initialize view mode on load
    setTimeout(function() {
        if (typeof window.setChainMode === 'function') {
            window.setChainMode('all');
        }
    }, 200);

    // Expose optimizeOptionChainAndWalls globally
    window.optimizeOptionChainAndWalls = optimizeOptionChainAndWalls;
})();
