/**
 * ═══════════════════════════════════════════════════════════════════════════
 * F-INTEL QUANTITATIVE TERMINAL — CORE DASHBOARD CONTROLLER
 * ═══════════════════════════════════════════════════════════════════════════
 * Manages tab switching, WebSocket streaming, fragment live-updates,
 * and institutional API integration.
 */

(function () {
    'use strict';

    // ── Theme Management (Permanently Dark Terminal Theme) ───────────────────
    document.documentElement.setAttribute('data-theme', 'dark');
    var meta = document.getElementById('theme-color-meta');
    if (meta) meta.content = '#131722';

    document.addEventListener('DOMContentLoaded', function () {
        var canvas = document.getElementById('quant-bg-canvas');
        if (canvas) { canvas.style.display = 'none'; canvas.style.animation = 'none'; }

        var fxWrap = document.querySelector('.fx-toggle-wrap');
        if (fxWrap) fxWrap.style.display = 'none';

        var themeWrap = document.querySelector('.theme-toggle-wrap');
        if (themeWrap) themeWrap.style.display = 'none';

        var timeEl = document.getElementById('time-display');
        if (timeEl && timeEl.textContent.includes('\u23F1')) {
            timeEl.textContent = timeEl.textContent.replace(/[^\d:]/g, '').trim();
        }
    });

    var activeTab = 'regime';
    var ws = null;
    var wsReconnectTimer = null;

    // ── Tab Management ──
    function switchTab(name) {
        if (name === 'vol') {
            name = 'regime';
        }
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
            scrubInstitutionalAndNseText();
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
    var _refreshInFlight = false;
    function refreshContent() {
        if (_refreshInFlight) return;
        _refreshInFlight = true;
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
                            // 1. Interactive Card Preservation for chain tab
                            var liveOiCard = null;
                            var liveGexCard = null;
                            if (t === 'chain') {
                                liveOiCard = dest.querySelector('#oi-velocity-card');
                                liveGexCard = dest.querySelector('#gex-rebalance-card');
                                // Remove static templates from incoming frag
                                var fragOi = frag.querySelector('#oi-velocity-card');
                                if (fragOi) fragOi.remove();
                                var fragGex = frag.querySelector('#gex-rebalance-card');
                                if (fragGex) fragGex.remove();
                                // Detach live elements so they aren't destroyed
                                if (liveOiCard && liveOiCard.parentNode) liveOiCard.parentNode.removeChild(liveOiCard);
                                if (liveGexCard && liveGexCard.parentNode) liveGexCard.parentNode.removeChild(liveGexCard);
                            }

                            // 2. Interactive Card Preservation for mm tab
                            var liveGeCard = null;
                            if (t === 'mm') {
                                liveGeCard = dest.querySelector('#ge-terminal-card');
                                var fragGe = frag.querySelector('#ge-terminal-card');
                                if (fragGe) fragGe.remove();
                                if (liveGeCard && liveGeCard.parentNode) liveGeCard.parentNode.removeChild(liveGeCard);
                            }

                            // 3. Interactive Card Preservation for iv tab
                            var liveIvCard = null;
                            if (t === 'iv') {
                                liveIvCard = dest.querySelector('#iv-surface-card');
                                var fragIv = frag.querySelector('#iv-surface-card');
                                if (fragIv) fragIv.remove();
                                if (liveIvCard && liveIvCard.parentNode) liveIvCard.parentNode.removeChild(liveIvCard);
                            }

                            // 4. Preserve active capital and strike inputs
                            var oldCap = dest.querySelector('#sz-capital-input');
                            var activeCapVal = (oldCap && document.activeElement === oldCap) ? oldCap.value : null;
                            var oldThStrike = (dest.querySelector('#sel-th-strike') || {}).value;

                            dest.innerHTML = frag.innerHTML;

                            // Re-append live interactive cards (charts & listeners intact!)
                            if (liveOiCard) dest.appendChild(liveOiCard);
                            if (liveGexCard) dest.appendChild(liveGexCard);
                            if (liveGeCard) dest.appendChild(liveGeCard);
                            if (liveIvCard) dest.appendChild(liveIvCard);

                            if (activeCapVal !== null) {
                                var newCap = dest.querySelector('#sz-capital-input');
                                if (newCap) {
                                    newCap.value = activeCapVal;
                                    newCap.focus();
                                }
                            }
                            if (oldThStrike) {
                                var newThStrike = dest.querySelector('#sel-th-strike');
                                if (newThStrike) newThStrike.value = oldThStrike;
                            }

                            executeScripts(dest);
                        }
                    } catch (tabErr) {
                        console.warn('Error updating tab ' + t + ':', tabErr);
                    }
                });

                // Re-apply preserved chain mode (All / Seller / GEX)
                if (typeof window.setChainMode === 'function' && window._currentChainMode) {
                    window.setChainMode(window._currentChainMode);
                }

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

                    // Verdict pill removed
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
                    // Scrub any lingering institutional/NSE text and re-apply live GEX state
                    scrubInstitutionalAndNseText();
                    if (window._lastGexData) updateGexUI(window._lastGexData);
                    if (window._lastDealerData) updateDealerUI(window._lastDealerData);
                } catch (e) {
                    console.warn('Post-update hook error:', e);
                }
            })
            .catch(function (err) {
                console.warn('Fragment refresh error (retrying):', err);
            })
            .finally(function () {
                _refreshInFlight = false;
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
                        var numSpot = parseFloat(msg.spot);
                        if (spotEl && !isNaN(numSpot)) {
                            spotEl.innerHTML = 'SPOT: <span class="spot-val">' + numSpot.toLocaleString('en-IN', { minimumFractionDigits: 2 }) + '</span>';
                            if (window._prevSpotPrice !== undefined && window._prevSpotPrice !== null) {
                                spotEl.classList.remove('flash-up', 'flash-down');
                                void spotEl.offsetWidth; // force reflow
                                if (numSpot > window._prevSpotPrice) {
                                    spotEl.classList.add('flash-up');
                                } else if (numSpot < window._prevSpotPrice) {
                                    spotEl.classList.add('flash-down');
                                }
                            }
                            window._prevSpotPrice = numSpot;
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

                    if (msg.type === 'gex_update') {
                        if (msg.payload) updateGexUI(msg.payload);
                        else pollGexAndDealer();
                    }

                    if (msg.type === 'dealer_update') {
                        if (msg.payload) updateDealerUI(msg.payload);
                        else pollGexAndDealer();
                    }

                    if (msg.type === 'gamma_explosion_update' && window.GammaExplosionTerminal) {
                        window.GammaExplosionTerminal.handleWsMessage(msg);
                    }

                    if (msg.type === 'gex_rebalance_update' && window.GexRebalanceRadar) {
                        window.GexRebalanceRadar.handleWsMessage(msg);
                    }

                    if (msg.type === 'oi_velocity_update' && window.OiVelocityRadar) {
                        window.OiVelocityRadar.handleWsMessage(msg);
                    }

                    if (msg.type === 'iv_surface_update' && window.IvSurfaceTerminal) {
                        window.IvSurfaceTerminal.handleWsMessage(msg);
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

    // ── Client-Side Scrubbing Safeguard (Strict No "Institutional", No "NSE") ──
    function scrubInstitutionalAndNseText(root) {
        var context = root || document.body;
        if (!context) return;
        var walker = document.createTreeWalker(context, NodeFilter.SHOW_TEXT, null, false);
        var node;
        var textNodes = [];
        while ((node = walker.nextNode())) {
            textNodes.push(node);
        }
        textNodes.forEach(function (tn) {
            var val = tn.nodeValue;
            if (!val) return;
            if (val.indexOf('INSTITUTIONAL') !== -1 || val.indexOf('Institutional') !== -1 || val.indexOf('NSE') !== -1) {
                var newVal = val
                    .replace(/INSTITUTIONAL\s+GEX\s+DISTRIBUTION/gi, 'GEX DISTRIBUTION')
                    .replace(/INSTITUTIONAL\s+GEX\s*&\s*DEALER\s+POSITIONING/gi, 'GEX & DEALER POSITIONING')
                    .replace(/PER-STRIKE\s+INSTITUTIONAL\s+INVENTORY/gi, 'PER-STRIKE INVENTORY')
                    .replace(/INSTITUTIONAL\s+ROLE/gi, 'DEALER ROLE / WALL')
                    .replace(/Institutional\s+Role/gi, 'Dealer Role / Wall')
                    .replace(/INSTITUTIONAL\s+BOUNDARIES/gi, 'BOUNDARIES')
                    .replace(/INSTITUTIONAL\s+DISTRIBUTION/gi, 'GEX DISTRIBUTION')
                    .replace(/Institutional\s+Gamma\s+Exposure\s*\(GEX\)/gi, 'Gamma Exposure (GEX)')
                    .replace(/INSTITUTIONAL\s+GEX/gi, 'GEX')
                    .replace(/Institutional\s+GEX/gi, 'GEX')
                    .replace(/INSTITUTIONAL/gi, '')
                    .replace(/Institutional/gi, '')
                    .replace(/\bNSE\b/g, '')
                    .replace(/\s{2,}/g, ' ')
                    .trim();
                if (newVal !== val) {
                    tn.nodeValue = newVal;
                }
            }
        });
    }
    window.scrubInstitutionalAndNseText = scrubInstitutionalAndNseText;

    // ── Real-Time GEX & Dealer Live Engine ──
    function updateGexUI(gexData) {
        if (!gexData) return;
        window._lastGexData = gexData;
        var netCr = gexData.net_gex_cr !== undefined ? gexData.net_gex_cr : (gexData.net_gex_crores !== undefined ? gexData.net_gex_crores : (gexData.net_gex ? gexData.net_gex / 1e7 : 0));
        var isLong = netCr >= 0;
        var color = isLong ? '#00e676' : '#ff3366';
        var sign = isLong ? '+' : '';
        var regimeText = isLong ? 'LONG GAMMA (DEALER PINNING)' : 'SHORT GAMMA (TREND ACCELERATION)';
        var regimeShort = isLong ? 'Long Gamma Pin' : 'Short Gamma Trend';

        // 1. Live Net GEX Badge (Spot Movement Card Header)
        var liveBadge = document.getElementById('live-net-gex-badge');
        if (liveBadge) {
            liveBadge.textContent = 'LIVE NET GEX: ' + sign + netCr.toFixed(1) + ' Cr (' + regimeShort + ')';
            liveBadge.style.color = color;
            liveBadge.style.borderColor = isLong ? 'rgba(0,230,118,0.5)' : 'rgba(255,51,102,0.5)';
            liveBadge.style.background = isLong ? 'rgba(0,230,118,0.1)' : 'rgba(255,51,102,0.1)';
        }

        // 2. Option Chain Key Metrics Box (LIVE NET GEX)
        var chainNetEl = document.getElementById('chain-net-gex-val');
        if (chainNetEl) {
            chainNetEl.textContent = sign + netCr.toFixed(1) + ' Cr';
            chainNetEl.style.color = color;
        }
        var chainNetBox = document.getElementById('chain-net-gex-box');
        if (chainNetBox) {
            chainNetBox.style.borderLeftColor = color;
        }
        var chainSubEl = document.getElementById('chain-gex-regime-sub');
        if (chainSubEl) {
            chainSubEl.textContent = regimeShort;
        }

        // 3. Tab 5: Executive Dealer Positioning Cards
        var dealerGexVal = document.getElementById('dealer-gex-val');
        if (dealerGexVal) {
            dealerGexVal.textContent = sign + netCr.toFixed(1) + ' Cr';
            dealerGexVal.style.color = color;
        }
        var dealerRegimeBadge = document.getElementById('dealer-regime-badge');
        if (dealerRegimeBadge) {
            dealerRegimeBadge.textContent = isLong ? 'LONG GAMMA' : 'SHORT GAMMA';
            dealerRegimeBadge.style.color = color;
        }
        var dealerRegimeCard = document.getElementById('dealer-regime-card');
        if (dealerRegimeCard) {
            dealerRegimeCard.style.borderTopColor = color;
        }
        var dealerRegimeDesc = document.getElementById('dealer-regime-desc');
        if (dealerRegimeDesc) {
            dealerRegimeDesc.textContent = isLong 
                ? 'Dealers counter-trade momentum: selling rallies and buying dips, compressing realized volatility.'
                : 'Dealers amplify momentum: buying rallies and selling dips, creating rapid breakout runaway risk.';
        }

        // 4. Spot Movement Tab: Intraday Projection & Strike Dynamics
        var spotVal = gexData.spot || (window._prevSpotPrice || 0);
        var pw1 = gexData.put_wall_1 || gexData.put_wall || 0;
        var cw1 = gexData.call_wall_1 || gexData.call_wall || 0;
        var pw2 = gexData.put_wall_2 || 0;
        var cw2 = gexData.call_wall_2 || 0;
        var flip = gexData.zero_gamma_level || gexData.gex_flip_point || 0;
        var maxPain = gexData.max_pain || (pw1 && cw1 ? Math.round((pw1 + cw1)/100)*50 : 0);
        var em = gexData.expected_move_pts || (spotVal ? spotVal * 0.0075 : 150);

        var moveRegimeBadge = document.getElementById('spot-move-regime-badge');
        if (moveRegimeBadge) {
            moveRegimeBadge.textContent = regimeText;
            moveRegimeBadge.style.color = color;
            moveRegimeBadge.style.borderColor = isLong ? 'rgba(0,230,118,0.4)' : 'rgba(255,51,102,0.4)';
            moveRegimeBadge.style.background = isLong ? 'rgba(0,230,118,0.12)' : 'rgba(255,51,102,0.12)';
        }

        var movePwEl = document.getElementById('spot-move-put-wall');
        if (movePwEl && pw1) {
            var pwDist = spotVal ? (spotVal - pw1) : 0;
            movePwEl.textContent = pw1.toFixed(0) + ' (' + (pwDist >= 0 ? '-' : '+') + Math.abs(pwDist).toFixed(0) + ' pts)';
        }
        var moveCwEl = document.getElementById('spot-move-call-wall');
        if (moveCwEl && cw1) {
            var cwDist = spotVal ? (cw1 - spotVal) : 0;
            moveCwEl.textContent = cw1.toFixed(0) + ' (+' + cwDist.toFixed(0) + ' pts)';
        }
        var moveFlipEl = document.getElementById('spot-move-flip-strike');
        if (moveFlipEl && flip) {
            var fDist = spotVal ? (spotVal - flip) : 0;
            moveFlipEl.textContent = flip.toFixed(0) + ' (' + Math.abs(fDist).toFixed(0) + ' pts ' + (fDist >= 0 ? 'below' : 'above') + ')';
        }
        var moveEmEl = document.getElementById('spot-move-expected-move');
        if (moveEmEl && em && spotVal) {
            moveEmEl.textContent = '±' + em.toFixed(0) + ' pts (' + (spotVal - em).toFixed(0) + ' – ' + (spotVal + em).toFixed(0) + ')';
        }

        // Live Real-Time Intraday Projection Narrative
        var outlookEl = document.getElementById('spot-move-outlook-text');
        if (outlookEl && pw1 && cw1) {
            var cw2Txt = cw2 ? ' → CW ② (' + cw2 + ')' : '';
            var pw2Txt = pw2 ? ' → PW ② (' + pw2 + ')' : '';
            if (isLong) {
                outlookEl.innerHTML = 'Dealer hedging dampens volatility within <strong>' + pw1 + ' – ' + cw1 + '</strong> pinning corridor. ' +
                    'Spot magnetically pulled toward Max Pain (<strong>' + maxPain + '</strong>). ' +
                    'Upside breakout &gt; ' + (cw1 + 25) + ' triggers dealer short squeeze' + cw2Txt + '. ' +
                    'Downside breakdown &lt; ' + (pw1 - 25) + ' flips dealer gamma, triggering cascade selling' + pw2Txt + '.';
            } else {
                outlookEl.innerHTML = 'Dealers are <strong>SHORT GAMMA (' + netCr.toFixed(1) + ' Cr)</strong>. Market in trend-acceleration mode. ' +
                    'Upside push above ' + (cw1 - 25) + ' sparks runaway short covering' + cw2Txt + '. ' +
                    'Downside slip below ' + (pw1 + 25) + ' accelerates dealer delta dumping towards cascade targets' + pw2Txt + '. Expect wide intraday swings.';
            }
        }

        // 5. Per-Strike Option Chain GEX Table Updates
        if (Array.isArray(gexData.strikes)) {
            gexData.strikes.forEach(function (s) {
                var cEl = document.querySelector('td[data-call-gex="' + s.strike + '"]');
                if (cEl && s.call_gex_cr !== undefined) {
                    var cg = s.call_gex_cr;
                    cEl.textContent = (cg >= 0.05 ? '+' : '') + cg.toFixed(1) + ' Cr';
                    cEl.style.color = cg > 0 ? '#00e676' : (cg < 0 ? '#ff5252' : '#64748b');
                }
                var pEl = document.querySelector('td[data-put-gex="' + s.strike + '"]');
                if (pEl && s.put_gex_cr !== undefined) {
                    var pg = s.put_gex_cr;
                    pEl.textContent = (pg >= 0.05 ? '+' : '') + pg.toFixed(1) + ' Cr';
                    pEl.style.color = pg < 0 ? '#ff5252' : (pg > 0 ? '#00e676' : '#64748b');
                }
            });
        }

        // 6. Plotly Bar Chart Dynamic Restyle (ATM-centered ±550 band to prevent squishing)
        var chartEl = document.getElementById('gex-distribution-chart');
        if (chartEl && chartEl.data && chartEl.data.length > 0 && Array.isArray(gexData.strikes) && gexData.strikes.length > 0 && window.Plotly) {
            var currentSpot = spotVal || (window._prevSpotPrice || 23200);
            var lo = currentSpot - 550;
            var hi = currentSpot + 550;
            var filtered = gexData.strikes.filter(function (s) {
                return s.strike >= lo && s.strike <= hi;
            });
            if (filtered.length < 8) {
                filtered = gexData.strikes.slice(0, 25);
            }

            // Sort strikes ascending
            filtered.sort(function (a, b) { return a.strike - b.strike; });

            var yVals = [];
            var xVals = [];
            var colors = [];
            var texts = [];
            var tickVals = [];
            var tickText = [];

            filtered.forEach(function (st) {
                yVals.push(st.strike);
                tickVals.push(st.strike);
                tickText.push(String(Math.round(st.strike)));
                var val = st.gex_cr !== undefined ? st.gex_cr : (st.gex / 1e7);
                xVals.push(val);
                colors.push(val >= 0 ? '#66bb6a' : '#ff4444');
                texts.push((val >= 0 ? '+' : '') + val.toFixed(1) + ' Cr');
            });

            var minY = yVals[0] - 25;
            var maxY = yVals[yVals.length - 1] + 25;

            try {
                window.Plotly.restyle(chartEl, {
                    x: [xVals],
                    y: [yVals],
                    text: [texts],
                    'marker.color': [colors]
                }, [0]);

                var chartTitle = 'GEX DISTRIBUTION & Spot Movement Trajectory | Net GEX: ' + sign + netCr.toFixed(1) + ' Cr';
                window.Plotly.relayout(chartEl, {
                    'title.text': chartTitle,
                    'yaxis.range': [minY, maxY],
                    'yaxis.tickvals': tickVals,
                    'yaxis.ticktext': tickText,
                    'yaxis.tickmode': 'array',
                    'yaxis.autorange': false,
                    'yaxis.showticklabels': true,
                    'yaxis.title': 'Strike Price'
                });
            } catch (pErr) {
                console.warn('Plotly restyle error:', pErr);
            }
        }
    }

    function updateDealerUI(dealerData) {
        if (!dealerData) return;
        window._lastDealerData = dealerData;
        var dexCr = dealerData.net_dex_cr !== undefined ? dealerData.net_dex_cr : (dealerData.net_dex_crores !== undefined ? dealerData.net_dex_crores : 0);
        var dexVal = dealerData.net_dex || 0;
        var dexColor = dexVal < 0 ? '#00e5ff' : '#ff9100';

        var dexValEl = document.getElementById('dealer-dex-val');
        if (dexValEl) {
            dexValEl.textContent = '₹' + (dexCr >= 0 ? '+' : '') + dexCr.toFixed(1) + ' Cr';
            dexValEl.style.color = dexColor;
        }
        var dexBadgeEl = document.getElementById('dealer-dex-badge');
        if (dexBadgeEl) {
            dexBadgeEl.textContent = dexVal < 0 ? 'SHORT DELTA (LONG SPOT)' : 'LONG DELTA (SHORT SPOT)';
            dexBadgeEl.style.color = dexColor;
        }
        var dexCardEl = document.getElementById('dealer-dex-card');
        if (dexCardEl) {
            dexCardEl.style.borderTopColor = dexColor;
        }

        var hedgeValEl = document.getElementById('dealer-hedge-val');
        if (hedgeValEl && dealerData.hedge_spot_up !== undefined) {
            var hShares = dealerData.hedge_spot_up;
            var hSign = hShares >= 0 ? 'BUY ' : 'SELL ';
            hedgeValEl.textContent = hSign + Math.abs(hShares).toLocaleString() + ' shares';
            hedgeValEl.style.color = hShares >= 0 ? '#00e676' : '#ff3366';
        }
    }

    function pollGexAndDealer() {
        Promise.all([
            fetch('/api/gex?t=' + Date.now()).then(function (r) { return r.ok ? r.json() : null; }),
            fetch('/api/dealer?t=' + Date.now()).then(function (r) { return r.ok ? r.json() : null; })
        ]).then(function (results) {
            if (results[0]) updateGexUI(results[0]);
            if (results[1]) updateDealerUI(results[1]);
            scrubInstitutionalAndNseText();
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
        scrubInstitutionalAndNseText();

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

        // Start WebSocket and Periodic Pollers (High-Speed Low-Latency Mode)
        initWebSocket();
        refreshContent();
        setInterval(refreshContent, 1500);
        setInterval(pollGexAndDealer, 1500);
        pollGexAndDealer();
    });

    // ── Master Option Chain View Switcher ──
    window._currentChainMode = 'all';
    try {
        var savedMode = localStorage.getItem('fintel_chain_mode');
        if (savedMode) window._currentChainMode = savedMode;
    } catch (e) {}

    window.setChainMode = function (mode) {
        if (!mode) mode = 'all';
        window._currentChainMode = mode;
        try { localStorage.setItem('fintel_chain_mode', mode); } catch (e) {}

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
            window.setChainMode(window._currentChainMode || 'all');
        }
    }, 200);

    // Expose optimizeOptionChainAndWalls globally
    window.optimizeOptionChainAndWalls = optimizeOptionChainAndWalls;
})();
