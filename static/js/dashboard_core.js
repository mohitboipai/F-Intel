/**
 * ═══════════════════════════════════════════════════════════════════════════
 * F-INTEL QUANTITATIVE TERMINAL — CORE DASHBOARD CONTROLLER
 * ═══════════════════════════════════════════════════════════════════════════
 * Manages tab switching, WebSocket streaming, fragment live-updates,
 * and institutional API integration.
 */

(function () {
    'use strict';

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
            if (name === 'mm' && window.GammaExplosionTerminal && typeof window.GammaExplosionTerminal.refresh === 'function') {
                window.GammaExplosionTerminal.refresh();
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

    // ── Live Fragment Polling (HTTP Hot-Reload) ──
    function refreshContent() {
        fetch('/fragment?t=' + Date.now())
            .then(function (res) {
                if (!res.ok) throw new Error('Fragment fetch failed');
                return res.text();
            })
            .then(function (html) {
                var parser = new DOMParser();
                var doc = parser.parseFromString(html, 'text/html');

                var tabNames = ['regime', 'iv', 'vol', 'chain', 'theta', 'prob', 'mm'];
                tabNames.forEach(function (t) {
                    var frag = doc.getElementById('frag-' + t);
                    var dest = document.getElementById('tab-' + t);
                    if (frag && dest) {
                        dest.innerHTML = frag.innerHTML;
                        executeScripts(dest);
                    }
                });

                // Update Spot Pill & Verdict
                var spotFrag = doc.getElementById('frag-spot');
                if (spotFrag) {
                    var spotVal = spotFrag.getAttribute('data-spot');
                    var timeVal = spotFrag.getAttribute('data-time');

                    var spotEl = document.getElementById('spot-display');
                    if (spotEl && spotVal) {
                        spotEl.innerHTML = 'SPOT: <span class="spot-val">' + Number(spotVal).toLocaleString('en-IN', { minimumFractionDigits: 2 }) + '</span>';
                    }

                    var timeEl = document.getElementById('time-display');
                    if (timeEl && timeVal && (!ws || ws.readyState !== WebSocket.OPEN)) {
                        timeEl.innerHTML = '&#128339; ' + timeVal;
                    }

                    var verdTransfer = doc.getElementById('frag-verdict-transfer');
                    var verdDest = document.getElementById('top-verdict-pill');
                    if (verdTransfer && verdDest) {
                        verdDest.innerHTML = verdTransfer.innerHTML;
                    }
                }

                // Re-apply filters and simulation after DOM injection
                if (typeof window.applyThetaFilters === 'function') {
                    window.applyThetaFilters();
                }
                if (typeof window.updateThetaSim === 'function') {
                    window.updateThetaSim();
                }
                if (window.GammaExplosionTerminal && typeof window.GammaExplosionTerminal.refresh === 'function') {
                    window.GammaExplosionTerminal.refresh();
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
                    if (msg.type === 'spot_tick' && msg.spot) {
                        var spotEl = document.getElementById('spot-display');
                        if (spotEl) {
                            spotEl.innerHTML = 'SPOT: <span class="spot-val">' + Number(msg.spot).toLocaleString('en-IN', { minimumFractionDigits: 2 }) + '</span>';
                        }
                    }
                    if (msg.server_time) {
                        var timeEl = document.getElementById('time-display');
                        if (timeEl) {
                            timeEl.innerHTML = '&#128339; ' + msg.server_time + ' &nbsp;|&nbsp; Live';
                        }
                    }
                    if (msg.type === 'gamma_explosion_update' && window.GammaExplosionTerminal) {
                        window.GammaExplosionTerminal.handleWsMessage(msg);
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
})();
