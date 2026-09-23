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

    // ── Table Auto-Centering on ATM Strike ──
    function centerTableOnATM(containerOrId, atmRowOrId) {
        var container = typeof containerOrId === 'string' ? document.getElementById(containerOrId) : containerOrId;
        if (!container) return;
        var atmRow = typeof atmRowOrId === 'string' ? (document.getElementById(atmRowOrId.replace(/^#/, '')) || container.querySelector(atmRowOrId)) : atmRowOrId;
        if (!atmRow) atmRow = container.querySelector('tr[data-is-atm="true"]') || container.querySelector('.glow-atm');
        if (atmRow) {
            var cRect = container.getBoundingClientRect();
            var rRect = atmRow.getBoundingClientRect();
            if (cRect.height > 0) {
                var currentScroll = container.scrollTop;
                var relativeTop = rRect.top - cRect.top + currentScroll;
                var targetScroll = Math.max(0, relativeTop - (cRect.height / 2) + (rRect.height / 2));
                container.scrollTop = targetScroll;
                container.setAttribute('data-has-scrolled', 'true');
            }
        }
    }
    window.centerTableOnATM = centerTableOnATM;

    var activeTab = 'regime';
    var ws = null;
    var wsReconnectTimer = null;

    // ── Tab Management ──
    function switchTab(name) {
        if (name === 'vol') {
            name = 'regime';
        }
        if (!name) return;
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

        // First time tab is activated, ensure its initial Plotly scripts run in visible context
        initInitialTabPlotly(name);

        // Execute any pending Plotly updates for this tab
        executePendingPlotlyForTab(name);

        // Trigger Plotly chart resize for visible layout & center tables
        setTimeout(function () {
            window.dispatchEvent(new Event('resize'));
            if (name === 'theta') {
                var thCont = document.getElementById('theta-table-container');
                if (thCont && thCont.getAttribute('data-has-scrolled') !== 'true') {
                    centerTableOnATM(thCont, '#th-row-atm');
                }
                var pBleed = document.getElementById('theta-plotly-bleed');
                if (pBleed && window.Plotly && typeof window.Plotly.Plots.resize === 'function') {
                    window.Plotly.Plots.resize(pBleed);
                }
                var pGamma = document.getElementById('theta-plotly-gamma');
                if (pGamma && window.Plotly && typeof window.Plotly.Plots.resize === 'function') {
                    window.Plotly.Plots.resize(pGamma);
                }
            }
            if (name === 'mm') {
                var mmCont = document.getElementById('dealer-inventory-container');
                if (mmCont && mmCont.getAttribute('data-has-scrolled') !== 'true') {
                    centerTableOnATM(mmCont, '#mm-row-atm');
                }
                if (window.GammaExplosionTerminal && typeof window.GammaExplosionTerminal.refresh === 'function') {
                    window.GammaExplosionTerminal.refresh();
                }
            }
            if (name === 'chain') {
                var chainCont = document.getElementById('master-chain-container') ||
                    (document.getElementById('master-chain-table') ? document.getElementById('master-chain-table').parentElement : null);
                if (chainCont && chainCont.getAttribute('data-has-scrolled') !== 'true') {
                    centerTableOnATM(chainCont, '#row-atm');
                }
                optimizeOptionChainAndWalls();
                var gexChart = document.getElementById('gex-distribution-chart');
                if (gexChart && window.Plotly && typeof window.Plotly.Plots.resize === 'function') {
                    window.Plotly.Plots.resize(gexChart);
                }
                var oiChart = document.getElementById('oi-velocity-chart');
                if (oiChart && window.Plotly && typeof window.Plotly.Plots.resize === 'function') {
                    window.Plotly.Plots.resize(oiChart);
                }
                if (window.OiVelocityRadar && typeof window.OiVelocityRadar.refresh === 'function') {
                    window.OiVelocityRadar.refresh();
                }
                if (window.GexRebalanceRadar && typeof window.GexRebalanceRadar.refresh === 'function') {
                    window.GexRebalanceRadar.refresh();
                }
                if (window.IgnitionScanner && typeof window.IgnitionScanner.refresh === 'function') {
                    window.IgnitionScanner.refresh();
                }
            }
            if (name === 'iv') {
                if (window.IvSurfaceTerminal && typeof window.IvSurfaceTerminal.onTabActivated === 'function') {
                    window.IvSurfaceTerminal.onTabActivated();
                } else if (window.IvSurfaceTerminal && typeof window.IvSurfaceTerminal.returnToLive === 'function') {
                    window.IvSurfaceTerminal.returnToLive();
                }
                var smileChart = document.getElementById('iv-smile-plot');
                if (smileChart && window.Plotly && typeof window.Plotly.Plots.resize === 'function') {
                    window.Plotly.Plots.resize(smileChart);
                }
                var surfChart = document.getElementById('iv-surface-3d-plot');
                if (surfChart && window.Plotly && typeof window.Plotly.Plots.resize === 'function') {
                    window.Plotly.Plots.resize(surfChart);
                }
            }
            if (name === 'regime') {
                var volPlot = document.getElementById('vol-history-evolution-plot');
                if (volPlot && window.Plotly && typeof window.Plotly.Plots.resize === 'function') {
                    window.Plotly.Plots.resize(volPlot);
                }
            }
            if (name === 'builder') {
                if (window.StrategyBuilder && typeof window.StrategyBuilder.onTabActivated === 'function') {
                    window.StrategyBuilder.onTabActivated();
                }
                var payoffDiv = document.getElementById('sb-payoff-chart') || document.getElementById('payoff-chart');
                if (payoffDiv && window.Plotly && typeof window.Plotly.Plots.resize === 'function') {
                    window.Plotly.Plots.resize(payoffDiv);
                }
            }
            scrubInstitutionalAndNseText();
            // Fast-sync active tab immediately on switch
            refreshContent(true);
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

    // ── Option Chain & GEX Wall Enhancements (Throttled & Non-Destructive) ──
    var _optChainAnimFrame = null;
    function optimizeOptionChainAndWalls() {
        if (_optChainAnimFrame) cancelAnimationFrame(_optChainAnimFrame);
        _optChainAnimFrame = requestAnimationFrame(function () {
            _optimizeOptionChainAndWallsCore();
        });
    }

    function _optimizeOptionChainAndWallsCore() {
        var chainTab = document.getElementById('tab-chain');
        if (!chainTab) return;

        // Move #gex-rebalance-card to bottom only if not already last
        var rebCard = chainTab.querySelector('#gex-rebalance-card');
        if (rebCard && rebCard !== chainTab.lastElementChild) {
            chainTab.appendChild(rebCard);
        }

        // Clean up legacy runway if present
        var existingRunway = chainTab.querySelector('.gex-corridor-runway-bar');
        if (existingRunway) existingRunway.remove();

        var table = chainTab.querySelector('#master-chain-table') || chainTab.querySelector('table.data-table');
        if (!table) return;
        var rows = table.querySelectorAll('tbody tr');
        if (!rows || rows.length === 0) return;

        // Compute Max OI for depth bars
        var maxCeOi = 1, maxPeOi = 1;
        for (var i = 0; i < rows.length; i++) {
            var tds = rows[i].children;
            if (tds.length >= 5) {
                var ceVal = parseFloat(tds[0].textContent.replace(/,/g, '')) || 0;
                var peVal = parseFloat(tds[tds.length - 1].textContent.replace(/,/g, '')) || 0;
                if (ceVal > maxCeOi) maxCeOi = ceVal;
                if (peVal > maxPeOi) maxPeOi = peVal;
            }
        }

        for (var j = 0; j < rows.length; j++) {
            var r = rows[j];
            var tds = r.children;
            if (tds.length < 5) continue;
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
        }
    }

    // ── Row-Level DOM Diffing (Zero Layout Collapse, Zero Scroll Reset) ──
    function diffRowsInPlace(srcTbody, dstTbody) {
        if (!srcTbody || !dstTbody) return;
        var srcRows = srcTbody.children;
        var dstRows = dstTbody.children;
        if (srcRows.length === dstRows.length && srcRows.length > 0) {
            for (var i = 0; i < srcRows.length; i++) {
                var s = srcRows[i];
                var d = dstRows[i];
                if (s.className !== d.className) d.className = s.className;
                if (s.getAttribute('style') !== d.getAttribute('style')) {
                    d.setAttribute('style', s.getAttribute('style'));
                }
                if (s.getAttribute('data-strike') !== d.getAttribute('data-strike')) {
                    d.setAttribute('data-strike', s.getAttribute('data-strike'));
                }
                if (s.id !== d.id) d.id = s.id;
                if (s.innerHTML !== d.innerHTML) {
                    d.innerHTML = s.innerHTML;
                }
            }
        } else {
            dstTbody.innerHTML = srcTbody.innerHTML;
        }
    }

    // ── Resilient Plotly Engine (Zero Layout Collapse, Zero DOM Destruction) ──
    window._pendingPlotlyScripts = window._pendingPlotlyScripts || {};
    window._tabPlotlyInitialized = window._tabPlotlyInitialized || {};

    function updatePlotlyScriptFromFrag(frag, tabName) {
        if (!frag) return;
        var scripts = frag.querySelectorAll('script');
        var isTabVisible = (activeTab === tabName);

        scripts.forEach(function (s) {
            var text = s.textContent || s.innerText || '';
            if (text.indexOf('Plotly.newPlot') === -1 && text.indexOf('Plotly.react') === -1) return;

            var m = text.match(/document\.getElementById\(["']([^"']+)["']\)/);
            var divId = m ? m[1] : null;
            if (!divId) {
                var m2 = text.match(/Plotly\.(?:newPlot|react)\(\s*["']([^"']+)["']/);
                if (m2) divId = m2[1];
            }
            if (!divId) return;

            if (isTabVisible && document.getElementById(divId) && typeof window.Plotly !== 'undefined') {
                try {
                    (new Function(text))();
                    var el = document.getElementById(divId);
                    if (el && typeof window.Plotly.Plots.resize === 'function') {
                        try { window.Plotly.Plots.resize(el); } catch (e) {}
                    }
                } catch (err) {
                    console.warn('Plotly render error for ' + divId + ':', err);
                }
            } else {
                window._pendingPlotlyScripts[divId] = text;
            }
        });
    }

    function executePendingPlotlyForTab(tabName) {
        if (typeof window.Plotly === 'undefined') {
            setTimeout(function () { executePendingPlotlyForTab(tabName); }, 150);
            return;
        }

        var targets = [];
        if (tabName === 'chain') targets = ['gex-distribution-chart', 'oi-velocity-chart'];
        else if (tabName === 'regime') targets = ['vol-history-evolution-plot'];
        else if (tabName === 'theta') targets = ['theta-plotly-bleed', 'theta-plotly-gamma'];
        else if (tabName === 'iv') targets = ['iv-smile-plot', 'iv-surface-3d-plot'];
        else if (tabName === 'builder') targets = ['sb-payoff-chart', 'payoff-chart'];

        targets.forEach(function (id) {
            var script = window._pendingPlotlyScripts[id];
            if (script && document.getElementById(id)) {
                try {
                    (new Function(script))();
                    delete window._pendingPlotlyScripts[id];
                } catch (e) {
                    console.warn('Pending Plotly render error for ' + id + ':', e);
                }
            }
            var el = document.getElementById(id);
            if (el && window.Plotly && typeof window.Plotly.Plots.resize === 'function') {
                try { window.Plotly.Plots.resize(el); } catch (e) {}
            }
        });
    }

    function initInitialTabPlotly(tabName) {
        if (window._tabPlotlyInitialized[tabName]) return;
        var tab = document.getElementById('tab-' + tabName);
        if (!tab) return;
        if (typeof window.Plotly === 'undefined') {
            setTimeout(function () { initInitialTabPlotly(tabName); }, 100);
            return;
        }
        window._tabPlotlyInitialized[tabName] = true;
        var scripts = tab.querySelectorAll('script');
        scripts.forEach(function (s) {
            var text = s.textContent || s.innerText || '';
            if (text.indexOf('Plotly.newPlot') !== -1) {
                try {
                    (new Function(text))();
                } catch (e) {
                    console.warn('Initial Plotly run error for tab ' + tabName + ':', e);
                }
            }
        });
    }

    // ── Surgical In-Place Tab Updaters (Zero Jitter / Zero Height Collapse) ──
    function updateMMTabInPlace(dest, frag) {
        var existingContainer = dest.querySelector('#dealer-inventory-container');
        if (!existingContainer) {
            dest.innerHTML = frag.innerHTML;
            executeScripts(dest);
            var cont = dest.querySelector('#dealer-inventory-container');
            if (cont) {
                centerTableOnATM(cont, '#mm-row-atm');
                cont.addEventListener('scroll', function () {
                    cont.setAttribute('data-has-scrolled', 'true');
                }, { passive: true });
            }
            return;
        }

        // 1. Metric text & badge in-place updates
        var mmMetricIds = [
            'dealer-regime-badge', 'dealer-gex-val', 'dealer-regime-desc',
            'dealer-dex-badge', 'dealer-dex-val', 'dealer-dex-desc',
            'dealer-hedge-badge', 'dealer-hedge-val', 'dealer-hedge-desc',
            'dealer-pain-val', 'dealer-pain-dist',
            'dealer-upper-barrier', 'dealer-lower-barrier', 'dealer-tactical-text'
        ];
        mmMetricIds.forEach(function (id) {
            var src = frag.querySelector('#' + id);
            var dst = dest.querySelector('#' + id);
            if (src && dst) {
                if (src.innerHTML !== dst.innerHTML) dst.innerHTML = src.innerHTML;
                if (src.style && src.style.cssText && src.style.cssText !== dst.style.cssText) {
                    dst.style.cssText = src.style.cssText;
                }
            }
        });

        // 2. Card border accents
        ['dealer-regime-card', 'dealer-dex-card', 'dealer-hedge-card', 'dealer-pain-card'].forEach(function (id) {
            var src = frag.querySelector('#' + id);
            var dst = dest.querySelector('#' + id);
            if (src && dst && src.style && src.style.borderTop && dst.style.borderTop !== src.style.borderTop) {
                dst.style.borderTop = src.style.borderTop;
            }
        });

        // 3. Update table body in-place (ZERO DOM destruction, ZERO scroll bounce)
        var srcTbody = frag.querySelector('#dealer-inventory-tbody');
        var dstTbody = dest.querySelector('#dealer-inventory-tbody');
        if (srcTbody && dstTbody && existingContainer) {
            var curScroll = existingContainer.scrollTop;
            var hasScrolled = existingContainer.getAttribute('data-has-scrolled') === 'true';
            diffRowsInPlace(srcTbody, dstTbody);
            if (hasScrolled) {
                existingContainer.scrollTop = curScroll;
            } else {
                centerTableOnATM(existingContainer, '#mm-row-atm');
            }
        }
    }

    function updateThetaTabInPlace(dest, frag) {
        var existingContainer = dest.querySelector('#theta-table-container');
        if (!existingContainer) {
            dest.innerHTML = frag.innerHTML;
            executeScripts(dest);
            var cont = dest.querySelector('#theta-table-container');
            if (cont) {
                centerTableOnATM(cont, '#th-row-atm');
                cont.addEventListener('scroll', function () {
                    cont.setAttribute('data-has-scrolled', 'true');
                }, { passive: true });
            }
            return;
        }

        // 1. Cockpit cards update (Layer 1)
        var thetaCardIds = [
            'card-strad-day', 'card-strad-hour', 'card-strad-pts',
            'card-asym-verdict', 'card-asym-diff', 'card-cushion-pts'
        ];
        thetaCardIds.forEach(function (id) {
            var src = frag.querySelector('#' + id);
            var dst = dest.querySelector('#' + id);
            if (src && dst) {
                if (src.innerHTML !== dst.innerHTML) dst.innerHTML = src.innerHTML;
                if (src.style && src.style.cssText && src.style.cssText !== dst.style.cssText) {
                    dst.style.cssText = src.style.cssText;
                }
            }
        });

        var srcCardsGrid = frag.querySelector('div[style*="grid-template-columns: repeat(3, 1fr)"]');
        var dstCardsGrid = dest.querySelector('div[style*="grid-template-columns: repeat(3, 1fr)"]');
        if (srcCardsGrid && dstCardsGrid && srcCardsGrid.innerHTML !== dstCardsGrid.innerHTML) {
            dstCardsGrid.innerHTML = srcCardsGrid.innerHTML;
        }

        // 2. Plotly charts update in-place (ZERO DOM destruction, ZERO flashing)
        updatePlotlyScriptFromFrag(frag, 'theta');

        // 3. Update table header & body in-place (Layer 3) (ZERO DOM destruction, ZERO scroll bounce)
        var srcThead = frag.querySelector('#theta-decay-table thead');
        var dstThead = dest.querySelector('#theta-decay-table thead');
        if (srcThead && dstThead && srcThead.innerHTML !== dstThead.innerHTML) {
            dstThead.innerHTML = srcThead.innerHTML;
        }

        var srcTbody = frag.querySelector('#theta-decay-tbody');
        var dstTbody = dest.querySelector('#theta-decay-tbody');
        if (srcTbody && dstTbody && existingContainer) {
            var curScroll = existingContainer.scrollTop;
            var hasScrolled = existingContainer.getAttribute('data-has-scrolled') === 'true';
            diffRowsInPlace(srcTbody, dstTbody);
            if (hasScrolled) {
                existingContainer.scrollTop = curScroll;
            } else {
                centerTableOnATM(existingContainer, '#th-row-atm');
            }
        }
    }

    // ── Option Chain In-Place Updater (ZERO DOM Destruction, Zero Jump) ──
    function updateChainTabInPlace(dest, frag) {
        var existingContainer = dest.querySelector('#master-chain-container') ||
            (dest.querySelector('#master-chain-table') ? dest.querySelector('#master-chain-table').parentElement : null);

        if (!existingContainer) {
            dest.innerHTML = frag.innerHTML;
            executeScripts(dest);
            var cont = dest.querySelector('#master-chain-container') ||
                (dest.querySelector('#master-chain-table') ? dest.querySelector('#master-chain-table').parentElement : null);
            if (cont) {
                if (!cont.id) cont.id = 'master-chain-container';
                centerTableOnATM(cont, '#row-atm');
                cont.addEventListener('scroll', function () {
                    cont.setAttribute('data-has-scrolled', 'true');
                }, { passive: true });
            }
            return;
        }

        if (!existingContainer.id) existingContainer.id = 'master-chain-container';
        if (!existingContainer.getAttribute('data-scroll-listener')) {
            existingContainer.setAttribute('data-scroll-listener', 'true');
            existingContainer.addEventListener('scroll', function () {
                existingContainer.setAttribute('data-has-scrolled', 'true');
            }, { passive: true });
        }

        // 1. Lock and record exact scroll position
        var curScroll = existingContainer.scrollTop;
        var hasScrolled = existingContainer.getAttribute('data-has-scrolled') === 'true';

        // 2. Update Key Metrics & Sell Zones Card (First card in tab)
        var srcMetrics = frag.querySelector('.card');
        var dstMetrics = dest.querySelector('.card');
        if (srcMetrics && dstMetrics) {
            if (srcMetrics.innerHTML !== dstMetrics.innerHTML) {
                dstMetrics.innerHTML = srcMetrics.innerHTML;
            }
        }

        // 3. Update Master Option Chain Rows In-Place
        var srcTbody = frag.querySelector('#master-chain-tbody') || frag.querySelector('#master-chain-table tbody');
        var dstTbody = dest.querySelector('#master-chain-tbody') || dest.querySelector('#master-chain-table tbody');
        if (srcTbody && dstTbody) {
            if (!dstTbody.id) dstTbody.id = 'master-chain-tbody';
            diffRowsInPlace(srcTbody, dstTbody);
            if (hasScrolled) {
                existingContainer.scrollTop = curScroll;
            } else {
                centerTableOnATM(existingContainer, '#row-atm');
            }
        }

        // 4. Update GEX Dynamic text if present without touching chart or radars
        var srcGexDyn = frag.querySelector('div[style*="font-size:13px;font-weight:900;letter-spacing:1.5px;color:#00e5ff;"]');
        if (srcGexDyn && srcGexDyn.parentElement) {
            var srcParentCard = srcGexDyn.closest('.card');
            var dstGexDyn = dest.querySelector('div[style*="font-size:13px;font-weight:900;letter-spacing:1.5px;color:#00e5ff;"]');
            var dstParentCard = dstGexDyn ? dstGexDyn.closest('.card') : null;
            if (srcParentCard && dstParentCard) {
                var srcPanels = srcParentCard.querySelectorAll('.metric-box, .metric-val');
                var dstPanels = dstParentCard.querySelectorAll('.metric-box, .metric-val');
                if (srcPanels.length === dstPanels.length) {
                    for (var p = 0; p < srcPanels.length; p++) {
                        if (srcPanels[p].innerHTML !== dstPanels[p].innerHTML) {
                            dstPanels[p].innerHTML = srcPanels[p].innerHTML;
                        }
                    }
                }
            }
        }

        // 5. Update GEX Distribution Bar Chart in-place
        updatePlotlyScriptFromFrag(frag, 'chain');

        // 6. Ensure scroll listener
        if (!existingContainer.hasAttribute('data-bound')) {
            existingContainer.setAttribute('data-bound', 'true');
            existingContainer.addEventListener('scroll', function () {
                existingContainer.setAttribute('data-has-scrolled', 'true');
            }, { passive: true });
        }
    }

    // ── Regime Tab In-Place Updater (Zero Layout Shift) ──
    function updateRegimeTabInPlace(dest, frag) {
        if (!dest || !frag) return;
        if (!dest.firstElementChild) {
            dest.innerHTML = frag.innerHTML;
            executeScripts(dest);
            return;
        }
        var srcCards = frag.querySelectorAll('.card');
        var dstCards = dest.querySelectorAll('.card');
        if (srcCards.length === dstCards.length && srcCards.length > 0) {
            for (var i = 0; i < srcCards.length; i++) {
                // Card containing Plotly chart: update header/text only, do not wipe out canvas!
                if (dstCards[i].querySelector('#vol-history-evolution-plot')) {
                    var srcHeader = srcCards[i].firstElementChild;
                    var dstHeader = dstCards[i].firstElementChild;
                    if (srcHeader && dstHeader && srcHeader.innerHTML !== dstHeader.innerHTML) {
                        dstHeader.innerHTML = srcHeader.innerHTML;
                    }
                    continue;
                }
                if (srcCards[i].innerHTML !== dstCards[i].innerHTML) {
                    dstCards[i].innerHTML = srcCards[i].innerHTML;
                }
            }
        } else {
            dest.innerHTML = frag.innerHTML;
            executeScripts(dest);
        }

        // Update Plotly historical chart in-place
        updatePlotlyScriptFromFrag(frag, 'regime');
    }

    // ── Live Fragment Polling (High-Speed Tab Targeted & Gzip Accelerated) ──
    var _refreshInFlight = false;
    var _lastEtag = '';
    var _refreshTick = 0;

    function refreshContent(forceActiveOnly) {
        if (_refreshInFlight) return;
        _refreshInFlight = true;

        _refreshTick++;
        // Request activeTab for 95%+ payload reduction;
        // every 10th refresh sync full fragment across all background tabs.
        var targetTab = (forceActiveOnly || _refreshTick % 10 !== 0) ? (activeTab || 'chain') : '';
        var url = '/fragment?t=' + Date.now();
        if (targetTab) {
            url += '&tab=' + encodeURIComponent(targetTab);
        }

        var headers = {};
        if (_lastEtag) {
            headers['If-None-Match'] = _lastEtag;
        }

        fetch(url, { headers: headers })
            .then(function (res) {
                if (res.status === 304) {
                    return null; // Zero changes, skip DOM work
                }
                if (!res.ok) throw new Error('Fragment fetch failed with status ' + res.status);
                var etag = res.headers.get('ETag');
                if (etag) _lastEtag = etag;
                return res.text();
            })
            .then(function (html) {
                if (!html || html.length < 50) return;
                var parser = new DOMParser();
                var doc = parser.parseFromString(html, 'text/html');

                var tabNames = ['chain', 'regime', 'mm', 'theta', 'iv', 'builder'];
                tabNames.forEach(function (t) {
                    try {
                        var frag = doc.getElementById('frag-' + t);
                        var dest = document.getElementById('tab-' + t);
                        if (!frag || !dest) return;

                        // Surgical in-place tab updates
                        if (t === 'chain') {
                            updateChainTabInPlace(dest, frag);
                            return;
                        }
                        if (t === 'regime') {
                            updateRegimeTabInPlace(dest, frag);
                            return;
                        }
                        if (t === 'mm') {
                            updateMMTabInPlace(dest, frag);
                            return;
                        }
                        if (t === 'theta') {
                            updateThetaTabInPlace(dest, frag);
                            return;
                        }

                        // Preserved fallback for other tabs
                        if (t === 'iv' || t === 'builder') {
                            // Client-side terminals: do NOT overwrite mounted interactive DOM with static shell
                            if (!dest.firstElementChild || dest.children.length === 0) {
                                dest.innerHTML = frag.innerHTML;
                                executeScripts(dest);
                            }
                            return;
                        }

                        dest.innerHTML = frag.innerHTML;
                        executeScripts(dest);
                    } catch (tabErr) {
                        console.warn('Error updating tab ' + t + ':', tabErr);
                    }
                });

                // Re-apply preserved chain mode
                if (typeof window.setChainMode === 'function' && window._currentChainMode) {
                    window.setChainMode(window._currentChainMode);
                }

                // Visual optimizations
                optimizeOptionChainAndWalls();

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

                    var timeEl = document.getElementById('time-display');
                    if (timeEl && timeVal) {
                        var isWs = (ws && ws.readyState === WebSocket.OPEN);
                        timeEl.innerHTML = '&#128339; ' + timeVal + (isWs ? ' &nbsp;|&nbsp; Live WS' : ' &nbsp;|&nbsp; Polled');
                    }
                }

                var badge = document.getElementById('ws-status-badge');
                if (badge) {
                    badge.style.opacity = '0.35';
                    setTimeout(function () { badge.style.opacity = '1'; }, 300);
                }

                try {
                    if (typeof window.applyThetaFilters === 'function') window.applyThetaFilters();
                    if (typeof window.updateThetaSim === 'function') window.updateThetaSim();
                    if (activeTab === 'mm' && window.GammaExplosionTerminal && typeof window.GammaExplosionTerminal.refresh === 'function') {
                        window.GammaExplosionTerminal.refresh();
                    }
                    if (window.GexRebalanceRadar && typeof window.GexRebalanceRadar.refresh === 'function') {
                        window.GexRebalanceRadar.refresh();
                    }
                    if (window.IgnitionScanner && typeof window.IgnitionScanner.refresh === 'function') {
                        window.IgnitionScanner.refresh();
                    }
                    scrubInstitutionalAndNseText();
                    if (window._lastGexData) updateGexUI(window._lastGexData);
                    if (window._lastDealerData) updateDealerUI(window._lastDealerData);
                } catch (e) {
                    console.warn('Post-update hook error:', e);
                }
            })
            .catch(function (err) {
                console.warn('Fragment refresh error:', err);
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

                    if (msg.type === 'ignition_update' && window.IgnitionScanner) {
                        window.IgnitionScanner.handleWsMessage(msg);
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
        // If WebSocket is active and has populated data, skip HTTP poll to save bandwidth
        if (ws && ws.readyState === WebSocket.OPEN && window._lastGexData && window._lastDealerData) {
            return;
        }
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
        setTimeout(function () {
            centerTableOnATM('dealer-inventory-container', '#mm-row-atm');
            centerTableOnATM('theta-table-container', '#th-row-atm');
        }, 300);
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
        setInterval(refreshContent, 2500);
        setInterval(pollGexAndDealer, 5000);
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
