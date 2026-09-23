/**
 * ═══════════════════════════════════════════════════════════════════════════
 * F-INTEL QUANTITATIVE TERMINAL — GREEKS CONTROLLER
 * ═══════════════════════════════════════════════════════════════════════════
 * Streamlined 3-Layer Executive Greeks Desk:
 * - Layer 1: Net ATM Straddle Bleed & Put-Call Decay Asymmetry Cockpit
 * - Layer 2: Clean Two-Curve Visualizer (Theta Bleed & Gamma Convexity)
 * - Layer 3: Streamlined 8-Column Per-Strike Matrix centered on ATM
 */

(function () {
    'use strict';

    // ── Table ATM Locking & Smooth Centering ──
    function lockTableToATM() {
        var container = document.getElementById('theta-table-container');
        if (!container) return;
        var atmRow = document.getElementById('th-row-atm') || container.querySelector('tr[data-is-atm="true"]') || container.querySelector('.glow-atm');
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

    // ── Backward-compatible Safe APIs ──
    function applyThetaFilters() {
        // Table filtering is handled server-side / CSS
    }

    function updateThetaSim() {
        // Calculations are performed server-side with calibrated NIFTY 65 qty lot size
    }

    function selectSimStrike(strike) {
        // Strike highlighting if selected
        var tbl = document.getElementById('theta-decay-table');
        if (tbl) {
            tbl.querySelectorAll('tbody tr').forEach(function (r) {
                var s = parseFloat(r.getAttribute('data-strike')) || parseFloat((r.querySelector('td') || {}).textContent);
                if (s === strike) {
                    r.style.outline = '2px solid #00e5ff';
                } else {
                    r.style.outline = '';
                }
            });
        }
    }

    function refreshThetaDecay() {
        if (typeof window.refreshContent === 'function') {
            window.refreshContent();
        }
    }

    // ── Expose Global APIs for Dashboard bindings ──
    window.lockTableToATM = lockTableToATM;
    window.centerThetaATM = lockTableToATM;
    window.selectSimStrike = selectSimStrike;
    window.applyThetaFilters = applyThetaFilters;
    window.updateThetaSim = updateThetaSim;
    window.refreshThetaDecay = refreshThetaDecay;
    window.changeThetaRange = function () {};
    window.setThetaUnit = function () {};
    window.toggleThetaModel = function () {};
    window.toggleAutoLockATM = function () {};
    window.updateExecutiveDecayCards = function () {};
    window.updateGreekCauseAndEffect = function () {};
    window.updatePlotlyOverlayForecast = function () {};
    window.applySpotShock = function () {};
    window.applyTimeShock = function () {};
    window.applyIvShock = function () {};
    window.resetGreekShocks = function () {};

    // ── Initialize on DOM ready ──
    document.addEventListener('DOMContentLoaded', function () {
        setTimeout(lockTableToATM, 300);
    });
})();
