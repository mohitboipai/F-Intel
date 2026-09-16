/**
 * ═══════════════════════════════════════════════════════════════════════════
 * F-INTEL ADVANCED ECONOMETRIC VOLATILITY & STRANGLE SIZING TERMINAL
 * ═══════════════════════════════════════════════════════════════════════════
 * Dynamic capital allocation, Corsi HAR forward volatility, Realized
 * semi-variance skew, and Bipower jump radar.
 */

(function () {
    'use strict';

    var currentCapital = 2000000;

    function setStrangleCapital(amt) {
        currentCapital = parseFloat(amt) || 2000000;
        var inp = document.getElementById('sz-capital-input');
        if (inp) inp.value = currentCapital;

        // Update button active state
        var btns = document.querySelectorAll('.sz-quick-btn');
        btns.forEach(function (btn) {
            btn.classList.remove('active');
            btn.style.background = 'rgba(255,255,255,0.06)';
            btn.style.borderColor = '#2a2a4a';
            btn.style.color = '#cbd5e1';
        });

        var ev = (typeof event !== 'undefined' ? event : (window.event || null));
        if (ev && ev.target && ev.target.classList && ev.target.classList.contains('sz-quick-btn')) {
            ev.target.classList.add('active');
            ev.target.style.background = 'rgba(0,240,255,0.15)';
            ev.target.style.borderColor = '#00f0ff66';
            ev.target.style.color = '#00f0ff';
        }

        updateStrangleSizing(currentCapital);
    }

    function updateStrangleSizing(capitalVal) {
        var cap = parseFloat(capitalVal) || currentCapital;
        currentCapital = cap;

        fetch('/api/strangle/sizing?capital=' + cap + '&t=' + Date.now())
            .then(function (r) {
                if (!r.ok) throw new Error('Sizing fetch failed');
                return r.json();
            })
            .then(function (res) {
                if (!res.ok || !res.sizing) return;
                var sz = res.sizing;

                // Verdict Pill
                var vPill = document.getElementById('sz-verdict-pill');
                if (vPill) {
                    vPill.textContent = sz.verdict_badge;
                    vPill.style.color = sz.verdict_color;
                    vPill.style.borderColor = sz.verdict_color;
                    vPill.style.background = sz.verdict_color + '22';
                }

                // Cockpit card border
                var card = document.getElementById('card-strangle-cockpit');
                if (card && sz.verdict_color) {
                    card.style.borderLeftColor = sz.verdict_color;
                }

                // Optimal Lots
                var optLotsEl = document.getElementById('sz-opt-lots-val');
                if (optLotsEl) {
                    optLotsEl.textContent = sz.optimal_lots + ' Lots';
                    optLotsEl.style.color = sz.verdict_color;
                }

                // Base Lots sub
                var baseLotsEl = document.getElementById('sz-base-lots-sub');
                if (baseLotsEl) {
                    baseLotsEl.textContent = 'Base Capacity: ' + sz.base_lots + ' Lots (' + Math.round(sz.deployment_pct) + '%)';
                }

                // Leg Allocation
                var ceLbl = document.getElementById('sz-ce-lots-lbl');
                if (ceLbl) ceLbl.textContent = sz.ce_lots + ' CE';
                var peLbl = document.getElementById('sz-pe-lots-lbl');
                if (peLbl) peLbl.textContent = sz.pe_lots + ' PE';

                // Skew Bias sub
                var skewEl = document.getElementById('sz-skew-bias-sub');
                if (skewEl && sz.skew) {
                    skewEl.textContent = sz.skew.bias.replace(/_/g, ' ');
                }

                // Gap stress test
                var gapRiskEl = document.getElementById('sz-gap-risk-val');
                if (gapRiskEl && sz.stress_test) {
                    gapRiskEl.textContent = '₹' + Math.round(sz.stress_test.estimated_loss_inr).toLocaleString('en-IN');
                }
                var gapPctEl = document.getElementById('sz-gap-pct-sub');
                if (gapPctEl && sz.stress_test) {
                    gapPctEl.textContent = sz.stress_test.risk_pct_of_capital.toFixed(2) + '% of capital (' + (sz.stress_test.is_within_budget ? 'Safe' : 'Watch') + ')';
                    gapPctEl.style.color = sz.stress_test.is_within_budget ? '#10b981' : '#f59e0b';
                }
            })
            .catch(function () {});
    }

    // Expose functions globally
    window.setStrangleCapital = setStrangleCapital;
    window.updateStrangleSizing = updateStrangleSizing;

    // Hook into tab changes and initial load
    document.addEventListener('DOMContentLoaded', function () {
        var inp = document.getElementById('sz-capital-input');
        if (inp) {
            inp.addEventListener('input', function () {
                updateStrangleSizing(this.value);
            });
        }
    });
})();
