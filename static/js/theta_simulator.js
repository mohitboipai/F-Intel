/**
 * ═══════════════════════════════════════════════════════════════════════════
 * F-INTEL QUANTITATIVE TERMINAL — THETA DECAY & MULTI-GREEK SIMULATOR
 * ═══════════════════════════════════════════════════════════════════════════
 * Research-grounded Taylor Greek attribution engine (Gatheral 2006, Bouchaud-Sornette 1994)
 */

(function () {
    'use strict';

    var currentThetaRange = '10';
    var currentThetaFocus = 'all';
    var currentThetaUnit = 'INR';
    var autoLockATM = true;
    var activeSimStrike = null;

    var simMin = 0;
    var simDays = 0;
    var simSpotShock = 0;
    var simIVShock = 0;

    function lockTableToATM() {
        var container = document.getElementById('theta-table-container');
        if (!container) return;
        var atmRow = document.getElementById('th-row-atm') || container.querySelector('tr[data-is-atm="true"]');
        if (atmRow) {
            var rowTop = atmRow.offsetTop;
            var targetScroll = Math.max(0, rowTop - (container.clientHeight / 2) + (atmRow.clientHeight / 2));
            container.scrollTop = targetScroll;
        }
    }

    function toggleAutoLockATM(btn) {
        autoLockATM = !autoLockATM;
        try { localStorage.setItem('th_autolock', autoLockATM ? 'true' : 'false'); } catch (e) {}
        updateAutoLockBtn(btn);
        if (autoLockATM) {
            lockTableToATM();
        }
    }

    function updateAutoLockBtn(btn) {
        var b = btn || document.getElementById('btn-th-autolock');
        if (!b) return;
        if (autoLockATM) {
            b.innerHTML = '&#128274; Lock ATM: ON';
            b.style.background = 'var(--color-bullish-dim, rgba(16,185,129,0.18))';
            b.style.color = 'var(--color-bullish, #10b981)';
            b.style.borderColor = 'var(--color-bullish, #10b981)';
        } else {
            b.innerHTML = '&#128275; Lock ATM: OFF';
            b.style.background = 'rgba(255,255,255,0.05)';
            b.style.color = 'var(--text-muted, #888888)';
            b.style.borderColor = 'var(--border-subtle, #444444)';
        }
    }

    function setThetaTableFocus(focus) {
        currentThetaFocus = focus;
        ['all', 'seller', 'buyer'].forEach(function (f) {
            var el = document.getElementById('btn-focus-' + f);
            if (el) el.classList.toggle('active', f === focus);
        });
        applyThetaFilters();
    }

    function setThetaUnit(unit) {
        currentThetaUnit = unit;
        var btnInr = document.getElementById('btn-unit-inr');
        var btnPts = document.getElementById('btn-unit-pts');
        if (btnInr && btnPts) {
            if (unit === 'INR') {
                btnInr.style.background = 'var(--accent-cyan, #00f0ff)';
                btnInr.style.color = 'var(--text-inverse, #030712)';
                btnPts.style.background = 'transparent';
                btnPts.style.color = 'var(--text-muted, #94a3b8)';
            } else {
                btnPts.style.background = 'var(--accent-cyan, #00f0ff)';
                btnPts.style.color = 'var(--text-inverse, #030712)';
                btnInr.style.background = 'transparent';
                btnInr.style.color = 'var(--text-muted, #94a3b8)';
            }
        }
        updateExecutiveDecayCards(activeSimStrike);
    }

    function setThetaRange(val, btn) {
        currentThetaRange = val;
        var p = btn && btn.parentElement;
        if (p) {
            var btns = p.querySelectorAll('button');
            btns.forEach(function (b) {
                b.style.background = 'transparent';
                b.style.color = 'var(--text-muted, #888)';
                b.style.borderColor = 'var(--border-subtle, #333)';
            });
            btn.style.background = 'var(--accent-cyan-dim, rgba(0, 240, 255, 0.15))';
            btn.style.color = 'var(--accent-cyan, #00f0ff)';
            btn.style.borderColor = 'var(--accent-cyan-glow, #00f0ff)';
        }
        applyThetaFilters();
    }

    function updateExecutiveDecayCards(strike) {
        var container = document.getElementById('theta-table-container');
        if (!container) return;
        var row = null;
        if (strike) {
            row = container.querySelector('tr[data-strike="' + strike + '"]');
        }
        if (!row) {
            row = document.getElementById('th-row-atm') || container.querySelector('tr[data-is-atm="true"]') || container.querySelector('tbody tr');
        }
        if (!row) return;

        var sVal = parseFloat(row.getAttribute('data-strike')) || 24000;
        var cLtp = parseFloat(row.getAttribute('data-ce-ltp')) || 0;
        var pLtp = parseFloat(row.getAttribute('data-pe-ltp')) || 0;
        var stLtp = parseFloat(row.getAttribute('data-strad-ltp')) || (cLtp + pLtp);

        var cThLot = parseFloat(row.getAttribute('data-ce-theta')) || 0;
        var pThLot = parseFloat(row.getAttribute('data-pe-theta')) || 0;
        var stThLot = parseFloat(row.getAttribute('data-strad-theta')) || (cThLot + pThLot);

        var extPts = parseFloat(row.getAttribute('data-ext-pts')) || 0;
        var extInr = parseFloat(row.getAttribute('data-ext-inr')) || 0;
        var gamma = parseFloat(row.getAttribute('data-strad-gam')) || 0.003;
        var lot = 65;

        var elLot = document.getElementById('attr-lot-size');
        if (elLot) {
            var lotParsed = parseInt(elLot.textContent.replace(/[^0-9]/g, ''), 10);
            if (lotParsed > 0) lot = lotParsed;
        }

        // Update strike labels
        var lblCeS = document.getElementById('card-ce-strike');
        if (lblCeS) lblCeS.textContent = sVal.toLocaleString('en-IN');
        var lblPeS = document.getElementById('card-pe-strike');
        if (lblPeS) lblPeS.textContent = sVal.toLocaleString('en-IN');
        var lblStS = document.getElementById('card-strad-strike');
        if (lblStS) lblStS.textContent = sVal.toLocaleString('en-IN');

        // LTPs
        var elCeP = document.getElementById('card-ce-price');
        if (elCeP) elCeP.textContent = 'LTP: ₹' + cLtp.toFixed(1);
        var elPeP = document.getElementById('card-pe-price');
        if (elPeP) elPeP.textContent = 'LTP: ₹' + pLtp.toFixed(1);
        var elStP = document.getElementById('card-strad-price');
        if (elStP) elStP.textContent = 'LTP: ₹' + stLtp.toFixed(1);

        var isINR = (currentThetaUnit === 'INR');

        // CALL Card
        var elCeDay = document.getElementById('card-ce-day');
        if (elCeDay) elCeDay.textContent = isINR ? '-₹' + Math.round(Math.abs(cThLot)).toLocaleString('en-IN') : '-' + Math.abs(cThLot / lot).toFixed(1) + ' pts';
        var elCeHr = document.getElementById('card-ce-hour');
        if (elCeHr) elCeHr.textContent = isINR ? '-₹' + Math.round(Math.abs(cThLot / 6.25)).toLocaleString('en-IN') : '-' + Math.abs(cThLot / (lot * 6.25)).toFixed(2) + ' pts';
        var elCeMin = document.getElementById('card-ce-min');
        if (elCeMin) elCeMin.textContent = isINR ? '-₹' + Math.abs(cThLot / 375.0).toFixed(2) : '-' + Math.abs(cThLot / (lot * 375.0)).toFixed(3) + ' pts';

        // PUT Card
        var elPeDay = document.getElementById('card-pe-day');
        if (elPeDay) elPeDay.textContent = isINR ? '-₹' + Math.round(Math.abs(pThLot)).toLocaleString('en-IN') : '-' + Math.abs(pThLot / lot).toFixed(1) + ' pts';
        var elPeHr = document.getElementById('card-pe-hour');
        if (elPeHr) elPeHr.textContent = isINR ? '-₹' + Math.round(Math.abs(pThLot / 6.25)).toLocaleString('en-IN') : '-' + Math.abs(pThLot / (lot * 6.25)).toFixed(2) + ' pts';
        var elPeMin = document.getElementById('card-pe-min');
        if (elPeMin) elPeMin.textContent = isINR ? '-₹' + Math.abs(pThLot / 375.0).toFixed(2) : '-' + Math.abs(pThLot / (lot * 375.0)).toFixed(3) + ' pts';

        // STRADDLE Card
        var elStDay = document.getElementById('card-strad-day');
        if (elStDay) elStDay.textContent = isINR ? '-₹' + Math.round(Math.abs(stThLot)).toLocaleString('en-IN') : '-' + Math.abs(stThLot / lot).toFixed(1) + ' pts';
        var elStHr = document.getElementById('card-strad-hour');
        if (elStHr) elStHr.textContent = isINR ? '-₹' + Math.round(Math.abs(stThLot / 6.25)).toLocaleString('en-IN') : '-' + Math.abs(stThLot / (lot * 6.25)).toFixed(2) + ' pts';
        var elStMin = document.getElementById('card-strad-min');
        if (elStMin) elStMin.textContent = isINR ? '-₹' + Math.abs(stThLot / 375.0).toFixed(2) : '-' + Math.abs(stThLot / (lot * 375.0)).toFixed(3) + ' pts';

        // Renormalized Alpha Metric
        var em = 250.0;
        var emEl = document.getElementById('attr-em-pts');
        if (emEl) {
            var m = emEl.textContent.match(/[0-9,.]+/);
            if (m) em = parseFloat(m[0].replace(/,/g, '')) || em;
        }
        var gammaHazard = 0.5 * gamma * (em * em) * lot;
        var renormAlpha = Math.abs(stThLot) / Math.max(gammaHazard, 1.0);
        var elAlpha = document.getElementById('card-strad-alpha');
        if (elAlpha) {
            var aCol = renormAlpha >= 1.0 ? 'var(--color-bullish, #10b981)' : (renormAlpha >= 0.7 ? 'var(--color-warning, #f59e0b)' : 'var(--color-bearish, #ef4444)');
            var aLabel = renormAlpha >= 1.0 ? 'Alpha Edge' : (renormAlpha >= 0.7 ? 'Buffer Zone' : 'Gamma Drag');
            elAlpha.textContent = renormAlpha.toFixed(2) + ' (' + aLabel + ')';
            elAlpha.style.color = aCol;
        }

        // ── CALL vs PUT THETA ASYMMETRY COMPARATOR ──
        var cAbs = Math.abs(cThLot);
        var pAbs = Math.abs(pThLot);
        var totTh = cAbs + pAbs;
        var cPct = totTh > 0 ? (cAbs / totTh * 100) : 50;
        var pPct = totTh > 0 ? (pAbs / totTh * 100) : 50;
        var ratio = pAbs > 0 ? (cAbs / pAbs) : 1.0;

        var leader = 'BALANCED';
        var verdictText = 'THETA DECAY IS SYMMETRICAL';
        var leaderColor = 'var(--color-warning, #ffd54f)';
        var diffInr = Math.abs(pAbs - cAbs);
        var diffPts = diffInr / lot;
        var diffPct = 0;
        var insightText = 'Time bleed is evenly matched between Calls and Puts (neutral decay bias).';

        if (pAbs > cAbs * 1.02) {
            leader = 'PUTS';
            diffPct = cAbs > 0 ? ((diffInr / cAbs) * 100) : 0;
            verdictText = 'PUT THETA IS HIGHER (+' + diffPct.toFixed(1) + '% vs Calls)';
            leaderColor = '#ff7043';
            insightText = 'Put buyers bleeding faster (-₹' + Math.round(diffInr).toLocaleString('en-IN') + '/d more). Put writing offers higher time-decay harvest than Call writing.';
        } else if (cAbs > pAbs * 1.02) {
            leader = 'CALLS';
            diffPct = pAbs > 0 ? ((diffInr / pAbs) * 100) : 0;
            verdictText = 'CALL THETA IS HIGHER (+' + diffPct.toFixed(1) + '% vs Puts)';
            leaderColor = '#38bdf8';
            insightText = 'Call buyers bleeding faster (-₹' + Math.round(diffInr).toLocaleString('en-IN') + '/d more). Call writing offers higher time-decay harvest than Put writing.';
        }

        var elCompCard = document.getElementById('card-theta-comparison');
        if (elCompCard) elCompCard.style.borderLeftColor = leaderColor;

        var elVerdict = document.getElementById('th-verdict-title');
        if (elVerdict) {
            elVerdict.textContent = verdictText;
            var parentV = document.getElementById('th-asymmetry-verdict');
            if (parentV) parentV.style.color = leaderColor;
        }

        var elBadge = document.getElementById('th-leader-badge');
        if (elBadge) {
            elBadge.textContent = leader + ' BLEED DOMINANCE';
            elBadge.style.color = leaderColor;
            elBadge.style.background = leader === 'PUTS' ? 'rgba(255,112,67,0.15)' : (leader === 'CALLS' ? 'rgba(56,189,248,0.15)' : 'rgba(255,213,79,0.15)');
            elBadge.style.borderColor = leaderColor;
        }

        var elRatio = document.getElementById('th-ratio-display');
        if (elRatio) elRatio.textContent = ratio.toFixed(2) + 'x';

        var elGaugeCe = document.getElementById('th-gauge-ce');
        if (elGaugeCe) elGaugeCe.style.width = cPct.toFixed(1) + '%';
        var elGaugePe = document.getElementById('th-gauge-pe');
        if (elGaugePe) elGaugePe.style.width = pPct.toFixed(1) + '%';

        var elCeLbl = document.getElementById('th-ce-gauge-lbl');
        if (elCeLbl) elCeLbl.textContent = cPct.toFixed(1) + '%';
        var elPeLbl = document.getElementById('th-pe-gauge-lbl');
        if (elPeLbl) elPeLbl.textContent = pPct.toFixed(1) + '%';

        var elCeVal = document.getElementById('th-ce-val-lbl');
        if (elCeVal) elCeVal.textContent = isINR ? Math.round(cAbs).toLocaleString('en-IN') : (cAbs / lot).toFixed(1) + ' pts';
        var elPeVal = document.getElementById('th-pe-val-lbl');
        if (elPeVal) elPeVal.textContent = isINR ? Math.round(pAbs).toLocaleString('en-IN') : (pAbs / lot).toFixed(1) + ' pts';

        var elStrikeLbl = document.getElementById('th-gauge-center-strike');
        if (elStrikeLbl) elStrikeLbl.textContent = 'TARGET STRIKE ' + sVal.toLocaleString('en-IN');

        var elDiff = document.getElementById('th-diff-detail');
        if (elDiff) elDiff.textContent = isINR ? 'Δ ₹' + Math.round(diffInr).toLocaleString('en-IN') + ' / lot (' + diffPts.toFixed(1) + ' pts)' : 'Δ ' + diffPts.toFixed(1) + ' pts/share';
        var elDiffSub = document.getElementById('th-diff-sub');
        if (elDiffSub) {
            elDiffSub.textContent = leader === 'BALANCED' ? 'Even decay rate' : leader + ' bleeding faster';
            elDiffSub.style.color = leaderColor;
        }

        var elInsight = document.getElementById('th-insight-detail');
        if (elInsight) elInsight.textContent = insightText;

        // Update tags on Call and Put cards
        var elCeTag = document.getElementById('card-ce-leader-tag');
        if (elCeTag) {
            if (leader === 'CALLS') {
                elCeTag.textContent = '🔥 HIGHER (+' + diffPct.toFixed(0) + '%)';
                elCeTag.style.background = 'rgba(56,189,248,0.2)';
                elCeTag.style.color = '#38bdf8';
                elCeTag.style.border = '1px solid #38bdf8';
            } else {
                elCeTag.textContent = 'LOWER DECAY';
                elCeTag.style.background = 'rgba(255,255,255,0.06)';
                elCeTag.style.color = 'var(--text-muted, #94a3b8)';
                elCeTag.style.border = 'none';
            }
        }
        var elPeTag = document.getElementById('card-pe-leader-tag');
        if (elPeTag) {
            if (leader === 'PUTS') {
                elPeTag.textContent = '🔥 HIGHER (+' + diffPct.toFixed(0) + '%)';
                elPeTag.style.background = 'rgba(255,112,67,0.2)';
                elPeTag.style.color = '#ff7043';
                elPeTag.style.border = '1px solid #ff7043';
            } else {
                elPeTag.textContent = 'LOWER DECAY';
                elPeTag.style.background = 'rgba(255,255,255,0.06)';
                elPeTag.style.color = 'var(--text-muted, #94a3b8)';
                elPeTag.style.border = 'none';
            }
        }
    }

    function selectSimStrike(strike) {
        activeSimStrike = strike;
        var sel = document.getElementById('sel-th-strike');
        if (sel && sel.value !== String(strike)) {
            sel.value = String(strike);
        }
        var container = document.getElementById('theta-table-container');
        if (container) {
            var allRows = container.querySelectorAll('tbody tr');
            allRows.forEach(function (r) {
                var s = parseFloat(r.getAttribute('data-strike'));
                if (s === strike) {
                    r.style.boxShadow = 'inset 0 0 0 2px var(--accent-cyan, #00f0ff)';
                } else {
                    r.style.boxShadow = '';
                }
            });
        }
        updateExecutiveDecayCards(strike);
        updateThetaSim();
    }

    function updateThetaSim() {
        var elDays = document.getElementById('sim-days') || document.getElementById('slider-sim-days');
        var elSpot = document.getElementById('sim-spot') || document.getElementById('slider-sim-spot');
        var elIv = document.getElementById('sim-iv') || document.getElementById('slider-sim-iv');

        var curDays = elDays ? parseFloat(elDays.value) : 1.0;
        var curSpotShock = elSpot ? parseFloat(elSpot.value) : 0.0;
        var curIvShock = elIv ? parseFloat(elIv.value) : 0.0;

        var lblDays = document.getElementById('lbl-sim-days');
        if (lblDays) lblDays.textContent = '+' + curDays.toFixed(1) + ' days';

        var spotBase = 24000;
        var spotElDisplay = document.getElementById('spot-display');
        if (spotElDisplay) {
            var m = spotElDisplay.textContent.match(/[0-9,.]+/);
            if (m) spotBase = parseFloat(m[0].replace(/,/g, '')) || spotBase;
        }

        var em = 250.0;
        var emEl = document.getElementById('attr-em-pts');
        if (emEl) {
            var mEm = emEl.textContent.match(/[0-9,.]+/);
            if (mEm) em = parseFloat(mEm[0].replace(/,/g, '')) || em;
        }
        var zScore = (curSpotShock / Math.max(em, 1.0)).toFixed(1);

        var lblSpot = document.getElementById('lbl-sim-spot');
        if (lblSpot) lblSpot.textContent = (curSpotShock >= 0 ? '+' : '') + curSpotShock + ' pts (' + (zScore >= 0 ? '+' : '') + zScore + 'σ)';

        var lblIv = document.getElementById('lbl-sim-iv');
        if (lblIv) lblIv.textContent = (curIvShock >= 0 ? '+' : '') + curIvShock.toFixed(1) + '%';

        // Read Straddle Greeks
        var cardStrad = document.getElementById('card-strad-theta');
        var thPerDay = -16.5;
        if (cardStrad) {
            var thVal = parseFloat(cardStrad.getAttribute('data-theta'));
            if (!isNaN(thVal)) thPerDay = thVal;
        }

        var delta = 0.02;
        var gamma = 0.00045;
        var vega = 28.5;
        var vanna = -0.015;
        var charm = 0.008;
        var lot = 65;

        var elLot = document.getElementById('attr-lot-size');
        if (elLot) {
            var lotParsed = parseInt(elLot.textContent.replace(/[^0-9]/g, ''), 10);
            if (lotParsed > 0) lot = lotParsed;
        }

        var elMetrics = document.getElementById('straddle-greeks-holder');
        if (elMetrics) {
            delta = parseFloat(elMetrics.getAttribute('data-delta')) || delta;
            gamma = parseFloat(elMetrics.getAttribute('data-gamma')) || gamma;
            vega = parseFloat(elMetrics.getAttribute('data-vega')) || vega;
            vanna = parseFloat(elMetrics.getAttribute('data-vanna')) || vanna;
            charm = parseFloat(elMetrics.getAttribute('data-charm')) || charm;
        }

        // Multi-Factor Taylor Attribution
        var dt = curDays;
        var dS = curSpotShock;
        var dVol = curIvShock;

        var stTh = -Math.abs(thPerDay) * dt;
        var stDelta = delta * dS;
        var stGamma = 0.5 * gamma * (dS * dS);
        var stVega = vega * dVol;
        var stCross = (vanna * dS * dVol) + (charm * dS * dt);

        var stTotPerShare = stTh + stDelta + stGamma + stVega + stCross;
        var stPnlSeller = -stTotPerShare * lot;

        // PnL display
        var pnlEl = document.getElementById('sim-strad-pnl');
        if (pnlEl) {
            var sign = stPnlSeller >= 0 ? '+' : '-';
            var col = stPnlSeller >= 0 ? 'var(--color-bullish, #10b981)' : 'var(--color-bearish, #ef4444)';
            pnlEl.textContent = sign + '₹' + Math.abs(Math.round(stPnlSeller)).toLocaleString('en-IN');
            pnlEl.style.color = col;
        }

        var pnlPtsEl = document.getElementById('sim-strad-pts');
        if (pnlPtsEl) {
            var signPts = -stTotPerShare >= 0 ? '+' : '';
            pnlPtsEl.textContent = '(' + signPts + (-stTotPerShare).toFixed(1) + ' pts)';
        }

        // Update breakdown chips
        var thChip = document.getElementById('sim-chip-theta');
        if (thChip) thChip.textContent = '+₹' + Math.round(Math.abs(stTh) * lot).toLocaleString('en-IN');

        var deltaChip = document.getElementById('sim-chip-delta');
        if (deltaChip) {
            var sD = -stDelta * lot;
            deltaChip.textContent = (sD >= 0 ? '+₹' : '-₹') + Math.abs(Math.round(sD)).toLocaleString('en-IN');
            deltaChip.style.color = sD >= 0 ? 'var(--color-bullish, #10b981)' : 'var(--color-bearish, #ef4444)';
        }

        var gammaChip = document.getElementById('sim-chip-gamma');
        if (gammaChip) {
            var sG = -stGamma * lot;
            gammaChip.textContent = (sG >= 0 ? '+₹' : '-₹') + Math.abs(Math.round(sG)).toLocaleString('en-IN');
            gammaChip.style.color = sG >= 0 ? 'var(--color-bullish, #10b981)' : 'var(--color-bearish, #ef4444)';
        }

        var vegaChip = document.getElementById('sim-chip-vega');
        if (vegaChip) {
            var sV = -stVega * lot;
            vegaChip.textContent = (sV >= 0 ? '+₹' : '-₹') + Math.abs(Math.round(sV)).toLocaleString('en-IN');
            vegaChip.style.color = sV >= 0 ? 'var(--color-bullish, #10b981)' : 'var(--color-bearish, #ef4444)';
        }

        // Highlight matching row in Scenarios Table
        var curZ = dS / Math.max(em, 1.0);
        var scenTable = document.getElementById('scenarios-table');
        if (scenTable) {
            var scenRows = scenTable.querySelectorAll('tbody tr');
            var closestRow = null;
            var minZDiff = 999;
            scenRows.forEach(function (sr) {
                var zVal = parseFloat(sr.getAttribute('data-z'));
                if (!isNaN(zVal)) {
                    var diff = Math.abs(zVal - curZ);
                    if (diff < minZDiff) {
                        minZDiff = diff;
                        closestRow = sr;
                    }
                }
                sr.style.background = '';
            });
            if (closestRow && minZDiff <= 0.3) {
                closestRow.style.background = 'rgba(0, 240, 255, 0.15)';
            }
        }
    }

    function applyThetaFilters() {
        var rangePct = parseFloat(currentThetaRange) || 10;
        var spotEl = document.getElementById('spot-display');
        var spot = 24000;
        if (spotEl) {
            var m = spotEl.textContent.match(/[0-9,.]+/);
            if (m) spot = parseFloat(m[0].replace(/,/g, '')) || spot;
        }

        var tbl = document.getElementById('theta-decay-table');
        if (tbl) {
            var allTrs = tbl.querySelectorAll('tr');
            allTrs.forEach(function (r) {
                var cells = r.cells;
                if (!cells || cells.length < 5) return;
                var strkCell = cells[4];
                if (!strkCell) return;
                var strk = parseFloat(strkCell.textContent.replace(/[^0-9.]/g, ''));
                if (isNaN(strk)) return;
                if (rangePct >= 99) {
                    r.style.display = '';
                } else {
                    var diff = Math.abs(strk - spot) / spot * 100;
                    r.style.display = diff <= rangePct ? '' : 'none';
                }
            });
        }
    }

    // Expose functions globally for HTML event bindings
    window.setThetaRange = setThetaRange;
    window.setThetaTableFocus = setThetaTableFocus;
    window.setThetaUnit = setThetaUnit;
    window.toggleAutoLockATM = toggleAutoLockATM;
    window.lockTableToATM = lockTableToATM;
    window.selectSimStrike = selectSimStrike;
    window.updateExecutiveDecayCards = updateExecutiveDecayCards;
    window.updateThetaSim = updateThetaSim;
    window.applyThetaFilters = applyThetaFilters;

    document.addEventListener('DOMContentLoaded', function () {
        var elDays = document.getElementById('sim-days') || document.getElementById('slider-sim-days');
        var elSpot = document.getElementById('sim-spot') || document.getElementById('slider-sim-spot');
        var elIv = document.getElementById('sim-iv') || document.getElementById('slider-sim-iv');

        if (elDays) elDays.addEventListener('input', updateThetaSim);
        if (elSpot) elSpot.addEventListener('input', updateThetaSim);
        if (elIv) elIv.addEventListener('input', updateThetaSim);
    });
})();
