/**
 * ═══════════════════════════════════════════════════════════════════════════
 * F-INTEL QUANTITATIVE TERMINAL — GREEKS & CAUSE-EFFECT ENGINE
 * ═══════════════════════════════════════════════════════════════════════════
 * Pure Quantitative Greeks Engine:
 * 1. Dynamic Greek Cause & Effect (Spot Shift, Minutes-to-Days Time, IV Shock)
 * 2. Real vs Forecast Greeks Side-by-Side Comparative Matrix
 * 3. Synchronized Multi-Greek Desk Visualizer (4 Quadrants with Forecast Overlay)
 * 4. Dedicated Bottom Strike Cockpit (Side-by-Side Call, Straddle & Put)
 * 5. Per-Strike Greeks Matrix (Zero Position Advice, Pure Analysis)
 */

(function () {
    'use strict';

    var currentThetaRange = '10';
    var currentThetaUnit = 'INR';
    var autoLockATM = true;
    var activeSimStrike = null;

    // Shock states for Greek Cause & Effect
    var simSpotShock = 0;       // points (ΔS)
    var simTimeShock = 0;       // days (Δt)
    var simTimeMins = 0;        // minutes
    var simIvShock = 0;         // % IV (Δσ)

    // ── High-Precision Standard Normal Distributions ──
    function normalCDF(x) {
        var b1 =  0.319381530;
        var b2 = -0.356563782;
        var b3 =  1.781477937;
        var b4 = -1.821255978;
        var b5 =  1.330274429;
        var p  =  0.2316419;
        var c2 =  0.3989422804014327;

        if (x >= 0.0) {
            var t = 1.0 / (1.0 + p * x);
            return (1.0 - c2 * Math.exp(-x * x / 2.0) * t *
                (t * (t * (t * (t * b5 + b4) + b3) + b2) + b1));
        } else {
            var t = 1.0 / (1.0 - p * x);
            return (c2 * Math.exp(-x * x / 2.0) * t *
                (t * (t * (t * (t * b5 + b4) + b3) + b2) + b1));
        }
    }

    function normalPDF(x) {
        return 0.3989422804014327 * Math.exp(-0.5 * x * x);
    }

    function calcBSM(S, K, T, sigma, r) {
        r = (r !== undefined) ? r : 0.07;
        S = Math.max(S, 1.0);
        K = Math.max(K, 1.0);
        T = Math.max(T, 0.0001); // in years
        sigma = Math.max(sigma, 0.01); // decimal

        var sqrtT = Math.sqrt(T);
        var d1 = (Math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrtT);
        var d2 = d1 - sigma * sqrtT;

        var nd1 = normalCDF(d1);
        var nd2 = normalCDF(d2);
        var n_neg_d2 = normalCDF(-d2);
        var npd1 = normalPDF(d1);
        var expRt = Math.exp(-r * T);

        var callPrice = S * nd1 - K * expRt * nd2;
        var putPrice = K * expRt * n_neg_d2 - S * normalCDF(-d1);

        var callDelta = nd1;
        var putDelta = nd1 - 1.0;
        var gamma = npd1 / (S * sigma * sqrtT);

        // Theta per calendar day (divide annual theta by 365)
        var thetaCommon = -(S * npd1 * sigma) / (2.0 * sqrtT);
        var callTheta = (thetaCommon - r * K * expRt * nd2) / 365.0;
        var putTheta = (thetaCommon + r * K * expRt * n_neg_d2) / 365.0;

        // Vega per 1% IV shift = S * sqrt(T) * pdf(d1) * 0.01
        var vega = S * sqrtT * npd1 * 0.01;

        return {
            callPrice: Math.max(0.05, callPrice),
            putPrice: Math.max(0.05, putPrice),
            callDelta: callDelta,
            putDelta: putDelta,
            gamma: gamma,
            callTheta: callTheta,
            putTheta: putTheta,
            vega: vega
        };
    }

    // ── Table ATM Locking & Scrolling ──
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
            b.style.background = 'rgba(16,185,129,0.18)';
            b.style.color = '#10b981';
            b.style.borderColor = '#10b981';
        } else {
            b.innerHTML = '&#128275; Lock ATM: OFF';
            b.style.background = 'rgba(255,255,255,0.05)';
            b.style.color = '#888888';
            b.style.borderColor = '#444444';
        }
    }

    function setThetaUnit(unit) {
        currentThetaUnit = unit;
        var btnInr = document.getElementById('btn-unit-inr');
        var btnPts = document.getElementById('btn-unit-pts');
        if (btnInr && btnPts) {
            if (unit === 'INR') {
                btnInr.style.background = '#0284c7';
                btnInr.style.color = '#ffffff';
                btnPts.style.background = 'transparent';
                btnPts.style.color = '#94a3b8';
            } else {
                btnPts.style.background = '#0284c7';
                btnPts.style.color = '#ffffff';
                btnInr.style.background = 'transparent';
                btnInr.style.color = '#94a3b8';
            }
        }
        updateExecutiveDecayCards(activeSimStrike);
        updateGreekCauseAndEffect();
    }

    function toggleThetaModel(modelType) {
        var m = (modelType || 'bsm').toLowerCase();
        ['bsm', 'heston', 'both'].forEach(function (mod) {
            var el = document.getElementById('btn-th-' + mod);
            if (el) el.classList.toggle('active', mod === m);
        });
    }

    function changeThetaRange(val) {
        currentThetaRange = String(val);
        applyThetaFilters();
    }

    function refreshThetaDecay(btn) {
        if (btn) {
            var origText = btn.innerHTML;
            btn.innerHTML = '&#8635; Loading...';
            btn.disabled = true;
            setTimeout(function () {
                btn.innerHTML = origText;
                btn.disabled = false;
                updateExecutiveDecayCards(activeSimStrike);
                updateGreekCauseAndEffect();
                updatePlotlyOverlayForecast();
            }, 400);
        }
    }

    // ── Helper to resolve target row in table ──
    function getStrikeRow(strike) {
        var container = document.getElementById('theta-table-container');
        if (!container) return null;
        var row = null;
        if (strike !== null && strike !== undefined) {
            row = container.querySelector('tr[data-strike="' + strike + '"]');
        }
        if (!row) {
            row = document.getElementById('th-row-atm') ||
                  container.querySelector('tr[data-is-atm="true"]') ||
                  container.querySelector('tbody tr');
        }
        return row;
    }

    function getLotSize() {
        var lot = 65;
        var elLot = document.getElementById('attr-lot-size');
        if (elLot) {
            var lotParsed = parseInt(elLot.textContent.replace(/[^0-9]/g, ''), 10);
            if (lotParsed > 0) lot = lotParsed;
        }
        return lot;
    }

    function getSpotPrice() {
        var spot = 24000;
        var spotEl = document.getElementById('spot-display');
        if (spotEl) {
            var m = spotEl.textContent.match(/[0-9,.]+/);
            if (m) spot = parseFloat(m[0].replace(/,/g, '')) || spot;
        }
        return spot;
    }

    function getDTE() {
        var dte = 1.0;
        var elDte = document.getElementById('attr-dte') || document.querySelector('[data-dte]');
        if (elDte) {
            var dteVal = parseFloat(elDte.getAttribute('data-dte') || elDte.textContent.replace(/[^0-9.]/g, ''));
            if (!isNaN(dteVal) && dteVal > 0) dte = dteVal;
        }
        return dte;
    }

    // ── 1. DEDICATED BOTTOM STRIKE COCKPIT (Side-by-Side Call & Put with Straddle) ──
    function updateExecutiveDecayCards(strike) {
        var row = getStrikeRow(strike);
        if (!row) return;

        var sVal = parseFloat(row.getAttribute('data-strike')) || 24000;
        var lot = getLotSize();
        var isINR = (currentThetaUnit === 'INR');

        // Target Strike Header
        var deckLbl = document.getElementById('deck-target-strike-lbl');
        if (deckLbl) deckLbl.textContent = sVal.toLocaleString('en-IN');

        var cLtp = parseFloat(row.getAttribute('data-ce-ltp')) || 0;
        var pLtp = parseFloat(row.getAttribute('data-pe-ltp')) || 0;
        var stLtp = parseFloat(row.getAttribute('data-strad-ltp')) || (cLtp + pLtp);

        var cThLot = parseFloat(row.getAttribute('data-ce-theta')) || 0;
        var pThLot = parseFloat(row.getAttribute('data-pe-theta')) || 0;
        var stThLot = parseFloat(row.getAttribute('data-strad-theta')) || (cThLot + pThLot);

        var cDelta = parseFloat(row.getAttribute('data-ce-delta')) || 0.50;
        var pDelta = parseFloat(row.getAttribute('data-pe-delta')) || -0.50;
        var netDelta = parseFloat(row.getAttribute('data-net-delta')) || (cDelta + pDelta);

        var cGam = parseFloat(row.getAttribute('data-ce-gamma')) || 0.0015;
        var pGam = parseFloat(row.getAttribute('data-pe-gamma')) || 0.0015;
        var stGam = parseFloat(row.getAttribute('data-strad-gam')) || (cGam + pGam);

        var cVg = parseFloat(row.getAttribute('data-ce-vega')) || 600;
        var pVg = parseFloat(row.getAttribute('data-pe-vega')) || 600;
        var stVg = parseFloat(row.getAttribute('data-strad-vega')) || (cVg + pVg);

        var cushion = parseFloat(row.getAttribute('data-cushion')) || 60.0;
        var yieldPct = parseFloat(row.getAttribute('data-yield')) || 0.0;

        // CALL Card (Left)
        var elCeP = document.getElementById('card-ce-price');
        if (elCeP) elCeP.textContent = 'LTP: ₹' + cLtp.toFixed(1);
        var elCeDay = document.getElementById('card-ce-day');
        if (elCeDay) elCeDay.textContent = isINR ? '-₹' + Math.round(Math.abs(cThLot)).toLocaleString('en-IN') : '-' + Math.abs(cThLot / lot).toFixed(1) + ' pts';
        var elCeHr = document.getElementById('card-ce-hour');
        if (elCeHr) elCeHr.textContent = isINR ? '-₹' + Math.round(Math.abs(cThLot / 6.25)).toLocaleString('en-IN') : '-' + Math.abs(cThLot / (lot * 6.25)).toFixed(2) + ' pts';
        var elCeDel = document.getElementById('card-ce-delta');
        if (elCeDel) elCeDel.textContent = (cDelta >= 0 ? '+' : '') + cDelta.toFixed(2);
        var elCeGam = document.getElementById('card-ce-gamma');
        if (elCeGam) elCeGam.textContent = cGam.toFixed(5);
        var elCeVg = document.getElementById('card-ce-vega');
        if (elCeVg) elCeVg.textContent = isINR ? '₹' + Math.round(cVg).toLocaleString('en-IN') : (cVg / lot).toFixed(2) + ' pts';

        // STRADDLE Card (Center Together)
        var elStP = document.getElementById('card-strad-price');
        if (elStP) elStP.textContent = 'LTP: ₹' + stLtp.toFixed(1);
        var elStDay = document.getElementById('card-strad-day');
        if (elStDay) elStDay.textContent = isINR ? '-₹' + Math.round(Math.abs(stThLot)).toLocaleString('en-IN') : '-' + Math.abs(stThLot / lot).toFixed(1) + ' pts';
        var elStBe = document.getElementById('card-strad-be');
        if (elStBe) elStBe.textContent = '±' + Math.round(cushion) + ' pts';
        var elStNetDel = document.getElementById('card-strad-net-delta');
        if (elStNetDel) elStNetDel.textContent = (netDelta >= 0 ? '+' : '') + netDelta.toFixed(2);
        var elStGam = document.getElementById('card-strad-gamma');
        if (elStGam) elStGam.textContent = stGam.toFixed(5);
        var elStVg = document.getElementById('card-strad-vega');
        if (elStVg) elStVg.textContent = isINR ? '₹' + Math.round(stVg).toLocaleString('en-IN') : (stVg / lot).toFixed(2) + ' pts';
        var elStYield = document.getElementById('card-strad-yield');
        if (elStYield) elStYield.textContent = yieldPct.toFixed(1) + '%/d';

        // PUT Card (Right)
        var elPeP = document.getElementById('card-pe-price');
        if (elPeP) elPeP.textContent = 'LTP: ₹' + pLtp.toFixed(1);
        var elPeDay = document.getElementById('card-pe-day');
        if (elPeDay) elPeDay.textContent = isINR ? '-₹' + Math.round(Math.abs(pThLot)).toLocaleString('en-IN') : '-' + Math.abs(pThLot / lot).toFixed(1) + ' pts';
        var elPeHr = document.getElementById('card-pe-hour');
        if (elPeHr) elPeHr.textContent = isINR ? '-₹' + Math.round(Math.abs(pThLot / 6.25)).toLocaleString('en-IN') : '-' + Math.abs(pThLot / (lot * 6.25)).toFixed(2) + ' pts';
        var elPeDel = document.getElementById('card-pe-delta');
        if (elPeDel) elPeDel.textContent = pDelta.toFixed(2);
        var elPeGam = document.getElementById('card-pe-gamma');
        if (elPeGam) elPeGam.textContent = pGam.toFixed(5);
        var elPeVg = document.getElementById('card-pe-vega');
        if (elPeVg) elPeVg.textContent = isINR ? '₹' + Math.round(pVg).toLocaleString('en-IN') : (pVg / lot).toFixed(2) + ' pts';
    }

    // ── 2. DYNAMIC GREEK CAUSE & EFFECT ENGINE & SHOCK HANDLERS ──
    function applySpotShock(val) {
        simSpotShock = parseFloat(val) || 0;
        var sl = document.getElementById('slider-shock-spot');
        if (sl && parseFloat(sl.value) !== simSpotShock) sl.value = simSpotShock;
        updateShockButtonStates();
        updateGreekCauseAndEffect();
        updatePlotlyOverlayForecast();
    }

    function applyTimeShock(days, mins) {
        simTimeShock = Math.max(0, parseFloat(days) || 0);
        simTimeMins = parseInt(mins, 10) || 0;
        var sl = document.getElementById('slider-shock-time');
        if (sl && Math.abs(parseFloat(sl.value) - simTimeShock) > 0.02) sl.value = simTimeShock;
        updateShockButtonStates();
        updateGreekCauseAndEffect();
        updatePlotlyOverlayForecast();
    }

    function applyIvShock(val) {
        simIvShock = parseFloat(val) || 0;
        var sl = document.getElementById('slider-shock-iv');
        if (sl && parseFloat(sl.value) !== simIvShock) sl.value = simIvShock;
        updateShockButtonStates();
        updateGreekCauseAndEffect();
        updatePlotlyOverlayForecast();
    }

    function resetGreekShocks() {
        simSpotShock = 0;
        simTimeShock = 0;
        simTimeMins = 0;
        simIvShock = 0;

        var slSpot = document.getElementById('slider-shock-spot');
        if (slSpot) slSpot.value = 0;
        var slTime = document.getElementById('slider-shock-time');
        if (slTime) slTime.value = 0;
        var slIv = document.getElementById('slider-shock-iv');
        if (slIv) slIv.value = 0;

        updateShockButtonStates();
        updateGreekCauseAndEffect();
        updatePlotlyOverlayForecast();
    }

    function updateShockButtonStates() {
        // Spot shock label
        var elSpot = document.getElementById('val-shock-spot');
        if (elSpot) elSpot.textContent = (simSpotShock >= 0 ? '+' : '') + simSpotShock + ' pts';

        // Time shock label
        var elTime = document.getElementById('val-shock-time');
        if (elTime) {
            if (simTimeMins > 0) {
                var hrStr = simTimeMins >= 60 ? (simTimeMins / 60).toFixed(1) + 'h' : simTimeMins + 'm';
                elTime.textContent = '+' + simTimeShock.toFixed(2) + ' d (' + hrStr + ')';
            } else {
                elTime.textContent = '+' + simTimeShock.toFixed(1) + ' d';
            }
        }

        // IV shock label
        var elIv = document.getElementById('val-shock-iv');
        if (elIv) elIv.textContent = (simIvShock >= 0 ? '+' : '') + simIvShock.toFixed(1) + '%';

        // Quick buttons active states
        var card = document.getElementById('card-greek-cause-effect');
        if (!card) return;

        var spotBtns = card.querySelectorAll('button[onclick^="applySpotShock"]');
        spotBtns.forEach(function (btn) {
            var m = btn.getAttribute('onclick').match(/applySpotShock\(([-0-9.]+)\)/);
            if (m && parseFloat(m[1]) === simSpotShock) {
                btn.classList.add('active');
            } else {
                btn.classList.remove('active');
            }
        });

        var timeBtns = card.querySelectorAll('button[onclick^="applyTimeShock"]');
        timeBtns.forEach(function (btn) {
            var m = btn.getAttribute('onclick').match(/applyTimeShock\(([-0-9.]+)/);
            if (m && Math.abs(parseFloat(m[1]) - simTimeShock) < 0.03) {
                btn.classList.add('active-time');
            } else {
                btn.classList.remove('active-time');
            }
        });

        var ivBtns = card.querySelectorAll('button[onclick^="applyIvShock"]');
        ivBtns.forEach(function (btn) {
            var m = btn.getAttribute('onclick').match(/applyIvShock\(([-0-9.]+)\)/);
            if (m && parseFloat(m[1]) === simIvShock) {
                btn.classList.add('active-iv');
            } else {
                btn.classList.remove('active-iv');
            }
        });
    }

    function updateGreekCauseAndEffect() {
        var row = getStrikeRow(activeSimStrike);
        if (!row) return;

        var lot = getLotSize();
        var spot0 = getSpotPrice();
        var dte0 = getDTE();
        var strike = parseFloat(row.getAttribute('data-strike')) || spot0;

        // Baseline Real Values from Row
        var cLtp0 = parseFloat(row.getAttribute('data-ce-ltp')) || 0.1;
        var pLtp0 = parseFloat(row.getAttribute('data-pe-ltp')) || 0.1;
        var stLtp0 = cLtp0 + pLtp0;

        var cDel0 = parseFloat(row.getAttribute('data-ce-delta')) || 0.50;
        var pDel0 = parseFloat(row.getAttribute('data-pe-delta')) || -0.50;
        var stDel0 = cDel0 + pDel0;

        var cGam0 = parseFloat(row.getAttribute('data-ce-gamma')) || 0.0015;
        var pGam0 = parseFloat(row.getAttribute('data-pe-gamma')) || 0.0015;
        var stGam0 = cGam0 + pGam0;

        var cThInr0 = parseFloat(row.getAttribute('data-ce-theta')) || -1800.0;
        var pThInr0 = parseFloat(row.getAttribute('data-pe-theta')) || -1950.0;
        var stThInr0 = cThInr0 + pThInr0;

        var cVgInr0 = parseFloat(row.getAttribute('data-ce-vega')) || 600.0;
        var pVgInr0 = parseFloat(row.getAttribute('data-pe-vega')) || 600.0;
        var stVgInr0 = cVgInr0 + pVgInr0;

        var cIv0 = parseFloat(row.getAttribute('data-ce-iv')) || 14.0;
        var pIv0 = parseFloat(row.getAttribute('data-pe-iv')) || 14.0;

        // Perturbed Market State:
        var S_sim = spot0 + simSpotShock;
        var dte_sim = Math.max(0.001, dte0 - simTimeShock);
        var T_sim = dte_sim / 365.0;
        var cSigma_sim = Math.max(0.01, (cIv0 + simIvShock) / 100.0);
        var pSigma_sim = Math.max(0.01, (pIv0 + simIvShock) / 100.0);

        // Exact Analytical BSM Simulation under Perturbed State:
        var bsmCe = calcBSM(S_sim, strike, T_sim, cSigma_sim, 0.07);
        var bsmPe = calcBSM(S_sim, strike, T_sim, pSigma_sim, 0.07);

        var cLtp_sim = bsmCe.callPrice;
        var pLtp_sim = bsmPe.putPrice;
        var stLtp_sim = cLtp_sim + pLtp_sim;

        var cDel_sim = bsmCe.callDelta;
        var pDel_sim = bsmPe.putDelta;
        var stDel_sim = cDel_sim + pDel_sim;

        var cGam_sim = bsmCe.gamma;
        var pGam_sim = bsmPe.gamma;
        var stGam_sim = cGam_sim + pGam_sim;

        var cThInr_sim = bsmCe.callTheta * lot;
        var pThInr_sim = bsmPe.putTheta * lot;
        var stThInr_sim = cThInr_sim + pThInr_sim;

        var cVgInr_sim = bsmCe.vega * lot;
        var pVgInr_sim = bsmPe.vega * lot;
        var stVgInr_sim = cVgInr_sim + pVgInr_sim;

        // Differences: Forecast - Real
        var cLtpDiff = cLtp_sim - cLtp0;
        var pLtpDiff = pLtp_sim - pLtp0;
        var stLtpDiff = stLtp_sim - stLtp0;

        var cDelDiff = cDel_sim - cDel0;
        var pDelDiff = pDel_sim - pDel0;
        var stDelDiff = stDel_sim - stDel0;

        var cGamDiff = cGam_sim - cGam0;
        var pGamDiff = pGam_sim - pGam0;
        var stGamDiff = stGam_sim - stGam0;

        var cThDiff = cThInr_sim - cThInr0;
        var pThDiff = pThInr_sim - pThInr0;
        var stThDiff = stThInr_sim - stThInr0;

        var cVgDiff = cVgInr_sim - cVgInr0;
        var pVgDiff = pVgInr_sim - pVgInr0;
        var stVgDiff = stVgInr_sim - stVgInr0;

        // Update Real vs Forecast Comparative Table
        // Prices
        setElText('sim-ce-price-real', '₹' + cLtp0.toFixed(1));
        setElText('sim-ce-price-fc', '₹' + cLtp_sim.toFixed(1));
        setElShift('sim-ce-price-diff', cLtpDiff, (cLtpDiff / Math.max(cLtp0, 0.1) * 100), '₹', false);

        setElText('sim-strad-price-real', '₹' + stLtp0.toFixed(1));
        setElText('sim-strad-price-fc', '₹' + stLtp_sim.toFixed(1));
        setElShift('sim-strad-price-diff', stLtpDiff, (stLtpDiff / Math.max(stLtp0, 0.1) * 100), '₹', false);

        setElText('sim-pe-price-real', '₹' + pLtp0.toFixed(1));
        setElText('sim-pe-price-fc', '₹' + pLtp_sim.toFixed(1));
        setElShift('sim-pe-price-diff', pLtpDiff, (pLtpDiff / Math.max(pLtp0, 0.1) * 100), '₹', false);

        // Deltas
        setElText('sim-ce-del-real', (cDel0 >= 0 ? '+' : '') + cDel0.toFixed(2));
        setElText('sim-ce-del-fc', (cDel_sim >= 0 ? '+' : '') + cDel_sim.toFixed(2));
        setElShift('sim-ce-del-diff', cDelDiff, (cDelDiff / Math.max(Math.abs(cDel0), 0.01) * 100), '', false, 2);

        setElText('sim-strad-del-real', (stDel0 >= 0 ? '+' : '') + stDel0.toFixed(2));
        setElText('sim-strad-del-fc', (stDel_sim >= 0 ? '+' : '') + stDel_sim.toFixed(2));
        setElShift('sim-strad-del-diff', stDelDiff, 0, '', false, 2);

        setElText('sim-pe-del-real', pDel0.toFixed(2));
        setElText('sim-pe-del-fc', pDel_sim.toFixed(2));
        setElShift('sim-pe-del-diff', pDelDiff, (pDelDiff / Math.max(Math.abs(pDel0), 0.01) * 100), '', false, 2);

        // Gammas
        setElText('sim-ce-gam-real', cGam0.toFixed(5));
        setElText('sim-ce-gam-fc', cGam_sim.toFixed(5));
        setElShift('sim-ce-gam-diff', cGamDiff, (cGamDiff / Math.max(cGam0, 1e-6) * 100), '', false, 5);

        setElText('sim-strad-gam-real', stGam0.toFixed(5));
        setElText('sim-strad-gam-fc', stGam_sim.toFixed(5));
        setElShift('sim-strad-gam-diff', stGamDiff, (stGamDiff / Math.max(stGam0, 1e-6) * 100), '', false, 5);

        setElText('sim-pe-gam-real', pGam0.toFixed(5));
        setElText('sim-pe-gam-fc', pGam_sim.toFixed(5));
        setElShift('sim-pe-gam-diff', pGamDiff, (pGamDiff / Math.max(pGam0, 1e-6) * 100), '', false, 5);

        // Thetas
        setElText('sim-ce-th-real', '-₹' + Math.round(Math.abs(cThInr0)).toLocaleString('en-IN'));
        setElText('sim-ce-th-fc', '-₹' + Math.round(Math.abs(cThInr_sim)).toLocaleString('en-IN'));
        setElShift('sim-ce-th-diff', cThDiff, (cThDiff / Math.max(Math.abs(cThInr0), 1) * 100), '₹', true);

        setElText('sim-strad-th-real', '-₹' + Math.round(Math.abs(stThInr0)).toLocaleString('en-IN'));
        setElText('sim-strad-th-fc', '-₹' + Math.round(Math.abs(stThInr_sim)).toLocaleString('en-IN'));
        setElShift('sim-strad-th-diff', stThDiff, (stThDiff / Math.max(Math.abs(stThInr0), 1) * 100), '₹', true);

        setElText('sim-pe-th-real', '-₹' + Math.round(Math.abs(pThInr0)).toLocaleString('en-IN'));
        setElText('sim-pe-th-fc', '-₹' + Math.round(Math.abs(pThInr_sim)).toLocaleString('en-IN'));
        setElShift('sim-pe-th-diff', pThDiff, (pThDiff / Math.max(Math.abs(pThInr0), 1) * 100), '₹', true);

        // Vegas
        setElText('sim-ce-veg-real', '₹' + Math.round(cVgInr0).toLocaleString('en-IN'));
        setElText('sim-ce-veg-fc', '₹' + Math.round(cVgInr_sim).toLocaleString('en-IN'));
        setElShift('sim-ce-veg-diff', cVgDiff, (cVgDiff / Math.max(cVgInr0, 1) * 100), '₹', false);

        setElText('sim-strad-veg-real', '₹' + Math.round(stVgInr0).toLocaleString('en-IN'));
        setElText('sim-strad-veg-fc', '₹' + Math.round(stVgInr_sim).toLocaleString('en-IN'));
        setElShift('sim-strad-veg-diff', stVgDiff, (stVgDiff / Math.max(stVgInr0, 1) * 100), '₹', false);

        setElText('sim-pe-veg-real', '₹' + Math.round(pVgInr0).toLocaleString('en-IN'));
        setElText('sim-pe-veg-fc', '₹' + Math.round(pVgInr_sim).toLocaleString('en-IN'));
        setElShift('sim-pe-veg-diff', pVgDiff, (pVgDiff / Math.max(pVgInr0, 1) * 100), '₹', false);

        // Waterfall Attribution Cards (using Straddle as baseline portfolio)
        var dS = simSpotShock;
        var dt = simTimeShock;
        var dSig = simIvShock;

        var deltaEffectPts = stDel0 * dS;
        var gammaEffectPts = 0.5 * stGam0 * (dS * dS);
        var thetaEffectPts = -(Math.abs(stThInr0) / lot) * dt;
        var vegaEffectPts = (stVgInr0 / lot) * dSig;

        var deltaINR = deltaEffectPts * lot;
        var gammaINR = gammaEffectPts * lot;
        var thetaINR = thetaEffectPts * lot;
        var vegaINR = vegaEffectPts * lot;

        var totDpINR = deltaINR + gammaINR + thetaINR + vegaINR;
        var totDpPts = totDpINR / lot;

        var sumAbs = Math.abs(deltaINR) + Math.abs(gammaINR) + Math.abs(thetaINR) + Math.abs(vegaINR);
        var delPct = sumAbs > 0 ? (Math.abs(deltaINR) / sumAbs * 100) : 0;
        var gamPct = sumAbs > 0 ? (Math.abs(gammaINR) / sumAbs * 100) : 0;
        var thPct = sumAbs > 0 ? (Math.abs(thetaINR) / sumAbs * 100) : 0;
        var vegPct = sumAbs > 0 ? (Math.abs(vegaINR) / sumAbs * 100) : 0;

        var elDelVal = document.getElementById('cause-delta-val');
        if (elDelVal) {
            elDelVal.textContent = (deltaINR >= 0 ? '+₹' : '-₹') + Math.abs(Math.round(deltaINR)).toLocaleString('en-IN');
            elDelVal.style.color = deltaINR > 0 ? '#10b981' : (deltaINR < 0 ? '#ef4444' : '#ffffff');
        }
        var elDelPct = document.getElementById('cause-delta-pct');
        if (elDelPct) elDelPct.textContent = delPct.toFixed(0) + '% of move (' + (deltaEffectPts >= 0 ? '+' : '') + deltaEffectPts.toFixed(1) + ' pts)';

        var elGamVal = document.getElementById('cause-gamma-val');
        if (elGamVal) {
            elGamVal.textContent = '+₹' + Math.round(gammaINR).toLocaleString('en-IN');
            elGamVal.style.color = gammaINR > 0 ? '#fbbf24' : '#ffffff';
        }
        var elGamPct = document.getElementById('cause-gamma-pct');
        if (elGamPct) elGamPct.textContent = gamPct.toFixed(0) + '% of move (+' + gammaEffectPts.toFixed(1) + ' pts)';

        var elThVal = document.getElementById('cause-theta-val');
        if (elThVal) {
            elThVal.textContent = (thetaINR >= 0 ? '₹0' : '-₹' + Math.abs(Math.round(thetaINR)).toLocaleString('en-IN'));
            elThVal.style.color = thetaINR < 0 ? '#10b981' : '#ffffff';
        }
        var elThPct = document.getElementById('cause-theta-pct');
        if (elThPct) elThPct.textContent = thPct.toFixed(0) + '% of move (' + thetaEffectPts.toFixed(1) + ' pts)';

        var elVegVal = document.getElementById('cause-vega-val');
        if (elVegVal) {
            elVegVal.textContent = (vegaINR >= 0 ? '+₹' : '-₹') + Math.abs(Math.round(vegaINR)).toLocaleString('en-IN');
            elVegVal.style.color = vegaINR > 0 ? '#10b981' : (vegaINR < 0 ? '#ef4444' : '#ffffff');
        }
        var elVegPct = document.getElementById('cause-vega-pct');
        if (elVegPct) elVegPct.textContent = vegPct.toFixed(0) + '% of move (' + (vegaEffectPts >= 0 ? '+' : '') + vegaEffectPts.toFixed(1) + ' pts)';

        var elTotDp = document.getElementById('cause-total-dp');
        if (elTotDp) {
            var sign = totDpINR >= 0 ? '+' : '-';
            elTotDp.textContent = sign + '₹' + Math.abs(Math.round(totDpINR)).toLocaleString('en-IN') + ' (' + (totDpPts >= 0 ? '+' : '') + totDpPts.toFixed(1) + ' pts)';
            elTotDp.style.color = totDpINR > 0 ? '#10b981' : (totDpINR < 0 ? '#ef4444' : '#ffd54f');
        }

        // Automated Root Cause Diagnostics
        var elDiag = document.getElementById('cause-diagnosis-text');
        if (elDiag) {
            if (sumAbs === 0) {
                elDiag.innerHTML = 'Baseline equilibrium state (no market shock active). Select spot shift, time elapsed, or IV shock above to diagnose price drivers in real time.';
            } else {
                var maxVal = Math.max(Math.abs(deltaINR), Math.abs(gammaINR), Math.abs(thetaINR), Math.abs(vegaINR));
                if (maxVal === Math.abs(thetaINR)) {
                    elDiag.innerHTML = '<span style="color:#10b981;">⏳ TIME BLEED DOMINATING (' + thPct.toFixed(0) + '% of impact):</span> Time decay erodes premium by <strong style="color:#10b981;">-₹' + Math.abs(Math.round(thetaINR)).toLocaleString('en-IN') + '</strong> as ' + dt.toFixed(2) + ' day(s) expire.';
                } else if (maxVal === Math.abs(deltaINR)) {
                    var dirWord = deltaINR >= 0 ? 'expanding' : 'contracting';
                    elDiag.innerHTML = '<span style="color:#38bdf8;">🎯 DIRECTIONAL DELTA DOMINATING (' + delPct.toFixed(0) + '% of impact):</span> Spot move of ' + (dS >= 0 ? '+' : '') + dS + ' pts shifts premium ' + dirWord + ' by <strong style="color:#38bdf8;">' + (deltaINR >= 0 ? '+' : '-') + '₹' + Math.abs(Math.round(deltaINR)).toLocaleString('en-IN') + '</strong>.';
                } else if (maxVal === Math.abs(gammaINR)) {
                    elDiag.innerHTML = '<span style="color:#fbbf24;">⚡ GAMMA CONVEXITY DOMINATING (' + gamPct.toFixed(0) + '% of impact):</span> Underlying spot displacement triggers non-linear acceleration cushioning position by <strong style="color:#fbbf24;">+₹' + Math.round(gammaINR).toLocaleString('en-IN') + '</strong>.';
                } else {
                    var volWord = vegaINR >= 0 ? 'expansion' : 'crush';
                    elDiag.innerHTML = '<span style="color:#c084fc;">📊 VEGA VOLATILITY DOMINATING (' + vegPct.toFixed(0) + '% of impact):</span> Implied volatility ' + (dSig >= 0 ? '+' : '') + dSig.toFixed(1) + '% shock causes ' + volWord + ' of <strong style="color:#c084fc;">' + (vegaINR >= 0 ? '+' : '-') + '₹' + Math.abs(Math.round(vegaINR)).toLocaleString('en-IN') + '</strong>.';
                }
            }
        }
    }

    function setElText(id, text) {
        var el = document.getElementById(id);
        if (el) el.textContent = text;
    }

    function setElShift(id, diff, pct, prefix, invertGoodBad, decimals) {
        var el = document.getElementById(id);
        if (!el) return;
        var dec = (decimals !== undefined) ? decimals : 1;
        var sign = diff > 0 ? '+' : (diff < 0 ? '-' : '');
        var absDiff = Math.abs(diff);
        var diffStr = sign + (prefix || '') + (dec === 0 ? Math.round(absDiff).toLocaleString('en-IN') : absDiff.toFixed(dec));
        if (pct !== undefined && pct !== 0 && !isNaN(pct)) {
            diffStr += ' (' + (pct > 0 ? '+' : '') + pct.toFixed(0) + '%)';
        }
        el.textContent = diffStr;
        if (Math.abs(diff) < 1e-5) {
            el.style.color = '#94a3b8';
        } else if (diff > 0) {
            el.style.color = invertGoodBad ? '#ef4444' : '#10b981';
        } else {
            el.style.color = invertGoodBad ? '#10b981' : '#ef4444';
        }
    }

    // ── 3. SYNCHRONIZED MULTI-GREEK DESK VISUALIZER (4 Quadrants Dynamic Overlay) ──
    function updatePlotlyOverlayForecast() {
        var chartEl = document.getElementById('theta-chart-synced');
        if (!chartEl || !window.Plotly || !chartEl.data || chartEl.data.length < 13) return;

        var container = document.getElementById('theta-table-container');
        if (!container) return;

        var allRows = container.querySelectorAll('tbody tr');
        if (!allRows || allRows.length === 0) return;

        var spot0 = getSpotPrice();
        var dte0 = getDTE();
        var lot = getLotSize();

        var S_sim = spot0 + simSpotShock;
        var dte_sim = Math.max(0.001, dte0 - simTimeShock);
        var T_sim = dte_sim / 365.0;

        var fcStrikes = [];
        var fcCeDeltas = [];
        var fcPeDeltas = [];
        var fcGammas = [];
        var fcStraddleThetas = [];
        var fcStraddleVegas = [];

        allRows.forEach(function (r) {
            var k = parseFloat(r.getAttribute('data-strike'));
            if (isNaN(k)) return;

            var cIv = parseFloat(r.getAttribute('data-ce-iv')) || 14.0;
            var pIv = parseFloat(r.getAttribute('data-pe-iv')) || 14.0;

            var cSigma_sim = Math.max(0.01, (cIv + simIvShock) / 100.0);
            var pSigma_sim = Math.max(0.01, (pIv + simIvShock) / 100.0);

            var bsmCe = calcBSM(S_sim, k, T_sim, cSigma_sim, 0.07);
            var bsmPe = calcBSM(S_sim, k, T_sim, pSigma_sim, 0.07);

            fcStrikes.push(k);
            fcCeDeltas.push(bsmCe.callDelta);
            fcPeDeltas.push(bsmPe.putDelta);
            fcGammas.push(bsmCe.gamma + bsmPe.gamma);
            fcStraddleThetas.push(Math.abs((bsmCe.callTheta + bsmPe.putTheta) * lot));
            fcStraddleVegas.push((bsmCe.vega + bsmPe.vega) * lot);
        });

        // Trace Indices:
        // 3: Call Δ (Forecast) [row 1, col 1]
        // 4: Put Δ (Forecast) [row 1, col 1]
        // 6: Gamma (Forecast) [row 1, col 2]
        // 10: Straddle θ (Forecast) [row 2, col 1]
        // 12: Straddle Vega (Forecast) [row 2, col 2]
        try {
            window.Plotly.restyle(chartEl, {
                x: [fcStrikes, fcStrikes, fcStrikes, fcStrikes, fcStrikes],
                y: [fcCeDeltas, fcPeDeltas, fcGammas, fcStraddleThetas, fcStraddleVegas]
            }, [3, 4, 6, 10, 12]);

            // Update forecast vertical line (shape with line.color = '#00f0ff')
            if (chartEl.layout && chartEl.layout.shapes) {
                var shapes = chartEl.layout.shapes;
                var modified = false;
                for (var i = 0; i < shapes.length; i++) {
                    if (shapes[i].line && shapes[i].line.color === '#00f0ff') {
                        shapes[i].x0 = S_sim;
                        shapes[i].x1 = S_sim;
                        modified = true;
                    }
                }
                if (modified) {
                    window.Plotly.relayout(chartEl, { shapes: shapes });
                }
            }
        } catch (e) {
            console.warn('Plotly restyle warning:', e);
        }
    }

    // ── 4. STRIKE SELECTION & TABLE FILTERING ──
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
                    r.style.boxShadow = 'inset 0 0 0 2px #00f0ff';
                } else {
                    r.style.boxShadow = '';
                }
            });
        }
        updateExecutiveDecayCards(strike);
        updateGreekCauseAndEffect();
    }

    function applyThetaFilters() {
        var rangePct = parseFloat(currentThetaRange) || 10;
        var spot = getSpotPrice();

        var tbl = document.getElementById('theta-decay-table');
        if (tbl) {
            var allTrs = tbl.querySelectorAll('tbody tr');
            allTrs.forEach(function (r) {
                var strk = parseFloat(r.getAttribute('data-strike'));
                if (isNaN(strk)) return;

                var show = true;
                if (rangePct < 99) {
                    var diff = Math.abs(strk - spot) / spot * 100;
                    if (diff > rangePct) show = false;
                }
                r.style.display = show ? '' : 'none';
            });
        }
    }

    // ── Expose Global APIs for Dashboard bindings ──
    window.changeThetaRange = changeThetaRange;
    window.setThetaUnit = setThetaUnit;
    window.toggleThetaModel = toggleThetaModel;
    window.toggleAutoLockATM = toggleAutoLockATM;
    window.lockTableToATM = lockTableToATM;
    window.selectSimStrike = selectSimStrike;
    window.updateExecutiveDecayCards = updateExecutiveDecayCards;
    window.updateGreekCauseAndEffect = updateGreekCauseAndEffect;
    window.updatePlotlyOverlayForecast = updatePlotlyOverlayForecast;
    window.applySpotShock = applySpotShock;
    window.applyTimeShock = applyTimeShock;
    window.applyIvShock = applyIvShock;
    window.resetGreekShocks = resetGreekShocks;
    window.refreshThetaDecay = refreshThetaDecay;
    window.applyThetaFilters = applyThetaFilters;

    // ── Initialization on DOM ready ──
    document.addEventListener('DOMContentLoaded', function () {
        try {
            var savedLock = localStorage.getItem('th_autolock');
            if (savedLock !== null) {
                autoLockATM = (savedLock === 'true');
            }
        } catch (e) {}

        updateAutoLockBtn();

        var sel = document.getElementById('sel-th-strike');
        if (sel && sel.value) {
            activeSimStrike = parseFloat(sel.value);
        }

        updateExecutiveDecayCards(activeSimStrike);
        updateGreekCauseAndEffect();
        updateShockButtonStates();

        if (autoLockATM) {
            setTimeout(lockTableToATM, 300);
        }

        // Initialize forecast overlay after chart renders
        setTimeout(function () {
            updatePlotlyOverlayForecast();
        }, 600);
    });
})();
