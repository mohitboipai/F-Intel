/**
 * static/js/gex_rebalance_radar.js
 * ═══════════════════════════════════════════════════════════════════════════
 * F-INTEL: OPTION BUYER RADAR & GEX REBALANCE CONTROLLER
 * ═══════════════════════════════════════════════════════════════════════════
 * Listens for WebSocket broadcasts and polls /api/gex-rebalance to update:
 * 1. Status Badge (COILING, ARMED, IGNITED, REBALANCING, TARGET REACHED)
 * 2. The 4 Essential Levels: Spot, Ignition Trigger, Rebalance Target, Fortress Pin
 * 3. Runway Progress Bar & Dealer Futures Demand
 * 4. Dual Strikes: Primary ATM (Steady) + 0DTE OTM Gamma Rocket (Explosive)
 * 5. One-sentence Action Directives
 */

(function () {
    'use strict';

    let _lastData = null;
    let _pollTimer = null;

    async function fetchRadarData() {
        try {
            const res = await fetch('/api/gex-rebalance?t=' + Date.now());
            if (!res.ok) return null;
            const data = await res.json();
            return data;
        } catch (e) {
            console.warn('[GexRebalanceRadar] Fetch error:', e);
            return null;
        }
    }

    function updateRadarUI(data) {
        if (!data || !data.ok) return;
        _lastData = data;

        // 1. Status & Direction Badges
        const statusEl = document.getElementById('gr-status-badge');
        const dirEl = document.getElementById('gr-direction-badge');
        const descEl = document.getElementById('gr-desc');
        const actionEl = document.getElementById('gr-action-text');
        const tsEl = document.getElementById('gr-update-ts');

        const status = data.status || 'MONITORING';
        if (statusEl) {
            statusEl.textContent = status.replace('_', ' ');
            if (status === 'IGNITED') {
                statusEl.style.background = 'rgba(0, 230, 118, 0.2)';
                statusEl.style.color = '#00e676';
                statusEl.style.borderColor = '#00e676';
                statusEl.style.boxShadow = '0 0 12px rgba(0, 230, 118, 0.4)';
            } else if (status === 'COILING' || status === 'ARMED') {
                statusEl.style.background = 'rgba(255, 213, 79, 0.2)';
                statusEl.style.color = '#ffd54f';
                statusEl.style.borderColor = '#ffd54f';
                statusEl.style.boxShadow = 'none';
            } else if (status === 'REBALANCING') {
                statusEl.style.background = 'rgba(0, 240, 255, 0.2)';
                statusEl.style.color = '#00f0ff';
                statusEl.style.borderColor = '#00f0ff';
                statusEl.style.boxShadow = '0 0 10px rgba(0, 240, 255, 0.3)';
            } else if (status === 'TARGET_REACHED') {
                statusEl.style.background = 'rgba(16, 185, 129, 0.25)';
                statusEl.style.color = '#10b981';
                statusEl.style.borderColor = '#10b981';
                statusEl.style.boxShadow = '0 0 15px rgba(16, 185, 129, 0.5)';
            } else {
                statusEl.style.background = 'rgba(255, 255, 255, 0.08)';
                statusEl.style.color = '#94a3b8';
                statusEl.style.borderColor = '#334155';
                statusEl.style.boxShadow = 'none';
            }
        }

        if (dirEl) {
            const dir = data.direction || 'NEUTRAL';
            if (dir === 'BULLISH_CE') {
                dirEl.textContent = 'CALL BUY (CE)';
                dirEl.style.background = 'rgba(0, 230, 118, 0.15)';
                dirEl.style.color = '#00e676';
                dirEl.style.borderColor = 'rgba(0, 230, 118, 0.4)';
            } else if (dir === 'BEARISH_PE') {
                dirEl.textContent = 'PUT BUY (PE)';
                dirEl.style.background = 'rgba(255, 68, 68, 0.15)';
                dirEl.style.color = '#ff4444';
                dirEl.style.borderColor = 'rgba(255, 68, 68, 0.4)';
            } else {
                dirEl.textContent = 'NO SETUP';
                dirEl.style.background = 'rgba(255, 255, 255, 0.05)';
                dirEl.style.color = '#94a3b8';
                dirEl.style.borderColor = '#334155';
            }
        }

        if (descEl && data.status_desc) {
            descEl.textContent = data.status_desc;
        }

        if (actionEl && data.action_summary) {
            actionEl.textContent = data.action_summary;
        }

        if (tsEl && data.timestamp) {
            tsEl.textContent = 'Updated: ' + data.timestamp;
        }

        // 2. The 4 Essential Levels
        const spotEl = document.getElementById('gr-spot-val');
        const trigEl = document.getElementById('gr-trigger-val');
        const tgtEl = document.getElementById('gr-target-val');
        const fortEl = document.getElementById('gr-fortress-val');

        if (spotEl && data.spot) spotEl.textContent = Number(data.spot).toLocaleString('en-IN', { minimumFractionDigits: 1 });
        if (trigEl && data.trigger_strike) trigEl.textContent = Number(data.trigger_strike).toLocaleString('en-IN', { maximumFractionDigits: 0 });
        if (tgtEl && data.rebalance_target) tgtEl.textContent = Number(data.rebalance_target).toLocaleString('en-IN', { maximumFractionDigits: 0 });
        if (fortEl && data.terminal_fortress) fortEl.textContent = Number(data.terminal_fortress).toLocaleString('en-IN', { maximumFractionDigits: 0 });

        // 3. Runway Progress Bar
        const barStart = document.getElementById('gr-bar-start');
        const barRunway = document.getElementById('gr-bar-runway');
        const barTarget = document.getElementById('gr-bar-target');
        const barFill = document.getElementById('gr-progress-fill');
        const pctEl = document.getElementById('gr-progress-pct');
        const fuelEl = document.getElementById('gr-fuel-indicator');

        if (barStart && data.trigger_strike) barStart.textContent = Number(data.trigger_strike).toFixed(0);
        if (barRunway && data.runway_pts) barRunway.textContent = Number(data.runway_pts).toFixed(0) + ' pts';
        if (barTarget && data.rebalance_target) barTarget.textContent = Number(data.rebalance_target).toFixed(0);

        const progress = Math.max(0, Math.min(100, data.progress_pct || 0));
        if (barFill) barFill.style.width = progress + '%';
        if (pctEl) pctEl.textContent = progress.toFixed(0) + '%';
        if (fuelEl && data.dealer_fuel_lots) {
            fuelEl.textContent = 'Dealer Futures Fuel: ' + Number(data.dealer_fuel_lots).toLocaleString('en-IN') + ' Lots';
        }

        // 4. Primary ATM Option Card
        const pOpt = data.primary_option;
        if (pOpt) {
            const pStrike = document.getElementById('gr-p-strike-name');
            const pLtp = document.getElementById('gr-p-ltp');
            const pBuy = document.getElementById('gr-p-buy');
            const pT1 = document.getElementById('gr-p-t1');
            const pT1Pct = document.getElementById('gr-p-t1-pct');
            const pT2 = document.getElementById('gr-p-t2');
            const pT2Pct = document.getElementById('gr-p-t2-pct');
            const pSl = document.getElementById('gr-p-sl');
            const pSlPct = document.getElementById('gr-p-sl-pct');

            const ltpVal = pOpt.current_price ?? pOpt.ltp ?? 0;
            const t1Val = pOpt.target_1 ?? 0;
            const t1PctVal = pOpt.target_gain_pct ?? pOpt.target_1_roi_pct ?? 0;
            const t2Val = pOpt.runner_target ?? pOpt.target_2 ?? 0;
            const t2PctVal = pOpt.runner_gain_pct ?? pOpt.target_2_roi_pct ?? 0;
            const slVal = pOpt.stop_loss ?? 0;
            const slPctVal = pOpt.stop_loss_pct ?? 0;

            if (pStrike) pStrike.textContent = `${pOpt.strike} ${pOpt.type}`;
            if (pLtp) pLtp.textContent = '₹' + Number(ltpVal).toFixed(1);
            if (pBuy) pBuy.textContent = pOpt.buy_zone || '--';
            if (pT1) pT1.textContent = '₹' + Number(t1Val).toFixed(1);
            if (pT1Pct) pT1Pct.textContent = '+' + Number(t1PctVal).toFixed(0) + '%';
            if (pT2) pT2.textContent = '₹' + Number(t2Val).toFixed(1);
            if (pT2Pct) pT2Pct.textContent = '+' + Number(t2PctVal).toFixed(0) + '%';
            if (pSl) pSl.textContent = '₹' + Number(slVal).toFixed(1);
            if (pSlPct) pSlPct.textContent = Number(slPctVal).toFixed(0) + '%';
        }

        // 5. 0DTE OTM Gamma Rocket Option Card
        const oOpt = data.otm_gamma_rocket;
        if (oOpt) {
            const oStrike = document.getElementById('gr-o-strike-name');
            const oLtp = document.getElementById('gr-o-ltp');
            const oBuy = document.getElementById('gr-o-buy');
            const oT1 = document.getElementById('gr-o-t1');
            const oT1Pct = document.getElementById('gr-o-t1-pct');
            const oT2 = document.getElementById('gr-o-t2');
            const oT2Pct = document.getElementById('gr-o-t2-pct');
            const oSl = document.getElementById('gr-o-sl');
            const oSlPct = document.getElementById('gr-o-sl-pct');
            const oStatus = document.getElementById('gr-o-status-tag');

            const ltpVal = oOpt.current_price ?? oOpt.ltp ?? 0;
            const t1Val = oOpt.target_1 ?? 0;
            const t1PctVal = oOpt.target_gain_pct ?? oOpt.target_1_roi_pct ?? 0;
            const t2Val = oOpt.runner_target ?? oOpt.target_2 ?? 0;
            const t2PctVal = oOpt.runner_gain_pct ?? oOpt.target_2_roi_pct ?? 0;
            const slVal = oOpt.stop_loss ?? 0;
            const slPctVal = oOpt.stop_loss_pct ?? 0;

            if (oStrike) oStrike.textContent = `${oOpt.strike} ${oOpt.type}`;
            if (oLtp) oLtp.textContent = '₹' + Number(ltpVal).toFixed(1);
            if (oBuy) oBuy.textContent = oOpt.buy_zone || '--';
            if (oT1) oT1.textContent = '₹' + Number(t1Val).toFixed(1);
            if (oT1Pct) oT1Pct.textContent = '+' + Number(t1PctVal).toFixed(0) + '%';
            if (oT2) oT2.textContent = '₹' + Number(t2Val).toFixed(1);
            if (oT2Pct) oT2Pct.textContent = '+' + Number(t2PctVal).toFixed(0) + '%';
            if (oSl) oSl.textContent = '₹' + Number(slVal).toFixed(1);
            if (oSlPct) oSlPct.textContent = Number(slPctVal).toFixed(0) + '%';
            if (oStatus) {
                if (oOpt.is_active) {
                    oStatus.textContent = '🔥 Convexity In Vacuum Runway';
                    oStatus.style.color = '#00e676';
                } else {
                    oStatus.textContent = 'Outside Runway / High DTE';
                    oStatus.style.color = '#94a3b8';
                }
            }
        }
    }

    // ─────────────────────────────────────────────────────────────────────
    // INTRADAY SIGNAL RENDERER
    // ─────────────────────────────────────────────────────────────────────

    function updateIntradaySignalUI(data) {
        if (!data || !data.ok) return;

        const swing   = data.swing_quality || {};
        const entry   = data.entry_signal  || null;
        const abs_    = data.absorption    || {};
        const oiVel   = data.oi_velocity   || {};
        const score   = data.score || 0;

        // ── Swing Quality Badge ──────────────────────────────────────────
        const qualityEl = document.getElementById('ids-quality-badge');
        if (qualityEl) {
            const q = swing.quality || 'UNKNOWN';
            const colors = { TRENDING: '#00e676', COILED: '#ffd54f', CHOPPY: '#ff5252', UNKNOWN: '#94a3b8' };
            const icons  = { TRENDING: '🟢', COILED: '🟡', CHOPPY: '🔴', UNKNOWN: '⚪' };
            qualityEl.textContent = `${icons[q] || '⚪'} ${q}`;
            qualityEl.style.color = colors[q] || '#94a3b8';
            qualityEl.style.borderColor = colors[q] || '#334155';
            qualityEl.style.background = `${colors[q] || '#334155'}22`;
        }

        // ── Session Phase Badge ─────────────────────────────────────────
        const phaseEl = document.getElementById('ids-phase-badge');
        if (phaseEl) {
            const phase = swing.session_phase || '';
            const phaseMap = {
                OPENING_RANGE: '⏰ Opening Range',
                MID_TREND:     '📈 Mid Trend',
                EXPIRY_HEAT:   '🔥 Expiry Heat',
                LATE:          '🌙 Late Session'
            };
            phaseEl.textContent = phaseMap[phase] || phase;
        }

        // ── ADR & RSI line ────────────────────────────────────────────────
        const adrEl = document.getElementById('ids-adr-rsi');
        if (adrEl) {
            adrEl.textContent =
                `ADR ${swing.adr_pct !== undefined ? swing.adr_pct.toFixed(0) + '%' : '–'} · ` +
                `RSI₅ ${swing.rsi_5min !== undefined ? swing.rsi_5min.toFixed(0) : '–'} · ` +
                `Dir bars ${swing.directional_bars !== undefined ? swing.directional_bars : '–'}/10`;
        }

        // ── Absorption Badge ──────────────────────────────────────────────
        const absEl = document.getElementById('ids-absorption-badge');
        if (absEl) {
            const aq = abs_.setup_quality || 'NONE';
            const absColors = { STRONG: '#00e676', MODERATE: '#ffd54f', WEAK: '#ff9800', NONE: '#94a3b8' };
            absEl.textContent = `Absorption: ${aq} (${abs_.wick_bars || 0} bars · ${(abs_.vol_accel || 0).toFixed(1)}× vol)`;
            absEl.style.color = absColors[aq] || '#94a3b8';
        }

        // ── OI Velocity ───────────────────────────────────────────────────
        const oiEl = document.getElementById('ids-oi-velocity');
        if (oiEl) {
            if (oiVel.available) {
                const violent = (oiVel.violently_unwinding || []).length;
                oiEl.textContent = violent > 0
                    ? `🚨 VIOLENT UNWIND (${violent} strikes)`
                    : oiVel.has_capitulation ? '⚠️ Capitulation detected' : '✅ OI stable';
                oiEl.style.color = violent > 0 ? '#ff5252' : oiVel.has_capitulation ? '#ffd54f' : '#00e676';
            } else {
                oiEl.textContent = 'OI velocity: warming up…';
                oiEl.style.color = '#94a3b8';
            }
        }

        // ── Score Bar ────────────────────────────────────────────────────
        const scoreBar = document.getElementById('ids-score-bar');
        const scoreVal = document.getElementById('ids-score-value');
        if (scoreBar) scoreBar.style.width = Math.min(score, 100).toFixed(0) + '%';
        if (scoreVal) {
            scoreVal.textContent = score.toFixed(0) + ' / 100';
            scoreVal.style.color = score >= 65 ? '#00e676' : score >= 45 ? '#ffd54f' : '#94a3b8';
        }

        // ── Entry Signal Panel ────────────────────────────────────────────
        const entryPanel = document.getElementById('ids-entry-panel');
        if (entryPanel) {
            if (data.actionable && entry) {
                entryPanel.style.display = 'block';

                const dir    = entry.direction || 'NEUTRAL';
                const dirEl  = document.getElementById('ids-entry-direction');
                if (dirEl) {
                    dirEl.textContent = dir === 'BULLISH' ? '🟢 BULLISH CE BUY' : '🔴 BEARISH PE BUY';
                    dirEl.style.color = dir === 'BULLISH' ? '#00e676' : '#ff5252';
                }

                const setText = (id, val) => { const el = document.getElementById(id); if (el) el.textContent = val; };
                const ez = entry.entry_zone || [0, 0];
                setText('ids-entry-zone',    `${(ez[0] || 0).toFixed(0)} – ${(ez[1] || 0).toFixed(0)}`);
                setText('ids-entry-sl',      (entry.sl_spot || 0).toFixed(0));
                setText('ids-entry-t1',      (entry.t1 || 0).toFixed(0));
                setText('ids-entry-t2',      (entry.t2 || 0).toFixed(0));
                setText('ids-entry-strike',  `${entry.option_strike || '–'} ${entry.option_type || ''}`);
                setText('ids-entry-rr',      `${(entry.rr_ratio || 0).toFixed(1)} : 1`);

                const rationaleEl = document.getElementById('ids-entry-rationale');
                if (rationaleEl && Array.isArray(entry.rationale)) {
                    rationaleEl.innerHTML = entry.rationale
                        .map(l => `<div class="ids-rationale-line">${l}</div>`)
                        .join('');
                }

                // OTM Gamma Rocket
                const otmEl = document.getElementById('ids-otm-rocket');
                if (otmEl && entry.otm_rocket) {
                    const r = entry.otm_rocket;
                    otmEl.style.display = 'block';
                    otmEl.innerHTML = `🚀 OTM Gamma Rocket: ${r.strike || '–'} ${r.type || ''} @ ₹${(r.current_price || 0).toFixed(0)} → ₹${(r.target_premium || 0).toFixed(0)}`;
                } else if (otmEl) {
                    otmEl.style.display = 'none';
                }
            } else {
                entryPanel.style.display = 'none';
            }
        }

        // ── Rationale List ──────────────────────────────────────────────
        const rationaleListEl = document.getElementById('ids-rationale-list');
        if (rationaleListEl && Array.isArray(data.rationale)) {
            rationaleListEl.innerHTML = data.rationale
                .map(l => `<li class="ids-rationale-item">${l}</li>`)
                .join('');
        }
    }

    async function fetchIntradaySignal() {
        try {
            const res = await fetch('/api/intraday-signal?t=' + Date.now());
            if (!res.ok) return null;
            return await res.json();
        } catch (e) {
            console.warn('[IntradaySignal] Fetch error:', e);
            return null;
        }
    }

    async function pollIntraday() {
        const data = await fetchIntradaySignal();
        if (data) updateIntradaySignalUI(data);
    }

    async function poll() {
        const data = await fetchRadarData();
        if (data) {
            updateRadarUI(data);
        }
        // Also refresh intraday panel on same 3s cycle
        await pollIntraday();
    }

    function startPolling() {
        if (_pollTimer) clearInterval(_pollTimer);
        poll();
        _pollTimer = setInterval(poll, 3000);
    }

    // Export to window
    window.GexRebalanceRadar = {
        refresh: poll,
        updateUI: updateRadarUI,
        handleWsMessage: function (msg) {
            if (!msg) return;
            if (msg.type === 'gex_rebalance_update') {
                const payload = msg.payload || msg.data;
                if (payload) updateRadarUI(payload);
            }
            if (msg.type === 'intraday_signal') {
                const payload = msg.payload || msg.data;
                if (payload) updateIntradaySignalUI(payload);
            }
        }
    };

    // Auto-init on DOM ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', startPolling);
    } else {
        startPolling();
    }
})();
