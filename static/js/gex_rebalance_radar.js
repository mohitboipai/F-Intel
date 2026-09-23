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
    let _lastPOpt = null;
    let _lastOOpt = null;
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

        // 1. Status, Direction & Active Tier Badges
        const statusEl = document.getElementById('gr-status-badge');
        const dirEl = document.getElementById('gr-direction-badge');
        const tierEl = document.getElementById('gr-tier-badge');
        const archEl = document.getElementById('gr-archetype-badge');
        const descEl = document.getElementById('gr-desc');
        const actionEl = document.getElementById('gr-action-text');
        const tsEl = document.getElementById('gr-update-ts');

        // Expected Move Range Badge (Big vs Small move forecast)
        const moveEl = document.getElementById('gr-expected-move-badge');
        if (moveEl) {
            const expMove = data.expected_move;
            if (expMove && expMove.tier_name) {
                const minP = Math.round(expMove.expected_move_min_pts || 0);
                const maxP = Math.round(expMove.expected_move_max_pts || 0);
                const isBig = expMove.tier === 'TIER_3_BIG_MOVE';
                const isSmall = expMove.tier === 'TIER_1_SMALL_MOVE';
                if (isBig) {
                    moveEl.textContent = `⚡ BIG MOVE (${minP} – ${maxP} pts)`;
                    moveEl.style.color = '#f87171';
                    moveEl.style.background = 'rgba(239, 68, 68, 0.15)';
                    moveEl.style.borderColor = 'rgba(239, 68, 68, 0.35)';
                } else if (isSmall) {
                    moveEl.textContent = `🛡️ PIN / CHOP (${minP} – ${maxP} pts)`;
                    moveEl.style.color = '#34d399';
                    moveEl.style.background = 'rgba(16, 185, 129, 0.15)';
                    moveEl.style.borderColor = 'rgba(16, 185, 129, 0.35)';
                } else {
                    moveEl.textContent = `EXP MOVE: ${minP} – ${maxP} pts`;
                    moveEl.style.color = '#38bdf8';
                    moveEl.style.background = '#171d30';
                    moveEl.style.borderColor = '#232b45';
                }
                moveEl.style.display = 'inline-block';
            } else {
                moveEl.textContent = 'EXPECTED MOVE: 60 – 160 pts';
                moveEl.style.color = '#94a3b8';
                moveEl.style.display = 'inline-block';
            }
        }

        // Setup Archetype Badge (0DTE Gamma Rocket vs Weekly Breakout vs Macro Expansion)
        if (archEl) {
            const arch = data.setup_archetype || '0DTE_GAMMA_ROCKET';
            const archName = data.archetype_name || (arch === '0DTE_GAMMA_ROCKET' ? '0DTE Breakout' : arch === 'WEEKLY_MOMENTUM_BREAKOUT' ? 'Weekly Breakout' : 'Vol Expansion');
            archEl.textContent = archName;
            archEl.style.color = '#cbd5e1';
            archEl.style.borderColor = '#232b45';
            archEl.style.background = '#171d30';
        }

        const status = data.status || 'MONITORING';
        if (statusEl) {
            statusEl.textContent = status.replace(/_/g, ' ');
            if (status === 'IGNITED') {
                statusEl.style.background = 'rgba(16, 185, 129, 0.15)';
                statusEl.style.color = '#10b981';
                statusEl.style.borderColor = 'rgba(16, 185, 129, 0.35)';
                statusEl.style.boxShadow = 'none';
            } else if (status === 'STAND_ASIDE') {
                statusEl.style.background = 'rgba(239, 68, 68, 0.12)';
                statusEl.style.color = '#ef4444';
                statusEl.style.borderColor = 'rgba(239, 68, 68, 0.3)';
                statusEl.style.boxShadow = 'none';
            } else if (status === 'COILING' || status === 'ARMED') {
                statusEl.style.background = 'rgba(245, 158, 11, 0.12)';
                statusEl.style.color = '#f59e0b';
                statusEl.style.borderColor = 'rgba(245, 158, 11, 0.3)';
                statusEl.style.boxShadow = 'none';
            } else if (status === 'REBALANCING') {
                statusEl.style.background = 'rgba(56, 189, 248, 0.15)';
                statusEl.style.color = '#38bdf8';
                statusEl.style.borderColor = 'rgba(56, 189, 248, 0.35)';
                statusEl.style.boxShadow = 'none';
            } else if (status === 'TARGET_REACHED') {
                statusEl.style.background = 'rgba(16, 185, 129, 0.2)';
                statusEl.style.color = '#10b981';
                statusEl.style.borderColor = 'rgba(16, 185, 129, 0.4)';
                statusEl.style.boxShadow = 'none';
            } else {
                statusEl.style.background = '#171d30';
                statusEl.style.color = '#94a3b8';
                statusEl.style.borderColor = '#232b45';
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

        if (tierEl) {
            if (data.tier_name) {
                tierEl.textContent = data.tier_name;
                tierEl.style.display = 'inline-block';
                if (data.active_tier === 'TIER_3_EXPIRY_MEGA_MOVE') {
                    tierEl.style.background = 'rgba(255, 112, 67, 0.2)';
                    tierEl.style.color = '#ff7043';
                    tierEl.style.borderColor = '#ff7043';
                } else if (data.active_tier === 'TIER_2_RUNWAY_SQUEEZE') {
                    tierEl.style.background = 'rgba(0, 240, 255, 0.18)';
                    tierEl.style.color = '#00f0ff';
                    tierEl.style.borderColor = '#00f0ff';
                } else {
                    tierEl.style.background = 'rgba(0, 230, 118, 0.15)';
                    tierEl.style.color = '#00e676';
                    tierEl.style.borderColor = 'rgba(0, 230, 118, 0.3)';
                }
            } else {
                tierEl.style.display = 'none';
            }
        }

        // 2. Confluence Score & Multi-Module Sensor Strip
        const confValEl = document.getElementById('gr-confluence-val');
        const confBarEl = document.getElementById('gr-confluence-bar');
        const score = Number(data.confluence_score !== undefined ? data.confluence_score : 50);

        if (confValEl) {
            confValEl.textContent = score.toFixed(0) + ' / 100';
            if (score >= 70) {
                confValEl.style.color = '#00e676';
            } else if (score >= 50) {
                confValEl.style.color = '#ffd54f';
            } else {
                confValEl.style.color = '#ff5252';
            }
        }

        if (confBarEl) {
            confBarEl.style.width = Math.min(100, Math.max(0, score)) + '%';
            if (score >= 70) {
                confBarEl.style.background = 'linear-gradient(90deg, #ffd54f, #00e676)';
            } else if (score >= 50) {
                confBarEl.style.background = 'linear-gradient(90deg, #ff9800, #ffd54f)';
            } else {
                confBarEl.style.background = 'linear-gradient(90deg, #b71c1c, #ff5252)';
            }
        }

        // Sensor Badges
        const checklist = data.confluence_checklist || {};
        const updateSensor = (id, checkKey, defaultIcon, defaultName) => {
            const el = document.getElementById(id);
            if (!el) return;
            const item = checklist[checkKey];
            if (item) {
                el.textContent = `${item.label}: ${item.detail}`;
                if (item.status === 'PASS') {
                    el.style.color = '#00e676';
                    el.style.borderColor = 'rgba(0, 230, 118, 0.4)';
                    el.style.background = 'rgba(0, 230, 118, 0.1)';
                } else if (item.status === 'WARN') {
                    el.style.color = '#ffd54f';
                    el.style.borderColor = 'rgba(255, 213, 79, 0.4)';
                    el.style.background = 'rgba(255, 213, 79, 0.1)';
                } else if (item.status === 'FAIL') {
                    el.style.color = '#ff5252';
                    el.style.borderColor = 'rgba(255, 82, 82, 0.4)';
                    el.style.background = 'rgba(255, 82, 82, 0.1)';
                } else {
                    el.style.color = '#94a3b8';
                    el.style.borderColor = '#334155';
                    el.style.background = 'rgba(255, 255, 255, 0.05)';
                }
            } else {
                el.textContent = `${defaultIcon} ${defaultName}: Checking`;
                el.style.color = '#94a3b8';
            }
        };

        updateSensor('gr-sensor-gex', 'dealer_gex', '⚡', 'GEX');
        updateSensor('gr-sensor-oi', 'oi_flow', '🌊', 'OI Flow');
        updateSensor('gr-sensor-abs', 'absorption', '🕯️', 'Absorption');
        updateSensor('gr-sensor-vol', 'vol_skew', '📊', 'Vol Skew');
        updateSensor('gr-sensor-pin', 'pin_cascade', '⏱️', 'Pin Status');

        // 3. Stand-Aside Shield Banner
        const standAsideBanner = document.getElementById('gr-stand-aside-banner');
        const standAsideText = document.getElementById('gr-stand-aside-text');
        const isStandAside = (data.trade_ready === false) || (data.status === 'STAND_ASIDE');

        if (standAsideBanner) {
            if (isStandAside) {
                standAsideBanner.style.display = 'flex';
                if (standAsideText) {
                    if (data.rejection_reasons && data.rejection_reasons.length > 0) {
                        standAsideText.textContent = 'STAND ASIDE: ' + data.rejection_reasons[0];
                    } else if (data.status_desc) {
                        standAsideText.textContent = data.status_desc.replace(/🛡️\s*/, '');
                    } else {
                        standAsideText.textContent = 'STAND ASIDE: Market in chop / low confluence. No high-ROI edge. Cash is a position.';
                    }
                }
            } else {
                standAsideBanner.style.display = 'none';
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

        // 4. The 4 Essential Levels
        const spotEl = document.getElementById('gr-spot-val');
        const trigEl = document.getElementById('gr-trigger-val');
        const tgtEl = document.getElementById('gr-target-val');
        const fortEl = document.getElementById('gr-fortress-val');

        if (spotEl && data.spot) spotEl.textContent = Number(data.spot).toLocaleString('en-IN', { minimumFractionDigits: 1 });
        if (trigEl && data.trigger_strike) trigEl.textContent = Number(data.trigger_strike).toLocaleString('en-IN', { maximumFractionDigits: 0 });
        if (tgtEl && data.rebalance_target) tgtEl.textContent = Number(data.rebalance_target).toLocaleString('en-IN', { maximumFractionDigits: 0 });
        if (fortEl && data.terminal_fortress) fortEl.textContent = Number(data.terminal_fortress).toLocaleString('en-IN', { maximumFractionDigits: 0 });

        // 4b. Time Horizon, Holding Duration & Theta Decay Strip
        const horizonEl = document.getElementById('gr-horizon-val');
        const maxHoldEl = document.getElementById('gr-max-hold-val');
        const timeStopEl = document.getElementById('gr-time-stop-val');
        const thetaBurnEl = document.getElementById('gr-theta-burn-val');

        if (horizonEl) {
            horizonEl.textContent = data.recommended_horizon || '15 – 35 Mins';
        }
        if (maxHoldEl) {
            if (data.max_hold_mins) {
                maxHoldEl.textContent = `${data.max_hold_mins} Mins Max (Auto-Veto)`;
            } else if (data.recommended_horizon) {
                maxHoldEl.textContent = data.recommended_horizon;
            }
        }
        if (timeStopEl) {
            timeStopEl.textContent = data.hard_time_stop || 'Exit by 14:45 or 35m inactivity';
        }
        if (thetaBurnEl) {
            thetaBurnEl.textContent = data.theta_decay_burn_15m || (data.primary_option && data.primary_option.theta_burn_15m_str) || '--';
        }

        // 5. Runway Progress Bar
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

        // 6. Primary ATM Option Card
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
            const pRr = document.getElementById('gr-p-rr');

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
            if (pRr) pRr.textContent = `R:R ${(pOpt.rr_ratio || 1.8).toFixed(1)} : 1`;
            _lastPOpt = pOpt;
        }

        // 7. 0DTE OTM Gamma Rocket Option Card
        const oOpt = data.otm_gamma_rocket;
        if (oOpt) {
            _lastOOpt = oOpt;
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
                    const thetaStr = oOpt.theta_burn_15m_str ? ` · Burn: ${oOpt.theta_burn_15m_str}` : '';
                    oStatus.textContent = `🔥 0DTE Convexity Hero Active${thetaStr}`;
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
        _pollTimer = setInterval(function () {
            // Only poll if tab-chain is visible
            var chainTab = document.getElementById('tab-chain');
            if (chainTab && !chainTab.classList.contains('active')) return;
            poll();
        }, 4000);
    }

    async function trackPrimary() {
        if (!_lastPOpt || !_lastPOpt.strike) {
            alert("No active primary strike recommendation.");
            return;
        }
        const select = document.getElementById('gr-p-lots');
        const lots = select ? parseInt(select.value, 10) : 1;
        const ltp = _lastPOpt.current_price ?? _lastPOpt.ltp ?? 0;
        if (window.IgnitionScanner && typeof window.IgnitionScanner.trackGenericStrike === 'function') {
            await window.IgnitionScanner.trackGenericStrike(
                `Radar Primary: ${_lastPOpt.strike} ${_lastPOpt.type}`,
                _lastPOpt.type,
                _lastPOpt.strike,
                ltp,
                _lastPOpt.stop_loss,
                _lastPOpt.target_1,
                _lastPOpt.runner_target ?? _lastPOpt.target_2,
                lots,
                'RADAR'
            );
        }
    }

    async function trackRocket() {
        if (!_lastOOpt || !_lastOOpt.strike) {
            alert("No active 0DTE rocket recommendation.");
            return;
        }
        const select = document.getElementById('gr-o-lots');
        const lots = select ? parseInt(select.value, 10) : 1;
        const ltp = _lastOOpt.current_price ?? _lastOOpt.ltp ?? 0;
        if (window.IgnitionScanner && typeof window.IgnitionScanner.trackGenericStrike === 'function') {
            await window.IgnitionScanner.trackGenericStrike(
                `0DTE Rocket: ${_lastOOpt.strike} ${_lastOOpt.type}`,
                _lastOOpt.type,
                _lastOOpt.strike,
                ltp,
                _lastOOpt.stop_loss,
                _lastOOpt.target_1,
                _lastOOpt.runner_target ?? _lastOOpt.target_2,
                lots,
                'RADAR'
            );
        }
    }

    // Export to window
    window.GexRebalanceRadar = {
        refresh: poll,
        updateUI: updateRadarUI,
        trackPrimary: trackPrimary,
        trackRocket: trackRocket,
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
