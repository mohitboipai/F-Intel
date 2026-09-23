/**
 * static/js/ignition_scanner.js
 * ═══════════════════════════════════════════════════════════════════════════
 * F-INTEL: 0DTE GAMMA IGNITION SCANNER & ONE-CLICK POSITION TRACKER
 * ═══════════════════════════════════════════════════════════════════════════
 * 
 * Scans cross-strike compression → ignition setups across ATM ± N strikes.
 * Features:
 *  - Ranked candidates display with live confluence score
 *  - Compression & ignition diagnostic telemetry
 *  - Full Greeks breakdown at entry & live
 *  - One-click "TRACK THIS" directly into Strategy Builder Paper Portfolio
 */

(function () {
    'use strict';

    let _lastData = null;
    let _pollTimer = null;
    let _lotSizeMultiplier = 65;

    // Toast notification helper
    function showToast(msg, isSuccess = true) {
        let toast = document.getElementById('ignition-toast');
        if (!toast) {
            toast = document.createElement('div');
            toast.id = 'ignition-toast';
            toast.style.cssText = `
                position: fixed;
                bottom: 24px;
                right: 24px;
                z-index: 99999;
                padding: 12px 20px;
                border-radius: 8px;
                font-family: 'Inter', -apple-system, sans-serif;
                font-size: 13px;
                font-weight: 700;
                box-shadow: 0 10px 30px rgba(0,0,0,0.5);
                transition: opacity 0.3s ease, transform 0.3s ease;
                display: flex;
                align-items: center;
                gap: 8px;
            `;
            document.body.appendChild(toast);
        }
        toast.style.background = isSuccess ? '#0d2818' : '#3d0c11';
        toast.style.color = isSuccess ? '#00e676' : '#ff5252';
        toast.style.border = `1px solid ${isSuccess ? '#00e676' : '#ff5252'}`;
        toast.innerHTML = (isSuccess ? '✅ ' : '⚠️ ') + msg;
        toast.style.opacity = '1';
        toast.style.transform = 'translateY(0)';

        setTimeout(() => {
            toast.style.opacity = '0';
            toast.style.transform = 'translateY(10px)';
        }, 4000);
    }

    async function fetchCandidates() {
        try {
            const res = await fetch('/api/ignition-candidates?t=' + Date.now());
            if (!res.ok) return null;
            return await res.json();
        } catch (e) {
            console.warn('[IgnitionScanner] Fetch error:', e);
            return null;
        }
    }

    function renderUI(data) {
        if (!data || !data.ok) return;
        _lastData = data;

        const countEl = document.getElementById('is-active-count');
        const compEl = document.getElementById('is-compressed-count');
        const spotCompEl = document.getElementById('is-spot-comp-status');
        const tsEl = document.getElementById('is-update-ts');
        const container = document.getElementById('is-candidates-container');

        if (countEl) countEl.textContent = (data.candidates || []).length;
        if (compEl) compEl.textContent = `${data.compressed_count || 0} / ${data.scan_count || 0}`;
        if (spotCompEl) {
            const isComp = !!data.spot_compressed;
            spotCompEl.textContent = isComp ? '⚡ Coiling (Tight Range)' : 'Normal Movement';
            spotCompEl.style.color = isComp ? '#ffd54f' : '#94a3b8';
        }
        if (tsEl && data.timestamp) tsEl.textContent = 'Updated: ' + data.timestamp;

        if (!container) return;

        const candidates = data.candidates || [];
        if (candidates.length === 0) {
            container.innerHTML = `
                <div style="grid-column: 1 / -1; padding: 32px 16px; text-align: center; background: rgba(255,255,255,0.02); border: 1px dashed #222744; border-radius: 8px;">
                    <div style="font-size: 24px; margin-bottom: 8px;">📡</div>
                    <div style="font-size: 13px; font-weight: 700; color: #94a3b8;">Scanning ATM ± 6 strikes for Compression → Ignition setups...</div>
                    <div style="font-size: 11px; color: #64748b; margin-top: 4px;">Candidates will appear when option premium coils in a tight base and spot triggers a high-velocity gamma move.</div>
                </div>
            `;
            return;
        }

        container.innerHTML = '';

        candidates.forEach((cand, idx) => {
            const isIgniting = cand.status === 'IGNITING';
            const statusColor = isIgniting ? '#00e676' : '#ffd54f';
            const statusBg = isIgniting ? 'rgba(0, 230, 118, 0.15)' : 'rgba(255, 213, 79, 0.15)';
            const statusBorder = isIgniting ? '#00e676' : '#ffd54f';
            const cardBg = isIgniting
                ? 'linear-gradient(135deg, rgba(16,28,44,0.95), rgba(12,32,24,0.95))'
                : 'linear-gradient(135deg, rgba(20,24,44,0.9), rgba(18,22,36,0.9))';
            const cardBorder = isIgniting ? '1px solid rgba(0, 230, 118, 0.35)' : '1px solid #222744';

            const greeks = cand.greeks_at_entry || {};
            const card = document.createElement('div');
            card.style.cssText = `
                background: ${cardBg};
                border: ${cardBorder};
                border-radius: 10px;
                padding: 14px 16px;
                display: flex;
                flex-direction: column;
                gap: 12px;
                box-shadow: 0 4px 16px rgba(0,0,0,0.3);
                transition: transform 0.2s ease, border-color 0.2s ease;
            `;

            const rankNum = idx + 1;
            const candJson = encodeURIComponent(JSON.stringify(cand));

            card.innerHTML = `
                <!-- Card Header -->
                <div style="display:flex; justify-content:space-between; align-items:center; border-bottom:1px solid rgba(255,255,255,0.06); padding-bottom:10px;">
                    <div style="display:flex; align-items:center; gap:8px;">
                        <span style="font-size:11px; font-weight:900; color:#38bdf8; background:rgba(0,240,255,0.12); padding:3px 8px; border-radius:4px; font-family:'JetBrains Mono',monospace;">
                            #${rankNum}
                        </span>
                        <div>
                            <div style="font-size:16px; font-weight:900; color:#ffffff; font-family:'JetBrains Mono',monospace;">
                                ${cand.strike} ${cand.type}
                            </div>
                            <div style="font-size:10px; color:#94a3b8;">
                                LTP: <span style="color:#00f0ff; font-weight:800;">₹${cand.entry_premium.toFixed(1)}</span>
                            </div>
                        </div>
                    </div>
                    <div style="text-align:right; display:flex; flex-direction:column; align-items:flex-end; gap:4px;">
                        <span style="font-size:10px; font-weight:900; padding:3px 10px; border-radius:4px; background:${statusBg}; color:${statusColor}; border:1px solid ${statusBorder}; letter-spacing:0.5px;">
                            ${isIgniting ? '🔥 IGNITING' : '⚡ COILING'}
                        </span>
                        <div style="font-size:11px; font-weight:800; color:#ffd54f; font-family:'JetBrains Mono',monospace;">
                            Score: ${cand.confluence_score}/100
                        </div>
                    </div>
                </div>

                <!-- Price Targets & Stops Grid -->
                <div style="display:grid; grid-template-columns: repeat(4, 1fr); gap:6px; text-align:center;">
                    <div style="background:rgba(255,255,255,0.03); padding:6px 4px; border-radius:5px; border:1px solid #1e2442;">
                        <div style="font-size:9px; color:#94a3b8; font-weight:700;">BUY AT</div>
                        <div style="font-size:12px; font-weight:800; color:#ffffff; font-family:'JetBrains Mono',monospace;">₹${cand.entry_premium.toFixed(1)}</div>
                        <div style="font-size:8px; color:#64748b;">Current</div>
                    </div>
                    <div style="background:rgba(255,68,68,0.06); padding:6px 4px; border-radius:5px; border:1px solid rgba(255,68,68,0.25);">
                        <div style="font-size:9px; color:#ef5350; font-weight:700;">STOP LOSS</div>
                        <div style="font-size:12px; font-weight:800; color:#ef5350; font-family:'JetBrains Mono',monospace;">₹${cand.sl_premium.toFixed(1)}</div>
                        <div style="font-size:8px; color:#ef5350;">-${cand.max_risk_pct}%</div>
                    </div>
                    <div style="background:rgba(0,230,118,0.06); padding:6px 4px; border-radius:5px; border:1px solid rgba(0,230,118,0.25);">
                        <div style="font-size:9px; color:#00e676; font-weight:700;">TARGET ①</div>
                        <div style="font-size:12px; font-weight:800; color:#00e676; font-family:'JetBrains Mono',monospace;">₹${cand.target_1.toFixed(1)}</div>
                        <div style="font-size:8px; color:#00e676;">+100% (2×)</div>
                    </div>
                    <div style="background:rgba(0,240,255,0.06); padding:6px 4px; border-radius:5px; border:1px solid rgba(0,240,255,0.25);">
                        <div style="font-size:9px; color:#00f0ff; font-weight:700;">TARGET ②</div>
                        <div style="font-size:12px; font-weight:800; color:#00f0ff; font-family:'JetBrains Mono',monospace;">₹${cand.target_2.toFixed(1)}</div>
                        <div style="font-size:8px; color:#00f0ff;">+150% (2.5×)</div>
                    </div>
                </div>

                <!-- Greeks Strip (Real-time Greeks snapshot) -->
                <div style="background:rgba(0,0,0,0.3); border:1px solid rgba(255,255,255,0.06); border-radius:6px; padding:6px 10px; display:flex; justify-content:space-between; font-size:10px; font-family:'JetBrains Mono',monospace;">
                    <span title="Option Delta">Δ <b style="color:#ffffff;">${greeks.delta ?? '--'}</b></span>
                    <span title="Option Gamma">Γ <b style="color:#00f0ff;">${greeks.gamma ?? '--'}</b></span>
                    <span title="Option Theta / day">θ <b style="color:#ff7043;">${greeks.theta ?? '--'}</b></span>
                    <span title="Option Vega">V <b style="color:#a855f7;">${greeks.vega ?? '--'}</b></span>
                    <span title="Implied Volatility">IV <b style="color:#ffd54f;">${greeks.iv ? greeks.iv + '%' : '--'}</b></span>
                </div>

                <!-- Diagnostics & Trigger Reason -->
                <div style="font-size:10px; color:#94a3b8; line-height:1.4; background:rgba(255,255,255,0.02); padding:6px 8px; border-radius:4px;">
                    <div>⚡ <b>Trigger:</b> <span style="color:#e2e8f0;">${cand.ignition_source || 'Compression building'}</span></div>
                    <div style="margin-top:2px;">🛡️ <b>Invalidation:</b> <span style="color:#64748b;">${cand.invalidation || 'Spot reverses inside range'}</span></div>
                </div>

                <!-- Action Controls: Lot Size Selector + TRACK THIS Button -->
                <div style="display:flex; align-items:center; gap:8px; margin-top:2px;">
                    <select id="is-lots-${idx}" style="background:#0f172a; border:1px solid #334155; color:#ffffff; font-size:11px; font-weight:700; padding:6px 8px; border-radius:6px; cursor:pointer; outline:none;">
                        <option value="1">1 Lot (65 qty)</option>
                        <option value="2" selected>2 Lots (130 qty)</option>
                        <option value="3">3 Lots (195 qty)</option>
                        <option value="5">5 Lots (325 qty)</option>
                        <option value="10">10 Lots (650 qty)</option>
                    </select>

                    <button onclick="window.IgnitionScanner.trackCandidate('${candJson}', ${idx})" style="flex:1; background:linear-gradient(135deg, #00e676, #00b0ff); border:none; color:#0b0f19; font-size:11px; font-weight:900; padding:8px 12px; border-radius:6px; cursor:pointer; letter-spacing:0.5px; display:flex; align-items:center; justify-content:center; gap:6px; box-shadow:0 0 12px rgba(0,230,118,0.3); transition:opacity 0.2s ease;">
                        <span>🚀 TRACK THIS</span>
                    </button>
                </div>
            `;

            container.appendChild(card);
        });
    }

    // Deploy candidate to Strategy Builder Portfolio
    async function trackCandidate(candEncoded, lotSelectIdx) {
        try {
            const cand = JSON.parse(decodeURIComponent(candEncoded));
            const selectEl = document.getElementById(`is-lots-${lotSelectIdx}`);
            const lots = selectEl ? parseInt(selectEl.value, 10) : 1;

            const payload = {
                name: `Ignition: ${cand.strike} ${cand.type}`,
                source: 'IGNITION_SCANNER',
                legs: [{
                    action: 'BUY',
                    type: cand.type,
                    strike: cand.strike,
                    price: cand.entry_premium,
                    lots: lots,
                    iv: cand.greeks_at_entry?.iv || 15.0
                }],
                sl_premium: cand.sl_premium,
                target_premiums: [cand.target_1, cand.target_2],
                greeks_at_entry: cand.greeks_at_entry || {}
            };

            const res = await fetch('/api/portfolio/deploy', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });

            const result = await res.json();
            if (result.ok) {
                showToast(`Tracked ${cand.strike} ${cand.type} (${lots} lot${lots > 1 ? 's' : ''}) in Portfolio!`);
                // Trigger Portfolio refresh in Strategy Builder if loaded
                if (window.StrategyBuilder && typeof window.StrategyBuilder.fetchPortfolio === 'function') {
                    window.StrategyBuilder.fetchPortfolio();
                }
            } else {
                showToast(`Failed to deploy: ${result.error || 'Unknown error'}`, false);
            }
        } catch (e) {
            console.error('[IgnitionScanner] Track error:', e);
            showToast(`Error: ${e.message}`, false);
        }
    }

    // Generic helper for tracking any strike from other widgets (like Radar cards)
    async function trackGenericStrike(name, type, strike, price, sl, t1, t2, lots = 1, source = 'RADAR') {
        try {
            const payload = {
                name: name || `Radar: ${strike} ${type}`,
                source: source,
                legs: [{
                    action: 'BUY',
                    type: type,
                    strike: strike,
                    price: price,
                    lots: lots,
                    iv: 15.0
                }],
                sl_premium: sl,
                target_premiums: [t1, t2].filter(t => t != null && t > 0),
                greeks_at_entry: {}
            };

            const res = await fetch('/api/portfolio/deploy', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });

            const result = await res.json();
            if (result.ok) {
                showToast(`Tracked ${strike} ${type} (${lots} lot${lots > 1 ? 's' : ''}) in Portfolio!`);
                if (window.StrategyBuilder && typeof window.StrategyBuilder.fetchPortfolio === 'function') {
                    window.StrategyBuilder.fetchPortfolio();
                }
            } else {
                showToast(`Failed to deploy: ${result.error || 'Unknown error'}`, false);
            }
        } catch (e) {
            console.error('[IgnitionScanner] Track generic error:', e);
            showToast(`Error: ${e.message}`, false);
        }
    }

    async function refresh() {
        const data = await fetchCandidates();
        if (data) renderUI(data);
    }

    function init() {
        refresh();
        if (_pollTimer) clearInterval(_pollTimer);
        _pollTimer = setInterval(refresh, 6000);
    }

    function handleWsMessage(msg) {
        if (!msg) return;
        if (msg.type === 'ignition_update' && msg.payload) {
            renderUI(msg.payload);
        }
    }

    // Expose public API
    window.IgnitionScanner = {
        init: init,
        refresh: refresh,
        renderUI: renderUI,
        handleWsMessage: handleWsMessage,
        trackCandidate: trackCandidate,
        trackGenericStrike: trackGenericStrike
    };

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }
})();
