/**
 * ═══════════════════════════════════════════════════════════════════════════
 * F-INTEL QUANTITATIVE TERMINAL — STRATEGY BUILDER CONTROLLER
 * ═══════════════════════════════════════════════════════════════════════════
 * High-performance Sensibull-style Strategy Builder & Paper Trading Engine.
 * Self-contained & namespaced to prevent collisions with dashboard core.
 */

window.StrategyBuilder = (function () {
    'use strict';

    let currentSpot = 0;
    let optionChain = [];
    let strikesList = [];
    let activeLegs = [];
    let debounceTimer = null;
    let lastStrategyName = "Custom Strategy";
    let activePositionsDict = {};
    let sdDays = 14;
    let currentChainView = 'LTP';
    let initialized = false;

    const chartLayout = {
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        font: { color: '#868993', family: 'JetBrains Mono, monospace', size: 10 },
        margin: { t: 25, r: 20, l: 50, b: 35 },
        xaxis: {
            gridcolor: 'rgba(255, 255, 255, 0.04)',
            zerolinecolor: 'rgba(255, 255, 255, 0.15)',
            tickfont: { color: '#868993', size: 10 }
        },
        yaxis: {
            title: { text: "Profit / Loss (₹)", font: { color: "#00e5ff", size: 10 } },
            gridcolor: 'rgba(255, 255, 255, 0.04)',
            zerolinecolor: 'rgba(255, 255, 255, 0.2)',
            tickfont: { color: '#868993', size: 10 }
        },
        showlegend: true,
        legend: {
            orientation: 'h',
            yanchor: 'bottom',
            y: 1.02,
            xanchor: 'right',
            x: 1,
            bgcolor: 'rgba(15, 23, 42, 0.6)'
        },
        hovermode: 'x unified'
    };

    function init() {
        if (initialized) return;
        initialized = true;

        const chartDiv = document.getElementById('sb-payoff-chart');
        if (chartDiv && window.Plotly) {
            Plotly.newPlot(chartDiv, [], chartLayout, { responsive: true, displayModeBar: false });
        }

        setupEventListeners();
        fetchChain();
        fetchPortfolio();
        fetchWizardHistory();

        // Polling loop
        setInterval(function () {
            const tabSection = document.getElementById('tab-builder');
            if (tabSection && tabSection.classList.contains('active')) {
                fetchChain();
                fetchPortfolio();
            }
        }, 5000);
    }

    function onTabActivated() {
        if (!initialized) {
            init();
        }
        setTimeout(function () {
            const chartDiv = document.getElementById('sb-payoff-chart');
            if (chartDiv && window.Plotly && typeof window.Plotly.Plots.resize === 'function') {
                window.Plotly.Plots.resize(chartDiv);
            }
            fetchChain();
            fetchPortfolio();
        }, 60);
    }

    function setupEventListeners() {
        const slideSpot = document.getElementById('sb-slide-spot');
        if (slideSpot) {
            slideSpot.addEventListener('input', function (e) {
                const v = parseFloat(e.target.value);
                const tSpot = currentSpot * (1 + v / 100);
                const valEl = document.getElementById('sb-val-target-spot');
                if (valEl) {
                    valEl.innerText = (v > 0 ? '+' : '') + v.toFixed(1) + '% | ' + tSpot.toFixed(0);
                    valEl.style.color = v >= 0 ? '#00e676' : '#ff3366';
                }
                triggerAnalyze();
            });
        }

        const slideDate = document.getElementById('sb-slide-date');
        if (slideDate) {
            slideDate.addEventListener('input', function (e) {
                const daysOffset = parseInt(e.target.value);
                const targetDate = new Date(Date.now() + daysOffset * 86400000);
                const options = { weekday: 'short', month: 'short', day: 'numeric' };
                const valEl = document.getElementById('sb-val-target-date');
                if (valEl) {
                    valEl.innerText = `T+${daysOffset}D | ` + targetDate.toLocaleDateString('en-US', options);
                }
                triggerAnalyze();
            });
        }

        const slideTime = document.getElementById('sb-slide-time');
        if (slideTime) {
            slideTime.addEventListener('input', function (e) {
                const valEl = document.getElementById('sb-val-target-time');
                if (valEl) {
                    valEl.innerText = getTimeString(parseInt(e.target.value));
                }
                triggerAnalyze();
            });
        }
    }

    function getTimeString(ticks) {
        const totalMinutes = 9 * 60 + 15 + ticks * 15;
        const h = Math.floor(totalMinutes / 60);
        const m = totalMinutes % 60;
        return `${h.toString().padStart(2, '0')}:${m.toString().padStart(2, '0')}`;
    }

    function switchSubTab(tabId) {
        document.querySelectorAll('.sb-subtab-btn').forEach(b => b.classList.remove('active'));
        document.querySelectorAll('.sb-subtab-content').forEach(c => c.classList.remove('active'));

        const btn = document.querySelector(`.sb-subtab-btn[data-subtab="${tabId}"]`);
        if (btn) btn.classList.add('active');

        const content = document.getElementById(tabId);
        if (content) content.classList.add('active');
    }

    function setChainView(view) {
        currentChainView = view;
        const btnLtp = document.getElementById('sb-view-ltp');
        const btnGreeks = document.getElementById('sb-view-greeks');
        if (btnLtp) btnLtp.classList.toggle('active', view === 'LTP');
        if (btnGreeks) btnGreeks.classList.toggle('active', view === 'GREEKS');
        renderChain();
    }

    async function fetchChain() {
        try {
            const res = await fetch('/api/chain/live');
            const data = await res.json();
            if (data.ok) {
                currentSpot = data.spot || currentSpot;
                const spotEl = document.getElementById('sb-spot-display');
                if (spotEl) spotEl.innerText = currentSpot.toLocaleString('en-IN', { minimumFractionDigits: 2, maximumFractionDigits: 2 });

                optionChain = data.options || [];
                strikesList = [...new Set(optionChain.map(o => o.strike))].sort((a, b) => a - b);

                const expirySelect = document.getElementById('sb-expiry-select');
                if (expirySelect && (expirySelect.options.length === 0 || expirySelect.options[0].value !== data.expiry)) {
                    expirySelect.innerHTML = `<option value="${data.expiry}">${data.expiry}</option>`;
                }

                renderChain();
            }
        } catch (e) {
            console.error("StrategyBuilder fetchChain error:", e);
        }
    }

    function renderChain() {
        const tbody = document.getElementById('sb-chain-body');
        if (!tbody) return;
        tbody.innerHTML = '';

        const thead = document.querySelector('.sb-chain-table thead tr');
        if (thead) {
            if (currentChainView === 'LTP') {
                thead.innerHTML = '<th>Call OI</th><th>Call Δ</th><th>Call LTP</th><th class="sb-strike-col">Strike</th><th>Put LTP</th><th>Put Δ</th><th>Put OI</th>';
            } else {
                thead.innerHTML = '<th>Call Δ</th><th class="sb-strike-col">Strike</th><th>IV</th><th>Put Δ</th><th>Theta</th><th>Vega</th><th>Gamma</th>';
            }
        }

        let spotUpperStrikeIndex = strikesList.findIndex(s => s > currentSpot);
        if (spotUpperStrikeIndex === -1) spotUpperStrikeIndex = strikesList.length;

        strikesList.forEach((strike, index) => {
            const isAtm = (index === spotUpperStrikeIndex || (index > 0 && strikesList[index - 1] < currentSpot && strike >= currentSpot));
            const call = optionChain.find(o => o.strike === strike && o.type === 'CE') || {};
            const put = optionChain.find(o => o.strike === strike && o.type === 'PE') || {};

            const tr = document.createElement('tr');
            if (isAtm) tr.className = 'atm-row';

            if (currentChainView === 'LTP') {
                const callLtp = call.ltp || 0;
                const putLtp = put.ltp || 0;
                tr.innerHTML = `
                    <td style="color:#868993;">${(call.oi || 0).toLocaleString()}</td>
                    <td style="color:#00e676;">${(call.delta || 0).toFixed(2)}</td>
                    <td>
                        <span style="font-weight:700; color:#fff;">₹${callLtp.toFixed(2)}</span>
                        <button class="sb-quick-btn sb-quick-buy" onclick="StrategyBuilder.addLeg('BUY', ${strike}, 'CE', ${callLtp}, ${call.iv || 15})">+B</button>
                        <button class="sb-quick-btn sb-quick-sell" onclick="StrategyBuilder.addLeg('SELL', ${strike}, 'CE', ${callLtp}, ${call.iv || 15})">+S</button>
                    </td>
                    <td class="sb-strike-col">${strike}</td>
                    <td>
                        <span style="font-weight:700; color:#fff;">₹${putLtp.toFixed(2)}</span>
                        <button class="sb-quick-btn sb-quick-buy" onclick="StrategyBuilder.addLeg('BUY', ${strike}, 'PE', ${putLtp}, ${put.iv || 15})">+B</button>
                        <button class="sb-quick-btn sb-quick-sell" onclick="StrategyBuilder.addLeg('SELL', ${strike}, 'PE', ${putLtp}, ${put.iv || 15})">+S</button>
                    </td>
                    <td style="color:#ff3366;">${(put.delta || 0).toFixed(2)}</td>
                    <td style="color:#868993;">${(put.oi || 0).toLocaleString()}</td>
                `;
            } else {
                tr.innerHTML = `
                    <td style="color:#00e676;">${(call.delta || 0).toFixed(2)}</td>
                    <td class="sb-strike-col">${strike}</td>
                    <td style="color:#ffd54f;">${(call.iv || 0).toFixed(1)}%</td>
                    <td style="color:#ff3366;">${(put.delta || 0).toFixed(2)}</td>
                    <td style="color:#868993;">${(call.theta || 0).toFixed(2)}</td>
                    <td style="color:#868993;">${(call.vega || 0).toFixed(2)}</td>
                    <td style="color:#c084fc;">${(call.gamma || 0).toFixed(4)}</td>
                `;
            }
            tbody.appendChild(tr);
        });
    }

    function addLeg(action, strike, optType, price, iv) {
        const expirySelect = document.getElementById('sb-expiry-select');
        const expiry = expirySelect ? expirySelect.value : 'LIVE';
        activeLegs.push({
            id: Date.now() + Math.random().toString(36).substr(2, 4),
            action: action,
            lots: 1,
            expiry: expiry,
            strike: strike,
            opt_type: optType,
            price: price || 100,
            iv: iv || 15
        });
        renderBasket();
        triggerAnalyze();
    }

    function removeLeg(index) {
        activeLegs.splice(index, 1);
        renderBasket();
        triggerAnalyze();
    }

    function toggleLegAction(index) {
        if (!activeLegs[index]) return;
        activeLegs[index].action = activeLegs[index].action === 'BUY' ? 'SELL' : 'BUY';
        renderBasket();
        triggerAnalyze();
    }

    function updateLegLots(index, delta) {
        if (!activeLegs[index]) return;
        activeLegs[index].lots = Math.max(1, activeLegs[index].lots + delta);
        renderBasket();
        triggerAnalyze();
    }

    function clearBasket() {
        activeLegs = [];
        lastStrategyName = "Custom Strategy";
        renderBasket();
        triggerAnalyze();
    }

    function renderBasket() {
        const tbody = document.getElementById('sb-basket-body');
        if (!tbody) return;
        tbody.innerHTML = '';

        if (activeLegs.length === 0) {
            tbody.innerHTML = '<tr><td colspan="8" style="text-align:center; color:#868993; padding: 24px 0;">No active legs. Click preset on left or +B / +S from option chain below.</td></tr>';
            return;
        }

        activeLegs.forEach((leg, index) => {
            const tr = document.createElement('tr');
            tr.innerHTML = `
                <td>
                    <div class="sb-action-btn-group">
                        <button class="${leg.action === 'BUY' ? 'active buy' : ''}" onclick="StrategyBuilder.toggleLegAction(${index})">BUY</button>
                        <button class="${leg.action === 'SELL' ? 'active sell' : ''}" onclick="StrategyBuilder.toggleLegAction(${index})">SELL</button>
                    </div>
                </td>
                <td>
                    <div style="display:inline-flex; align-items:center; gap:6px;">
                        <button style="background:#131722; border:1px solid #2a2e39; color:#fff; width:22px; height:22px; border-radius:3px; cursor:pointer;" onclick="StrategyBuilder.updateLegLots(${index}, -1)">-</button>
                        <span style="font-family:var(--font-mono); font-weight:700;">${leg.lots}</span>
                        <button style="background:#131722; border:1px solid #2a2e39; color:#fff; width:22px; height:22px; border-radius:3px; cursor:pointer;" onclick="StrategyBuilder.updateLegLots(${index}, 1)">+</button>
                    </div>
                </td>
                <td style="color:#868993;">${leg.expiry}</td>
                <td style="font-weight:800; color:#fff;">${leg.strike}</td>
                <td style="font-weight:700; color:${leg.opt_type === 'CE' ? '#00e676' : '#ff3366'}">${leg.opt_type}</td>
                <td style="font-family:var(--font-mono);">₹${leg.price.toFixed(2)}</td>
                <td style="color:#ffd54f;">${leg.iv.toFixed(1)}%</td>
                <td>
                    <button style="background:transparent; border:none; color:#ff3366; cursor:pointer; font-weight:bold; font-size:14px;" onclick="StrategyBuilder.removeLeg(${index})">✕</button>
                </td>
            `;
            tbody.appendChild(tr);
        });
    }

    function buildStrategy(name) {
        if (!strikesList.length) return;
        activeLegs = [];
        lastStrategyName = name;

        const atmStrike = strikesList.reduce((prev, curr) => Math.abs(curr - currentSpot) < Math.abs(prev - currentSpot) ? curr : prev);
        const atmIndex = strikesList.indexOf(atmStrike);
        const step = strikesList.length > 1 ? (strikesList[1] - strikesList[0]) : 50;

        function getOption(strike, type) {
            return optionChain.find(o => o.strike === strike && o.type === type) || { ltp: 100, iv: 15 };
        }

        if (name === 'Bull Call Spread') {
            const buyStrike = atmStrike;
            const sellStrike = strikesList[Math.min(strikesList.length - 1, atmIndex + 2)];
            addLeg('BUY', buyStrike, 'CE', getOption(buyStrike, 'CE').ltp, getOption(buyStrike, 'CE').iv);
            addLeg('SELL', sellStrike, 'CE', getOption(sellStrike, 'CE').ltp, getOption(sellStrike, 'CE').iv);
        } else if (name === 'Bull Put Spread') {
            const sellStrike = atmStrike;
            const buyStrike = strikesList[Math.max(0, atmIndex - 2)];
            addLeg('SELL', sellStrike, 'PE', getOption(sellStrike, 'PE').ltp, getOption(sellStrike, 'PE').iv);
            addLeg('BUY', buyStrike, 'PE', getOption(buyStrike, 'PE').ltp, getOption(buyStrike, 'PE').iv);
        } else if (name === 'Bear Put Spread') {
            const buyStrike = atmStrike;
            const sellStrike = strikesList[Math.max(0, atmIndex - 2)];
            addLeg('BUY', buyStrike, 'PE', getOption(buyStrike, 'PE').ltp, getOption(buyStrike, 'PE').iv);
            addLeg('SELL', sellStrike, 'PE', getOption(sellStrike, 'PE').ltp, getOption(sellStrike, 'PE').iv);
        } else if (name === 'Bear Call Spread') {
            const sellStrike = atmStrike;
            const buyStrike = strikesList[Math.min(strikesList.length - 1, atmIndex + 2)];
            addLeg('SELL', sellStrike, 'CE', getOption(sellStrike, 'CE').ltp, getOption(sellStrike, 'CE').iv);
            addLeg('BUY', buyStrike, 'CE', getOption(buyStrike, 'CE').ltp, getOption(buyStrike, 'CE').iv);
        } else if (name === 'Short Straddle') {
            addLeg('SELL', atmStrike, 'CE', getOption(atmStrike, 'CE').ltp, getOption(atmStrike, 'CE').iv);
            addLeg('SELL', atmStrike, 'PE', getOption(atmStrike, 'PE').ltp, getOption(atmStrike, 'PE').iv);
        } else if (name === 'Short Strangle') {
            const callStrike = strikesList[Math.min(strikesList.length - 1, atmIndex + 2)];
            const putStrike = strikesList[Math.max(0, atmIndex - 2)];
            addLeg('SELL', callStrike, 'CE', getOption(callStrike, 'CE').ltp, getOption(callStrike, 'CE').iv);
            addLeg('SELL', putStrike, 'PE', getOption(putStrike, 'PE').ltp, getOption(putStrike, 'PE').iv);
        } else if (name === 'Iron Condor') {
            const sellCall = strikesList[Math.min(strikesList.length - 1, atmIndex + 2)];
            const buyCall = strikesList[Math.min(strikesList.length - 1, atmIndex + 4)];
            const sellPut = strikesList[Math.max(0, atmIndex - 2)];
            const buyPut = strikesList[Math.max(0, atmIndex - 4)];
            addLeg('SELL', sellCall, 'CE', getOption(sellCall, 'CE').ltp, getOption(sellCall, 'CE').iv);
            addLeg('BUY', buyCall, 'CE', getOption(buyCall, 'CE').ltp, getOption(buyCall, 'CE').iv);
            addLeg('SELL', sellPut, 'PE', getOption(sellPut, 'PE').ltp, getOption(sellPut, 'PE').iv);
            addLeg('BUY', buyPut, 'PE', getOption(buyPut, 'PE').ltp, getOption(buyPut, 'PE').iv);
        } else if (name === 'Iron Butterfly') {
            const wingBuyCall = strikesList[Math.min(strikesList.length - 1, atmIndex + 2)];
            const wingBuyPut = strikesList[Math.max(0, atmIndex - 2)];
            addLeg('SELL', atmStrike, 'CE', getOption(atmStrike, 'CE').ltp, getOption(atmStrike, 'CE').iv);
            addLeg('SELL', atmStrike, 'PE', getOption(atmStrike, 'PE').ltp, getOption(atmStrike, 'PE').iv);
            addLeg('BUY', wingBuyCall, 'CE', getOption(wingBuyCall, 'CE').ltp, getOption(wingBuyCall, 'CE').iv);
            addLeg('BUY', wingBuyPut, 'PE', getOption(wingBuyPut, 'PE').ltp, getOption(wingBuyPut, 'PE').iv);
        } else if (name === 'Call Ratio Backspread') {
            const sellStrike = atmStrike;
            const buyStrike = strikesList[Math.min(strikesList.length - 1, atmIndex + 2)];
            addLeg('SELL', sellStrike, 'CE', getOption(sellStrike, 'CE').ltp, getOption(sellStrike, 'CE').iv);
            addLeg('BUY', buyStrike, 'CE', getOption(buyStrike, 'CE').ltp, getOption(buyStrike, 'CE').iv);
            activeLegs[1].lots = 2;
        } else if (name === 'Put Ratio Backspread') {
            const sellStrike = atmStrike;
            const buyStrike = strikesList[Math.max(0, atmIndex - 2)];
            addLeg('SELL', sellStrike, 'PE', getOption(sellStrike, 'PE').ltp, getOption(sellStrike, 'PE').iv);
            addLeg('BUY', buyStrike, 'PE', getOption(buyStrike, 'PE').ltp, getOption(buyStrike, 'PE').iv);
            activeLegs[1].lots = 2;
        }

        const nameEl = document.getElementById('sb-summary-strategy-name');
        if (nameEl) nameEl.innerText = name;
        renderBasket();
        triggerAnalyze();
    }

    function triggerAnalyze() {
        clearTimeout(debounceTimer);
        debounceTimer = setTimeout(analyzeStrategy, 150);
    }

    async function analyzeStrategy() {
        const slideSpot = document.getElementById('sb-slide-spot');
        const slideDate = document.getElementById('sb-slide-date');
        const slideTime = document.getElementById('sb-slide-time');

        const spotShift = slideSpot ? parseFloat(slideSpot.value) : 0;
        const daysOffset = slideDate ? parseFloat(slideDate.value) : 0;
        const timeTicks = slideTime ? parseInt(slideTime.value) : 25;

        const fraction = 1 - (timeTicks / 25.0);
        const totalFractionalDays = daysOffset + fraction;

        const chartDiv = document.getElementById('sb-payoff-chart');

        if (activeLegs.length === 0) {
            if (chartDiv && window.Plotly) {
                Plotly.react(chartDiv, [], chartLayout);
            }
            updateSummary({ max_profit: 0, max_loss: 0, breakevens: [], pop: 0, net_premium: 0 }, { delta: 0, gamma: 0, theta: 0, vega: 0 });
            return;
        }

        try {
            const res = await fetch('/api/builder/analyze', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    legs: activeLegs,
                    spot_shift_pct: spotShift,
                    target_days_offset: totalFractionalDays
                })
            });
            const data = await res.json();
            if (data.ok) {
                updateSummary(data.summary, data.greeks);
                renderChart(data.chart_data);
                const projEl = document.getElementById('sb-projected-profit');
                if (projEl) {
                    const sign = data.projected_pnl >= 0 ? '+' : '';
                    projEl.innerText = `${sign}₹${data.projected_pnl.toLocaleString('en-IN', { maximumFractionDigits: 0 })}`;
                    projEl.style.color = data.projected_pnl >= 0 ? '#00e676' : '#ff3366';
                }
            }
        } catch (e) {
            console.error("StrategyBuilder analyzeStrategy error:", e);
        }
    }

    function updateSummary(summary, greeks) {
        const maxP = document.getElementById('sb-stat-max-profit');
        const maxL = document.getElementById('sb-stat-max-loss');
        const popEl = document.getElementById('sb-stat-pop');
        const premEl = document.getElementById('sb-stat-premium');
        const beEl = document.getElementById('sb-stat-breakevens');

        if (maxP) {
            maxP.innerText = summary.max_profit > 1e6 ? 'Unlimited' : `₹${Math.round(summary.max_profit).toLocaleString('en-IN')}`;
            maxP.style.color = '#00e676';
        }
        if (maxL) {
            maxL.innerText = summary.max_loss < -1e6 ? 'Unlimited' : `₹${Math.round(Math.abs(summary.max_loss)).toLocaleString('en-IN')}`;
            maxL.style.color = '#ff3366';
        }
        if (popEl) popEl.innerText = `${(summary.pop || 0).toFixed(1)}%`;
        if (premEl) {
            const sign = summary.net_premium >= 0 ? '+' : '-';
            premEl.innerText = `${sign}₹${Math.round(Math.abs(summary.net_premium)).toLocaleString('en-IN')}`;
        }
        if (beEl) {
            beEl.innerText = (summary.breakevens && summary.breakevens.length) ? summary.breakevens.map(b => b.toFixed(0)).join(', ') : '-';
        }

        const dEl = document.getElementById('sb-greek-delta');
        const tEl = document.getElementById('sb-greek-theta');
        const gEl = document.getElementById('sb-greek-gamma');
        const vEl = document.getElementById('sb-greek-vega');

        if (dEl) dEl.innerText = (greeks.delta || 0).toFixed(2);
        if (tEl) tEl.innerText = (greeks.theta || 0).toFixed(2);
        if (gEl) gEl.innerText = (greeks.gamma || 0).toFixed(4);
        if (vEl) vEl.innerText = (greeks.vega || 0).toFixed(2);
    }

    function renderChart(cData) {
        const chartDiv = document.getElementById('sb-payoff-chart');
        if (!chartDiv || !window.Plotly) return;

        const traceExpiry = {
            x: cData.spots,
            y: cData.pnl_expiry,
            mode: 'lines',
            name: 'At Expiry P&L',
            line: { color: '#64748b', width: 2, dash: 'dot' }
        };

        const traceTarget = {
            x: cData.spots,
            y: cData.pnl_target,
            mode: 'lines',
            name: 'Target Date P&L',
            line: { color: '#00e5ff', width: 2.5 }
        };

        const atmIv = 0.14;
        const tAnnual = sdDays / 365.0;
        const sd1 = currentSpot * atmIv * Math.sqrt(tAnnual);

        const shapes = [
            { type: 'line', x0: currentSpot, x1: currentSpot, y0: 0, y1: 1, yref: 'paper', line: { color: 'rgba(255, 255, 255, 0.4)', width: 1.5, dash: 'dash' } },
            { type: 'line', x0: currentSpot - sd1, x1: currentSpot - sd1, y0: 0, y1: 1, yref: 'paper', line: { color: 'rgba(0, 230, 118, 0.5)', width: 1, dash: 'dash' } },
            { type: 'line', x0: currentSpot + sd1, x1: currentSpot + sd1, y0: 0, y1: 1, yref: 'paper', line: { color: 'rgba(0, 230, 118, 0.5)', width: 1, dash: 'dash' } }
        ];

        const annotations = [
            { x: currentSpot, y: 0.05, yanchor: 'bottom', yref: 'paper', text: `Spot ${currentSpot.toFixed(0)}`, showarrow: false, font: { size: 10, color: '#ffffff' } },
            { x: currentSpot - sd1, y: 1.0, yanchor: 'bottom', yref: 'paper', text: '-1σ', showarrow: false, font: { size: 10, color: '#00e676' } },
            { x: currentSpot + sd1, y: 1.0, yanchor: 'bottom', yref: 'paper', text: '+1σ', showarrow: false, font: { size: 10, color: '#00e676' } }
        ];

        const layout = { ...chartLayout, shapes: shapes, annotations: annotations };
        Plotly.react(chartDiv, [traceExpiry, traceTarget], layout);
    }

    async function deployStrategy() {
        if (!activeLegs.length) {
            alert("No legs to deploy!");
            return;
        }
        try {
            const res = await fetch('/api/portfolio/deploy', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    name: lastStrategyName,
                    legs: activeLegs
                })
            });
            const data = await res.json();
            if (data.ok) {
                alert(`Strategy '${lastStrategyName}' deployed successfully as paper trade!`);
                switchSubTab('sb-subtab-portfolio');
                fetchPortfolio();
            } else {
                alert(`Deploy failed: ${data.error}`);
            }
        } catch (e) {
            alert(`Deploy error: ${e}`);
        }
    }

    async function fetchPortfolio() {
        try {
            const res = await fetch('/api/portfolio/status');
            const data = await res.json();
            if (data.ok) {
                const active = (data.portfolio && data.portfolio.active) ? data.portfolio.active : (data.active || []);
                renderPortfolio(active);
            }
        } catch (e) {
            console.error("StrategyBuilder fetchPortfolio error:", e);
        }
    }

    function renderPortfolio(active) {
        const container = document.getElementById('sb-active-positions-container');
        const badge = document.getElementById('sb-pos-badge');
        if (badge) {
            badge.innerText = active.length;
            badge.style.display = active.length > 0 ? 'inline-block' : 'none';
        }
        if (!container) return;
        container.innerHTML = '';

        if (active.length === 0) {
            container.innerHTML = '<div style="color:#868993; font-size:12px; text-align:center; padding: 24px 0;">No active paper positions. Build and click Deploy or track from Ignition Scanner.</div>';
            return;
        }

        active.forEach(pos => {
            const pnl = pos.live_pnl || 0;
            const pnlPct = pos.pnl_pct != null ? pos.pnl_pct.toFixed(1) : '0.0';
            const colorClass = pnl >= 0 ? '#00e676' : '#ff3366';
            const sign = pnl >= 0 ? '+' : '';

            // Source Badge
            const src = pos.source || 'MANUAL';
            let srcColor = '#38bdf8';
            let srcBg = 'rgba(56,189,248,0.12)';
            let srcBorder = 'rgba(56,189,248,0.3)';
            if (src === 'IGNITION_SCANNER') {
                srcColor = '#ffd54f';
                srcBg = 'rgba(255,213,79,0.15)';
                srcBorder = '#ffd54f';
            } else if (src === 'RADAR') {
                srcColor = '#a855f7';
                srcBg = 'rgba(168,85,247,0.15)';
                srcBorder = '#a855f7';
            }

            // SL and Targets
            let slBadgeHtml = '';
            if (pos.sl_premium != null) {
                const isHit = !!pos.sl_hit;
                slBadgeHtml += `
                    <span style="font-size:10px; font-family:'JetBrains Mono',monospace; padding:2px 6px; border-radius:4px; ${isHit ? 'background:#ff3366; color:#fff; font-weight:900;' : 'background:rgba(255,51,102,0.12); color:#ff5252; border:1px solid rgba(255,51,102,0.3);'}">
                        ${isHit ? '⚠️ SL HIT: ₹' + pos.sl_premium.toFixed(1) : 'SL: ₹' + pos.sl_premium.toFixed(1)}
                    </span>
                `;
            }
            if (pos.target_premiums && pos.target_premiums.length > 0) {
                const t1 = pos.target_premiums[0];
                const t1Hit = !!pos.target_1_hit;
                slBadgeHtml += `
                    <span style="font-size:10px; font-family:'JetBrains Mono',monospace; padding:2px 6px; border-radius:4px; ${t1Hit ? 'background:#00e676; color:#0b0f19; font-weight:900;' : 'background:rgba(0,230,118,0.12); color:#00e676; border:1px solid rgba(0,230,118,0.3);'}">
                        ${t1Hit ? '✅ T1: ₹' + t1.toFixed(1) : 'T1: ₹' + t1.toFixed(1)}
                    </span>
                `;
                if (pos.target_premiums.length > 1) {
                    const t2 = pos.target_premiums[1];
                    const t2Hit = !!pos.target_2_hit;
                    slBadgeHtml += `
                        <span style="font-size:10px; font-family:'JetBrains Mono',monospace; padding:2px 6px; border-radius:4px; ${t2Hit ? 'background:#00f0ff; color:#0b0f19; font-weight:900;' : 'background:rgba(0,240,255,0.12); color:#00f0ff; border:1px solid rgba(0,240,255,0.3);'}">
                            ${t2Hit ? '🎯 T2: ₹' + t2.toFixed(1) : 'T2: ₹' + t2.toFixed(1)}
                        </span>
                    `;
                }
            }

            // Greeks Waterfall Strip
            const lg = pos.live_greeks || {};
            const eg = pos.greeks_at_entry || {};
            let greeksHtml = '';
            if (lg && (lg.delta !== undefined || lg.gamma !== undefined)) {
                let deltaChangeStr = '';
                if (pos.delta_change_pct !== undefined && pos.delta_change_pct !== 0) {
                    const dSign = pos.delta_change_pct >= 0 ? '+' : '';
                    deltaChangeStr = ` (${dSign}${pos.delta_change_pct}%)`;
                }

                greeksHtml = `
                    <div style="background:rgba(0,0,0,0.35); border:1px solid rgba(255,255,255,0.06); border-radius:6px; padding:6px 10px; margin-top:8px; display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:8px; font-size:10px; font-family:'JetBrains Mono',monospace;">
                        <span title="Net Delta (Live vs Entry)">
                            Δ <b style="color:#ffffff;">${lg.delta ?? '--'}</b>
                            ${eg.delta !== undefined ? `<span style="color:#64748b; font-size:9px;"> (entry ${eg.delta}${deltaChangeStr})</span>` : ''}
                        </span>
                        <span title="Net Gamma">Γ <b style="color:#00f0ff;">${lg.gamma ?? '--'}</b></span>
                        <span title="Net Theta / day">θ <b style="color:#ff7043;">${lg.theta ?? '--'}</b></span>
                        <span title="Net Vega">V <b style="color:#a855f7;">${lg.vega ?? '--'}</b></span>
                    </div>
                `;
            }

            // Legs breakdown
            let legsHtml = '';
            if (pos.legs && pos.legs.length > 0) {
                legsHtml = pos.legs.map(l => {
                    const legPnl = l.live_pnl || 0;
                    const legColor = legPnl >= 0 ? '#00e676' : '#ff3366';
                    return `
                        <div style="display:flex; justify-content:space-between; font-size:10px; color:#94a3b8; font-family:'JetBrains Mono',monospace; padding:2px 0;">
                            <span>${l.action} ${l.lots}× ${l.strike} ${l.type} @ ₹${l.price.toFixed(1)} &rarr; ₹${(l.live_price || l.price).toFixed(1)}</span>
                            <span style="color:${legColor}; font-weight:700;">${legPnl >= 0 ? '+' : ''}₹${legPnl.toFixed(1)}</span>
                        </div>
                    `;
                }).join('');
            }

            const div = document.createElement('div');
            div.className = 'sb-pos-card';
            div.style.cssText = `
                background: #141824;
                border: 1px solid ${pos.sl_hit ? '#ff3366' : '#222744'};
                border-radius: 8px;
                padding: 12px;
                margin-bottom: 10px;
                box-shadow: 0 4px 12px rgba(0,0,0,0.25);
            `;

            div.innerHTML = `
                <!-- Header -->
                <div style="display:flex; justify-content:space-between; align-items:flex-start;">
                    <div>
                        <div style="display:flex; align-items:center; gap:6px;">
                            <span style="font-weight:700; color:#fff; font-size:13px;">${pos.name}</span>
                            <span style="font-size:9px; font-weight:800; padding:2px 6px; border-radius:4px; background:${srcBg}; color:${srcColor}; border:1px solid ${srcBorder};">
                                ${src.replace('_', ' ')}
                            </span>
                        </div>
                        <div style="font-size:10px; color:#868993; margin-top:2px;">
                            Entered: ${(pos.entry_time || '').split(' ')[1] || ''} · Margin: ₹${Math.round(pos.estimated_margin || 0).toLocaleString()}
                        </div>
                    </div>
                    <div style="text-align:right;">
                        <div style="font-family:var(--font-mono); font-weight:900; font-size:15px; color:${colorClass};">
                            ${sign}₹${pnl.toFixed(2)} <span style="font-size:11px;">(${sign}${pnlPct}%)</span>
                        </div>
                        <div style="font-size:9px; color:#868993;">Live P&L</div>
                    </div>
                </div>

                <!-- SL & Targets -->
                ${slBadgeHtml ? `<div style="display:flex; gap:6px; flex-wrap:wrap; margin-top:8px;">${slBadgeHtml}</div>` : ''}

                <!-- Legs List -->
                ${legsHtml ? `<div style="margin-top:8px; border-top:1px dashed #222744; padding-top:6px;">${legsHtml}</div>` : ''}

                <!-- Greeks Waterfall -->
                ${greeksHtml}

                <!-- Footer Actions -->
                <div style="display:flex; justify-content:space-between; align-items:center; margin-top:10px; padding-top:6px; border-top:1px solid rgba(255,255,255,0.05);">
                    <div style="font-size:10px; color:#868993;">
                        ${pos.live_pop ? `POP: <b style="color:#00e676;">${pos.live_pop.toFixed(1)}%</b>` : ''}
                    </div>
                    <button style="background:rgba(255,51,102,0.15); border:1px solid #ff3366; color:#ff3366; padding:5px 12px; border-radius:4px; font-size:10px; font-weight:700; cursor:pointer;" onclick="StrategyBuilder.exitPosition('${pos.id}')">
                        Exit Position
                    </button>
                </div>
            `;
            container.appendChild(div);
        });
    }

    async function exitPosition(id) {
        try {
            const res = await fetch('/api/portfolio/exit', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ id: id })
            });
            const data = await res.json();
            if (data.ok) {
                alert(`Position exited! Realized P&L: ₹${(data.position.realized_pnl || 0).toFixed(2)}`);
                fetchPortfolio();
            }
        } catch (e) {
            console.error("StrategyBuilder exitPosition error:", e);
        }
    }

    async function fetchWizardRecommendation() {
        const btn = document.getElementById('sb-btn-fetch-wizard');
        if (btn) btn.innerText = "Analyzing Live Market...";
        try {
            const res = await fetch('/api/wizard', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ view: 'NEUTRAL', risk: 'MODERATE', capital: 150000 })
            });
            const data = await res.json();
            if (data.ok) {
                fetchWizardHistory();
            }
        } catch (e) {
            console.error("StrategyBuilder fetchWizardRecommendation error:", e);
        } finally {
            if (btn) btn.innerText = "⚡ Suggest AI Strategy";
        }
    }

    async function fetchWizardHistory() {
        try {
            const res = await fetch('/api/wizard/history');
            const data = await res.json();
            if (data.ok) {
                const cont = document.getElementById('sb-ai-presets-container');
                if (!cont) return;
                cont.innerHTML = '';
                if (!data.history || data.history.length === 0) {
                    cont.innerHTML = '<div style="color:#868993; font-size:11px; text-align:center; padding: 18px 0;">No AI recommendations yet today. Click Suggest button above.</div>';
                    return;
                }
                data.history.slice().reverse().forEach(s => {
                    const div = document.createElement('div');
                    div.className = 'sb-ai-card';
                    div.innerHTML = `
                        <div style="font-size:10px; color:#868993;">${s.time}</div>
                        <div style="font-size:12px; font-weight:800; color:#00e5ff;">${s.strategy}</div>
                        <div style="font-size:11px; color:#d1d4dc; font-style:italic;">"${s.rationale}"</div>
                    `;
                    div.onclick = function () {
                        lastStrategyName = s.strategy;
                        activeLegs = [];
                        (s.legs || []).forEach(l => addLeg(l.action, l.strike, l.opt_type, l.entry_price, l.iv));
                        renderBasket();
                        triggerAnalyze();
                        switchSubTab('sb-subtab-ready');
                    };
                    cont.appendChild(div);
                });
            }
        } catch (e) {
            console.error("StrategyBuilder fetchWizardHistory error:", e);
        }
    }

    function openOrderbook() {
        fetch('/api/portfolio/history')
            .then(res => res.json())
            .then(data => {
                if (data.ok) {
                    const tbody = document.getElementById('sb-history-body');
                    if (!tbody) return;
                    tbody.innerHTML = '';
                    const history = data.history || [];
                    if (history.length === 0) {
                        tbody.innerHTML = '<tr><td colspan="4" style="text-align:center; padding: 20px; color:#868993;">No exited positions recorded.</td></tr>';
                    } else {
                        history.slice().reverse().forEach(pos => {
                            const pnl = pos.realized_pnl || 0;
                            const color = pnl >= 0 ? '#00e676' : '#ff3366';
                            const tr = document.createElement('tr');
                            tr.innerHTML = `
                                <td>${pos.exit_time || '-'}</td>
                                <td style="font-weight:700; color:#fff;">${pos.name}</td>
                                <td>₹${Math.round(pos.estimated_margin || 0).toLocaleString()}</td>
                                <td style="font-weight:800; color:${color}; font-family:var(--font-mono);">${pnl >= 0 ? '+' : ''}₹${pnl.toFixed(2)}</td>
                            `;
                            tbody.appendChild(tr);
                        });
                    }
                    const modal = document.getElementById('sb-orderbook-modal');
                    if (modal) modal.style.display = 'flex';
                }
            });
    }

    function closeOrderbook() {
        const modal = document.getElementById('sb-orderbook-modal');
        if (modal) modal.style.display = 'none';
    }

    return {
        init: init,
        onTabActivated: onTabActivated,
        switchSubTab: switchSubTab,
        setChainView: setChainView,
        addLeg: addLeg,
        removeLeg: removeLeg,
        toggleLegAction: toggleLegAction,
        updateLegLots: updateLegLots,
        clearBasket: clearBasket,
        buildStrategy: buildStrategy,
        deployStrategy: deployStrategy,
        exitPosition: exitPosition,
        fetchPortfolio: fetchPortfolio,
        fetchWizardRecommendation: fetchWizardRecommendation,
        openOrderbook: openOrderbook,
        closeOrderbook: closeOrderbook
    };
})();
