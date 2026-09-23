/**
 * static/js/backtest_studio.js
 * =============================
 * Interactive Backtest Studio & GEX Move Forecaster frontend logic.
 * Handles running backtests, rendering Plotly charts, and displaying the granular trade log.
 */

let currentBacktestData = null;
let currentGexData = null;

function initBacktestStudio() {
    console.log("[BacktestStudio] Initialized.");
}

async function runStudioBacktest() {
    const stratSelect = document.getElementById('bt-strategy-select');
    const daysSelect = document.getElementById('bt-days-select');
    const targetSelect = document.getElementById('bt-target-select');
    const slSelect = document.getElementById('bt-sl-select');
    const confSelect = document.getElementById('bt-conf-select');
    const lotsInput = document.getElementById('bt-lots-input');
    const runBtn = document.getElementById('btn-run-backtest');
    const statusMsg = document.getElementById('bt-status-msg');

    const strategy = stratSelect ? stratSelect.value : 'RADAR_ATM';
    const days = daysSelect ? parseInt(daysSelect.value) : 365;
    const target = targetSelect ? parseFloat(targetSelect.value) : 0.80;
    const sl = slSelect ? parseFloat(slSelect.value) : 0.40;
    const confluence = confSelect ? parseFloat(confSelect.value) : 50.0;
    const lots = lotsInput ? parseInt(lotsInput.value) : 1;

    if (runBtn) {
        runBtn.disabled = true;
        runBtn.innerHTML = `<span class="spinner" style="display:inline-block;width:14px;height:14px;border:2px solid rgba(255,255,255,0.3);border-top-color:#fff;border-radius:50%;animation:spin 0.8s linear infinite;margin-right:6px;"></span> Running...`;
    }
    if (statusMsg) statusMsg.textContent = "Simulating historical trades across Bhavcopy data...";

    try {
        if (strategy === 'GEX_MOVES') {
            const resp = await fetch(`/api/backtest/gex-moves?days=${days}`);
            const data = await resp.json();
            if (data.ok) {
                currentGexData = data;
                renderGexMoveResults(data);
                if (statusMsg) statusMsg.textContent = `Completed: Evaluated ${data.summary.total_days_evaluated} trading days.`;
            } else {
                alert("GEX Backtest failed: " + (data.error || "Unknown error"));
            }
        } else {
            const resp = await fetch('/api/backtest/run', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ strategy, days, sl, target, confluence, lots })
            });
            const data = await resp.json();
            if (data.ok) {
                currentBacktestData = data;
                renderStrategyBacktestResults(data);
                if (statusMsg) statusMsg.textContent = `Completed: Simulated ${data.summary.total_trades} trades across ${days} days.`;
            } else {
                alert("Strategy Backtest failed: " + (data.error || "Unknown error"));
            }
        }
    } catch (err) {
        console.error("Backtest execution error:", err);
        alert("Backtest failed to execute: " + err);
    } finally {
        if (runBtn) {
            runBtn.disabled = false;
            runBtn.innerHTML = `<span>Run Backtest 🚀</span>`;
        }
    }
}

function renderStrategyBacktestResults(data) {
    const s = data.summary;
    document.getElementById('bt-metric-net-pnl').textContent = formatInr(s.total_net_pnl);
    document.getElementById('bt-metric-net-pnl').className = s.total_net_pnl >= 0 ? 'stat-value text-green' : 'stat-value text-red';

    document.getElementById('bt-metric-win-rate').textContent = `${s.win_rate_pct}%`;
    document.getElementById('bt-metric-profit-factor').textContent = s.profit_factor.toFixed(2);
    document.getElementById('bt-metric-max-dd').textContent = formatInr(s.max_drawdown);
    document.getElementById('bt-metric-sharpe').textContent = s.sharpe_ratio.toFixed(2);
    document.getElementById('bt-metric-trades').textContent = s.total_trades;

    document.getElementById('bt-target-hits-sub').textContent = `${s.target_1_hits} Target / ${s.stop_loss_hits} SL / ${s.expiry_settlements} Expiry`;

    // Render Equity Curve via Plotly
    renderEquityCurve(data.equity_curve, s.strategy);

    // Render Granular Trade Table
    renderTradeTable(data.trades);

    // Show strategy panels, hide GEX panels
    document.getElementById('bt-strategy-panels').style.display = 'block';
    document.getElementById('bt-gex-panels').style.display = 'none';
    document.getElementById('btn-export-csv').style.display = 'inline-flex';
}

function renderGexMoveResults(data) {
    const s = data.summary;
    document.getElementById('bt-metric-net-pnl').textContent = `${s.avg_next_day_range_pts} pts`;
    document.getElementById('bt-metric-net-pnl').className = 'stat-value text-green';
    document.getElementById('bt-net-pnl-label').textContent = 'Avg Daily Range';

    document.getElementById('bt-metric-win-rate').textContent = `${s.tier_accuracy_pct}%`;
    document.getElementById('bt-win-rate-label').textContent = 'Tier Accuracy';

    document.getElementById('bt-metric-profit-factor').textContent = `${s.range_hit_rate_pct}%`;
    document.getElementById('bt-pf-label').textContent = 'Range Hit Rate';

    document.getElementById('bt-metric-max-dd').textContent = `${s.big_move_days_count} days`;
    document.getElementById('bt-max-dd-label').textContent = 'Big Move Days';

    document.getElementById('bt-metric-sharpe').textContent = `${s.big_move_realization_rate}%`;
    document.getElementById('bt-sharpe-label').textContent = 'Big Move Realization';

    document.getElementById('bt-metric-trades').textContent = s.total_days_evaluated;
    document.getElementById('bt-trades-label').textContent = 'Days Evaluated';

    document.getElementById('bt-target-hits-sub').textContent = `Correlation: ${s.net_gex_range_correlation}`;

    // Render Scatter Chart
    renderGexScatterChart(data.scatter_points);

    // Render GEX Day Log Table
    renderGexRecordsTable(data.records);

    document.getElementById('bt-strategy-panels').style.display = 'none';
    document.getElementById('bt-gex-panels').style.display = 'block';
    document.getElementById('btn-export-csv').style.display = 'inline-flex';
}

function renderEquityCurve(equityCurve, stratName) {
    const chartDiv = document.getElementById('bt-equity-chart');
    if (!chartDiv || !equityCurve || !equityCurve.length) return;

    const x = equityCurve.map(p => p.date);
    const y = equityCurve.map(p => p.cum_pnl);

    const trace = {
        x: x,
        y: y,
        type: 'scatter',
        mode: 'lines',
        line: { color: '#38bdf8', width: 2.5 },
        fill: 'tozeroy',
        fillcolor: 'rgba(56, 189, 248, 0.08)',
        name: 'Cumulative P&L'
    };

    const layout = {
        title: { text: `${stratName} — Cumulative P&L Curve (₹)`, font: { color: '#f8fafc', size: 14 } },
        paper_bgcolor: 'transparent',
        plot_bgcolor: 'transparent',
        margin: { l: 60, r: 20, t: 40, b: 40 },
        xaxis: { color: '#94a3b8', gridcolor: 'rgba(255,255,255,0.05)', showgrid: true },
        yaxis: { color: '#94a3b8', gridcolor: 'rgba(255,255,255,0.05)', showgrid: true, tickprefix: '₹' },
    };

    Plotly.newPlot(chartDiv, [trace], layout, { responsive: true, displayModeBar: false });
}

function renderGexScatterChart(points) {
    const chartDiv = document.getElementById('bt-gex-scatter-chart');
    if (!chartDiv || !points || !points.length) return;

    const bigPoints = points.filter(p => p.tier === 'BIG MOVE');
    const medPoints = points.filter(p => p.tier === 'MEDIUM MOVE');
    const smallPoints = points.filter(p => p.tier === 'SMALL MOVE');

    const traceBig = {
        x: bigPoints.map(p => p.x),
        y: bigPoints.map(p => p.y),
        mode: 'markers',
        type: 'scatter',
        name: 'Big Move (>180 pts)',
        marker: { color: '#ef4444', size: 8 }
    };

    const traceMed = {
        x: medPoints.map(p => p.x),
        y: medPoints.map(p => p.y),
        mode: 'markers',
        type: 'scatter',
        name: 'Medium Move (80-180 pts)',
        marker: { color: '#f59e0b', size: 7 }
    };

    const traceSmall = {
        x: smallPoints.map(p => p.x),
        y: smallPoints.map(p => p.y),
        mode: 'markers',
        type: 'scatter',
        name: 'Small Move (<80 pts)',
        marker: { color: '#10b981', size: 7 }
    };

    const layout = {
        title: { text: 'Net GEX (₹ Cr) vs Realized Next-Day Range (pts)', font: { color: '#f8fafc', size: 14 } },
        paper_bgcolor: 'transparent',
        plot_bgcolor: 'transparent',
        margin: { l: 60, r: 20, t: 40, b: 40 },
        xaxis: { title: 'Net GEX (₹ Crores)', color: '#94a3b8', gridcolor: 'rgba(255,255,255,0.05)' },
        yaxis: { title: 'Next-Day Spot Range (pts)', color: '#94a3b8', gridcolor: 'rgba(255,255,255,0.05)' },
        legend: { font: { color: '#cbd5e1' } }
    };

    Plotly.newPlot(chartDiv, [traceBig, traceMed, traceSmall], layout, { responsive: true, displayModeBar: false });
}

function renderTradeTable(trades) {
    const tbody = document.getElementById('bt-trades-table-body');
    if (!tbody) return;
    tbody.innerHTML = '';

    if (!trades || !trades.length) {
        tbody.innerHTML = `<tr><td colspan="11" style="text-align:center;color:#64748b;padding:2rem;">No trades simulated.</td></tr>`;
        return;
    }

    trades.forEach(t => {
        const tr = document.createElement('tr');
        const pnlClass = t.lot_pnl >= 0 ? 'text-green' : 'text-red';
        const pnlSign = t.lot_pnl >= 0 ? '+' : '';
        const reasonColor = t.exit_reason === 'TARGET_1' ? '#10b981' : (t.exit_reason === 'STOP_LOSS' ? '#ef4444' : '#94a3b8');

        tr.innerHTML = `
            <td style="font-weight:700;color:#cbd5e1;">#${t.trade_id}</td>
            <td class="font-mono" style="font-size:0.8rem;">${t.entry_date}</td>
            <td class="font-mono" style="font-weight:600;">${t.contract}</td>
            <td><span style="font-size:0.75rem;padding:2px 6px;border-radius:4px;background:${t.action === 'BUY' ? 'rgba(16,185,129,0.15)' : 'rgba(239,68,68,0.15)'};color:${t.action === 'BUY' ? '#10b981' : '#ef4444'};">${t.action}</span></td>
            <td class="font-mono">₹${t.entry_price.toFixed(1)}</td>
            <td class="font-mono" style="font-size:0.8rem;">${t.exit_date}</td>
            <td class="font-mono">₹${t.exit_price.toFixed(1)}</td>
            <td><span style="font-size:0.75rem;padding:2px 8px;border-radius:4px;background:rgba(255,255,255,0.05);color:${reasonColor};font-weight:600;border:1px solid rgba(255,255,255,0.1);">${t.exit_reason}</span></td>
            <td class="font-mono ${pnlClass}" style="font-weight:700;">${pnlSign}₹${t.lot_pnl.toLocaleString('en-IN')}</td>
            <td class="font-mono ${pnlClass}">${pnlSign}${t.roi_pct.toFixed(1)}%</td>
            <td style="color:#94a3b8;font-size:0.8rem;">${t.duration_days}d</td>
        `;
        tbody.appendChild(tr);
    });
}

function renderGexRecordsTable(records) {
    const tbody = document.getElementById('bt-gex-table-body');
    if (!tbody) return;
    tbody.innerHTML = '';

    if (!records || !records.length) {
        tbody.innerHTML = `<tr><td colspan="8" style="text-align:center;color:#64748b;padding:2rem;">No GEX records.</td></tr>`;
        return;
    }

    records.forEach(r => {
        const tr = document.createElement('tr');
        const tierColor = r.actual_tier === 'BIG MOVE' ? '#ef4444' : (r.actual_tier === 'SMALL MOVE' ? '#10b981' : '#f59e0b');
        tr.innerHTML = `
            <td class="font-mono" style="font-size:0.8rem;">${r.date}</td>
            <td class="font-mono" style="font-weight:600;">${Number(r.spot).toFixed(1)}</td>
            <td class="font-mono" style="color:${r.net_gex_crores < 0 ? '#ef4444' : '#10b981'};">${r.net_gex_crores} Cr</td>
            <td style="font-size:0.8rem;color:#94a3b8;">${r.predicted_tier}</td>
            <td class="font-mono" style="font-size:0.8rem;">${r.predicted_range}</td>
            <td class="font-mono" style="font-weight:700;">${r.actual_range_pts} pts</td>
            <td><span style="font-size:0.75rem;padding:2px 8px;border-radius:4px;background:rgba(255,255,255,0.05);color:${tierColor};font-weight:600;">${r.actual_tier}</span></td>
            <td>${r.tier_hit ? '<span style="color:#10b981;font-weight:700;">✓ HIT</span>' : '<span style="color:#64748b;">MISSED</span>'}</td>
        `;
        tbody.appendChild(tr);
    });
}

function exportCurrentBacktestCsv() {
    if (currentBacktestData && currentBacktestData.job_id) {
        window.location.href = `/api/backtest/export?job_id=${currentBacktestData.job_id}`;
    } else {
        const stratSelect = document.getElementById('bt-strategy-select');
        const daysSelect = document.getElementById('bt-days-select');
        const strat = stratSelect ? stratSelect.value : 'RADAR_ATM';
        const days = daysSelect ? daysSelect.value : 365;
        window.location.href = `/api/backtest/export?strategy=${strat}&days=${days}`;
    }
}

function filterTradeTable() {
    const input = document.getElementById('bt-table-search');
    if (!input || !currentBacktestData || !currentBacktestData.trades) return;
    const filter = input.value.toLowerCase();
    const filtered = currentBacktestData.trades.filter(t => 
        t.contract.toLowerCase().includes(filter) ||
        t.exit_reason.toLowerCase().includes(filter) ||
        t.entry_date.includes(filter)
    );
    renderTradeTable(filtered);
}

function formatInr(val) {
    if (val === undefined || val === null) return '₹0';
    const sign = val >= 0 ? '+' : '-';
    return `${sign}₹${Math.abs(val).toLocaleString('en-IN', { maximumFractionDigits: 0 })}`;
}
