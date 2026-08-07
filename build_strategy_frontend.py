import os

html_content = """<!DOCTYPE html>
<html lang="en" data-theme="dark">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Option Strategy Builder</title>
    <!-- Google Fonts -->
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;700&display=swap" rel="stylesheet">
    <!-- Plotly.js -->
    <script src="https://cdn.plot.ly/plotly-2.24.1.min.js"></script>
    <!-- Feather Icons -->
    <script src="https://unpkg.com/feather-icons"></script>
    <style>
        /* CSS Variables - Theme Engine */
        :root[data-theme="dark"] {
            --bg-base: #0f172a;
            --bg-glass: rgba(30, 41, 59, 0.7);
            --bg-glass-solid: #1e293b;
            --border-glass: rgba(255, 255, 255, 0.1);
            --border-highlight: rgba(255, 255, 255, 0.2);
            --text-primary: #f8fafc;
            --text-secondary: #94a3b8;
            --accent-blue: #3b82f6;
            --accent-blue-hover: #2563eb;
            --accent-green: #10b981;
            --accent-green-bg: rgba(16, 185, 129, 0.15);
            --accent-red: #ef4444;
            --accent-red-bg: rgba(239, 68, 68, 0.15);
            --accent-warn: #f59e0b;
            --shadow-glass: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
            --table-hover: rgba(255,255,255,0.03);
            --spot-glow: 0 0 15px rgba(59, 130, 246, 0.5);
            --grad-bg: radial-gradient(circle at 15% 50%, rgba(59, 130, 246, 0.15) 0%, transparent 50%), radial-gradient(circle at 85% 30%, rgba(16, 185, 129, 0.15) 0%, transparent 50%);
        }

        :root[data-theme="light"] {
            --bg-base: #f1f5f9;
            --bg-glass: rgba(255, 255, 255, 0.85);
            --bg-glass-solid: #ffffff;
            --border-glass: rgba(0, 0, 0, 0.08);
            --border-highlight: rgba(0, 0, 0, 0.15);
            --text-primary: #0f172a;
            --text-secondary: #475569;
            --accent-blue: #2563eb;
            --accent-blue-hover: #1d4ed8;
            --accent-green: #059669;
            --accent-green-bg: rgba(16, 185, 129, 0.15);
            --accent-red: #dc2626;
            --accent-red-bg: rgba(239, 68, 68, 0.15);
            --accent-warn: #d97706;
            --shadow-glass: 0 8px 32px 0 rgba(0, 0, 0, 0.05);
            --table-hover: rgba(0,0,0,0.02);
            --spot-glow: 0 0 15px rgba(37, 99, 235, 0.3);
            --grad-bg: radial-gradient(circle at 15% 50%, rgba(37, 99, 235, 0.08) 0%, transparent 50%), radial-gradient(circle at 85% 30%, rgba(5, 150, 105, 0.08) 0%, transparent 50%);
        }

        * { margin: 0; padding: 0; box-sizing: border-box; }
        
        body { font-family: 'Inter', sans-serif; background-color: var(--bg-base); background-image: var(--grad-bg); background-attachment: fixed; color: var(--text-primary); min-height: 100vh; display: flex; flex-direction: column; overflow-x: hidden; transition: background-color 0.3s, color 0.3s; }

        .text-green { color: var(--accent-green) !important; }
        .text-red { color: var(--accent-red) !important; }
        .font-mono { font-family: 'JetBrains Mono', monospace; }
        .glass-panel { background: var(--bg-glass); backdrop-filter: blur(16px); -webkit-backdrop-filter: blur(16px); border: 1px solid var(--border-glass); border-radius: 12px; box-shadow: var(--shadow-glass); transition: border-color 0.3s; }
        
        /* Header */
        header { padding: 1rem 2rem; display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid var(--border-glass); background: var(--bg-glass); backdrop-filter: blur(10px); position: sticky; top: 0; z-index: 100; }
        .header-logo { font-size: 1.5rem; font-weight: 700; background: linear-gradient(90deg, #3b82f6, #10b981); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
        .header-actions { display: flex; gap: 1.5rem; align-items: center; }
        
        .spot-price-box { background: var(--bg-glass-solid); padding: 0.4rem 1rem; border-radius: 8px; border: 1px solid var(--border-glass); display: flex; align-items: baseline; gap: 0.5rem; transition: box-shadow 0.3s; }
        .spot-price-box.pulse { box-shadow: var(--spot-glow); border-color: var(--accent-blue); }
        .spot-value { font-size: 1.4rem; font-weight: 700; font-family: 'JetBrains Mono'; }
        
        .theme-toggle { background: transparent; border: none; color: var(--text-primary); cursor: pointer; display: flex; align-items: center; padding: 0.5rem; border-radius: 50%; transition: background 0.2s;}
        .theme-toggle:hover { background: var(--border-glass); }

        /* Main Layout */
        main { display: grid; grid-template-columns: 420px 1fr; gap: 1.5rem; padding: 1.5rem; flex-grow: 1; max-width: 1800px; margin: 0 auto; width: 100%; }
        
        /* Sidebar */
        .sidebar { display: flex; flex-direction: column; height: calc(100vh - 120px); position: sticky; top: 90px; }
        .tabs { display: flex; border-bottom: 1px solid var(--border-glass); margin-bottom: 1rem; }
        .tab-btn { flex: 1; padding: 0.75rem 0.2rem; background: transparent; border: none; color: var(--text-secondary); cursor: pointer; font-weight: 600; font-size: 0.85rem; font-family: 'Inter', sans-serif; border-bottom: 2px solid transparent; transition: 0.2s; }
        .tab-btn.active { color: var(--accent-blue); border-bottom: 2px solid var(--accent-blue); }
        .tab-btn:hover:not(.active) { color: var(--text-primary); background: var(--table-hover); }

        .tab-content { display: none; flex-direction: column; gap: 1rem; overflow-y: auto; flex-grow: 1; padding-right: 0.5rem; scrollbar-width: thin; scrollbar-color: var(--accent-blue) transparent; }
        .tab-content.active { display: flex; animation: fadeIn 0.3s ease; }
        @keyframes fadeIn { from { opacity: 0; transform: translateY(5px); } to { opacity: 1; transform: translateY(0); } }

        .strategy-category { margin-bottom: 1rem; }
        .category-title { font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.05em; color: var(--text-secondary); margin-bottom: 0.5rem; padding-left: 0.2rem; }
        .preset-btn { width: 100%; padding: 0.75rem 1rem; background: var(--bg-glass-solid); border: 1px solid var(--border-glass); border-radius: 8px; cursor: pointer; text-align: left; color: var(--text-primary); font-weight: 500; transition: all 0.2s; margin-bottom: 0.5rem; display: flex; justify-content: space-between; align-items: center; }
        .preset-btn:hover { border-color: var(--accent-blue); transform: translateX(4px); box-shadow: 0 4px 12px rgba(0,0,0,0.1); }
        
        .ai-card { padding: 1rem; background: var(--bg-glass-solid); border: 1px solid var(--border-glass); border-radius: 8px; cursor: pointer; transition: 0.2s; }
        .ai-card:hover { border-color: var(--accent-blue); transform: translateY(-2px); box-shadow: 0 4px 12px rgba(59, 130, 246, 0.1); }
        .ai-title { font-weight: 600; font-size: 1.05rem; margin-bottom: 0.5rem; color: var(--text-primary); }
        .ai-desc { font-size: 0.85rem; color: var(--text-secondary); margin-bottom: 0.5rem;}
        
        /* Portfolio Tab */
        .pos-card { padding: 1rem; background: var(--bg-glass-solid); border: 1px solid var(--border-glass); border-radius: 8px; margin-bottom: 1rem; position: relative; overflow: hidden; cursor: pointer; transition: 0.2s; }
        .pos-card:hover { border-color: var(--accent-blue); }
        .pos-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem; }
        .pos-name { font-weight: 600; color: var(--text-primary); display: flex; align-items: center; gap: 0.5rem; }
        .pos-margin { font-size: 0.75rem; color: var(--text-secondary); }
        .pos-pnl { font-size: 1.2rem; font-weight: 700; font-family: 'JetBrains Mono'; text-align: right; }
        .pos-pnl.pulse-green { animation: flashGreen 1s; }
        .pos-pnl.pulse-red { animation: flashRed 1s; }
        
        .pop-badge { padding: 0.15rem 0.4rem; border-radius: 4px; font-size: 0.7rem; font-weight: 600; font-family: 'JetBrains Mono'; }
        .pop-green { background: var(--accent-green-bg); color: var(--accent-green); }
        .pop-red { background: var(--accent-red-bg); color: var(--accent-red); }
        .pop-warn { background: rgba(245, 158, 11, 0.15); color: var(--accent-warn); }

        .pos-leg-table { width: 100%; border-collapse: collapse; margin: 0.5rem 0; font-size: 0.75rem; background: var(--bg-base); border-radius: 6px; overflow: hidden; }
        .pos-leg-table th { background: rgba(0,0,0,0.2); padding: 0.4rem; color: var(--text-secondary); font-weight: 500; text-align: right; }
        .pos-leg-table th:first-child { text-align: left; }
        .pos-leg-table td { padding: 0.4rem; text-align: right; border-bottom: 1px solid rgba(255,255,255,0.05); }
        .pos-leg-table td:first-child { text-align: left; }
        
        .pos-footer { margin-top: 1rem; display: flex; justify-content: space-between; align-items: center; }
        .btn-exit { background: var(--accent-red-bg); color: var(--accent-red); border: 1px solid var(--accent-red); padding: 0.4rem 0.8rem; border-radius: 6px; cursor: pointer; font-size: 0.8rem; font-weight: 600; transition: 0.2s; z-index: 2; position: relative; }
        .btn-exit:hover { background: var(--accent-red); color: white; }

        @keyframes flashGreen { 0% { text-shadow: 0 0 10px var(--accent-green); } 100% { text-shadow: none; } }
        @keyframes flashRed { 0% { text-shadow: 0 0 10px var(--accent-red); } 100% { text-shadow: none; } }
        
        /* Dashboard Right */
        .builder-dashboard { display: flex; flex-direction: column; gap: 1.5rem; min-width: 0; }
        .dashboard-top { display: grid; grid-template-columns: 1.5fr 1fr; gap: 1.5rem; }
        
        .chart-container { padding: 1rem; display: flex; flex-direction: column; min-height: 380px; }
        #payoff-chart { flex-grow: 1; width: 100%; }
        
        .controls-row { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 1rem; margin-bottom: 1rem; padding: 0 1rem; }
        .slider-group { display: flex; flex-direction: column; gap: 0.5rem; }
        .slider-group label { font-size: 0.75rem; color: var(--text-secondary); display: flex; justify-content: space-between; text-transform: uppercase; letter-spacing: 0.05em; font-weight: 600;}
        input[type="range"] { width: 100%; accent-color: var(--accent-blue); cursor: pointer; }
        
        .summary-container { padding: 1.5rem; display: flex; flex-direction: column; gap: 1rem; justify-content: space-between; }
        .summary-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; }
        .stat-box { background: var(--bg-glass-solid); padding: 1rem; border-radius: 8px; border: 1px solid var(--border-glass); }
        .stat-label { font-size: 0.75rem; color: var(--text-secondary); text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.4rem;}
        .stat-value { font-size: 1.3rem; font-weight: 600; font-family: 'JetBrains Mono'; }
        
        .greeks-row { display: grid; grid-template-columns: repeat(4, 1fr); gap: 0.5rem; margin-top: 0.5rem; padding-top: 1rem; border-top: 1px solid var(--border-glass); }
        .greek-box { text-align: center; }
        .greek-label { font-size: 0.75rem; color: var(--text-secondary); margin-bottom: 0.2rem;}
        .greek-value { font-size: 0.95rem; font-family: 'JetBrains Mono'; }
        
        /* Basket */
        .basket-container { padding: 1rem; }
        .basket-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem; }
        .leg-table { width: 100%; border-collapse: collapse; text-align: left; }
        .leg-table th { color: var(--text-secondary); font-size: 0.85rem; font-weight: 500; padding: 0.75rem 1rem; border-bottom: 1px solid var(--border-glass); }
        .leg-table td { padding: 0.75rem 1rem; border-bottom: 1px solid var(--border-glass); font-size: 0.95rem; }
        .leg-table tr:hover td { background: var(--table-hover); }
        
        .action-toggle { display: inline-flex; border-radius: 6px; overflow: hidden; border: 1px solid var(--border-glass); }
        .action-toggle button { background: var(--bg-glass-solid); color: var(--text-primary); border: none; padding: 0.25rem 0.75rem; font-size: 0.85rem; cursor: pointer; transition: 0.2s; }
        .action-toggle button.active.buy { background: var(--accent-green); color: white; font-weight: 600; }
        .action-toggle button.active.sell { background: var(--accent-red); color: white; font-weight: 600; }
        
        .qty-control { display: inline-flex; align-items: center; gap: 0.5rem; }
        .qty-control button { background: var(--bg-glass-solid); border: 1px solid var(--border-glass); color: var(--text-primary); width: 26px; height: 26px; border-radius: 4px; cursor: pointer; display: flex; align-items: center; justify-content: center;}
        
        .btn-delete { background: transparent; border: none; color: var(--text-secondary); cursor: pointer; transition: 0.2s; display: flex; align-items: center; justify-content: center;}
        .btn-delete:hover { color: var(--accent-red); transform: scale(1.1); }
        .btn-deploy { background: var(--accent-green); color: white; border: none; padding: 0.5rem 1rem; border-radius: 6px; font-weight: 600; cursor: pointer; display: flex; align-items: center; gap: 0.5rem; transition: 0.2s; box-shadow: 0 4px 12px var(--accent-green-bg); }
        .btn-deploy:hover { background: #059669; transform: translateY(-2px); }
        
        /* Chain */
        .chain-container { padding: 0; margin-bottom: 2rem; overflow: hidden; display: flex; flex-direction: column; }
        .chain-header { display: flex; justify-content: space-between; align-items: center; padding: 1rem 1.5rem; border-bottom: 1px solid var(--border-glass); }
        .chain-table-wrapper { max-height: 600px; overflow-y: auto; scrollbar-width: thin; scrollbar-color: var(--accent-blue) transparent; }
        .chain-table { width: 100%; border-collapse: collapse; text-align: center; font-size: 0.9rem; position: relative;}
        .chain-table th { background: var(--bg-glass-solid); position: sticky; top: 0; padding: 0.75rem; color: var(--text-secondary); border-bottom: 1px solid var(--border-glass); z-index: 10; font-weight: 500; font-size: 0.8rem; text-transform: uppercase;}
        .chain-table td { padding: 0.6rem 0.5rem; border-bottom: 1px solid var(--border-glass); transition: background 0.2s; }
        .strike-col { font-weight: 600; font-family: 'JetBrains Mono'; background: var(--bg-glass-solid); }
        .chain-table tbody tr:hover { background: var(--table-hover); }
        
        .hover-actions { opacity: 0; display: inline-flex; gap: 0.25rem; margin-left: 0.5rem; transition: 0.2s; vertical-align: middle;}
        .chain-table tr:hover .hover-actions { opacity: 1; }
        .btn-add { padding: 0.15rem 0.4rem; border-radius: 4px; border: none; font-size: 0.7rem; font-weight: 600; cursor: pointer; color: white;}
        .btn-add.buy { background: var(--accent-green-bg); color: var(--accent-green); border: 1px solid var(--accent-green); }
        .btn-add.sell { background: var(--accent-red-bg); color: var(--accent-red); border: 1px solid var(--accent-red); }
        .btn-add.buy:hover { background: var(--accent-green); }
        .btn-add.sell:hover { background: var(--accent-red); }
        
        .spot-line-row td { padding: 0 !important; border: none !important; height: 2px !important; background: var(--accent-blue); position: relative; box-shadow: var(--spot-glow); z-index: 5;}
        .spot-line-label { position: absolute; left: 50%; top: 50%; transform: translate(-50%, -50%); background: var(--accent-blue); color: white; font-size: 0.7rem; font-weight: bold; padding: 0.2rem 0.6rem; border-radius: 12px; font-family: 'JetBrains Mono'; box-shadow: var(--spot-glow); }

        /* Modal Orderbook */
        .modal-overlay { position: fixed; top: 0; left: 0; width: 100vw; height: 100vh; background: rgba(0,0,0,0.5); backdrop-filter: blur(5px); z-index: 1000; display: none; align-items: center; justify-content: center; }
        .modal-content { background: var(--bg-glass-solid); border: 1px solid var(--border-glass); border-radius: 12px; padding: 2rem; width: 800px; max-width: 90vw; max-height: 80vh; overflow-y: auto; box-shadow: var(--shadow-glass); position: relative; }
        .modal-close { position: absolute; top: 1rem; right: 1rem; background: transparent; border: none; color: var(--text-secondary); cursor: pointer; }
    </style>
</head>
<body>
    <header>
        <div class="header-logo">Strategy Builder & Monitor</div>
        <div class="header-actions">
            <div class="spot-price-box" id="spot-container">
                <span style="color:var(--text-secondary); font-size:0.85rem;">NIFTY Spot</span>
                <span class="spot-value" id="spot-price">---</span>
            </div>
            <button class="theme-toggle" id="theme-btn" onclick="toggleTheme()"><i data-feather="sun" id="theme-icon"></i></button>
        </div>
    </header>

    <main>
        <!-- Sidebar -->
        <aside class="sidebar glass-panel">
            <div class="tabs">
                <button class="tab-btn active" onclick="switchTab('tab-ready')">Builder</button>
                <button class="tab-btn" onclick="switchTab('tab-portfolio')">Live Positions <span id="pos-badge" style="background:var(--accent-blue); color:white; border-radius:50%; padding:0.1rem 0.4rem; font-size:0.7rem; margin-left:0.2rem; display:none;">0</span></button>
                <button class="tab-btn" onclick="switchTab('tab-ai')">AI Wizard</button>
            </div>
            
            <div id="tab-ready" class="tab-content active" style="padding: 0 1rem 1rem 1rem;">
                <div class="strategy-category">
                    <div class="category-title">Bullish</div>
                    <button class="preset-btn" onclick="buildStrategy('Bull Call Spread')">Bull Call Spread <i data-feather="trending-up" width="16"></i></button>
                    <button class="preset-btn" onclick="buildStrategy('Bull Put Spread')">Bull Put Spread <i data-feather="trending-up" width="16"></i></button>
                </div>
                <div class="strategy-category">
                    <div class="category-title">Bearish</div>
                    <button class="preset-btn" onclick="buildStrategy('Bear Put Spread')">Bear Put Spread <i data-feather="trending-down" width="16"></i></button>
                    <button class="preset-btn" onclick="buildStrategy('Bear Call Spread')">Bear Call Spread <i data-feather="trending-down" width="16"></i></button>
                </div>
                <div class="strategy-category">
                    <div class="category-title">Neutral / Range Bound</div>
                    <button class="preset-btn" onclick="buildStrategy('Short Straddle')">Short Straddle <i data-feather="target" width="16"></i></button>
                    <button class="preset-btn" onclick="buildStrategy('Short Strangle')">Short Strangle <i data-feather="target" width="16"></i></button>
                    <button class="preset-btn" onclick="buildStrategy('Iron Condor')">Iron Condor <i data-feather="anchor" width="16"></i></button>
                    <button class="preset-btn" onclick="buildStrategy('Iron Butterfly')">Iron Butterfly <i data-feather="anchor" width="16"></i></button>
                </div>
                <div class="strategy-category">
                    <div class="category-title">Advanced</div>
                    <button class="preset-btn" onclick="buildStrategy('Call Ratio Backspread')">Call Ratio Backspread <i data-feather="trending-up" width="16"></i></button>
                    <button class="preset-btn" onclick="buildStrategy('Put Ratio Backspread')">Put Ratio Backspread <i data-feather="trending-down" width="16"></i></button>
                </div>
            </div>

            <div id="tab-portfolio" class="tab-content" style="padding: 0 1rem 1rem 1rem;">
                <div style="display:flex; justify-content:space-between; margin-bottom:1rem;">
                    <h3 style="font-size:1rem;">Live Portfolio</h3>
                    <button onclick="document.getElementById('orderbook-modal').style.display='flex'" style="background:transparent; border:1px solid var(--border-glass); color:var(--text-secondary); border-radius:6px; padding:0.3rem 0.6rem; cursor:pointer; font-size:0.75rem;">
                        <i data-feather="book-open" width="12"></i> History
                    </button>
                </div>
                <div id="active-positions-container">
                    <div style="color:var(--text-secondary); font-size:0.9rem; text-align:center; padding: 2rem 0;">No active positions deployed.</div>
                </div>
            </div>

            <div id="tab-ai" class="tab-content" style="padding: 0 1rem 1rem 1rem;">
                <button id="btn-fetch-wizard" onclick="fetchWizardRecommendation()" style="width:100%; padding:0.75rem; background:var(--accent-blue); color:white; border:none; border-radius:8px; cursor:pointer; font-weight:600; margin-bottom:1rem; display:flex; justify-content:center; align-items:center; gap:0.5rem; transition:0.2s;"><i data-feather="cpu" width="18"></i> Analyze Live Market & Suggest Strategy</button>
                <h3 style="font-size:0.9rem; color:var(--text-secondary); margin-bottom:0.5rem; border-bottom:1px solid var(--border-glass); padding-bottom:0.5rem;">Today's Insights</h3>
                <div id="ai-presets-container" style="display:flex; flex-direction:column; gap:1rem;">
                    <div style="color:var(--text-secondary); font-size:0.9rem; text-align:center; padding: 2rem 0;">No recommendations generated today.</div>
                </div>
            </div>
        </aside>

        <!-- Builder Dashboard -->
        <section class="builder-dashboard">
            <div class="dashboard-top">
                <div class="chart-container glass-panel">
                    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.5rem; padding:0 1.5rem;">
                        <div style="display:flex; align-items:center; gap:0.5rem;">
                            <span style="font-size:0.75rem; color:var(--text-secondary); text-transform:uppercase; font-weight:600;">SD Mode</span>
                            <div class="qty-control">
                                <button onclick="updateSDDays(Math.max(1, sdDays-1))">-</button>
                                <span class="font-mono" id="val-sd-days" style="min-width:24px; text-align:center;">14</span>
                                <button onclick="updateSDDays(sdDays+1)">+</button>
                            </div>
                            <span style="font-size:0.75rem; color:var(--text-secondary);">Days</span>
                        </div>
                    </div>
                    <div class="controls-row">
                        <div class="slider-group">
                            <label><span>Target Spot Shift</span> <span id="val-target-spot" class="font-mono text-green">+0%</span></label>
                            <input type="range" id="slide-spot" min="-15" max="15" value="0" step="0.5">
                        </div>
                        <div class="slider-group">
                            <label><span>Target Date</span> <span id="val-target-date" class="font-mono">T+0 Days</span></label>
                            <input type="range" id="slide-date" min="0" max="30" value="0" step="1">
                        </div>
                        <div class="slider-group">
                            <label><span>Target Time</span> <span id="val-target-time" class="font-mono">Market Close</span></label>
                            <input type="range" id="slide-time" min="0" max="25" value="25" step="1">
                        </div>
                    </div>
                    <div id="payoff-chart"></div>
                    <div style="text-align:center; font-size:1.1rem; font-weight:600; margin-top:-10px; color:var(--text-primary);">Projected Profit: <span id="projected-profit" class="font-mono">₹0</span></div>
                </div>
                
                <div class="summary-container glass-panel">
                    <h3 id="summary-strategy-name" style="font-weight: 600;">Custom Strategy</h3>
                    <div class="summary-grid">
                        <div class="stat-box"><div class="stat-label">Max Profit</div><div class="stat-value text-green" id="stat-max-profit">₹0</div></div>
                        <div class="stat-box"><div class="stat-label">Max Loss</div><div class="stat-value text-red" id="stat-max-loss">₹0</div></div>
                        <div class="stat-box"><div class="stat-label">Prob. of Profit</div><div class="stat-value" id="stat-pop">0%</div></div>
                        <div class="stat-box"><div class="stat-label">Net Premium</div><div class="stat-value" id="stat-premium">₹0</div></div>
                    </div>
                    <div style="background: var(--bg-glass-solid); padding: 0.75rem 1rem; border-radius: 8px; border: 1px solid var(--border-glass); display: flex; flex-direction:column; gap:0.5rem; justify-content: space-between;">
                        <div style="display:flex; justify-content: space-between;"><span class="stat-label" style="margin:0;">Target Breakevens</span><span class="font-mono" id="stat-target-breakevens" style="font-size: 1rem;">-</span></div>
                        <div style="display:flex; justify-content: space-between;"><span class="stat-label" style="margin:0;">Expiry Breakevens</span><span class="font-mono text-secondary" id="stat-breakevens" style="font-size: 1rem;">-</span></div>
                    </div>
                    <div class="greeks-row">
                        <div class="greek-box"><div class="greek-label">Delta</div><div class="greek-value" id="greek-delta">0.0</div></div>
                        <div class="greek-box"><div class="greek-label">Theta</div><div class="greek-value" id="greek-theta">0.0</div></div>
                        <div class="greek-box"><div class="greek-label">Gamma</div><div class="greek-value" id="greek-gamma">0.0</div></div>
                        <div class="greek-box"><div class="greek-label">Vega</div><div class="greek-value" id="greek-vega">0.0</div></div>
                    </div>
                </div>
            </div>

            <div class="basket-container glass-panel">
                <div class="basket-header">
                    <h3 style="font-weight: 600;">Leg Builder</h3>
                    <div style="display:flex; gap:1rem;">
                        <button id="btn-clear-basket" style="padding: 0.4rem 0.8rem; background:transparent; border:1px solid var(--border-glass); color:var(--text-secondary); border-radius:6px; cursor:pointer; display:flex; align-items:center; gap:0.4rem;"><i data-feather="trash-2" width="14"></i> Clear</button>
                        <button id="btn-deploy-strategy" class="btn-deploy" onclick="deployStrategy()"><i data-feather="send" width="14"></i> Deploy as Paper Trade</button>
                    </div>
                </div>
                <table class="leg-table">
                    <thead><tr><th>Action</th><th>Lots (x65)</th><th>Expiry</th><th>Strike</th><th>Type</th><th>Entry Price</th><th>IV</th><th></th></tr></thead>
                    <tbody id="basket-body"><tr><td colspan="8" style="text-align:center; color:var(--text-secondary); padding: 2rem 0;">No legs added.</td></tr></tbody>
                </table>
            </div>

            <div class="chain-container glass-panel">
                <div class="chain-header">
                    <h3 style="font-weight: 600;">Option Chain</h3>
                    <div style="display:flex; gap:1rem; align-items:center;">
                        <select id="expiry-select" style="padding:0.3rem; border-radius:4px; background:var(--bg-glass-solid); color:var(--text-primary); border:1px solid var(--border-glass);">
                        </select>
                        <div class="action-toggle">
                            <button id="view-ltp" class="active" onclick="setChainView('LTP')">LTP / OI</button>
                            <button id="view-greeks" onclick="setChainView('Greeks')">Greeks</button>
                        </div>
                    </div>
                </div>
                <div class="chain-table-wrapper" id="chain-wrapper">
                    <table class="chain-table">
                        <thead><tr><th>Call OI</th><th>Call Delta</th><th>Call LTP</th><th class="strike-col">Strike</th><th>Put LTP</th><th>Put Delta</th><th>Put OI</th></tr></thead>
                        <tbody id="chain-body"><tr><td colspan="7" style="padding: 3rem 0;">Loading chain...</td></tr></tbody>
                    </table>
                </div>
            </div>
        </section>
    </main>

    <!-- Modal -->
    <div id="orderbook-modal" class="modal-overlay">
        <div class="modal-content">
            <button class="modal-close" onclick="document.getElementById('orderbook-modal').style.display='none'"><i data-feather="x"></i></button>
            <h2 style="margin-bottom:1rem;">P&L History Directory</h2>
            <div style="max-height: 400px; overflow-y:auto;">
                <table class="leg-table" style="width:100%;">
                    <thead><tr><th>Date</th><th>Strategy</th><th>Net Margin</th><th>Realized P&L</th></tr></thead>
                    <tbody id="history-body"></tbody>
                </table>
            </div>
        </div>
    </div>

    <script>
        feather.replace();
        let currentSpot = 0; let optionChain = []; let strikesList = []; let activeLegs = []; 
        let debounceTimer; let isDarkMode = true; let lastStrategyName = "Custom Strategy"; let activePositionsDict = {};

        function toggleTheme() {
            isDarkMode = !isDarkMode;
            document.documentElement.setAttribute('data-theme', isDarkMode ? 'dark' : 'light');
            document.getElementById('theme-icon').setAttribute('data-feather', isDarkMode ? 'sun' : 'moon');
            feather.replace();
            const fontColor = isDarkMode ? '#94a3b8' : '#475569';
            const gridColor = isDarkMode ? 'rgba(255,255,255,0.05)' : 'rgba(0,0,0,0.05)';
            const zeroColor = isDarkMode ? 'rgba(255,255,255,0.2)' : 'rgba(0,0,0,0.2)';
            Plotly.relayout('payoff-chart', { 'font.color': fontColor, 'xaxis.gridcolor': gridColor, 'xaxis.zerolinecolor': zeroColor, 'yaxis.gridcolor': gridColor, 'yaxis.zerolinecolor': zeroColor });
        }

        function switchTab(tabId) {
            document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
            event.target.classList.add('active');
            document.getElementById(tabId).classList.add('active');
        }

        const chartLayout = {
            paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)', font: { color: '#94a3b8', family: 'Inter' },
            margin: { t: 20, r: 20, l: 50, b: 40 }, xaxis: { gridcolor: 'rgba(255,255,255,0.05)', zerolinecolor: 'rgba(255,255,255,0.2)' },
            yaxis: { gridcolor: 'rgba(255,255,255,0.05)', zerolinecolor: 'rgba(255,255,255,0.2)' }, showlegend: true, legend: { orientation: 'h', y: 1.1 }, hovermode: 'x unified'
        };
        Plotly.newPlot('payoff-chart', [], chartLayout, {responsive: true, displayModeBar: false});

        let currentChainView = 'LTP';
        function setChainView(view) {
            currentChainView = view;
            document.getElementById('view-ltp').classList.toggle('active', view === 'LTP');
            document.getElementById('view-greeks').classList.toggle('active', view === 'Greeks');
            renderChain();
        }
        
        async function fetchChain() {
            try {
                const res = await fetch('/api/chain/live'); const data = await res.json();
                if(data.ok) {
                    if (currentSpot !== data.spot) { currentSpot = data.spot; const box = document.getElementById('spot-container'); box.classList.add('pulse'); setTimeout(() => box.classList.remove('pulse'), 1000); }
                    document.getElementById('spot-price').innerText = currentSpot.toFixed(2);
                    optionChain = data.options; strikesList = [...new Set(optionChain.map(o => o.strike))].sort((a,b) => a-b);
                    
                    const expirySelect = document.getElementById('expiry-select');
                    if (expirySelect.options.length === 0 || expirySelect.options[0].value !== data.expiry) {
                        expirySelect.innerHTML = `<option value="${data.expiry}">${data.expiry}</option>`;
                    }
                    
                    renderChain();
                }
            } catch(e) { console.error(e); }
        }

        function renderChain() {
            const tbody = document.getElementById('chain-body'); tbody.innerHTML = '';
            const thead = document.querySelector('.chain-table thead tr');
            
            if (currentChainView === 'LTP') {
                thead.innerHTML = '<th>Call OI</th><th>Call Delta</th><th>Call LTP</th><th class="strike-col">Strike</th><th>Put LTP</th><th>Put Delta</th><th>Put OI</th>';
            } else {
                thead.innerHTML = '<th>Call Delta</th><th class="strike-col">Strike</th><th>IV</th><th>Put Delta</th><th>Theta</th><th>Vega</th><th>Gamma</th>';
            }
            
            let spotUpperStrikeIndex = strikesList.findIndex(s => s > currentSpot);
            if(spotUpperStrikeIndex === -1) spotUpperStrikeIndex = strikesList.length;

            strikesList.forEach((strike, index) => {
                if (index === spotUpperStrikeIndex) {
                    const spotLine = document.createElement('tr'); spotLine.className = 'spot-line-row';
                    spotLine.innerHTML = `<td colspan="7"><div class="spot-line-label">${currentSpot.toFixed(2)}</div></td>`; tbody.appendChild(spotLine);
                }
                const calls = optionChain.filter(o => o.strike === strike && o.type === 'CE'); const puts = optionChain.filter(o => o.strike === strike && o.type === 'PE');
                const call = calls[0] || {}; const put = puts[0] || {};
                const tr = document.createElement('tr');
                
                if (currentChainView === 'LTP') {
                    tr.innerHTML = `
                        <td>${(call.oi || 0).toLocaleString()}</td><td class="font-mono text-secondary">${(call.greeks?.delta || 0).toFixed(2)}</td>
                        <td class="font-mono">${(call.ltp || 0).toFixed(2)}<div class="hover-actions"><button class="btn-add buy" onclick="addLeg('BUY', ${strike}, 'CE', ${call.ltp}, ${call.iv})">B</button><button class="btn-add sell" onclick="addLeg('SELL', ${strike}, 'CE', ${call.ltp}, ${call.iv})">S</button></div></td>
                        <td class="strike-col">${strike}</td>
                        <td class="font-mono">${(put.ltp || 0).toFixed(2)}<div class="hover-actions"><button class="btn-add buy" onclick="addLeg('BUY', ${strike}, 'PE', ${put.ltp}, ${put.iv})">B</button><button class="btn-add sell" onclick="addLeg('SELL', ${strike}, 'PE', ${put.ltp}, ${put.iv})">S</button></div></td>
                        <td class="font-mono text-secondary">${(put.greeks?.delta || 0).toFixed(2)}</td><td>${(put.oi || 0).toLocaleString()}</td>
                    `;
                } else {
                    tr.innerHTML = `
                        <td class="font-mono">${(call.greeks?.delta || 0).toFixed(2)}</td>
                        <td class="strike-col">${strike}</td>
                        <td class="font-mono text-secondary">${((call.iv || put.iv || 0)).toFixed(1)}</td>
                        <td class="font-mono">${(put.greeks?.delta || 0).toFixed(2)}</td>
                        <td class="font-mono">${(call.greeks?.theta || 0).toFixed(2)}</td>
                        <td class="font-mono">${(call.greeks?.vega || 0).toFixed(2)}</td>
                        <td class="font-mono">${(call.greeks?.gamma || 0).toFixed(4)}</td>
                    `;
                    // Click on Greek view to revert to LTP and show basket buttons
                    tr.style.cursor = 'pointer';
                    tr.onclick = (e) => { setChainView('LTP'); };
                }
                tbody.appendChild(tr);
            });
            
            // Auto center logic
            const wrapper = document.getElementById('chain-wrapper'); 
            const spotEl = wrapper.querySelector('.spot-line-row'); 
            if(spotEl) { 
                wrapper.scrollTop = spotEl.offsetTop - wrapper.offsetHeight / 2;
            }
        }

        function getOption(strike, type) { return optionChain.find(o => o.strike === strike && o.type === type) || { ltp: 0, iv: 15 }; }
        function buildStrategy(presetName) {
            if (strikesList.length === 0 || currentSpot === 0) return;
            activeLegs = []; lastStrategyName = presetName;
            let atmStrike = strikesList[0]; let minDiff = Infinity; let atmIndex = 0;
            strikesList.forEach((s, i) => { const diff = Math.abs(s - currentSpot); if(diff < minDiff) { minDiff = diff; atmStrike = s; atmIndex = i; } });
            const getStrike = (offset) => { const idx = Math.max(0, Math.min(strikesList.length - 1, atmIndex + offset)); return strikesList[idx]; };
            let c1, c2, p1, p2;
            switch(presetName) {
                case 'Short Straddle': c1 = getOption(atmStrike, 'CE'); p1 = getOption(atmStrike, 'PE'); addLegRaw('SELL', atmStrike, 'CE', c1.ltp, c1.iv); addLegRaw('SELL', atmStrike, 'PE', p1.ltp, p1.iv); break;
                case 'Bull Call Spread': c1 = getOption(atmStrike, 'CE'); c2 = getOption(getStrike(2), 'CE'); addLegRaw('BUY', atmStrike, 'CE', c1.ltp, c1.iv); addLegRaw('SELL', getStrike(2), 'CE', c2.ltp, c2.iv); break;
                case 'Bull Put Spread': p1 = getOption(getStrike(-1), 'PE'); p2 = getOption(getStrike(-3), 'PE'); addLegRaw('SELL', getStrike(-1), 'PE', p1.ltp, p1.iv); addLegRaw('BUY', getStrike(-3), 'PE', p2.ltp, p2.iv); break;
                case 'Bear Put Spread': p1 = getOption(atmStrike, 'PE'); p2 = getOption(getStrike(-2), 'PE'); addLegRaw('BUY', atmStrike, 'PE', p1.ltp, p1.iv); addLegRaw('SELL', getStrike(-2), 'PE', p2.ltp, p2.iv); break;
                case 'Bear Call Spread': c1 = getOption(getStrike(1), 'CE'); c2 = getOption(getStrike(3), 'CE'); addLegRaw('SELL', getStrike(1), 'CE', c1.ltp, c1.iv); addLegRaw('BUY', getStrike(3), 'CE', c2.ltp, c2.iv); break;
                case 'Iron Condor': addLegRaw('SELL', getStrike(2), 'CE', getOption(getStrike(2), 'CE').ltp, getOption(getStrike(2), 'CE').iv); addLegRaw('BUY', getStrike(4), 'CE', getOption(getStrike(4), 'CE').ltp, getOption(getStrike(4), 'CE').iv); addLegRaw('SELL', getStrike(-2), 'PE', getOption(getStrike(-2), 'PE').ltp, getOption(getStrike(-2), 'PE').iv); addLegRaw('BUY', getStrike(-4), 'PE', getOption(getStrike(-4), 'PE').ltp, getOption(getStrike(-4), 'PE').iv); break;
                case 'Short Strangle': c1 = getOption(getStrike(2), 'CE'); p1 = getOption(getStrike(-2), 'PE'); addLegRaw('SELL', getStrike(2), 'CE', c1.ltp, c1.iv); addLegRaw('SELL', getStrike(-2), 'PE', p1.ltp, p1.iv); break;
                case 'Iron Butterfly': c1 = getOption(atmStrike, 'CE'); p1 = getOption(atmStrike, 'PE'); c2 = getOption(getStrike(2), 'CE'); p2 = getOption(getStrike(-2), 'PE'); addLegRaw('SELL', atmStrike, 'CE', c1.ltp, c1.iv); addLegRaw('SELL', atmStrike, 'PE', p1.ltp, p1.iv); addLegRaw('BUY', getStrike(2), 'CE', c2.ltp, c2.iv); addLegRaw('BUY', getStrike(-2), 'PE', p2.ltp, p2.iv); break;
                case 'Call Ratio Backspread': c1 = getOption(getStrike(-1), 'CE'); c2 = getOption(getStrike(1), 'CE'); addLegRaw('SELL', getStrike(-1), 'CE', c1.ltp, c1.iv, 1); addLegRaw('BUY', getStrike(1), 'CE', c2.ltp, c2.iv, 2); break;
                case 'Put Ratio Backspread': p1 = getOption(getStrike(1), 'PE'); p2 = getOption(getStrike(-1), 'PE'); addLegRaw('SELL', getStrike(1), 'PE', p1.ltp, p1.iv, 1); addLegRaw('BUY', getStrike(-1), 'PE', p2.ltp, p2.iv, 2); break;
            }
            renderBasket(); triggerAnalyze();
        }

        function addLegRaw(action, strike, type, price, iv, lots=1) { activeLegs.push({ id: Date.now() + Math.random(), action, lots, strike, type, price, iv, expiry: 'Current' }); }
        function addLeg(action, strike, type, price, iv) { lastStrategyName = "Custom Strategy"; addLegRaw(action, strike, type, price, iv, 1); renderBasket(); triggerAnalyze(); }
        function removeLeg(id) { activeLegs = activeLegs.filter(l => l.id !== id); renderBasket(); triggerAnalyze(); }
        function updateLeg(id, field, value) { const leg = activeLegs.find(l => l.id === id); if(leg) { leg[field] = value; renderBasket(); triggerAnalyze(); } }

        function renderBasket() {
            const tbody = document.getElementById('basket-body');
            if(activeLegs.length === 0) { tbody.innerHTML = '<tr><td colspan="8" style="text-align:center; color:var(--text-secondary); padding: 2rem 0;">No legs added.</td></tr>'; document.getElementById('btn-deploy-strategy').style.display = 'none'; return; }
            document.getElementById('btn-deploy-strategy').style.display = 'flex';
            tbody.innerHTML = '';
            activeLegs.forEach(leg => {
                const tr = document.createElement('tr');
                tr.innerHTML = `
                    <td><div class="action-toggle"><button class="${leg.action === 'BUY' ? 'active buy' : ''}" onclick="updateLeg(${leg.id}, 'action', 'BUY')">B</button><button class="${leg.action === 'SELL' ? 'active sell' : ''}" onclick="updateLeg(${leg.id}, 'action', 'SELL')">S</button></div></td>
                    <td><div class="qty-control"><button onclick="updateLeg(${leg.id}, 'lots', Math.max(1, ${leg.lots}-1))">-</button><span class="font-mono">${leg.lots}</span><button onclick="updateLeg(${leg.id}, 'lots', ${leg.lots}+1)">+</button></div></td>
                    <td>${leg.expiry}</td><td class="font-mono" style="font-weight:600;">${leg.strike}</td>
                    <td><span style="color:${leg.type==='CE'?'var(--accent-blue)':'var(--accent-red)'}; font-weight:600;">${leg.type}</span></td>
                    <td class="font-mono">₹${leg.price.toFixed(2)}</td><td class="font-mono text-secondary">${leg.iv.toFixed(1)}%</td>
                    <td><button class="btn-delete" onclick="removeLeg(${leg.id})"><i data-feather="x" width="16"></i></button></td>
                `; tbody.appendChild(tr);
            });
            feather.replace();
        }
        document.getElementById('btn-clear-basket').onclick = () => { activeLegs = []; renderBasket(); triggerAnalyze(); };

        // Portfolio Manager
        window.activePortfolioData = [];
        async function deployStrategy() {
            if (activeLegs.length === 0) return;
            try { const res = await fetch('/api/portfolio/deploy', { method: 'POST', body: JSON.stringify({ legs: activeLegs, name: lastStrategyName }) }); const data = await res.json();
                if(data.ok) { alert("Strategy Deployed to Paper Trading Portfolio!"); activeLegs = []; renderBasket(); triggerAnalyze(); switchTab('tab-portfolio'); fetchPortfolio(); }
            } catch(e) { console.error(e); }
        }

        async function fetchPortfolio() {
            try { const res = await fetch('/api/portfolio/status'); const data = await res.json();
                if(data.ok) { window.activePortfolioData = data.portfolio.active; renderPortfolio(data.portfolio.active); renderOrderbook(data.portfolio.history); }
            } catch (e) { console.error(e); }
        }

        function loadPosToBuilder(posId) {
            const pos = window.activePortfolioData.find(p => p.id === posId);
            if(pos) { activeLegs = pos.legs.map(l => ({...l, id: Date.now() + Math.random()})); lastStrategyName = pos.name; renderBasket(); triggerAnalyze(); }
        }

        function renderPortfolio(active) {
            const container = document.getElementById('active-positions-container');
            const badge = document.getElementById('pos-badge');
            badge.innerText = active.length; badge.style.display = active.length > 0 ? 'inline-block' : 'none';
            if(active.length === 0) { container.innerHTML = '<div style="color:var(--text-secondary); font-size:0.9rem; text-align:center; padding: 2rem 0;">No active positions deployed.</div>'; return; }
            container.innerHTML = '';
            
            active.forEach(pos => {
                const pnl = pos.live_pnl || 0;
                let pnlClass = ''; const lastPnl = activePositionsDict[pos.id];
                if (lastPnl !== undefined) { if (pnl > lastPnl) pnlClass = 'pulse-green'; else if (pnl < lastPnl) pnlClass = 'pulse-red'; }
                activePositionsDict[pos.id] = pnl;

                const colorClass = pnl >= 0 ? 'text-green' : 'text-red';
                const sign = pnl >= 0 ? '+' : '';
                
                const pop = pos.live_pop || 0;
                let popClass = 'pop-warn';
                if(pop > 55) popClass = 'pop-green'; else if(pop < 45) popClass = 'pop-red';

                let legsHtml = pos.legs.map(l => {
                    const lPnl = l.live_pnl || 0;
                    const lColor = lPnl >= 0 ? 'text-green' : 'text-red';
                    const actColor = l.action==='BUY' ? 'var(--accent-green)' : 'var(--accent-red)';
                    return `<tr><td><span style="color:${actColor}">[${l.action[0]}]</span> ${l.lots}x ${l.strike}${l.type}</td>
                            <td class="font-mono">₹${l.price.toFixed(2)}</td>
                            <td class="font-mono">₹${(l.live_price||l.price).toFixed(2)}</td>
                            <td class="font-mono ${lColor}">${lPnl >= 0 ? '+' : ''}₹${lPnl.toFixed(0)}</td></tr>`;
                }).join('');

                const div = document.createElement('div'); div.className = 'pos-card';
                div.onclick = (e) => { if(!e.target.closest('button')) loadPosToBuilder(pos.id); };
                div.innerHTML = `
                    <div class="pos-header">
                        <div class="pos-name">${pos.name} <span class="pop-badge ${popClass}">POP ${pop.toFixed(0)}%</span></div>
                        <div class="pos-pnl ${colorClass} ${pnlClass}">${sign}₹${pnl.toFixed(2)}</div>
                    </div>
                    <div class="pos-margin">Req. Margin: ₹${Math.round(pos.estimated_margin).toLocaleString()}</div>
                    <table class="pos-leg-table">
                        <thead><tr><th>Leg</th><th>Entry</th><th>CMP</th><th>P&L</th></tr></thead>
                        <tbody>${legsHtml}</tbody>
                    </table>
                    <div class="pos-footer">
                        <span style="font-size:0.7rem; color:var(--text-secondary)">Entered: ${pos.entry_time.split(' ')[1]}</span>
                        <button class="btn-exit" onclick="exitPosition('${pos.id}')">Exit at Market</button>
                    </div>
                `; container.appendChild(div);
            });
        }

        async function exitPosition(id) {
            try { const res = await fetch('/api/portfolio/exit', { method: 'POST', body: JSON.stringify({id}) }); const data = await res.json();
                if(data.ok) { alert(`Exited position! Realized P&L: ₹${data.position.realized_pnl.toFixed(2)}`); fetchPortfolio(); }
            } catch(e) { console.error(e); }
        }

        function renderOrderbook(history) {
            const tbody = document.getElementById('history-body'); tbody.innerHTML = '';
            if(history.length === 0) { tbody.innerHTML = '<tr><td colspan="4" style="text-align:center; padding: 2rem 0; color:var(--text-secondary)">No exited positions.</td></tr>'; return; }
            history.slice().reverse().forEach(pos => {
                const pnl = pos.realized_pnl || 0; const colorClass = pnl >= 0 ? 'text-green' : 'text-red'; const sign = pnl >= 0 ? '+' : '';
                const tr = document.createElement('tr');
                tr.innerHTML = `<td style="font-size:0.8rem; color:var(--text-secondary)">${pos.exit_time}</td><td style="font-weight:600">${pos.name}</td><td class="font-mono">₹${Math.round(pos.estimated_margin).toLocaleString()}</td><td class="font-mono ${colorClass}" style="font-weight:700;">${sign}₹${pnl.toFixed(2)}</td>`;
                tbody.appendChild(tr);
            });
        }

        // Analysis
        function triggerAnalyze() { clearTimeout(debounceTimer); debounceTimer = setTimeout(analyzeStrategy, 200); }
        
        // Time Slider Logic
        function getTimeString(ticks) {
            // ticks 0 to 25. 0 = 09:15. 1 = 09:30.
            const totalMinutes = 9*60 + 15 + ticks * 15;
            const h = Math.floor(totalMinutes / 60); const m = totalMinutes % 60;
            return `${h.toString().padStart(2, '0')}:${m.toString().padStart(2, '0')}`;
        }

        async function analyzeStrategy() {
            const spotShift = parseFloat(document.getElementById('slide-spot').value);
            const daysOffset = parseFloat(document.getElementById('slide-date').value);
            const timeTicks = parseInt(document.getElementById('slide-time').value); // 0 to 25
            
            // Current Time (approx 15:30 if outside market hours, but let's just use ticks)
            // A full trading day is 6h15m (375 mins).
            // Fraction of day = timeTicks / 25
            // So total offset = daysOffset + (1 - timeTicks/25) 
            // We want T_target to be smaller as daysOffset increases.
            
            // To simplify, just pass the fractional days to backend.
            // Backend will do T_now - offset.
            // If slider is T+1 Day, and Time is 15:30, offset is 1.0.
            // If Time is 09:15, that is earlier in the day, meaning more time remaining.
            const fraction = 1 - (timeTicks / 25.0); // 0 at close, 1 at open
            const totalFractionalDays = daysOffset + fraction;

            if (activeLegs.length === 0) { Plotly.react('payoff-chart', [], chartLayout); updateSummary({ max_profit: 0, max_loss: 0, breakevens: [], pop: 0, net_premium: 0 }, { delta:0, gamma:0, theta:0, vega:0 }); return; }

            try {
                const res = await fetch('/api/builder/analyze', {
                    method: 'POST', body: JSON.stringify({ legs: activeLegs, target_date_offset: totalFractionalDays, target_spot_offset: spotShift })
                });
                const data = await res.json();
                if (data.ok && !data.empty) { if(data.atm_iv) lastAtmIv = data.atm_iv; updateSummary(data.summary, data.greeks); plotChart(data.chart); }
            } catch(e) { console.error(e); }
        }

        function updateSummary(summary, greeks) {
            document.getElementById('summary-strategy-name').innerText = lastStrategyName;
            const fmt = v => { if(v === "Infinity" || v === "-Infinity" || v === Infinity || v === -Infinity || v > 999999 || v < -999999) return "Unlimited"; return (v > 0 ? "+" : "") + "₹" + Math.round(v).toLocaleString(); };
            document.getElementById('stat-max-profit').innerText = fmt(summary.max_profit);
            document.getElementById('stat-max-loss').innerText = fmt(summary.max_loss);
            document.getElementById('stat-pop').innerText = summary.pop.toFixed(1) + "%";
            document.getElementById('stat-premium').innerText = fmt(summary.net_premium);
            const bes = summary.breakevens.length > 0 ? summary.breakevens.map(b => b.toFixed(0)).join(', ') : 'None';
            const tbes = summary.target_breakevens && summary.target_breakevens.length > 0 ? summary.target_breakevens.map(b => b.toFixed(0)).join(', ') : 'None';
            document.getElementById('stat-breakevens').innerText = bes;
            document.getElementById('stat-target-breakevens').innerText = tbes;
            
            const pp = document.getElementById('projected-profit');
            if(summary.projected_pnl) {
                pp.innerText = fmt(summary.projected_pnl);
                pp.className = 'font-mono ' + (summary.projected_pnl >= 0 ? 'text-green' : 'text-red');
            } else { pp.innerText = '₹0'; pp.className = 'font-mono'; }
            
            document.getElementById('greek-delta').innerText = greeks.delta.toFixed(2);
            document.getElementById('greek-theta').innerText = greeks.theta.toFixed(2);
            document.getElementById('greek-gamma').innerText = greeks.gamma.toFixed(4);
            document.getElementById('greek-vega').innerText = greeks.vega.toFixed(2);
        }

        let sdDays = 14;
        let lastAtmIv = 15.0;
        function updateSDDays(days) { sdDays = days; document.getElementById('val-sd-days').innerText = days; triggerAnalyze(); }

        function plotChart(chartData) {
            const x = chartData.spot_prices;
            const traceExpiry = { x: x, y: chartData.pnl_expiry, name: 'P&L at Expiry', type: 'scatter', mode: 'lines', line: { color: 'rgba(148, 163, 184, 0.8)', width: 2, dash: 'dash' }, hovertemplate: 'Spot: %{x:.0f}<br>Expiry P&L: ₹%{y:.0f}<extra></extra>' };
            const traceTarget = { x: x, y: chartData.pnl_target, name: 'P&L (Target)', type: 'scatter', mode: 'lines', line: { color: '#3b82f6', width: 4 }, fill: 'tozeroy', fillcolor: 'rgba(59, 130, 246, 0.1)', hovertemplate: 'Spot: %{x:.0f}<br>Target P&L: ₹%{y:.0f}<extra></extra>' };
            
            const tSpot = currentSpot * (1 + parseFloat(document.getElementById('slide-spot').value)/100);
            const sd1 = currentSpot * (lastAtmIv / 100) * Math.sqrt(sdDays / 365);
            
            const shapes = [
                { type: 'line', x0: tSpot, x1: tSpot, y0: 0, y1: 1, yref: 'paper', line: { color: 'rgba(59, 130, 246, 0.6)', width: 2, dash: 'dot' } }, // Target Spot
                { type: 'line', x0: currentSpot, x1: currentSpot, y0: 0, y1: 1, yref: 'paper', line: { color: '#10b981', width: 1, dash: 'dot' } },
                { type: 'rect', x0: currentSpot - sd1, x1: currentSpot + sd1, y0: 0, y1: 1, yref: 'paper', fillcolor: 'rgba(16, 185, 129, 0.15)', line: { width: 0 }, layer: 'below' },
                { type: 'rect', x0: currentSpot - sd1*2, x1: currentSpot + sd1*2, y0: 0, y1: 1, yref: 'paper', fillcolor: 'rgba(239, 68, 68, 0.1)', line: { width: 0 }, layer: 'below' },
                { type: 'line', x0: currentSpot - sd1, x1: currentSpot - sd1, y0: 0, y1: 1, yref: 'paper', line: { color: 'rgba(16, 185, 129, 0.5)', width: 1, dash: 'dash' } },
                { type: 'line', x0: currentSpot + sd1, x1: currentSpot + sd1, y0: 0, y1: 1, yref: 'paper', line: { color: 'rgba(16, 185, 129, 0.5)', width: 1, dash: 'dash' } },
                { type: 'line', x0: currentSpot - sd1*2, x1: currentSpot - sd1*2, y0: 0, y1: 1, yref: 'paper', line: { color: 'rgba(239, 68, 68, 0.5)', width: 1, dash: 'dash' } },
                { type: 'line', x0: currentSpot + sd1*2, x1: currentSpot + sd1*2, y0: 0, y1: 1, yref: 'paper', line: { color: 'rgba(239, 68, 68, 0.5)', width: 1, dash: 'dash' } }
            ];
            const annotations = [
                { x: currentSpot - sd1, y: 1.0, yanchor: 'bottom', yref: 'paper', text: '-1SD', showarrow: false, font: {size: 11, color: '#10b981'} },
                { x: currentSpot + sd1, y: 1.0, yanchor: 'bottom', yref: 'paper', text: '+1SD', showarrow: false, font: {size: 11, color: '#10b981'} },
                { x: currentSpot - sd1*2, y: 1.0, yanchor: 'bottom', yref: 'paper', text: '-2SD', showarrow: false, font: {size: 11, color: '#ef4444'} },
                { x: currentSpot + sd1*2, y: 1.0, yanchor: 'bottom', yref: 'paper', text: '+2SD', showarrow: false, font: {size: 11, color: '#ef4444'} }
            ];
            
            const layout = { ...chartLayout, shapes: shapes, annotations: annotations };
            Plotly.react('payoff-chart', [traceExpiry, traceTarget], layout);
        }

        document.getElementById('slide-spot').addEventListener('input', (e) => { 
            const v = parseFloat(e.target.value); 
            const tSpot = currentSpot * (1 + v/100);
            document.getElementById('val-target-spot').innerText = (v > 0 ? '+' : '') + v + '% | ' + tSpot.toFixed(0); 
            document.getElementById('val-target-spot').className = 'font-mono ' + (v>=0 ? 'text-green' : 'text-red'); 
            triggerAnalyze(); 
        });
        document.getElementById('slide-date').addEventListener('input', (e) => { 
            const daysOffset = parseInt(e.target.value);
            const targetDate = new Date(Date.now() + daysOffset * 86400000);
            const options = { weekday: 'short', month: 'short', day: 'numeric' };
            document.getElementById('val-target-date').innerText = `T+${daysOffset} Days | ` + targetDate.toLocaleDateString('en-US', options); 
            triggerAnalyze(); 
        });
        document.getElementById('slide-time').addEventListener('input', (e) => { document.getElementById('val-target-time').innerText = getTimeString(parseInt(e.target.value)); triggerAnalyze(); });

        async function fetchWizardRecommendation() {
            const btn = document.getElementById('btn-fetch-wizard');
            btn.innerHTML = '<i data-feather="loader" class="spin"></i> Analyzing Dashboard...'; feather.replace();
            try {
                const res = await fetch('/api/wizard', { method: 'POST', body: JSON.stringify({ view: 'NEUTRAL', risk: 'MODERATE', capital: 150000 }) });
                const data = await res.json();
                if(data.ok) {
                    fetchWizardHistory();
                }
            } catch(e) { console.error(e); }
            btn.innerHTML = '<i data-feather="cpu" width="18"></i> Analyze Live Market & Suggest Strategy'; feather.replace();
        }

        async function fetchWizardHistory() {
            try {
                const res = await fetch('/api/wizard/history');
                const data = await res.json();
                if(data.ok) {
                    const cont = document.getElementById('ai-presets-container'); 
                    cont.innerHTML = '';
                    if(data.history.length === 0) {
                        cont.innerHTML = '<div style="color:var(--text-secondary); font-size:0.9rem; text-align:center; padding: 2rem 0;">No recommendations generated today.</div>';
                        return;
                    }
                    // Reverse to show latest first
                    data.history.slice().reverse().forEach(s => {
                        const div = document.createElement('div'); div.className = 'ai-card';
                        div.innerHTML = `<div style="font-size:0.75rem; color:var(--text-secondary); margin-bottom:0.3rem;">${s.time}</div>
                                         <div class="ai-title" style="color:var(--accent-blue)">${s.strategy}</div>
                                         <div class="ai-desc" style="font-style:italic;">"${s.rationale}"</div>
                                         <div style="font-size:0.75rem; color:var(--text-primary); margin-top:0.5rem;"><i data-feather="mouse-pointer" width="12"></i> Click to Load</div>`;
                        div.onclick = () => { lastStrategyName = s.strategy; activeLegs = []; s.legs.forEach(l => addLegRaw(l.action, l.strike, l.opt_type, l.entry_price, l.iv)); renderBasket(); triggerAnalyze(); switchTab('tab-ready'); };
                        cont.appendChild(div);
                    });
                    feather.replace();
                }
            } catch(e) { console.error(e); }
        }

        fetchChain(); fetchPortfolio(); fetchWizardHistory();
        setInterval(fetchChain, 10000);
        setInterval(fetchPortfolio, 3000);
    </script>
</body>
</html>
"""

with open('strategy_builder.html', 'w', encoding='utf-8') as f:
    f.write(html_content)

print("Updated strategy_builder.html for new UI & Sliders")
