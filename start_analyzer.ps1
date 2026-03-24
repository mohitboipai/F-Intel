# start_analyzer.ps1
# Automates the startup sequence for the Volatility Analyzer System

Write-Host "=========================================" -ForegroundColor Cyan
Write-Host "  Starting Volatility Analyzer System" -ForegroundColor Cyan
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host ""

# 1. Start the Data Server in a new background window
Write-Host "1. Launching DataHub Server (Background)..." -ForegroundColor Yellow
Start-Process powershell -WindowStyle Minimized -ArgumentList "-NoExit -Command `"cd '$PSScriptRoot'; .\.venv\Scripts\python.exe DataServer.py`""
# Give it a moment to initialize the Fyers WebSocket
Start-Sleep -Seconds 3

# 2. Start the main Volatility Analyzer interactively in the current window
Write-Host "2. Launching Volatility Analyzer Dashboard..." -ForegroundColor Green
Write-Host ""
& .\.venv\Scripts\python.exe VolatilityAnalyzer.py
