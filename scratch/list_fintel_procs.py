import subprocess, json

ps_cmd = 'Get-CimInstance Win32_Process | Where-Object { $_.Name -like "*python*" } | Select-Object ProcessId, ParentProcessId, CommandLine | ConvertTo-Json'
res = subprocess.run(["powershell", "-NoProfile", "-Command", ps_cmd], capture_output=True, text=True)
data = json.loads(res.stdout)
if isinstance(data, dict): data = [data]
for p in data:
    cmd = p.get('CommandLine') or ''
    if 'F-Intel' in cmd or 'VolatilityAnalyzer' in cmd or 'FIntelLauncher' in cmd:
        print(f"PID: {p.get('ProcessId')} | Parent: {p.get('ParentProcessId')} | CMD: {cmd}")
