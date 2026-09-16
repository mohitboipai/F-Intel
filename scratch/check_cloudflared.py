import subprocess

cmd = """Get-CimInstance Win32_Process -Filter "Name like 'cloudflared%'" | Select-Object ProcessId, ParentProcessId, CommandLine | Format-List"""
res = subprocess.run(["powershell", "-NoProfile", "-Command", cmd], capture_output=True, text=True)
print(res.stdout)
