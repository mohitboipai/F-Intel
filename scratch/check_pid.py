import subprocess

cmd = """Get-CimInstance Win32_Process -Filter "ProcessId = 30308" | Select-Object ProcessId, Name, CommandLine | Format-List"""
res = subprocess.run(["powershell", "-NoProfile", "-Command", cmd], capture_output=True, text=True)
print(res.stdout)
