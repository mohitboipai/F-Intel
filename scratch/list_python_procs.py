import subprocess

cmd = """Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" | Select-Object ProcessId, ParentProcessId, CommandLine | Format-Table -AutoSize"""
res = subprocess.run(["powershell", "-NoProfile", "-Command", cmd], capture_output=True, text=True)
print(res.stdout)
