import subprocess

try:
    cmd = 'powershell "Get-CimInstance Win32_Process | Where-Object { $_.Name -like \'*python*\' } | Select-Object ProcessId, CommandLine"'
    res = subprocess.check_output(cmd, shell=True, encoding='utf-8')
    print(res)
except Exception as e:
    print('Error:', e)
