@echo off
setlocal

echo Stopping the 189 reverse proxy tunnel...
powershell -NoProfile -Command ^
  "$procs = Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'ssh.exe' -and $_.CommandLine -like '*-R 17890:127.0.0.1:7890 189*' }; " ^
  "if (-not $procs) { Write-Host 'No matching tunnel process found.'; exit 0 }; " ^
  "$procs | ForEach-Object { Stop-Process -Id $_.ProcessId -Force; Write-Host ('Stopped PID ' + $_.ProcessId) }"

echo Done.
pause
