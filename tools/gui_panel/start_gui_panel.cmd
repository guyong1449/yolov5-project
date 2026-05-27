@echo off
setlocal
set "REPO=F:\1\yolov5-master"
set "PWSH=C:\Program Files\PowerShell\7\pwsh.exe"
set "PYTHON=D:\Miniconda3\envs\f312\python.exe"
set "URL=http://127.0.0.1:8752/"
set "CMD=Set-Location -LiteralPath '%REPO%'; Start-Process -FilePath '%PWSH%' -WindowStyle Hidden -ArgumentList '-NoLogo','-NoProfile','-Command','Start-Sleep -Seconds 2; Start-Process ''%URL%'''; & '%PYTHON%' tools/gui_panel/start_gui_panel.py"
"%PWSH%" -NoLogo -NoProfile -NoExit -Command "%CMD%"
endlocal
