@echo off
setlocal

set "SCRIPT_DIR=%~dp0"
powershell.exe -ExecutionPolicy Bypass -File "%SCRIPT_DIR%start_fiftyone_voc.ps1"

endlocal
