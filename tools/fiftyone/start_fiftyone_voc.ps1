$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$pythonExe = "D:\Miniconda3\envs\fiftyone312\python.exe"
$importScript = Join-Path $projectRoot "tools\fiftyone\fiftyone_import_voc.py"
$dataDir = "F:\1\labelimg\data\test1_stride10\fiftyone_voc\data"
$labelsDir = "F:\1\labelimg\data\test1_stride10\fiftyone_voc\labels"
$datasetName = "live_app_check"
$sessionUrl = "http://localhost:5151/"

Write-Host ""
Write-Host "=== FiftyOne VOC Launcher ===" -ForegroundColor Cyan
Write-Host "Python      : $pythonExe"
Write-Host "Dataset     : $datasetName"
Write-Host "Data dir    : $dataDir"
Write-Host "Labels dir  : $labelsDir"
Write-Host "Browser URL : $sessionUrl"
Write-Host ""
Write-Host "Stop hint   : press Ctrl + C in this window to stop the FiftyOne session" -ForegroundColor Yellow
Write-Host ""

if (-not (Test-Path -LiteralPath $pythonExe)) {
    throw "Python executable not found: $pythonExe"
}

if (-not (Test-Path -LiteralPath $importScript)) {
    throw "Import script not found: $importScript"
}

Push-Location $projectRoot
try {
    & $pythonExe $importScript `
        --name $datasetName `
        --data-dir $dataDir `
        --labels-dir $labelsDir `
        --overwrite `
        --wait
}
finally {
    Pop-Location
}
