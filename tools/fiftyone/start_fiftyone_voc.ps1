$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$workspaceRoot = Split-Path -Parent (Split-Path -Parent $projectRoot)

# Load .env if present
$envFile = Join-Path $projectRoot ".env"
if (Test-Path -LiteralPath $envFile) {
    Get-Content $envFile | ForEach-Object {
        $line = $_.Trim()
        if ($line -and -not $line.StartsWith("#") -and $line -match "^([^#=]+)=(.*)$") {
            $key = $Matches[1].Trim()
            $val = $Matches[2].Trim().Trim('"').Trim("'")
            if (-not [Environment]::GetEnvironmentVariable($key)) {
                [Environment]::SetEnvironmentVariable($key, $val, "Process")
            }
        }
    }
}

$datasetDir = if ($env:YOLO_DATASET_DIR) { $env:YOLO_DATASET_DIR } else { Join-Path $workspaceRoot "data\labelimg\test1_stride10" }
$pythonExe = "python"
$importScript = Join-Path $projectRoot "tools\fiftyone\fiftyone_import_voc.py"
$dataDir = Join-Path $datasetDir "fiftyone_voc\data"
$labelsDir = Join-Path $datasetDir "fiftyone_voc\labels"
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
