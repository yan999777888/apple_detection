param(
    [string]$ConfigPath,
    [string]$OutputPath,
    [switch]$InspectDataOnly
)

$ErrorActionPreference = 'Stop'
$SmokeRepo = Split-Path -Parent $PSScriptRoot
. (Join-Path $PSScriptRoot 'runtime_local_rgbd.ps1')
if (-not $ConfigPath) {
    $ConfigPath = Join-Path $SmokeRepo 'configs\local_apple_smoke_20261104.json'
}
$SmokeConfig = Get-Content -LiteralPath $ConfigPath -Raw -Encoding UTF8 | ConvertFrom-Json
if ($SmokeConfig.schema -ne 'local_apple_visible_training_smoke_v1') {
    throw 'Unsupported local training smoke configuration.'
}
if (($SmokeConfig.steps -lt 1) -or ($SmokeConfig.steps -gt 32)) {
    throw 'This entry permits 1..32 smoke steps only.'
}
if ($SmokeConfig.workers -ne 0 -or $SmokeConfig.pin_memory -ne $false -or $SmokeConfig.amp -ne $true) {
    throw 'This entry requires workers=0, pin_memory=False and AMP=True.'
}
if (-not $OutputPath) { $OutputPath = Join-Path $SmokeRepo $SmokeConfig.output }
$SmokeOutput = [System.IO.Path]::GetFullPath($OutputPath)
if ([System.IO.Path]::GetPathRoot($SmokeOutput) -ne 'F:\') {
    throw 'The output must be on F:.'
}
if (Test-Path -LiteralPath $SmokeOutput) {
    if (@(Get-ChildItem -LiteralPath $SmokeOutput -Force).Count -gt 0) {
        throw 'The output folder is not empty; select a new -OutputPath.'
    }
}
$SmokeArguments = @(
    (Join-Path $PSScriptRoot 'train_local_segmenter.py'),
    '--manifest', (Join-Path $SmokeRepo $SmokeConfig.train_manifest),
    '--weights', (Join-Path $SmokeRepo $SmokeConfig.initial_weights),
    '--output', $SmokeOutput,
    '--steps', [string]$SmokeConfig.steps,
    '--batch-size', [string]$SmokeConfig.batch_size,
    '--imgsz', [string]$SmokeConfig.image_size,
    '--mask-ratio', [string]$SmokeConfig.mask_ratio,
    '--seed', [string]$SmokeConfig.seed,
    '--lr', $SmokeConfig.learning_rate.ToString([System.Globalization.CultureInfo]::InvariantCulture),
    '--amp-init-scale', [string]$SmokeConfig.amp_initial_scale,
    '--device', 'cuda:0'
)
if ($InspectDataOnly) { $SmokeArguments += '--inspect-data-only' }
Push-Location -LiteralPath $SmokeRepo
try {
    & $LocalRgbdPython @SmokeArguments
    if ($LASTEXITCODE -ne 0) { throw "Local smoke process failed with exit code $LASTEXITCODE." }
} finally {
    Pop-Location
}
