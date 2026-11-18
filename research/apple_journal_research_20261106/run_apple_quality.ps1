<#
.SYNOPSIS
Produce an annotated apple image and per-fruit quality, maturity and visible defect predictions.
.PARAMETER Image
Path to an image. By default apples must be detected in the scene before grading.
.PARAMETER SingleFruit
Use only for an image already known to be a single fruit. Grade the full image as one target.
.PARAMETER LongEdge
Maximum inference image edge. Use 1280 for the measured camera-size mode or zero for native size.
.PARAMETER GradeDirectory
Optional directory containing a frozen grade candidate and its selection receipt.
.PARAMETER OpenCVThreads
Number of CPU threads for high-resolution frame preparation. The default is one.
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory, Position = 0)]
    [ValidateScript({ Test-Path -LiteralPath $_ -PathType Leaf })]
    [string]$Image,
    [switch]$SingleFruit,
    [ValidateSet(0, 1280)]
    [int]$LongEdge = 0,
    [ValidateSet(1, 8)]
    [int]$OpenCVThreads = 1,
    [ValidateScript({ Test-Path -LiteralPath $_ -PathType Container })]
    [string]$GradeDirectory
)

$ErrorActionPreference = 'Stop'
$sourceImage = (Resolve-Path -LiteralPath $Image).Path
$python = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..\apple_detection\envs\local_rgbd_v1\Scripts\python.exe')).Path
$entry = Join-Path $PSScriptRoot 'apple_research_v2\afruit_visible_quality.py'
$runName = (Get-Date -Format 'yyyyMMdd_HHmmss_fff') + '_' + [guid]::NewGuid().ToString('N').Substring(0, 6)
$destination = Join-Path $PSScriptRoot "experiments\afruit_visible_quality_v1\predictions\$runName"
$arguments = @('-B', $entry, '--image', $sourceImage, '--output', $destination, '--opencv-threads', $OpenCVThreads.ToString())
if ($SingleFruit) {
    $arguments += '--known-single-fruit'
}
if ($LongEdge) {
    $arguments += @('--long-edge', $LongEdge.ToString())
}
if ($GradeDirectory) {
    $arguments += @('--grade-directory', (Resolve-Path -LiteralPath $GradeDirectory).Path)
}

Push-Location -LiteralPath $PSScriptRoot
try {
    . (Join-Path $PSScriptRoot 'set_run_env.ps1')
    & $python @arguments
    if ($LASTEXITCODE -ne 0) { throw "Inference exited with code $LASTEXITCODE" }
    Get-Item -LiteralPath (Join-Path $destination 'prediction.jpg'), (Join-Path $destination 'prediction.json') |
        Select-Object -ExpandProperty FullName
} finally {
    Pop-Location
}
