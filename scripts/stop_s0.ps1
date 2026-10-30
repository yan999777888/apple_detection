$ErrorActionPreference = 'Stop'
$AppleRepo = Split-Path -Parent $PSScriptRoot
$AppleConfig = Get-Content -LiteralPath (Join-Path $AppleRepo 'configs\query_s0_4060.json') -Raw -Encoding UTF8 | ConvertFrom-Json
$AppleRun = [System.IO.Path]::GetFullPath((Join-Path $AppleRepo $AppleConfig.paths.output))
if (-not $AppleRun.StartsWith($AppleRepo + '\', [System.StringComparison]::OrdinalIgnoreCase)) { throw 'Run path escaped repository' }
if (-not (Test-Path -LiteralPath $AppleRun)) { throw 'S0 run directory does not exist' }
[System.IO.File]::WriteAllText((Join-Path $AppleRun 'STOP'), (Get-Date -Format o))
Write-Output 'Stop requested. The current update or full validation will finish, then last.pt will be saved.'
