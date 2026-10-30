param(
    [ValidateSet('pressure','train','resume')]
    [string]$Action = 'pressure',
    [switch]$Background
)
. (Join-Path $PSScriptRoot 'runtime_f_drive.ps1')
$AppleConfig = Join-Path $AppleRepo ('configs\' + $(if ($Action -eq 'pressure') { 'query_s0_pressure_4060.json' } else { 'query_s0_4060.json' }))
$AppleEntry = Join-Path $PSScriptRoot 'train_s0.py'
$AppleArguments = @('-B', ('"' + $AppleEntry + '"'), '--config', ('"' + $AppleConfig + '"'), '--device', 'cuda')
if ($Action -eq 'pressure') { $AppleArguments += @('--smoke','--max-updates','2') }
if ($Action -eq 'resume') {
    $AppleArguments += '--resume'
    $AppleResolvedConfig = Get-Content -LiteralPath $AppleConfig -Raw -Encoding UTF8 | ConvertFrom-Json
    $AppleRun = [System.IO.Path]::GetFullPath((Join-Path $AppleRepo $AppleResolvedConfig.paths.output))
    if (-not $AppleRun.StartsWith($AppleRepo + '\', [System.StringComparison]::OrdinalIgnoreCase)) { throw 'Run path escaped repository' }
    $AppleStop = Join-Path $AppleRun 'STOP'
    if (Test-Path -LiteralPath $AppleStop) { Remove-Item -LiteralPath $AppleStop }
}
if ($Background) {
    $AppleLogDirectory = Join-Path $AppleRepo 'logs'
    New-Item -ItemType Directory -Path $AppleLogDirectory -Force | Out-Null
    $AppleStamp = Get-Date -Format 'yyyyMMdd_HHmmss'
    $AppleStdout = Join-Path $AppleLogDirectory ("s0_${Action}_${AppleStamp}.stdout.log")
    $AppleStderr = Join-Path $AppleLogDirectory ("s0_${Action}_${AppleStamp}.stderr.log")
    $AppleProcess = Start-Process -FilePath $ApplePython -ArgumentList $AppleArguments -WorkingDirectory $AppleRepo -WindowStyle Hidden -PassThru -RedirectStandardOutput $AppleStdout -RedirectStandardError $AppleStderr
    $AppleStarted = [pscustomobject]@{action=$Action; pid=$AppleProcess.Id; started_local=(Get-Date -Format o); stdout=$AppleStdout; stderr=$AppleStderr; config=$AppleConfig}
    $AppleStarted | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $AppleLogDirectory 'latest_s0_launch.json') -Encoding UTF8
    $AppleStarted | ConvertTo-Json -Compress
} else {
    $AppleDirectArgs = @('-B',$AppleEntry,'--config',$AppleConfig,'--device','cuda')
    if ($Action -eq 'pressure') { $AppleDirectArgs += @('--smoke','--max-updates','2') }
    if ($Action -eq 'resume') { $AppleDirectArgs += '--resume' }
    & $ApplePython @AppleDirectArgs
    exit $LASTEXITCODE
}
