param([int]$AdoptBasePid = 0)
. (Join-Path $PSScriptRoot 'runtime_f_drive.ps1')
$AppleReportDirectory = Join-Path $AppleRepo 'reports\strong_c0c1_20261030'
New-Item -ItemType Directory -Path $AppleReportDirectory -Force | Out-Null
$AppleEntry = Join-Path $PSScriptRoot 'strong_pipeline.py'
$AppleArgs = @('-B',('"' + $AppleEntry + '"'))
if ($AdoptBasePid -gt 0) { $AppleArgs += @('--adopt-base-pid', "$AdoptBasePid") }
$AppleStamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$AppleStdout = Join-Path $AppleReportDirectory ("pipeline_${AppleStamp}.stdout.log")
$AppleStderr = Join-Path $AppleReportDirectory ("pipeline_${AppleStamp}.stderr.log")
$AppleProcess = Start-Process -FilePath $ApplePython -ArgumentList $AppleArgs -WorkingDirectory $AppleRepo -WindowStyle Hidden -PassThru -RedirectStandardOutput $AppleStdout -RedirectStandardError $AppleStderr
$AppleLaunch = [ordered]@{pid=$AppleProcess.Id; started_local=(Get-Date -Format o); adopted_base_pid=$AdoptBasePid; stdout=$AppleStdout; stderr=$AppleStderr}
$AppleLaunch | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $AppleReportDirectory 'pipeline_launch.json') -Encoding UTF8
$AppleLaunch | ConvertTo-Json -Compress
