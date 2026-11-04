param([string]$ConfigPath = 'configs\local_formal_papple_20261104.json')
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'runtime_local_rgbd.ps1')
$FormalConfig = Get-Content -LiteralPath (Join-Path $LocalRgbdRepo $ConfigPath) -Raw -Encoding UTF8 | ConvertFrom-Json
$FormalOutput = Join-Path $LocalRgbdRepo $FormalConfig.output
if ((Test-Path -LiteralPath (Join-Path $FormalOutput 'pipeline_status.json')) -or
    (Test-Path -LiteralPath (Join-Path $LocalRgbdRepo 'reports\local_formal_pipeline.lock')) -or
    (Test-Path -LiteralPath (Join-Path $LocalRgbdRepo 'reports\local_training_gpu.lock'))) {
    throw 'Existing pipeline or GPU lock found. Inspect it; do not launch a duplicate.'
}
# Arguments are repository-relative and contain no spaces; the working directory
# and interpreter paths are passed as structured Start-Process parameters.
$FormalArguments = @('scripts\local_formal_pipeline.py', '--config', $ConfigPath)
if ($ConfigPath -match '\s') { throw 'Use a repository-relative config path without spaces.' }
$FormalJob = Start-Process -FilePath $LocalRgbdPython -ArgumentList $FormalArguments `
    -WorkingDirectory $LocalRgbdRepo -WindowStyle Hidden -PassThru `
    -RedirectStandardOutput (Join-Path $LocalRgbdRepo 'reports\local_formal_20261104\controller.stdout.log') `
    -RedirectStandardError (Join-Path $LocalRgbdRepo 'reports\local_formal_20261104\controller.stderr.log')
@{ launcher_pid = $FormalJob.Id; config = $ConfigPath; status = (Join-Path $FormalOutput 'pipeline_status.json') } | ConvertTo-Json
