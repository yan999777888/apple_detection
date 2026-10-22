param([ValidateSet('check','stress','train','resume','feedback')][string]$Action = 'check')
. "$PSScriptRoot/runtime.ps1"
if (!(Test-Path -LiteralPath $Python)) { throw 'Run 01_setup.cmd first.' }
$Config = Join-Path $RepoRoot 'configs/pilot.json'
switch ($Action) {
    'check' { & $Python (Join-Path $RepoRoot 'scripts/preflight.py') --config $Config --require-cuda --verify-all-assets }
    'stress' {
        $Stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
        & $Python (Join-Path $RepoRoot 'scripts/train_mixed.py') --config $Config --device cuda --smoke --max-updates 2 --input-size 768 --output (Join-Path $RepoRoot "runs/gpu_preflight_$Stamp")
    }
    'train' { & $Python (Join-Path $RepoRoot 'scripts/train_mixed.py') --config $Config --device cuda }
    'resume' { & $Python (Join-Path $RepoRoot 'scripts/train_mixed.py') --config $Config --device cuda --resume }
    'feedback' { & $Python (Join-Path $RepoRoot 'scripts/collect_feedback.py') --root $RepoRoot }
}
if ($LASTEXITCODE -ne 0) { throw "Action $Action failed (exit $LASTEXITCODE). Run 05_feedback.cmd to collect evidence." }
