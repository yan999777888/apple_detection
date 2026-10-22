param([string]$BasePython = '')
. "$PSScriptRoot/runtime.ps1"
$Reports = Join-Path $RepoRoot 'reports'
New-Item -ItemType Directory -Path $Reports -Force | Out-Null
Start-Transcript -Path (Join-Path $Reports ("setup_" + (Get-Date -Format 'yyyyMMdd_HHmmss') + '.log')) | Out-Null
try {
if (!(Test-Path -LiteralPath $Python)) {
    if ($BasePython) { & $BasePython -m venv (Join-Path $RepoRoot '.venv') }
    elseif (Get-Command py -ErrorAction SilentlyContinue) { & py -3.11 -m venv (Join-Path $RepoRoot '.venv') }
    elseif (Get-Command python -ErrorAction SilentlyContinue) {
        & python -c "import sys; assert sys.version_info[:2] == (3,11), 'Python 3.11 required'"
        if ($LASTEXITCODE -ne 0) { throw 'Install Python 3.11 or pass -BasePython.' }
        & python -m venv (Join-Path $RepoRoot '.venv')
    } else { throw 'Python 3.11 not found. Use: powershell -File scripts/setup.ps1 -BasePython PATH_TO_PYTHON_EXE' }
    if ($LASTEXITCODE -ne 0) { throw 'Creating environment failed.' }
}
& $Python -c "import sys; assert sys.version_info[:2] == (3,11), 'Python 3.11 required'"
if ($LASTEXITCODE -ne 0) { throw 'Wrong Python version.' }
& $Python -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw 'pip upgrade failed.' }
& $Python -m pip install 'torch==2.12.1+cu132' 'torchvision==0.27.1+cu132' --index-url https://download.pytorch.org/whl/cu132
if ($LASTEXITCODE -ne 0) { throw 'Installing CUDA PyTorch failed. Check network and NVIDIA driver.' }
& $Python -m pip install -r (Join-Path $RepoRoot 'requirements.txt')
if ($LASTEXITCODE -ne 0) { throw 'Installing requirements failed.' }
& $Python -m pip check
if ($LASTEXITCODE -ne 0) { throw 'Dependency check failed.' }
Write-Host 'Environment ready. Run 02_check.cmd next.'
} finally {
    Stop-Transcript | Out-Null
}
