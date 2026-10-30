$ErrorActionPreference = 'Stop'
$AppleRepo = Split-Path -Parent $PSScriptRoot
$AppleWorkspace = Split-Path -Parent $AppleRepo
$ApplePython = Join-Path $AppleWorkspace 'apple_vision\local_amodal\envs\query_dual_v1\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $ApplePython)) { throw "Checked F-drive Python environment missing: $ApplePython" }
$env:PYTHONPATH = Join-Path $AppleRepo 'src'
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONUNBUFFERED = '1'
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:PYTHONPYCACHEPREFIX = Join-Path $AppleRepo 'cache\pycache'
$env:HF_HOME = Join-Path $AppleRepo 'cache\huggingface'
$env:TORCH_HOME = Join-Path $AppleRepo 'cache\torch'
$env:PIP_CACHE_DIR = Join-Path $AppleRepo 'cache\pip'
$env:CUDA_CACHE_PATH = Join-Path $AppleRepo 'cache\cuda'
$env:XDG_CACHE_HOME = Join-Path $AppleRepo 'cache'
$env:TEMP = Join-Path $AppleRepo 'tmp'
$env:TMP = $env:TEMP
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
$env:TOKENIZERS_PARALLELISM = 'false'
$env:OMP_NUM_THREADS = '4'
$env:MKL_NUM_THREADS = '4'
foreach ($AppleDirectory in @($env:TEMP,$env:CUDA_CACHE_PATH,$env:HF_HOME,$env:TORCH_HOME)) {
    New-Item -ItemType Directory -Path $AppleDirectory -Force | Out-Null
}
