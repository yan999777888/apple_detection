$ErrorActionPreference = 'Stop'
$LocalRgbdRepo = Split-Path -Parent $PSScriptRoot
$LocalRgbdPython = Join-Path $LocalRgbdRepo 'envs\local_rgbd_v1\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $LocalRgbdPython)) { throw 'Create the F-drive local_rgbd_v1 environment first.' }
$env:PYTHONPATH = Join-Path $LocalRgbdRepo 'src'
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONUNBUFFERED = '1'
$env:PYTHONNOUSERSITE = '1'
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:PYTHONPYCACHEPREFIX = Join-Path $LocalRgbdRepo 'cache\local_rgbd_pycache'
$env:YOLO_CONFIG_DIR = Join-Path $LocalRgbdRepo 'cache\ultralytics_local_rgbd'
$env:TORCH_HOME = Join-Path $LocalRgbdRepo 'cache\torch'
$env:HF_HOME = Join-Path $LocalRgbdRepo 'cache\huggingface'
$env:PIP_CACHE_DIR = Join-Path $LocalRgbdRepo 'cache\pip'
$env:XDG_CACHE_HOME = Join-Path $LocalRgbdRepo 'cache'
$env:CUDA_CACHE_PATH = Join-Path $LocalRgbdRepo 'cache\cuda'
$env:MPLCONFIGDIR = Join-Path $LocalRgbdRepo 'cache\matplotlib_local_rgbd'
$env:TORCH_EXTENSIONS_DIR = Join-Path $LocalRgbdRepo 'cache\torch_extensions'
$env:TEMP = Join-Path $LocalRgbdRepo 'tmp\local_rgbd'
$env:TMP = $env:TEMP
$env:OMP_NUM_THREADS = '4'
$env:MKL_NUM_THREADS = '4'
foreach ($LocalRgbdDirectory in @($env:TEMP, $env:CUDA_CACHE_PATH, $env:YOLO_CONFIG_DIR, $env:MPLCONFIGDIR, $env:TORCH_EXTENSIONS_DIR)) {
    New-Item -ItemType Directory -Path $LocalRgbdDirectory -Force | Out-Null
}
# This entry point reads the existing falldet dependencies. Never upgrade that shared environment.
