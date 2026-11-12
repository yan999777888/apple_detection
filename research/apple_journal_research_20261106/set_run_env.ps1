$ErrorActionPreference = 'Stop'
$taskResearchRoot = $PSScriptRoot
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:PYTHONIOENCODING = 'utf-8'
$taskCachePaths = @{
 TEMP='runtime/tmp'; TMP='runtime/tmp'; TORCH_HOME='runtime/torch';
 YOLO_CONFIG_DIR='runtime/ultralytics'; MPLCONFIGDIR='runtime/matplotlib';
 HF_HOME='runtime/huggingface'; XDG_CACHE_HOME='runtime/cache';
 CUDA_CACHE_PATH='runtime/cuda'; NUMBA_CACHE_DIR='runtime/numba';
 TORCHINDUCTOR_CACHE_DIR='runtime/torchinductor'; TRITON_CACHE_DIR='runtime/triton'
}
foreach ($item in $taskCachePaths.GetEnumerator()) {
 $cachePath=Join-Path $taskResearchRoot $item.Value
 New-Item -ItemType Directory -Path $cachePath -Force | Out-Null
 [Environment]::SetEnvironmentVariable($item.Key,$cachePath,'Process')
}
$OutputEncoding=[Console]::OutputEncoding=[System.Text.UTF8Encoding]::new()
