$ErrorActionPreference = 'Stop'
$researchRoot = [System.IO.Path]::GetFullPath($PSScriptRoot)
if ($researchRoot -ne 'F:\研究生毕业设计\模块一定位补偿以及预测\apple_journal_research_20261106') { throw 'Unexpected research root' }
Set-Location -LiteralPath $researchRoot
. ./set_run_env.ps1
$env:PYTHONHASHSEED = '20261106'
$taskPython = 'F:\研究生毕业设计\模块一定位补偿以及预测\apple_detection\envs\local_rgbd_v1\Scripts\python.exe'
$eventPath = Join-Path $researchRoot 'experiments/papple_continuation_v1_events.jsonl'
if (Test-Path -LiteralPath $eventPath) { throw 'Continuation already started; no automatic restart' }
function Invoke-RecordedPython {
    param([string]$Step, [string[]]$Arguments)
    $logPath = Join-Path $researchRoot ('logs/papple_continuation_v1_' + $Step + '.log')
    @{utc=[DateTime]::UtcNow.ToString('o');phase='step_start';step=$Step;argv=$Arguments;log=$logPath} | ConvertTo-Json -Compress -Depth 5 | Add-Content -LiteralPath $eventPath -Encoding utf8
    & $taskPython @Arguments *> $logPath
    $stepExit = $LASTEXITCODE
    @{utc=[DateTime]::UtcNow.ToString('o');phase='step_end';step=$Step;exit_code=$stepExit} | ConvertTo-Json -Compress | Add-Content -LiteralPath $eventPath -Encoding utf8
    if ($stepExit -ne 0) { throw ('Step failed: ' + $Step + ', exit ' + $stepExit) }
}
try {
    Invoke-RecordedPython -Step 'train40' -Arguments @('./apple_research_v2/train_yolo_full.py','--task','segment','--data','./apple_research_v2/datasets/papple_visible_stage_seg_v1/data.yaml','--manifest','./apple_research_v2/datasets/papple_visible_stage_seg_v1/conversion_audit.json','--expected-manifest-sha256','9e93f9a8a86018cc9567f284c74a4ecc0c6516bba5e96e825a5ae29f46eff825','--weights','./assets/ultralytics_yolo26_coco_20261106/yolo26n-seg.pt','--expected-weights-sha256','361fbfabab285c3237700b6bb91d7ecfa602cd945fffda8dbe1242829b71e73f','--run-name','papple_yolo26n_seg_640_e40_s20261106','--mask-ratio','4')
    $statusPath = './runs/papple_yolo26n_seg_640_e40_s20261106/status.json'
    $trainingState = Get-Content -LiteralPath $statusPath -Raw -Encoding utf8 | ConvertFrom-Json
    if ($trainingState.phase -ne 'complete' -or $trainingState.epochs_completed -ne 40) { throw 'Training did not complete expected budget' }
    $selection = Get-Content -LiteralPath './runs/papple_yolo26n_seg_640_e40_s20261106/final_selection.json' -Raw -Encoding utf8 | ConvertFrom-Json
    $lastPath = [System.IO.Path]::GetFullPath($selection.primary.path)
    if (-not $lastPath.StartsWith($researchRoot + '\', [StringComparison]::OrdinalIgnoreCase)) { throw 'Unexpected checkpoint boundary' }
    $lastHash = (Get-FileHash -LiteralPath $lastPath -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($lastHash -ne $selection.primary.sha256) { throw 'Checkpoint hash mismatch' }
    Invoke-RecordedPython -Step 'common8_benchmark' -Arguments @('./apple_research_v2/benchmark_yolo_pipeline.py','--weights',$lastPath,'--data','./experiments/benchmarks/global_common8_input/data.yaml','--output','./apple_research_v2/benchmarks/papple_common8_native640_v1','--num-frames','8','--repeats','5','--warmup','5','--imgsz','640')
    Invoke-RecordedPython -Step 'dev40_benchmark' -Arguments @('./apple_research_v2/benchmark_yolo_pipeline.py','--weights',$lastPath,'--data','./apple_research_v2/datasets/papple_visible_stage_seg_v1/data.yaml','--output','./apple_research_v2/benchmarks/papple_dev40_native640_v1','--num-frames','40','--repeats','3','--warmup','5','--imgsz','640')
    Invoke-RecordedPython -Step 'onnx_export' -Arguments @('./apple_research_v2/export_onnx_checked.py','--weights',$lastPath,'--expected-weights-sha256',$lastHash,'--training-status',$statusPath,'--task','segment','--output','./exports/papple_yolo26n_seg_640_e40_fp32')
    @{utc=[DateTime]::UtcNow.ToString('o');phase='queue_complete';onnx_inference_parity_complete=$false;independent_papple_test=$false} | ConvertTo-Json -Compress | Add-Content -LiteralPath $eventPath -Encoding utf8
} catch {
    @{utc=[DateTime]::UtcNow.ToString('o');phase='queue_failed';error=$_.Exception.Message;automatic_retry=$false} | ConvertTo-Json -Compress | Add-Content -LiteralPath $eventPath -Encoding utf8
    throw
}
