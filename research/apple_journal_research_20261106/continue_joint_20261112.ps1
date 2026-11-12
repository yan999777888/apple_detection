$ErrorActionPreference='Stop'
$root=$PSScriptRoot
Set-Location -LiteralPath $root
. ./set_run_env.ps1
$env:PYTHONNOUSERSITE='1'
$env:PYTHONHASHSEED='20261106'
$python='F:\研究生毕业设计\模块一定位补偿以及预测\apple_detection\envs\local_rgbd_v1\Scripts\python.exe'
$folder=Join-Path $root 'experiments/restart_20261112'
$events=Join-Path $folder 'events.jsonl'
if(Test-Path -LiteralPath $events){throw 'This queue was already started; inspect state before restarting'}
function Invoke-Step([string]$name,[string[]]$argv){
  $log=Join-Path $root ('logs/restart_20261112_'+$name+'.log')
  @{utc=[DateTime]::UtcNow.ToString('o');phase='step_start';step=$name;argv=$argv;log=$log} | ConvertTo-Json -Compress -Depth 6 | Add-Content -LiteralPath $events -Encoding utf8
  & $python -B @argv *> $log
  $exit=$LASTEXITCODE
  @{utc=[DateTime]::UtcNow.ToString('o');phase='step_end';step=$name;exit_code=$exit} | ConvertTo-Json -Compress | Add-Content -LiteralPath $events -Encoding utf8
  if($exit -ne 0){throw ('Step failed: '+$name+' exit '+$exit)}
}
try {
 if(@(Get-Process python*,yolo* -ErrorAction SilentlyContinue).Count -gt 0){throw 'Python process already running; do not overlap training'}
 $checkpoint='./runs/joint_shared_e40_s20261106_v1/checkpoints/epoch_004.pt'
 if((Get-FileHash -LiteralPath $checkpoint -Algorithm SHA256).Hash.ToLowerInvariant() -ne 'b79015609bf126f34ced2a1f9e9780154f4b7f7cf2cdb2f387770a8d0a98d675'){throw 'Checkpoint checksum mismatch'}
 $common=@('./apple_research_v2/joint/train_joint_full_v1.py','train','--init-checkpoint','./experiments/joint_training_v1/canonical_init_seed20261106.pt','--device','cuda:0')
 Invoke-Step 'joint_resume' ($common+@('--mode','joint','--output','./runs/joint_shared_e40_s20261106_v1','--resume',$checkpoint))
 Invoke-Step 'separate_control' ($common+@('--mode','separate','--output','./runs/separate_matched_e40_s20261112_v1'))
 @{utc=[DateTime]::UtcNow.ToString('o');phase='queue_complete'} | ConvertTo-Json -Compress | Add-Content -LiteralPath $events -Encoding utf8
} catch {
 @{utc=[DateTime]::UtcNow.ToString('o');phase='queue_failed';error=$_.Exception.Message} | ConvertTo-Json -Compress | Add-Content -LiteralPath $events -Encoding utf8
 throw
}
