from __future__ import annotations
import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
import json,hashlib,time,copy,sys
from pathlib import Path
from datetime import datetime,timezone
ROOT = Path(__file__).resolve().parents[1]
if ROOT.drive.upper() != "F:":
 raise RuntimeError("This local recovery probe and its evidence must stay on F:")
sys.path.insert(0, str(ROOT / "src"))
for key, suffix in {"TEMP":"tmp/local_rgbd", "TMP":"tmp/local_rgbd", "YOLO_CONFIG_DIR":"cache/ultralytics_local_training", "TORCH_HOME":"cache/torch", "MPLCONFIGDIR":"cache/matplotlib_local_rgbd"}.items():
 location = ROOT / suffix; location.mkdir(parents=True, exist_ok=True); os.environ[key] = str(location)
import torch,cv2
from apple_local_rgbd.local_training import load_local_checkpoint,collate_native,compute_e2e_loss
from apple_local_rgbd.formal_training import PreparedVisibleDataset,restore_criterion

torch.set_num_threads(2);cv2.setNumThreads(1)
root=ROOT; out=root/'reports/local_formal_20261104/recovery_v3'
ckpt=root/'runs/local_visible_papple_20261104/recovery_v3/train_to_002/checkpoints/epoch_002.pt'
source=root/'runs/local_visible_papple_20261104/train_to_001/checkpoints/epoch_001.pt'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
original_sha=sha(ckpt);source_sha=sha(source)
c=torch.load(ckpt,map_location='cpu',weights_only=False)
a=torch.load(source,map_location='cpu',weights_only=False)
m=load_local_checkpoint(ckpt);frozen=set(c['initialization']['frozen_parameter_names'])
for name,param in m.named_parameters():param.requires_grad_(name not in frozen)
m.train();m.criterion=m.init_criterion();restore_criterion(m.criterion,c['formal_training']['criterion'])
named=[(n,p) for n,p in m.named_parameters() if p.requires_grad]
ids=c['formal_training']['optimizer']['param_groups'][0]['params'];assert len(named)==len(ids)==442
optimizer=torch.optim.AdamW((p for n,p in named),lr=1e-4,weight_decay=1e-4)
optimizer.load_state_dict(c['formal_training']['optimizer'])
mapping=[]
for pid,(name,param) in zip(ids,named):
 st=c['formal_training']['optimizer']['state'][pid]
 assert tuple(st['exp_avg'].shape)==tuple(param.shape)
 mapping.append({'id':pid,'name':name,'shape':list(param.shape),'epoch1_step':float(a['formal_training']['optimizer']['state'][pid]['step']),'epoch2_step':float(st['step'])})
short=[x for x in mapping if x['epoch2_step']!=2304]
assert len(short)==48
steps=[json.loads(l) for l in (ckpt.parents[1]/'steps.jsonl').read_text(encoding='utf-8').splitlines()]
empty=[s for s in steps if s['instances']==0]
assert len(empty)==1 and empty[0]['batch']==196
manifest=root/'cache/datasets/local_apple_v1/train_manifest.json'
dataset=PreparedVisibleDataset(manifest,imgsz=640,mask_ratio=2,prepared_cache=root/'cache/datasets/local_apple_prepared_v1')
indices={r['image_id']:i for i,r in enumerate(dataset.rows)}
probes=[]
for scope,ids_image in [('real_empty_batch196',empty[0]['image_ids']),('positive_batch658',steps[657]['image_ids'])]:
 optimizer.zero_grad(set_to_none=True)
 batch=collate_native([dataset[indices[k]] for k in ids_image]);meta=batch.pop('metadata')
 before={name:float(optimizer.state[param]['step']) for name,param in named}
 tick=time.perf_counter();pred=m(batch['img']);loss,logs=compute_e2e_loss(m.criterion,pred,batch);loss.backward()
 none=[name for name,param in named if param.grad is None]
 zero=[name for name,param in named if param.grad is not None and not bool(torch.count_nonzero(param.grad))]
 nonfinite=[name for name,param in named if param.grad is not None and not bool(torch.isfinite(param.grad).all())]
 assert not nonfinite
 torch.nn.utils.clip_grad_norm_(m.parameters(),10.0,error_if_nonfinite=True);optimizer.step()
 increments={name:float(optimizer.state[param]['step'])-before[name] for name,param in named}
 expected_none=set(x['name'] for x in short) if scope.startswith('real_empty') else set()
 record={'scope':scope,'image_ids':ids_image,'instances':sum(len(x['instance_ids']) for x in meta),'loss':float(loss.detach()),'loss_components':logs,'grad_none_names':none,'zero_grad_names':zero,'nonfinite_grad_names':nonfinite,'optimizer_step_increments':increments,'grad_none_exact_expected':set(none)==expected_none,'adamw_increment_matches_grad_presence':all(increments[n]==(0 if n in none else 1) for n,p in named),'seconds':time.perf_counter()-tick}
 probes.append(record);print(json.dumps({'scope':scope,'instances':record['instances'],'none_count':len(none),'zero_count':len(zero),'passed':record['grad_none_exact_expected'] and record['adamw_increment_matches_grad_presence'],'seconds':record['seconds']}),flush=True)
 del pred,loss,batch
loss_source=Path(__import__('ultralytics.utils.loss',fromlist=['__file__']).__file__)
adam_source=Path(__import__('torch.optim.adam',fromlist=['__file__']).__file__)
r={'schema':'local_epoch2_optimizer_participation_cpu_audit_v1','created_utc':datetime.now(timezone.utc).isoformat(),'mode':'CPU FP32 structural gradient participation and in-memory AdamW probe; not training continuation and no checkpoint writes','cuda_initialized':torch.cuda.is_initialized(),'epoch2_checkpoint_sha256':original_sha,'epoch1_checkpoint_sha256':source_sha,'checkpoint_files_unchanged':sha(ckpt)==original_sha and sha(source)==source_sha,'strict_parameter_mapping_count':len(mapping),'parameter_mapping':mapping,'parameters_one_fewer_update':short,'one_fewer_update_count':len(short),'empty_training_batches':empty,'probes':probes,'loss_source':{'path':str(loss_source),'sha256':sha(loss_source),'empty_target_box_branch':'get_assigned_targets_and_loss: regression losses only when fg_mask.sum()>0','empty_target_mask_branch':'v8SegmentationLoss.loss explicitly adds proto*0, mask_coeff*0, pred_semantic*0'},'adam_source':{'path':str(adam_source),'sha256':sha(adam_source),'semantics':'AdamW inherits Adam; params_with_grad populated only when p.grad is not None; per-parameter step advances only for those params'},'epoch2_all_batch_global_updates':len(steps),'epoch2_amp_tracker':c['formal_training']['scaler']['_growth_tracker'],'epoch2_numerical_retries':sum(s['numerical_retries'] for s in steps),'passed':all(x['grad_none_exact_expected'] and x['adamw_increment_matches_grad_presence'] and not x['nonfinite_grad_names'] for x in probes) and not torch.cuda.is_initialized() and sha(ckpt)==original_sha and sha(source)==source_sha,'conclusion':'The unique all-empty batch196 legitimately gives grad=None to exactly 48 box-head tensors, so their AdamW counters advance 1151 times while all other 394 advance1152. Global optimizer invocation count is1152; no whole optimizer skip. Blanket all-parameter-step=global-step audit is invalid for this graph. Exact parameter-specific participation is required; no range tolerance or checkpoint edits.'}
(out/'optimizer_participation_audit.json').write_text(json.dumps(r,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'passed':r['passed'],'cuda_initialized':r['cuda_initialized'],'report':str(out/'optimizer_participation_audit.json')}),flush=True)
