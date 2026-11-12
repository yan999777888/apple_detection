"""CPU real-image integration checks; metrics here are NOT research results."""
from pathlib import Path
import json
import os
import sys
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'experiments/joint_eval_contract_v1'
if OUT.exists():raise FileExistsError('Use a new explicit test output; no overwrite')
os.environ['YOLO_OFFLINE']='1'
os.environ['YOLO_AUTOINSTALL']='0'
os.environ['PYTHONIOENCODING']='utf-8'
from joint_state_v1 import configure
configure(OUT,'cpu')
from joint_eval_v1 import *
from joint_model_v1 import build_model,ActiveEMA
from torch.utils.data import DataLoader,Subset
from ultralytics.data.utils import check_det_dataset
import cv2

torch.set_num_threads(2);torch.set_num_interop_threads(1);cv2.setNumThreads(1)
model=build_model(mode='joint',seed=20261106,epochs=40,device='cpu')
assert model.trunk_for('stage') is model.trunk_for('defect')
assert TaskView(model,'stage').shared_model is TaskView(model,'defect').shared_model
assert {id(p) for p in TaskView(model,'stage').parameters()}=={id(p) for p in model.parameters()}
model.train()
next(model.parameters()).requires_grad_(False)
next(m for m in model.modules() if isinstance(m,nn.BatchNorm2d)).eval()
initial_flags=[m.training for m in model.modules()]
initial_grad=[p.requires_grad for p in model.parameters()]
initial_sha=state_digest(model)
initial_criteria=model.criterion_state()
receipts={};subset_images={};loaders={}
for task,base in [('stage',SegmentationValidator),('defect',DetectionValidator)]:
    spec=source_spec(task)
    args=dict(data=spec['data'],task=TASK_TYPES[task],imgsz=640,batch=1,rect=True,workers=0,quantize=32,overlap_mask=False,mask_ratio=4,plots=False,save_json=False,save_txt=False,device='cpu')
    with offline_network():
        seed_validator=base(save_dir=OUT/('loader_'+task),args=args)
        seed_validator.data=check_det_dataset(spec['data'],autodownload=False,split='val')
        seed_validator.stride=32;seed_validator.device=torch.device('cpu');seed_validator.training=False
        full_loader=seed_validator.get_dataloader(seed_validator.data['val'],1)
    dataset=full_loader.dataset
    loader=DataLoader(Subset(dataset,[0]),batch_size=1,shuffle=False,num_workers=0,collate_fn=dataset.collate_fn)
    loaders[task]=loader;subset_images[task]=dataset.im_files[0]
    receipts[task]=evaluate_task(model,task,output_dir=OUT/('raw_'+task),device='cpu',dataloader=loader,scope='cpu_contract_only',expected_images=1)
    assert [m.training for m in model.modules()]==initial_flags
    assert [p.requires_grad for p in model.parameters()]==initial_grad
    assert state_digest(model)==initial_sha
    assert model.criterion_state()==initial_criteria
    assert 'fitness' not in receipts[task]['metrics']
    assert receipts[task]['noop_fuse_calls']==1 and not receipts[task]['fusion_performed']
    print('official real CPU validator and raw-model restoration passed:',task,flush=True)

ema=ActiveEMA(model).model
ema_sha=state_digest(ema)
receipts['ema_stage']=evaluate_task(ema,'stage',output_dir=OUT/'ema_stage',device='cpu',dataloader=loaders['stage'],scope='cpu_contract_only',expected_images=1)
assert state_digest(ema)==ema_sha and all(not p.requires_grad for p in ema.parameters()) and all(not m.training for m in ema.modules())
assert ema.criteria=={}
image_path=subset_images['stage'];image=cv2.imread(image_path)
assert image is not None
counts={'trunk':0,'stage':0,'defect':0}
def hook(key):
    def called(*_):counts[key]+=1
    return called
handles=[ema.trunk_for('stage')[0].register_forward_hook(hook('trunk'))]
handles += [ema.head_for(task).register_forward_hook(hook(task)) for task in ('stage','defect')]
adapter=JointInferenceAdapter(ema,audit_values=True)
results=adapter.predict(image,paths=[image_path])
assert counts=={'trunk':1,'stage':1,'defect':1}
assert set(results)=={'stage','defect'}
for task in results:
    assert len(results[task])==1 and results[task][0].boxes.data.shape[1]==6
    assert results[task][0].names==ema.names[task]
    assert results[task][0].boxes.data.device.type=='cpu'
counts.update(trunk=0,stage=0,defect=0)
only=adapter.predict(image,paths=[image_path],tasks=('stage',))
assert counts=={'trunk':1,'stage':1,'defect':0} and set(only)=={'stage'}
# Structural probe of the nonempty native-mask path. It does not change the
# deployed default threshold or evaluate model quality: fresh heads are untrained.
probe=JointInferenceAdapter(ema,conf=0.0,max_det=2,audit_values=True)
probed=probe.predict(image,paths=[image_path])
mask_result=probed['stage'][0]
assert len(mask_result.boxes)>0 and mask_result.masks is not None
assert tuple(mask_result.masks.data.shape[1:])==image.shape[:2]
assert len(mask_result.boxes)==len(mask_result.masks)
for h in handles:h.remove()
with preserve_eval_state(ema):
    tensor=adapter.predictors['stage'].preprocess([image])
raw_both=adapter.predict_raw(tensor)
raw_stage=adapter.predict_raw(tensor,tasks=('stage',))
raw_defect=adapter.predict_raw(tensor,tasks=('defect',))
assert torch.equal(raw_both['stage'][0][0],raw_stage['stage'][0][0])
assert torch.equal(raw_both['stage'][0][1],raw_stage['stage'][0][1])
assert torch.equal(raw_both['defect'][0],raw_defect['defect'][0])
assert state_digest(ema)==ema_sha and ema.criteria=={}
# A real inference failure must also restore flags/weights/criteria.
try:
    with preserve_eval_state(model):
        for p in model.parameters():p.requires_grad_(False)
        raise RuntimeError('intentional restoration-path probe')
except RuntimeError as exc:
    assert str(exc)=='intentional restoration-path probe'
assert [m.training for m in model.modules()]==initial_flags
assert [p.requires_grad for p in model.parameters()]==initial_grad
assert state_digest(model)==initial_sha and model.criterion_state()==initial_criteria
assert not torch.cuda.is_initialized()
report={'schema':'joint_eval_cpu_contract_tests_v1','phase':'complete','purpose':'API/identity/state-restoration checks only; untrained fresh-head AP is not a research result','real_images':subset_images,'model_initial_sha256':initial_sha,'ema_sha256':ema_sha,'model_code_sha256':sha_file(Path(__file__).with_name('joint_model_v1.py')),'adapter_sha256':sha_file(Path(__file__).with_name('joint_eval_v1.py')),'test_sha256':sha_file(__file__),'cuda_initialized':torch.cuda.is_initialized(),'torch_threads':torch.get_num_threads(),'interop_threads':torch.get_num_interop_threads(),'tests':['same shared trunk and wrapper parameter references','official segmentation validator full call on one real batch','official detection validator full call on one real batch','mixed training and gradient flags restored','parameter/BN values and existing gradients preserved','source criteria and schedules unchanged','actual official validator preserves EMA values and flags','both-task inference one trunk pass and one pass per head','single-task inference bypasses inactive head','standard Results boxes/classes plus native-resolution masks','raw predict_both matches source-specific decoded outputs','exception path restores model state','no CUDA initialization'],'validator_receipts':{k:{'seen':v['seen'],'state_guard':v['state_guard'],'output':'ema_stage' if k=='ema_stage' else 'raw_'+k} for k,v in receipts.items()},'mask_probe':{'confidence':0.0,'max_det':2,'scope':'structure only, not research metric or deployment threshold','native_hw':list(image.shape[:2]),'output_masks':len(mask_result.masks)}}
save_json(OUT/'contract_test_receipt.json',report)
print(json.dumps({'phase':'complete','receipt':str(OUT/'contract_test_receipt.json'),'tests':len(report['tests']),'cuda_initialized':False}),flush=True)
