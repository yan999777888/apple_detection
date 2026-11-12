"""Actual CPU batch and deterministic augmentation contracts; never fit/run GPU."""
import os
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='2'
os.environ.update(CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',YOLO_AUTOINSTALL='False',YOLO_OFFLINE='true',NO_ALBUMENTATIONS_UPDATE='1')
for key,name in {'YOLO_CONFIG_DIR':'yolo','MPLCONFIGDIR':'matplotlib','TEMP':'tmp','TMP':'tmp','TORCH_HOME':'torch','XDG_CACHE_HOME':'cache'}.items():
    p=ROOT/'runtime/joint_data_cpu_contract'/name;p.mkdir(parents=True,exist_ok=True);os.environ[key]=str(p)
sys.dont_write_bytecode=True
import hashlib,json,random,pickle,traceback
from datetime import datetime,timezone
from collections import Counter
import numpy as np
import torch
import cv2
import ultralytics
from joint_data import JointDataModule,TASKS,validate_batch,sha,save,DEFAULT_DIR,walk_transforms
torch.set_num_threads(2);cv2.setNumThreads(1)
OUT=DEFAULT_DIR/'data_cpu_contract.json'
assert not OUT.exists(),'Refuse overwrite'
report={'schema':'joint_data_cpu_contract_v1','phase':'running','GPU_used':False,'model_fitted':False,'torch_threads':torch.get_num_threads(),'checks':[]}
def check(ok,name):
    if not ok:raise AssertionError(name)
    report['checks'].append(name)
def rng():return (pickle.dumps(random.getstate()),pickle.dumps(np.random.get_state()),torch.get_rng_state().numpy().tobytes())
def tensor_hash(batch):
    h=hashlib.sha256()
    for k in sorted(batch):
        v=batch[k]
        if isinstance(v,torch.Tensor):h.update(k.encode());h.update(str(v.shape).encode());h.update(v.numpy().tobytes())
    return h.hexdigest()
def label_hash(dataset):
    h=hashlib.sha256()
    for row in dataset.labels:
        for k in ('cls','bboxes'):h.update(row[k].tobytes())
        for x in row['segments']:h.update(x.tobytes())
    return h.hexdigest()
try:
    dm=JointDataModule();report.update(manifest_sha256=dm.manifest_sha256,protocol_sha256=dm.protocol_sha256,runner_sha256=sha(Path(__file__).with_name('joint_data.py')),source_bound_files=len(dm.source_hashes))
    fingerprints=[]
    for epoch in range(40):
        plan=dm.epoch_schedule(epoch);fingerprints.append(dm.schedule_fingerprint(epoch))
        assert len(plan)==87
        for task,expected,blocks in (('stage',772,49),('defect',596,38)):
            chosen=[w for w in plan if w['task']==task];anchors=[x for w in chosen for x in w['anchor_ids']]
            assert len(chosen)==blocks and len(anchors)==len(set(anchors))==expected
            assert set(anchors)=={r['sample_id'] for r in dm.records[task]}
            assert chosen[-1]['anchor_count']==4 and len(chosen[-1]['microbatch_indices'])==1
            assert all(w['anchor_count']==sum(map(len,w['microbatch_indices'])) for w in chosen)
        assert sum(w['anchor_count'] for w in plan)==1368
    check(len(set(fingerprints))==40,'All40epochs: once-per-source anchor coverage,49+38windows,4anchor tails,distinct deterministic orders')
    report['epoch_schedule_sha256']=fingerprints
    for task in TASKS:dm.dataset(task)
    original_labels={task:label_hash(dm.dataset(task)) for task in TASKS}
    plans={task:[w for w in dm.epoch_schedule(0) if w['task']==task] for task in TASKS}
    hashes={};donors={task:0 for task in TASKS};tested_anchors={task:[] for task in TASKS}
    # Four actual batches/source, alternated; then reverse source/order and revisit.
    requests=[(t,indices) for j in range(4) for t in TASKS for indices in [plans[t][0]['microbatch_indices'][j]]]
    for task,indices in requests:
        state=rng();batch=dm.batch(task,indices,0);check(rng()==state,'Caller Python/NumPy/TorchCPU RNG restored '+task+str(indices[0]))
        hashes[(task,tuple(indices))]=tensor_hash(batch);donors[task]+=sum(len(x)>1 for x in batch['augmentation_source_ids']);tested_anchors[task]+=batch['anchor_ids']
        check(all(all(x.startswith(task+'/train/') for x in draw) for draw in batch['augmentation_source_ids']),'All donors belong to same source training split '+task+str(indices[0]))
    for task,indices in reversed(requests):
        state=rng();batch=dm.batch(task,indices,0)
        check(tensor_hash(batch)==hashes[(task,tuple(indices))] and rng()==state,'Exact tensor equality after different global retrieval order '+task+str(indices[0]))
    check(all(donors[t]>0 for t in TASKS),'Actual native mosaic augmentation exercised for both sources')
    for task in TASKS:
        idx=plans[task][0]['microbatch_indices'][0]
        dm.batch(task,idx,1);again=dm.batch(task,idx,0)
        check(tensor_hash(again)==hashes[(task,tuple(idx))],'Epoch recreation reproduces image/box/class/native-mask tensors '+task)
        closed=dm.batch(task,idx,35)
        check(all(len(draw)==1 for draw in closed['augmentation_source_ids']),'Mosaic disabled last5epochs '+task)
        check(label_hash(dm.dataset(task))==original_labels[task],'Persistent raw labels/polygons unchanged '+task)
        tail=plans[task][-1];batch=dm.batch(task,tail['sample_indices'],0)
        check(len(batch['anchor_ids'])==4,'Real source tail batch validates '+task)
    # Actual grouped-window API used by core, including source-only control filter.
    joint=next(dm.iter_epoch(0));solo=next(dm.iter_epoch(0,task='stage'))
    check(joint['anchor_count']==16 and len(joint['batches'])==4 and [tensor_hash(b) for b in joint['batches']]==[tensor_hash(b) for b in solo['batches']],'Joint/separate grouped-window tensor parity')
    stage=dm.batch('stage',plans['stage'][0]['microbatch_indices'][0],0);defect=dm.batch('defect',plans['defect'][0]['microbatch_indices'][0],0)
    negatives=[]
    bad=dict(stage);bad['cls']=None;negatives.append((bad,'stage'))
    bad=dict(stage);bad['annotation_available']={'stage':True,'defect':None};negatives.append((bad,'stage'))
    bad=dict(stage);bad['masks']=None;negatives.append((bad,'stage'))
    bad=dict(stage);bad['mask_annotation_available']=None;negatives.append((bad,'stage'))
    bad=dict(defect);bad['masks']=torch.zeros(0,160,160);negatives.append((bad,'defect'))
    bad=dict(defect);bad['annotation_available']={'stage':True,'defect':True};negatives.append((bad,'defect'))
    for i,(batch,task) in enumerate(negatives):
        try:validate_batch(batch,task)
        except ValueError:pass
        else:raise AssertionError('Null/missing task annotation accepted '+str(i))
    check(True,'6 null/missing/inactive-task fabrication negative contracts rejected')
    dm.verify_sources()
    report.update(phase='passed',completed_utc=datetime.now(timezone.utc).isoformat(),native_version=ultralytics.__version__,
        tested_anchor_ids=tested_anchors,mosaic_anchor_count_among_first16=donors,source_files_unchanged=True,
        stage_tensor_shapes={k:list(stage[k].shape) for k in ('img','cls','bboxes','batch_idx','masks','sem_masks')},
        defect_tensor_shapes={k:list(defect[k].shape) for k in ('img','cls','bboxes','batch_idx')},
        class_support={t:{s:[sum(r['class_support'][c] for r in dm.manifest['records'] if r['source']==t and r['split']==s) for c in range(len(dm.source_specs[t]['names']))] for s in ('train','val')} for t in TASKS},
        limitations=['CPU tensor contract tests and schedule coverage only; no model training or accuracy results.','Each anchor once does not mean each underlying image appears only once: mosaic donors are recorded augmentation appearances.','No physical-fruit/orchard independence claim.'])
except BaseException as error:
    report.update(phase='failed',error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc());raise
finally:
    save(OUT,report);print(json.dumps({'phase':report['phase'],'report':str(OUT),'checks':len(report['checks'])},ensure_ascii=False))