"""Source-aware deterministic YOLO data for matched joint/separate controls.

No GPU calls. workers=0 only. Each source anchor occurs once per source epoch;
mosaic donors are augmentation appearances, not additional anchors. The two
heads have disjoint annotation availability and class vocabularies.
"""
from __future__ import annotations
import argparse
from collections import Counter
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys

sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[2]
DEFAULT_DIR=ROOT/'experiments/joint_training_v1'
DEFAULT_MANIFEST=DEFAULT_DIR/'data_manifest.json'
DEFAULT_PROTOCOL=DEFAULT_DIR/'data_protocol.json'
TASKS=('stage','defect')
EXPECTED={'stage':{'train':772,'val':72},'defect':{'train':596,'val':113}}
NAMES={'stage':['immature apple','semi-mature apple','mature apple'],'defect':['class_0','class_1']}
AUGMENT={'mosaic':.25,'mixup':0.,'copy_paste':0.,'cutmix':0.,'degrees':0.,'translate':.1,'scale':.25,'shear':0.,'perspective':0.,'flipud':0.,'fliplr':.5,'hsv_h':0.,'hsv_s':0.,'hsv_v':.1,'bgr':0.,'overlap_mask':False,'mask_ratio':4,'deterministic':True}


def require(ok,message):
    if not ok:raise ValueError(message)


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def canonical(value):return json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False)
def digest(value):return hashlib.sha256(canonical(value).encode('utf-8')).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def save(path,value):
    with Path(path).open('x',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False)


def fpath(value):
    import re
    path=Path(value).resolve()
    require(path.drive.upper()=='F:','All inputs/output must stay on F')
    require(not any(set(re.split('[^a-z0-9]+',p.lower()))&{'test','tests','testing'} for p in path.parts),'No test paths')
    return path


def parse_label(row,names):
    """Read existing labels; missing/null active labels are errors, not negatives."""
    path=fpath(row['label_path']);require(path.is_file(),'Missing annotation is not an empty background')
    result=[]
    for number,line in enumerate(path.read_text(encoding='utf-8-sig').splitlines(),1):
        if not line.strip():continue
        vals=[float(x) for x in line.split()]
        require(all(math.isfinite(x) for x in vals),'Nonfinite label')
        require(vals[0].is_integer() and int(vals[0]) in range(len(names)),'Source class outside declared head')
        if row['source']=='stage':require(len(vals)>=7 and len(vals)%2==1,'Stage annotation must be a polygon')
        else:require(len(vals)==5 and vals[3]>0 and vals[4]>0,'Defect annotation must be normalized xywh')
        require(all(0<=x<=1 for x in vals[1:]),'Annotation outside normalized range')
        result.append(vals)
    return result


def prepare(manifest_path=DEFAULT_MANIFEST,protocol_path=DEFAULT_PROTOCOL):
    """Freeze only actual converted train/dev files and their source provenance."""
    require(not Path(manifest_path).exists() and not Path(protocol_path).exists(),'Refuse existing data freeze')
    import yaml
    files={};rows=[];specs={}
    def bind(path,expected=None):
        path=fpath(path);actual=sha(path)
        require(expected is None or actual==expected,'Source SHA mismatch '+str(path))
        require(str(path) not in files or files[str(path)]==actual,'Conflicting source hash')
        files[str(path)]=actual;return str(path)
    for task,folder,meta in (('stage','mendeley_stage_seg_v1','conversion_audit.json'),('defect','ssda_date_v2','manifest.json')):
        base=ROOT/'apple_research_v2/datasets'/folder
        data=base/'data.yaml';doc=yaml.safe_load(data.read_text(encoding='utf-8-sig'))
        require([doc['names'][i] for i in range(len(doc['names']))]==NAMES[task],'Unexpected source classes')
        bind(data);audit_path=base/meta;bind(audit_path);audit=read(audit_path)
        candidates=audit['images'] if task=='stage' else audit['samples']
        for item in candidates:
            split=item.get('split') if task=='stage' else {'train':'train','dev':'val'}.get(item['partition'])
            if split not in ('train','val'):continue
            if task=='stage':
                image=item['image_path'];ih=item['image_sha256'];group=item['group_id'];native=item['native_hw']
                origin=bind(item['source_image_path'],ih);annotation=bind(item['annotation_path'],item['annotation_sha256'])
            else:
                require(item['training_eligible'] is True,'Included SSDA sample is source-quarantined')
                image=item['rgb_path'];ih=item['rgb_sha256'];group=item['split_group'];native=item['native_hw'];origin=str(fpath(image));annotation=str(fpath(item['label_path']))
            image=bind(image,ih);label=bind(item['label_path'],item['label_sha256'])
            row={'source':task,'split':split,'sample_id':task+'/'+split+'/'+Path(image).name,
                'image_path':image,'label_path':label,'image_sha256':ih,'label_sha256':item['label_sha256'],
                'native_hw':native,'source_image_path':origin,'source_annotation_path':annotation,'known_group_id':group,
                'fruit_id':None,'physical_fruit_identity_verified':False,
                'annotation_available':{t:t==task for t in TASKS},'mask_annotation_available':task=='stage'}
            labels=parse_label(row,NAMES[task]);row['raw_instances']=len(labels);row['class_support']=[sum(int(v[0])==c for v in labels) for c in range(len(NAMES[task]))]
            require(len(labels)==len(item['instances']),'Converted annotation instance count differs')
            rows.append(row)
        # Resolve only train/val YAML paths and prove membership equals frozen rows.
        for split in ('train','val'):
            spec=fpath(Path(doc['path'])/doc[split])
            if spec.is_dir():paths={str(p.resolve()) for p in spec.iterdir() if p.suffix.lower() in ('.jpg','.jpeg','.png')}
            else:
                bind(spec);paths={str(fpath(line.strip())) for line in spec.read_text(encoding='utf-8-sig').splitlines() if line.strip()}
            selected=[r for r in rows if r['source']==task and r['split']==split]
            require(len(selected)==EXPECTED[task][split] and {r['image_path'] for r in selected}==paths,'YAML/source split mismatch')
        specs[task]={'data':str(data),'task':'segment' if task=='stage' else 'detect','names':NAMES[task],'counts':EXPECTED[task],
            'semantics':'Author color stages and polygon outlines; not physiological maturity/verified visible-only masks' if task=='stage' else 'Numeric defect-region types; no normal-vs-abnormal or whole-fruit grade labels'}
    rows.sort(key=lambda r:r['sample_id']);require(len({r['sample_id'] for r in rows})==len(rows),'Duplicate sample IDs')
    overlap={}
    for task in TASKS:
        a=[r for r in rows if r['source']==task and r['split']=='train'];b=[r for r in rows if r['source']==task and r['split']=='val']
        overlap[task]={'image_sha256':len({r['image_sha256'] for r in a}&{r['image_sha256'] for r in b}),'known_group_ids':len({r['known_group_id'] for r in a}&{r['known_group_id'] for r in b})}
        require(overlap[task]['image_sha256']==0,'Exact train/dev image leakage')
    prior=ROOT/'experiments/initial_full_training_protocol.json';bind(prior)
    protocol={'schema':'joint_data_protocol_v1','epochs':40,'seed':20261106,'imgsz':640,'microbatch_size':4,'effective_anchor_count':16,'workers':0,
        'source_order':list(TASKS),'augmentation':AUGMENT,'close_mosaic_last_epochs':5,
        'native_version':'8.4.115','mosaic_donor_selection':'Set native Mosaic.buffer_enabled=False; uniform draws within the current training source, independent of historical loader buffer',
        'augmentation_seed':'SHA256(seed,source,zero-based epoch,stable sample_id); isolated Python/NumPy/Torch CPU RNG; independently seeded Albumentations when active',
        'sample_order':'SHA256(seed,source,epoch,sample_id,order), stable sample_id ties; identical for joint and separate controls',
        'blocks':'Within each source consecutive <=16 anchors, microbatches<=4; interleave complete stage/defect blocks by source block index, never resample',
        'tail_loss_normalization':'Sum native criterion batch-summed loss vectors over a source block divided by its actual anchor count; flush every block, including final 4-anchor tail',
        'exposure':{'anchors_per_cycle':1368,'stage_blocks_per_cycle':49,'defect_blocks_per_cycle':38,'joint_shared_steps_40cycles':3480,'stage_head_steps_40cycles':1960,'defect_head_steps_40cycles':1520,
            'separate_control_total_steps_40cycles':3480,'mosaic_donors_are_extra_augmentation_appearances':True,'silent_source_resampling':False},
        'control_comparability':'New joint and separate controls must share this data module/order/augment seed/accumulation. Earlier independent Ultralytics runs used dynamic warmup accumulation and are not exact optimizer-exposure controls.',
        'evaluation':'Development only; no physical-fruit/orchard independence claim; stage only on stage source and defect only on defect source',
        'missing_annotations':'Inactive head availability is False, never fabricated negative labels; active label path/cls must exist; known empty annotation files remain annotated backgrounds',
        'prior_recipe_path':str(prior),'prior_recipe_sha256':files[str(prior)]}
    for p in (Path(manifest_path),Path(protocol_path)):
        require(fpath(p).is_relative_to(DEFAULT_DIR),'Freeze path outside owned joint data directory');p.parent.mkdir(parents=True,exist_ok=True)
    save(protocol_path,protocol)
    manifest={'schema':'joint_data_manifest_v1','source_specs':specs,'records':rows,'source_hashes':files,'counts':EXPECTED,'overlap_audit':overlap,
        'protocol_sha256':sha(protocol_path),'physical_fruit_independent':False,'PApple_included':False,'test_read':False}
    save(manifest_path,manifest)
    return manifest


def sample_seed(seed,task,epoch,sample_id,purpose='augmentation'):
    return int.from_bytes(hashlib.sha256(canonical([seed,task,epoch,sample_id,purpose]).encode()).digest()[:8],'big')%(2**63-1)


@contextmanager
def isolated_rng(seed):
    import numpy as np
    import torch
    py,npstate,ts=random.getstate(),np.random.get_state(),torch.get_rng_state()
    try:
        random.seed(seed);np.random.seed(seed%(2**32));torch.random.default_generator.manual_seed(seed)
        yield
    finally:
        random.setstate(py);np.random.set_state(npstate);torch.set_rng_state(ts)


def walk_transforms(node,seen=None):
    seen=set() if seen is None else seen
    if id(node) in seen:return
    seen.add(id(node));yield node
    for child in getattr(node,'transforms',[]):yield from walk_transforms(child,seen)
    pre=getattr(node,'pre_transform',None)
    if pre is not None:yield from walk_transforms(pre,seen)


def dataset_for(rows,task,protocol):
    import numpy as np
    from ultralytics.cfg import get_cfg
    from ultralytics.data.dataset import YOLODataset
    from ultralytics.utils.ops import segments2boxes
    class SourceDataset(YOLODataset):
        def __init__(self):
            self.rows=rows;self.epoch=0;self.draws=None
            hyp=get_cfg(overrides=dict(protocol['augmentation']))
            super().__init__(img_path='frozen_joint_manifest',imgsz=protocol['imgsz'],cache=False,augment=True,hyp=hyp,
                prefix=task+': ',rect=False,batch_size=4,stride=32,pad=0.,data={'names':dict(enumerate(NAMES[task])),'channels':3},task='segment' if task=='stage' else 'detect')
            self.base_hyp=hyp
            # Never consume or repair hidden source-adjacent .npy image caches.
            self.npy_files=[ROOT/'runtime/joint_data/no_numpy_cache'/f'{digest(r["sample_id"])}.npy' for r in rows]
            require(not any(p.exists() for p in self.npy_files),'Unexpected image cache')
            self.set_epoch(0)
        def get_img_files(self,img_path):return [r['image_path'] for r in self.rows]
        def get_labels(self):
            labels=[]
            for r in self.rows:
                raw=parse_label(r,NAMES[task]);cls=np.array([v[0] for v in raw],dtype=np.float32).reshape(-1,1)
                segments=[np.array(v[1:],dtype=np.float32).reshape(-1,2) for v in raw] if task=='stage' else []
                boxes=segments2boxes(segments) if segments else np.array([v[1:] for v in raw],dtype=np.float32).reshape(-1,4)
                labels.append({'im_file':r['image_path'],'shape':tuple(r['native_hw']),'cls':cls,'bboxes':boxes,'segments':segments,'keypoints':None,'normalized':True,'bbox_format':'xywh'})
            self.label_files=[r['label_path'] for r in self.rows]
            return labels
        def load_image(self,*args,**kwargs):
            image,hw0,hw=super().load_image(*args,**kwargs)
            return image.copy(),hw0,hw
        def get_image_and_label(self,index):
            if self.draws is not None:self.draws.append(self.rows[index]['sample_id'])
            return super().get_image_and_label(index)
        def set_epoch(self,epoch):
            require(0<=epoch<protocol['epochs'],'Epoch outside frozen schedule')
            self.epoch=epoch;hyp=deepcopy(self.base_hyp)
            if epoch>=protocol['epochs']-protocol['close_mosaic_last_epochs']:hyp.mosaic=0.
            with isolated_rng(sample_seed(protocol['seed'],task,epoch,'transform_build')):
                self.transforms=self.build_transforms(hyp)
            for t in walk_transforms(self.transforms):
                if t.__class__.__name__=='Mosaic':t.buffer_enabled=False
        def __getitem__(self,index):
            row=self.rows[index];seed=sample_seed(protocol['seed'],task,self.epoch,row['sample_id'])
            with isolated_rng(seed):
                for t in walk_transforms(self.transforms):
                    if t.__class__.__name__=='Albumentations' and getattr(t,'transform',None) is not None:
                        require(hasattr(t.transform,'set_random_seed'),'Installed Albumentations lacks isolated seed interface')
                        t.transform.set_random_seed(seed%(2**32))
                self.draws=[]
                try:
                    item=super().__getitem__(index);draws=list(self.draws)
                finally:self.draws=None
            item['_joint_anchor']=row['sample_id'];item['_joint_draws']=draws;item['_joint_seed']=seed
            return item
    with isolated_rng(sample_seed(protocol['seed'],task,0,'dataset_constructor')):return SourceDataset()


def validate_batch(batch,task,imgsz=640):
    import torch
    require(task in TASKS and batch.get('source')==task,'Unknown/mixed batch source')
    flags=batch.get('annotation_available')
    require(isinstance(flags,dict) and set(flags)==set(TASKS) and all(type(v) is bool for v in flags.values()),'Missing/null task availability')
    require(flags=={t:t==task for t in TASKS},'Unavailable task labels must not be fabricated')
    require(type(batch.get('mask_annotation_available')) is bool and batch['mask_annotation_available']==(task=='stage'),'Missing/null/incorrect mask annotation availability')
    for key in ('img','batch_idx','cls','bboxes'):require(isinstance(batch.get(key),torch.Tensor),'Missing/null active '+key)
    im=batch['img'];b=im.shape[0];cls=batch['cls'];boxes=batch['bboxes'];idx=batch['batch_idx'];n=len(cls)
    require(im.shape==(b,3,imgsz,imgsz) and im.dtype==torch.float32 and im.device.type=='cpu','Image tensor contract')
    require(torch.isfinite(im).all().item() and im.min().item()>=0 and im.max().item()<=1,'Image normalization/finite')
    require(cls.shape==(n,1) and boxes.shape==(n,4) and idx.shape==(n,) and idx.dtype==torch.int64,'Target tensor shapes')
    require(cls.dtype==boxes.dtype==torch.float32 and all(t.device.type=='cpu' for t in (cls,boxes,idx)),'CPU float32 target contract')
    require(all(torch.isfinite(t).all().item() for t in (cls,boxes)),'Nonfinite targets')
    if n:
        require(torch.equal(cls,cls.round()) and cls.min().item()>=0 and cls.max().item()<len(NAMES[task]),'Invalid source class IDs')
        require(boxes.min().item()>=0 and boxes.max().item()<=1 and (boxes[:,2:]>0).all().item(),'Invalid normalized boxes')
        require(idx.min().item()>=0 and idx.max().item()<b and torch.equal(idx,idx.sort().values),'Unsorted/out-of-range batch indices')
    require(len(batch['anchor_ids'])==b and len(set(batch['anchor_ids']))==b,'Anchor count/uniqueness')
    if task=='stage':
        masks,sem=batch.get('masks'),batch.get('sem_masks')
        require(isinstance(masks,torch.Tensor) and isinstance(sem,torch.Tensor),'Missing/null active instance/semantic masks')
        require(masks.shape==(n,imgsz//4,imgsz//4) and sem.shape==(b,imgsz//4,imgsz//4),'Native mask dimensions')
        require(torch.isfinite(masks).all().item() and ((masks==0)|(masks==1)).all().item(),'Invalid instance masks')
        require(torch.isfinite(sem).all().item() and torch.equal(sem,sem.round()) and sem.min().item()>=0 and sem.max().item()<3,'Invalid native semantic mask')
    else:require('masks' not in batch and 'sem_masks' not in batch,'Defect masks are unannotated, not synthetic negatives')
    return True


class JointDataModule:
    def __init__(self,manifest_path=DEFAULT_MANIFEST,protocol_path=DEFAULT_PROTOCOL,workers=0):
        require(workers==0,'Only workers=0 is frozen and deterministic-tested')
        self.manifest_path=fpath(manifest_path);self.protocol_path=fpath(protocol_path)
        self.manifest=read(self.manifest_path);self.protocol=read(self.protocol_path)
        self.manifest_sha256=sha(self.manifest_path);self.protocol_sha256=sha(self.protocol_path)
        require(self.manifest['protocol_sha256']==self.protocol_sha256,'Data protocol hash mismatch')
        require(self.manifest['schema']=='joint_data_manifest_v1' and self.protocol['schema']=='joint_data_protocol_v1','Unexpected data freeze')
        require(self.protocol['augmentation']==AUGMENT and self.protocol['microbatch_size']==4 and self.protocol['effective_anchor_count']==16 and self.protocol['workers']==0,'Unfrozen data recipe')
        self.source_hashes=dict(self.manifest['source_hashes']);self.source_hashes[str(self.manifest_path)]=self.manifest_sha256;self.source_hashes[str(self.protocol_path)]=self.protocol_sha256
        self.source_hashes[str(Path(__file__).resolve())]=sha(__file__)
        self.source_specs=self.manifest['source_specs'];self.counts=self.manifest['counts'];require(self.counts==EXPECTED,'Unexpected source coverage');self.verify_sources()
        self.records={task:[r for r in self.manifest['records'] if r['source']==task and r['split']=='train'] for task in TASKS}
        self._datasets={}
    def verify_sources(self):
        for path,expected in self.source_hashes.items():require(sha(fpath(path))==expected,'Frozen source changed '+path)
    def dataset(self,task):
        require(task in TASKS,'Unknown task')
        if task not in self._datasets:
            import ultralytics
            require(ultralytics.__version__==self.protocol['native_version'],'Unexpected native loader version')
            self._datasets[task]=dataset_for(self.records[task],task,self.protocol)
        return self._datasets[task]
    def epoch_schedule(self,epoch):
        require(0<=epoch<self.protocol['epochs'],'Invalid source epoch')
        blocks={}
        for task in TASKS:
            ordered=sorted(range(len(self.records[task])),key=lambda i:(sample_seed(self.protocol['seed'],task,epoch,self.records[task][i]['sample_id'],'order'),self.records[task][i]['sample_id']))
            blocks[task]=[ordered[i:i+16] for i in range(0,len(ordered),16)]
        result=[]
        for source_index in range(max(map(len,blocks.values()))):
            for task in TASKS:
                if source_index<len(blocks[task]):
                    ids=blocks[task][source_index]
                    result.append({'task':task,'window_index':len(result),'source_window_index':source_index,'source_window_count':len(blocks[task]),
                        'anchor_count':len(ids),'sample_indices':ids,'anchor_ids':[self.records[task][i]['sample_id'] for i in ids],
                        'microbatch_indices':[ids[j:j+4] for j in range(0,len(ids),4)]})
        return result
    def schedule_fingerprint(self,epoch):return digest(self.epoch_schedule(epoch))
    def batch(self,task,indices,epoch):
        from ultralytics.data.dataset import YOLODataset
        data=self.dataset(task)
        if data.epoch!=epoch:data.set_epoch(epoch)
        items=[data[i] for i in indices]
        anchors=[r.pop('_joint_anchor') for r in items];draws=[r.pop('_joint_draws') for r in items];seeds=[r.pop('_joint_seed') for r in items]
        batch=YOLODataset.collate_fn(items)
        batch['img']=batch['img'].float().div_(255);batch['batch_idx']=batch['batch_idx'].long()
        batch.update(source=task,task=task,anchor_ids=anchors,augmentation_source_ids=draws,augmentation_seeds=seeds,
            annotation_available={t:t==task for t in TASKS},mask_annotation_available=task=='stage')
        validate_batch(batch,task,self.protocol['imgsz'])
        return batch
    def iter_epoch(self,epoch,start_window=0,task=None):
        schedule=self.epoch_schedule(epoch)
        require(0<=start_window<=len(schedule),'Invalid completed-window cursor')
        require(task is None or task in TASKS,'Unknown control task')
        for window in schedule[start_window:]:
            if task is not None and window['task']!=task:continue
            batches=[self.batch(window['task'],indices,epoch) for indices in window['microbatch_indices']]
            yield {**window,'batches':batches}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--prepare',action='store_true');args=p.parse_args()
    require(args.prepare,'Use --prepare to freeze data; training imports JointDataModule')
    result=prepare();print(json.dumps({'phase':'data_frozen','manifest':str(DEFAULT_MANIFEST),'manifest_sha256':sha(DEFAULT_MANIFEST),'protocol':str(DEFAULT_PROTOCOL),'protocol_sha256':sha(DEFAULT_PROTOCOL),'counts':result['counts']},ensure_ascii=False))