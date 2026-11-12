"""Shared-trunk/source-aware YOLO model; local official initialization only."""
from __future__ import annotations
import copy, hashlib, math
from pathlib import Path
from types import SimpleNamespace
import torch
from torch import nn
from ultralytics.nn.modules.head import Detect, Segment26
from ultralytics.utils.loss import E2ELoss, v8SegmentationLoss
from ultralytics.utils.torch_utils import initialize_weights

ROOT=Path(__file__).resolve().parents[2]
OFFICIAL=ROOT/'assets/ultralytics_yolo26_coco_20261106/yolo26n-seg.pt'
OFFICIAL_SHA='361fbfabab285c3237700b6bb91d7ecfa602cd945fffda8dbe1242829b71e73f'
TASKS=('stage','defect')
NAMES={'stage':{0:'immature apple',1:'semi-mature apple',2:'mature apple'},'defect':{0:'class_0',1:'class_1'}}
SOURCES={'mendeley_stage3':'stage','ssda_defect2':'defect','stage':'stage','defect':'defect'}

def require(ok,message):
    if not ok: raise ValueError(message)

def file_sha(path):
    with Path(path).open('rb') as stream: return hashlib.file_digest(stream,'sha256').hexdigest()

def tensor_sha(state):
    digest=hashlib.sha256()
    for name,value in sorted(state.items()):
        a=value.detach().cpu().contiguous().numpy()
        digest.update(name.encode());digest.update(str(a.dtype).encode());digest.update(str(a.shape).encode());digest.update(a.tobytes())
    return digest.hexdigest()

def finite_state(state,description,max_chunk_elements=16000000):
    """Bounded concatenation reduces device synchronization; never repairs values."""
    groups={}
    for name,value in state.items():
        if torch.is_tensor(value) and (value.is_floating_point() or value.is_complex()):
            groups.setdefault((value.device,value.dtype),[]).append((name,value.detach()))
    for items in groups.values():
        chunk=[];names=[];size=0
        def check(parts,labels):
            if not parts:return
            if not torch.isfinite(torch.cat(parts)).all().item():
                bad=[name for name,value in items if not torch.isfinite(value).all().item()]
                raise ValueError(description+' has nonfinite tensors: '+str(bad[:8]))
        for name,value in items:
            flat=value.reshape(-1)
            for part in flat.split(max_chunk_elements):
                if size+part.numel()>max_chunk_elements:check(chunk,names);chunk=[];names=[];size=0
                chunk.append(part);names.append(name);size+=part.numel()
        check(chunk,names)
class LossView(nn.Module):
    def __init__(self,head,epochs):
        super().__init__();self.model=nn.ModuleList([head])
        self.args=SimpleNamespace(box=7.5,cls=.5,dfl=1.5,overlap_mask=False,epochs=epochs)

class SourceAwareYOLO(nn.Module):
    def __init__(self,reference,mode='joint',seed=20261106,epochs=40):
        super().__init__();require(mode in ('joint','separate'),'Unknown mode')
        self.mode=mode;self.epochs=epochs;self.names=copy.deepcopy(NAMES)
        self.args={'task':'segment','imgsz':640};self.save_indices=set(reference.save)
        self.feature_indices=tuple(reference.model[-1].f);require(self.feature_indices==(16,19,22),'Unsupported trunk graph')
        chosen=nn.ModuleList(copy.deepcopy(list(reference.model[:-1]))).float().requires_grad_(True)
        self.trunks=nn.ModuleDict({'shared':chosen} if mode=='joint' else {task:copy.deepcopy(chosen) for task in TASKS})
        channels=tuple(branch[0].conv.in_channels for branch in reference.model[-1].cv2)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.heads=nn.ModuleDict({'stage':Segment26(nc=3,nm=32,npr=64,reg_max=1,end2end=True,ch=channels),'defect':Detect(nc=2,reg_max=1,end2end=True,ch=channels)})
            for head in self.heads.values():
                head.stride=reference.model[-1].stride.detach().clone().float();initialize_weights(head);head.bias_init()
        self.criteria={};self.criterion_epochs={task:0 for task in TASKS}
        self.initialization={'official_path':str(OFFICIAL),'official_sha256':OFFICIAL_SHA,'trunk_tensor_sha256':tensor_sha(chosen.state_dict()),'heads':{task:tensor_sha(head.state_dict()) for task,head in self.heads.items()},'seed':seed,'mode':mode,'trained_head_transfer':False}
    def _apply(self,fn):
        super()._apply(fn)
        for head in self.heads.values():
            for name in ('stride','anchors','strides'): setattr(head,name,fn(getattr(head,name)))
        self.criteria={}  # rebuild device-bound loss tensors after movement
        return self
    def head_for(self,task): return self.heads[task]
    def trunk_for(self,task): return self.trunks['shared' if self.mode=='joint' else task]
    def parameter_groups(self,weight_decay=.0005):
        components=[('trunks.'+name,module) for name,module in self.trunks.items()]+[('heads.'+task,self.heads[task]) for task in TASKS]
        groups=[]
        for component,module in components:
            for decay_kind,condition in [('decay',True),('no_decay',False)]:
                params=[parameter for parameter in module.parameters() if (parameter.ndim>1)==condition]
                if params:groups.append({'name':component+'.'+decay_kind,'component':component,'params':params,'weight_decay':weight_decay if condition else 0.})
        return groups
    def active_groups(self,task): return ['trunks.'+('shared' if self.mode=='joint' else task),'heads.'+task]
    def features(self,images,task):
        require(images.ndim==4 and images.shape[1]==3 and images.shape[-1]%32==0 and images.shape[-2]%32==0,'Invalid RGB input dimensions')
        x=images;outputs=[]
        for layer in self.trunk_for(task):
            if layer.f!=-1: x=outputs[layer.f] if isinstance(layer.f,int) else [x if i==-1 else outputs[i] for i in layer.f]
            x=layer(x);outputs.append(x if layer.i in self.save_indices else None)
        return [outputs[i] for i in self.feature_indices]
    def predict_source(self,images,task): return self.heads[task](self.features(images,task))
    def predict_both(self,images):
        if self.mode=='separate': return {task:self.predict_source(images,task) for task in TASKS}
        features=self.features(images,'stage');return {task:self.heads[task](list(features)) for task in TASKS}
    def prepare_criteria(self):
        for task in TASKS:
            criterion=E2ELoss(LossView(self.heads[task],self.epochs),v8SegmentationLoss) if task=='stage' else E2ELoss(LossView(self.heads[task],self.epochs))
            for _ in range(self.criterion_epochs[task]): criterion.update()
            self.criteria[task]=criterion
    def criterion_state(self):
        return {task:{'updates':self.criterion_epochs[task],'hyp_epochs':self.epochs,**{name:getattr(self.criteria[task],name) for name in ('o2m','o2o','total','o2m_copy','final_o2m')}} for task in TASKS}
    def set_criterion_epoch(self,epoch):
        require(0<=epoch<=self.epochs,'Invalid loss schedule epoch')
        self.criterion_epochs={task:epoch for task in TASKS};self.prepare_criteria()
    def validate_batch(self,task,batch):
        require(task in TASKS,'Unknown task');require(SOURCES.get(batch.get('source'))==task,'Source/task mismatch')
        require(batch.get('annotation_available')=={name:name==task for name in TASKS},'Missing task annotation declaration')
        images=batch['img'];require(images.is_floating_point() and torch.isfinite(images).all().item() and images.min().item()>=0 and images.max().item()<=1,'Images must be finite RGB [0,1]')
        require(images.device==next(self.parameters()).device,'Model/input device mismatch')
        labels,boxes,indices=batch['cls'],batch['bboxes'],batch['batch_idx'];n=len(labels)
        require(labels.shape==(n,1) and boxes.shape==(n,4) and indices.shape==(n,),'Invalid targets')
        require(all(torch.isfinite(x).all().item() for x in (labels,boxes,indices)),'Nonfinite targets')
        require(torch.all(indices==indices.floor()).item() and torch.all(indices[1:]>=indices[:-1]).item() and torch.all((indices>=0)&(indices<len(images))).item(),'Invalid image indices')
        require(torch.all(labels==labels.floor()).item() and torch.all((labels>=0)&(labels<len(self.names[task]))).item(),'Invalid source class IDs')
        require(torch.all((boxes>=0)&(boxes<=1)).item() and torch.all(boxes[:,2:]>0).item(),'Invalid normalized boxes')
        if task=='stage':
            require('masks' in batch and 'sem_masks' in batch,'Missing stage masks')
            masks,semantic=batch['masks'],batch['sem_masks'];require(masks.ndim==3 and semantic.ndim==3 and masks.shape[0]==n and semantic.shape[0]==len(images),'Mask indexing mismatch')
            require(torch.isfinite(masks).all().item() and torch.all((masks==0)|(masks==1)).item(),'Invalid instance masks')
            require(torch.isfinite(semantic).all().item() and torch.all(semantic==semantic.floor()).item() and torch.all((semantic>=0)&(semantic<len(self.names[task]))).item(),'Invalid semantic masks')
        else: require('masks' not in batch and 'sem_masks' not in batch,'Detection data must not carry fabricated masks')
    def source_loss(self,task,batch):
        require(self.training,'Loss requires training mode');self.validate_batch(task,batch)
        if not self.criteria: self.prepare_criteria()
        vector,details=self.criteria[task](self.predict_source(batch['img'],task),batch)
        require(torch.isfinite(vector).all().item(),'Nonfinite loss')
        return vector,details

def build_model(mode='joint',seed=20261106,epochs=40,device='cpu'):
    require(file_sha(OFFICIAL)==OFFICIAL_SHA,'Official initialization checksum changed')
    payload=torch.load(OFFICIAL,map_location='cpu',weights_only=False);reference=payload.get('ema') or payload['model']
    model=SourceAwareYOLO(reference,mode,seed,epochs).to(device).float();model.prepare_criteria()
    return model

class ActiveEMA:
    """EMA only updates the active trunk/head, with per-group update counts."""
    def __init__(self,model,decay=.9999,tau=2000.):
        criteria=model.criteria;model.criteria={}
        try:self.model=copy.deepcopy(model).eval().requires_grad_(False)
        finally:model.criteria=criteria
        self.model.criteria={};self.decay=decay;self.tau=tau
        self.updates={group['component']:0 for group in model.parameter_groups()}
    @torch.no_grad()
    def update(self,model,task):
        source=model.state_dict();target=self.model.state_dict()
        for prefix in model.active_groups(task):
            self.updates[prefix]+=1;factor=self.decay*(1-math.exp(-self.updates[prefix]/self.tau))
            for key,value in target.items():
                if key.startswith(prefix+'.'):
                    if value.is_floating_point():value.mul_(factor).add_(source[key].detach(),alpha=1-factor)
                    else:value.copy_(source[key])
    def state_dict(self):return {'model':self.model.state_dict(),'updates':self.updates,'decay':self.decay,'tau':self.tau}
    def load_state_dict(self,state):
        require(state['decay']==self.decay and state['tau']==self.tau,'EMA policy mismatch')
        self.model.load_state_dict(state['model'],strict=True);self.updates=dict(state['updates']);finite_state(self.model.state_dict(),'EMA')

def create_initialization(path,seed=20261106):
    from joint_state_v1 import atomic_checkpoint
    model=build_model('joint',seed,40,'cpu')
    artifact={'schema':'joint_matched_initialization_v1','resumable':False,'epoch_completed':0,'seed':seed,'official_sha256':OFFICIAL_SHA,'trunk':model.trunks['shared'].state_dict(),'heads':{task:model.heads[task].state_dict() for task in TASKS},'initialization':model.initialization,'purpose':'Identical common official trunk and fresh head states for joint and matched separate controls; not a trained model.'}
    return atomic_checkpoint(path,artifact)

def apply_initialization(model,path):
    import json
    path=Path(path);receipt=json.loads(path.with_suffix('.sha256.json').read_text(encoding='utf8'))
    require(file_sha(path)==receipt['sha256'],'Canonical initialization checksum mismatch')
    state=torch.load(path,map_location='cpu',weights_only=False)
    require(state['schema']=='joint_matched_initialization_v1' and state['official_sha256']==OFFICIAL_SHA and state['seed']==model.initialization['seed'],'Canonical initialization identity mismatch')
    for trunk in model.trunks.values():trunk.load_state_dict(state['trunk'],strict=True)
    for task in TASKS:model.heads[task].load_state_dict(state['heads'][task],strict=True)
    require(all(tensor_sha(trunk.state_dict())==state['initialization']['trunk_tensor_sha256'] for trunk in model.trunks.values()),'Canonical trunk restoration failed')
    require(all(tensor_sha(model.heads[task].state_dict())==state['initialization']['heads'][task] for task in TASKS),'Canonical head restoration failed')
    model.initialization.update(canonical_checkpoint=str(path.resolve()),canonical_sha256=receipt['sha256'])
    return receipt
