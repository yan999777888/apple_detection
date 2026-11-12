"""F-only runtime, exact training state IO, and source-progress LR schedule."""
from __future__ import annotations
import hashlib,json,math,os,random,sys,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]

def configure(output,device):
    output=Path(output).resolve()
    if output.drive.lower()!='f:' or not output.is_relative_to(ROOT):raise ValueError('All output must remain in the F research project')
    sys.dont_write_bytecode=True
    os.environ.update(PYTHONDONTWRITEBYTECODE='1',YOLO_AUTOINSTALL='False',HF_HUB_OFFLINE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',CUBLAS_WORKSPACE_CONFIG=':4096:8')
    if str(device)=='cpu':os.environ['CUDA_VISIBLE_DEVICES']=''
    for key in ('TEMP','TMP','YOLO_CONFIG_DIR','MPLCONFIGDIR','TORCH_HOME','HF_HOME','XDG_CACHE_HOME','CUDA_CACHE_PATH','TORCH_EXTENSIONS_DIR'):
        path=output/'runtime'/key.lower();path.mkdir(parents=True,exist_ok=True);os.environ[key]=str(path)
    return output

def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

def canonical_sha(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()

def atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fd,temp=tempfile.mkstemp(prefix=path.name+'.',suffix='.tmp',dir=path.parent)
    try:
        with os.fdopen(fd,'w',encoding='utf8') as stream:json.dump(value,stream,indent=2,ensure_ascii=False,allow_nan=False);stream.flush();os.fsync(stream.fileno())
        os.replace(temp,path)
    finally:
        if os.path.exists(temp):os.unlink(temp)

def atomic_checkpoint(path,state):
    import torch
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():raise ValueError('Immutable checkpoint already exists: '+str(path))
    fd,temp=tempfile.mkstemp(prefix=path.name+'.',suffix='.tmp',dir=path.parent)
    try:
        with os.fdopen(fd,'wb') as stream:torch.save(state,stream);stream.flush();os.fsync(stream.fileno())
        os.replace(temp,path)
    finally:
        if os.path.exists(temp):os.unlink(temp)
    receipt={'sha256':sha(path),'file':str(path),'schema':'joint_exact_checkpoint_sha_v1','resumable':state['resumable'],'epoch_completed':state['epoch_completed']}
    atomic_json(path.with_suffix('.sha256.json'),receipt)
    return receipt

def load_checkpoint(path):
    import torch
    path=Path(path);receipt=json.loads(path.with_suffix('.sha256.json').read_text(encoding='utf8'))
    if sha(path)!=receipt['sha256']:raise ValueError('Checkpoint checksum mismatch')
    checkpoint=torch.load(path,map_location='cpu',weights_only=False)
    if checkpoint.get('schema')!='joint_exact_training_checkpoint_v1' or not checkpoint.get('resumable'):raise ValueError('Not an exact-resume checkpoint')
    return checkpoint

def get_rng():
    import numpy as np,torch
    return {'python':random.getstate(),'numpy':np.random.get_state(),'torch_cpu':torch.get_rng_state(),'torch_cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else []}

def set_rng(state):
    import numpy as np,torch
    random.setstate(state['python']);np.random.set_state(state['numpy']);torch.set_rng_state(state['torch_cpu'])
    if state['torch_cuda']:
        if not torch.cuda.is_available():raise ValueError('CUDA RNG checkpoint requires CUDA')
        torch.cuda.set_rng_state_all(state['torch_cuda'])

def seed_everything(seed):
    import numpy as np,torch
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True,warn_only=True)

def finite_optimizer(optimizer):
    from joint_model_v1 import finite_state
    tensors={str(index)+'.'+key:value for index,state in enumerate(optimizer.state.values()) for key,value in state.items()}
    finite_state(tensors,'Optimizer state')
def optimizer_steps(optimizer):
    result={}
    for group in optimizer.param_groups:
        steps=[int(optimizer.state[p].get('step',0)) for p in group['params']]
        result[group['name']]={'component':group['component'],'parameters':len(steps),'step_min':min(steps,default=0),'step_max':max(steps,default=0),'step_histogram':{str(x):steps.count(x) for x in sorted(set(steps))}}
    return result

class SourceProgressSchedule:
    """Next-step source progress: warmup 0 to lr0 over 3 cycles; cosine to .01*lr0 at cycle40."""
    def __init__(self,optimizer,mode,windows,epochs=40,lr0=.001,warmup=3.,lrf=.01):
        self.optimizer=optimizer;self.mode=mode;self.windows=dict(windows);self.epochs=epochs;self.lr0=lr0;self.warmup=warmup;self.lrf=lrf
        self.completed={task:0 for task in windows};self.applied=0
    def factor(self,cycle_progress):
        p=min(max(float(cycle_progress),0.),float(self.epochs))
        if p<=self.warmup:return p/self.warmup
        return self.lrf+(1-self.lrf)*.5*(1+math.cos(math.pi*(p-self.warmup)/(self.epochs-self.warmup)))
    def apply(self,task):
        for group in self.optimizer.param_groups:
            name=group['name'];source=group['component'].split('.')[-1]
            progress=(sum(self.completed.values())+1)/sum(self.windows.values()) if source=='shared' else (self.completed[source]+1)/self.windows[source]
            group['lr']=self.lr0*self.factor(progress)
        self.applied+=1
        return {group['name']:group['lr'] for group in self.optimizer.param_groups}
    def step(self,task):self.completed[task]+=1
    def state_dict(self):return {'mode':self.mode,'windows':self.windows,'epochs':self.epochs,'lr0':self.lr0,'warmup':self.warmup,'lrf':self.lrf,'completed':self.completed.copy(),'applied':self.applied,'formula':'p=(component_completed_steps+1)/component_steps_per_cycle; p<=3: p/3; else .01+.99*(1+cos(pi*(p-3)/(epochs-3)))/2'}
    def load_state_dict(self,state):
        for key in ('mode','windows','epochs','lr0','warmup','lrf'):
            if state[key]!=getattr(self,key):raise ValueError('LR schedule mismatch '+key)
        self.completed=dict(state['completed']);self.applied=state['applied']
