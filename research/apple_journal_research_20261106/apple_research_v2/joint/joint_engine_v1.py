"""Full source-aware training core; all checkpoints retain raw training state."""
from __future__ import annotations
import copy,json,time
from pathlib import Path
import torch
from joint_model_v1 import TASKS,ActiveEMA,finite_state,require,tensor_sha
from joint_state_v1 import SourceProgressSchedule,atomic_checkpoint,atomic_json,canonical_sha,finite_optimizer,get_rng,load_checkpoint,optimizer_steps,set_rng

def device_batch(batch,device):
    return {key:value.to(device,non_blocking=False) if torch.is_tensor(value) else value for key,value in batch.items()}

class TrainingEngine:
    def __init__(self,model,windows,identity,lr0=.001,weight_decay=.0005,grad_clip=10.):
        self.model=model;self.identity=copy.deepcopy(identity);self.windows=dict(windows)
        self.optimizer=torch.optim.AdamW(model.parameter_groups(weight_decay),lr=lr0,betas=(.9,.999),eps=1e-8,weight_decay=weight_decay)
        self.scheduler=SourceProgressSchedule(self.optimizer,model.mode,windows,epochs=model.epochs,lr0=lr0)
        self.ema=ActiveEMA(model);self.grad_clip=grad_clip;self.epoch_completed=0;self.global_updates=0
        self.group_updates={group['component']:0 for group in self.optimizer.param_groups}
        self.parameter_updates={name:0 for name,_ in model.named_parameters()}
        self.history=[];self.epochs=[];self.validation=[];self.initialization=copy.deepcopy(model.initialization)
    def train_window(self,window,epoch):
        task=window['task'];batches=window['batches'];count=sum(len(batch['img']) for batch in batches)
        require(task in TASKS and batches and count==window['anchor_count'] and 0<count<=16,'Invalid source window')
        require(epoch==self.epoch_completed,'Invalid training epoch cursor')
        require(self.model.criterion_epochs[task]==epoch,'Loss schedule is not at active source epoch')
        self.model.train();self.optimizer.zero_grad(set_to_none=True)
        lrs=self.scheduler.apply(task);loss_total=0.;vector_total=None;anchor_ids=[]
        start=time.perf_counter();device=next(self.model.parameters()).device
        for batch in batches:
            require(len(batch['img'])<=4,'Microbatch exceeds fixed protocol')
            moved=device_batch(batch,device);vector,_=self.model.source_loss(task,moved)
            # Upstream vector is already multiplied by microbatch image count.
            # Sum it once and divide by the actual window count, including tail.
            normalized=vector.sum()/count;normalized.backward()
            loss_total+=float(normalized.detach());values=vector.detach().cpu()/count
            vector_total=values if vector_total is None else vector_total+values
            anchor_ids.extend(map(str,batch.get('anchor_ids',batch.get('im_file',[]))))
        active=set(self.model.active_groups(task));inactive_params=[]
        for group in self.optimizer.param_groups:
            if group['component'] not in active:inactive_params.extend(group['params'])
        require(all(parameter.grad is None for parameter in inactive_params),'Inactive source received gradient')
        parameters=[p for group in self.optimizer.param_groups if group['component'] in active for p in group['params'] if p.grad is not None]
        require(parameters,'Active loss produced no gradient')
        participating=[name for name,p in self.model.named_parameters() if p.grad is not None]
        norm=torch.nn.utils.clip_grad_norm_(parameters,self.grad_clip,error_if_nonfinite=True)
        require(torch.isfinite(norm).item(),'Nonfinite gradient norm')
        finite_state(self.model.state_dict(),'Raw model before optimizer step');finite_optimizer(self.optimizer)
        self.optimizer.step();self.scheduler.step(task);self.global_updates+=1
        for name in participating:self.parameter_updates[name]+=1
        for name in active:self.group_updates[name]+=1
        self.ema.update(self.model,task)
        finite_state(self.model.state_dict(),'Raw model after optimizer step');finite_optimizer(self.optimizer)
        row={'epoch':epoch+1,'global_update':self.global_updates,'task':task,'window_index':window['window_index'],'source_window_index':window['source_window_index'],'anchor_count':count,'microbatch_sizes':[len(b['img']) for b in batches],'loss':loss_total,'loss_vector':vector_total.tolist(),'pre_clip_gradient_norm':float(norm),'lr':{group['name']:group['lr'] for group in self.optimizer.param_groups if group['component'] in active},'active_groups':sorted(active),'inactive_gradients_none':True,'participating_parameter_tensors':len(participating),'anchor_ids':anchor_ids,'elapsed_seconds_not_benchmark':time.perf_counter()-start}
        self.history.append(row);self.optimizer.zero_grad(set_to_none=True)
        return row
    def complete_epoch(self,epoch,coverage):
        require(epoch==self.epoch_completed,'Wrong completion cursor')
        expected={task:(epoch+1)*self.windows[task] for task in TASKS}
        require(self.scheduler.completed==expected,'Incomplete per-source optimizer coverage')
        self.epoch_completed=epoch+1
        # Source criterion schedule advances once each completed full source cycle.
        self.model.set_criterion_epoch(self.epoch_completed)
        self.epochs.append({'epoch':self.epoch_completed,'coverage':coverage,'source_updates':dict(self.scheduler.completed),'criterion_state':self.model.criterion_state()})
    def verify_update_counts(self):
        steps=optimizer_steps(self.optimizer)
        for name,parameter in self.model.named_parameters():
            actual=int(self.optimizer.state.get(parameter,{}).get('step',0))
            require(actual==self.parameter_updates[name],'Optimizer parameter participation mismatch '+name)
        require(self.ema.updates==self.group_updates,'EMA update counts differ from active optimizer groups')
        return steps
    def checkpoint(self,path):
        require(self.global_updates==self.epoch_completed*sum(self.windows.values()),'Only completed epoch boundaries are resumable')
        require(all(p.grad is None for p in self.model.parameters()),'Checkpoint contains pending accumulated gradients')
        finite_state(self.model.state_dict(),'Raw checkpoint model');finite_state(self.ema.model.state_dict(),'EMA checkpoint model');finite_optimizer(self.optimizer)
        steps=self.verify_update_counts()
        state={'schema':'joint_exact_training_checkpoint_v1','resumable':True,'resume_scope':'completed full-source-cycle epoch boundary; no pending gradients','identity':self.identity,'identity_sha256':canonical_sha(self.identity),'mode':self.model.mode,'epoch_completed':self.epoch_completed,'global_updates':self.global_updates,'group_updates':self.group_updates,'parameter_updates':self.parameter_updates,'raw_model':self.model.state_dict(),'ema':self.ema.state_dict(),'optimizer':self.optimizer.state_dict(),'scheduler':self.scheduler.state_dict(),'criterion_state':self.model.criterion_state(),'rng':get_rng(),'initialization':self.initialization,'history':self.history,'epochs':self.epochs,'validation':self.validation,'optimizer_step_verification':steps,'raw_tensor_sha256':tensor_sha(self.model.state_dict()),'ema_tensor_sha256':tensor_sha(self.ema.model.state_dict()),'amp':False,'scaler':None,'inference_cache_policy':'Non-parameter head anchor caches rebuilt deterministically; raw/EMA parameters and persistent buffers saved.'}
        return atomic_checkpoint(path,state)
    def restore(self,path):
        state=load_checkpoint(path)
        require(state['identity']==self.identity and state['identity_sha256']==canonical_sha(self.identity),'Resume identity/input/source/protocol mismatch')
        require(state['mode']==self.model.mode,'Resume mode mismatch')
        self.model.load_state_dict(state['raw_model'],strict=True);self.ema.load_state_dict(state['ema'])
        self.optimizer.load_state_dict(state['optimizer']);self.scheduler.load_state_dict(state['scheduler'])
        self.epoch_completed=state['epoch_completed'];self.global_updates=state['global_updates'];self.group_updates=dict(state['group_updates']);self.parameter_updates=dict(state['parameter_updates'])
        self.model.set_criterion_epoch(self.epoch_completed)
        require(self.model.criterion_state()==state['criterion_state'],'E2E criterion schedule mismatch after restore')
        self.history=state['history'];self.epochs=state['epochs'];self.validation=state['validation'];self.initialization=state['initialization']
        require(tensor_sha(self.model.state_dict())==state['raw_tensor_sha256'] and tensor_sha(self.ema.model.state_dict())==state['ema_tensor_sha256'],'Checkpoint tensor reconstruction mismatch')
        finite_optimizer(self.optimizer);finite_state(self.model.state_dict(),'Restored model');self.verify_update_counts();set_rng(state['rng'])
        return state
