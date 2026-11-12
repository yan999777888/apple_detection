"""Bounded CPU integration using real source images/labels; no performance claim."""
from __future__ import annotations
import argparse,copy,json,random,sys
from pathlib import Path
sys.dont_write_bytecode=True
from joint_state_v1 import configure,seed_everything,atomic_json,sha,optimizer_steps

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    if args.output.exists():raise ValueError('New test directory required')
    output=configure(args.output,'cpu');seed_everything(20261106)
    import numpy as np,torch
    import torch.nn.functional as F
    from joint_model_v1 import build_model,create_initialization,apply_initialization,tensor_sha,finite_state,require,TASKS
    from joint_state_v1 import SourceProgressSchedule,finite_optimizer
    from joint_engine_v1 import TrainingEngine
    from joint_data import JointDataModule
    checks=[]
    data=JointDataModule();windows=[];fixture_records=[]
    for index,task in enumerate(TASKS):
        first=next(row for row in data.epoch_schedule(0) if row['task']==task);sample_index=first['sample_indices'][0]
        batch=data.batch(task,[sample_index],0)
        batch['img']=F.interpolate(batch['img'],size=(96,96),mode='bilinear',align_corners=False)
        if task=='stage':
            batch['masks']=F.interpolate(batch['masks'].unsqueeze(1).float(),size=(24,24),mode='nearest').squeeze(1)
            batch['sem_masks']=F.interpolate(batch['sem_masks'].unsqueeze(1).float(),size=(24,24),mode='nearest').squeeze(1).long()
        windows.append({'task':task,'batches':[batch],'anchor_count':1,'window_index':index,'source_window_index':0,'source_window_count':1})
        fixture_records.append({'task':task,'anchor_ids':batch['anchor_ids'],'augmentation_source_ids':batch['augmentation_source_ids'],'actual_class_ids':batch['cls'].flatten().tolist(),'objects':len(batch['cls']),'engineering_resize':96,'semantic_scope':'real labels; reduced-resolution contract fixture only, not accuracy/training experiment'})
    checks.append('two_actual_sources_loaded_no_fabricated_labels')
    canonical=output/'canonical_init.pt';create_initialization(canonical)
    joint=build_model('joint');separate=build_model('separate');apply_initialization(joint,canonical);apply_initialization(separate,canonical)
    for task in TASKS:
        require(tensor_sha(joint.heads[task].state_dict())==tensor_sha(separate.heads[task].state_dict()),'Initial head mismatch')
        require(tensor_sha(joint.trunks['shared'].state_dict())==tensor_sha(separate.trunks[task].state_dict()),'Initial trunk mismatch')
    checks.append('canonical_joint_separate_initial_heads_and_trunks_exact')
    # Device-bound head attributes are included in _apply and criterion is rebuilt.
    joint.to(dtype=torch.float64);require(all(head.stride.dtype==torch.float64 for head in joint.heads.values()),'Head stride dtype did not move')
    joint.float();joint.prepare_criteria();checks.append('head_stride_apply_and_criterion_rebuild')
    del separate
    identity={'scope':'CPU_real_data_two_source_minimal_epoch_fixture','windows':{'stage':1,'defect':1},'epochs':40,'source_manifest_sha256':data.manifest_sha256,'data_protocol_sha256':data.protocol_sha256,'canonical_init_sha256':sha(canonical)}
    engine=TrainingEngine(joint,{'stage':1,'defect':1},identity)
    for group in engine.optimizer.param_groups:
        require(all((p.ndim>1)==bool(group['weight_decay']) for p in group['params']),'Weight decay included vector/BN or excluded matrix')
    checks.append('matrix_only_weight_decay')
    require(abs(engine.scheduler.factor(3)-1)<1e-15 and abs(engine.scheduler.factor(40)-.01)<1e-15 and engine.scheduler.factor(0)==0,'Wrong cosine endpoints')
    dummy=torch.nn.Parameter(torch.zeros(1));opt=torch.optim.AdamW([{'params':[dummy],'name':'heads.stage.no_decay','component':'heads.stage'}])
    schedule=SourceProgressSchedule(opt,'joint',{'stage':49,'defect':38});rates=schedule.apply('stage')
    require(abs(rates['heads.stage.no_decay']-.001/49/3)<1e-15,'First head LR progress incorrect')
    checks.append('next_step_cosine_schedule_endpoints_and_first_step')
    # Controlled sum loss proves no extra microbatch factor; native YOLO itself is not batch-invariant.
    targets=torch.tensor([1.,2.,3.,4.,5.,6.,7.,8.]);parameter=torch.tensor(2.,requires_grad=True)
    ((parameter*targets).sum()/8).backward();single=parameter.grad.clone();parameter.grad=None
    for chunk in targets.split(4):((parameter*chunk).sum()/8).backward()
    require(torch.equal(single,parameter.grad),'Double-weighted accumulation')
    parameter.grad=None
    ((parameter*targets[:3]).sum()/3).backward();require(float(parameter.grad)==2.,'Tail denominator not actual count')
    checks.append('synthetic_batch_scaled_sum_normalization_and_tail')
    finite_state({'noncontiguous':torch.arange(20.).reshape(4,5).T,'scalar':torch.tensor(1.),'empty':torch.empty(0)},'finite contract',max_chunk_elements=7)
    rejected=[]
    for label,value in [('NaN',torch.tensor(float('nan'))),('Inf',torch.tensor(float('inf'))),('noncontiguous_bad',torch.tensor([[1.,float('nan')],[2.,3.]]).T)]:
        try:finite_state({label:value},'negative contract',max_chunk_elements=2)
        except ValueError:rejected.append(label)
        else:raise AssertionError('Nonfinite values accepted')
    checks.append('finite_chunk_noncontiguous_scalar_empty_NaN_Inf_contract')
    # First tiny epoch: inspect inactive raw and EMA tensors across each active step.
    for window in windows:
        inactive='defect' if window['task']=='stage' else 'stage'
        before=tensor_sha(engine.model.heads[inactive].state_dict());ema_before=tensor_sha(engine.ema.model.heads[inactive].state_dict());count=engine.ema.updates['heads.'+inactive]
        engine.train_window(window,0)
        require(before==tensor_sha(engine.model.heads[inactive].state_dict()),'Inactive raw head changed')
        require(ema_before==tensor_sha(engine.ema.model.heads[inactive].state_dict()) and count==engine.ema.updates['heads.'+inactive],'Inactive EMA changed')
    engine.complete_epoch(0,{'scope':'two-source one-real-anchor fixture'});checks.append('real_loss_backward_optimizer_and_inactive_head_EMA_unchanged')
    checkpoint=output/'epoch_001.pt';receipt=engine.checkpoint(checkpoint)
    draws_reference=[random.random(),float(np.random.rand()),float(torch.rand(1))]
    for window in windows:engine.train_window(window,1)
    engine.complete_epoch(1,{'scope':'same reduced real fixture; restore-equivalence only'})
    reference_raw=tensor_sha(engine.model.state_dict());reference_ema=tensor_sha(engine.ema.model.state_dict())
    reference_losses=[row['loss'] for row in engine.history[-2:]]
    reference_optimizer={str(i)+'.'+key:value for i,state in enumerate(engine.optimizer.state.values()) for key,value in state.items() if torch.is_tensor(value)}
    reference_optimizer_sha=tensor_sha(reference_optimizer);reference_rng=torch.get_rng_state().clone()
    restored_model=build_model('joint');apply_initialization(restored_model,canonical)
    restored=TrainingEngine(restored_model,{'stage':1,'defect':1},identity);restored.restore(checkpoint)
    draws_restored=[random.random(),float(np.random.rand()),float(torch.rand(1))];require(draws_reference==draws_restored,'RNG restoration differs')
    for window in windows:restored.train_window(window,1)
    restored.complete_epoch(1,{'scope':'same reduced real fixture; restore-equivalence only'})
    restored_optimizer={str(i)+'.'+key:value for i,state in enumerate(restored.optimizer.state.values()) for key,value in state.items() if torch.is_tensor(value)}
    require(tensor_sha(restored.model.state_dict())==reference_raw and tensor_sha(restored.ema.model.state_dict())==reference_ema,'Continued raw/EMA tensors differ after checkpoint restore')
    require(tensor_sha(restored_optimizer)==reference_optimizer_sha,'Optimizer continuation differs')
    require([row['loss'] for row in restored.history[-2:]]==reference_losses,'Continued losses differ')
    require(torch.equal(torch.get_rng_state(),reference_rng),'Continued torch RNG differs')
    require(restored.scheduler.state_dict()==engine.scheduler.state_dict() and restored.model.criterion_state()==engine.model.criterion_state(),'Schedule continuation differs')
    checks.append('disk_save_reload_next_real_steps_bitwise_raw_EMA_optimizer_loss_RNG_schedule')
    require(all(state['updates']==2 and state['hyp_epochs']==40 for state in restored.model.criterion_state().values()),'Source criterion did not update each completed cycle')
    finite_optimizer(restored.optimizer);steps=restored.verify_update_counts();checks.append('actual_parameter_steps_and_full_E2E_criterion_schedule')
    # An explicitly synthetic empty/background unit fixture exercises legitimate
    # missing gradients within an ACTIVE task. It is never a research annotation.
    background_model=build_model('joint');apply_initialization(background_model,canonical)
    background_engine=TrainingEngine(background_model,{'stage':1,'defect':1},{'scope':'synthetic_empty_background_contract'})
    background={'source':'defect','annotation_available':{'stage':False,'defect':True},'img':torch.zeros(1,3,96,96),'batch_idx':torch.empty(0,dtype=torch.long),'cls':torch.empty(0,1),'bboxes':torch.empty(0,4),'anchor_ids':['SYNTHETIC_UNIT_FIXTURE_EMPTY_BACKGROUND']}
    bg_row=background_engine.train_window({'task':'defect','batches':[background],'anchor_count':1,'window_index':0,'source_window_index':0,'source_window_count':1},0)
    background_engine.verify_update_counts();checks.append('synthetic_active_empty_background_tracks_actual_parameter_participation')
    opt_state=next(state for state in restored.optimizer.state.values() if 'exp_avg' in state)
    saved=opt_state['exp_avg'].clone();opt_state['exp_avg'].view(-1)[0]=float('inf')
    try:finite_optimizer(restored.optimizer)
    except ValueError:checks.append('optimizer_nonfinite_state_rejected')
    else:raise AssertionError('Nonfinite AdamW moment accepted')
    finally:opt_state['exp_avg'].copy_(saved)
    data.verify_sources();require(not torch.cuda.is_initialized(),'CPU test initialized CUDA')
    report={'schema':'joint_full_cpu_integration_v1','phase':'passed','checks':checks,'checks_count':len(checks),'source_fixtures':fixture_records,'source_hash_recheck_count':len(data.source_hashes),'canonical_checkpoint':str(canonical),'checkpoint_receipt':receipt,'full_40cycle_training':False,'actual_optimizer_steps_executed':7,'synthetic_background_participating_parameters':bg_row['participating_parameter_tensors'],'reference_logical_updates':engine.global_updates,'restored_logical_updates':restored.global_updates,'parameter_steps':steps,'component_EMA_updates':restored.ema.updates,'criterion_state':restored.model.criterion_state(),'restore_continuation_bitwise_equal':True,'continued_losses':reference_losses,'RNG_draws_equal':True,'nonfinite_rejections':rejected,'CPU_threads':torch.get_num_threads(),'CUDA_initialized':False,'source_code_sha256':{str(path):sha(path) for path in Path(__file__).parent.glob('*v1.py')},'limitations':['Tiny real-data resize96 contract test is not training/evaluation evidence.','Native YOLO loss/BN is not generally invariant to microbatch partition; only controlled sum normalization is tested.','CUDA resume equivalence must be empirically verified on the target GPU; CPU test cannot prove it.']}
    atomic_json(output/'cpu_integration_report.json',report);print(json.dumps({'phase':'passed','checks':len(checks),'output':str(output/'cpu_integration_report.json'),'actual_optimizer_steps':7},ensure_ascii=False),flush=True)

if __name__=='__main__':main()
