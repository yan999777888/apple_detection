"""Root-scheduled CUDA contract smoke, never an accuracy or speed experiment."""
from __future__ import annotations
import argparse,json,sys,traceback,warnings
from pathlib import Path
sys.dont_write_bytecode=True
from joint_state_v1 import configure,seed_everything,atomic_json,sha

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--init-checkpoint',type=Path,required=True);parser.add_argument('--device',default='cuda:0');parser.add_argument('--concurrent-context',required=True,help='Explicit other GPU workload status; no timings are benchmarks')
    args=parser.parse_args()
    if args.output.exists():raise ValueError('New smoke output required')
    if not args.device.startswith('cuda'):raise ValueError('This smoke is explicit CUDA only; use CPU test script for CPU')
    output=configure(args.output,args.device);seed_everything(20261106)
    import torch
    from joint_model_v1 import build_model,apply_initialization,require,tensor_sha,TASKS,finite_state
    from joint_engine_v1 import TrainingEngine
    from joint_data import JointDataModule
    from joint_eval_v1 import evaluate_task
    device=torch.device(args.device);require(torch.cuda.is_available(),'CUDA is unavailable')
    torch.cuda.set_device(device);free_before,total=torch.cuda.mem_get_info(device);torch.cuda.reset_peak_memory_stats(device)
    data=JointDataModule();model=build_model('joint',device=device);apply_initialization(model,args.init_checkpoint)
    identity={'schema':'joint_cuda_two_source_fixture_v1','purpose':'actual640 microbatch4 GPU contract; fixture cycle has2 source windows, NOT full87-window protocol','source_manifest_sha256':data.manifest_sha256,'data_protocol_sha256':data.protocol_sha256,'canonical_init_sha256':sha(args.init_checkpoint),'concurrent_context':args.concurrent_context}
    engine=TrainingEngine(model,{'stage':1,'defect':1},identity);rows=[];captured=[]
    try:
        for window in data.iter_epoch(0):
            if len(rows)==2:break
            require(window['source_window_index']==0 and window['anchor_count']==16,'Expected first full source windows')
            require([len(batch['img']) for batch in window['batches']]==[4,4,4,4],'Actual microbatch4 required')
            task=window['task'];inactive='defect' if task=='stage' else 'stage'
            before=tensor_sha(engine.model.heads[inactive].state_dict());ema_before=tensor_sha(engine.ema.model.heads[inactive].state_dict())
            with warnings.catch_warnings(record=True) as notices:
                warnings.simplefilter('always');row=engine.train_window(window,0)
            captured.extend(str(notice.message) for notice in notices)
            require(before==tensor_sha(engine.model.heads[inactive].state_dict()) and ema_before==tensor_sha(engine.ema.model.heads[inactive].state_dict()),'Inactive task changed')
            require(all(getattr(head,name).device==device for head in model.heads.values() for name in ('stride','anchors','strides')),'Head auxiliary device mismatch')
            require(all(criterion.one2many.device==device and criterion.one2one.device==device for criterion in model.criteria.values()),'Loss tensor device mismatch')
            rows.append(row);print(json.dumps({'phase':'GPU_contract_step','task':task,'anchors':16,'microbatch':4,'updates':engine.global_updates},ensure_ascii=False),flush=True)
        require(len(rows)==2 and {row['task'] for row in rows}==set(TASKS),'Missing actual source smoke')
        engine.complete_epoch(0,{'scope':'engineering fixture cycle2 blocks only; not full data coverage','anchor_counts':{'stage':16,'defect':16}})
        receipt=engine.checkpoint(output/'fixture_cycle_001.pt')
        raw_before=tensor_sha(engine.model.state_dict());ema_before=tensor_sha(engine.ema.model.state_dict())
        # Restore into the same device-resident engine to exercise optimizer-device
        # reconstruction without unnecessary duplicate GPU model memory.
        engine.restore(output/'fixture_cycle_001.pt')
        require(raw_before==tensor_sha(engine.model.state_dict()) and ema_before==tensor_sha(engine.ema.model.state_dict()),'GPU checkpoint tensor roundtrip differs')
        validation={}
        for task in TASKS:
            # Exercise the unchanged production validator on its entire fixed
            # development source. These are still two-step engineering weights.
            result=evaluate_task(engine.ema.model,task,data.source_specs[task]['data'],output/('validator_'+task),device=str(device),batch=1,imgsz=640,scope='full_development')
            require(result['phase']=='complete' and result['seen']==data.counts[task]['val'],'Incomplete actual GPU development coverage')
            validation[task]={'seen':result['seen'],'state_guard':result['state_guard'],'scope':result['scope'],'metric_scope':'complete development contract on two-step engineering weights; not final research accuracy','receipt_directory':str(output/('validator_'+task))}
        finite_state(engine.model.state_dict(),'GPU raw final');finite_state(engine.ema.model.state_dict(),'GPU EMA final');data.verify_sources()
        free_after,_=torch.cuda.mem_get_info(device)
        report={'schema':'joint_cuda_contract_smoke_v1_1','phase':'passed','device':str(device),'GPU_name':torch.cuda.get_device_name(device),'concurrent_context':args.concurrent_context,'isolated_speed_benchmark':False,'scientific_accuracy_result':False,'full_training_completed':False,'actual_optimizer_updates':engine.global_updates,'real_anchors':32,'imgsz':640,'microbatch':4,'source_blocks':[{'task':row['task'],'anchors':row['anchor_count'],'microbatch_sizes':row['microbatch_sizes'],'loss_finite':True,'inactive_gradients_none':row['inactive_gradients_none'],'participating_parameter_tensors':row['participating_parameter_tensors']} for row in rows],'parameter_steps':engine.verify_update_counts(),'component_EMA_updates':engine.ema.updates,'checkpoint':receipt,'roundtrip_raw_EMA_exact':True,'next_step_CUDA_resume_equivalence_tested':False,'GPU_memory_bytes':{'peak_allocated_this_process':torch.cuda.max_memory_allocated(device),'peak_reserved_this_process':torch.cuda.max_memory_reserved(device),'global_free_before':free_before,'global_free_after':free_after,'global_total':total},'fixed_model_validators':validation,'warnings':sorted(set(captured)),'source_hash_recheck':{'passed':True,'count':len(data.source_hashes)},'script_sha256':sha(__file__),'source_code_sha256':{str(path):sha(path) for path in Path(__file__).parent.glob('joint_*v1.py')},'limitations':['Only two actual training source blocks, not one full research epoch; full72/113 development passes validate interface and coverage only.','Global free memory includes concurrent jobs; no speed/FPS claims.','GPU checkpoint tensors roundtrip exactly; next-step numerical continuation was not tested here.']}
        atomic_json(output/'cuda_contract_report.json',report);print(json.dumps({'phase':'passed','report':str(output/'cuda_contract_report.json'),'updates':2,'peak_allocated':report['GPU_memory_bytes']['peak_allocated_this_process']},ensure_ascii=False),flush=True)
    except Exception as error:
        atomic_json(output/'cuda_contract_failure.json',{'phase':'failed','error':repr(error),'traceback':traceback.format_exc(),'actual_optimizer_updates':engine.global_updates,'concurrent_context':args.concurrent_context,'full_training_completed':False});raise

if __name__=='__main__':main()
