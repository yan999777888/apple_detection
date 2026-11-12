"""Fixed 40-cycle joint or matched separate-source YOLO training. No downloads."""
from __future__ import annotations
import argparse,json,sys,time,traceback
from pathlib import Path
sys.dont_write_bytecode=True
from joint_state_v1 import ROOT,configure,atomic_json,sha,canonical_sha,get_rng,set_rng,seed_everything

def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    init=sub.add_parser('initialize');init.add_argument('--output',type=Path,required=True);init.add_argument('--seed',type=int,default=20261106)
    run=sub.add_parser('train');run.add_argument('--mode',choices=('joint','separate'),required=True);run.add_argument('--output',type=Path,required=True);run.add_argument('--init-checkpoint',type=Path,required=True);run.add_argument('--device',default='cpu');run.add_argument('--check-only',action='store_true');run.add_argument('--resume',type=Path);run.add_argument('--manifest',type=Path,default=ROOT/'experiments/joint_training_v1/data_manifest.json');run.add_argument('--data-protocol',type=Path,default=ROOT/'experiments/joint_training_v1/data_protocol.json')
    args=parser.parse_args();exists=args.output.exists()
    if args.command=='initialize':
        output=configure(args.output.parent/'initialization_runtime','cpu');seed_everything(args.seed)
        from joint_model_v1 import create_initialization
        receipt=create_initialization(args.output.resolve(),args.seed);print(json.dumps(receipt,ensure_ascii=False));return
    if exists and args.resume is None:raise ValueError('New run output already exists; use explicit compatible resume')
    if args.check_only and (args.device!='cpu' or args.resume):raise ValueError('Preflight requires CPU and fresh state')
    output=configure(args.output,args.device)
    if args.resume and not args.resume.resolve().is_relative_to(output):raise ValueError('Resume checkpoint must belong to the selected output')
    import torch,ultralytics,importlib.metadata
    from joint_model_v1 import build_model,apply_initialization,require,file_sha,TASKS,OFFICIAL,OFFICIAL_SHA,tensor_sha
    from joint_engine_v1 import TrainingEngine
    from joint_data import JointDataModule
    from joint_eval_v1 import evaluate_task
    require(ultralytics.__version__=='8.4.115','Installed Ultralytics version differs')
    data=JointDataModule(args.manifest,args.data_protocol,workers=0)
    require(data.protocol['epochs']==40 and data.protocol['seed']==20261106 and data.protocol['imgsz']==640,'Unfrozen full training recipe')
    seed_everything(data.protocol['seed']);device=torch.device(args.device)
    require(device.type in ('cpu','cuda'),'Unsupported training device')
    schedule=data.epoch_schedule(0);windows={task:sum(row['task']==task for row in schedule) for task in TASKS};require(windows=={'stage':49,'defect':38},'Unexpected full source coverage')
    code_paths=list(Path(__file__).parent.glob('joint_*v1.py'))+[Path(__file__).parent/'joint_data.py',Path(__file__).resolve()]
    library_root=Path(ultralytics.__file__).parent
    code_paths += [library_root/p for p in ('nn/modules/head.py','nn/tasks.py','utils/loss.py','utils/torch_utils.py','data/augment.py','data/dataset.py','data/base.py','utils/ops.py')]
    source_code={str(path.resolve()):sha(path) for path in sorted(set(code_paths))}
    init_sha=file_sha(args.init_checkpoint)
    recipe={'schema':'joint_matched_full40_protocol_v1','mode':args.mode,'epochs':40,'seed':20261106,'imgsz':640,'microbatch_size':4,'source_window_anchors_max':16,'windows_per_cycle':windows,'data_manifest_sha256':data.manifest_sha256,'data_protocol_sha256':data.protocol_sha256,'optimizer':'AdamW','lr0':.001,'betas':[.9,.999],'epsilon':1e-8,'weight_decay':.0005,'weight_decay_scope':'parameter ndim>1 only; bias/BN/vector tensors zero decay','grad_clip':10.,'loss_normalization':'sum native batch-scaled loss vectors / actual source-block anchor count; preserve identical microbatch partitions across arms; no large-batch invariance claim','lr_formula':'p=(component completed steps+1)/component steps per cycle; p<=3: lr0*p/3; otherwise lr0*(.01+.99*(1+cos(pi*(p-3)/37))/2)','shared_trunk_progress':'all87 source blocks per cycle','task_progress':'own49 stage or38 defect blocks per cycle','EMA':'only active trunk/head; separate per-component update counters; decay=.9999*(1-exp(-updates/2000))','criterion_schedule':'E2ELoss.update once after each fully completed source cycle; all schedule fields saved','AMP':False,'workers':0,'resume_scope':'exact full-source-cycle epoch boundaries, no pending accumulated gradients','validation_cycles':[1,5,10,15,20,25,30,35,40],'primary_selection':'fixed final cycle40 EMA; raw final secondary; no best-development selection','evaluation_scope':'development_non_independent; source-specific labels; no test; no physiological maturity or market grade claim','official_sha256':OFFICIAL_SHA,'canonical_initialization_sha256':init_sha}
    optional_environment={name:(importlib.metadata.version(name) if importlib.metadata.packages_distributions().get(name) else 'absent') for name in ('albumentations','numpy')}
    import cv2
    optional_environment['opencv']=cv2.__version__
    identity={'optional_environment':optional_environment,'recipe':recipe,'source_files':data.source_hashes,'source_code':source_code,'torch':str(torch.__version__),'ultralytics':ultralytics.__version__,'device_type':device.type}
    model=build_model(args.mode,20261106,40,device);apply_initialization(model,args.init_checkpoint);engine=TrainingEngine(model,windows,identity)
    if args.resume:engine.restore(args.resume)
    else:atomic_json(output/'training_protocol.json',recipe);atomic_json(output/'initialization_receipt.json',model.initialization)
    atomic_json(output/'identity.json',identity)
    if args.check_only:
        require(not torch.cuda.is_initialized(),'CPU preflight initialized CUDA')
        receipt={'phase':'preflight_only_passed','mode':args.mode,'source_files_verified':len(data.source_hashes),'canonical_initialization':model.initialization,'source_code_sha256':source_code,'optimizer_updates':0,'CUDA_initialized':False,'parameter_groups':[{'name':group['name'],'parameters':len(group['params']),'weight_decay':group['weight_decay']} for group in engine.optimizer.param_groups],'windows_per_cycle':windows,'criterion_state':model.criterion_state()}
        atomic_json(output/'preflight_report.json',receipt);print(json.dumps({'phase':receipt['phase'],'output':str(output)},ensure_ascii=False),flush=True);return
    try:
        for epoch in range(engine.epoch_completed,40):
            seen={task:[] for task in TASKS};anchors={task:0 for task in TASKS};window_counts={task:0 for task in TASKS}
            for window in data.iter_epoch(epoch):
                row=engine.train_window(window,epoch);task=window['task'];seen[task].extend(row['anchor_ids']);anchors[task]+=row['anchor_count'];window_counts[task]+=1
                atomic_json(output/'status.json',{'phase':'training','epoch':epoch+1,'epochs':40,'global_update':engine.global_updates,'window':window['window_index']+1,'windows':len(schedule),'task':task,'loss':row['loss'],'source_updates':dict(engine.scheduler.completed)})
                print(json.dumps({'event':'optimizer_step','epoch':epoch+1,'step':engine.global_updates,'task':task,'loss':row['loss'],'anchors':row['anchor_count']},ensure_ascii=False),flush=True)
            require(window_counts==windows,'Incomplete source windows')
            for task in TASKS:require(anchors[task]==data.counts[task]['train'] and len(seen[task])==anchors[task] and len(set(seen[task]))==anchors[task],'Missing/duplicate anchor coverage '+task)
            coverage={'source_anchor_counts':anchors,'source_window_counts':window_counts,'anchor_order_sha256':{task:canonical_sha(seen[task]) for task in TASKS},'schedule_sha256':data.schedule_fingerprint(epoch),'no_pending_tail':True}
            engine.complete_epoch(epoch,coverage)
            if epoch+1 in (1,5,10,15,20,25,30,35,40):
                variants=[('ema',engine.ema.model)]+([('raw',engine.model)] if epoch+1==40 else [])
                for variant,evaluated in variants:
                    for task in TASKS:
                        destination=output/'validation'/f'cycle{epoch+1:02d}_{variant}_{task}'
                        attempt=0
                        while destination.exists():
                            attempt+=1;destination=output/'validation'/f'cycle{epoch+1:02d}_{variant}_{task}_attempt{attempt}'
                        rng=get_rng()
                        try:result=evaluate_task(evaluated,task,data.source_specs[task]['data'],destination,device=str(device),batch=1,imgsz=640)
                        finally:set_rng(rng)
                        engine.validation.append({'variant':variant,'task':task,'epoch':epoch+1,'primary':variant=='ema' and epoch+1==40,'receipt':result})
            receipt=engine.checkpoint(output/'checkpoints'/f'epoch_{epoch+1:03d}.pt');atomic_json(output/'latest_checkpoint.json',receipt)
        data.verify_sources()
        for path,expected in source_code.items():require(sha(path)==expected,'Training code changed during run '+path)
        require(file_sha(OFFICIAL)==OFFICIAL_SHA and file_sha(args.init_checkpoint)==init_sha,'Initialization source changed')
        final_path=output/'checkpoints'/'final40_complete.pt';receipt=engine.checkpoint(final_path)
        report={'schema':'joint_full40_training_report_v1','phase':'complete','recipe':recipe,'initialization':engine.initialization,'epochs_completed':engine.epoch_completed,'optimizer_updates':engine.global_updates,'component_updates':engine.group_updates,'parameter_steps':engine.verify_update_counts(),'EMA_updates':engine.ema.updates,'criterion_state':engine.model.criterion_state(),'source_hash_recheck':{'passed':True,'count':len(data.source_hashes)},'source_code_hash_recheck':True,'final_checkpoint':receipt,'validation':engine.validation,'elapsed_scope':'training step durations are diagnostics, not deployment benchmark','primary_selection':'fixed final40 EMA','resume_exact_scope':'epoch boundary on matching software/device; CUDA deterministic warn-only operations require empirical reproducibility verification'}
        atomic_json(output/'training_report.json',report);atomic_json(output/'status.json',{'phase':'complete','epochs':40,'optimizer_updates':engine.global_updates,'final_checkpoint':receipt})
        print(json.dumps({'phase':'complete','output':str(output),'optimizer_updates':engine.global_updates},ensure_ascii=False),flush=True)
    except Exception as error:
        atomic_json(output/'status.json',{'phase':'failed','error':repr(error),'traceback':traceback.format_exc(),'epoch_completed':engine.epoch_completed,'optimizer_updates':engine.global_updates,'last_complete_epoch_checkpoint_only':True})
        raise

if __name__=='__main__':main()
