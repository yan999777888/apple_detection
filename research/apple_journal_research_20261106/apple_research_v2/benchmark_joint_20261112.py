"""Matched shared/separate YOLO26 FP32 benchmark on identical real development frames.
Measures batch-one model-only and BGR-array-to-CPU-Results latency, not camera/robot latency.
Refuses incomplete, unhashed checkpoints or output overwrite. No installs/downloads.
"""
from __future__ import annotations
import argparse, gc, hashlib, json, math, os, statistics, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def summary(values):
    values=sorted(values)
    def quantile(q):
        pos=(len(values)-1)*q; low=int(pos); high=min(low+1,len(values)-1)
        return values[low]+(values[high]-values[low])*(pos-low)
    if not values or not all(math.isfinite(x) and x>0 for x in values):raise ValueError('Invalid timing observations')
    return dict(n=len(values),mean_ms=statistics.mean(values),p50_ms=quantile(.5),p95_ms=quantile(.95),serial_fps=1000/statistics.mean(values))

def verify_run(folder):
    folder=Path(folder).resolve()
    if not folder.is_relative_to(ROOT):raise ValueError('Input outside research root')
    report=json.loads((folder/'training_report.json').read_text(encoding='utf8'))
    status=json.loads((folder/'status.json').read_text(encoding='utf8'))
    if report.get('phase')!='complete' or report.get('epochs_completed')!=40 or status.get('phase')!='complete':raise ValueError('Only completed fixed40 models may be compared')
    cp=Path(report['final_checkpoint']['file']).resolve()
    if not cp.is_relative_to(folder) or cp.name!='final40_complete.pt':raise ValueError('Unexpected final checkpoint')
    receipt=json.loads(cp.with_suffix('.sha256.json').read_text(encoding='utf8'))
    if sha(cp)!=receipt['sha256'] or receipt['sha256']!=report['final_checkpoint']['sha256']:raise ValueError('Checkpoint checksum mismatch')
    return cp,report

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--joint',type=Path);parser.add_argument('--separate',type=Path);parser.add_argument('--output',type=Path)
    parser.add_argument('--frames-per-source',type=int,default=20);parser.add_argument('--repeats',type=int,default=3);parser.add_argument('--warmup',type=int,default=10);parser.add_argument('--self-test',action='store_true')
    args=parser.parse_args()
    if args.self_test:
        r=summary([1.,2.,3.,4.]);assert r['p50_ms']==2.5 and abs(r['p95_ms']-3.85)<1e-9
        try:summary([0.])
        except ValueError:pass
        else:raise AssertionError('Invalid duration accepted')
        print('PASS: timing statistics and invalid-input guard');return
    if not all((args.joint,args.separate,args.output)) or min(args.frames_per_source,args.repeats,args.warmup)<1:parser.error('Two completed runs and fresh output are required')
    output=args.output.resolve()
    if output.exists() or not output.is_relative_to(ROOT):raise ValueError('Fresh output inside F research root required')
    sources={mode:verify_run(getattr(args,mode)) for mode in ('joint','separate')}
    for key in ('canonical_initialization_sha256','data_manifest_sha256','data_protocol_sha256'):
        if sources['joint'][1]['recipe'][key]!=sources['separate'][1]['recipe'][key]:raise ValueError('Unmatched source '+key)
    output.mkdir(parents=True)
    sys.path.insert(0,str(ROOT/'apple_research_v2/joint'))
    from joint_state_v1 import configure,atomic_json,load_checkpoint
    configure(output,'cuda:0')
    os.environ['YOLO_OFFLINE']='1';os.environ['YOLO_AUTOINSTALL']='0'
    import cv2,numpy as np,torch,ultralytics
    from joint_model_v1 import build_model,tensor_sha
    from joint_eval_v1 import JointInferenceAdapter,source_spec,offline_network
    torch.set_num_threads(2);torch.backends.cudnn.benchmark=False
    paths=[]
    for task in ('stage','defect'):
        eligible=source_spec(task)['images'];n=min(args.frames_per_source,len(eligible))
        indices=[round(i*(len(eligible)-1)/max(n-1,1)) for i in range(n)]
        paths.extend(eligible[i] for i in indices)
    images=[cv2.imread(p) for p in paths]
    if any(im is None for im in images):raise ValueError('Cannot read benchmark image')
    result={'scope':'development_image_arrays_only_no_camera_depth_tracking_or_robot','precision':'FP32','imgsz':640,'batch':1,'sources':{k:{'checkpoint':str(v[0]),'sha256':sha(v[0])} for k,v in sources.items()},'images':[{'path':p,'sha256':sha(p)} for p in paths],'torch':str(torch.__version__),'ultralytics':ultralytics.__version__,'gpu':torch.cuda.get_device_name(0),'warmup':args.warmup,'repeats':args.repeats,'models':{},'script_sha256':sha(__file__),'timing_includes_adapter_state_guard':True,'warning':'Sequential isolated microbenchmark; not robot FPS or independent grading accuracy.'}
    atomic_json(output/'protocol.json',result)
    for mode in ('joint','separate'):
        state=load_checkpoint(sources[mode][0]);model=build_model(mode,device='cpu');model.load_state_dict(state['ema']['model'],strict=True)
        if tensor_sha(model.state_dict())!=state['ema_tensor_sha256']:raise ValueError('EMA reconstruction mismatch')
        del state
        model=model.eval().requires_grad_(False).to('cuda:0');adapter=JointInferenceAdapter(model,audit_values=False)
        tensors=[adapter.predictors['stage'].preprocess([im]) for im in images]
        observations=[]
        torch.cuda.reset_peak_memory_stats()
        with offline_network(),torch.inference_mode():
            for i in range(args.warmup):adapter.predict(images[i%len(images)],cpu=True)
            torch.cuda.synchronize()
            for repeat in range(args.repeats):
                for i,im in enumerate(images):
                    torch.cuda.synchronize();start=time.perf_counter();raw=model.predict_both(tensors[i]);torch.cuda.synchronize();raw_ms=(time.perf_counter()-start)*1000;del raw
                    torch.cuda.synchronize();start=time.perf_counter();pred=adapter.predict(im,paths=[paths[i]],cpu=True);torch.cuda.synchronize();full_ms=(time.perf_counter()-start)*1000
                    counts={task:len(pred[task][0].boxes) for task in ('stage','defect')};del pred
                    observations.append({'repeat':repeat,'image':paths[i],'model_only_ms':raw_ms,'array_to_cpu_results_ms':full_ms,'detections':counts})
        result['models'][mode]={'parameters':sum(p.numel() for p in model.parameters()),'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'model_only':summary([r['model_only_ms'] for r in observations]),'array_to_cpu_results':summary([r['array_to_cpu_results_ms'] for r in observations]),'observations':observations}
        atomic_json(output/'results_partial.json',result)
        del tensors,adapter,model;gc.collect();torch.cuda.empty_cache()
    result['phase']='complete';atomic_json(output/'results.json',result)
    print(json.dumps({mode:{k:v for k,v in row.items() if k!='observations'} for mode,row in result['models'].items()},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
