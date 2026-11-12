"""Full source validation + interleaved FP32/AMP deployment benchmark.

No training, source data edits, resolution changes or confidence tuning.
Source AP uses the frozen official rect640/conf.001 protocol. Deployment speed
uses square640/conf.25, batch1, BOTH tasks including native full-size CPU masks.
"""
import argparse,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'apple_research_v2/joint'))
from joint_state_v1 import configure,sha,atomic_json


def fixed_defect_metrics(worker):
    """Micro P/R at the SAME square640/.25 inference used by timing.

    Match score-descending predictions to unused same-class GT at box IoU>=.5.
    This explicitly named metric is separate from Ultralytics' optimal-F1 P/R.
    The stage output is still computed; SSDA has no stage annotation.
    """
    import cv2,numpy as np
    from joint_eval_v1 import source_spec
    from ultralytics.data.utils import img2label_paths
    rows=[];by_class={k:{'TP':0,'FP':0,'FN':0} for k in (0,1)}
    for path in source_spec('defect')['images']:
        im=cv2.imread(path);height,width=im.shape[:2]
        label=Path(img2label_paths([path])[0]);gt=[]
        for line in label.read_text(encoding='utf-8-sig').splitlines():
            if not line.strip():continue
            cls,x,y,w,h=map(float,line.split());assert cls in (0,1)
            gt.append((int(cls),np.array([(x-w/2)*width,(y-h/2)*height,(x+w/2)*width,(y+h/2)*height])))
        pred=worker.predict(im,cpu=True)['defect'][0].boxes
        coords=pred.xyxy.numpy();scores=pred.conf.numpy();classes=pred.cls.numpy().astype(int)
        used=set();matches=[];fp=[]
        for pi in np.argsort(-scores,kind='stable'):
            box=coords[pi];best=-1;best_iou=0.
            for gi,(cls,target) in enumerate(gt):
                if gi in used or cls!=classes[pi]:continue
                inter=np.maximum(np.minimum(box[2:],target[2:])-np.maximum(box[:2],target[:2]),0).prod()
                union=np.maximum(box[2:]-box[:2],0).prod()+np.maximum(target[2:]-target[:2],0).prod()-inter
                iou=float(inter/max(union,1e-9))
                if iou>best_iou:best_iou=iou;best=gi
            cls=int(classes[pi])
            if best>=0 and best_iou>=.5:
                used.add(best);by_class[cls]['TP']+=1;matches.append(best)
            else:by_class[cls]['FP']+=1;fp.append(int(pi))
        missing=[gi for gi in range(len(gt)) if gi not in used]
        for gi in missing:by_class[gt[gi][0]]['FN']+=1
        rows.append({'image':path,'gt_count':len(gt),'matched_gt_indices':matches,'missed_gt_indices':missing,'fp_count':len(fp)})
    total={key:sum(v[key] for v in by_class.values()) for key in ('TP','FP','FN')}
    assert len(rows)==113 and total['TP']+total['FN']==230
    total.update(P=total['TP']/max(total['TP']+total['FP'],1),R=total['TP']/230)
    return {'protocol':'class-aware score-descending greedy matching; boxIoU>=.5 square640 conf.25 maxdet300 both heads computed','total':total,'per_class':by_class,'per_image':rows}


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--sha256',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise ValueError('Fresh output only')
    if not a.checkpoint.resolve().is_relative_to(ROOT) or sha(a.checkpoint)!=a.sha256:raise ValueError('Source checksum/location mismatch')
    out=configure(a.output,'cuda:0')
    import cv2,torch
    from joint_model_v1 import build_model,tensor_sha
    from joint_eval_v1 import source_spec,evaluate_task
    from deployment_adapter_v1 import FrozenDeploymentAdapter
    from deployment_amp_20261112 import FrozenAMPDeploymentAdapter
    from benchmark_joint_20261112 import summary
    torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.backends.cudnn.benchmark=False
    result={'checkpoint':str(a.checkpoint.resolve()),'sha256':a.sha256,'no_training':True,'validation':{},'benchmark':{},'gpu':torch.cuda.get_device_name(0),'torch':str(torch.__version__)}
    try:
        state=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
        if state.get('schema') not in ('retained_task_composite_v1','strong_defect_bn_refine_v1'):raise ValueError('Unsupported checkpoint')
        model=build_model('separate',device='cpu');model.load_state_dict(state['model'],strict=True);del state
        model.eval().requires_grad_(False).to('cuda:0');before=tensor_sha(model.state_dict())
        for precision in ('FP32','AMP_FP16'):
            result['validation'][precision]={}
            for task in ('stage','defect'):
                atomic_json(out/'status.json',{'phase':'validating','precision':precision,'task':task})
                receipt=evaluate_task(model,task,output_dir=out/f'{precision}_{task}',device='cuda:0',inference_precision=precision)
                result['validation'][precision][task]={'metrics':receipt['metrics'],'per_class':receipt['per_class'],'seen':receipt['seen'],'inference_precision':receipt['inference_precision']}
        workers={'FP32':FrozenDeploymentAdapter(model),'AMP_FP16':FrozenAMPDeploymentAdapter(model)}
        result['fixed_deployment_defect']={}
        for precision,worker in workers.items():
            atomic_json(out/'status.json',{'phase':'fixed_deployment_accuracy','precision':precision})
            result['fixed_deployment_defect'][precision]=fixed_defect_metrics(worker)
        paths=[]
        for task in ('stage','defect'):
            images=source_spec(task)['images'];paths.extend(images[round(i*(len(images)-1)/19)] for i in range(20))
        images=[cv2.imread(p) for p in paths]
        if any(im is None for im in images):raise ValueError('Unreadable image')
        times={k:[] for k in workers};counts={k:[] for k in workers}
        atomic_json(out/'status.json',{'phase':'benchmarking'})
        # Reverse order every round to reduce drift bias; warm each precision
        # independently, then alternate 40-image blocks. Keep all measurements.
        for worker in workers.values():
            for i in range(10):worker.predict(images[i],cpu=True)
        orders=[('FP32','AMP_FP16'),('AMP_FP16','FP32')]*2
        for order in orders:
            for precision in order:
                for im in images:
                    torch.cuda.synchronize();start=time.perf_counter();pred=workers[precision].predict(im,cpu=True);torch.cuda.synchronize()
                    times[precision].append((time.perf_counter()-start)*1000)
                    counts[precision].append({task:len(pred[task][0].boxes) for task in ('stage','defect')});del pred
        for precision in workers:
            result['benchmark'][precision]={'summary':summary(times[precision]),'observations_ms':times[precision],'detections':counts[precision]}
        result['benchmark_protocol']={'images':paths,'warmup_per_precision':10,'rounds':4,'orders':orders,'batch':1,'imgsz':640,'rect':False,'conf':.25,'max_det':300,'cpu_results':True,'full_resolution_stage_masks':True,'scope':'BGR image array to both tasks CPU Results; excludes disk, camera, tracking, depth, robot'}
        assert tensor_sha(model.state_dict())==before
        assert sha(a.checkpoint)==a.sha256
        result.update(phase='complete',weights_unchanged=True,automatic_promotion=False)
        atomic_json(out/'results.json',result);atomic_json(out/'status.json',{'phase':'complete'})
    except BaseException as error:
        atomic_json(out/'status.json',{'phase':'failed','error':repr(error),'traceback':traceback.format_exc()});raise


if __name__=='__main__':main()
