"""Reconstruct best retained task weights in the fast dual-output adapter.
Strict state transfer, no optimization; independently revalidate both tasks.
"""
import argparse,sys,time,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'apple_research_v2/joint'))
from joint_state_v1 import configure,sha,atomic_json,load_checkpoint

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise ValueError('Fresh output only')
    out=configure(a.output,'cuda:0')
    import torch,cv2
    from joint_model_v1 import build_model,tensor_sha
    from joint_eval_v1 import evaluate_task,source_spec
    from deployment_adapter_v1 import FrozenDeploymentAdapter
    from benchmark_joint_20261112 import summary
    stage_cp=ROOT/'runs/separate_matched_e40_s20261112_v1/checkpoints/final40_complete.pt'
    defect_cp=ROOT/'runs/ssda_yolo26n_detect_640_e40_s20261106/weights/last.pt'
    assert sha(defect_cp)=='1d10a4a02a852c85bdd4ea5a34eed11f72c70367445ec6176cd411b9b8b674bc'
    torch.set_num_threads(2);torch.backends.cudnn.benchmark=False
    model=build_model('separate',device='cpu');state=load_checkpoint(stage_cp)
    model.load_state_dict(state['ema']['model']);del state
    original_stage=tensor_sha({k:v for k,v in model.state_dict().items() if k.startswith(('trunks.stage.','heads.stage.'))})
    payload=torch.load(defect_cp,map_location='cpu',weights_only=False)
    native=(payload.get('ema') or payload['model']).float().eval()
    assert native.model[-1].nc==2 and tuple(native.model[-1].f)==model.feature_indices
    trunk=torch.nn.ModuleList(list(native.model[:-1]))
    model.trunks['defect'].load_state_dict(trunk.state_dict(),strict=True)
    model.heads['defect'].load_state_dict(native.model[-1].state_dict(),strict=True)
    assert tensor_sha(model.trunks['defect'].state_dict())==tensor_sha(trunk.state_dict())
    assert tensor_sha(model.heads['defect'].state_dict())==tensor_sha(native.model[-1].state_dict())
    del native,trunk,payload
    model.eval().requires_grad_(False).to('cuda:0')
    report={'phase':'running','no_training':True,'stage_source_sha256':sha(stage_cp),'defect_source_sha256':sha(defect_cp),'validation':{}}
    atomic_json(out/'status.json',{'phase':'validating'})
    for task in ('stage','defect'):
        r=evaluate_task(model,task,output_dir=out/('validation_'+task),device='cuda:0')
        report['validation'][task]={'metrics':r['metrics'],'per_class':r['per_class'],'seen':r['seen']}
    adapter=FrozenDeploymentAdapter(model);paths=[]
    for task in ('stage','defect'):
        images=source_spec(task)['images'];paths.extend(images[round(i*(len(images)-1)/19)] for i in range(20))
    arrays=[cv2.imread(path) for path in paths];times=[]
    for i in range(10):adapter.predict(arrays[i],cpu=True)
    for _ in range(4):
        for im in arrays:
            torch.cuda.synchronize();start=time.perf_counter();pred=adapter.predict(im,cpu=True);torch.cuda.synchronize()
            times.append((time.perf_counter()-start)*1000);del pred
    report['speed']={'summary':summary(times),'observations_ms':times,'images':paths,'conf':.25,'max_det':300,'precision':'FP32','scope':'both heads array-to-CPU; no camera/depth/robot'}
    assert tensor_sha({k:v for k,v in model.state_dict().items() if k.startswith(('trunks.stage.','heads.stage.'))})==original_stage
    cp=out/'composite_frozen.pt'
    torch.save({'schema':'retained_task_composite_v1','model':model.cpu().state_dict(),'source_receipt':report,'exact_resume':False},cp)
    report.update(phase='complete',checkpoint=str(cp),checkpoint_sha256=sha(cp),stage_state_unchanged=True)
    atomic_json(out/'results.json',report);atomic_json(out/'status.json',{'phase':'complete'})

if __name__=='__main__':main()
