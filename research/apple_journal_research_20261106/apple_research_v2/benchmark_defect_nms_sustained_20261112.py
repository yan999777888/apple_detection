"""1200-frame serial replay to investigate the marginal 50ms P95 gate.
Same 40 frozen images repeated 30 times; not camera or independent video data.
No weights, labels, scores, thresholds or default deployment entry are changed.
"""
import sys,time,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'apple_research_v2/joint'))
from joint_state_v1 import configure,sha,atomic_json

def main():
    out=ROOT/'experiments/defect_cause_audit_20261112_v1/sustained_speed'
    if out.exists():raise ValueError('Fresh output only')
    configure(out,'cuda:0')
    import torch,cv2
    from joint_model_v1 import build_model,tensor_sha
    from deployment_adapter_v1 import FrozenDeploymentAdapter
    from benchmark_joint_20261112 import summary
    from audit_defect_causes_20261112 import nms_indices
    torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.backends.cudnn.benchmark=False
    source=json.loads((out.parent/'summary.json').read_text(encoding='utf8'))
    cp=ROOT/'experiments/strong_defect_composite_20261112_v1/composite_frozen.pt'
    assert sha(cp)==source['protocol']['checkpoint_sha256']
    state=torch.load(cp,map_location='cpu',weights_only=False);model=build_model('separate',device='cpu')
    model.load_state_dict(state['model'],strict=True);del state
    worker=FrozenDeploymentAdapter(model.eval().requires_grad_(False).to('cuda:0'));before=tensor_sha(model.state_dict())
    paths=source['speed_protocol']['images'];images=[cv2.imread(p) for p in paths]
    assert len(images)==40 and all(im is not None for im in images)
    def infer(im):
        result=worker.predict(im,cpu=True);r=result['defect'][0];result['defect'][0]=r[nms_indices(r.boxes.data.tolist())];return result
    for i in range(20):infer(images[i%40])
    times=[];cycles=[]
    for cycle in range(30):
        for im in images:
            torch.cuda.synchronize();start=time.perf_counter();pred=infer(im);torch.cuda.synchronize();times.append((time.perf_counter()-start)*1000);del pred
        cycles.append(summary(times[-40:]));atomic_json(out/'status.json',{'phase':'running','frames':len(times),'total':1200})
    assert tensor_sha(model.state_dict())==before
    r={'phase':'complete','scope':'1200 serial calls, same 40 existing source images repeated 30 times, both heads native CPU output+NMS .7; not real camera video','source_summary_sha256':sha(out.parent/'summary.json'),'warmup':20,'precision':'FP32','imgsz':640,'conf':.25,'maxdet':300,'summary':summary(times),'worst_40_frame_cycle_fps':min(c['serial_fps'] for c in cycles),'over50ms_frames':sum(t>50 for t in times),'cycles':cycles,'observations_ms':times,'weights_unchanged':True}
    atomic_json(out/'results.json',r);atomic_json(out/'status.json',{'phase':'complete'})

if __name__=='__main__':main()
