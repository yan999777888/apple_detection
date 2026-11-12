"""Validate each mode on its ACTUAL deployment batch shape; cache parity is not assumed."""
import sys,json,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'apple_research_v2/joint'))
from joint_state_v1 import configure,atomic_json,sha
def main():
    out=ROOT/'experiments/defect_algorithm_steps_20261112_v2/mode_deployment_validation'
    if out.exists():raise ValueError('Fresh output only')
    configure(out,'cuda:0')
    import torch,cv2
    from joint_model_v1 import build_model,tensor_sha
    from defect_algorithm_modes_20261112 import AlgorithmAdapter,MODES
    from run_defect_zero_train_20261112 import metrics
    torch.set_num_threads(2);torch.backends.cudnn.benchmark=False
    cp=ROOT/'experiments/strong_defect_composite_20261112_v1/composite_frozen.pt';payload=torch.load(cp,map_location='cpu',weights_only=False);model=build_model('separate',device='cpu');model.load_state_dict(payload['model']);del payload
    worker=AlgorithmAdapter(model.eval().requires_grad_(False).to('cuda:0'));before=tensor_sha(model.state_dict())
    manifest=json.loads((ROOT/'apple_research_v2/datasets/ssda_date_v2/manifest.json').read_text(encoding='utf8'));lookup={r['image_id']:r for r in manifest['samples']}
    source=json.loads((out.parent/'phase1/source_predictions.json').read_text(encoding='utf8'));old=json.loads((out.parent/'phase1/validation.json').read_text(encoding='utf8'));result={}
    for mode in MODES:
        rows=[]
        for r in source:
            im=cv2.imread(lookup[r['image_id']]['rgb_path']);p=worker.predict_mode(im,mode,both=False)['defect'][0].boxes.data.tolist();rows.append({'image_id':r['image_id'],'GT':r['GT'],'pred':p})
        m=metrics(rows);m['aggregate_matches_batched_cache']=m['total']==old[mode]['total'];m['missed_GT_matches_batched_cache']=m['missed_GT']==old[mode]['missed_GT'];result[mode]=m
        atomic_json(out/f'{mode}_predictions.json',rows);atomic_json(out/'status.json',{'phase':'validation','completed_mode':mode})
    baseline=result['o2o_nms'];assert [baseline['total'][k] for k in ('TP','FP','FN')]==[179,50,51]
    for mode,m in result.items():m['new_FN_vs_baseline']=sorted(set(m['missed_GT'])-set(baseline['missed_GT']))
    assert tensor_sha(model.state_dict())==before
    atomic_json(out/'results.json',{'phase':'complete','modes':result,'checkpoint_sha256':sha(cp),'actual_deployment_shapes':True});atomic_json(out/'status.json',{'phase':'complete'})
if __name__=='__main__':main()
