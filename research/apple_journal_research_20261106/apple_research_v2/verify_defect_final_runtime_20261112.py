"""Recheck fixed operating point under the sustained benchmark's runtime flags."""
import sys,json,gc
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'apple_research_v2/joint'));BASE=ROOT/'experiments/defect_algorithm_steps_20261112_v2'
from joint_state_v1 import configure,atomic_json,sha
def main():
    out=BASE/'runtime_fixed_verification'
    if out.exists():raise ValueError('Fresh output only')
    configure(out,'cuda:0')
    import torch,cv2
    from joint_model_v1 import build_model
    from defect_algorithm_modes_20261112 import AlgorithmAdapter
    from run_defect_zero_train_20261112 import metrics
    torch.set_num_threads(2);torch.backends.cudnn.benchmark=False
    review=json.loads((BASE/'final_review/results.json').read_text(encoding='utf8'));modes={'baseline':review['baseline']}
    if review['review_candidate']:modes['candidate']=review['review_candidate']
    samples=[s for s in json.loads((ROOT/'apple_research_v2/datasets/ssda_date_v2/manifest.json').read_text(encoding='utf8'))['samples'] if s['partition']=='dev'];results={}
    for k,c in modes.items():
        payload=torch.load(c['checkpoint'],map_location='cpu',weights_only=False);model=build_model('separate',device='cpu');model.load_state_dict(payload['model']);del payload;worker=AlgorithmAdapter(model.eval().requires_grad_(False).to('cuda:0'));rows=[]
        for s in samples:
            im=cv2.imread(s['rgb_path']);p=worker.predict_mode(im,c['mode'],c['thresholds'],both=False)['defect'][0].boxes.data.tolist();gt=[]
            for g in s['instances']:
                x,y,w,h=g['xywh_normalized'];gt.append({'class':g['class_id'],'box':[(x-w/2)*1024,(y-h/2)*1024,(x+w/2)*1024,(y+h/2)*1024]})
            rows.append({'image_id':s['image_id'],'GT':gt,'pred':p})
        m=metrics(rows);old=c['fixed'];results[k]={'fixed':m,'counts_equal':m['total']==old['total'] and m['class']==old['class'] and m['large']==old['large'],'missed_GT_equal':m['missed_GT']==old['missed_GT'],'checkpoint_sha256':sha(c['checkpoint'])};atomic_json(out/f'{k}_predictions.json',rows);del worker,model;gc.collect();torch.cuda.empty_cache()
    atomic_json(out/'results.json',{'modes':results,'all_counts_and_GT_equal':all(r['counts_equal'] and r['missed_GT_equal'] for r in results.values()),'runtime':{'torch':torch.__version__,'cudnn_deterministic':torch.backends.cudnn.deterministic,'cudnn_benchmark':torch.backends.cudnn.benchmark,'float32_matmul_precision':torch.get_float32_matmul_precision(),'input_weights_dtype':'float32'},'scope':'training-arm short benchmarks used deterministic=True; final paired benchmark and this full fixed-point check use fresh-process defaults with benchmark=False'})
if __name__=='__main__':main()
