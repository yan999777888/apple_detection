"""Full-development fixed-policy head/WBF/TTA ablation; phase 1, no training."""
import sys,json,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'apple_research_v2/joint'))
from joint_state_v1 import configure,sha,atomic_json

def metrics(records):
    from audit_defect_causes_20261112 import match,geometry
    totals={'TP':0,'FP':0,'FN':0};groups={str(c):{'GT':0,'TP':0,'FP':0} for c in (0,1)};large={'GT':0,'TP':0};miss=[]
    for r in records:
        p=r['pred'];g=r['GT'];mp,_=match(p,g);seen=set(mp.values())
        totals['TP']+=len(mp);totals['FP']+=len(p)-len(mp);totals['FN']+=len(g)-len(mp)
        for gi,t in enumerate(g):
            c=str(t['class']);groups[c]['GT']+=1;groups[c]['TP']+=gi in seen
            if geometry(t['box'],[1024,1024])['size_bin']=='>=96':large['GT']+=1;large['TP']+=gi in seen
            if gi not in seen:miss.append(f"{r['image_id']}#{gi}")
        for pi,t in enumerate(p):
            if pi not in mp:groups[str(int(t[5]))]['FP']+=1
    totals.update(P=totals['TP']/max(totals['TP']+totals['FP'],1),R=totals['TP']/max(totals['TP']+totals['FN'],1))
    for x in groups.values():x.update(R=x['TP']/max(x['GT'],1),P=x['TP']/max(x['TP']+x['FP'],1))
    large['R']=large['TP']/max(large['GT'],1)
    return {'total':totals,'class':groups,'large':large,'missed_GT':miss}

def main():
    out=ROOT/'experiments/defect_algorithm_steps_20261112_v2/phase1'
    if out.exists():raise ValueError('Fresh output only')
    configure(out,'cuda:0')
    import cv2,torch
    from joint_model_v1 import build_model,tensor_sha
    from joint_eval_v1 import source_spec
    from defect_algorithm_modes_20261112 import AlgorithmAdapter,MODES,combine
    from benchmark_joint_20261112 import summary
    torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.backends.cudnn.benchmark=False
    cp=ROOT/'experiments/strong_defect_composite_20261112_v1/composite_frozen.pt';assert sha(cp)=='f8becdeff0e042fa19d681141aa56e133393130b4a2af23133706fd7766367f6'
    state=torch.load(cp,map_location='cpu',weights_only=False);model=build_model('separate',device='cpu');model.load_state_dict(state['model']);del state
    worker=AlgorithmAdapter(model.eval().requires_grad_(False).to('cuda:0'));before=tensor_sha(model.state_dict())
    atomic_json(out/'protocol.json',{'checkpoint_sha256':sha(cp),'modes':MODES,'conf':.25,'source_floor':.001,'NMS':.7,'WBF_iou':.55,'WBF_scores':'mean member scores times unique source support/source count','imgsz':640,'precision':'FP32','GT_unchanged':True,'OOF_thresholds_pending':True,'TTA':'defect-only batch2 original/hflip, stage single view','trained_head_parameters':{k:sum(p.numel() for p in m.parameters()) for k,m in [('one2many_cv2',model.heads['defect'].cv2),('one2many_cv3',model.heads['defect'].cv3)]}})
    samples=[r for r in json.loads((ROOT/'apple_research_v2/datasets/ssda_date_v2/manifest.json').read_text(encoding='utf8'))['samples'] if r['partition']=='dev'];rows={m:[] for m in MODES};sources_all=[]
    for i,s in enumerate(samples):
        im=cv2.imread(s['rgb_path']);tensor=worker.predictors['stage'].preprocess([im]);src=worker.sources(tensor,im.shape[:2],('o2o','o2m'),True)
        gt=[]
        for g in s['instances']:
            x,y,w,h=g['xywh_normalized'];gt.append({'class':g['class_id'],'box':[(x-w/2)*1024,(y-h/2)*1024,(x+w/2)*1024,(y+h/2)*1024]})
        sources_all.append({'image_id':s['image_id'],'GT':gt,'sources':{k:v.tolist() for k,v in src.items()}})
        for mode in MODES:rows[mode].append({'image_id':s['image_id'],'GT':gt,'pred':combine(src,mode).tolist()})
        atomic_json(out/'status.json',{'phase':'validation','images':i+1,'total':113})
    results={m:metrics(r) for m,r in rows.items()};assert [results['o2o_nms']['total'][k] for k in ('TP','FP','FN')]==[179,50,51]
    for m in MODES:results[m]['new_FN_vs_baseline']=sorted(set(results[m]['missed_GT'])-set(results['o2o_nms']['missed_GT']))
    atomic_json(out/'validation.json',results);atomic_json(out/'source_predictions.json',sources_all)
    paths=[]
    for task in ('stage','defect'):
        p=source_spec(task)['images'];paths.extend(p[round(i*(len(p)-1)/19)] for i in range(20))
    images=[cv2.imread(p) for p in paths]
    for mode in MODES:
        atomic_json(out/'status.json',{'phase':'benchmark','mode':mode})
        for i in range(5):worker.predict_mode(images[i],mode)
        times=[]
        for _ in range(3):
            for im in images:
                torch.cuda.synchronize();start=time.perf_counter();pred=worker.predict_mode(im,mode);torch.cuda.synchronize();times.append((time.perf_counter()-start)*1000);del pred
        results[mode]['speed']=summary(times);results[mode]['observations_ms']=times
        atomic_json(out/'results_partial.json',results)
    assert tensor_sha(model.state_dict())==before
    atomic_json(out/'results.json',{'phase':'complete','modes':results,'speed_scope':'both tasks BGR-array to native CPU Results; no disk/camera/depth/robot','weights_unchanged':True});atomic_json(out/'status.json',{'phase':'complete'})

if __name__=='__main__':main()
