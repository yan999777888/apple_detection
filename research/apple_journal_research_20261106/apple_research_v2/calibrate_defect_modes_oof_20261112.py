"""Calibrate all retained head/view modes on genuine OOF only; common FP/P budget.
Run after OOF checkpoints exist and no other GPU experiment is active.
"""
import sys,json,gc,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'apple_research_v2/joint'))
from joint_state_v1 import configure,atomic_json,sha
BASE=ROOT/'experiments/defect_algorithm_steps_20261112_v2'
def main():
    out=BASE/'all_mode_OOF_calibration'
    if out.exists():raise ValueError('Fresh output only')
    configure(out,'cuda:0')
    protocol={'modes':['o2o_nms','o2m_nms','heads_wbf_nms','o2o_wbf','flip_nms','flip_wbf'],'class_threshold_grid':[round(i/100,2) for i in range(1,91)],'common_budget':'For each class: OOF FP <= original o2o NMS .25 FP, and P>= original o2o .25 P. Maximize TP, then minimize FP, then nearest .25. If none feasible, mark rejected and use .25 for disclosure only.','selection_source':'only 596 true grouped OOF images, fixed last40 COCO-init folds','dev_used_to_choose_thresholds':False,'freeze_before_dev_evaluation':True}
    atomic_json(out/'protocol.json',protocol)
    import torch,numpy as np,cv2
    from joint_model_v1 import build_model,tensor_sha
    from defect_algorithm_modes_20261112 import AlgorithmAdapter,MODES,combine
    from run_defect_zero_train_20261112 import metrics
    from audit_defect_causes_20261112 import match
    from joint_eval_v1 import source_spec
    from benchmark_joint_20261112 import summary
    torch.set_num_threads(2);torch.backends.cudnn.benchmark=False
    manifest=json.loads((ROOT/'apple_research_v2/datasets/ssda_date_v2/manifest.json').read_text(encoding='utf8'))['samples'];lookup={s['image_id']:s for s in manifest};allrows={m:[] for m in MODES}
    folds=json.loads((BASE/'oof_v2/protocol.json').read_text(encoding='utf8'))['fold_images']
    for fi,ids in enumerate(folds):
        atomic_json(out/'status.json',{'phase':'OOF_heads_inference','fold':fi})
        cp=ROOT/f'runs/defect_oof3_fold{fi}_e40_20261112_v1/weights/last.pt';payload=torch.load(cp,map_location='cpu',weights_only=False);native=(payload.get('ema') or payload['model']).float();model=build_model('separate',device='cpu');model.trunks['defect'].load_state_dict(torch.nn.ModuleList(list(native.model[:-1])).state_dict());model.heads['defect'].load_state_dict(native.model[-1].state_dict());del native,payload
        worker=AlgorithmAdapter(model.eval().requires_grad_(False).to('cuda:0'));fold=[]
        for sid in ids:
            s=lookup[sid];assert s['partition']=='train';im=cv2.imread(s['rgb_path']);t=worker.predictors['stage'].preprocess([im]);src=worker.sources(t,im.shape[:2],('o2o','o2m'),True);gt=[]
            for g in s['instances']:
                x,y,w,h=g['xywh_normalized'];gt.append({'class':g['class_id'],'box':[(x-w/2)*1024,(y-h/2)*1024,(x+w/2)*1024,(y+h/2)*1024]})
            for mode in MODES:allrows[mode].append({'image_id':sid,'GT':gt,'pred':combine(src,mode,(.001,.001)).tolist()})
            fold.append({'image_id':sid,'GT':gt,'sources':{k:v.tolist() for k,v in src.items()}})
        atomic_json(out/f'fold{fi}_sources.json',{'checkpoint_sha256':sha(cp),'rows':fold});del worker,model;gc.collect();torch.cuda.empty_cache()
    base=metrics([{**r,'pred':[p for p in r['pred'] if p[4]>=.25]} for r in allrows['o2o_nms']]);selected={}
    for mode,rows in allrows.items():
        assert len(rows)==len({r['image_id'] for r in rows})==596
        # Greedy NMS/matching high-score prefixes preserve per-prediction TP flags.
        flags={0:[],1:[]};gtcount={0:0,1:0}
        for r in rows:
            mp,_=match(r['pred'],r['GT'])
            for i,p in enumerate(r['pred']):flags[int(p[5])].append((p[4],int(i in mp)))
            for g in r['GT']:gtcount[g['class']]+=1
        ts=[];curves={};valid=True
        for c in (0,1):
            curve=[]
            for t in protocol['class_threshold_grid']:
                kept=[tp for score,tp in flags[c] if score>=t];tp=sum(kept);fp=len(kept)-tp;curve.append({'threshold':t,'TP':tp,'FP':fp,'FN':gtcount[c]-tp,'P':tp/max(tp+fp,1),'R':tp/gtcount[c]})
            b=base['class'][str(c)];feasible=[r for r in curve if r['FP']<=b['FP'] and r['P']>=b['P']]
            if feasible:chosen=max(feasible,key=lambda r:(r['TP'],-r['FP'],-abs(r['threshold']-.25)));ts.append(chosen['threshold'])
            else:valid=False;ts.append(.25)
            curves[str(c)]=curve
        selected[mode]={'thresholds':ts,'feasible':valid,'curves':curves,'OOF_selected':metrics([{**r,'pred':[p for p in r['pred'] if p[4]>=ts[int(p[5])]]} for r in rows])}
    atomic_json(out/'thresholds_frozen.json',{'baseline_OOF':base,'modes':selected,'protocol':protocol})
    # No dev labels used in fitting above thresholds (manifest also lists dev paths).
    cp=ROOT/'experiments/strong_defect_composite_20261112_v1/composite_frozen.pt';payload=torch.load(cp,map_location='cpu',weights_only=False);model=build_model('separate',device='cpu');model.load_state_dict(payload['model']);del payload;worker=AlgorithmAdapter(model.eval().requires_grad_(False).to('cuda:0'));before=tensor_sha(model.state_dict());results={}
    dev=[s for s in manifest if s['partition']=='dev'];images=[]
    for task in ('stage','defect'):
        pp=source_spec(task)['images'];images.extend(cv2.imread(pp[round(i*(len(pp)-1)/19)]) for i in range(20))
    baseline=json.loads((BASE/'phase1/validation.json').read_text(encoding='utf8'))['o2o_nms']
    for mode in MODES:
        ts=selected[mode]['thresholds'];rows=[];atomic_json(out/'status.json',{'phase':'dev_actual_deployment','mode':mode,'frozen_thresholds':ts})
        for s in dev:
            im=cv2.imread(s['rgb_path']);pred=worker.predict_mode(im,mode,ts,both=False)['defect'][0].boxes.data.tolist();gt=[]
            for g in s['instances']:
                x,y,w,h=g['xywh_normalized'];gt.append({'class':g['class_id'],'box':[(x-w/2)*1024,(y-h/2)*1024,(x+w/2)*1024,(y+h/2)*1024]})
            rows.append({'image_id':s['image_id'],'GT':gt,'pred':pred})
        m=metrics(rows);m['new_FN_vs_baseline']=sorted(set(m['missed_GT'])-set(baseline['missed_GT']));m['recovered_FN']=sorted(set(baseline['missed_GT'])-set(m['missed_GT']))
        for im in images[:5]:worker.predict_mode(im,mode,ts)
        times=[]
        for _ in range(3):
            for im in images:
                torch.cuda.synchronize();start=time.perf_counter();p=worker.predict_mode(im,mode,ts);torch.cuda.synchronize();times.append((time.perf_counter()-start)*1000);del p
        results[mode]={'thresholds':ts,'OOF_feasible':selected[mode]['feasible'],'fixed':m,'speed':summary(times),'speed_ms':times};atomic_json(out/f'{mode}_predictions.json',rows);atomic_json(out/'results_partial.json',results)
    assert tensor_sha(model.state_dict())==before
    atomic_json(out/'results.json',{'phase':'complete','modes':results,'checkpoint_sha256':sha(cp),'no_training_of_strong_model':True});atomic_json(out/'status.json',{'phase':'complete'})
if __name__=='__main__':main()
