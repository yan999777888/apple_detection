"""Read-only cause audit of retained weights on frozen SSDA train/development.

No training, relabeling, threshold search or checkpoint selection. NMS=.7 was
declared in the existing Nov 6 audit; here it is independently rechecked on the
actual FP32 composite. Train metrics are resubstitution diagnostics only.
"""
import argparse,sys,json,time,collections,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'apple_research_v2/joint'))
from joint_state_v1 import configure,sha,atomic_json

def ious(a,b):
    import numpy as np
    a=np.asarray(a,float).reshape(-1,4);b=np.asarray(b,float).reshape(-1,4)
    inter=np.maximum(np.minimum(a[:,None,2:],b[None,:,2:])-np.maximum(a[:,None,:2],b[None,:,:2]),0).prod(2)
    return inter/np.maximum((a[:,2:]-a[:,:2]).prod(1)[:,None]+(b[:,2:]-b[:,:2]).prod(1)[None,:]-inter,1e-12)

def match(pred,gt,ignore_class=False):
    m=ious([p[:4] for p in pred],[g['box'] for g in gt]);used=set();pairs={}
    for pi in sorted(range(len(pred)),key=lambda i:(-pred[i][4],i)):
        cand=[j for j,g in enumerate(gt) if j not in used and (ignore_class or g['class']==int(pred[pi][5]))]
        if cand:
            gi=max(cand,key=lambda j:m[pi,j])
            if m[pi,gi]>=.5:pairs[pi]=gi;used.add(gi)
    return pairs,m

def nms_indices(pred):
    import torch
    from torchvision.ops import batched_nms
    if not pred:return []
    x=torch.tensor(pred,dtype=torch.float32)
    return batched_nms(x[:,:4],x[:,4],x[:,5],.7).tolist()

def geometry(box,hw):
    h,w=hw;side=math.sqrt((box[2]-box[0])*(box[3]-box[1]))*640/max(h,w)
    return {'size640':side,'size_bin':'<16' if side<16 else '16-32' if side<32 else '32-96' if side<96 else '>=96'}

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise ValueError('Fresh output only')
    out=configure(a.output,'cuda:0')
    import torch,cv2,numpy as np
    from joint_model_v1 import build_model,tensor_sha
    from deployment_adapter_v1 import FrozenDeploymentAdapter
    from joint_eval_v1 import source_spec
    from benchmark_joint_20261112 import summary
    torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.backends.cudnn.benchmark=False
    cp=ROOT/'experiments/strong_defect_composite_20261112_v1/composite_frozen.pt'
    assert sha(cp)=='f8becdeff0e042fa19d681141aa56e133393130b4a2af23133706fd7766367f6'
    manifest=ROOT/'apple_research_v2/datasets/ssda_date_v2/manifest.json'
    samples=json.loads(manifest.read_text(encoding='utf8'))['samples']
    protocol={'no_training':True,'no_GT_edits':True,'checkpoint_sha256':sha(cp),'manifest_sha256':sha(manifest),'export_conf':.001,'deployment_conf':.25,'imgsz':640,'rect':False,'precision':'FP32','match_iou':.5,'NMS_iou':.7,'NMS_chosen_from_prior_audit_no_scan':True,'train_role':'seen-training-set resubstitution, NOT independent validation','hypotheses':['duplicate predictions','score suppression','class confusion','extent mismatch','train/dev source or appearance shift','annotation incompleteness requires visual review'],'script_sha256':sha(__file__)}
    atomic_json(out/'protocol.json',protocol)
    model=build_model('separate',device='cpu');state=torch.load(cp,map_location='cpu',weights_only=False)
    model.load_state_dict(state['model'],strict=True);del state
    model.eval().requires_grad_(False).to('cuda:0');before=tensor_sha(model.state_dict())
    low=FrozenDeploymentAdapter(model,conf=.001);records=[];gtrows=[];fprows=[]
    (out/'predictions').mkdir()
    for ix,s in enumerate(x for x in samples if x['partition'] in ('train','dev')):
        assert sha(s['rgb_path'])==s['rgb_sha256'] and sha(s['label_path'])==s['label_sha256']
        im=cv2.imread(s['rgb_path']);h,w=im.shape[:2];gt=[]
        for g in s['instances']:
            x,y,bw,bh=g['xywh_normalized'];gt.append({'class':g['class_id'],'box':[(x-bw/2)*w,(y-bh/2)*h,(x+bw/2)*w,(y+bh/2)*h]})
        pred=low.predict(im,tasks=('defect',),cpu=True)['defect'][0].boxes.data.tolist()
        high=[q for q in pred if q[4]>=.25];kept=nms_indices(high);after=[high[i] for i in kept]
        mp,m=match(high,gt);mn,mnmat=match(after,gt);ml,mlmat=match(pred,gt);mi,_=match(high,gt,True)
        common={'image_id':s['image_id'],'image_path':s['rgb_path'],'partition':s['partition'],'date':s['capture_filename_tokens']['date'],'camera':s['capture_filename_tokens']['camera']}
        row={**common,'GT':gt,'low_predictions':pred,'high_predictions':high,'NMS_keep':kept,'TP':len(mp),'FP':len(high)-len(mp),'FN':len(gt)-len(mp),'NMS_TP':len(mn),'NMS_FP':len(after)-len(mn),'NMS_FN':len(gt)-len(mn),'low_conf_TP':len(ml),'low_conf_FP':len(pred)-len(ml),'class_agnostic_TP':len(mi),'new_NMS_FN':sorted(set(mp.values())-set(mn.values())),'matched_GT':sorted(mp.values())}
        atomic_json(out/'predictions'/f"{Path(s['image_id']).stem}.json",row)
        records.append({k:v for k,v in row.items() if k not in ('low_predictions','high_predictions','GT')})
        for gi,g in enumerate(gt):
            same=[j for j,q in enumerate(high) if int(q[5])==g['class']];other=[j for j,q in enumerate(high) if int(q[5])!=g['class']]
            sv=max((m[j,gi] for j in same),default=0);ov=max((m[j,gi] for j in other),default=0)
            lowgood=[q[4] for j,q in enumerate(pred) if int(q[5])==g['class'] and mlmat[j,gi]>=.5]
            lowmax=max(lowgood,default=0)
            cat='TP' if gi in mp.values() else 'same_class_competition' if sv>=.5 else 'wrong_class_high_overlap' if ov>=.5 else 'low_score_correct_box' if lowmax>0 else 'localization_IoU_0.1_to_0.5' if max(sv,ov)>=.1 else 'no_high_near_or_low_correct_box'
            gtrows.append({**common,'GT_index':gi,'class':g['class'],'box':g['box'],'category':cat,'TP':gi in mp.values(),'best_high_same_IoU':sv,'best_high_other_IoU':ov,'max_score_correct_box':lowmax,**geometry(g['box'],[h,w])})
        for pi,q in enumerate(high):
            if pi in mp:continue
            same=[j for j,g in enumerate(gt) if g['class']==int(q[5])];other=[j for j,g in enumerate(gt) if g['class']!=int(q[5])]
            sv=max((m[pi,j] for j in same),default=0);ov=max((m[pi,j] for j in other),default=0)
            cat='duplicate_same_class' if sv>=.5 else 'wrong_class_high_overlap' if ov>=.5 else 'localization_IoU_0.1_to_0.5' if max(sv,ov)>=.1 else 'no_near_annotated_GT'
            fprows.append({**common,'prediction_index':pi,'class':int(q[5]),'box':q[:4],'score':q[4],'category':cat,'survives_NMS':pi in kept,'best_same_IoU':sv,'best_other_IoU':ov,**geometry(q[:4],[h,w])})
        if ix%25==0:atomic_json(out/'status.json',{'phase':'inference','images_completed':ix+1,'total':709})
    summaries={}
    for part in ('train','dev'):
        rr=[r for r in records if r['partition']==part];gg=[r for r in gtrows if r['partition']==part];ff=[r for r in fprows if r['partition']==part]
        counts={k:sum(r[k] for r in rr) for k in ('TP','FP','FN','NMS_TP','NMS_FP','NMS_FN','low_conf_TP','low_conf_FP','class_agnostic_TP')}
        counts.update(P=counts['TP']/(counts['TP']+counts['FP']),R=counts['TP']/len(gg),NMS_P=counts['NMS_TP']/(counts['NMS_TP']+counts['NMS_FP']),NMS_R=counts['NMS_TP']/len(gg))
        strata={}
        for field in ('class','size_bin','date','camera'):
            strata[field]={str(val):{'n':len(gp:=[g for g in gg if g[field]==val]),'TP':sum(g['TP'] for g in gp),'FN':sum(not g['TP'] for g in gp),'FP':sum(g[field]==val for g in ff) if field!='size_bin' else None} for val in sorted({g[field] for g in gg})}
        summaries[part]={'images':len(rr),'GT':len(gg),'counts':counts,'FN_categories':dict(collections.Counter(g['category'] for g in gg if not g['TP'])),'FP_categories':dict(collections.Counter(g['category'] for g in ff)),'post_NMS_FP_categories':dict(collections.Counter(g['category'] for g in ff if g['survives_NMS'])),'new_NMS_FN':sum(len(r['new_NMS_FN']) for r in rr),'strata':strata}
    assert (summaries['dev']['counts']['TP'],summaries['dev']['counts']['FP'],summaries['dev']['counts']['FN'])==(179,98,51)
    assert len(records)==709 and len(gtrows)==1506
    atomic_json(out/'image_records.json',records);atomic_json(out/'GT_records.json',gtrows);atomic_json(out/'FP_records.json',fprows)
    atomic_json(out/'summary.json',{'phase':'inference_complete','protocol':protocol,'splits':summaries})
    worker=FrozenDeploymentAdapter(model,conf=.25);paths=[]
    for task in ('stage','defect'):
        images=source_spec(task)['images'];paths.extend(images[round(i*(len(images)-1)/19)] for i in range(20))
    arrays=[cv2.imread(p) for p in paths];timings={'baseline':[],'fixed_NMS07':[]}
    for i in range(10):worker.predict(arrays[i],cpu=True)
    for order in [('baseline','fixed_NMS07'),('fixed_NMS07','baseline')]*2:
        for mode in order:
            for im in arrays:
                torch.cuda.synchronize();start=time.perf_counter();pred=worker.predict(im,cpu=True)
                if mode=='fixed_NMS07':
                    r=pred['defect'][0];keep=nms_indices(r.boxes.data.tolist());pred['defect'][0]=r[keep]
                torch.cuda.synchronize();timings[mode].append((time.perf_counter()-start)*1000);del pred
    assert tensor_sha(model.state_dict())==before
    result={'phase':'complete','protocol':protocol,'splits':summaries,'speed':{k:summary(v) for k,v in timings.items()},'observations_ms':timings,'speed_protocol':{'images':paths,'warmup':10,'rounds':4,'scope':'both tasks BGR array to CPU results plus optional CPU class-aware NMS .7','excluded':'disk load, model init, camera, depth, robot'},'weights_unchanged':True}
    atomic_json(out/'summary.json',result);atomic_json(out/'status.json',{'phase':'complete'})

if __name__=='__main__':main()
