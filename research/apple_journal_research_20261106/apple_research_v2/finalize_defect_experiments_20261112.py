"""Audit evidence and sustained paired benchmark. Never overwrites default model/config."""
import sys,json,time,gc
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'apple_research_v2/joint'));BASE=ROOT/'experiments/defect_algorithm_steps_20261112_v2'
from joint_state_v1 import configure,atomic_json,sha
def read(p):return json.loads(p.read_text(encoding='utf8'))
def main():
    out=BASE/'final_review'
    if out.exists():raise ValueError('Fresh output only')
    configure(out,'cuda:0')
    import torch,cv2
    from joint_model_v1 import build_model,tensor_sha
    from defect_algorithm_modes_20261112 import AlgorithmAdapter
    from joint_eval_v1 import source_spec
    from run_defect_zero_train_20261112 import metrics
    from report_defect_algorithm_steps_20261112 import deployment_ap
    from benchmark_joint_20261112 import summary
    torch.set_num_threads(2);torch.backends.cudnn.benchmark=False
    original=ROOT/'experiments/strong_defect_composite_20261112_v1/composite_frozen.pt';baseline=read(BASE/'mode_deployment_validation/results.json')['modes']['o2o_nms'];aps=read(BASE/'deployment_ap.json')
    candidates=[]
    for name,r in read(BASE/'all_mode_OOF_calibration/results.json')['modes'].items():
        if r['OOF_feasible']:candidates.append({'name':'OOF_'+name,'checkpoint':str(original),'mode':name,'thresholds':r['thresholds'],'fixed':r['fixed'],'speed':r['speed'],'custom_AP':aps[name]})
    for name,r in read(BASE/'train/results.json')['arms'].items():candidates.append({'name':name,'checkpoint':r['checkpoint'],'mode':'o2o_nms','thresholds':[.25,.25],'fixed':r['fixed'],'speed':r['speed'],'custom_AP':aps[name]})
    b=baseline['total']
    for c in candidates:
        t=c['fixed']['total'];c['F1']=2*t['TP']/max(2*t['TP']+t['FP']+t['FN'],1)
        c['review_eligible']=(t['P']>=b['P'] and t['R']>=b['R'] and (t['P']>b['P'] or t['R']>b['R']) and c['fixed']['large']['TP']>=baseline['large']['TP'] and c['speed']['serial_fps']>=20 and c['custom_AP']['mAP50_95']>=aps['o2o_nms']['mAP50_95'])
    ranked=sorted([c for c in candidates if c['review_eligible']],key=lambda c:(-c['F1'],-c['fixed']['total']['P']))
    def load(c):
        state=torch.load(c['checkpoint'],map_location='cpu',weights_only=False);model=build_model('separate',device='cpu');model.load_state_dict(state['model']);del state
        return AlgorithmAdapter(model.eval().requires_grad_(False).to('cuda:0'))
    reference={'name':'retained_NMS_baseline','checkpoint':str(original),'mode':'o2o_nms','thresholds':[.25,.25],'fixed':baseline}
    worker=load(reference);stage0=tensor_sha({k:v for k,v in worker.model.state_dict().items() if k.startswith(('trunks.stage.','heads.stage.'))})
    manifest=read(ROOT/'apple_research_v2/datasets/ssda_date_v2/manifest.json')['samples'];dev=[s for s in manifest if s['partition']=='dev']
    def exact_ap(adapter,mode):
        rows=[]
        for s in dev:
            im=cv2.imread(s['rgb_path']);pred=adapter.predict_mode(im,mode,(.001,.001),both=False)['defect'][0].boxes.data.tolist();gt=[]
            for g in s['instances']:
                x,y,w,h=g['xywh_normalized'];gt.append({'class':g['class_id'],'box':[(x-w/2)*1024,(y-h/2)*1024,(x+w/2)*1024,(y+h/2)*1024]})
            rows.append({'image_id':s['image_id'],'GT':gt,'pred':pred})
        return deployment_ap(rows)
    baseline_ap=exact_ap(worker,'o2o_nms');chosen=None;other=None
    for c in ranked:
        atomic_json(out/'status.json',{'phase':'candidate_exact_AP','candidate':c['name']});adapter=load(c);ap=exact_ap(adapter,c['mode']);c['exact_deployment_AP']=ap
        assert tensor_sha({k:v for k,v in adapter.model.state_dict().items() if k.startswith(('trunks.stage.','heads.stage.'))})==stage0
        if ap['mAP50_95']>=baseline_ap['mAP50_95'] and ap['mAP50']>=baseline_ap['mAP50']:chosen=c;other=adapter;break
        del adapter;gc.collect();torch.cuda.empty_cache()
    paths=[]
    for task in ('stage','defect'):
        pp=source_spec(task)['images'];paths.extend(pp[round(i*(len(pp)-1)/19)] for i in range(20))
    ims=[cv2.imread(p) for p in paths];workers={'baseline':(worker,reference)}
    if chosen:workers['candidate']=(other,chosen)
    for ad,c in workers.values():
        for im in ims[:10]:ad.predict_mode(im,c['mode'],c['thresholds'])
    obs={k:[] for k in workers};cycles={k:[] for k in workers}
    for repeat in range(15):
        order=list(workers) if repeat%2==0 else list(reversed(workers))
        for k in order:
            ad,c=workers[k];one=[]
            for im in ims:
                torch.cuda.synchronize();start=time.perf_counter();p=ad.predict_mode(im,c['mode'],c['thresholds']);torch.cuda.synchronize();one.append((time.perf_counter()-start)*1000);del p
            obs[k].extend(one);cycles[k].append(sum(one)/len(one));atomic_json(out/'status.json',{'phase':'sustained_benchmark','cycle':repeat+1,'mode':k,'cycles':15})
    speed={k:{**summary(v),'slow_frames_over50ms':sum(t>50 for t in v),'worst_40_frame_cycle_fps':1000/max(cycles[k]),'observations_ms':v,'cycle_mean_ms':cycles[k]} for k,v in obs.items()}
    if chosen:chosen['sustained_speed']=speed['candidate'];chosen['speed_target_met']=speed['candidate']['serial_fps']>=20
    # All original source labels and images, including the development set, remain intact.
    for s in manifest:
        if s['partition'] in ('train','dev'):assert sha(s['label_path'])==s['label_sha256'] and sha(s['rgb_path'])==s['rgb_sha256']
    assert sha(original)=='f8becdeff0e042fa19d681141aa56e133393130b4a2af23133706fd7766367f6'
    result={'phase':'complete','baseline':reference,'baseline_exact_AP':baseline_ap,'candidates':candidates,'review_candidate':chosen,'speed':speed,'image_paths':paths,'scope':'40 real images repeated15 cycles per mode, both tasks array->CPU Results, no camera/depth/robot','original_GT_and_weights_unchanged':True,'deployment_default_replaced':False,'selection_caveat':'development-selected candidate, not independent test or multi-seed proof; no automatic promotion'}
    atomic_json(out/'results.json',result);atomic_json(out/'status.json',{'phase':'complete','candidate':chosen['name'] if chosen else None})
if __name__=='__main__':main()
