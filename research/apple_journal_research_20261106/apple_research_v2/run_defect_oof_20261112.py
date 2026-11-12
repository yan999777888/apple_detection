"""Three grouped folds, independent COCO initialization, fixed final40; never select on dev."""
import os,sys,json,subprocess,traceback,hashlib,gc
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'apple_research_v2/joint'))
from joint_state_v1 import configure,atomic_json,sha

def main():
    out=ROOT/'experiments/defect_algorithm_steps_20261112_v2/oof_v2'
    if out.exists():raise ValueError('Fresh output only')
    configure(out,'cuda:0')
    import yaml
    samples=[r for r in json.loads((ROOT/'apple_research_v2/datasets/ssda_date_v2/manifest.json').read_text(encoding='utf8'))['samples'] if r['partition']=='train']
    groups={}
    for r in samples:groups.setdefault(r['same_second_group'],[]).append(r)
    # Deterministic group-balanced three folds, no outcome/label stratification.
    folds=[[],[],[]]
    for k,rr in sorted(groups.items(),key=lambda kv:(-len(kv[1]),hashlib.sha256(kv[0].encode()).hexdigest())):
        folds[min(range(3),key=lambda f:(len(folds[f]),f))].extend(rr)
    weights=ROOT/'assets/ultralytics_yolo26_coco_20261106/yolo26n.pt'
    protocol={'seed':20261106,'folds':3,'epochs_per_fold':40,'init':str(weights),'init_sha256':sha(weights),'selection':'fixed last epoch, NOT best fold validation','group_key':'same_second_group (known capture-time group; physical fruit identity unverified)','threshold_rule':'per class maximize OOF TP subject to FP <= OOF FP at .25 AND Precision >= OOF Precision at .25; tie smaller FP then closest .25','grid':[round(i/100,2) for i in range(1,61)],'no_dev_in_training_or_threshold_choice':True,'fold_images':[[r['image_id'] for r in f] for f in folds]}
    atomic_json(out/'protocol.json',protocol)
    predictions=[]
    try:
        for fi,held in enumerate(folds):
            folder=out/f'fold{fi}';folder.mkdir();heldids={r['image_id'] for r in held};train=[r for r in samples if r['image_id'] not in heldids]
            assert not ({r['same_second_group'] for r in train}&{r['same_second_group'] for r in held})
            assert train and held
            assert len(train)+len(held)==596
            for key,rr in [('train',train),('val',held)]:
                (folder/f'{key}.txt').write_text('\n'.join(r['rgb_path'] for r in rr)+'\n',encoding='utf8')
            atomic_json(folder/'membership.json',{'train':[r['image_id'] for r in train],'heldout':[r['image_id'] for r in held]})
            data={'path':str(folder),'train':str(folder/'train.txt'),'val':str(folder/'val.txt'),'names':{0:'class_0',1:'class_1'}}
            (folder/'data.yaml').write_text(yaml.safe_dump(data,allow_unicode=True),encoding='utf8')
            run=f'defect_oof3_fold{fi}_e40_20261112_v1'
            args=[sys.executable,'-B',str(ROOT/'apple_research_v2/train_yolo_full.py'),'--task','detect','--data',str(folder/'data.yaml'),'--manifest',str(folder/'membership.json'),'--weights',str(weights),'--expected-weights-sha256',sha(weights),'--run-name',run,'--epochs','40']
            atomic_json(out/'status.json',{'phase':'training_fold','fold':fi,'epochs':40,'run':run})
            env=os.environ.copy();env['PYTHONHASHSEED']='20261106'
            with (folder/'training.log').open('w',encoding='utf8') as log:subprocess.run(args,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
            import torch,cv2
            from joint_model_v1 import build_model
            from defect_algorithm_modes_20261112 import AlgorithmAdapter,nms
            torch.set_num_threads(2)
            cp=ROOT/'runs'/run/'weights/last.pt';payload=torch.load(cp,map_location='cpu',weights_only=False);native=(payload.get('ema') or payload['model']).float()
            model=build_model('separate',device='cpu');model.trunks['defect'].load_state_dict(torch.nn.ModuleList(list(native.model[:-1])).state_dict());model.heads['defect'].load_state_dict(native.model[-1].state_dict());del native,payload
            worker=AlgorithmAdapter(model.eval().requires_grad_(False).to('cuda:0'));foldpred=[]
            atomic_json(out/'status.json',{'phase':'predict_heldout','fold':fi})
            for s in held:
                im=cv2.imread(s['rgb_path']);tensor=worker.predictors['stage'].preprocess([im]);p=nms(worker.sources(tensor,im.shape[:2])['o2o']);gt=[]
                for g in s['instances']:
                    x,y,w,h=g['xywh_normalized'];gt.append({'class':g['class_id'],'box':[(x-w/2)*1024,(y-h/2)*1024,(x+w/2)*1024,(y+h/2)*1024]})
                foldpred.append({'image_id':s['image_id'],'fold':fi,'GT':gt,'pred':p.tolist()})
            atomic_json(folder/'predictions.json',{'checkpoint_sha256':sha(cp),'rows':foldpred});predictions.extend(foldpred)
            del worker,model;gc.collect();torch.cuda.empty_cache()
        from run_defect_zero_train_20261112 import metrics
        def filtered(rows,ts):return [{**r,'pred':[p for p in r['pred'] if p[4]>=ts[int(p[5])]]} for r in rows]
        assert len(predictions)==len({r['image_id'] for r in predictions})==596
        baseline=metrics(filtered(predictions,[.25,.25]));thresholds=[];curves={}
        for c in (0,1):
            curve=[];b=baseline['class'][str(c)]
            for t in protocol['grid']:
                ts=[.25,.25];ts[c]=t;m=metrics(filtered(predictions,ts))['class'][str(c)];curve.append({'threshold':t,**m})
            feasible=[r for r in curve if r['FP']<=b['FP'] and r['P']>=b['P']]
            chosen=max(feasible,key=lambda r:(r['TP'],-r['FP'],-abs(r['threshold']-.25)))
            thresholds.append(chosen['threshold']);curves[str(c)]=curve
        # Freeze threshold selection before reading dev cache.
        atomic_json(out/'thresholds_frozen.json',{'thresholds':thresholds,'OOF_baseline':baseline,'OOF_selected':metrics(filtered(predictions,thresholds)),'curves':curves,'protocol':protocol})
        atomic_json(out/'all_predictions.json',predictions)
        dev=json.loads((out.parent/'phase1/source_predictions.json').read_text(encoding='utf8'))
        dev=[{'image_id':r['image_id'],'GT':r['GT'],'pred':nms(r['sources']['o2o']).tolist()} for r in dev]
        b=metrics(filtered(dev,[.25,.25]));m=metrics(filtered(dev,thresholds));m['new_FN_vs_baseline']=sorted(set(m['missed_GT'])-set(b['missed_GT']))
        atomic_json(out/'results.json',{'phase':'complete','thresholds':thresholds,'dev_baseline':b,'dev_selected':m,'OOF_source':'independent COCO initializations, grouped final40 heldout predictions','speed_pending':True})
        atomic_json(out/'status.json',{'phase':'complete'})
    except BaseException as e:
        atomic_json(out/'status.json',{'phase':'failed','error':repr(e),'traceback':traceback.format_exc()});raise

if __name__=='__main__':main()

