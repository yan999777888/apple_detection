"""Fixed-budget, train-only shield ablations. Original GT files never edited.
Geometry is applied jointly to true and ignore boxes; ignore boxes are NEVER
passed as targets. Every arm shares this same explicit affine data pipeline.
"""
import sys,json,math,random,copy,gc,time,traceback,argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'apple_research_v2/joint'))
from joint_state_v1 import configure,atomic_json,sha

def prepare_ignore(out):
    from audit_defect_causes_20261112 import ious,nms_indices
    samples=[r for r in json.loads((ROOT/'apple_research_v2/datasets/ssda_date_v2/manifest.json').read_text(encoding='utf8'))['samples'] if r['partition']=='train']
    rows=[]
    for s in samples:
        p=ROOT/'experiments/defect_cause_audit_20261112_v1/predictions'/f"{Path(s['image_id']).stem}.json";r=json.loads(p.read_text(encoding='utf8'))
        assert r['partition']=='train' and r['image_id']==s['image_id']
        high=[x for x in r['low_predictions'] if x[4]>=.15];high=[high[i] for i in nms_indices(high)];mat=ious([x[:4] for x in high],[g['box'] for g in r['GT']])
        candidate=[{'box':x[:4],'class':int(x[5]),'score':x[4],'max_IoU':float(mat[i].max())} for i,x in enumerate(high) if mat[i].max()<.1]
        rows.append({'image_id':s['image_id'],'image_path':s['rgb_path'],'label_sha256':s['label_sha256'],'candidates':candidate,'GT':r['GT']})
    doc={'schema':'train_only_negative_shield_v1','teacher_sha256':'f8becdeff0e042fa19d681141aa56e133393130b4a2af23133706fd7766367f6','teacher_role':'in-sample original trained model proposals, NOT OOF or confirmed missed labels','confidence':.15,'max_IoU_all_GT_lt':.1,'images':len(rows),'images_with_candidates':sum(bool(r['candidates']) for r in rows),'candidate_count':sum(len(r['candidates']) for r in rows),'positive_protection':'no assigned fg anchor OR anchor center inside ANY true GT is shielded','rows':rows}
    atomic_json(out/'defect_train_ignore.json',doc);return samples,doc

def transform_boxes(boxes,scale,tx,ty,flip):
    import numpy as np
    b=np.asarray(boxes,dtype=np.float32).reshape(-1,4)*(.625*scale);b[:,[0,2]]+=tx;b[:,[1,3]]+=ty
    before=b.copy();b=b.clip(0,640)
    if flip:b[:,[0,2]]=640-b[:,[2,0]]
    wh=b[:,2:]-b[:,:2];origwh=before[:,2:]-before[:,:2]
    keep=(wh[:,0]>2)&(wh[:,1]>2)&(wh.prod(1)/(origwh.prod(1)+1e-9)>.1)&(np.maximum(wh[:,0]/(wh[:,1]+1e-9),wh[:,1]/(wh[:,0]+1e-9))<100)
    return b,keep

def make_batch(indices,samples,ignores,epoch,draw_start,wide,device):
    import numpy as np,cv2,torch
    ims=[];classes=[];boxes=[];batchidx=[];shield=[];ids=[];dropped=0
    for bi,idx in enumerate(indices):
        s=samples[idx];rng=np.random.default_rng(20261112+epoch*100000+draw_start+bi)
        scale=float(rng.uniform(.5,1.3) if wide else rng.uniform(.75,1.25));tx=320*(1-scale)+float(rng.uniform(-64,64));ty=320*(1-scale)+float(rng.uniform(-64,64));flip=bool(rng.random()<.5)
        im=cv2.imread(s['rgb_path']);assert im.shape[:2]==(1024,1024)
        im=cv2.warpAffine(im,np.float32([[.625*scale,0,tx],[0,.625*scale,ty]]),(640,640),flags=cv2.INTER_LINEAR,borderValue=(114,114,114))
        if flip:im=im[:,::-1].copy()
        hsv=cv2.cvtColor(im,cv2.COLOR_BGR2HSV);hsv[:,:,2]=np.clip(hsv[:,:,2].astype(float)*(1+rng.uniform(-.1,.1)),0,255).astype(np.uint8);im=cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
        gt=ignores[s['image_id']]['GT'];b,keep=transform_boxes([g['box'] for g in gt],scale,tx,ty,flip);dropped+=int((~keep).sum());b=b[keep]
        c=[g['class'] for g,k in zip(gt,keep) if k];xywh=np.c_[(b[:,:2]+b[:,2:])/2,b[:,2:]-b[:,:2]]/640
        boxes.extend(xywh.tolist());classes.extend(c);batchidx.extend([bi]*len(c))
        ib,ik=transform_boxes([g['box'] for g in ignores[s['image_id']]['candidates']],scale,tx,ty,flip);shield.append(torch.tensor(ib[ik],device=device))
        ims.append(np.ascontiguousarray(im[:,:,::-1].transpose(2,0,1)));ids.append(s['image_id'])
    return {'img':torch.from_numpy(np.stack(ims)).to(device).float()/255,'cls':torch.tensor(classes,device=device,dtype=torch.float32).reshape(-1,1),'bboxes':torch.tensor(boxes,device=device,dtype=torch.float32).reshape(-1,4),'batch_idx':torch.tensor(batchidx,device=device,dtype=torch.int64),'ignore_boxes':shield,'source':'defect','annotation_available':{'stage':False,'defect':True},'anchor_ids':ids,'aug_clipped_GT':dropped}

def attach_shield(criterion):
    import torch
    from ultralytics.utils.loss import v8DetectionLoss
    from ultralytics.utils.ops import xywh2xyxy
    class CaptureBCE(torch.nn.BCEWithLogitsLoss):
        def forward(self,input,target):self.target=target;return super().forward(input,target)
    class ShieldLoss(v8DetectionLoss):
        def get_assigned_targets_and_loss(self,preds,batch):
            result=super().get_assigned_targets_and_loss(preds,batch);(fg,_,_,anchors,strides),loss,det=result
            target=self.bce.target;self.last_shielded=0
            if not any(len(x) for x in batch['ignore_boxes']):return result
            points=anchors*strides;mask=torch.zeros_like(fg)
            for bi,ib in enumerate(batch['ignore_boxes']):
                def inside(bb):return ((points[:,None,:]>=bb[None,:,:2])&(points[:,None,:]<=bb[None,:,2:])).all(-1).any(-1)
                if len(ib):mask[bi]=inside(ib)
                gt=xywh2xyxy(batch['bboxes'][batch['batch_idx']==bi])*640
                if len(gt):mask[bi]&=~inside(gt)
            mask&=~fg;mask&=(target==0).all(-1)
            self.last_shielded=int(mask.sum());self.last_fg=int(fg.sum())
            # Only target-zero background anchors; positive weights/box loss unchanged.
            logits=preds['scores'].permute(0,2,1).contiguous()
            remove=torch.nn.functional.softplus(logits)*mask.unsqueeze(-1)
            if self.class_weights is not None:remove=remove*self.class_weights
            newloss=loss.clone();newloss[1]=loss[1]-remove.sum()/max(target.sum(),1)*self.hyp.cls
            return result[0],newloss,dict(zip(self.loss_names,newloss.detach()))
    criterion.__class__=ShieldLoss;criterion.bce=CaptureBCE(reduction='none');criterion.last_shielded=0

ARMS={'G0_control':{},'G1_shield':{'shield':True},'G2_sampling':{'sampling':True},'G3_scale':{'wide':True},'G4_beta3':{'beta':3},'G5_combined':{'shield':True,'sampling':True,'wide':True,'beta':3}}

def main():
    p=argparse.ArgumentParser();p.add_argument('--prepare-only',action='store_true');p.add_argument('--smoke',action='store_true');a=p.parse_args()
    out=ROOT/'experiments/defect_algorithm_steps_20261112_v2'/('shield_prepare' if a.prepare_only else 'train_smoke' if a.smoke else 'train')
    if out.exists():raise ValueError('Fresh output only')
    configure(out,'cpu' if a.prepare_only else 'cuda:0');samples,doc=prepare_ignore(out)
    if a.prepare_only:return
    import torch,numpy as np,cv2
    from joint_model_v1 import build_model,tensor_sha,finite_state
    from joint_state_v1 import seed_everything,get_rng,set_rng
    from joint_eval_v1 import evaluate_task,source_spec
    from defect_algorithm_modes_20261112 import AlgorithmAdapter,nms
    from run_defect_zero_train_20261112 import metrics
    from benchmark_joint_20261112 import summary
    cp=ROOT/'experiments/strong_defect_composite_20261112_v1/composite_frozen.pt';assert sha(cp)==doc['teacher_sha256']
    seed_everything(20261112);initial_rng=get_rng();ignore={r['image_id']:r for r in doc['rows']}
    protocol={'arms':ARMS,'epochs':3,'updates_per_arm':114,'lr':1e-5,'FP32':True,'fresh_AdamW':True,'seed':20261112,'microbatch':4,'effective_batch':16,'draws_per_epoch':596,'baseline_scale':[.75,1.25],'wide_scale':[.5,1.3],'affine_translate':.1,'flip':.5,'HSV_v':.1,'mosaic':0,'sampling':'2.5x probability for image containing class0 OR sqrt(box area at640)>=96; 596 replacement draws, not 2.5x total updates','G5':'combined predeclared, individual changes separately measured','all_from_same_strong_checkpoint':sha(cp),'GT_edits':False,'criterion':'final40 o2m .1/o2o .9; default box loss retained; TAL beta only in G4/G5','code_sha256':sha(__file__),'selection':'fixed epoch3; no auto promotion; accuracy first near20FPS','smoke':a.smoke}
    atomic_json(out/'protocol.json',protocol);results={}
    try:
        for arm,settings in ARMS.items():
            folder=out/arm;folder.mkdir();set_rng(initial_rng);state=torch.load(cp,map_location='cpu',weights_only=False);model=build_model('separate',device='cuda:0');model.load_state_dict(state['model']);del state
            model.set_criterion_epoch(40);model.trunks['stage'].requires_grad_(False);model.heads['stage'].requires_grad_(False)
            def stagehash():return tensor_sha({k:v for k,v in model.state_dict().items() if k.startswith(('trunks.stage.','heads.stage.'))})
            frozen=stagehash()
            for loss in (model.criteria['defect'].one2many,model.criteria['defect'].one2one):
                loss.assigner.beta=settings.get('beta',6)
                if settings.get('shield'):attach_shield(loss)
            groups=[g for g in model.parameter_groups() if g['component'] in ('trunks.defect','heads.defect')];opt=torch.optim.AdamW(groups,lr=1e-5,betas=(.9,.999),eps=1e-8);params=[p for g in groups for p in g['params']]
            history=[];steps=0
            for epoch in range(1 if a.smoke else 3):
                rng=np.random.default_rng(20261112+epoch);weights=np.array([2.5 if any(g['class']==0 or math.sqrt((g['box'][2]-g['box'][0])*(g['box'][3]-g['box'][1]))*.625>=96 for g in ignore[s['image_id']]['GT']) else 1 for s in samples])
                order=rng.choice(len(samples),len(samples),replace=True,p=weights/weights.sum()).tolist() if settings.get('sampling') else rng.permutation(len(samples)).tolist()
                atomic_json(folder/f'epoch{epoch+1}_draws.json',{'image_ids':[samples[i]['image_id'] for i in order],'unique':len(set(order))})
                for start in range(0,len(order),16):
                    indices=order[start:start+16];model.train();model.trunks['stage'].eval();model.heads['stage'].eval();opt.zero_grad(set_to_none=True);value=0;masked=0;dropped=0
                    for offset in range(0,len(indices),4):
                        batch=make_batch(indices[offset:offset+4],samples,ignore,epoch,start+offset,settings.get('wide',False),'cuda:0')
                        loss,_=model.source_loss('defect',batch);loss=loss.sum()/len(indices)
                        if not torch.isfinite(loss):raise ValueError('Nonfinite loss; no skipped batch')
                        loss.backward();value+=float(loss.detach());dropped+=batch['aug_clipped_GT'];masked+=sum(getattr(c,'last_shielded',0) for c in (model.criteria['defect'].one2many,model.criteria['defect'].one2one))
                    torch.nn.utils.clip_grad_norm_(params,10,error_if_nonfinite=True);opt.step();steps+=1
                    history.append({'epoch':epoch+1,'step':steps,'loss':value,'shielded_anchor_branches':masked,'aug_clipped_GT':dropped})
                    atomic_json(out/'status.json',{'phase':'training','arm':arm,'epoch':epoch+1,'step':steps,'budget':114})
                    if a.smoke and steps==2:break
                assert stagehash()==frozen;finite_state(model.state_dict(),'trained model')
            assert steps==(2 if a.smoke else 114)
            path=folder/'final.pt';torch.save({'model':model.state_dict(),'optimizer':opt.state_dict(),'rng':get_rng(),'steps':steps,'protocol':protocol,'arm':arm},path);atomic_json(folder/'history.json',history)
            result={'checkpoint':str(path),'checkpoint_sha256':sha(path),'steps':steps,'stage_unchanged':True}
            if not a.smoke:
                atomic_json(out/'status.json',{'phase':'validation','arm':arm});model.eval();official=evaluate_task(model,'defect',output_dir=folder/'official_val',device='cuda:0');result['official']=official['metrics']
                worker=AlgorithmAdapter(model.eval().requires_grad_(False));dev=json.loads((out.parent/'phase1/source_predictions.json').read_text(encoding='utf8'));allmanifest=json.loads((ROOT/'apple_research_v2/datasets/ssda_date_v2/manifest.json').read_text(encoding='utf8'))['samples'];lookup={s['image_id']:s for s in allmanifest};records=[];low=[]
                for r in dev:
                    im=cv2.imread(lookup[r['image_id']]['rgb_path']);t=worker.predictors['stage'].preprocess([im]);p=nms(worker.sources(t,im.shape[:2])['o2o']);low.append({'image_id':r['image_id'],'GT':r['GT'],'pred':p.tolist()});records.append({'image_id':r['image_id'],'GT':r['GT'],'pred':p[p[:,4]>=.25].tolist()})
                result['fixed']=metrics(records);base=json.loads((out.parent/'phase1/validation.json').read_text(encoding='utf8'))['o2o_nms'];result['new_FN_vs_baseline']=sorted(set(result['fixed']['missed_GT'])-set(base['missed_GT']));result['recovered_FN']=sorted(set(base['missed_GT'])-set(result['fixed']['missed_GT']))
                atomic_json(folder/'predictions.json',low)
                images=[]
                for task in ('stage','defect'):
                    pp=source_spec(task)['images'];images.extend(cv2.imread(pp[round(i*(len(pp)-1)/19)]) for i in range(20))
                for im in images[:5]:worker.predict_mode(im)
                times=[]
                for _ in range(3):
                    for im in images:
                        torch.cuda.synchronize();t=time.perf_counter();pred=worker.predict_mode(im);torch.cuda.synchronize();times.append((time.perf_counter()-t)*1000);del pred
                result['speed']=summary(times);result['speed_ms']=times;del worker
            results[arm]=result;atomic_json(folder/'result.json',result);atomic_json(out/'results_partial.json',results)
            del model,opt,groups,params;gc.collect();torch.cuda.empty_cache()
        for s in samples:assert sha(s['label_path'])==s['label_sha256'] and sha(s['rgb_path'])==s['rgb_sha256']
        assert sha(cp)==doc['teacher_sha256'];atomic_json(out/'results.json',{'phase':'complete','arms':results,'GT_unchanged':True,'retained_weights_unchanged':True});atomic_json(out/'status.json',{'phase':'complete'})
    except BaseException as e:atomic_json(out/'status.json',{'phase':'failed','error':repr(e),'traceback':traceback.format_exc()});raise

if __name__=='__main__':main()
