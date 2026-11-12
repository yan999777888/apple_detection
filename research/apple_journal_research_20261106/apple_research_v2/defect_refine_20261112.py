"""Bounded defect-only C0/C1 fine-tuning from retained separate final40 EMA.
No architecture/data changes. C1 changes only box loss gain, 7.5 -> 11.25.
Three full source epochs per arm; frozen stage branch, FP32, fresh AdamW.
"""
import argparse, gc, json, sys, time, traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'apple_research_v2/joint'))
from joint_state_v1 import configure,atomic_json,sha,load_checkpoint,seed_everything,get_rng,set_rng

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--smoke',action='store_true');a=p.parse_args()
    if a.output.exists():raise ValueError('Fresh output required')
    out=configure(a.output,'cuda:0')
    import torch,cv2
    from joint_model_v1 import build_model,tensor_sha,finite_state
    from joint_data import JointDataModule
    from joint_engine_v1 import device_batch
    from joint_eval_v1 import evaluate_task,source_spec
    from deployment_adapter_v1 import FrozenDeploymentAdapter
    from benchmark_joint_20261112 import summary
    cp=ROOT/'runs/separate_matched_e40_s20261112_v1/checkpoints/final40_complete.pt'
    cp_sha=sha(cp);assert cp_sha=='a2743af0f1bdbbde9a3f6f3a030e6c53c79ad814905d8b974630f2ad08298340'
    protocol={'source':str(cp),'source_sha256':cp_sha,'seed':20261112,'arms':{'C0':7.5,'C1':11.25},
        'epochs':3,'source_windows_per_epoch':38,'images_per_epoch':596,'microbatch':4,'effective_batch_max':16,
        'lr':.0001,'weight_decay':.0005,'optimizer':'fresh AdamW','precision':'FP32','imgsz':640,
        'data_epochs':[35,36,37],'criterion_schedule':'fixed final40 o2m .1/o2o .9',
        'selection':'fixed epoch3 per arm; epoch1 diagnostic; development only; no automatic promotion',
        'speed_gate':'two-task BGR-array to CPU Results mean and P95 <=50ms; conf.25 square640',
        'precision_gate':'candidate defect AP50-95 above retained baseline AND C0; AP50 not below retained; stage state exact',
        'smoke':a.smoke,'script_sha256':sha(__file__)}
    atomic_json(out/'protocol.json',protocol)
    try:
        seed_everything(20261112)
        data=JointDataModule();results={};initial_rng=get_rng()
        for arm,gain in protocol['arms'].items():
            set_rng(initial_rng);folder=out/arm;folder.mkdir()
            state=load_checkpoint(cp);model=build_model('separate',device='cuda:0')
            model.load_state_dict(state['ema']['model']);del state
            model.set_criterion_epoch(40)
            for loss in (model.criteria['defect'].one2many,model.criteria['defect'].one2one):loss.hyp.box=gain
            model.trunks['stage'].requires_grad_(False);model.heads['stage'].requires_grad_(False)
            def stage_hash():return tensor_sha({k:v for k,v in model.state_dict().items() if k.startswith(('trunks.stage.','heads.stage.'))})
            frozen=stage_hash()
            groups=[g for g in model.parameter_groups() if g['component'] in ('trunks.defect','heads.defect')]
            opt=torch.optim.AdamW(groups,lr=.0001,betas=(.9,.999),eps=1e-8)
            rows=[];steps=0;validations=[]
            for epoch in range(1 if a.smoke else 3):
                seen=[]
                for wi,window in enumerate(data.iter_epoch(35+epoch,task='defect')):
                    model.train();model.trunks['stage'].eval();model.heads['stage'].eval();opt.zero_grad(set_to_none=True)
                    value=0
                    for batch in window['batches']:
                        loss,_=model.source_loss('defect',device_batch(batch,'cuda:0'))
                        loss=loss.sum()/window['anchor_count'];loss.backward();value+=float(loss.detach())
                        seen.extend(batch['anchor_ids'])
                    params=[p for g in groups for p in g['params']]
                    torch.nn.utils.clip_grad_norm_(params,10,error_if_nonfinite=True);opt.step();opt.zero_grad(set_to_none=True)
                    finite_state(model.state_dict(),'refined model');steps+=1
                    rows.append({'epoch':epoch+1,'window':wi+1,'loss':value,'anchor_ids':list(seen[-window['anchor_count']:])})
                    atomic_json(out/'status.json',{'phase':'training','arm':arm,'epoch':epoch+1,'window':wi+1,'steps':steps})
                    if a.smoke and wi==1:break
                assert stage_hash()==frozen
                if not a.smoke:assert len(seen)==len(set(seen))==596 and wi+1==38
                path=folder/f'epoch_{epoch+1:03d}.pt'
                torch.save({'schema':'defect_refine_v1','model':model.state_dict(),'optimizer':opt.state_dict(),
                    'rng':get_rng(),'epoch':epoch+1,'steps':steps,'protocol':protocol,'arm':arm,'box_gain':gain},path)
                atomic_json(path.with_suffix('.sha256.json'),{'sha256':sha(path),'stage_sha256':frozen})
                if not a.smoke and epoch in (0,2):
                    atomic_json(out/'status.json',{'phase':'validating','arm':arm,'epoch':epoch+1})
                    model.eval();receipt=evaluate_task(model,'defect',output_dir=folder/f'val_{epoch+1:03d}',device='cuda:0')
                    validations.append({'epoch':epoch+1,'metrics':receipt['metrics'],'per_class':receipt['per_class']})
                    atomic_json(folder/'validation_summary.json',validations)
            timings=None
            if not a.smoke:
                model.eval().requires_grad_(False);adapter=FrozenDeploymentAdapter(model)
                paths=[]
                for task in ('stage','defect'):
                    eligible=source_spec(task)['images'];paths.extend(eligible[round(i*(len(eligible)-1)/19)] for i in range(20))
                images=[cv2.imread(x) for x in paths];measure=[]
                for i in range(10):adapter.predict(images[i],cpu=True)
                for repeat in range(4):
                    for im in images:
                        torch.cuda.synchronize();start=time.perf_counter();pred=adapter.predict(im,cpu=True);torch.cuda.synchronize()
                        measure.append((time.perf_counter()-start)*1000);del pred
                timings={'summary':summary(measure),'observations_ms':measure,'images':paths,'conf':.25,'max_det':300,'precision':'FP32','scope':'two-task array to CPU Results; excludes camera/depth/robot'}
                del adapter
            results[arm]={'phase':'complete','steps':steps,'stage_state_unchanged':stage_hash()==frozen,'checkpoint':str(path),'checkpoint_sha256':sha(path),'validation':validations,'speed':timings}
            atomic_json(folder/'history.json',rows);atomic_json(out/'results_partial.json',results)
            del model,opt,groups,params;gc.collect();torch.cuda.empty_cache()
        data.verify_sources();assert sha(cp)==cp_sha
        atomic_json(out/'results.json',{'phase':'complete','protocol':protocol,'arms':results,'promotion':'none; review paired evidence'})
        atomic_json(out/'status.json',{'phase':'complete','smoke':a.smoke})
    except BaseException as error:
        atomic_json(out/'status.json',{'phase':'failed','error':repr(error),'traceback':traceback.format_exc()});raise

if __name__=='__main__':main()
