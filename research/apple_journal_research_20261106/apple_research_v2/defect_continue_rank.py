"""Train the single-backbone quality head on fit490, select on calib106, evaluate once on dev113."""

import argparse
from copy import deepcopy
from time import perf_counter
import math
import cv2
import numpy as np
import torch
from ultralytics.utils.loss import E2ELoss
from ultralytics.utils.torch_utils import ModelEMA
from class0_patch_refinement import ROOT, CP, atomic_json, configure, sha, read, load_image
from defect9485_small import BASE_SHA
from defect9485_p2 import samples
from defect_continue_center import resized, labels, reference_gt, pack, measured, calibrate
from defect_continue_transplant import split_train
from defect_continue_rank_net import QualityDetector, joint_loss
from defect_algorithm_modes_20261112 import nms
from joint_model_v1 import LossView

OUT=ROOT/"experiments/defect_continue_rank"
SIDE=768
EPOCHS=36
CHECKS=(12,24,36)
BATCH=4
ACCUM=3
SEED=61385


def spatial(image, boxes, flip, turns):
    side=image.shape[0]
    if image.shape[1]!=side:
        raise ValueError("A square RGB input is required")
    b=boxes.copy()
    if flip:
        image=image[:,::-1]
        b[:,[0,2]]=side-b[:,[2,0]]
    image=np.rot90(image,turns)
    for _ in range(turns):
        b[:,:4]=np.column_stack([b[:,1],side-b[:,2],b[:,3],side-b[:,0]])
    return np.ascontiguousarray(image),b


def train_pair(row,rng,side=SIDE):
    image=resized(row["rgb_path"],side)
    b=labels(row,side,side)
    image,b=spatial(image,b,rng.random()<.5,int(rng.integers(4)))
    gains=rng.uniform(.83,1.19)*rng.uniform(.92,1.08,3)
    bias=rng.uniform(-6,6)
    lut=np.clip(np.arange(256)[:,None]*gains+bias,0,255).astype(np.uint8)
    return cv2.LUT(image,lut[:,None,:]),torch.from_numpy(b.copy())


@torch.inference_mode()
def boxes(model, im, device):
    h,w=im.shape[:2]
    y=pack([cv2.resize(im,(SIDE,SIDE),interpolation=cv2.INTER_AREA)],device).float().div_(255)
    with torch.autocast("cuda",dtype=torch.bfloat16,enabled=y.is_cuda):
        p=model(y)[0].float()
    p=p[p[:,4]>=.001].cpu().numpy()
    p[:,[0,2]]=(p[:,[0,2]]*(w/SIDE)).clip(0,w)
    p[:,[1,3]]=(p[:,[1,3]]*(h/SIDE)).clip(0,h)
    return nms(p)


def predict(model,rows,device="cuda"):
    model.eval()
    result=[]
    for r in rows:
        im=load_image(r["rgb_path"])
        result.append(dict(image_id=r["image_id"],GT=reference_gt(r,*im.shape[:2]),pred=boxes(model,im,device).tolist()))
    return result


def train():
    if (OUT/"history.json").exists() or (OUT/"frozen.json").exists():
        raise RuntimeError("Existing training evidence exists; refusing replacement")
    fit,cal=split_train()
    assert (len(fit),len(cal))==(490,106)
    fids={v["image_id"] for v in fit};cids={v["image_id"] for v in cal}
    assert not fids&cids
    atomic_json(OUT/"protocol.json",dict(source="COCO generic YOLO26s",source_sha256=BASE_SHA,
                                         fit_ids=sorted(fids),calibration_ids=sorted(cids),seed=SEED,
                                         epochs=EPOCHS,calibration_epochs=CHECKS,size=SIDE,
                                         native_loss="E2ELoss",quality_gains=[.30,.15,.65],dev_used_for_selection=False))
    torch.manual_seed(SEED)
    rng=np.random.default_rng(SEED)
    model=QualityDetector().cuda().to(memory_format=torch.channels_last).train()
    ema=ModelEMA(model,decay=.997,tau=200)
    crit=E2ELoss(LossView(model.head,EPOCHS))
    decay=[p for p in model.parameters() if p.requires_grad and p.ndim>1]
    nodecay=[p for p in model.parameters() if p.requires_grad and p.ndim<=1]
    opt=torch.optim.AdamW([dict(params=decay,weight_decay=.0005),dict(params=nodecay,weight_decay=0)],lr=.0008)
    history=[];selections=[];updates=0
    for e in range(EPOCHS):
        started=perf_counter()
        ids=rng.permutation(len(fit));accumulated={}
        for start in range(0,len(ids),BATCH*ACCUM):
            group=ids[start:start+BATCH*ACCUM];step=e+start/len(ids)
            ratio=(.05+.95*(1+math.cos(math.pi*max(0,step-2)/(EPOCHS-2)))/2)*min(1,(step+.05)/2)
            for g in opt.param_groups:g["lr"]=.0008*ratio
            opt.zero_grad(set_to_none=True)
            for offset in range(0,len(group),BATCH):
                inds=group[offset:offset+BATCH]
                pairs=[train_pair(fit[i],rng) for i in inds]
                x=pack([p[0] for p in pairs],"cuda").float().div_(255)
                target=[p[1].cuda() for p in pairs]
                with torch.autocast("cuda",dtype=torch.bfloat16):
                    raw,q=model(x)
                total,terms=joint_loss(model,crit,raw,q,target,(SIDE,SIDE))
                (total*(len(inds)/len(group))).backward()
                for key,v in terms.items():
                    value=v.detach()*len(inds)
                    accumulated[key]=accumulated.get(key,0)+value
            torch.nn.utils.clip_grad_norm_(model.parameters(),10.,error_if_nonfinite=True)
            opt.step();ema.update(model);updates+=1
        crit.update()
        ep=e+1
        row=dict(epoch=ep,updates=updates,seconds=perf_counter()-started,
                 losses={k:float(v/len(ids)) for k,v in accumulated.items()})
        if ep in CHECKS:
            name=f"epoch_{ep:02d}.pt"
            # Save tensors usable with the unfused and single-forward production model.
            torch.save(ema.ema.state_dict(),OUT/name)
            inference_model=deepcopy(ema.ema).eval()
            inference_model.base.fuse(verbose=False)
            cp=predict(inference_model,cal)
            del inference_model
            profiles=calibrate(cp)
            row["calibration"]=profiles
            selections.append((ep,profiles))
        history.append(row);atomic_json(OUT/"history.json",history)
        print(row,flush=True)
    ep,profiles=max(selections,key=lambda v:(min(v[1]["joint"]["P"]/.94,v[1]["joint"]["R"]/.85),
                                             v[1]["joint"]["TP"],v[1]["joint"]["P"]))
    name=f"epoch_{ep:02d}.pt"
    atomic_json(OUT/"frozen.json",dict(epoch=ep,weight=name,weight_sha256=sha(OUT/name),
                                       points=profiles,source_sha256=BASE_SHA,fit=len(fit),calibration=len(cal)))


def restored():
    receipt=read(OUT/"frozen.json")
    p=OUT/receipt["weight"]
    assert sha(p)==receipt["weight_sha256"]
    model=QualityDetector()
    model.load_state_dict(torch.load(p,map_location="cpu",weights_only=True))
    model=model.eval().cuda().to(memory_format=torch.channels_last)
    model.base.fuse(verbose=False)
    return model,receipt


def evaluate():
    if (OUT/"dev_metrics.json").exists():
        raise RuntimeError("Development evaluation already completed")
    model,f=restored()
    predictions=predict(model,samples("dev"))
    assert len(predictions)==113 and sum(len(v["GT"]) for v in predictions)==230
    atomic_json(OUT/"dev_low_predictions.json",predictions)
    result={}
    for k,prof in f["points"].items():
        fixed,result[k]=measured(predictions,prof)
        atomic_json(OUT/f"dev_{k}_predictions.json",fixed)
    atomic_json(OUT/"dev_metrics.json",result)
    print({k:v["total"] for k,v in result.items()},flush=True)


@torch.inference_mode()
def benchmark():
    from defect_four_orientations_20261118 import make_worker
    from benchmark_joint_20261112 import summary
    from joint_eval_v1 import source_spec
    from ultralytics.engine.results import Results
    model,f=restored()
    worker=make_worker(CP,False);stage=worker.predictors["stage"]
    cuts=np.array(f["points"]["joint"]["thresholds"])
    def infer(im):
        det=boxes(model,im,"cuda")
        det=det[det[:,4]>=cuts[det[:,5].astype(int)]]
        defect=Results(im,"frame",worker.model.names["defect"],boxes=torch.from_numpy(det))
        raw=stage.preprocess([im]);stage.batch=(["frame"],[im],[""])
        maturity=[r.cpu() for r in stage.postprocess(worker.model.predict_source(raw,"stage"),raw,[im])]
        return maturity,defect
    ims=[]
    for name in ("stage","defect"):
        pool=source_spec(name)["images"]
        ims.extend(cv2.imread(pool[round(i*(len(pool)-1)/19)]) for i in range(20))
    for im in ims[:5]:infer(im)
    ms=[]
    for _ in range(15):
        for im in ims:
            torch.cuda.synchronize();t=perf_counter();infer(im);torch.cuda.synchronize()
            ms.append((perf_counter()-t)*1000)
    atomic_json(OUT/"dual_speed.json",dict(timing=summary(ms),source_sha256=f["weight_sha256"],thresholds=cuts.tolist()))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode",choices=("cpu-check","train","evaluate","benchmark"))
    mode=parser.parse_args().mode
    configure(OUT,"cpu" if mode=="cpu-check" else "cuda")
    cv2.setNumThreads(1);torch.set_num_threads(2)
    if mode=="cpu-check":
        from defect_continue_rank_checks import check
        check()
    else:
        {"train":train,"evaluate":evaluate,"benchmark":benchmark}[mode]()
