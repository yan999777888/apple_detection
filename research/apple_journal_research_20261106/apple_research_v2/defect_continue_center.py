"""Fit, calibrate and test a stride-4 CenterNet-style defect detector."""

import argparse
import time
from functools import lru_cache

import cv2
import numpy as np
import torch
from torch import nn

from class0_patch_refinement import ROOT, CP, configure, atomic_json, sha, load_image, read
from defect_continue_center_net import SIDE, STRIDE, WEIGHTS, CenterDetector, heat_targets, objective, decode
from defect_continue_transplant import split_train
from defect9485_p2 import samples
from defect9485_scene import policy
from run_defect_zero_train_20261112 import metrics

OUT = ROOT / "experiments/defect_continue_center"
EPOCHS = 30
CHECKS = (10, 20, 30)
BATCH = 4
SEED = 29851


@lru_cache(maxsize=150)
def resized(path, side=SIDE):
    return cv2.resize(load_image(path),(side,side),interpolation=cv2.INTER_AREA)


def labels(row,h,w):
    boxes=[]
    for g in row["instances"]:
        x,y,bw,bh=g["xywh_normalized"]
        boxes.append([(x-bw*.5)*w,(y-bh*.5)*h,(x+bw*.5)*w,(y+bh*.5)*h,g["class_id"]])
    return np.asarray(boxes,np.float32).reshape(-1,5)


def reference_gt(row,h,w):
    out=[]
    for g in row["instances"]:
        x,y,bw,bh=g["xywh_normalized"]
        out.append({"class":g["class_id"],"box":[(x-bw/2)*w,(y-bh/2)*h,
                                                (x+bw/2)*w,(y+bh/2)*h]})
    return out


def draw_training(row, rng, side=SIDE):
    im = resized(row["rgb_path"],side)
    boxes = labels(row,side,side)
    if rng.random()<.5:
        im = im[:,::-1]
        boxes[:,[0,2]] = side-boxes[:,[2,0]]
    rotation=int(rng.integers(4))
    im=np.rot90(im,rotation).copy()
    for _ in range(rotation):
        boxes[:,:4]=np.column_stack([boxes[:,1],side-boxes[:,2],boxes[:,3],side-boxes[:,0]])
    if rng.random()<.6:
        factor=rng.uniform(.86,1.09)
        shift=(1-factor)*side/2 + rng.uniform(-.025,.025,2)*side
        matrix=np.float32([[factor,0,shift[0]],[0,factor,shift[1]]])
        im=cv2.warpAffine(im,matrix,(side,side),flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT,borderValue=(0,0,0))
        boxes[:,:4]=boxes[:,:4]*factor+np.tile(shift,2)
        old=(boxes[:,2:4]-boxes[:,:2]).prod(1)
        boxes[:,:4]=boxes[:,:4].clip(0,side)
        extent=boxes[:,2:4]-boxes[:,:2]
        boxes=boxes[(extent.min(1)>=2)&(extent.prod(1)>.55*old)]
    gain=rng.uniform(.83,1.19) * rng.uniform(.96,1.04,3)
    lut=np.clip(np.arange(256)[:,None]*gain+rng.uniform(-7,7),0,255).astype(np.uint8)
    im=cv2.LUT(im,lut[:,None,:])
    return im,boxes


def pack(images,device):
    rgb=np.stack([im[:,:,::-1] for im in images])
    return torch.from_numpy(rgb).permute(0,3,1,2).to(device=device,memory_format=torch.channels_last)


@torch.inference_mode()
def predict(model,rows):
    result=[]
    for r in rows:
        full=load_image(r["rgb_path"])
        H,W=full.shape[:2]
        im=cv2.resize(full,(SIDE,SIDE),interpolation=cv2.INTER_AREA)
        with torch.autocast("cuda",dtype=torch.bfloat16):
            pred=decode(model(pack([im],"cuda")),[(H,W)])[0]
        gt=reference_gt(r,H,W)
        result.append(dict(image_id=r["image_id"],GT=gt,pred=pred.tolist()))
    return result


def calibrate(predictions):
    return policy([dict(oof=np.array(r["pred"]).reshape(-1,6),GT=r["GT"]) for r in predictions])


def measured(raw, point):
    rows = []
    for r in raw:
        prediction = [p for p in r["pred"] if p[4]>=point["thresholds"][int(p[5])]]
        rows.append(dict(image_id=r["image_id"],GT=r["GT"],pred=prediction))
    return rows,metrics(rows)


def fit():
    if (OUT / "frozen.json").exists():
        raise RuntimeError("Completed training receipt exists")
    fit_data,calib = split_train()
    assert (len(fit_data),len(calib)) == (490,106)
    model=CenterDetector().cuda().to(memory_format=torch.channels_last)
    body=[*model.stem.parameters(),*model.layer1.parameters(),*model.layer2.parameters(),*model.layer3.parameters()]
    bodyids={id(v) for v in body}
    other=[p for p in model.parameters() if id(p) not in bodyids]
    opt=torch.optim.AdamW([dict(params=body,lr=.00010),dict(params=other,lr=.0012)],
                          weight_decay=.008)
    steps=(len(fit_data)+BATCH-1)//BATCH
    schedule=torch.optim.lr_scheduler.OneCycleLR(opt,max_lr=[.00010,.0012],total_steps=steps*EPOCHS,
                                                 pct_start=.12,div_factor=16,final_div_factor=140)
    saved=[]
    rng=np.random.default_rng(SEED)
    rawlog=[]
    for ep in range(1,EPOCHS+1):
        model.train()
        for mod in (model.stem, model.layer1, model.layer2, model.layer3):
            for m in mod.modules():
                if isinstance(m,nn.BatchNorm2d):
                    m.eval()
        idx=rng.permutation(len(fit_data)); total=0.
        epoch_start=time.monotonic()
        for lo in range(0,len(idx),BATCH):
            records=[draw_training(fit_data[j],rng) for j in idx[lo:lo+BATCH]]
            target=heat_targets([g for _,g in records],(SIDE//STRIDE,)*2)
            inp=pack([im for im,_ in records],"cuda")
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda",dtype=torch.bfloat16):
                result=model(inp)
            loss=objective(result,target)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Nonfinite training loss at {ep=} {lo=}")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),8.)
            opt.step()
            schedule.step()
            total+=loss.item()
        entry=dict(epoch=ep,mean_loss=total/steps,seconds=time.monotonic()-epoch_start)
        if ep in CHECKS:
            model.eval()
            raw=predict(model,calib)
            options=calibrate(raw)
            atomic_json(OUT/f"calib_{ep:02}_low.json",raw)
            torch.save(model.state_dict(),OUT/f"epoch_{ep:02}.pt")
            entry["calibration"]=options
            saved.append((ep,options))
        rawlog.append(entry)
        atomic_json(OUT/"history.json",rawlog)
        print(entry,flush=True)
    def criterion(x):
        p=x[1]["joint"]
        return (min(p["P"]/.94,p["R"]/.85),p["P"]*p["R"])
    ep,pts=max(saved,key=criterion)
    weight=OUT/f"epoch_{ep:02}.pt"
    atomic_json(OUT/"frozen.json",dict(epoch=ep,weights=weight.name,weights_sha=sha(weight),
                points=pts,encoder_sha=sha(WEIGHTS),fit_ids=[r["image_id"] for r in fit_data],
                calib_ids=[r["image_id"] for r in calib],input=SIDE,stride=STRIDE,
                predeclared_epochs=CHECKS,calibration_rule="max min(P/.94,R/.85) then P*R",
                fit_gt=sum(len(r["instances"]) for r in fit_data),
                calibration_gt=sum(len(r["instances"]) for r in calib)))


def check():
    fit_data,calib=split_train()
    assert (len(fit_data),len(calib))==(490,106)
    model=CenterDetector().train()
    params=sum(p.numel() for p in model.parameters())
    rng=np.random.default_rng(SEED)
    short=224
    ims=[draw_training(r,rng,short) for r in fit_data[:2]]
    batch=pack([im for im,g in ims],"cpu")
    lab=[g for im,g in ims]
    sample=model(batch)
    target=heat_targets(lab,(short//STRIDE,)*2)
    loss=objective(sample,target)
    loss.backward()
    assert torch.isfinite(loss) and model.layer3[-1].conv3.weight.grad.abs().sum()>0
    assert all(v[-1].weight.grad.abs().sum()>0 for v in (model.hm,model.reg,model.quality))
    assert sample[0].shape==(2,2,short//4,short//4)
    empty=objective(sample,heat_targets([[],[]],(short//STRIDE,)*2))
    assert torch.isfinite(empty)
    # At unit stride resolution a handcrafted off-grid center decodes into exactly its GT.
    h=torch.full((1,2,10,10),-30.)
    h[0,1,4,6]=20; size=torch.zeros(1,8,10,10); q=h.clone()
    size[0,4:,4,6]=torch.tensor([.25,.75,np.log(3.),np.log(2.)])
    pred=decode((h,size,q),[(40,40)])[0]
    assert pred.shape==(1,6) and int(pred[0,5])==1
    np.testing.assert_allclose(pred[0,:4],[19,15,31,23],atol=1e-5)
    # Exponentiated low-confidence background peaks are finite and suppressed.
    a=heat_targets([np.array([[19,15,31,23,1]])],(10,10))
    assert a[0][0,1,4,6]==1. and len(a[1])==9
    with torch.no_grad():
        model.eval()
        out=decode(model(pack([resized(fit_data[0]["rgb_path"])],"cpu")),[(1024,1024)])[0]
    assert np.isfinite(out).all()
    atomic_json(OUT/"preflight.json",dict(training_images=len(fit_data),calibration_images=len(calib),
                fit_gt=sum(len(r["instances"]) for r in fit_data),dev_images=113,
                threads=torch.get_num_threads(),device="cpu",parameters=params,local_coco_sha=sha(WEIGHTS),
                real_224_batch2_loss=float(loss.detach()),empty_loss=float(empty.detach()),
                real_704_forward=True,backbone_and_all_heads_have_gradient=True,
                offset_size_decode_exact=True,nine_cell_regression_targets=True))
    print(read(OUT/"preflight.json"))


@torch.inference_mode()
def evaluate():
    if (OUT/"fresh_metrics.json").exists():
        raise RuntimeError("Locked development evaluation already completed")
    f=read(OUT/"frozen.json")
    p=OUT/f["weights"]
    assert sha(p)==f["weights_sha"]
    model=CenterDetector(pretrained=False).cuda().eval().to(memory_format=torch.channels_last)
    model.load_state_dict(torch.load(p,map_location="cpu",weights_only=True))
    raw=predict(model,samples("dev"))
    assert len(raw)==113 and sum(len(r["GT"]) for r in raw)==230
    atomic_json(OUT/"fresh_low_predictions.json",raw)
    res={}
    for name,pt in f["points"].items():
        rows,res[name]=measured(raw,pt)
        atomic_json(OUT/f"fresh_{name}_predictions.json",rows)
    atomic_json(OUT/"fresh_metrics.json",res)
    print("Fresh development set",{k:r["total"] for k,r in res.items()},flush=True)


@torch.inference_mode()
def benchmark():
    from defect_four_orientations_20261118 import make_worker
    from benchmark_joint_20261112 import summary
    from joint_eval_v1 import source_spec
    from ultralytics.engine.results import Results
    f=read(OUT/"frozen.json")
    weight=OUT/f["weights"]
    assert sha(weight)==f["weights_sha"]
    model=CenterDetector(pretrained=False).cuda().eval().to(memory_format=torch.channels_last)
    model.load_state_dict(torch.load(weight,map_location="cpu",weights_only=True))
    worker=make_worker(CP,False)
    stage=worker.predictors["stage"]
    ts=np.asarray(f["points"]["joint"]["thresholds"])
    def once(im):
        h,w=im.shape[:2]
        xs=pack([cv2.resize(im,(SIDE,SIDE),interpolation=cv2.INTER_AREA)],"cuda")
        with torch.autocast("cuda",dtype=torch.bfloat16):
            det=decode(model(xs),[(h,w)])[0]
        det=det[det[:,4]>=ts[det[:,5].astype(int)]]
        defect=Results(im,"frame",worker.model.names["defect"],boxes=torch.from_numpy(det))
        t=stage.preprocess([im])
        stage.batch=(["frame"],[im],[""])
        maturity=[r.cpu() for r in stage.postprocess(worker.model.predict_source(t,"stage"),t,[im])]
        return maturity,defect
    images=[]
    for task in ("stage","defect"):
        paths=source_spec(task)["images"]
        images.extend(cv2.imread(paths[round(i*(len(paths)-1)/19)]) for i in range(20))
    for im in images[:5]:
        once(im)
    ms=[]
    for _ in range(15):
        for im in images:
            torch.cuda.synchronize()
            t=time.perf_counter()
            once(im)
            torch.cuda.synchronize()
            ms.append(1000*(time.perf_counter()-t))
    timing=summary(ms)
    atomic_json(OUT/"benchmark.json",dict(timing=timing,frozen_point=f["points"]["joint"],
                device=torch.cuda.get_device_name(),images=40,repeats=15,warmup=5,
                precision="bf16 defect 704, fp32 original maturity",excludes="disk read and acquisition"))
    print(timing,flush=True)


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("action",choices=["check","fit","evaluate","benchmark","run"])
    act=ap.parse_args().action
    configure(OUT,"cpu" if act=="check" else "cuda:0")
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    cv2.setNumThreads(1)
    torch.manual_seed(SEED)
    if act=="check":
        check()
    if act in ("fit","run"):
        fit()
    if act in ("evaluate","run"):
        evaluate()
    if act in ("benchmark","run"):
        benchmark()
