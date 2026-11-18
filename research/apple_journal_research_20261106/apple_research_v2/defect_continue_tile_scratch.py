"""Held-fold test of native-resolution tiled scratches with a fixed insect-damage specialist."""

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from audit_defect_causes_20261112 import match
from audit_defect_target9485 import PHASE
from class0_patch_refinement import ROOT, OLD, configure, read, sha, atomic_json, load_image, manifest
from compare_defect_target9485 import indexed, load_candidates
from defect9485_p2 import NATIVE, SHA
from defect_algorithm_modes_20261112 import nms, wbf
from run_defect_zero_train_20261112 import metrics

OUT = ROOT / "experiments/defect_continue_tile_scratch"
MODES = ("full", "tile", "consensus")


def loader(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    return (checkpoint.get("ema") or checkpoint["model"]).float().eval().fuse(verbose=False)


@torch.inference_mode()
def propose(model, im):
    h, w = im.shape[:2]
    dh, dw = round(h * .6), round(w * .6)
    windows = [(0, 0, w, h)] + [(x, y, x+dw, y+dh) for y in (0,h-dh) for x in (0,w-dw)]
    views = [cv2.resize(im[t:b,l:r,::-1],(640,640),interpolation=cv2.INTER_LINEAR)
             for l,t,r,b in windows]
    x = torch.from_numpy(np.stack(views)).permute(0,3,1,2).float().div_(255)
    out = model(x)
    raw = out[0] if isinstance(out,tuple) else out
    branches = []
    for i,(l,t,r,b) in enumerate(windows):
        p = raw[i].detach().cpu().float().numpy()
        p = p[(p[:,5]==0)&(p[:,4]>=.005)].copy()
        p[:,[0,2]] = (p[:,[0,2]] * (r-l)/640 + l).clip(0,w)
        p[:,[1,3]] = (p[:,[1,3]] * (b-t)/640 + t).clip(0,h)
        if i:
            cx=(p[:,0]+p[:,2])/2
            cy=(p[:,1]+p[:,3])/2
            p=p[((cx >= w/2) == bool(l)) & ((cy >= h/2) == bool(t))]
        branches.append(p)
    full = nms(branches[0])
    tile = nms(np.concatenate(branches[1:]))
    return {"full":full, "tile":tile, "consensus":nms(wbf([full,tile]))}


def calibration():
    lookup = manifest()
    outrows=[]
    for f in range(3):
        folder=ROOT/f"runs/defect_oof3_fold{f}_e40_20261112_v1"
        cp=folder/"weights/last.pt"
        original=read(OLD/f"all_mode_OOF_calibration/fold{f}_sources.json")
        assert sha(cp)==original["checkpoint_sha256"]
        train={Path(s).name for s in (folder/"train_audited_images.txt").read_text("utf-8-sig").splitlines()}
        val={Path(s).name for s in (folder/"val_audited_images.txt").read_text("utf-8-sig").splitlines()}
        assert not train&val and val=={r["image_id"] for r in original["rows"]}
        model=loader(cp)
        rows=[]
        for r in original["rows"]:
            sid=r["image_id"]
            assert lookup[sid]["partition"]=="train"
            pp=propose(model,load_image(lookup[sid]["rgb_path"]))
            rows.append(dict(image_id=sid,GT=r["GT"],pred={k:v.tolist() for k,v in pp.items()}))
        atomic_json(OUT/f"held_fold{f}.json",dict(weight_sha=sha(cp),rows=rows))
        outrows+=rows
        print("held fold complete",f,len(rows),flush=True)
    assert len({r["image_id"] for r in outrows})==596
    n=sum(g["class"]==0 for r in outrows for g in r["GT"])
    summary={}
    for mode in MODES:
        evidence=[]
        for r in outrows:
            p=r["pred"][mode]
            pairs,_=match(p,r["GT"])
            evidence += [(b[4],i in pairs) for i,b in enumerate(p)]
        ev=np.array(evidence)
        points=[]
        for t in np.linspace(.005,.995,199):
            keep=ev[ev[:,0]>=t]
            tp=int(keep[:,1].sum())
            fp=len(keep)-tp
            points.append(dict(threshold=float(t),TP=tp,FP=fp,FN=n-tp,P=tp/max(1,len(keep)),R=tp/n))
        valid=[a for a in points if a["TP"]*87>=n*73]
        summary[mode]=max(valid,key=lambda a:(a["P"],a["R"])) if valid else max(points,key=lambda a:(a["R"],a["P"]))
    eligible=[k for k in MODES if summary[k]["TP"]*87>=n*73]
    chosen=(max(eligible,key=lambda k:(summary[k]["P"],summary[k]["R"])) if eligible
            else max(MODES,key=lambda k:(summary[k]["R"],summary[k]["P"])))
    atomic_json(OUT/"frozen.json",dict(summary=summary,chosen=chosen,train_scratch_gt=n,
                inputs="five 640 views, full and four 60% tiles with center ownership; original native OOF nano"))
    print("held-fold operating points",summary,"chosen",chosen,flush=True)


def development():
    spec=read(OUT/"frozen.json")
    mode=spec["chosen"]
    cut=spec["summary"][mode]["threshold"]
    canonical=indexed(read(PHASE))
    warm=load_candidates(ROOT/"experiments/defect_continue_warm_rank/fixed_predictions.json",canonical)
    lookup=manifest()
    assert sha(NATIVE)==SHA
    model=loader(NATIVE)
    rows=[];lo=[]
    for sid,r in canonical.items():
        branches=propose(model,load_image(lookup[sid]["rgb_path"]))
        insects=warm[sid][warm[sid][:,5]==1]
        scr=branches[mode]
        rows.append(dict(image_id=sid,GT=r["GT"],pred=np.concatenate([scr[scr[:,4]>=cut],insects]).tolist()))
        lo.append(dict(image_id=sid,GT=r["GT"],pred=np.concatenate([scr,insects]).tolist()))
    m=metrics(rows)
    assert (m["class"]["1"]["TP"],m["class"]["1"]["FP"])==(123,7)
    atomic_json(OUT/"fixed_predictions.json",rows)
    atomic_json(OUT/"low_predictions.json",lo)
    atomic_json(OUT/"metrics.json",dict(fixed=m,chosen=mode,cut=cut,
                insect_source="existing verified frozen quality-head predictions; no live system speed measured"))
    print("fixed",mode,cut,m["total"],m["class"],flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("action",choices=["calibrate","development","run"])
    args=p.parse_args()
    configure(OUT,"cpu")
    cv2.setNumThreads(1)
    torch.set_num_threads(2)
    if args.action in ("calibrate","run"):
        calibration()
    if args.action in ("development","run"):
        development()
