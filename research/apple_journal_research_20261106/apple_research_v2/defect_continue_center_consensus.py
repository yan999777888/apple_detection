"""CPU-only same-frame single-view consensus for Center and out-of-fold Nano."""
import hashlib
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from class0_patch_refinement import ROOT, OLD, configure, read, sha, atomic_json
from defect_continue_center import OUT as CENTER, measured
from defect9485_scene import policy
from defect_algorithm_modes_20261112 import nms
from audit_defect_causes_20261112 import ious,match
from run_defect_zero_train_20261112 import metrics

OUT = ROOT / "experiments/defect_continue_center_consensus"
NAMES = ("geometry_consensus","learned_consensus")
FLOOR=.003


def cleaned(pred):
    p=np.asarray(pred,np.float32).reshape(-1,6)
    return nms(p[p[:,4]>=FLOOR])


def logit(x):
    x=np.clip(x,.001,.999)
    return np.log(x/(1-x))


def prepare(center,nano):
    a,b=cleaned(center),cleaned(nano)
    ov=ious(a[:,:4],b[:,:4])
    matched=np.zeros(len(a),bool)
    used=np.zeros(len(b),bool)
    pairs=[]
    if len(a) and len(b):
        wh_a=np.maximum(1.,a[:,2:4]-a[:,:2])
        wh_b=np.maximum(1.,b[:,2:4]-b[:,:2])
        offset=((a[:,None,:2]+a[:,None,2:4])-(b[None,:,:2]+b[None,:,2:4]))*.5
        d=np.linalg.norm(offset/np.maximum(wh_a[:,None],wh_b[None]),axis=2)
        candidates=(a[:,None,5]==b[None,:,5]) & ((ov>=.32)|((ov>=.18)&(d<=.22)))
        ii,jj=np.where(candidates)
        sim=ov[ii,jj]*np.maximum(.0001,a[ii,4]*b[jj,4])**.20
        for z in np.argsort(-sim,kind="stable"):
            i,j=ii[z],jj[z]
            if not matched[i] and not used[j]:
                pairs.append((i,j))
                matched[i]=True
                used[j]=True
    pairs.extend((i,-1) for i in np.flatnonzero(~matched))
    pairs.extend((-1,j) for j in np.flatnonzero(~used))
    boxes,features=[],[]
    for i,j in pairs:
        aa=a[i] if i>=0 else None
        bb=b[j] if j>=0 else None
        ca=float(aa[4]) if i>=0 else 0.
        cb=float(bb[4]) if j>=0 else 0.
        both=i>=0 and j>=0
        cls=int(aa[5] if i>=0 else bb[5])
        if both:
            # Arithmetic vote scales with agreement; uncertain pairs retain the stronger box.
            w=ca/(ca+cb+1e-12)
            voted=w*aa[:4]+(1-w)*bb[:4]
            alpha=min(1.,ov[i,j]/.58)
            coords=alpha*voted+(1-alpha)*(aa[:4] if ca>cb else bb[:4])
            overlap=ov[i,j]
        else:
            coords=(aa if i>=0 else bb)[:4]
            overlap=0.
        other_c=0. if j<0 or not np.any(a[:,5]!=cls) else (ov[a[:,5]!=cls,j]**2*a[a[:,5]!=cls,4]).max()
        other_n=0. if i<0 or not np.any(b[:,5]!=cls) else (ov[i,b[:,5]!=cls]**2*b[b[:,5]!=cls,4]).max()
        aligned=np.sqrt(ca*cb)
        score=.60*max(ca,cb)+.4*aligned if both else .50*max(ca,cb)
        extent=np.maximum(1.,coords[2:]-coords[:2])
        boxes.append([*coords,score,cls])
        features.append([logit(ca) if ca else 0.,logit(cb) if cb else 0.,i>=0,j>=0,both,
                         overlap,float(other_c),float(other_n),np.log(extent[0]/48),np.log(extent[1]/48)])
    return np.asarray(boxes,np.float32).reshape(-1,6),np.asarray(features,np.float64).reshape(-1,10)


def learn(rows):
    xs=[[] for _ in range(2)]
    ys=[[] for _ in range(2)]
    for row in rows:
        b,x=row["data"]
        pairs,_=match(b,row["GT"])
        label=np.asarray([int(i in pairs) for i in range(len(b))])
        for c in (0,1):
            m=b[:,5]==c
            xs[c].extend(x[m])
            ys[c].extend(label[m])
    out=[]
    for x,y in zip(xs,ys):
        model=LogisticRegression(C=.25,max_iter=400,solver="lbfgs").fit(np.array(x),np.array(y))
        out.append(dict(w=model.coef_[0].tolist(),b=float(model.intercept_[0])))
    return out


def rank(item,coeff=None):
    b,x=item["data"]
    b=b.copy()
    if coeff is not None:
        for c,p in enumerate(coeff):
            m=b[:,5]==c
            z=x[m]@np.array(p["w"])+p["b"]
            b[m,4]=1/(1+np.exp(-np.clip(z,-30,30)))
    return nms(b[b[:,4]>=.001])


def index(rows):
    res={r["image_id"]:r for r in rows}
    assert len(res)==len(rows)
    return res


def attach(crows,nrows):
    out=[]
    for r in crows:
        other=nrows[r["image_id"]]
        assert r["GT"]==other["GT"]
        out.append(dict(image_id=r["image_id"],GT=r["GT"],data=prepare(r["pred"],other["sources"]["o2o"])))
    return out


def source_folds():
    allrows={}
    receipt=[]
    for i in range(3):
        fold=OLD/f"oof_v2/fold{i}"
        train={Path(v).name for v in (fold/"train.txt").read_text("utf-8-sig").splitlines()}
        held={Path(v).name for v in (fold/"val.txt").read_text("utf-8-sig").splitlines()}
        src=OLD/f"all_mode_OOF_calibration/fold{i}_sources.json"
        data=read(src)
        rows=index(data["rows"])
        cp=ROOT/f"runs/defect_oof3_fold{i}_e40_20261112_v1/weights/last.pt"
        assert set(rows)==held and not (train&held) and len(train|held)==596
        assert not (allrows.keys()&rows.keys()) and sha(cp)==data["checkpoint_sha256"]
        allrows.update(rows)
        receipt.append(dict(fold=i,oof_sha=sha(src),weights_sha=sha(cp),
                            held=len(held),train=len(train),held_vs_train_overlap=0))
    assert len(allrows)==596
    return allrows,receipt


def main():
    configure(OUT,"cpu")
    f=read(CENTER/"frozen.json")
    cfile=CENTER/f"calib_{f['epoch']:02}_low.json"
    crows=read(cfile)
    assert len(crows)==106 and set(f["calib_ids"])==set(index(crows))
    nr,receipt=source_folds()
    calibration=attach(crows,nr)
    def group(r):
        _,date,clock,*other=r["image_id"].split("_")
        return date+clock[:4]
    groups=sorted({group(r) for r in calibration},
                  key=lambda g:hashlib.sha256(("center-nano:"+g).encode()).hexdigest())
    fold_ids={g:i%3 for i,g in enumerate(groups)}
    raw={name:[] for name in NAMES}
    for fold in range(3):
        fit=[r for r in calibration if fold_ids[group(r)]!=fold]
        model=learn(fit)
        for r in calibration:
            if fold_ids[group(r)]==fold:
                for key,p in zip(NAMES,(None,model)):
                    raw[key].append(dict(image_id=r["image_id"],GT=r["GT"],oof=rank(r,p)))
    points={k:policy(rows) for k,rows in raw.items()}
    def utility(p):
        return min(p["P"]/.94,p["R"]/.85),p["P"]*p["R"]
    primary=max(NAMES,key=lambda k:utility(points[k]["joint"]))
    weights=learn(calibration)
    atomic_json(OUT/"frozen.json",dict(primary=primary,calibration=points,coefficients=weights,
                detector="COCO-init original Nano trained last40, o2o single original image only",
                detector_receipts=receipt,calib_counts=[len(raw[k]) for k in NAMES],center_calib_sha=sha(cfile),
                meta_folds=fold_ids,center_weights_sha=f["weights_sha"],uses_small=False,
                proxy_note="Grouped held-out Nano folds used fewer training images than deployed 596-image Nano; Center checkpoint was selected on the 106, so reported calibration is a selection estimate."))
    cdev=CENTER/"fresh_low_predictions.json"
    ndev=OLD/"phase1/source_predictions.json"
    dev=attach(read(cdev),index(read(ndev)))
    assert len(dev)==113 and sum(len(r["GT"]) for r in dev)==230
    summary={}
    for k in NAMES:
        coeff=None if k=="geometry_consensus" else weights
        low=[dict(image_id=r["image_id"],GT=r["GT"],pred=rank(r,coeff).tolist()) for r in dev]
        atomic_json(OUT/f"{k}_low.json",low)
        summary[k]={}
        for mode,p in points[k].items():
            rows,m=measured(low,p)
            atomic_json(OUT/f"{k}_{mode}.json",rows)
            summary[k][mode]=m
    atomic_json(OUT/"metrics.json",dict(primary=primary,scores=summary,center_dev_sha=sha(cdev),nano_dev_sha=sha(ndev)))
    print("Frozen on 106",primary,points[primary],flush=True)
    print("Development full cache comparison",{n:{k:v["total"] for k,v in s.items()} for n,s in summary.items()},flush=True)


if __name__=="__main__":
    main()
