"""Learn proposal arbitration from paired held-out detections, then test Small transfer."""

from collections import Counter
from time import perf_counter
import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, ExtraTreesRegressor

from class0_patch_refinement import ROOT, OLD, atomic_json, configure, load_image, manifest, read, sha, appearance
from audit_defect_causes_20261112 import ious, match
from compare_defect_target9485 import BASELINES, PHASE, inspect, indexed, load_candidates, delta
from defect_continue_teacher_proxy import OUT
from defect_coordinate_vote_20261118 import vote
from defect9485_set import operating_points, cut

FLOOR = 0.008


def matrix(raw):
    p = np.asarray(raw, np.float32).reshape(-1, 6)
    return p[p[:, 4] >= FLOOR]


def describe(im, seed):
    # Compare the inside of a defect to its immediate exterior; local changes persist
    # across capture-day shifts in fruit color and exposure.
    import cv2
    l, t, r, b = seed[:4]
    w, h = max(3., r-l), max(3., b-t)
    crop = im[max(0, int(t-.4*h)):min(im.shape[0],int(b+.4*h)+1),
              max(0, int(l-.4*w)):min(im.shape[1],int(r+.4*w)+1)]
    if not crop.size:
        crop = np.zeros((2, 2, 3), np.uint8)
    lab = cv2.cvtColor(cv2.resize(crop,(24,24)), cv2.COLOR_BGR2LAB).astype(float)
    core = lab[5:-5,5:-5]
    outer = lab[np.maximum.outer(abs(np.arange(24)-11.5),abs(np.arange(24)-11.5)) >= 8]
    med = np.median(outer,axis=0)
    scale = np.maximum(np.median(abs(outer-med),axis=0),4.)
    dev = (core-med)/scale
    return np.r_[appearance(im,seed[:4]),np.percentile(dev,[10,50,90],axis=(0,1)).ravel(),
                 np.percentile(np.linalg.norm(dev,axis=2),[50,85,95]), (dev[...,0]<-1.5).mean(),
                 (dev[...,1]>1.5).mean(), (dev[...,2]<-1.5).mean()]


def build(raw, second, im):
    main = vote(raw, 0.5, [FLOOR, FLOOR])
    aux = matrix(second)
    pools = [main, aux, matrix(raw["o2o"]), matrix(raw["flip"]), matrix(raw["o2m"])]
    pair = []
    feats = []
    for c in (0,1):
        anchors = [p[p[:,5] == c] for p in (main,aux)]
        merged = np.concatenate(anchors)
        if not len(merged):
            continue
        order = np.argsort(-merged[:,4],kind="stable")
        overlap = ious(merged[:,:4],merged[:,:4])
        used = np.zeros(len(merged),bool)
        chosen = []
        for i in order:
            if not used[i]:
                chosen.append(i)
                used |= overlap[i] >= .46
        seeds = merged[chosen].copy()
        xywh = np.maximum(seeds[:,2:4]-seeds[:,:2],2)
        desc = []
        nn = []
        for pool_i, pool in enumerate(pools):
            for same in (True,False):
                a = pool[pool[:,5] == (c if same else 1-c)]
                v = np.zeros((len(seeds),8),np.float32)
                coord = np.zeros((len(seeds),4))
                if len(a):
                    ov = ious(seeds[:,:4],a[:,:4])
                    strength = ov**2 * a[None,:,4]
                    j = np.argmax(strength,axis=1)
                    idx = np.arange(len(seeds))
                    support = ov[idx,j]
                    near = support >= .18
                    v[near,0] = a[j[near],4]
                    v[near,1] = support[near]
                    v[:,2] = strength[idx,j]
                    v[:,3] = ((ov>=.2) & (a[:,4][None,:] >= .04)).sum(1)
                    coord[near] = a[j[near],:4]
                    v[near,4:] = ((coord[near]-seeds[near,:4])/np.tile(xywh[near],(1,2))).clip(-2,2)
                desc.append(v)
                if same and pool_i < 2:
                    nn.append((coord,v[:,0],v[:,1]))
        loc = seeds.copy()
        for i in range(len(seeds)):
            supported = [(q[i],sc[i],ov[i]) for q,sc,ov in nn if ov[i] >= .4]
            if supported:
                loc[i,:4] = np.average([u[0] for u in supported],axis=0,
                                        weights=[np.sqrt(u[1])*u[2] for u in supported])
        wh = np.maximum(loc[:,2:4]-loc[:,:2],1)
        ctx = np.c_[np.full(len(loc),c),np.log(wh),np.log(wh[:,0]/wh[:,1]),
                    np.maximum(nn[0][1],nn[1][1]),np.minimum(nn[0][1],nn[1][1])]
        feats.extend(np.column_stack([ctx,*desc,np.stack([describe(im,b) for b in loc])]))
        pair.extend(loc)
    return np.asarray(pair,np.float32).reshape(-1,6),np.asarray(feats,np.float32)


def targets(row):
    b,x = row["data"]
    grade = np.zeros(len(b))
    offset = np.zeros((len(b),4))
    if len(b) and row["GT"]:
        g = np.asarray([v["box"] for v in row["GT"]])
        cls = np.asarray([v["class"] for v in row["GT"]])
        ov = ious(b[:,:4],g)*(b[:,5,None] == cls[None,:])
        best = ov.argmax(1)
        grade = ov[np.arange(len(b)),best]
        wh = np.maximum(b[:,2:4]-b[:,:2],2)
        offset = ((g[best]-b[:,:4])/np.tile(wh,(1,2))).clip(-.6,.6)
    return grade,offset


def fit(rows):
    x = np.concatenate([r["data"][1] for r in rows if len(r["data"][0])])
    y = [targets(r) for r in rows if len(r["data"][0])]
    grade = np.concatenate([g for g,d in y])
    offset = np.concatenate([d for g,d in y])
    gate = HistGradientBoostingClassifier(max_iter=135,max_leaf_nodes=12,min_samples_leaf=28,
                                         l2_regularization=5,early_stopping=False,random_state=1989)
    gate.fit(x,grade>=.5)
    valid = grade >= .29
    extent = ExtraTreesRegressor(n_estimators=40,min_samples_leaf=17,max_features=.65,
                                  n_jobs=1,random_state=3117).fit(x[valid],offset[valid])
    return gate,extent


def infer(row,model):
    b,x = row["data"]
    if not len(b):
        return b
    p = b.copy()
    p[:,4] = model[0].predict_proba(x)[:,1]
    wh = np.maximum(b[:,2:4]-b[:,:2],2)
    p[:,:4] += .65*model[1].predict(x)*np.tile(wh,(1,2))
    p[:,:4] = p[:,:4].clip(0,1024)
    ov = ious(p[:,:4],p[:,:4])
    used = np.zeros(len(p),bool)
    keep = []
    for i in np.argsort(-p[:,4],kind="stable"):
        if not used[i]:
            keep.append(i)
            used |= (p[:,5] == p[i,5]) & (ov[i] >= .55)
    return p[keep]


def main():
    import cv2
    import torch
    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    lookup = manifest()
    train = []
    origin = {}
    for f in range(3):
        path = OUT / f"proxy_fold{f}.json"
        proxy = read(path)
        orig = OLD / f"all_mode_OOF_calibration/fold{f}_sources.json"
        assert proxy["source_sha256"] == sha(orig)
        origin[str(f)] = dict(weights=proxy["weight_sha256"],proxy_sha256=sha(path),native_sha256=proxy["native_sha256"])
        for r in read(orig)["rows"]:
            sid = r["image_id"]
            assert lookup[sid]["partition"] == "train"
            data = build(r["sources"],proxy["pred"][sid],load_image(lookup[sid]["rgb_path"]))
            train.append(dict(image_id=sid,GT=r["GT"],fold=f,data=data))
        print("features/fold", f, flush=True)
    assert len(train) == len({r["image_id"] for r in train}) == 596
    for f in range(3):
        model = fit([r for r in train if r["fold"] != f])
        for r in train:
            if r["fold"] == f:
                r["raw"] = infer(r,model)
    profiles = operating_points(train)
    model = fit(train)
    joblib.dump(model,OUT/"head.joblib",compress=3)
    atomic_json(OUT/"frozen.json",dict(OOF=profiles,primary="tradeoff",source=origin,
                                       script_sha256=sha(__file__),head_sha256=sha(OUT/"head.joblib"),
                                       secondary_training="held-fold nano epoch30 768 proxy",secondary_dev="YOLO26s epoch36 768 actual",
                                       selection="trained on 596 only; classwise cross-fit operating profiles; no development fitting"))
    print("frozen",profiles,flush=True)
    canonical = indexed(read(PHASE))
    actual = ROOT/"experiments/defect9485_small_v1/low_predictions.json"
    second = load_candidates(actual,canonical)
    assert len(canonical) == 113 and sum(len(r["GT"]) for r in canonical.values()) == 230
    outputs = {}
    times = []
    for sid,r in canonical.items():
        assert lookup[sid]["partition"] == "dev"
        im = load_image(lookup[sid]["rgb_path"])
        t = perf_counter()
        outputs[sid] = infer(dict(data=build(r["sources"],second[sid],im)),model)
        times.append((perf_counter()-t)*1000)
    def save(name,pred):
        atomic_json(OUT/f"{name}_predictions.json",[dict(image_id=sid,GT=r["GT"],pred=pred[sid].tolist()) for sid,r in canonical.items()])
    save("low",outputs)
    report = {}
    base = inspect(load_candidates(BASELINES["original_balanced"],canonical),canonical)[1]
    joined = ROOT/"experiments/defect9485_cross/yolo26s/one_fixed_plus_other_support_and_frozen_set_predictions.json"
    both = inspect(load_candidates(joined,canonical),canonical)[1]
    for name,cfg in profiles.items():
        pred = {k:cut(p,cfg["thresholds"]) for k,p in outputs.items()}
        stats,hits,fps,_ = inspect(pred,canonical)
        save(name,pred)
        report[name] = dict(fixed=stats,fp_reasons=fps,vs_original=delta(hits,base,canonical),vs_old_cross=delta(hits,both,canonical))
    curves = inspect(outputs,canonical)[3]
    atomic_json(OUT/"report.json",dict(frozen=report,dev_full_PR=curves,frozen_sha256=sha(OUT/"frozen.json"),
                                      secondary_sha256=sha(actual),CPU_gate_only_ms=dict(mean=np.mean(times),p95=np.percentile(times,95)),
                                      candidates=sum(map(len,outputs.values())),development_total=230))
    print("development", {k:v["fixed"]["total"] for k,v in report.items()},curves["classwise_overall"],flush=True)


if __name__ == "__main__":
    configure(OUT,"cpu")
    main()
