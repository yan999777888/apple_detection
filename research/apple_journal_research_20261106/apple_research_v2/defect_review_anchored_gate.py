"""Leave one reviewed capture second out when testing a localized, prior-preserving correction."""
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
from collections import defaultdict
import hashlib
import numpy as np
import torch
from class0_patch_refinement import ROOT, read, sha, atomic_json, configure
from audit_defect_causes_20261112 import ious
import defect_blind_review_update as base

OUT = base.OUT / "anchored_train_gate"
CONFIG = {"unreviewed_similarity_percentile":99, "kernel_temperature":.12, "ridge":.015,
          "unreviewed_logit_distillation_weight":4., "review_positive_probability":.85,
          "review_negative_probability":.10, "max_abs_logit_change":6.,
          "required_net_new_centers":2, "min_outside_decision_agreement":.995}


def group(name):
    parts = name.split("_")
    return tuple(parts[1:3])


def native_gt(row):
    values = []
    for inst in row["instances"]:
        x,y,w,h=inst["xywh_normalized"]
        values.append([(x-w/2)*1024,(y-h/2)*1024,(x+w/2)*1024,(y+h/2)*1024,inst["class_id"]])
    return np.asarray(values,np.float32).reshape(-1,5)


def logit(p):
    p=np.clip(p,.00001,.99999)
    return np.log(p/(1-p))


def kernel_delta(embedding, prior, class_ids, frame, held, has_review, weak, fit_class):
    # Fit the class neighborhood and its similarity cutoff on the other acquisition seconds.
    source=(~held) & (class_ids==fit_class)
    donors=np.flatnonzero(source & (weak>=0))
    anchor=np.flatnonzero(source & ~has_review)
    query=np.flatnonzero(held & (class_ids==fit_class))
    if not len(query) or not len(donors) or not len(anchor):
        return query, np.zeros(len(query)), {}
    center=embedding[anchor].mean(0)
    normalized=embedding-center
    normalized/=np.maximum(np.linalg.norm(normalized,axis=1,keepdims=True),1e-5)
    sim=(normalized@normalized[donors].T).clip(-1,1)
    limit=float(np.quantile(sim[anchor].max(1), CONFIG["unreviewed_similarity_percentile"]/100))
    basis=np.exp((sim-1)/CONFIG["kernel_temperature"]).astype(np.float64)
    basis[sim.max(1)<limit]=0.
    r=basis[donors]
    u=basis[anchor]
    desired=logit(np.where(weak[donors]>0,CONFIG["review_positive_probability"],
                           CONFIG["review_negative_probability"]))-logit(prior[donors])
    matrix=(r.T@r)/len(r) + CONFIG["unreviewed_logit_distillation_weight"]*(u.T@u)/len(u)
    matrix+=np.eye(len(r))*CONFIG["ridge"]
    weights=np.linalg.solve(matrix,r.T@desired/len(r))
    return query, np.clip(basis[query]@weights,-CONFIG["max_abs_logit_change"],CONFIG["max_abs_logit_change"]),dict(
        source_donors=len(donors),unreviewed_distillation_anchors=len(anchor),threshold=limit)


def run():
    torch.set_num_threads(1)
    if not (OUT/"protocol.json").exists():
        atomic_json(OUT/"protocol.json",dict(config=CONFIG,train_only=True,run_precedes_development=True,
           training_export=read(base.OUT/"frozen_review.json")["export_sha256"],source_sha=sha(__file__)))
    spec=read(OUT/"protocol.json")
    assert spec["config"]==CONFIG and spec["source_sha"]==sha(__file__)
    data=torch.load(base.OUT/"train_features.pt",map_location="cpu",weights_only=True)
    rows=base.replay.original.samples("train")
    assert len(rows)==596 and sum(len(r["instances"]) for r in rows)==1276
    frozen=read(base.OUT/"frozen_review.json")
    assert data["export_sha256"]==frozen["export_sha256"]
    digest=hashlib.sha256()
    p=data["p"].numpy()
    cls=p[:,5].astype(int)
    frame=data["frame"].numpy().astype(int)
    z=data["feats"][:,:-4].float().numpy()
    weak=data["weak"].numpy()
    cases={x["image"]:x for x in frozen["cases"]}
    excluded=set()
    seconds=defaultdict(list)
    for i,r in enumerate(rows):
        seconds[group(r["image_id"])].append(i)
        if r["image_id"] in cases:
            excluded.add(i)
    # Distill on unreviewed samples. Every frame from a held second is excluded from all fitting.
    reviewed_images=np.isin(frame,list(excluded))
    key_groups=sorted({group(k) for k in cases})
    operating=np.asarray([.40,.34])[cls]
    pre=p[:,4]>=operating
    scores=p[:,4].copy()
    held_all=np.zeros(len(p),bool)
    total_outside=0
    changed_outside=0
    old_gt_lost=0
    old_gt_hits=0
    out=[]
    for g in key_groups:
        held=np.isin(frame,seconds[g])
        assert not np.any(held_all&held)
        held_all|=held
        d=np.zeros(len(p))
        notes=[]
        for c in (0,1):
            idx,value,note=kernel_delta(z,p[:,4],cls,frame,held,reviewed_images,weak,c)
            d[idx]=value
            notes.append(note)
        moved=held&(d!=0.)
        scores[moved]=1/(1+np.exp(-(logit(p[moved,4])+d[moved])))
        after=scores>=operating
        for i in seconds[g]:
            idx=np.flatnonzero(frame==i)
            local=p[idx]
            gt=native_gt(rows[i])
            if len(gt):
                ov=ious(local[:,:4],gt[:,:4])
                same=cls[idx,None]==gt[None,:,4]
                before_hit=((pre[idx,None]&same)&(ov>=.5)).any(0)
                after_hit=((after[idx,None]&same)&(ov>=.5)).any(0)
                old_gt_lost+=int((before_hit&~after_hit).sum())
                old_gt_hits+=int(before_hit.sum())
            name=rows[i]["image_id"]
            outside=np.ones(len(idx),bool)
            if name in cases:
                case=cases[name]
                eligible,_,_=base.reviewed_region(local,gt,case)
                pos=case["response"]>=0
                focus=eligible & ((local[:,5]==case["response"]) if pos else True)
                before=bool(np.any(pre[idx]&focus))
                later=bool(np.any(after[idx]&focus))
                out.append(dict(case=case["case"],target_class=case["response"],before=before,
                       after=later,source_neighbors=notes,eligible_rows=int(focus.sum())))
                outside&=~eligible
            total_outside+=int(outside.sum())
            changed_outside+=int((outside&(pre[idx]!=after[idx])).sum())
    plus=sum(int(r["after"])-int(r["before"]) for r in out if r["target_class"]>=0)
    more_background=sum(int(r["after"]) for r in out if r["target_class"]<0 and not r["before"])
    agree=1-changed_outside/total_outside
    passes=(plus>=CONFIG["required_net_new_centers"] and more_background==0 and
            agree>=CONFIG["min_outside_decision_agreement"] and old_gt_lost==0)
    assert len(out)==20 and len(key_groups)==20
    digest.update(scores[held_all].tobytes())
    report=dict(train_seconds=len(key_groups),train_frames_held=sum(len(seconds[g]) for g in key_groups),
             accepted_sites=len(out),holdout_positive_before=sum(r["before"] for r in out if r["target_class"]>=0),
             holdout_positive_after=sum(r["after"] for r in out if r["target_class"]>=0),
             holdout_ordinary_before=sum(r["before"] for r in out if r["target_class"]<0),
             holdout_ordinary_after=sum(r["after"] for r in out if r["target_class"]<0),
             new_ordinary_false_alerts=more_background,net_new_positive_centers=plus,unchanged_outside_fraction=agree,
             changed_other_candidate_decisions=changed_outside,other_candidate_rows=total_outside,
             native_previously_covered=old_gt_hits,native_previously_covered_lost=old_gt_lost,
             output_digest=digest.hexdigest(),gate_passed=bool(passes),weak_cases=out,
             scope="Leave-one-second-out for review adaptation only. Frozen base detector previously saw original 596 training images.")
    atomic_json(OUT/"result.json",report)
    print({k:v for k,v in report.items() if k!="weak_cases"},flush=True)


if __name__=="__main__":
    configure(OUT,"cpu")
    run()
