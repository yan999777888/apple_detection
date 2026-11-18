"""Describe teacher-specific false positives and complementarity without fitting on development labels."""
from collections import Counter,defaultdict
import numpy as np
from class0_patch_refinement import read,atomic_json
from defect_continue_scratch_audit import P,OUT,PHASE,indexed,native,label,inspect
from audit_defect_causes_20261112 import match,ious
from pathlib import Path


def scratch_rows(path,ref,key=None):
    rows=native(read(P/path),ref)
    pred={i:np.asarray(r["raw"][key] if key else r["pred"],np.float64).reshape(-1,6) for i,r in rows.items()}
    return {i:p[p[:,5]==0] for i,p in pred.items()}


def cutoff(rows,ref,mode):
    # Derive a retrospective point exclusively to attribute errors at the specified
    # budget. No numerical cutoff or candidate list is exported for deployment.
    bins=defaultdict(lambda:[0,0])
    for i,a in rows.items():
        m,_=match(a,ref[i]["GT"])
        for j,p in enumerate(a):
            bins[p[4]][int(j not in m)]+=1
    t=f=0;values=[]
    for score,(tp,fp) in sorted(bins.items(),reverse=True):
        t+=tp;f+=fp
        if (mode=="73tp" and t>=73) or (mode=="5fp" and f<=5):
            objective=(-f,t) if mode=="73tp" else (t,-f)
            values.append((objective,score))
    threshold=max(values)[1]
    return {i:a[a[:,4]>=threshold] for i,a in rows.items()}


def details(rows,ref,case):
    out={};hits=set(); fps=[]
    for sid,a in rows.items():
        m,ov=match(a,ref[sid]["GT"])
        local={}
        for j,box in enumerate(a):
            nearest=int(ov[j].argmax()) if ov.shape[1] else -1
            val=float(ov[j,nearest]) if nearest>=0 else 0.
            if j in m:
                gid=label(sid,m[j]);hits.add(gid); reason="TP"
            else:
                gid=None
                c=ref[sid]["GT"][nearest]["class"] if nearest>=0 else None
                reason=("duplicate" if c==0 else "wrong_class") if val>=.5 else ("extent" if val>=.1 else "no_nearby_GT")
                fps.append(dict(id=sid,box=box[:4].tolist(),reason=reason,
                                nearest_gt_class=c,nearest_gt_iou=val,near_miss_04_05=(c==0 and .4<=val<.5)))
            local[j]=(gid,reason)
        out[sid]=local
    return out,hits,fps


def main():
    ref=indexed(read(PHASE))
    original=read(OUT/"original29.json")
    ids={q["reference"]:q["code"] for q in original}
    source={
        "teacher73":("defect_continue_teacher/low_predictions.json","73tp",None),
        "rcnn5fp":("defect_continue_rcnn/transfer_v2/low_predictions.json","5fp",None),
        "DINO5fp":("defect_continue_metric/fresh_low_predictions.json","5fp","0.65"),
        "agreement_fixed":("defect9485_local_fusion/agreement_predictions.json",None,None),
        "gate99_fixed":("defect9485_cross_deploy/gate99/fixed_predictions.json",None,None),
        "teacher_fixed_high_recall":("defect_continue_teacher/R85_predictions.json",None,None),
        "small_fixed":("defect9485_small_v1/fixed_predictions.json",None,None)}
    allrows={};outputs={};allhits={};allmeta={}
    for k,(p,mode,key) in source.items():
        archive=scratch_rows(p,ref,key)
        a=cutoff(archive,ref,mode) if mode else archive
        allrows[k]=a
        meta,hits,events=details(a,ref,mode)
        allhits[k]=hits;allmeta[k]=meta
        outputs[k]=dict(source=p,status="retrospective attribution at fixed numeric TP/FP budget" if mode else "archive fixed",
                        TP=len(hits),FP=len(events),FN=87-len(hits),fp_reason=dict(Counter(v["reason"] for v in events)),
                        close_boundary=sum(v["near_miss_04_05"] for v in events),FP_rows=events,
                        original29_covered=sorted(ids[v] for v in hits if v in ids))
    comparisons={}
    anchor="teacher73"
    for k,b in allrows.items():
        if k==anchor:continue
        both=allhits[k]&allhits[anchor]
        union=allhits[k]|allhits[anchor]
        signals=Counter()
        for sid,pred in allrows[anchor].items():
            ov=ious(pred[:,:4],b[sid][:,:4])
            for j,p in enumerate(pred):
                true=allmeta[anchor][sid][j][0] is not None
                supported=bool(ov.shape[1] and ov[j].max()>=.5)
                signals[("TP" if true else "FP")+("_supported" if supported else "_unsupported")]+=1
        comparisons[k]=dict(common=len(both),union=len(union),teacher_only=len(allhits[anchor]-allhits[k]),
                            other_only=len(allhits[k]-allhits[anchor]),union_original29=sorted(ids[v] for v in union if v in ids),
                            teacher73_same_location_support=dict(signals))
    atomic_json(OUT/"teacher_attribution.json",dict(sources=outputs,teacher73_vs=comparisons,
                warning="Unions/intersections of matched GT identities are diagnostic ceilings, not implementable selectors. Fixed non-retuned insect branch 123TP 7FP. Audit points defined by TP/FP budgets are never used as validated thresholds."))
    print({k:(v["TP"],v["FP"],v["fp_reason"],v["close_boundary"],len(v["original29_covered"])) for k,v in outputs.items()})
    print({k:(v["common"],v["union"],v["teacher73_same_location_support"]) for k,v in comparisons.items()})
if __name__=="__main__":
    main()
