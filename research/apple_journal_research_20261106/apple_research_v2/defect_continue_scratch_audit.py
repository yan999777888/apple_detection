"""Retrospective scratch-specific evidence with the archived insect deployment point locked."""
from collections import Counter,defaultdict
from itertools import combinations
from pathlib import Path
import json
import numpy as np

from compare_defect_target9485 import PHASE,ROOT,read,indexed
from audit_defect_target9485 import oracle
from audit_defect_causes_20261112 import match,ious
from joint_state_v1 import sha,atomic_json,configure

P=ROOT/"experiments";OUT=P/"defect_continue_scratch_audit"
WARM=P/"defect_continue_warm_rank/fixed_predictions.json"
EXPECTED=(123,7,20)
CLASSES=(87,143)


def overall(a,b):
    t=a+EXPECTED[0];f=b+EXPECTED[1]
    return dict(TP=t,FP=f,FN=230-t,P=t/max(1,t+f),R=t/230)


def category(p):
    n=p.name.lower()
    if "low" in n or "/train/G" in p.as_posix():
        return "low"
    return "fixed"


def label(sid,g):
    return f"{sid}#{g}"


def native(records,ref):
    r=indexed(records)
    if r.keys()!=ref.keys():
        raise ValueError("image universe differs")
    for sid,row in r.items():
        if "GT" in row:
            want=ref[sid]["GT"]
            got=row["GT"]
            if [v["class"] for v in got]!=[v["class"] for v in want] or not np.allclose(
                    [v["box"] for v in got],[v["box"] for v in want],atol=.00001,rtol=0):
                raise ValueError("canonical reference differs")
    return r


def families(path):
    if "rcnn" in path: return "R-CNN"
    if "metric" in path or "repr" in path: return "crop reclassifier"
    if "warm_rank" in path: return "Small quality head"
    if "continue_rank" in path: return "anchor-free rank"
    if "center" in path: return "center detector"
    if "teacher" in path: return "teacher"
    if "cross" in path or "fusion" in path or "consensus" in path: return "fusion"
    if "small" in path or "continue_native" in path or "continue_public" in path: return "Small detector"
    return path.split("/")[0]


def inspect(result,ref,old,stream):
    ledger=set(); covered=set(); fp_types=Counter(); missed_types=Counter(); bins=defaultdict(lambda:[0,0])
    misses={}
    for sid,a in result.items():
        a=np.asarray(a,np.float64).reshape(-1,6)
        assert np.isfinite(a).all() and (a[:,2:4]>=a[:,:2]).all() and np.isin(a[:,5],[0,1]).all()
        g=ref[sid]["GT"]
        g0=[i for i,t in enumerate(g) if t["class"]==0]
        s=a[a[:,5]==0]
        hits,ov=match(s,g)
        got={label(sid,j) for j in hits.values()}
        ledger.update(got)
        if stream=="low":
            covered.update(label(sid,j) for j in oracle(s,g))
        for i,p in enumerate(s):
            matched=i in hits
            bins[float(p[4])][0 if matched else 1]+=1
            if matched: continue
            if len(g):
                j=ov[i].argmax()
                v=ov[i,j]
                reason=("duplicate" if g[j]["class"]==0 else "wrong_class") if v>=.5 else ("extent" if v>=.1 else "no_nearby_GT")
            else:
                reason="no_nearby_GT"
            fp_types[reason]+=1
        for j in g0:
            if label(sid,j) in got:continue
            can=bool(len(s) and ov[:,j].max()>=.5)
            near=bool(len(s) and ov[:,j].max()>=.20)
            reason="assignment_to_neighbour" if can else ("partial_extent" if near else "no_nearby_class0_box")
            missed_types[reason]+=1
            misses[label(sid,j)]=reason
    tp=len(ledger); fp=sum(x[1] for x in bins.values())
    base=dict(TP=tp,FP=fp,FN=87-tp,combined=overall(tp,fp),hits=sorted(ledger),
              hits_original29=sorted(ledger&old),fp_types=dict(fp_types),miss_reason=misses)
    if stream=="low":
        t=f=0; prefixes=[(0,0)]
        for _,(x,y) in sorted(bins.items(),reverse=True):
            t+=x;f+=y
            prefixes.append((t,f))
        def point(i):
            if i is None:return None
            return dict(scratch_TP=i[0],scratch_FP=i[1],combined=overall(*i))
        def pick(points,func):
            return point(max(points,key=func,default=None))
        base.update(oracle_GTs=sorted(covered),
          max_P_at_R85=pick([(a,b) for a,b in prefixes if a>=73],lambda p:(overall(*p)["P"],p[0])),
          max_R_at_P94=pick([(a,b) for a,b in prefixes if overall(a,b)["P"]>=.94],lambda p:(p[0],-p[1])),
          min_FP_at_73=pick([(a,b) for a,b in prefixes if a>=73],lambda p:(-p[1],p[0])),
          max_TP_at_5FP=pick([(a,b) for a,b in prefixes if b<=5],lambda p:(p[0],-p[1])),
          meets=any(a>=73 and overall(a,b)["P"]>=.94 for a,b in prefixes))
    return base


def discover(ref,old):
    selected=sorted(set(p for p in P.rglob("*.json") if (p.parts[len(P.parts)].startswith(("defect","class0","strong_defect")) and
           ("pred" in p.name.lower() or "_low" in p.name or p.parent.name=="defect_continue_center_consensus") and
           p.stat().st_size>1000 and "oof_v2" not in p.parts and not p.is_relative_to(OUT) and
           not any(v.startswith(("example","calib_","proxy_fold")) for v in p.parts))))
    modes={};skip={}
    for p in selected:
        name=p.relative_to(P).as_posix()
        rows=read(p)
        if not isinstance(rows,list) or not rows or name.endswith("all_predictions.json") or not isinstance(rows[0],dict) or "image_id" not in rows[0] or len(rows)!=113:
            skip[name]="not 113-row defect predictions";continue
        try:
            table=native(rows,ref)
        except ValueError as e:
            skip[name]=str(e);continue
        if "pred" in rows[0]:
            branches={"":{i:r["pred"] for i,r in table.items()}}
        elif "raw" in rows[0] and isinstance(rows[0]["raw"],dict):
            branches={f":{k}":{i:r["raw"][k] for i,r in table.items()} for k in rows[0]["raw"]}
        elif "sources" in rows[0]:
            branches={f":{k}":{i:r["sources"][k] for i,r in table.items()} for k in rows[0]["sources"]}
        else:
            skip[name]="no supported candidate matrix";continue
        for sub,preds in branches.items():
            kind="low" if sub or category(p)=="low" else "fixed"
            metric=inspect(preds,ref,old,kind)
            modes[name+sub]=dict(kind=kind,family=families(name),file_sha=sha(p),**metric)
    return modes,skip


def totals_only(ref):
    # Aggregate-only older ablations. No attempt is made to recreate absent boxes.
    dirs=["defect_coordinate_vote_20261118","defect_weight_average_20261118","defect_score_transfer_20261118",
          "defect_consensus_20261118","defect_four_orientations_20261118","defect_refine_20261112_v1",
          "defect_algorithm_steps_20261112_v2/phase1","defect_algorithm_steps_20261112_v1/phase1"]
    found={}
    def walk(obj,key):
        if not isinstance(obj,dict):return
        cls=obj.get("class");t=obj.get("total")
        if isinstance(cls,dict) and isinstance(t,dict) and t.get("TP",-1)+t.get("FN",-1)==230 and cls.get("0",{}).get("GT")==87 and cls.get("1",{}).get("GT")==143:
            c=cls["0"];found[key]=dict(TP=c["TP"],FP=c["FP"],FN=c["GT"]-c["TP"],combined=overall(c["TP"],c["FP"]))
            return
        for k,v in obj.items():
            if isinstance(v,dict) and not any(z in k.lower() for z in ("threshold","oracle","sweep","scan","grid","oof","diagnostic","curve","speed","ap","train")):
                walk(v,key+"/"+k)
    for d in dirs:
        path=P/d/"results.json"
        if path.exists():walk(read(path),d)
    return found


def main():
    configure(OUT,"cpu")
    ref=indexed(read(PHASE))
    assert len(ref)==113 and sum(len(x["GT"]) for x in ref.values())==230
    insect=native(read(WARM),ref)
    it=ip=0
    for i,r in insect.items():
        a=np.array(r["pred"]).reshape(-1,6);a=a[a[:,5]==1]
        hit,_=match(a,ref[i]["GT"]);it+=len(hit);ip+=len(a)-len(hit)
    assert (it,ip,143-it)==EXPECTED
    cases=read(P/"class0_visual_review_v1/audit.json")["cases"]["miss"]
    old={label(q["image"],q["index"]) for q in cases}
    assert len(old)==29
    modes,excluded=discover(ref,old)
    low={k:v for k,v in modes.items() if v["kind"]=="low"}
    fixed={k:v for k,v in modes.items() if v["kind"]=="fixed"}
    fixed_pairs=[]
    for (ka,a),(kb,b) in combinations(fixed.items(),2):
        if a["family"]==b["family"]:continue
        A=set(a["hits"]);B=set(b["hits"])
        fixed_pairs.append(dict(a=ka,b=kb,intersection=len(A&B),union=len(A|B),only_a=len(A-B),only_b=len(B-A),
                                union_original29=len((A|B)&old),fp_a=a["FP"],fp_b=b["FP"]))
    low_pair=[]
    for (ka,a),(kb,b) in combinations(low.items(),2):
        if a["family"]==b["family"]:continue
        A=set(a["oracle_GTs"]);B=set(b["oracle_GTs"])
        low_pair.append(dict(a=ka,b=kb,union=len(A|B),intersection=len(A&B),only_a=len(A-B),only_b=len(B-A),
                             original29_union=len((A|B)&old)))
    combined=set().union(*(set(v["oracle_GTs"]) for v in low.values()))
    allgt={label(k,j) for k,r in ref.items() for j,g in enumerate(r["GT"]) if g["class"]==0}
    by29=[]
    for q in cases:
        gid=label(q["image"],q["index"])
        by29.append(dict(code=q["code"],reference=gid,original_cause=q["category"],
                         fixed_hits=[k for k,v in fixed.items() if gid in v["hits"]],
                         low_geometry_hits=[k for k,v in low.items() if gid in v["oracle_GTs"]]))
    summary=dict(reference_sha=sha(PHASE),warm_insect_sha=sha(WARM),insect=EXPECTED,
                 files_considered=len(set(k.split(":")[0] for k in modes)),fixed_sets=len(fixed),low_sets=len(low),
                 original_29=len(old),fixed_meets=[k for k,v in fixed.items() if v["combined"]["P"]>=.94 and v["combined"]["R"]>=.85],
                 low_oracle_meets=[k for k,v in low.items() if v["meets"]],
                 any_geometry_coverage=len(combined),never_geometric_hits=sorted(allgt-combined),
                 no_fixed_model_recovers_original29=[r["code"] for r in by29 if not r["fixed_hits"]],
                 no_low_model_recovers_original29=[r["code"] for r in by29 if not r["low_geometry_hits"]])
    atomic_json(OUT/"summary.json",summary)
    atomic_json(OUT/"fixed.json",fixed)
    atomic_json(OUT/"candidate_frontiers.json",low)
    atomic_json(OUT/"aggregate_only_older_results.json",totals_only(ref))
    atomic_json(OUT/"pair_diagnostics.json",dict(fixed=fixed_pairs,low_geometry=low_pair))
    atomic_json(OUT/"original29.json",by29)
    atomic_json(OUT/"file_exclusions.json",excluded)
    print(summary,flush=True)
    print("fixed at R85",sorted([(v["combined"]["P"],k,v["TP"],v["FP"]) for k,v in fixed.items() if v["TP"]>=73],reverse=True)[:4])
    print("fixed at P94",sorted([(v["combined"]["R"],k,v["TP"],v["FP"]) for k,v in fixed.items() if v["combined"]["P"]>=.94],reverse=True)[:4])
    print("prefix best at R85",sorted([(v["max_P_at_R85"]["combined"]["P"],k,v["max_P_at_R85"]["scratch_TP"],v["max_P_at_R85"]["scratch_FP"]) for k,v in low.items() if v["max_P_at_R85"]],reverse=True)[:5])


if __name__=="__main__":
    main()
