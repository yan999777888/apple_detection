"""Trace every fixed-agreement false positive/miss; constrained same-center box correction ceiling."""
from collections import Counter
from functools import lru_cache
import numpy as np
import cv2
from scipy.optimize import linear_sum_assignment

from class0_patch_refinement import load_image,manifest
from defect_continue_scratch_audit import P,OUT,PHASE,WARM,indexed,read,native,label,atomic_json
from defect_continue_scratch_pairs import scratch_rows
from audit_defect_causes_20261112 import match,ious

AG="defect9485_local_fusion/agreement_predictions.json"


@lru_cache(maxsize=120)
def archive(name):
    source,*variant=name.split(":")
    rows=indexed(read(P/source))
    if variant:
        return {i:(r.get("raw") or r.get("sources"))[variant[0]] for i,r in rows.items()}
    return {i:r["pred"] for i,r in rows.items()}


def class_boxes(name,sid):
    a=np.asarray(archive(name)[sid],float).reshape(-1,6)
    return a[a[:,5]==0]


def best(name,sid,gt,fp=None):
    a=class_boxes(name,sid)
    if fp is not None and len(a):
        a=a[ious(a[:,:4],[fp])[:,0]>=.1]
    if not len(a):
        return dict(IoU=0.,score=None,box=None)
    ov=ious(a[:,:4],[gt])[:,0]
    j=int(np.argmax(ov))
    return dict(IoU=float(ov[j]),score=float(a[j,4]),box=a[j,:4].tolist())


def center_ceiling(fp,gt):
    b=np.asarray(fp);g=np.asarray(gt)
    c=(b[:2]+b[2:])/2
    # Intersection is affine between rectangle-edge breakpoints. Holding the
    # other axis fixed makes IoU fractional-linear, so an extremum is at a breakpoint.
    candidates=[np.r_[c-[dx,dy],c+[dx,dy]] for dx in np.maximum(.00001,np.abs(c[0]-g[[0,2]]))
                for dy in np.maximum(.00001,np.abs(c[1]-g[[1,3]]))]
    ov=ious(candidates,[g])[:,0];j=int(ov.argmax())
    return float(ov[j]),candidates[j].tolist()


def maximum_assignment(edges):
    sources=sorted({x[0] for x in edges});targets=sorted({x[1] for x in edges})
    if not sources or not targets:return []
    cost=np.zeros((len(sources),len(targets)))
    bonus=max(len(sources),len(targets))+1
    for uid,gid,qual in edges:cost[sources.index(uid),targets.index(gid)]=bonus+float(qual)
    ii,jj=linear_sum_assignment(-cost)
    return [dict(fp=sources[i],gt=targets[j]) for i,j in zip(ii,jj) if cost[i,j]>0]


def main():
    ref=indexed(read(PHASE));lookup=manifest()
    sources=list(read(OUT/"candidate_frontiers.json"))
    source_head="defect_algorithm_steps_20261112_v2/phase1/source_predictions.json:"
    # Never pooled at runtime; per-architecture upper bound is explicitly retrospective.
    input_streams=[source_head+k for k in ("o2o","o2m","flip")]
    low_branches=["defect9485_surface/low_predictions.json","defect9485_set/low_predictions.json"]
    arms=["class0_crop_deploy_v1/fp32/dev_predictions.json",
          "defect9485_surface/joint_predictions.json","defect9485_set/tradeoff_predictions.json"]
    original={r["reference"]:r["code"] for r in read(OUT/"original29.json")}
    ledger=[]; missed=[]; edges_raw=[];edges_fixedcenter=[];edges_alt=[]
    warm=native(read(WARM),ref)
    reference_rules=Counter()
    tiles=[]
    for sid,r in ref.items():
        gt=r["GT"];p=class_boxes(AG,sid)
        m,ov=match(p,gt)
        insect=np.array(warm[sid]["pred"],float).reshape(-1,6)
        insect=insect[insect[:,5]==1]
        with_insect=ious(p[:,:4],insect[:,:4])
        overlaps=ious(p[:,:4],p[:,:4])
        np.fill_diagonal(overlaps,0)
        keep_nms=[]
        for n in sorted(range(len(p)),key=lambda j:(-p[j,4],j)):
            if not keep_nms or max(overlaps[n,keep_nms])<.5:
                keep_nms.append(n)
        insect_flags=with_insect.max(1)>=.5 if with_insect.shape[1] else np.zeros(len(p),bool)
        for name,retained in [("insect_overlap_ge_050_veto",~insect_flags),
                              ("within_class_nms_050",[j in keep_nms for j in range(len(p))])]:
            tested=p[np.asarray(retained,dtype=bool)]
            actual,_=match(tested,gt)
            reference_rules[name+"_scratch_TP"]+=len(actual)
            reference_rules[name+"_scratch_FP"]+=len(tested)-len(actual)
        unhit={j for j,g in enumerate(gt) if g["class"]==0 and j not in m.values()}
        for j in sorted(unhit):
            box=gt[j]["box"]
            a={k:best(k,sid,box) for k in arms}
            b={k:best(k,sid,box) for k in low_branches}
            c={k:best(k,sid,box) for k in input_streams}
            d=max(((best(k,sid,box)["IoU"],k) for k in sources),default=(0.,None))
            reason=("agreement_suppressed_despite_fixed_upstream_box" if max(v["IoU"] for v in a.values())>=.5 else
                    "surface_or_set_low_pool_only" if max(v["IoU"] for v in b.values())>=.5 else
                    "original_Nano_geometry_lost_in_verification" if max(v["IoU"] for v in c.values())>=.5 else
                    "different_architecture_geometry_only" if d[0]>=.5 else "no_saved_class0_geometry")
            missed.append(dict(id=label(sid,j),original_29_code=original.get(label(sid,j)),GT=box,
                               reason=reason,upstream_fixed=a,upstream_low=b,nano_low=c,best_any=d))
        for n,box in enumerate(p):
            if n in m:continue
            i=int(ov[n].argmax()) if gt else None
            v=ov[n,i] if i is not None else 0
            reason=("duplicate" if gt[i]["class"]==0 else "wrong_class") if v>=.5 else ("extent" if v>=.1 else "no_nearby_GT")
            uid=f"F{len(ledger)+1:02}"
            record=dict(uid=uid,image=sid,box=box[:4].tolist(),score=float(box[4]),reason=reason,
                        nearest_id=None if i is None else label(sid,i),
                        nearest_class=None if i is None else gt[i]["class"],nearest_iou=float(v),
                        nearest_gt_already_detected_by_scratch=i in m.values(),nearby_same_class=[])
            record["warm_insect_prediction_iou"]=float(with_insect[n].max(initial=0.))
            record["other_agreement_scratch_iou"]=float(overlaps[n].max(initial=0.))
            record["nms050_kept"]=n in keep_nms
            for j,g in enumerate(gt):
                if g["class"]!=0 or ov[n,j]<.1:continue
                gid=label(sid,j)
                samecenter,oracle_box=center_ceiling(box[:4],g["box"])
                other=max(((best(k,sid,g["box"],box[:4])["IoU"],k) for k in sources))
                b=np.asarray(g["box"]); wh=b[2:]-b[:2]
                predwh=box[2:4]-box[:2]
                free=j in unhit
                record["nearby_same_class"].append(dict(id=gid,unmatched=free,original29=original.get(gid),
                    existing_iou=float(ov[n,j]),area_ratio_pred_gt=float(np.prod(predwh)/np.prod(wh)),
                    width_ratio=float(predwh[0]/wh[0]),height_ratio=float(predwh[1]/wh[1]),
                    center_offset_in_GT_wh=(((box[:2]+box[2:4])-(b[:2]+b[2:]))/(2*wh)).tolist(),
                    exact_same_center_max_iou=samecenter,oracle_same_center_box=oracle_box,
                    best_other_candidate=other))
                if free:
                    edges_raw.append((uid,gid,ov[n,j]))
                    if samecenter>=.5:edges_fixedcenter.append((uid,gid,samecenter))
                    if other[0]>=.5:edges_alt.append((uid,gid,other[0]))
            if i is not None:
                ng=np.asarray(gt[i]["box"]);record["nearest_GT_box"]=ng.tolist()
                record["nearest_area_ratio_pred_gt"]=float(np.prod(box[2:4]-box[:2])/np.prod(ng[2:]-ng[:2]))
            ledger.append(record)
            im=load_image(lookup[sid]["rgb_path"]);H,W=im.shape[:2]
            size=min(H,W,int(max(110,2*(box[2:4]-box[:2]).max())))
            x,y=np.clip(((box[2:4]+box[:2])/2-size/2).astype(int),[0,0],[W-size,H-size])
            crop=im[y:y+size,x:x+size].copy()
            overlay=crop.copy()
            for j,g in enumerate(gt):
                b=(np.array(g["box"])-[x,y,x,y]).astype(int)
                cv2.rectangle(overlay,tuple(b[:2]),tuple(b[2:]),(30,240,40) if g["class"]==0 else (255,135,15),1)
            b=(box[:4]-[x,y,x,y]).astype(int)
            cv2.rectangle(overlay,tuple(b[:2]),tuple(b[2:]),(15,30,255),2)
            title=np.full((22,248,3),250,np.uint8)
            cv2.putText(title,uid+" "+reason,(4,15),cv2.FONT_HERSHEY_SIMPLEX,.45,(20,20,20),1)
            tiles.append(np.vstack([title,np.hstack([cv2.resize(crop,(124,124)),cv2.resize(overlay,(124,124))])]))
    assert len(ledger)==13 and len(missed)==22
    while len(tiles)%3:tiles.append(np.full_like(tiles[0],250))
    img=np.vstack([np.hstack(tiles[i:i+3]) for i in range(0,len(tiles),3)])
    (OUT/"agreement_13_fp.jpg").write_bytes(cv2.imencode(".jpg",img)[1])
    output=dict(source=AG,misses=missed,miss_counts=dict(Counter(x["reason"] for x in missed)),false_positives=ledger,
                same_frame_fixed_outputs_diagnostic=dict(reference_rules),
                one_to_one_optimistic=dict(nearby_same_class_unmatched=maximum_assignment(edges_raw),
                    same_center_extent_only=maximum_assignment(edges_fixedcenter),
                    another_archived_candidate_near_this_FP=maximum_assignment(edges_alt)),
                independent_groundtruth="No second blind annotation or verified different view of these same locations was available; an unmatched image patch cannot establish biological absence.")
    atomic_json(OUT/"agreement_13fp_22fn.json",output)
    print(output["miss_counts"],output["one_to_one_optimistic"])
    for r in ledger:
        if r["reason"]=="extent":
            print(r["uid"],r["nearest_id"],r["nearest_class"],r["nearest_iou"],r["nearest_area_ratio_pred_gt"],
                  [(q["id"],q["unmatched"],q["exact_same_center_max_iou"],q["best_other_candidate"][0]) for q in r["nearby_same_class"]])
if __name__=="__main__":
    main()
