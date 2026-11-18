"""One frozen grouped evaluation and an image-paired audit against the last grade candidate."""
import argparse
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torch

import afruit_grade_model as grade
from afruit_grade_train import atomic,read
from matched_grade_loader import load,tensor_at_size


OLD=grade.ROOT/"experiments/grade_matched_review_v1"
PREVIOUS=grade.ROOT/"experiments/afruit_grade_matched_v1/paired"


def evaluate_groups(directory, manifest, dest):
    rowset=read(manifest)["samples"]
    by_group={};by_sha={}
    for r in rowset:
        for seen,key in [(by_group,(r["source"],r["group_id"])),(by_sha,r["sha256"])]:
            if seen.setdefault(key,r["split"])!=r["split"]:
                raise ValueError("Group or identical image crosses the reserved split")
    rows=[dict(r) for r in rowset if r["split"]=="test"]
    if not rows or any(r["grade"] not in (0,1,2) for r in rows):
        raise ValueError("Empty reserved set or unknown grades")
    models={name:load(path,"cuda") for name,path in {"previous":PREVIOUS,"repair":directory}.items()}
    for start in range(0,len(rows),12):
        part=rows[start:start+12];frames=[]
        for r in part:
            file=grade.ROOT/r["path"]
            if grade.file_sha(file)!=r["sha256"]:
                raise ValueError("Changed stored image")
            im=cv2.imdecode(np.fromfile(file,np.uint8),cv2.IMREAD_COLOR)
            if im is None:
                raise ValueError(str(file))
            frames.append(im)
        for name,(model,cut,side) in models.items():
            labels=grade.decide(model(tensor_at_size(frames,"cuda",side)),cut).tolist()
            for r,label in zip(part,labels,strict=True):
                r[name]=label
    out={}
    for name in models:
        matrix=[[0]*3 for _ in range(3)]
        for r in rows:
            matrix[r["grade"]][r[name]]+=1
        out[name]=matrix
    atomic(dest/"new_test_rows.json",rows)
    atomic(dest/"new_test.json",dict(confusion=out,groups=len({(r["source"],r["group_id"]) for r in rows}),
          grade_counts={str(g):sum(r["grade"]==g for r in rows) for g in range(3)},
          absent_grade_policy="No rate or safety claim for a class with zero reserved instances; refer to separately identified external panels.",
          source_counts=dict(Counter(r["source"] for r in rows)),manifest_sha256=grade.file_sha(manifest),
          previous_sha256=grade.file_sha(PREVIOUS/"weights.pt"),repair_sha256=grade.file_sha(directory/"weights.pt")))
    print("NEW HELD-OUT",out,flush=True)


def reconcile(root):
    def data(path):
        return {r["path"]:r for r in read(path)}
    locked=read(root/"frozen_field_candidate.json")["field_target"]["weight_sha256"]
    if (read(root/"field/receipt.json")["weight_sha256"]!=locked or
            read(root/"external/results.json")["models"]["candidate"]!=locked):
        raise ValueError("Field runs do not match the development-frozen challenger")
    ae=data(OLD/"field/paired_predictions.json");be=data(root/"field/paired_predictions.json")
    ah=data(OLD/"external/predictions.json");bh=data(root/"external/predictions.json")
    if set(ae)!=set(be) or set(ah)!=set(bh):
        raise ValueError("Different regression rows")
    af=[{"id":str(Path(key).stem),"truth":r["truth"],"group":r["group"],"before":r["linked_grade"],"after":be[key]["linked_grade"]}
        for key,r in ae.items()]
    hh=[{"id":str(Path(key).stem),"truth":r["target"],"group":r["group"],"before":r["candidate"],"after":bh[key]["candidate"]}
        for key,r in ah.items() if r["panel"]=="HDF"]
    def counts(items):
        return [sum(r["after"]==k for r in items) for k in [0,1,2,None]]
    summary={
        "previous_13_mild_as_good":counts([r for r in af if r["truth"]==1 and r["before"]==0]),
        "previous_36_fresh_as_severe":counts([r for r in hh if r["truth"]=="fresh" and r["before"]==2]),
        "previous_correct_39_auto_good":counts([r for r in af if r["truth"]==0 and r["before"]==0]),
        "previous_correct_14_auto_mild":counts([r for r in af if r["truth"]==1 and r["before"]==1]),
        "previous_correct_28_hdf_fresh":counts([r for r in hh if r["truth"]=="fresh" and r["before"]==0]),
        "previous_6_auto_severe_predicted_mild":counts([r for r in af if r["truth"]==2 and r["before"]==1]),
        "af_all_severe":counts([r for r in af if r["truth"]==2]),
        "hdf_all_source_fresh":counts([r for r in hh if r["truth"]=="fresh"]),
        "hdf_all_source_rot":counts([r for r in hh if r["truth"]=="rot_defect"])}
    cases=[r for r in af if r["truth"]==1 and r["before"]==0]+[r for r in hh if r["truth"]=="fresh" and r["before"]==2]
    if len(cases)!=49:
        raise ValueError("Historical error set changed")
    outcomes={r["id"]:r for r in cases}
    probes=read(root/"hdf_fresh/per_image_probes.json")
    hdf_notes=read(root/"hdf_fresh/visual_notes.json")["items"]
    af_notes=read(grade.ROOT/"experiments/grade_failure_audit_20261118/afruit13/visual_review.json")
    strata={}
    for title,predicate in (("hdf_yellow_edge",lambda i:2<=i<=22),("hdf_glossy_red",lambda i:24<=i<=36),
                           ("hdf_other",lambda i:i in (1,23))):
        strata[title]=counts([outcomes[Path(r["path"]).stem] for r in probes
                              if r["id"].startswith("H") and predicate(int(r["id"][1:]))])
    for category in sorted({r["visual_category"] for r in af_notes}):
        strata[category]=counts([outcomes[Path(r["path"]).stem] for r in af_notes if r["visual_category"]==category])
    atomic(root/"paired_changes.json",dict(model_sha256=locked,order=["good","minor","severe","unresolved"],
          summary=summary,visual_subgroups=strata,errors49=cases))
    labels={0:"好果",1:"轻瑕",2:"重瑕",None:"未解析"}
    lines=["# 49 张历史错例逐张复核","",f"复测权重 `{locked}`。判级使用实际部署输入。","",
           "## HDF 作者新鲜类","","| 文件 | 图中可见现象 | 上轮 | 本轮 |","|---|---|---|---|"]
    for r in sorted(probes,key=lambda row:row["id"]):
        if r["id"] in hdf_notes and r["candidate"]==2:
            key=Path(r["path"]).stem
            lines.append(f"| {key} | {hdf_notes[r['id']]} | 重瑕 | {labels[outcomes[key]['after']]} |")
    lines.extend(["","## AFruitDB 作者轻瑕类","","| 文件 | 可见证据 | 逐图观察 | 上轮 | 本轮 |","|---|---|---|---|---|"])
    for r in af_notes:
        key=Path(r["path"]).stem
        lines.append(f"| {key} | {r['visual_category']} | {r['visual_notes']} | 好果 | {labels[outcomes[key]['after']]} |")
    if len([s for s in lines if s.startswith("| SD_REAL_") or s.startswith("| IMG_")])!=49:
        raise ValueError("Case table missing observations")
    (root/"49张逐图修正结果.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(summary)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=("new","changes"))
    parser.add_argument("report",type=Path)
    parser.add_argument("--candidate",type=Path)
    parser.add_argument("--manifest",type=Path)
    args=parser.parse_args()
    cv2.setNumThreads(1);torch.set_num_threads(2)
    with torch.inference_mode():
        if args.action=="new":
            if (args.report/"new_test.json").exists():
                raise FileExistsError("New-group evaluation already recorded")
            if not args.candidate or not args.manifest:
                parser.error("Supply candidate and reserved manifest")
            evaluate_groups(args.candidate,args.manifest,args.report)
        else:
            reconcile(args.report)
