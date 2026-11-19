"""Verify saved sparse grader and measure proposal coverage on independently localized dev marks."""
from collections import defaultdict
import math
import torch

import afruit_grade_model as grade
import afruit_grade_train as io
from matched_grade_loader import load
import grade_pairs_20261119 as net
import train_grade_pairs_20261119 as job


def run():
    torch.set_num_threads(2);job.cv2.setNumThreads(1)
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    model,_,side=load(job.OUT,"cuda")
    if not isinstance(side,net.PairSide):raise ValueError("Unexpected input type")
    frozen=io.read(job.OUT/"development_predictions.json")
    predictions={(r["path"],r["panel"]):r for r in frozen["rows"]}
    cache={r["path"]:r for r in io.read(job.SOURCE/"af_proposals_val.json")["samples"]}
    regions=defaultdict(list)
    for r in io.read(job.SOURCE/"regions.json")["regions"]:
        if r["split"] in ("val","dev") and r["target"] in ("surface_defect","confirmed_normal"):
            regions[r["path"]].append(r)
    result=[];matches=0;scored=0;image_read={}
    with torch.inference_mode():
        for row,im,hints in job.views("val"):
            if im is None:continue
            key=(row["path"],row["panel"]);saved=predictions[key]
            x=net.prepare([im],"cuda",[hints])
            out=model(x)
            scored+=1;matches+=int(int(out.argmax(1))==saved["proposed"])
            candidates=net.proposals(im,hints)
            for actual in regions[row["path"]]:
                x1,y1,x2,y2=actual["source_image_xyxy"]
                if row["panel"]=="auto":
                    r=cache[row["path"]];offx,offy=r["crop"][:2]
                    ry,rx=[a/b for a,b in zip(r["hw"],actual["image_hw"],strict=True)]
                    x1,x2=x1*rx-offx,x2*rx-offx;y1,y2=y1*ry-offy,y2*ry-offy
                if x2<0 or y2<0 or x1>=im.shape[1] or y1>=im.shape[0]:continue
                def coverage(p):
                    cx,cy,size,typ=p
                    inter=max(0,min(x2,cx+size/2)-max(x1,cx-size/2))*max(0,min(y2,cy+size/2)-max(y1,cy-size/2))
                    return dict(type=typ,region_coverage=inter/max(1.,(x2-x1)*(y2-y1)),center_on_mark=bool(x1<=cx<=x2 and y1<=cy<=y2))
                spots=[coverage(p) for p in candidates]
                best=saved["best_proposal"]
                tested=coverage(best) if best else None
                result.append(dict(source=actual["source"],id=actual["id"],target=actual["target"],path=row["path"],
                    panel=row["panel"],subtype=actual["subtype"],centered=any(a["center_on_mark"] for a in spots),
                    covered50=any(a["region_coverage"]>=.5 for a in spots),
                    detector_covered50=any(a["region_coverage"]>=.5 and a["type"]=="detector" for a in spots),
                    best=tested,best_correct_grade=saved["proposed"]==row["grade"]))
            image_read[row["path"]]=True
    sums={}
    for k in sorted({(r["panel"],r["target"]) for r in result}):
        rows=[r for r in result if (r["panel"],r["target"])==k]
        sums["/".join(k)]=dict(n=len(rows),in_native_any=sum(r["covered50"] for r in rows),
            in_detector_any=sum(r["detector_covered50"] for r in rows),best_native=sum(bool(r["best"]) and r["best"]["region_coverage"]>=.5 for r in rows),
            exact_center_inside=sum(r["centered"] for r in rows))
    io.atomic(job.OUT/"native_replay_and_support.json",dict(grade_export_matches=matches,verified_inputs=scored,
        complete=matches==scored==len(predictions),photo_count=len(image_read),summary=sums,rows=result))
    print("Saved-weight argmax",matches,scored,"coverage",sums,flush=True)
    if matches!=scored or scored!=len(predictions):raise ValueError("Saved production model differs from dev cache")


if __name__=="__main__":
    run()
