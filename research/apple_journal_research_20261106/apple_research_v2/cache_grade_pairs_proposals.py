"""Replay the frozen detector on AF training and development photos for local defect verification."""
from collections import Counter

import cv2
import torch

import afruit_grade_crop_alignment as align
import afruit_grade_train as io
import afruit_visible_quality as visible

OUT=io.ROOT/"datasets/grade_pairs_sources_20261119"


@torch.inference_mode()
def run():
    cv2.setNumThreads(2)
    torch.set_num_threads(2)
    visible.configure(io.ROOT/"experiments/grade_pairs_20261119/proposal_runtime","cuda")
    pipe=visible.QualityPipeline()
    for split in ("val","train"):
        target=OUT/f"af_proposals_{split}.json"
        if target.exists():
            raise FileExistsError(target)
        regions=io.read(align.OUT/f"{split}_regions.json")
        rows=[]
        counts=Counter()
        for row,ref in zip(align.split_rows(split),regions,strict=True):
            image=visible.prepare_input(align.read_frame(row),1280)
            result,_=pipe(image)
            fruits=[{k:f[k] for k in ("fruit_id","bbox","crop","maturity","maturity_score")} for f in result["fruits"]]
            unique=fruits[0]["crop"] if len(fruits)==1 else None
            assert row["sha256"]==ref["sha256"] and unique==ref["crop"] and list(image.shape[:2])==ref["hw"]
            rows.append(dict(path=row["path"],sha256=row["sha256"],split=split,group_id=row["group_id"],
                             grade=row["grade"],hw=ref["hw"],crop=unique,fruits=fruits,defects=result["defects"]))
            counts["images"]+=1
            counts["eligible"]+=int(unique is not None)
            counts["lesions"]+=len(result["defects"])
            counts["unique_lesions"]+=sum(d["association"]=="unique" for d in result["defects"])
        io.atomic(target,dict(samples=rows,counts=counts,split=split,frame_long_edge=1280,
                  stage_sha=visible.STAGE_WEIGHT_SHA,defect_parent_sha=visible.SOURCE_SHA,defect_warm_sha=visible.WARM_SHA))
        print(split,counts,flush=True)


if __name__=="__main__":
    run()
