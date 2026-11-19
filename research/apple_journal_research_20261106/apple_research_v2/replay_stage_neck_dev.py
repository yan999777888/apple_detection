"""Replay the fixed two native deployment modes, independently of the cached fit."""
import cv2
import torch
import numpy as np
import grade_stage_neck as neck
import afruit_grade_crop_alignment as af
from train_stage_neck import evaluate,order


@torch.inference_mode()
def replay():
    out=neck.OUT
    chosen=neck.io.read(out/"selection.json")
    reference=neck.io.read(out/"development_predictions.json")
    pred={(r["sha"],r["view"]):r for r in reference["records"]}
    rt=neck.Runtime(out);mismatch=[];rows=[];mode_checks=0
    for r in af.split_rows("val"):
        frame=neck.visible.prepare_input(af.read_frame(r),1280)
        for mode in ("auto","whole"):
            known=mode=="whole"
            answer,masks=rt(frame,known_single_fruit=known)
            if known:z=answer["known_single_fruit_grade"]
            else:z=answer["fruits"][0] if len(answer["fruits"])==1 else None
            expect=pred.get((r["sha256"],mode));actual=None if z is None else z["grade_id"]
            if expect is None and actual is None:mode_checks+=1;continue
            if expect is None or actual!=expect["candidate"]:
                mismatch.append(dict(path=r["path"],mode=mode,cached=None if expect is None else expect["candidate"],actual=actual))
            mode_checks+=1
            rows.append(dict(path=r["path"],sha=r["sha256"],grade=r["grade"],mode=mode,predicted=actual,
                             cached=None if expect is None else expect["candidate"],parent=None if expect is None else expect["baseline"]))
    cases=[]
    for mode in ("auto","whole"):
        changed=[a for a in rows if a["mode"]==mode and a["predicted"]!=a["parent"]]
        cases+=changed
    neck.io.atomic(out/"deployment_dev_replay.json",dict(mode_count=mode_checks,independent_scene_frames=57,known_single_frames=57,expected_weight=chosen["weights_sha256"],
              zero_mismatches=len(mismatch)==0,mismatches=mismatch,changed_cases=cases,rows=rows))
    print("mode rows",mode_checks,"mismatches",len(mismatch),"changes",len(cases),flush=True)
    if mismatch:raise ValueError("Runtime labels differ from cached head")


if __name__=="__main__":
    cv2.setNumThreads(2);torch.set_num_threads(2);replay()

