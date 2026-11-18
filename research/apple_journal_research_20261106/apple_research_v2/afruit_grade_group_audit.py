"""After selection, count apple grading errors within held-out acquisition groups."""
import argparse
import collections
from pathlib import Path

import cv2
import numpy as np
import torch

import afruit_grade_crop_alignment as aligned
import afruit_grade_model as grade
import afruit_grade_train as original


@torch.inference_mode()
def audit(arm,device):
    chosen=original.read(arm/"selection.json")
    if grade.file_sha(arm/"weights.pt")!=chosen["weights_sha256"] or chosen["threshold"]!=1:
        raise ValueError("A frozen argmax grading checkpoint is required")
    rows,whole,truth=original.open_rows(("val",))["val"]
    crops=aligned.load_cache("val",rows)
    model=grade.network()
    model.load_state_dict(torch.load(arm/"weights.pt",map_location="cpu",weights_only=True))
    model=model.eval().to(device)
    def predict(items):
        batches=[model(grade.tensor_rgb(items[i:i+24],device)).float().cpu() for i in range(0,len(items),24)]
        return torch.cat(batches).argmax(1).numpy()
    valid=[i for i,v in enumerate(crops) if v is not None]
    auto=np.full(len(rows),3)
    auto[valid]=predict([crops[i] for i in valid])
    full=predict(whole)
    groups=collections.defaultdict(list)
    for i,row in enumerate(rows):
        groups[row["group_id"]].append(i)
    result={}
    reproduced={}
    for name,prediction in (("auto",auto),("whole",full)):
        reproduced[name]=aligned.score(truth,prediction)["confusion"]==chosen["metrics"][name]["confusion"]
        if device=="cuda" and not reproduced[name]:
            raise ValueError("Group audit failed to reproduce official validation counts")
        result[name]={g:aligned.score(truth[indices],prediction[indices])["confusion"]
                      for g,indices in sorted(groups.items())}
    output=dict(model_sha=chosen["weights_sha256"],manifest_sha256=grade.file_sha(original.MANIFEST),
                grade_rows=list(grade.LABELS),grade_columns=[*grade.LABELS,"unresolved"],
                precision_scope="groups are linked captures, not proven fruit identities",group_counts=result,device=device,
                reproduced_cuda_validation=reproduced)
    original.atomic(arm/f"validation_groups_{device}.json",output)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("arm",type=Path)
    parser.add_argument("--device",choices=("cpu","cuda"),default="cpu")
    arg=parser.parse_args()
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    audit(arg.arm,arg.device)
