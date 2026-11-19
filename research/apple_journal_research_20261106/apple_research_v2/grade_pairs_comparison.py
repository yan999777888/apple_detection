"""Freeze a development-qualified model, and compare models on a source-isolated blind panel."""
import argparse
from collections import Counter,defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch

import afruit_grade_model as grade
import afruit_grade_train as io
from matched_grade_loader import load,tensor_at_size
from select_grade_repair import publish


ROOT=grade.ROOT
OUT=ROOT/"experiments/grade_pairs_20261119"
MODELS=dict(original=ROOT/"experiments/afruit_grade_release_v1",woo=ROOT/"experiments/afruit_grade_matched_v1/paired",
            native320=ROOT/"experiments/grade_error49_v1/border_native320_woo_ref")
MANALAGI_MANIFEST_SHA="0cbd92a77598951cf13f3180c58dc96a542d165be3890460e0ccd16f508db7ac"


def freeze(dirs):
    arms=[]
    for folder in dirs:
        folder=folder.resolve();d=io.read(folder/"selection.json");sha=grade.file_sha(folder/"weights.pt")
        if sha!=d["weights_sha256"]:raise ValueError("Selected checkpoint changed")
        a,h=d["metrics"]["auto"],d["metrics"]["whole"]
        assert len(a["confusion"])==3 and [sum(row) for row in a["confusion"]]==[25,15,17]
        passed=a["mild"]>=6 and a["good"]>=20 and a["severe_good"]==0 and a["severe"]>=16 and h["severe_good"]==0
        arms.append(dict(directory=folder.relative_to(ROOT).as_posix(),sha256=sha,passed=passed,
                  rank=[a["mild"],a["good"]+a["severe"],a["good"],h["mild"]],metrics=d["metrics"]))
    eligible=[arm for arm in arms if arm["passed"]]
    if not eligible:raise ValueError("No qualified candidate. Keep blind results sealed.")
    chosen=max(eligible,key=lambda arm:arm["rank"])
    publish(OUT/"field_lock.json",dict(candidate=chosen["directory"],sha256=chosen["sha256"],
             protocol_sha=grade.file_sha(OUT/"protocol.json"),arms=arms))
    print(chosen["directory"],chosen["sha256"])


@torch.inference_mode()
def blind(manifest):
    lock=io.read(OUT/"field_lock.json")
    if (OUT/"blind_comparison.json").exists():
        raise FileExistsError("This blind decision has already been recorded")
    if lock["protocol_sha"]!=grade.file_sha(OUT/"protocol.json"):
        raise ValueError("Model-selection protocol changed after the candidate was frozen")
    if grade.file_sha(manifest)!=MANALAGI_MANIFEST_SHA:
        raise ValueError("The blind set differs from the published selection manifest")
    dirs={**MODELS,"repair":ROOT/lock["candidate"]}
    policy=io.read(OUT/"protocol.json")
    rows=io.read(manifest)["samples"]
    if (any(r["split"]!="test" for r in rows) or Counter(r["grade"] for r in rows)!={0:10,1:11,2:11}
            or len({r["group_id"] for r in rows})!=32 or len({r["sha256"] for r in rows})!=32):
        raise ValueError("Blind manifest must contain 32 independently grouped, reserved fruit photographs")
    predictions=[dict(r) for r in rows]
    hashes={}
    for name,folder in dirs.items():
        hashes[name]=grade.file_sha(folder/"weights.pt")
        assert hashes[name]==(lock["sha256"] if name=="repair" else policy["reference"][name])
        model,cut,side=load(folder,"cuda")
        for start in range(0,len(rows),8):
            frames=[]
            for row in rows[start:start+8]:
                path=ROOT/row["path"]
                if grade.file_sha(path)!=row["sha256"]:raise ValueError(f"Changed blind image {path.name}")
                frame=cv2.imdecode(np.fromfile(path,np.uint8),cv2.IMREAD_COLOR)
                if frame is None:raise ValueError(path)
                frames.append(frame)
            labels=grade.decide(model(tensor_at_size(frames,"cuda",side)),cut).tolist()
            for row,p in zip(predictions[start:start+8],labels,strict=True):row[name]=int(p)
        del model
    results={}
    for source in sorted({r["source"] for r in rows}):
        selected=[r for r in predictions if r["source"]==source]
        matrices={}
        for name in dirs:
            mat=np.zeros((3,3),int)
            for r in selected:mat[r["grade"],r[name]]+=1
            matrices[name]=mat.tolist()
        results[source]=dict(images=len(selected),groups=len({r["group_id"] for r in selected}),matrices=matrices)
    publish(OUT/"blind_comparison.json",dict(manifest_sha=grade.file_sha(manifest),weights=hashes,
            scope="Supplied single visible fruit, original author/visible proxy labels as registered; scene localisation and FPS measured separately.",
            results=results,predictions=predictions))
    print(results,flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest="action",required=True)
    sub.add_parser("freeze").add_argument("dirs",type=Path,nargs="+")
    sub.add_parser("blind").add_argument("manifest",type=Path)
    args=parser.parse_args()
    torch.set_num_threads(2);cv2.setNumThreads(1)
    if args.action=="freeze":freeze(args.dirs)
    else:blind(args.manifest)
