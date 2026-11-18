"""Choose one graded-apple checkpoint using development photos only."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile

import cv2
import numpy as np
import torch

import afruit_grade_crop_alignment as align
import afruit_grade_model as grade
import afruit_grade_train as io
import guard_grade_v2 as guard
from matched_grade_loader import load
from grade_detail_patch import ARCH as PATCH320, DetailSide, pack, square as native_square


ROOT = grade.ROOT
OUT = ROOT / "experiments/grade_error49_v1"
REFERENCE = ROOT / "experiments/afruit_grade_matched_v1/paired"
POLICY = OUT / "dev_selection_protocol.json"
FIELD = OUT / "field_protocol.json"
POLICY_SHA = "661e60330b74efa6fb948aad1ba14675bf619ae7760bd94514df829a0d164f4c"
FIELD_SHA = "43684adc5f471b79a3b1c93b9db6010f11e97e935278d7ae06b34098827f372a"
PANELS = ("auto", "whole", "guard")
SPATIAL = "mobilenet_v3_small_supervised_surface_v1"
CONTRAST = "mobilenet_v3_small_confirmed_mark_contrast_v1"


def relative(path):
    return Path(path).resolve().relative_to(ROOT.resolve()).as_posix()


def policy_digest(path):
    content=json.dumps(io.read(path),sort_keys=True,ensure_ascii=False,separators=(",",":")).encode()
    return hashlib.sha256(content).hexdigest()


def input_panels(detail=False):
    # The canonical guard loader hashes/decodes the supplied crop file and
    # letterboxes it exactly once. An embedded source_box is provenance only.
    af=align.split_rows("val")
    whole=[grade.rgb_square(align.read_frame(r)) for r in af]
    crops=align.load_cache("val",af)
    extra,images=guard.support("val")
    if len(af) != 57 or Counter(r["grade"] for r in af) != {0:25, 1:15, 2:17}:
        raise ValueError("Unexpected AF development set")
    if not extra or any(r["split"] != "val" for r in af + extra):
        raise ValueError("Unexpected development rows")
    if any("hdf" in r["source"].lower() for r in extra):
        raise ValueError("HDF is excluded from development")
    boxes = io.read(align.OUT / "val_regions.json")
    standard = io.read(align.OUT / "protocol.json")
    if (standard["long_edge"] != 1280 or standard["crop_context"] != 0.1
            or standard["manifest_sha256"] != grade.file_sha(io.MANIFEST)):
        raise ValueError("Different AF automatic crop")
    groups = {}
    for name, records, squares in (("auto", af, crops), ("whole", af, whole), ("guard", extra, images)):
        meta = []
        for i, (row, square) in enumerate(zip(records, squares, strict=True)):
            if square is not None and (square.shape != (224,224,3) or square.dtype != np.uint8):
                raise ValueError("Unexpected model input")
            item = {k: row[k] for k in ("path","grade","group_id","sha256","split")}
            item.update(source=row.get("source", "AFruitDB"), input_sha256=(
                hashlib.sha256(square.tobytes()).hexdigest() if square is not None else None))
            if name == "auto":
                box = boxes[i]
                if box["sha256"] != row["sha256"] or (square is None) != (box["crop"] is None):
                    raise ValueError("Crop manifest/pixel mismatch")
                item.update(long_edge_1280_xyxy=box["crop"], detector_frame_hw=box["hw"])
            elif name == "guard" and "source_box" in row:
                item["supplied_source_box_xyxy"] = row["source_box"]
            meta.append(item)
        groups[name] = (meta, squares)
    if detail:
        from afruit_visible_quality import prepare_input
        raw320={name:[] for name in PANELS}
        for i,row in enumerate(af):
            image=align.read_frame(row)
            frame=prepare_input(image,align.SIDE)
            box=boxes[i]
            if list(frame.shape[:2])!=box["hw"]:
                raise ValueError("Different detector frame dimensions")
            bounds=box["crop"]
            fruit=None if bounds is None else frame[bounds[1]:bounds[3],bounds[0]:bounds[2]]
            for panel,raw in (("whole",image),("auto",fruit)):
                if raw is not None and not np.array_equal(grade.rgb_square(raw),groups[panel][1][i]):
                    raise ValueError("Rebuilt 224 input differs from original development pixels")
                raw320[panel].append(native_square(raw,320) if raw is not None else None)
        for i,row in enumerate(extra):
            image=align.read_frame(row)
            if not np.array_equal(grade.rgb_square(image),images[i]):
                raise ValueError("Guard source crop changed")
            raw320["guard"].append(native_square(image,320))
        for name,(meta,squares) in list(groups.items()):
            for r,high in zip(meta,raw320[name],strict=True):
                r["input_sha256_320"]=hashlib.sha256(high.tobytes()).hexdigest() if high is not None else None
            groups[name]=(meta,squares,raw320[name])
    return groups


def counts(records, decisions):
    # Fourth column is missing fruit/detector output, never a grading class.
    table = [[0]*4 for _ in range(3)]
    for row, pred in zip(records, decisions, strict=True):
        table[row["grade"]][3 if pred is None else pred] += 1
    recalls = [table[i][i]/sum(table[i]) if sum(table[i]) else None for i in range(3)]
    return dict(n=len(records), confusion=table, recall=recalls, good=table[0][0], mild=table[1][1],
                severe=table[2][2], severe_good=table[2][0], good_severe=table[0][2],
                correct=sum(table[i][i] for i in range(3)))


@torch.inference_mode()
def evaluate(directory, panels, device):
    sha = grade.file_sha(directory / "weights.pt")
    receipt = io.read(directory / "selection.json")
    architecture = receipt.get("architecture")
    if architecture not in (None, "MobileNet V3 Small", "MobileNetV3-Small", SPATIAL, CONTRAST, PATCH320):
        raise ValueError(f"Unexpected repair architecture {architecture}")
    model, _, side = load(directory, device)
    if side != 224 and not isinstance(side,DetailSide):
        raise ValueError("This comparison permits the frozen 224 graders and the native local 320 verifier")
    metrics, outcomes, summaries = {}, {}, {}
    for panel, inputs in panels.items():
        rows, squares=inputs[:2]
        indices = [i for i, square in enumerate(squares) if square is not None]
        logits, decisions = [None] * len(rows), [None] * len(rows)
        for start in range(0, len(indices), io.BATCH):
            batch = indices[start:start+io.BATCH]
            small=[squares[i] for i in batch]
            x = pack(small,[inputs[2][i] for i in batch],device) if isinstance(side,DetailSide) else grade.tensor_rgb(small,device)
            y = model(x).float().cpu()
            if y.shape != (len(batch),3) or not torch.isfinite(y).all():
                raise ValueError("Unexpected or nonfinite grading output")
            for i, values, label in zip(batch, y.tolist(), y.argmax(1).tolist(), strict=True):
                logits[i], decisions[i] = values, label
        outcomes[panel] = [dict(row_index=i, logits=values, predicted=label)
                           for i,(values,label) in enumerate(zip(logits,decisions,strict=True))]
        metrics[panel] = counts(rows, decisions)
        group_ids = defaultdict(list)
        for i, row in enumerate(rows):
            group_ids[(row["source"],row["group_id"])].append(i)
        summaries[panel] = [dict(source=source,group_id=group,**counts([rows[i] for i in ix],
                                 [decisions[i] for i in ix])) for (source,group),ix in sorted(group_ids.items())]
    if grade.file_sha(directory / "weights.pt") != sha or receipt["weights_sha256"] != sha:
        raise ValueError("Weights changed during evaluation")
    return dict(directory=relative(directory), weight_sha256=sha, selection_sha256=grade.file_sha(
        directory / "selection.json"), architecture=architecture or "MobileNetV3-Small", metrics=metrics,
        groups=summaries, predictions=outcomes)


def compare(arm, reference):
    m,p=arm["metrics"],reference["metrics"]
    shortfalls={}
    for name in PANELS:
        shortfalls[name] = dict(extra_severe_good=max(0,m[name]["severe_good"]-p[name]["severe_good"]),
                               lost_severe=max(0,p[name]["severe"]-m[name]["severe"]),
                               lost_good=max(0,p[name]["good"]-m[name]["good"]))
    mild = {name:max(0,p[name]["mild"]+int(name=="auto")-m[name]["mild"]) for name in PANELS}
    all_safety = [n for panel in shortfalls.values() for n in panel.values()]
    metric_rank = [m["auto"]["mild"],m["whole"]["mild"],m["guard"]["mild"],m["guard"]["good"],
                   m["auto"]["good"],m["whole"]["good"], -sum(s["severe_good"] for s in m.values()),
                   sum(s["severe"] for s in m.values()),sum(s["correct"] for s in m.values())]
    risk_rank = [sum(s[k] for s in shortfalls.values()) for k in ("extra_severe_good","lost_severe","lost_good")]
    arm.update(passed=not any(all_safety) and not any(mild.values()), safety_shortfalls=shortfalls,
               mild_shortfalls=mild, performance_rank=metric_rank, diagnosis_shortfalls=risk_rank+[
                   sum(bool(x) for x in all_safety),sum(bool(x) for x in mild.values())])


def candidate_order(arm):
    return (not arm["passed"],tuple(arm["diagnosis_shortfalls"]),tuple(-v for v in arm["performance_rank"]),
            (2 if arm["architecture"]==PATCH320 else int(arm["architecture"] in (SPATIAL,CONTRAST))),
            Path(arm["directory"]).name,arm["weight_sha256"],arm["directory"])


def identify(arm):
    return {k:arm[k] for k in ("directory","weight_sha256","selection_sha256","architecture")}


def publish(path, data):
    # Link a fully fsync'd staging file into place, atomically refusing an existing lock.
    with tempfile.NamedTemporaryFile("w",encoding="utf-8",dir=path.parent,delete=False) as file:
        scratch=Path(file.name)
        try:
            json.dump(data,file,ensure_ascii=False,indent=2,allow_nan=False)
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())
        except BaseException:
            file.close()
            scratch.unlink(missing_ok=True)
            raise
    try:
        os.link(scratch,path)
    finally:
        scratch.unlink(missing_ok=True)


def run(folders, output, device):
    if output.exists():
        raise FileExistsError(f"A development decision already exists: {output}")
    if policy_digest(POLICY)!=POLICY_SHA or policy_digest(FIELD)!=FIELD_SHA:
        raise ValueError("The frozen policy or field protocol changed")
    policy=io.read(POLICY)
    if grade.file_sha(REFERENCE/"weights.pt") != policy["reference_sha256"]:
        raise ValueError("Wrong Woo reference")
    dirs=[Path(f).resolve() for f in folders]
    dirs=[d for d in dirs if d != REFERENCE.resolve()]
    if not dirs or len(set(dirs))!=len(dirs):
        raise ValueError("Supply at least one unique challenger directory")
    torch.set_num_threads(2)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False
    cv2.setNumThreads(1)
    panels=input_panels(detail=any(io.read(d/"selection.json").get("architecture")==PATCH320 for d in dirs))
    reference=evaluate(REFERENCE,panels,device)
    arms=[evaluate(d,panels,device) for d in dirs]
    for arm in arms:
        compare(arm,reference)
    best=min(arms,key=candidate_order)
    selected=best if best["passed"] else reference
    result=dict(created_utc=datetime.now(timezone.utc).isoformat(), selected=identify(selected),
                diagnostic_challenger=identify(best) if not best["passed"] else None,
                disposition="eligible_for_field_review" if best["passed"] else "retain_woo_diagnostic_only",
                field_target=identify(best),field_runs_allowed=1,policy_sha256=grade.file_sha(POLICY),
                field_protocol_sha256=grade.file_sha(FIELD),policy_canonical_sha256=POLICY_SHA,
                field_protocol_canonical_sha256=FIELD_SHA,script_sha256=grade.file_sha(__file__),
                software=dict(torch=torch.__version__,opencv=cv2.__version__,device=device,
                              gpu=torch.cuda.get_device_name() if device=="cuda" else None),
                manifests={"AF":grade.file_sha(io.MANIFEST),"guard":grade.file_sha(guard.INPUT),
                           "af_crop_protocol":grade.file_sha(align.OUT/"protocol.json"),
                           "val_crops":grade.file_sha(align.OUT/"val_crops.npy"),
                           "val_regions":grade.file_sha(align.OUT/"val_regions.json")},
                confusion_rows=["good","mild","severe"],confusion_columns=["good","mild","severe","unresolved"],
                panel_rows={key:value[0] for key,value in panels.items()},reference=reference,arms=arms)
    publish(output,result)
    print(json.dumps({k:result[k] for k in ("disposition","selected","diagnostic_challenger","field_target")},
                     ensure_ascii=False,indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directories",nargs="+",type=Path,help="Challenger directories with weights.pt/selection.json")
    parser.add_argument("--output",type=Path,default=OUT/"frozen_candidate.json")
    parser.add_argument("--device",choices=("cuda","cpu"),default="cuda")
    args=parser.parse_args()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    lock=args.output.with_suffix(".selecting")
    with lock.open("x",encoding="utf-8") as handle:
        handle.write(str(os.getpid()))
    try:
        run(args.directories,args.output,args.device)
    finally:
        lock.unlink()
