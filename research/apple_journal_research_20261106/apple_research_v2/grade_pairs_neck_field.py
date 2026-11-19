"""Evaluate a locked, scene-feature grading head in its native frame coordinates."""
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageOps

import afruit_visible_quality as visible
import benchmark_afruit_visible_quality as baseline
import grade_matched_external_check as ext
import grade_matched_field_check as field
from grade_stage_neck import Runtime


OUT = visible.ROOT / "experiments/grade_pairs_20261119"


class FieldRuntime(Runtime):
    def classify(self, frames):
        return [self(visible.prepare_input(im,1280),known_single_fruit=True)[0]["known_single_fruit_grade"]
                for im in frames]


def candidate():
    lock=visible.read(OUT/"field_lock.json")
    folder=visible.ROOT/lock["candidate"]
    if lock["protocol_sha"]!=visible.sha(OUT/"protocol.json") or lock["sha256"]!=visible.sha(folder/"weights.pt"):
        raise ValueError("Changed protocol or frozen checkpoint")
    if visible.read(folder/"selection.json")["architecture"]!="stage_neck_native320_v1":
        raise ValueError("This entry requires a shared scene-feature grading head")
    return folder


def af_panel(folder, dest, pipe):
    rows=sorted((r for r in visible.read(baseline.MANIFEST)["samples"] if r["split"]=="test"),key=lambda r:r["path"])
    historical=visible.read(baseline.OUT/"latency_market_frame_1280.json")
    photos=[(r["source"],Path(r["path"])) for r in historical["files"]]
    if {p.resolve() for _,p in photos}!={(visible.ROOT/r["path"]).resolve() for r in baseline.spaced(rows,20)}:
        raise ValueError("Timing-panel identity changed")
    visible.atomic_json(dest/"receipt.json",dict(weight_sha256=visible.sha(folder/"weights.pt"),
        selection_sha256=visible.sha(folder/"selection.json"),model=visible.read(folder/"selection.json")["architecture"],
        input_manifest=visible.sha(baseline.MANIFEST),gpu=torch.cuda.get_device_name(0),opencv_threads=cv2.getNumThreads(),
        input_long_edge=1280,rounds=30))
    result=field.paired_results(pipe,rows,dest)
    print("AF",result["auto"]["confusion"],flush=True)
    baseline.OUT=dest
    for mode in ("scene","single"):
        rate=baseline.benchmark(pipe,mode,photos,30,known=(mode=="single"),long_edge=1280)
        print(mode,rate["all"]["input_to_annotated_frame"],flush=True)


def external_panel(folder, dest, pipe):
    from afruit_grade_model import file_sha
    previous_root=visible.ROOT/"experiments/grade_error49_v1/external"
    old=visible.read(previous_root/"predictions.json")
    ref=visible.read(previous_root/"results.json")
    if ref["models"]["original"]!=visible.read(OUT/"protocol.json")["reference"]["original"]:
        raise ValueError("Original reference checkpoint changed")
    rows=ext.photos_and_labels()
    reference={r["sha256"]:r for r in old}
    if len(reference)!=len(old) or {r["sha256"] for r in rows}!=set(reference):
        raise ValueError("External reference images changed")
    for r in rows:
        src=visible.ROOT/r["path"]
        if file_sha(src)!=r["sha256"] or r["target"]!=reference[r["sha256"]]["target"]:
            raise ValueError(f"External sample changed: {src.name}")
        if r["panel"]=="HDF":
            with Image.open(src) as im:
                frame=np.asarray(ImageOps.exif_transpose(im).convert("RGB"))[:,:,::-1].copy()
        else:
            frame=cv2.imdecode(np.fromfile(src,np.uint8),cv2.IMREAD_COLOR)
        r["original"]=reference[r["sha256"]]["original"]
        r["candidate"]=pipe.classify([frame])[0]["grade_id"]
    result=dict(models={"original":ref["models"]["original"],"candidate":file_sha(folder/"weights.pt")},
        scope="Fresh candidate forward on GPU, standalone known-single deployment pass at up to 1280px. Published original CPU labels reused by exact image hash. HDF source labels; FruitLife repeated views of three physical fruit.",
        panels=ext.summarize(rows,("original","candidate")),reference_sha256=file_sha(previous_root/"predictions.json"))
    visible.atomic_json(dest/"predictions.json",rows)
    visible.atomic_json(dest/"results.json",result)
    print("HDF",result["panels"]["HDF"]["candidate"]["totals"],flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("panel",choices=("af","external"))
    args=parser.parse_args()
    folder=candidate()
    dest=OUT/"accepted"/args.panel
    if dest.exists() and any(dest.iterdir()):raise FileExistsError(dest)
    visible.configure(dest,"cuda");torch.set_num_threads(2);cv2.setNumThreads(8)
    runtime=FieldRuntime(folder)
    with torch.inference_mode():
        (af_panel if args.panel=="af" else external_panel)(folder,dest,runtime)
