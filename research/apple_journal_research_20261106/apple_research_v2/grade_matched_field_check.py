"""Recheck a frozen grade candidate inside the unmodified maturity and defect pipeline."""
import argparse
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torch

import benchmark_afruit_visible_quality as old
from matched_grade_loader import install
from afruit_visible_quality import QualityPipeline, ROOT, atomic_json, configure, read, sha


def paired_results(pipe, source, output):
    historical = {r["path"]: r for r in read(old.OUT / "test_predictions_1280.json")}
    if set(historical) != {r["path"] for r in source}:
        raise ValueError("Historical reference and new images differ")
    rows = []
    for rec in source:
        path = ROOT / rec["path"]
        if sha(path) != rec["sha256"]:
            raise ValueError(f"Changed test image: {path.name}")
        frame = old.prepare_input(old.market_image(path), 1280)
        scene, _ = pipe(frame)
        detections = scene["fruits"]
        linked = detections[0]["grade_id"] if len(detections) == 1 else None
        direct = pipe.classify([frame])[0]["grade_id"]
        previous = historical[rec["path"]]
        unchanged_boxes = (
            len(detections) == previous["fruit_count"]
            and all(f["crop"] == g["crop"] and f["maturity"] == g["maturity"]
                    for f, g in zip(detections, previous["fruits"]))
        )
        rows.append(dict(path=rec["path"], truth=rec["grade"], group=rec["group_id"],
                         linked_grade=linked, direct_grade=direct, previous_linked=previous["linked_grade"],
                         previous_direct=previous["direct_grade"], fruit_count=len(detections),
                         same_regions=unchanged_boxes, same_lesion_boxes=previous["defects"] == scene["defects"]))
    def counts(field):
        return old.quality_counts(rows, field)
    strata = {}
    for group in sorted({row["group"] for row in rows}):
        strata[group] = old.quality_counts([r for r in rows if r["group"] == group], "linked_grade")
    report = dict(auto=counts("linked_grade"), whole=counts("direct_grade"),
                  old_auto=counts("previous_linked"), old_whole=counts("previous_direct"),
                  per_group_auto=strata, same_regions=sum(r["same_regions"] for r in rows),
                  same_lesion_outputs=sum(r["same_lesion_boxes"] for r in rows),
                  frame_instance_counts=dict(Counter(r["fruit_count"] for r in rows)),
                  scope="Previously inspected fixed AFruitDB regression panel; not a new prospective test.")
    atomic_json(output / "paired_predictions.json", rows)
    atomic_json(output / "regression.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--speed-only", action="store_true")
    args = parser.parse_args()
    dest = args.output.resolve()
    if dest.exists() and any(dest.iterdir()):
        raise FileExistsError("Use a new output directory to preserve measured results")
    configure(dest, "cuda")
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    reference_report = read(old.OUT / "latency_market_frame_1280.json")
    samples = sorted((r for r in read(old.MANIFEST)["samples"] if r["split"] == "test"), key=lambda r: r["path"])
    timed_images = [(x["source"], Path(x["path"])) for x in reference_report["files"]]
    if {p.resolve() for _,p in timed_images} != {(ROOT / x["path"]).resolve() for x in old.spaced(samples,20)}:
        raise ValueError("Timing sample differs from fixed original benchmark")
    pipe = QualityPipeline()
    side = install(pipe,args.candidate)
    candidate_sha = sha(args.candidate/"weights.pt")
    atomic_json(dest / "receipt.json", dict(
        selection_sha256=sha(args.candidate / "selection.json"), weight_sha256=candidate_sha,
        original_grade=sha(old.GRADE_OUT / "weights.pt"), input_manifest=sha(old.MANIFEST),
        input_long_edge=1280, classifier_input_side=side,
        original_models=[old.STAGE_WEIGHT_SHA,old.WARM_SHA,old.SOURCE_SHA],
        gpu=torch.cuda.get_device_name(0), rounds=30, test_dataset="retrospective AFruitDB regression",
        speed="20 predecoded photos x 30 cycles; 5 warmup frames per mode. Includes CPU results and overlay; also records source downscale separately."))
    with torch.inference_mode():
        if not args.speed_only:
            result=paired_results(pipe,samples,dest)
            print("Regression, auto / whole",result["auto"]["confusion"],result["whole"]["confusion"],flush=True)
        # The original benchmark uses its own result directory; isolate this process's outputs.
        old.OUT=dest
        for name, known in (("scene",False),("single",True)):
            result=old.benchmark(pipe,name,timed_images,30,known=known,long_edge=1280)
            print(name,result["all"]["input_to_annotated_frame"],flush=True)


if __name__ == "__main__":
    main()
