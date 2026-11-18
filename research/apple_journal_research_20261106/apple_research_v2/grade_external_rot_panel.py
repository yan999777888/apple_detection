"""Check frozen quality predictions on other public photos with a rot/fresh label."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps
import torch

from afruit_grade_release import OUT as MODEL_DIR, ROOT, batch_preprocess, decide, file_sha, load_model
from class0_patch_refinement import atomic_json


OUT = ROOT / "experiments/afruit_external_appearance_v1"
SPLIT = ROOT / "diagnostics/hdf_peripheral_v1_s20261106/frozen_split_manifest.json"
AFRUIT = ROOT / "datasets/afruitdb_apples_v1/manifest.json"
TYPES = ("fresh", "rot_defect")


def freeze():
    rows = json.loads(SPLIT.read_bytes())["representatives"]
    selected = {}
    for row in sorted(rows, key=lambda r: r["sha256"]):
        if row["split"] != "development" or row["quarantined"] or row["author_class"] not in TYPES:
            continue
        key = row["recommended_proxy_group_id"]
        if key in selected and selected[key]["author_class"] != row["author_class"]:
            raise ValueError("Conflicting source labels in a proxy group")
        selected.setdefault(key, row)
    cases = [{"path": Path(row["image_path"]).relative_to(ROOT).as_posix(),
              "source_label": row["author_class"], "sha256": row["sha256"], "group": group}
             for group, row in sorted(selected.items())]
    for case in cases:
        if file_sha(ROOT / case["path"]) != case["sha256"]:
            raise ValueError("Source image changed")
    protocol = dict(source="HDF real-only, pre-existing development split", split_sha256=file_sha(SPLIT),
                    selection="lowest source SHA per eligible approximate group; no performance selection",
                    note="fresh/rot are source appearance labels, not project quality grades. "
                         "This panel reports model outputs, not grade accuracy. No physical fruit ID.",
                    cases=cases)
    path = OUT / "protocol.json"
    if path.exists() and json.loads(path.read_bytes()) != protocol:
        raise ValueError("Existing panel differs")
    atomic_json(path, protocol)
    print({label: sum(c["source_label"] == label for c in cases) for label in TYPES})


@torch.inference_mode()
def predict():
    protocol = json.loads((OUT / "protocol.json").read_bytes())
    rows = protocol["cases"]
    afruit_shas = {r["sha256"] for r in json.loads(AFRUIT.read_bytes())["samples"]}
    kept = [r for r in rows if r["sha256"] not in afruit_shas]
    model, cutoff = load_model("cpu")
    counts = {label: [0, 0, 0] for label in TYPES}
    for i in range(0, len(kept), 12):
        batch = kept[i:i + 12]
        frames = []
        for item in batch:
            with Image.open(ROOT / item["path"]) as im:
                frames.append(np.asarray(ImageOps.exif_transpose(im).convert("RGB"))[:, :, ::-1].copy())
        grades = decide(model(batch_preprocess(frames, "cpu")), cutoff).tolist()
        for r, g in zip(batch, grades, strict=True):
            counts[r["source_label"]][g] += 1
    report = dict(protocol_sha256=file_sha(OUT / "protocol.json"), model_sha256=file_sha(MODEL_DIR / "weights.pt"),
                  threshold=cutoff, grade_columns=["good", "mild", "severe"], by_source_label=counts,
                  removed_exact_afruit_overlap=len(rows) - len(kept), near_overlap_across_sources_verified=False,
                  scope=protocol["note"], input="full image with deployment preprocessing, CPU FP32")
    atomic_json(OUT / "results.json", report)
    print(report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["freeze", "predict"])
    args = parser.parse_args()
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    OUT.mkdir(parents=True, exist_ok=True)
    (freeze if args.action == "freeze" else predict)()
