"""Read-only, training-point diagnostics for the prospective center-proposal ablation."""

import argparse
from collections import defaultdict

import cv2
import torch

from defect_blind_center_add import (
    CUTS,
    OUT,
    REVIEW,
    attach,
    base_network,
    decode,
    maps,
    model,
    replay,
    split_training,
)
from class0_patch_refinement import ROOT, atomic_json, configure, load_image, read


def at_point(boxes, point):
    if len(boxes) == 0:
        return []
    cx, cy = point
    take = (
        (boxes[:, 0] <= cx)
        & (cx <= boxes[:, 2])
        & (boxes[:, 1] <= cy)
        & (cy <= boxes[:, 3])
    )
    return [{"class": int(row[5]), "score": float(row[4])} for row in boxes[take]]


@torch.inference_mode()
def inspect():
    src, _, audit = split_training()
    assert read(OUT / "protocol.json")["export_sha256"] == audit["export_sha256"]
    images = {r["image_id"]: r for r in src}
    prior = read(
        ROOT / "experiments/defect_review_impact_v1/training_point_behavior.json"
    )
    assert prior["export_sha"] == audit["export_sha256"]
    lower = {
        row["id"]: row["models"]["specialist"]["at_point"] for row in prior["records"]
    }
    detectors = {name: model(name) for name in ("control", "review")}
    det = base_network()
    examples = []
    counts = {name: {cut: defaultdict(int) for cut in CUTS} for name in detectors}
    for item in read(REVIEW)["rows"]:
        response = item["manual"]
        if item["kind"] != "unmatched" or response not in (
            "scratch",
            "insect",
            "other",
        ):
            continue
        image = load_image(images[item["image_id"]]["rgb_path"])
        grid, base, transform = maps(det, image)
        center = item["center_xy"]
        cls = {"scratch": 0, "insect": 1, "other": -1}[response]
        out = dict(
            id=item["id"], response=response, position=item["position"], models={}
        )
        source_missing = cls >= 0 and not any(
            row["cls"] == response for row in lower[item["id"]]
        )
        out["previously_missing_at_candidate_floor"] = source_missing
        for name, net in detectors.items():
            candidates = decode(net, grid, transform, image.shape[:2])
            scores = at_point(attach(base, candidates, min(CUTS))[len(base) :], center)
            out["models"][name] = scores
            for cut in CUTS:
                stat = counts[name][cut]
                tags = {row["class"] for row in scores if row["score"] >= cut}
                stat["previously_missing_positive_count"] += int(source_missing)
                stat["newly_covered_previously_missing"] += int(
                    source_missing and cls in tags
                )
                stat["human_other_count"] += int(cls == -1)
                stat["human_other_with_new_detection"] += int(cls == -1 and bool(tags))
        examples.append(out)
    atomic_json(
        OUT / "training_probe.json",
        dict(
            scope="Fit examples only. No new manual boxes. Coverage means the predicted box contains the reviewed pixel; it does not measure localization precision.",
            counts=counts,
            samples=examples,
        ),
    )
    print({k: dict(v[min(CUTS)]) for k, v in counts.items()}, flush=True)


def summarize():
    results = read(OUT / "training_probe.json")
    previous = read(ROOT / "experiments/defect_blind_review_update_v1/coverage.json")[
        "coverage"
    ]
    blocked = {
        v["case"] for v in previous if v["response"] >= 0 and 1 not in v["labels"]
    }
    rows = [r for r in results["samples"] if r["id"] in blocked]
    assert len(rows) == len(blocked)
    counts = {}
    for arm in ("control", "review"):
        counts[arm] = {}
        for cut in CUTS:
            found = []
            for r in rows:
                cls = {"scratch": 0, "insect": 1}[r["response"]]
                if any(
                    p["class"] == cls and p["score"] >= cut for p in r["models"][arm]
                ):
                    found.append(r["position"])
            counts[arm][cut] = found
    out = dict(
        measure="Fit location covered by an independently generated new box; no manually measured extent.",
        previous_rescorer_without_eligible_same_class=len(rows),
        coverage=counts,
    )
    atomic_json(OUT / "qualification_probe.json", out)
    print(out, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action", choices=["inspect", "summarize"], default="inspect", nargs="?"
    )
    action = parser.parse_args().action
    configure(OUT, "cuda" if action == "inspect" else "cpu")
    cv2.setNumThreads(1)
    torch.set_num_threads(2)
    with replay.exclusive_run(OUT):
        if action == "inspect":
            inspect()
        summarize()
