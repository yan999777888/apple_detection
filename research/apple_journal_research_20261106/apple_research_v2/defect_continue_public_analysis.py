"""Audit GT identities, missing-candidate geometry and false positives at fixed thresholds."""

import argparse
from collections import Counter

from audit_defect_causes_20261112 import match
from class0_patch_refinement import atomic_json, read, sha
from defect_continue_public_training import OUT, PARENT, PRIMARY, mode_folder


def reason_for_miss(gt_index, target, low, overlaps):
    cls = target["class"]
    score = PRIMARY[cls]
    same = [
        (float(overlaps[i, gt_index]), p[4])
        for i, p in enumerate(low)
        if int(p[5]) == cls
    ]
    other = [
        float(overlaps[i, gt_index])
        for i, p in enumerate(low)
        if int(p[5]) != cls and p[4] >= PRIMARY[int(p[5])]
    ]
    support = [confidence for iou, confidence in same if iou >= 0.5]
    if support:
        return (
            "matched geometry below fixed score"
            if max(support) < score
            else "same-class candidate assigned to neighbouring GT"
        )
    if max(other, default=0) >= 0.5:
        return "classified as other class at deployment score"
    if max((iou for iou, _ in same), default=0) >= 0.20:
        return "partial same-class localization (0.20 <= IoU < 0.50)"
    return "no nearby same-class proposal at exported score floor"


def read_mode(out):
    fixed = read(out / "fixed_predictions.json")
    exported = read(out / "low_predictions.json")
    assert len(fixed) == len(exported) == 113
    gt_records = {}
    categories = Counter()
    fp = 0
    for row, lowrow in zip(fixed, exported):
        assert row["image_id"] == lowrow["image_id"] and row["GT"] == lowrow["GT"]
        hits, high_iou = match(row["pred"], row["GT"])
        matched = set(hits.values())
        _, low_iou = match(lowrow["pred"], row["GT"])
        for i, g in enumerate(row["GT"]):
            key = f"{row['image_id']}#{i}"
            hit = i in matched
            gt_records[key] = dict(
                cls=g["class"],
                bbox=g["box"],
                hit=hit,
                reason=None if hit else reason_for_miss(i, g, lowrow["pred"], low_iou),
            )
        for pred_index, p in enumerate(row["pred"]):
            if pred_index in hits:
                continue
            fp += 1
            cls = int(p[5])
            same = [
                float(high_iou[pred_index, i])
                for i, g in enumerate(row["GT"])
                if g["class"] == cls
            ]
            other = [
                float(high_iou[pred_index, i])
                for i, g in enumerate(row["GT"])
                if g["class"] != cls
            ]
            if max(same, default=0) >= 0.5:
                tag = "duplicate same-class detection"
            elif max(other, default=0) >= 0.5:
                tag = "overlaps reference of the other class"
            elif max(same, default=0) >= 0.20:
                tag = "partial localization of same-class reference"
            else:
                tag = "no overlapping same-class reference (IoU >= 0.20)"
            categories[f"{cls} | {tag}"] += 1
    scores = read(out / "metrics.json")["fixed"]["total"]
    assert (
        len(gt_records) == 230
        and fp == scores["FP"]
        and sum(v["hit"] for v in gt_records.values()) == scores["TP"]
    )
    return dict(
        gt=gt_records,
        false_positives=dict(categories),
        fixed=scores,
        prediction_sha=sha(out / "fixed_predictions.json"),
    )


def comparison(reference, candidate):
    base, other = reference["gt"], candidate["gt"]
    assert base.keys() == other.keys()
    transitions = []
    counts = Counter()
    causes = Counter()
    for k, a in base.items():
        b = other[k]
        assert a["cls"] == b["cls"] and a["bbox"] == b["bbox"]
        transition = (
            ("retained" if b["hit"] else "lost")
            if a["hit"]
            else ("recovered" if b["hit"] else "persistent miss")
        )
        cls = a["cls"]
        counts[f"{cls} | {transition}"] += 1
        if not b["hit"]:
            causes[f"{cls} | {b['reason']}"] += 1
        if transition in ("lost", "recovered"):
            transitions.append(
                dict(
                    id=k,
                    cls=cls,
                    transition=transition,
                    previous_reason=a["reason"],
                    new_reason=b["reason"],
                )
            )
    return dict(
        transitions=dict(counts),
        candidate_miss_diagnostics=dict(causes),
        changed_GT=transitions,
    )


def main(snapshot, baseline_only=False):
    sources = {"parent": PARENT.parent}
    if not baseline_only:
        sources.update(
            replay=mode_folder("replay"),
            external=mode_folder("external", snapshot=snapshot),
        )
    modes = {tag: read_mode(folder) for tag, folder in sources.items()}
    result = {}
    for tag, source in modes.items():
        result[tag] = dict(
            fixed=source["fixed"],
            false_positives=source["false_positives"],
            fixed_prediction_sha=source["prediction_sha"],
            miss_reasons=dict(
                Counter(
                    f"{v['cls']} | {v['reason']}"
                    for v in source["gt"].values()
                    if not v["hit"]
                )
            ),
        )
    if not baseline_only:
        for ref, candidate in (
            ("parent", "replay"),
            ("parent", "external"),
            ("replay", "external"),
        ):
            result[f"{ref}_to_{candidate}"] = comparison(modes[ref], modes[candidate])
    target = OUT / (
        "baseline_error_attribution.json"
        if baseline_only
        else "paired_error_attribution.json"
    )
    atomic_json(
        target,
        dict(
            fixed_scores=PRIMARY,
            matching_iou=0.50,
            geometry_floor=0.20,
            prediction_floor=0.001,
            note="Predictions without a matching annotation are counted against fixed original GT. Their biological truth needs a separate annotation review.",
            result=result,
        ),
    )
    print(target)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--snapshot", required=True)
    p.add_argument("--baseline-only", action="store_true")
    a = p.parse_args()
    main(a.snapshot, a.baseline_only)
