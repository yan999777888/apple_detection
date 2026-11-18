"""Independent acceptance of an updated scratch head with numerically frozen insect outputs."""

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import math
from pathlib import Path

from class0_patch_refinement import ROOT, atomic_json, read, sha
from audit_defect_causes_20261112 import match
from run_defect_zero_train_20261112 import metrics

EXP = ROOT / "experiments"
WARM = EXP / "defect_continue_warm_rank"
CANDIDATE = EXP / "defect_continue_scratch_specialist"
MANIFEST = ROOT / "apple_research_v2/datasets/ssda_date_v2/manifest.json"
OLD29 = EXP / "class0_visual_review_v1/audit.json"


def canonical():
    rows = read(MANIFEST)["samples"]
    training_dates = {
        r["capture_filename_tokens"]["date"] for r in rows if r["partition"] == "train"
    }
    dev = {}
    for row in rows:
        if row["partition"] != "dev":
            continue
        assert row["image_id"] not in dev
        h, w = row["native_hw"]
        gt = []
        for v in row["instances"]:
            x, y, bw, bh = v["xywh_normalized"]
            gt.append(
                dict(
                    box=[
                        (x - bw / 2) * w,
                        (y - bh / 2) * h,
                        (x + bw / 2) * w,
                        (y + bh / 2) * h,
                    ],
                    **{"class": v["class_id"]},
                )
            )
        dev[row["image_id"]] = (row["capture_filename_tokens"]["date"], gt)
    if len(dev) != 113 or sum(len(v[1]) for v in dev.values()) != 230:
        raise ValueError("Unexpected canonical split")
    return dev, training_dates


def validate(rows, dev):
    if len(rows) != len(dev) or {r["image_id"] for r in rows} != set(dev):
        raise ValueError("Missing, duplicate or extraneous development image")
    for row in rows:
        if row["GT"] != dev[row["image_id"]][1]:
            raise ValueError(f"Canonical GT differs at {row['image_id']}")
        for p in row["pred"]:
            if (
                len(p) != 6
                or not all(math.isfinite(v) for v in p)
                or not (0 <= p[4] <= 1 and p[5] in (0, 1))
                or p[0] >= p[2]
                or p[1] >= p[3]
            ):
                raise ValueError("Malformed deployment candidate")
    return {r["image_id"]: r for r in rows}


def expect_locked(a, b, label):
    for key in a:
        x = sorted(tuple(p) for p in a[key]["pred"] if p[5] == 1)
        y = sorted(tuple(p) for p in b[key]["pred"] if p[5] == 1)
        if x != y:
            raise ValueError(f"Frozen class1 changed at {key} ({label})")


def expect_prefix(fixed, low, thresholds):
    for key, row in low.items():
        prefix = sorted(tuple(p) for p in row["pred"] if p[4] >= thresholds[int(p[5])])
        if prefix != sorted(tuple(p) for p in fixed[key]["pred"]):
            raise ValueError(
                f"Fixed candidates differ from their low-score source at {key}"
            )


def counts(tp, fp, n):
    return dict(
        TP=tp,
        FP=fp,
        FN=n - tp,
        GT=n,
        P=tp / (tp + fp) if tp + fp else None,
        R=tp / n if n else None,
    )


def frontier(low, insect):
    scores = defaultdict(lambda: [0, 0])
    for row in low.values():
        p = [v for v in row["pred"] if v[5] == 0]
        g = [v for v in row["GT"] if v["class"] == 0]
        matches, _ = match(p, g)
        for i, v in enumerate(p):
            scores[v[4]][int(i not in matches)] += 1
    tp, fp = insect["TP"], insect["FP"]
    pts = [dict(counts(tp, fp, 230), scratch_threshold=None)]
    for score, delta in sorted(scores.items(), reverse=True):
        tp += delta[0]
        fp += delta[1]
        pts.append(dict(counts(tp, fp, 230), scratch_threshold=score))
    r85 = [p for p in pts if p["R"] >= 0.85]
    p94 = [p for p in pts if p["P"] >= 0.94]
    return dict(
        P_at_R85=max(r85, key=lambda p: (p["P"], p["R"])) if r85 else None,
        R_at_P94=max(p94, key=lambda p: (p["R"], p["P"])) if p94 else None,
        any_dual_target_on_dev=any(p["P"] >= 0.94 and p["R"] >= 0.85 for p in pts),
        score_levels=len(scores),
        scope="diagnostic only, fixed insect; no change to deployed threshold",
    )


def hit_ids(rows, cls=None):
    hits = set()
    for key, row in rows.items():
        paired, _ = match(row["pred"], row["GT"])
        for i in paired.values():
            if cls is None or row["GT"][i]["class"] == cls:
                hits.add((key, i))
    return hits


def timing(speed):
    if speed.get("n") != 600 or any(
        not math.isfinite(speed[k]) or speed[k] <= 0
        for k in ("mean_ms", "serial_fps", "p95_ms")
    ):
        raise ValueError("Incomplete 600-frame timing or invalid latency")
    if not math.isclose(speed["mean_ms"] * speed["serial_fps"], 1000.0, rel_tol=1e-5):
        raise ValueError("Mean and FPS do not describe the same serialized timing")
    return speed


def audit(folder=CANDIDATE, thresholds=(0.40, 0.34), selftest=False):
    dev, dates = canonical()
    ref_fixed = validate(read(WARM / "fixed_predictions.json"), dev)
    ref_low = validate(read(WARM / "low_predictions.json"), dev)
    fixed = validate(read(folder / "fixed_predictions.json"), dev)
    low = validate(read(folder / "low_predictions.json"), dev)
    expect_locked(fixed, ref_fixed, "fixed")
    expect_locked(low, ref_low, "low")
    expect_prefix(fixed, low, thresholds)
    speed = timing(read(folder / "speed.json"))
    if selftest:

        def rejected(fn):
            try:
                fn()
            except ValueError:
                return
            raise AssertionError("Invalid result accepted")

        rows = list(fixed.values())
        rejected(lambda: validate(rows[:-1], dev))
        rejected(lambda: validate(rows[:-1] + [rows[0]], dev))
        altered = deepcopy(rows)
        record = next(r for r in altered if r["GT"])
        record["GT"][0]["box"][0] = math.nextafter(record["GT"][0]["box"][0], math.inf)
        rejected(lambda: validate(altered, dev))
        for stream in (fixed, low):
            bad = deepcopy(stream)
            p = next(p for row in bad.values() for p in row["pred"] if p[5] == 1)
            p[4] = math.nextafter(p[4], math.inf)
            rejected(lambda: expect_locked(stream, bad, "selftest"))
            rev = {
                key: dict(r, pred=list(reversed(r["pred"])))
                for key, r in stream.items()
            }
            expect_locked(stream, rev, "permuted")
        rejected(lambda: timing(dict(speed, n=599)))
        print(
            "Original warm fixtures validated; missing/duplicate images, 1-ULP reference/head drift, incomplete timing refused."
        )
        return
    summary = metrics(list(fixed.values()))
    if summary["class"]["1"] != {
        "TP": 123,
        "FP": 7,
        "GT": 143,
        "P": 123 / 130,
        "R": 123 / 143,
    }:
        raise ValueError("Insect counts do not reproduce the locked working point")
    out = dict(
        source_hashes={
            name: sha(folder / name)
            for name in ("fixed_predictions.json", "low_predictions.json", "speed.json")
        },
        reference_hashes={
            name: sha(WARM / name)
            for name in ("fixed_predictions.json", "low_predictions.json")
        },
        manifest_sha256=sha(MANIFEST),
        thresholds=thresholds,
        canonical_images=len(fixed),
        canonical_GT=230,
        IoU=0.5,
        unchanged_fixed_and_low_insect_predictions=True,
        metrics=summary,
        speed=speed,
        retrospective_frontier=frontier(low, summary["class"]["1"]),
        date_splits={},
        versus={},
    )
    by_date = {
        d: {k for k, (date, _) in dev.items() if date == d}
        for d in sorted({v[0] for v in dev.values()})
    }
    for key, ids in {
        **by_date,
        "same_date_as_train": {k for k, (d, _) in dev.items() if d in dates},
        "different_date_from_train": {k for k, (d, _) in dev.items() if d not in dates},
    }.items():
        rows = [r for k, r in fixed.items() if k in ids]
        m = metrics(rows) if rows else None
        out["date_splits"][key] = dict(
            images=len(rows),
            total=m["total"] if m else None,
            classes=m["class"] if m else None,
        )
    hits = hit_ids(fixed, 0)
    cases = read(OLD29)["cases"]["miss"]
    assert len(cases) == 29
    for case in cases:
        g = dev[case["image"]][1][case["index"]]
        if g["class"] != 0 or g["box"] != case["box"]:
            raise ValueError("A historical 29-case reference changed")
    out["historical_29"] = [
        dict(
            id=c["code"],
            image=c["image"],
            gt_index=c["index"],
            hit=(c["image"], c["index"]) in hits,
        )
        for c in cases
    ]
    for tag, ref in (
        ("warm", ref_fixed),
        (
            "parent",
            validate(read(EXP / "defect9485_small_v1/fixed_predictions.json"), dev),
        ),
        (
            "replay",
            validate(
                read(EXP / "defect_continue_public_v1/replay/fixed_predictions.json"),
                dev,
            ),
        ),
    ):
        a = hit_ids(ref)
        b = hit_ids(fixed)
        out["versus"][tag] = {
            name: [
                dict(image=k, id=i, cls=dev[k][1][i]["class"]) for k, i in sorted(ids)
            ]
            for name, ids in (("gained", b - a), ("lost", a - b))
        }
    atomic_json(folder / "independent_audit.json", out)
    print(summary["total"], out["retrospective_frontier"], speed)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--folder", type=Path, default=CANDIDATE)
    p.add_argument("--scratch-threshold", type=float, default=0.40)
    a = p.parse_args()
    audit(
        WARM if a.self_test else a.folder,
        (0.40 if a.self_test else a.scratch_threshold, 0.34),
        a.self_test,
    )
