"""Audit a frozen defect operating point and the joint PR limits of its saved candidates.

Input JSON is a list of rows with image_id and pred (xyxy, score, class). Supplied GT
is checked against the canonical 113-image development split; never re-evaluate
against candidate-provided GT. Run from the project's F: drive.
"""

import argparse
from collections import Counter, defaultdict
from pathlib import Path
import json
import math
import re

import numpy as np

from audit_defect_target9485 import PHASE, ROOT, MODES, oracle_prefix
from audit_defect_causes_20261112 import match
from joint_state_v1 import atomic_json, sha

BASELINES = {
    k: ROOT / "experiments" / MODES[k] for k in ("original_balanced", "precision92")
}


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def indexed(rows):
    items = {r["image_id"]: r for r in rows}
    if len(items) != len(rows):
        raise ValueError("Duplicate image ID")
    return items


def load_candidates(path, canonical):
    rows = indexed(read(path))
    if rows.keys() != canonical.keys():
        raise ValueError(f"Image coverage mismatch in {path}")
    result = {}
    for sid, row in rows.items():
        if "GT" in row:
            gt = row["GT"]
            expected = canonical[sid]["GT"]
            if [v["class"] for v in gt] != [
                v["class"] for v in expected
            ] or not np.allclose(
                [v["box"] for v in gt], [v["box"] for v in expected], atol=1e-5, rtol=0
            ):
                raise ValueError(f"Changed GT in {path}: {sid}")
        p = np.asarray(row["pred"], dtype=float)
        if p.shape == (0,):
            p = p.reshape(0, 6)
        if p.ndim != 2 or p.shape[1] != 6 or not np.isfinite(p).all():
            raise ValueError(f"Invalid candidate matrix: {sid}")
        if np.any(~np.isin(p[:, 5], [0, 1])) or np.any((p[:, 4] < 0) | (p[:, 4] > 1)):
            raise ValueError(f"Invalid score or category: {sid}")
        if np.any(p[:, 2:4] < p[:, :2]):
            raise ValueError(f"Reversed box: {sid}")
        result[sid] = p
    return result


def count_point(t, f, n):
    return dict(
        TP=int(t),
        FP=int(f),
        FN=int(n - t),
        GT=int(n),
        P=float(t / (t + f)) if t + f else 0.0,
        R=float(t / n) if n else 0.0,
    )


def target(p):
    return (
        100 * p["TP"] >= 94 * (p["TP"] + p["FP"])
        and 100 * p["TP"] >= 85 * p["GT"]
        and p["TP"] > 0
    )


def shared_threshold_envelope(evidence, total_gt):
    t = f = 0
    points = []
    for score, (ok, bad) in sorted(evidence.items(), reverse=True):
        t += ok
        f += bad
        points.append(count_point(t, f, total_gt))
    r85 = [p for p in points if 100 * p["TP"] >= 85 * total_gt]
    p94 = [p for p in points if 100 * p["TP"] >= 94 * (p["TP"] + p["FP"])]
    return dict(
        distinct_nonempty_prefixes=len(points),
        both_achievable=any(target(p) for p in points),
        max_P_at_R85=max(r85, key=lambda p: (p["P"], p["R"]), default=None),
        max_R_at_P94=max(p94, key=lambda p: (p["R"], p["P"]), default=None),
    )


def partition(sid):
    d = re.search(r"_(20\d{6})_", sid)
    if not d:
        raise ValueError(f"Missing capture date in {sid}")
    return d.group(1)


def inspect(predictions, canonical):
    counts = defaultdict(Counter)
    evidence = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    ledger = set()
    fp_types = Counter()
    for sid, r in canonical.items():
        gt = r["GT"]
        date = partition(sid)
        p = predictions[sid]
        pairs, ov = match(p, gt)
        for gi, g in enumerate(gt):
            for name in (
                "total",
                f"class/{g['class']}",
                f"date/{date}",
                f"date/{date}/class/{g['class']}",
            ):
                counts[name]["GT"] += 1
        for pi, b in enumerate(p):
            cls = int(b[5])
            correct = pi in pairs
            if correct:
                ledger.add((sid, pairs[pi]))
            else:
                j = int(ov[pi].argmax()) if gt else None
                v = ov[pi, j] if gt else 0
                reason = (
                    "no_nearby_GT"
                    if v < 0.1
                    else "extent"
                    if v < 0.5
                    else "wrong_class"
                    if gt[j]["class"] != cls
                    else "duplicate"
                )
                fp_types[f"class{cls}/{reason}"] += 1
            for name in (
                "total",
                f"class/{cls}",
                f"date/{date}",
                f"date/{date}/class/{cls}",
            ):
                counts[name]["TP" if correct else "FP"] += 1
                evidence[name][float(b[4])][0 if correct else 1] += 1
    stats = {
        k: count_point(s["TP"], s["FP"], s["GT"]) for k, s in sorted(counts.items())
    }
    envelopes = {
        k: shared_threshold_envelope(evidence[k], s["GT"]) for k, s in stats.items()
    }
    separate = oracle_prefix(
        {c: evidence[f"class/{c}"] for c in (0, 1)}, stats["total"]["GT"]
    )
    separate["both_achievable"] = (
        separate["max_P_with_R85"] is not None
        and separate["max_P_with_R85"]["P"] >= 0.94
    )
    return (
        stats,
        ledger,
        fp_types,
        dict(shared_by_group=envelopes, classwise_overall=separate),
    )


def delta(new, old, canonical):
    def describe(changes):
        changes = sorted(changes)
        return dict(
            count=len(changes),
            by_class=Counter(canonical[sid]["GT"][gi]["class"] for sid, gi in changes),
            by_date=Counter(partition(sid) for sid, gi in changes),
            gt_ids=[f"{sid}#{gi}" for sid, gi in changes],
        )

    return dict(gained=describe(new - old), lost=describe(old - new))


def run(args):
    canonical = indexed(read(PHASE))
    n = sum(len(s["GT"]) for s in canonical.values())
    if len(canonical) != 113 or n != 230:
        raise ValueError("Unexpected evaluation set")
    fixed = load_candidates(args.fixed, canonical)
    fixed_stats, new_ids, fp_types, limited_curve = inspect(fixed, canonical)
    candidates = args.candidates or args.fixed
    diag = (
        limited_curve
        if candidates == args.fixed
        else inspect(load_candidates(candidates, canonical), canonical)[3]
    )
    comparisons = {}
    for k, p in BASELINES.items():
        stats, ids, _, _ = inspect(load_candidates(p, canonical), canonical)
        comparisons[k] = dict(
            source_sha256=sha(p),
            previous=stats["total"],
            **delta(new_ids, ids, canonical),
        )
    speed = None
    if args.speed:
        speed = read(args.speed)
        if args.speed_key:
            speed = speed[args.speed_key]
        for key in ("n", "mean_ms", "p95_ms", "serial_fps"):
            if not math.isfinite(speed[key]) or speed[key] <= 0:
                raise ValueError(f"Invalid speed record: {key}")
    output = args.output.resolve()
    if output.drive.lower() != "f:" or not output.is_relative_to(ROOT / "experiments"):
        raise ValueError(
            "Choose an output JSON in this project's F: experiments hierarchy"
        )
    protected = {
        p.resolve() for p in [PHASE, args.fixed, candidates, *BASELINES.values()]
    }
    if args.speed:
        protected.add(args.speed.resolve())
    if output in protected:
        raise ValueError("Choose an output distinct from inputs")
    atomic_json(
        output,
        dict(
            canonical_sha256=sha(PHASE),
            fixed_sha256=sha(args.fixed),
            candidates_sha256=sha(candidates),
            fixed_source=str(args.fixed),
            candidate_source=str(candidates),
            tested_images=len(canonical),
            tested_gt=n,
            speed_source=str(args.speed) if args.speed else None,
            speed_sha256=sha(args.speed) if args.speed else None,
            speed_key=args.speed_key,
            reported_speed=speed,
            fixed_metrics=fixed_stats,
            fixed_meets_precision_recall=target(fixed_stats["total"]),
            fixed_fp_types=fp_types,
            operating_point_comparisons=comparisons,
            development_PR_diagnostic=diag,
            diagnostic_scope="All distinct score prefixes of supplied proposals after their upstream filtering; ties enter together. Shared and classwise thresholds are reported separately as retrospective diagnostics only.",
            candidates_default_to_final_output=args.candidates is None,
        ),
    )
    print(
        json.dumps(
            dict(
                fixed=fixed_stats["total"],
                counts_validated=len(canonical),
                classwise=diag["classwise_overall"],
                shared=diag["shared_by_group"]["total"],
            ),
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fixed", required=True, type=Path)
    p.add_argument(
        "--candidates",
        type=Path,
        help="Same model's low-floor candidates before its frozen deployment cutoff",
    )
    p.add_argument("--speed", type=Path)
    p.add_argument("--speed-key")
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()
    run(args)
