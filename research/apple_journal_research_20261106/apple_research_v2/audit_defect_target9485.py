"""CPU-only fixed-prediction audit for the simultaneous precision/recall target."""

from collections import Counter, defaultdict
import math
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "apple_research_v2/joint"))
from joint_state_v1 import atomic_json, configure, sha
from audit_defect_causes_20261112 import ious, match

OUT = ROOT / "experiments/defect_target9485_audit"
PHASE = (
    ROOT
    / "experiments/defect_algorithm_steps_20261112_v2/phase1/source_predictions.json"
)
MODES = {
    "original_balanced": "class0_crop_deploy_v1/fp32/dev_predictions.json",
    "context_balanced": "defect_precision92_v1/balanced_predictions.json",
    "precision92": "defect_precision92_consensus_v1/agreement_plus_high_predictions.json",
}


def oracle(boxes, gt, same_class=True):
    """Maximum-cardinality assignment; IoU breaks ties but does not reduce recall."""
    if not len(boxes) or not gt:
        return set()
    ov = ious(boxes[:, :4], [g["box"] for g in gt])
    eligible = ov >= 0.5
    if same_class:
        eligible &= boxes[:, 5, None] == [g["class"] for g in gt]
    rows, cols = linear_sum_assignment(-(eligible * (len(gt) + 1) + ov))
    return {int(g) for d, g in zip(rows, cols) if eligible[d, g]}


def as_counts(tp, fp, n):
    return dict(
        TP=int(tp), FP=int(fp), FN=int(n - tp), P=tp / max(1, tp + fp), R=tp / n
    )


def oracle_prefix(evidence, n):
    """Derive exact attainable micro PR for two independent per-class score cuts.

    Called on dev only to disprove threshold-only solutions, never returns thresholds.
    Compress dominated prefixes with equal TP before combining the two classes.
    """
    curves = []
    for cls in (0, 1):
        counts = {0: 0}
        tp = fp = 0
        for score, (pos, neg) in sorted(evidence[cls].items(), reverse=True):
            tp += pos
            fp += neg
            counts.setdefault(tp, fp)
        curves.append(counts)
    pts = [
        as_counts(a + b, x + y, n)
        for a, x in curves[0].items()
        for b, y in curves[1].items()
    ]
    with_recall = [p for p in pts if p["R"] >= 0.85]
    with_precision = [p for p in pts if p["P"] >= 0.94]
    return dict(
        max_P_with_R85=max(with_recall, key=lambda p: (p["P"], p["R"]), default=None),
        max_R_with_P94=max(
            with_precision, key=lambda p: (p["R"], p["P"]), default=None
        ),
    )


def main():
    import json
    import torch
    from defect_precision92 import proposals
    from defect_coordinate_vote_20261118 import vote

    torch.set_num_threads(1)
    rows = json.loads(PHASE.read_text("utf8"))
    paths = {k: ROOT / "experiments" / v for k, v in MODES.items()}
    md = {
        k: {r["image_id"]: r for r in json.loads(v.read_text("utf8"))}
        for k, v in paths.items()
    }
    stages = defaultdict(
        lambda: dict(candidates=0, per_class=Counter(), ignore_class=0)
    )
    models = defaultdict(
        lambda: dict(
            tp=Counter(),
            fp=Counter(),
            fp_type=Counter(),
            covered_fn=Counter(),
            uncovered_fn=Counter(),
        )
    )
    ledger = []
    prefix = defaultdict(lambda: {c: defaultdict(lambda: [0, 0]) for c in (0, 1)})
    n_by_class = Counter()
    all_seen = set()

    for r in rows:
        sid, gt = r["image_id"], r["GT"]
        assert sid not in all_seen
        all_seen.add(sid)
        for m in md.values():
            assert m[sid]["GT"] == gt
        s = {k: np.array(v, np.float32).reshape(-1, 6) for k, v in r["sources"].items()}
        raw = np.concatenate([s[k] for k in ("o2o", "flip", "o2m")])
        proposed = proposals(s)
        vote_pool = vote(s, 0.5, [0.001, 0.001])
        fixed = {
            k: np.array(m[sid]["pred"], np.float32).reshape(-1, 6)
            for k, m in md.items()
        }
        hits = {k: set(match(p, gt)[0].values()) for k, p in fixed.items()}
        pools = {
            **s,
            "all_heads_views": raw,
            "original_flip_voted": vote_pool,
            "verifier_input": proposed,
        }
        oracles = {}
        for k, p in pools.items():
            h = oracle(p, gt)
            h_relax = oracle(p, gt, False)
            stages[k]["candidates"] += len(p)
            stages[k]["per_class"].update(gt[g]["class"] for g in h)
            stages[k]["ignore_class"] += len(h_relax)
            oracles[k] = h
        for k, p in {**fixed, "original_flip_voted": vote_pool}.items():
            pairs, ov = match(p, gt)
            for i, box in enumerate(p):
                c = int(box[5])
                prefix[k][c][float(box[4])][0 if i in pairs else 1] += 1
                if k not in fixed:
                    continue
                if i in pairs:
                    models[k]["tp"][c] += 1
                else:
                    models[k]["fp"][c] += 1
                    j = int(ov[i].argmax()) if gt else None
                    value = ov[i, j] if gt else 0
                    reason = (
                        "no_nearby_GT"
                        if value < 0.1
                        else "extent"
                        if value < 0.5
                        else "wrong_class"
                        if gt[j]["class"] != c
                        else "duplicate"
                    )
                    models[k]["fp_type"][f"{c}/{reason}"] += 1
        for j, g in enumerate(gt):
            c = g["class"]
            n_by_class[c] += 1
            # Per-GT independent coverage is used for attribution to avoid arbitrary
            # tie breaking among equivalent maximum-cardinality assignments.
            p = proposed[proposed[:, 5] == c]
            best = float(ious(p[:, :4], [g["box"]]).max()) if len(p) else 0
            for k, h in hits.items():
                if j not in h:
                    category = "covered_fn" if best >= 0.5 else "uncovered_fn"
                    models[k][category][c] += 1
            ledger.append(
                dict(
                    image_id=sid,
                    gt=j,
                    cls=c,
                    best_verifier_input_iou=best,
                    oracle_stages=[k for k, h in oracles.items() if j in h],
                    detected_modes=[k for k, h in hits.items() if j in h],
                )
            )
    assert len(all_seen) == 113 and n_by_class == {0: 87, 1: 143}
    for m in md.values():
        assert set(m) == all_seen
    n = sum(n_by_class.values())
    result = dict(
        dev_images=len(all_seen),
        gt=n_by_class,
        hashes={k: sha(path) for k, path in paths.items()},
        cached_source_sha=sha(PHASE),
        target_TP_min=math.ceil(0.85 * n),
        target_FP_max_at_min_TP=12,
        theoretical_assignment=stages,
        operating_points=models,
        oracle_classwise_prefix_on_saved_boxes={
            k: oracle_prefix(v, n) for k, v in prefix.items()
        },
    )
    atomic_json(OUT / "summary.json", result)
    atomic_json(OUT / "per_gt.json", ledger)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    configure(OUT, "cpu")
    main()
