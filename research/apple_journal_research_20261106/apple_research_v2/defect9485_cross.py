"""Predeclared two-detector confirmation/ranking experiment using cached CPU outputs."""

import argparse
from collections import Counter
from pathlib import Path
import numpy as np

from audit_defect_target9485 import ROOT, PHASE, oracle
from audit_defect_causes_20261112 import ious
from compare_defect_target9485 import (
    BASELINES,
    indexed,
    load_candidates,
    inspect,
    delta,
    read,
)
from defect_coordinate_vote_20261118 import vote
from defect_algorithm_modes_20261112 import nms
from joint_state_v1 import sha, atomic_json, configure

OUT = ROOT / "experiments/defect9485_cross"
SET = ROOT / "experiments/defect9485_set/P94_predictions.json"


def extend(fixed, low):
    extra = []
    for cls in (0, 1):
        a = fixed[fixed[:, 5] == cls]
        b = low[low[:, 5] == cls]
        if len(a) and len(b):
            b = b[ious(b[:, :4], a[:, :4]).max(1) < 0.5]
        extra.append(b)
    p = np.concatenate([fixed, *extra])
    return p, np.arange(len(p)) < len(fixed)


def paired(primary, secondary):
    a, old = primary
    b, new = secondary
    if not len(a) or not len(b):
        return np.empty((0, 6), np.float32), np.empty(0, int)
    ov = ious(a[:, :4], b[:, :4])
    i, j = np.where((a[:, 5, None] == b[None, :, 5]) & (ov >= 0.5))
    level = old[i].astype(int) + new[j].astype(int)
    order = np.lexsort((j, i, -(a[i, 4] * b[j, 4]), -ov[i, j], -level))
    used_a = set()
    used_b = set()
    pair = []
    levels = []
    for pos in order:
        k, l = i[pos], j[pos]
        if k in used_a or l in used_b:
            continue
        p = a[k].copy()
        p[:4] = 0.5 * (p[:4] + b[l, :4])
        p[4] = np.sqrt(p[4] * b[l, 4])
        pair.append(p)
        levels.append(level[pos])
        used_a.add(k)
        used_b.add(l)
    return np.array(pair, np.float32).reshape(-1, 6), np.array(levels, int)


def with_supplement(p, supp):
    if len(p) and len(supp):
        overlap = ious(supp[:, :4], p[:, :4])
        supp = supp[~((overlap >= 0.5) & (supp[:, 5, None] == p[None, :, 5])).any(1)]
    primary = p.copy()
    other = supp.copy()
    primary[:, 4] = 0.5 + 0.5 * primary[:, 4]
    other[:, 4] *= 0.5
    return nms(np.concatenate([primary, other]))


def run(args):
    src = indexed(read(PHASE))
    strong = load_candidates(BASELINES["original_balanced"], src)
    high = load_candidates(BASELINES["precision92"], src)
    set_fixed = load_candidates(SET, src)
    secondary = load_candidates(args.secondary_fixed, src)
    low = load_candidates(args.secondary_low, src)
    names = (
        "both_fixed",
        "one_fixed_plus_other_support",
        "one_fixed_plus_other_support_and_frozen_set",
    )
    results = {k: {} for k in names}
    relaxed = {}
    relaxed_set = {}
    structure = Counter()
    for sid, r in src.items():
        low0 = vote(r["sources"], 0.5, [0.055, 0.055])
        a = extend(strong[sid], low0)
        b = extend(secondary[sid], low[sid][low[sid][:, 4] >= 0.055])
        p, level = paired(a, b)
        for v in (0, 1, 2):
            structure[f"{v}_frozen_votes"] += int(sum(level == v))
        relaxed[sid] = nms(p)
        p2 = nms(p[level == 2])
        p1 = nms(p[level >= 1])
        s = with_supplement(p1, set_fixed[sid])
        relaxed_set[sid] = np.concatenate([relaxed[sid], set_fixed[sid]])
        for k, v in zip(names, [p2, p1, s]):
            results[k][sid] = v
        for name, v in [
            ("paired", relaxed[sid]),
            ("paired_union_set", relaxed_set[sid]),
        ]:
            for j in oracle(v, r["GT"]):
                structure[f"oracle_{name}_class{r['GT'][j]['class']}"] += 1
    for k, p in [("primary", strong), ("high", high)]:
        stats, hits, *_ = inspect(p, src)
        if k == "primary":
            ph = hits
        else:
            hh = hits
    report = {}
    for k, outputs in results.items():
        stats, hits, fps, curves = inspect(outputs, src)
        report[k] = dict(
            fixed=stats,
            fp_attribution=fps,
            vs_balanced=delta(hits, ph, src),
            vs_high=delta(hits, hh, src),
            threshold_limits_of_fixed_output=curves,
        )
        atomic_json(
            OUT / args.arm / f"{k}_predictions.json",
            [
                dict(image_id=sid, GT=r["GT"], pred=outputs[sid].tolist())
                for sid, r in src.items()
            ],
        )
    # Unthresholded paired ranking is a geometry/ordering diagnostic, not the frozen union.
    _, _, _, envelope = inspect(relaxed, src)
    report["relaxed_all_pairs_diagnostic"] = envelope
    report["candidate_structure"] = structure
    report["input_sha256"] = {
        str(p.relative_to(ROOT)): sha(p)
        for p in (
            PHASE,
            BASELINES["original_balanced"],
            SET,
            args.secondary_fixed.resolve(),
            args.secondary_low.resolve(),
            OUT / "protocol.json",
        )
    }
    atomic_json(OUT / args.arm / "report.json", report)
    print({k: report[k]["fixed"]["total"] for k in names})
    print(structure, envelope["classwise_overall"])


if __name__ == "__main__":
    arg = argparse.ArgumentParser()
    arg.add_argument("--arm", choices=["p2", "yolo26s", "yolo26s_flip"], required=True)
    arg.add_argument("--secondary-fixed", type=Path, required=True)
    arg.add_argument("--secondary-low", type=Path, required=True)
    args = arg.parse_args()
    configure(OUT, "cpu")
    run(args)
