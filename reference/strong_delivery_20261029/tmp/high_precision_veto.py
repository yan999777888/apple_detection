"""One-sided high-precision veto on top of the locked four-view fusion.

Design constraints that make this safe to tune on 814 images:

* the candidate pool is exactly the Base+Small four-view fusion, so no external
  model can inject a new box; the extra models only *subtract* evidence;
* every rule is a ReLU on an anomaly margin, so a normal apple keeps its score
  bit-for-bit and the balanced operating point is preserved;
* every parameter set is chosen on one parity fold and reported on the other.

Target operating point: Recall >= 0.90, maximise Precision.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).absolute().parents[1]
sys.path.insert(0, str(ROOT / "tmp"))
import flip_tta  # noqa: E402
from flip_tta import metrics  # noqa: E402

RECALL_TARGET = 0.90

MAIN_VIEWS = [
    "runs/od_step4_convnextbase_c2_11/revalidation_live/raw_predictions.json",
    "runs/flip_tta_convnextbase_c2_2304/hflip/raw_predictions.json",
    "runs/od_step3_convnext_joint2_11/validation/update_002304_1793090722953859300/raw_predictions.json",
    "runs/flip_tta_convnext_joint2_2304/hflip/raw_predictions.json",
]
VIEW_WEIGHTS = [0.65, 0.65, 0.35, 0.35]
CONSENSUS_GROUPS = [0, 0, 1, 1]
VERIFIERS = {
    "v1280": "runs/od_step4_convnextbase_c2_11/view_1280/raw_predictions.json",
    "vr50": "runs/od_step1_vfl_ignore_joint4_11/validation/update_002304_1793008380974163600/raw_predictions.json",
    "vswin": "runs/od_step5_swins_c1_11/validation/update_002304_1793199196467574400/raw_predictions.json",
}


def logit(score):
    clipped = np.clip(np.asarray(score, dtype=np.float64), 1e-6, 1 - 1e-6)
    return np.log(clipped / (1 - clipped))


def sigmoid(value):
    return 1 / (1 + np.exp(-value))


def intersection_matrix(boxes):
    """Exact pairwise intersection area; IoU cannot be reused for containment."""
    x1 = np.maximum(boxes[:, None, 0], boxes[None, :, 0])
    y1 = np.maximum(boxes[:, None, 1], boxes[None, :, 1])
    x2 = np.minimum(boxes[:, None, 2], boxes[None, :, 2])
    y2 = np.minimum(boxes[:, None, 3], boxes[None, :, 3])
    return np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)


def box_area(boxes):
    return np.maximum(boxes[:, 2] - boxes[:, 0], 0) * np.maximum(boxes[:, 3] - boxes[:, 1], 0)


def load(path):
    return flip_tta.load_records(str(ROOT / path))


def build_candidates(main_layers, verifier_layers, match_iou=0.45):
    """Fuse the locked pool once and attach every veto feature to each cluster."""
    candidates = []
    for index in range(len(main_layers[0])):
        entries = [layer[index] for layer in main_layers]
        image_id = entries[0]["image_id"]
        meta = []
        boxes, scores = flip_tta.fuse_entries(
            entries, iou_threshold=match_iou, score_mode="mean", box_mode="score_weighted",
            view_weights=VIEW_WEIGHTS, consensus_groups=CONSENSUS_GROUPS,
            consensus_bonus=1.10, isolation_penalty=0.90, cluster_metadata=meta)
        verifier_boxes, verifier_scores = {}, {}
        for name, layer in verifier_layers.items():
            record = layer[index]
            if record["image_id"] != image_id:
                raise RuntimeError("Verifier pool image order differs: " + name)
            verifier_boxes[name] = np.asarray(record["pred"]["boxes"], dtype=np.float64).reshape(-1, 4)
            verifier_scores[name] = np.asarray(record["pred"]["scores"], dtype=np.float64)
        rows = []
        for position in range(len(boxes)):
            info = meta[position]
            member_boxes = np.asarray(info["member_boxes"], dtype=np.float64).reshape(-1, 4)
            if len(member_boxes) > 1:
                matrix = metrics.box_iou(member_boxes, member_boxes)
                jitter = float(matrix[np.triu_indices(len(member_boxes), k=1)].mean())
            else:
                jitter = 1.0
            verifier_best = {}
            for name in verifier_layers:
                vb, vs = verifier_boxes[name], verifier_scores[name]
                if not len(vb):
                    verifier_best[name] = 0.0
                    continue
                iou = metrics.box_iou(boxes[position:position + 1], vb)[0]
                valid = iou >= match_iou
                verifier_best[name] = float(vs[valid].max()) if valid.any() else 0.0
            rows.append({"box": boxes[position].tolist(), "score": float(scores[position]),
                         "views": len(info["member_views"]), "jitter": jitter,
                         "verifier": verifier_best})
        rows.sort(key=lambda row: -row["score"])
        image_boxes = np.asarray([row["box"] for row in rows], dtype=np.float64).reshape(-1, 4)
        image_scores = np.asarray([row["score"] for row in rows], dtype=np.float64)
        areas = box_area(image_boxes)
        inter = intersection_matrix(image_boxes) if len(image_boxes) > 1 else None
        for position, row in enumerate(rows):
            containment, bridge = 0.0, 0
            if inter is not None:
                higher = (image_scores > row["score"] + 0.05) & (image_scores > 0.60)
                if areas[position] > 0:
                    candidates_here = np.where(higher)[0]
                    if len(candidates_here):
                        containment = float((inter[position, candidates_here]
                                             / areas[position]).max())
                if areas[position] > 0:
                    covers = inter[position] / np.maximum(areas, 1e-9)
                    bridge = int(((covers > 0.45) & (image_scores > 0.65)
                                  & (np.arange(len(image_scores)) != position)).sum())
            row["ioa_in"] = containment
            row["bridge"] = int(bridge)
        candidates.append({"image_id": image_id, "gt": entries[0]["gt"], "rows": rows})
    return candidates


def veto(candidates, lam_iso, beta_jitter, beta_ioa, beta_bridge,
         iso_gate=(0.30, 0.30, 0.25), ambiguous_below=0.85, jitter_ref=0.80, ioa_ref=0.35):
    records = []
    for entry in candidates:
        rows = entry["rows"]
        scores = []
        for row in rows:
            z = float(logit(row["score"]))
            if row["score"] < ambiguous_below:
                gates = (row["verifier"]["v1280"], row["verifier"]["vr50"], row["verifier"]["vswin"])
                if gates[0] < iso_gate[0] and gates[1] < iso_gate[1] and gates[2] < iso_gate[2]:
                    z -= lam_iso
            z -= beta_jitter * max(0.0, jitter_ref - row["jitter"])
            z -= beta_ioa * max(0.0, row["ioa_in"] - ioa_ref)
            if row["bridge"] >= 2:
                z -= beta_bridge
            scores.append(float(sigmoid(z)))
        records.append({"image_id": entry["image_id"], "gt": entry["gt"],
                        "pred": {"boxes": [row["box"] for row in rows], "scores": scores}})
    return records


def counts(records, confidence):
    return metrics.evaluate_at_threshold(records, confidence, 0.5)


def sweep(records, window):
    rows = []
    for value in window:
        summary = counts(records, value)
        rows.append((summary["precision"], summary["recall"], float(value), summary))
    eligible = [row for row in rows if row[1] >= RECALL_TARGET]
    return max(eligible, key=lambda row: (row[0], row[1])) if eligible else None


def evaluate_arm(records, coarse=None, refine_steps=8):
    coarse = coarse if coarse is not None else np.linspace(0.68, 0.84, 15)
    first = sweep(records, coarse)
    if first is None:
        return None
    fine = sweep(records, np.linspace(first[2] - 0.010, first[2] + 0.010, refine_steps))
    best = first if fine is None or fine[0] <= first[0] else fine
    return {"precision": best[0], "recall": best[1], "threshold": best[2],
            "tp": int(best[3]["tp"]), "fp": int(best[3]["fp"]), "fn": int(best[3]["fn"])}


def subset(candidates, parity):
    return [entry for index, entry in enumerate(candidates) if index % 2 == parity]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--dump", help="Optional compressed cache of the fused candidates")
    parser.add_argument("--reuse", help="Reuse a previously dumped candidate cache")
    parser.add_argument("--top", type=int, default=3, help="How many finalists to score on full VAL")
    args = parser.parse_args()

    if args.reuse:
        cached = np.load(args.reuse, allow_pickle=True)
        candidates = cached["candidates"].tolist()
    else:
        main_layers = [load(path) for path in MAIN_VIEWS]
        verifier_layers = {name: load(path) for name, path in VERIFIERS.items()}
        candidates = build_candidates(main_layers, verifier_layers)
    if args.dump:
        np.savez_compressed(args.dump, candidates=np.asarray(candidates, dtype=object))

    report = {"recall_target": RECALL_TARGET, "candidate_pool": MAIN_VIEWS,
              "verifiers": VERIFIERS, "fold_baseline": {}, "search": [], "finalists": []}
    for parity in (0, 1):
        reference = evaluate_arm(veto(subset(candidates, parity), 0, 0, 0, 0))
        report["fold_baseline"][parity] = reference
        print(f"fold {parity} baseline P@R90 = {reference['precision'] * 100:.2f}% "
              f"R = {reference['recall'] * 100:.2f}% thr = {reference['threshold']:.4f}", flush=True)

    for lam in (0.0, 0.5, 1.0):
        for bj in (0.0, 6.0, 15.0):
            for bi in (0.0, 6.0, 15.0):
                for bb in (0.0, 5.0):
                    if lam == 0 and bj == 0 and bi == 0 and bb == 0:
                        continue
                    row = {"lam_iso": lam, "beta_jitter": bj, "beta_ioa": bi, "beta_bridge": bb}
                    for parity in (0, 1):
                        row[f"fold{parity}"] = evaluate_arm(
                            veto(subset(candidates, parity), lam, bj, bi, bb))
                    worst = min(row["fold0"]["precision"], row["fold1"]["precision"])
                    row["worst_fold"] = worst
                    report["search"].append(row)
                    print(f"lam={lam} bj={bj} bi={bi} bb={bb} "
                          f"f0={row['fold0']['precision'] * 100:.2f} "
                          f"f1={row['fold1']['precision'] * 100:.2f} worst={worst * 100:.2f}", flush=True)

    ranked = sorted(report["search"], key=lambda row: -row["worst_fold"])[:args.top]
    for row in ranked:
        params = (row["lam_iso"], row["beta_jitter"], row["beta_ioa"], row["beta_bridge"])
        final = evaluate_arm(veto(candidates, *params))
        report["finalists"].append({**{key: row[key] for key in
                                       ("lam_iso", "beta_jitter", "beta_ioa", "beta_bridge", "worst_fold")},
                                    "full": final})
        print(f"finalist {params} full P@R90 = {final['precision'] * 100:.2f}% "
              f"R = {final['recall'] * 100:.2f}% TP/FP/FN = {final['tp']}/{final['fp']}/{final['fn']}", flush=True)
    report["baseline_full"] = evaluate_arm(veto(candidates, 0, 0, 0, 0))
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print("wrote", args.out, flush=True)


if __name__ == "__main__":
    main()
