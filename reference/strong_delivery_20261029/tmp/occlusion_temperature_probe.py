"""Probe multiplicative occlusion-temperature scaling before any retraining.

Scores are converted back to a foreground logit, divided by
tau = clamp(r_V ** beta, min_tau, 1), and mapped back to probability. The
visible/amodal mask arrays are the genuine cached ROI probabilities from the
published single-view evaluation.
"""
import argparse
import json
from pathlib import Path

import numpy as np

import flip_tta

ROOT = Path(__file__).resolve().parents[1]


def logit(probability):
    probability = np.clip(np.asarray(probability, dtype=np.float64), 1e-7, 1 - 1e-7)
    return np.log(probability / (1 - probability))


def sigmoid(value):
    return 1. / (1. + np.exp(-np.asarray(value, dtype=np.float64)))


def mask_ratios(record, cache_file):
    with np.load(cache_file, allow_pickle=False) as archive:
        visible = archive["visible"].astype(np.float64)
        amodal = archive["amodal"].astype(np.float64)
    if visible.shape != amodal.shape or len(visible) != len(record["pred"]["scores"]):
        raise ValueError("Mask cache and prediction scope differ")
    numerator = visible.sum(axis=(1, 2))
    denominator = np.maximum(amodal.sum(axis=(1, 2)), 1e-6)
    return np.clip(numerator / denominator, 0., 1.)


def rescore(records, ratios, beta, min_tau):
    result = []
    for record, ratio in zip(records, ratios):
        scores = np.asarray(record["pred"]["scores"], dtype=np.float64)
        tau = np.clip(np.power(ratio, beta), min_tau, 1.)
        adjusted = sigmoid(logit(scores) / tau)
        result.append({"image_id": record["image_id"], "gt": record["gt"],
                       "pred": {"boxes": record["pred"]["boxes"], "scores": adjusted.tolist()}})
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--mask-cache", required=True)
    parser.add_argument("--betas", type=float, nargs="+", default=[.25, .5, .75])
    parser.add_argument("--min-taus", type=float, nargs="+", default=[.35, .5, .65])
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    records, ratios = [], []
    for index, record in enumerate(flip_tta.load_records(args.predictions)):
        cache_file = Path(args.mask_cache) / f"roi_{index:06d}.npz"
        if not cache_file.is_file():
            continue
        records.append(record)
        ratios.append(mask_ratios(record, cache_file))
    report = {"images": len(records), "gt": sum(len(row["gt"]["boxes"]) for row in records),
              "baseline": flip_tta.evaluate(records), "arms": []}
    print(f"subset images={report['images']} GT={report['gt']} baseline={report['baseline']}")
    best = None
    for beta in args.betas:
        for min_tau in args.min_taus:
            adjusted = rescore(records, ratios, beta, min_tau)
            row = flip_tta.evaluate(adjusted)
            row.update(beta=beta, min_tau=min_tau)
            report["arms"].append(row)
            print(f"beta={beta:.2f} min_tau={min_tau:.2f} thr={row['threshold']:.4f} "
                  f"P={row['precision'] * 100:.2f} R={row['recall'] * 100:.2f} "
                  f"TP/FP/FN={row['tp']}/{row['fp']}/{row['fn']}", flush=True)
            key = (row["min_precision_recall"], row["precision"] + row["recall"])
            if best is None or key > best[0]:
                best = (key, row)
    report["best"] = best[1]
    Path(args.out).write_text(json.dumps(report, indent=1), encoding="utf-8")
    print("BEST", json.dumps(best[1]), "\nwrote", args.out)


if __name__ == "__main__":
    main()
