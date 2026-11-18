"""Measure out-of-fold-selected set verifier's complementary hits against fixed detectors."""

from pathlib import Path
from class0_patch_refinement import ROOT, read, configure, atomic_json
from audit_defect_causes_20261112 import ious, match
from run_defect_zero_train_20261112 import metrics
import numpy as np

OUT = ROOT / "experiments/defect9485_set"


def audit_pair(old, new):
    gained = []
    lost = []
    same = 0
    for r, s in zip(old, new, strict=True):
        assert (r["image_id"], r["GT"]) == (s["image_id"], s["GT"])
        a = set(match(r["pred"], r["GT"])[0].values())
        b = set(match(s["pred"], s["GT"])[0].values())
        same += len(a & b)
        for keys, dest in [(b - a, gained), (a - b, lost)]:
            dest.extend(f"{r['image_id']}#{i}" for i in sorted(keys))
    return dict(common_TP=same, gained=gained, lost=lost)


def supplement(old, new):
    rows = []
    for r, s in zip(old, new, strict=True):
        b = np.array(r["pred"], np.float32).reshape(-1, 6)
        n = np.array(s["pred"], np.float32).reshape(-1, 6)
        if len(b) and len(n):
            ov = ious(n[:, :4], b[:, :4])
            n = n[~((ov >= 0.5) & (n[:, 5, None] == b[None, :, 5])).any(1)]
        # Incumbent predictions have priority to avoid claiming their matches as rescues.
        b[:, 4] = 0.5 + 0.5 * b[:, 4]
        n[:, 4] *= 0.5
        rows.append(
            dict(
                image_id=r["image_id"], GT=r["GT"], pred=np.concatenate((b, n)).tolist()
            )
        )
    return rows


if __name__ == "__main__":
    configure(OUT, "cpu")
    p = read(OUT / "P94_predictions.json")
    report = {}
    for name, path in [
        ("balanced", "class0_crop_deploy_v1/fp32/dev_predictions.json"),
        (
            "precision92",
            "defect_precision92_consensus_v1/agreement_plus_high_predictions.json",
        ),
    ]:
        raw = read(ROOT / "experiments" / path)
        high = supplement(raw, p)
        m = metrics(high)
        report[name] = dict(
            individual=audit_pair(raw, p),
            union_change=audit_pair(raw, high),
            union={k: m[k] for k in ["total", "class"]},
        )
        atomic_json(OUT / f"{name}_supplement_predictions.json", high)
    atomic_json(OUT / "supplement.json", report)
    print(
        {
            k: (len(v["individual"]["gained"]), v["union"]["total"])
            for k, v in report.items()
        }
    )
