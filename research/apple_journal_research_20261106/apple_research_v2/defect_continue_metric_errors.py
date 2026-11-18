"""Read-only pairwise error audit at the predeclared operating points."""
from collections import Counter
import numpy as np
from class0_patch_refinement import ROOT, read, atomic_json
from audit_defect_causes_20261112 import match

OUT = ROOT / "experiments/defect_continue_metric"
REF = ROOT / "experiments/defect9485_cross_deploy/gate99/fixed_predictions.json"


def hits(rows):
    accepted = set()
    reasons = Counter()
    for r in rows:
        pred = r["pred"]
        m, ov = match(pred, r["GT"])
        accepted.update((r["image_id"], gt, r["GT"][gt]["class"]) for gt in m.values())
        for i, p in enumerate(pred):
            if i in m:
                continue
            c = int(p[5])
            gts = r["GT"]
            overlap = ov[i]
            same = [j for j,g in enumerate(gts) if g["class"] == c]
            if same and overlap[same].max() >= .5:
                reason = "duplicate"
            elif len(gts) and overlap.max() >= .5:
                reason = "wrong_class"
            elif same and overlap[same].max() >= .1:
                reason = "extent_or_nearby"
            else:
                reason = "no_nearby_same_class_GT"
            reasons[f"{c}/{reason}"] += 1
    return accepted, dict(reasons)


def main():
    ref, _ = hits(read(REF))
    results = {}
    for k in ("joint", "precision94", "recall85"):
        got, reasons = hits(read(OUT / f"fresh_{k}_predictions.json"))
        results[k] = dict(gained=sorted(got-ref), lost=sorted(ref-got), FP_types=reasons)
    atomic_json(OUT/"paired_errors.json", dict(reference=str(REF.relative_to(ROOT)), results=results))
    for k,r in results.items():
        print(k, "gained", len(r["gained"]), "lost", len(r["lost"]), r["FP_types"])


if __name__ == "__main__":
    main()
