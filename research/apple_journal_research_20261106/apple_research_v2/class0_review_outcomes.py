"""Join reviewed defects and false positives to independently re-inferred outputs."""

from collections import Counter
from class0_patch_refinement import ROOT, OLD, read, configure, atomic_json
from audit_class0_visual import OUT


def main():
    from audit_defect_causes_20261112 import match, ious
    from defect_coordinate_vote_20261118 import vote
    from scipy.optimize import linear_sum_assignment

    review = read(OUT / "audit.json")["cases"]
    cases = {(c["image"], c["index"]): c["code"] for c in review["miss"]}
    runs = {}
    for variant in ("patch_refinement", "extent_selection", "crop_expert"):
        run = ROOT / f"experiments/class0_{variant}_v1"
        records = {r["image_id"]: r for r in read(run / "dev_predictions.json")}
        recovered = []
        lost = []
        newfp = []
        removedfp = []
        for row in read(OLD / "phase1/source_predictions.json"):
            sid = row["image_id"]
            old = vote(row["sources"], 0.5, [0.4, 0.34])
            gt = row["GT"]
            new = records[sid]["pred"]
            assert gt == records[sid]["GT"]
            pairs, _ = match(old, gt)
            other, mat = match(new, gt)
            a = {i for i in pairs.values() if gt[i]["class"] == 0}
            b = {i for i in other.values() if gt[i]["class"] == 0}
            recovered.extend(cases[sid, i] for i in sorted(b - a))
            lost.extend(sid + "#" + str(i) for i in sorted(a - b))
            assert {i for i in pairs.values() if gt[i]["class"] == 1} == {
                i for i in other.values() if gt[i]["class"] == 1
            }
            of = [i for i, p in enumerate(old) if p[5] == 0 and i not in pairs]
            nf = [i for i, p in enumerate(new) if p[5] == 0 and i not in other]
            ov = ious([old[i][:4] for i in of], [new[i][:4] for i in nf])
            rows, cols = linear_sum_assignment(-((ov >= 0.5) * 1000 + ov))
            ok = [(of[i], nf[j]) for i, j in zip(rows, cols) if ov[i, j] >= 0.5]
            removedfp.extend(
                sid + "#" + str(i) for i in of if all(i != j for j, _ in ok)
            )
            for i in nf:
                if any(i == j for _, j in ok):
                    continue
                j = mat[i].argmax() if len(gt) else None
                overlap = float(mat[i, j]) if j is not None else 0
                reason = (
                    "no nearby GT"
                    if overlap < 0.1
                    else "extent mismatch"
                    if overlap < 0.5
                    else "wrong class"
                    if gt[j]["class"] != 0
                    else "duplicate"
                )
                newfp.append(
                    dict(image=sid, index=i, reason=reason, nearest_iou=overlap)
                )
        runs[variant] = dict(
            recovered=sorted(recovered),
            lost=lost,
            new_false_positives=newfp,
            removed_false_positives=removedfp,
            new_fp_types=dict(Counter(p["reason"] for p in newfp)),
        )
    atomic_json(OUT / "outcomes.json", runs)


if __name__ == "__main__":
    configure(OUT, "cpu")
    main()
