"""Audit per-instance differences at the two frozen deployment points."""

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "apple_research_v2/joint"))
from joint_state_v1 import configure, atomic_json

OUT = ROOT / "experiments/class0_visual_review_v1"
OLD = ROOT / "experiments/defect_algorithm_steps_20261112_v2"


def read(path):
    return json.loads(path.read_text("utf8"))


def main():
    configure(OUT, "cpu")
    import numpy as np
    import cv2
    from PIL import Image, ImageDraw
    from scipy.optimize import linear_sum_assignment
    from audit_defect_causes_20261112 import match, ious
    from defect_consensus_20261118 import prepare, rank, cut
    from defect_coordinate_vote_20261118 import vote

    cfg = read(ROOT / "experiments/defect_consensus_20261118/frozen.json")
    manifest = read(ROOT / "apple_research_v2/datasets/ssda_date_v2/manifest.json")[
        "samples"
    ]
    paths = {r["image_id"]: r["rgb_path"] for r in manifest if r["partition"] == "dev"}
    cases = {"miss": [], "fp": [], "lost": []}
    panels = {k: [] for k in cases}
    verified = [[], []]
    diff = {"shared_FP": 0, "base_only_FP": 0}
    for r in read(OLD / "phase1/source_predictions.json"):
        gt = r["GT"]
        src = r["sources"]
        sid = r["image_id"]
        im = Image.open(paths[sid]).convert("RGB")
        low = vote(src, 0.5, [0.001, 0.001])
        base = cut(low, [0.40, 0.34])
        dual_all = rank(*prepare(src), cfg["coefficients"])
        dual = cut(dual_all, cfg["thresholds"])
        bm, bio = match(base, gt)
        dm, dio = match(dual, gt)
        for j, p in enumerate([base, dual]):
            verified[j].append(dict(image_id=sid, GT=gt, pred=p.tolist()))
        bfp = [i for i, p in enumerate(base) if p[5] == 0 and i not in bm]
        dfp = [i for i, p in enumerate(dual) if p[5] == 0 and i not in dm]
        mat = ious([base[i, :4] for i in bfp], [dual[i, :4] for i in dfp])
        ii, jj = linear_sum_assignment(-((mat >= 0.5).astype(float) * 1000 + mat))
        shared = {dfp[j] for i, j in zip(ii, jj) if mat[i, j] >= 0.5}
        diff["shared_FP"] += len(shared)
        diff["base_only_FP"] += len(bfp) - len(shared)

        def entry(kind, idx, box, details, overlay):
            n = len(cases[kind]) + 1
            code = kind[0].upper() + f"{n:02}"
            cases[kind].append(
                dict(
                    code=code,
                    image=sid,
                    index=idx,
                    box=list(map(float, box)),
                    **details,
                )
            )
            x, y, X, Y = box
            cx = (x + X) / 2
            cy = (y + Y) / 2
            side = max(76, 1.95 * max(X - x, Y - y))
            l, t = int(cx - side / 2), int(cy - side / 2)
            rr = (l, t, l + int(side), t + int(side))
            crop = im.crop(rr).resize((148, 148))
            panel = Image.new("RGB", (300, 174), "white")
            panel.paste(crop, (0, 24))
            local = crop.copy()
            d = ImageDraw.Draw(local)
            for b, color in overlay:
                if b is not None:
                    d.rectangle(
                        [
                            (b[0] - l) * 148 / int(side),
                            (b[1] - t) * 148 / int(side),
                            (b[2] - l) * 148 / int(side),
                            (b[3] - t) * 148 / int(side),
                        ],
                        outline=color,
                        width=2,
                    )
            panel.paste(local, (151, 24))
            ImageDraw.Draw(panel).text(
                (3, 3), code + "  " + details["category"], fill="black"
            )
            panels[kind].append(panel)

        for gi, g in enumerate(gt):
            if g["class"] != 0:
                continue

            def nearest(p):
                p = p[p[:, 5] == 0]
                v = ious(p[:, :4], [g["box"]]).ravel()
                if not len(v):
                    return None, 0.0, 0.0
                j = v.argmax()
                return p[j, :4], float(v[j]), float(p[j, 4])

            if gi not in bm.values():
                b, bi, bs = nearest(low)
                d, di, ds = nearest(dual_all)
                recovered = gi in dm.values()
                other = max(
                    (dio[k, gi] for k, x in enumerate(dual) if x[5] == 1), default=0
                )
                cat = (
                    "recovered"
                    if recovered
                    else "wrong class nearby"
                    if other >= 0.5
                    else "below score"
                    if max(bi, di) >= 0.5
                    else "extent mismatch"
                    if max(bi, di) >= 0.1
                    else "no nearby box"
                )
                entry(
                    "miss",
                    gi,
                    g["box"],
                    dict(
                        category=cat,
                        recovered=recovered,
                        base_iou=bi,
                        base_score=bs,
                        dual_iou=di,
                        dual_score=ds,
                    ),
                    [(g["box"], "lime"), (b, "cyan"), (d, "magenta")],
                )
            elif gi not in dm.values():
                entry(
                    "lost",
                    gi,
                    g["box"],
                    dict(category="lost by dual"),
                    [(g["box"], "lime")],
                )
        for pi in dfp:
            best = int(dio[pi].argmax()) if gt else None
            iou = float(dio[pi, best]) if best is not None else 0.0
            cat = (
                "no nearby GT"
                if iou < 0.1
                else "extent mismatch"
                if iou < 0.5
                else "wrong class"
                if gt[best]["class"] != 0
                else "duplicate"
            )
            entry(
                "fp",
                pi,
                dual[pi, :4],
                dict(
                    category=cat,
                    shared=pi in shared,
                    score=float(dual[pi, 4]),
                    nearest_iou=iou,
                ),
                [
                    (dual[pi, :4], "magenta"),
                    (gt[best]["box"] if iou > 0.05 else None, "lime"),
                ],
            )
    from run_defect_zero_train_20261112 import metrics

    for rr, expected in zip(verified, [(186, 45, 44), (192, 64, 38)]):
        t = metrics(rr)["total"]
        assert (t["TP"], t["FP"], t["FN"]) == expected
    assert len(cases["miss"]) == 29 and len(cases["fp"]) == 21
    for kind, ps in panels.items():
        for i in range(0, len(ps), 9):
            sheet = Image.new(
                "RGB", (900, 174 * ((min(len(ps) - i, 9) + 2) // 3)), "white"
            )
            for j, p in enumerate(ps[i : i + 9]):
                sheet.paste(p, (j % 3 * 300, j // 3 * 174))
            sheet.save(OUT / f"{kind}_{i // 9 + 1}.jpg", quality=94)
    atomic_json(OUT / "audit.json", dict(cases=cases, diff=diff))
    print(diff, {k: len(v) for k, v in cases.items()})


if __name__ == "__main__":
    main()
