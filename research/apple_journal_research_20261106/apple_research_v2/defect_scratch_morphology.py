"""Describe scratch annotations and frozen model errors without changing scores or labels."""

import argparse
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw
from scipy.stats import rankdata

from class0_patch_refinement import ROOT, atomic_json, read, sha
from audit_defect_causes_20261112 import match
from defect_scratch_specialist_audit import (
    canonical,
    validate,
    hit_ids,
    MANIFEST,
    EXP,
    WARM,
)

SNAP = EXP / "defect_public_intake/snapshots/75e57d602dc84195b8d705105909e37c"
OUT = EXP / "ssda_scratch_morphology"
METRICS = (
    "side_768",
    "aspect",
    "x_fraction",
    "y_fraction",
    "image_edge_fraction",
    "L_inner_minus_ring",
    "dE_median",
    "dark_tail",
    "light_tail",
    "surface_neighbor_gap",
)


def descriptor(rgb, box, others):
    h, w = rgb.shape[:2]
    l, t, r, b = box
    bw, bh = r - l, b - t
    side = math.sqrt(bw * bh)
    cx, cy = (l + r) / 2, (t + b) / 2
    pad = max(4, side * 0.22)
    a, c = int(max(0, l - pad)), int(max(0, t - pad))
    z, j = int(min(w, math.ceil(r + pad))), int(min(h, math.ceil(b + pad)))
    patch = rgb[c:j, a:z]
    # Colour is measured within the reference rectangle for both sources. The
    # SSDA boxes do not provide a per-pixel defect mask.
    lab = cv2.cvtColor(patch.astype(np.float32) / 255, cv2.COLOR_RGB2LAB)
    yy, xx = np.ogrid[c:j, a:z]
    inner = (xx >= l) & (xx < r) & (yy >= t) & (yy < b)
    ring = ~inner
    for L, T, R, B in others:
        ring &= ~((xx >= L) & (xx < R) & (yy >= T) & (yy < B))
    # Exclude near-black conveyor pixels from both measurement supports.
    inside = lab[inner & (patch.max(2) > 25)]
    outside = lab[ring & (patch.max(2) > 25)]
    dist = []
    for L, T, R, B in others:
        dist.append(math.hypot(max(L - r, l - R, 0), max(T - b, t - B, 0)) / side)
    desc = dict(
        side_768=side * 768 / max(h, w),
        aspect=max(bw, bh) / min(bw, bh),
        x_fraction=cx / w,
        y_fraction=cy / h,
        image_edge_fraction=min(cx / w, 1 - cx / w, cy / h, 1 - cy / h),
        native_width=bw,
        native_height=bh,
        surface_neighbor_gap=min(dist) if dist else None,
    )
    if len(inside) >= 8 and len(outside) >= 8:
        m = np.median(inside, axis=0)
        n = np.median(outside, axis=0)
        desc.update(
            L_inner_minus_ring=float(m[0] - n[0]),
            dE_median=float(np.linalg.norm(m - n)),
            dark_tail=float(
                np.percentile(outside[:, 0], 10) - np.percentile(inside[:, 0], 10)
            ),
            light_tail=float(
                np.percentile(inside[:, 0], 90) - np.percentile(outside[:, 0], 90)
            ),
        )
    else:
        desc.update(
            {
                k: None
                for k in ("L_inner_minus_ring", "dE_median", "dark_tail", "light_tail")
            }
        )
    return desc


def scene(src, box):
    with Image.open(src) as image:
        rgb = image.convert("RGB")
        w, h = rgb.size
        thumb = rgb.resize((94, 94), Image.Resampling.LANCZOS)
        x, y, X, Y = box
        cx, cy = (x + X) / 2, (y + Y) / 2
        side = min(max(w, h), max(86, math.ceil(1.35 * max(X - x, Y - y))))
        left = min(max(0, round(cx - side / 2)), w - side)
        top = min(max(0, round(cy - side / 2)), h - side)
        crop = rgb.crop((left, top, left + side, top + side)).resize(
            (158, 158), Image.Resampling.LANCZOS
        )
        d = ImageDraw.Draw(thumb)
        d.rectangle(
            (x / w * 94, y / h * 94, X / w * 94, Y / h * 94), outline="#4bf5e5", width=1
        )
        panel = Image.new("RGB", (254, 182), "white")
        panel.paste(thumb, (0, 0))
        panel.paste(crop, (96, 0))
        return panel


def stats(rows):
    out = {}
    for key in METRICS:
        values = [v["desc"][key] for v in rows if v["desc"][key] is not None]
        out[key] = (
            dict(n=len(values), quartiles=np.percentile(values, [25, 50, 75]).tolist())
            if values
            else None
        )
    return dict(n=len(rows), features=out)


def build():
    OUT.mkdir(parents=True, exist_ok=True)
    dev, dates = canonical()
    model_paths = {
        "warm": WARM,
        "Small": EXP / "defect9485_small_v1",
        "replay": EXP / "defect_continue_public_v1/replay",
        "RCNN": EXP / "defect_continue_rcnn/transfer_v2",
    }
    hits = {
        k: hit_ids(validate(read(p / "fixed_predictions.json"), dev), 0)
        for k, p in model_paths.items()
    }
    low = validate(read(WARM / "low_predictions.json"), dev)
    archive = {
        (c["image"], c["index"]): c["code"]
        for c in read(EXP / "class0_visual_review_v1/audit.json")["cases"]["miss"]
    }
    rows = []
    manifest = read(MANIFEST)["samples"]
    for row in manifest:
        if row["partition"] not in ("train", "dev"):
            continue
        gt = row["instances"]
        h, w = row["native_hw"]
        boxes = []
        for v in gt:
            x, y, bw, bh = v["xywh_normalized"]
            boxes.append(
                [(x - bw / 2) * w, (y - bh / 2) * h, (x + bw / 2) * w, (y + bh / 2) * h]
            )
        if not any(v["class_id"] == 0 for v in gt):
            continue
        with Image.open(row["rgb_path"]) as im:
            rgb = np.asarray(im.convert("RGB"))
        for i, g in enumerate(gt):
            if g["class_id"] != 0:
                continue
            key = row["image_id"]
            identity = (key, i)
            out = dict(
                source="SSDA",
                partition=row["partition"],
                image=key,
                index=i,
                date=row["capture_filename_tokens"]["date"],
                camera=row["capture_filename_tokens"]["camera"],
                path=row["rgb_path"],
                box=boxes[i],
                desc=descriptor(
                    rgb, boxes[i], [b for j, b in enumerate(boxes) if j != i]
                ),
            )
            if key in dev:
                out["hits"] = {tag: identity in seen for tag, seen in hits.items()}
                out["M"] = archive.get(identity)
                p = [v for v in low[key]["pred"] if v[5] == 0]
                _, overlap = match(p, low[key]["GT"])
                out["best_warm_same_iou"] = max(
                    (float(q) for q in overlap[:, i]), default=0.0
                )
                out["best_score_at_50"] = max(
                    (v[4] for j, v in enumerate(p) if overlap[j, i] >= 0.5),
                    default=None,
                )
            rows.append(out)
    positives = read(SNAP / "positives.json")
    assert (
        sha(SNAP / "positives.json")
        == "e37fa965b09dc6c1ba5839c88d781dfa785e70ceac837eba22d42859de464e75"
    )
    by_src = defaultdict(list)
    for r in positives:
        if r["target_class"] == 0:
            by_src[r["source_id"]].append(r)
    for key, items in by_src.items():
        first = items[0]
        path = SNAP / first["full_image"]
        with Image.open(path) as im:
            rgb = np.asarray(im.convert("RGB"))
        for r in items:
            with Image.open(SNAP / r["mask"]) as mask:
                mask_area = int(np.asarray(mask.convert("L")).astype(bool).sum())
            box = r["box_xyxy"]
            rows.append(
                dict(
                    source="Woo",
                    partition="official train",
                    image=key,
                    index=r["annotation"],
                    box=box,
                    mask_fill=mask_area / ((box[2] - box[0]) * (box[3] - box[1])),
                    desc=descriptor(
                        rgb,
                        box,
                        [
                            t["box_xyxy"]
                            for t in items
                            if t["annotation"] != r["annotation"]
                        ],
                    ),
                )
            )
    development = [r for r in rows if r["partition"] == "dev"]
    assert len(development) == 87 and sum(v["hits"]["warm"] for v in development) == 44
    order = sorted(development, key=lambda v: (v["image"], v["index"]))
    random.Random(18603).shuffle(order)
    for start in range(0, len(order), 12):
        chunk = order[start : start + 12]
        sheet = Image.new("RGB", (4 * 254, math.ceil(len(chunk) / 4) * 182), "white")
        for j, v in enumerate(chunk):
            code = f"S{start+j+1:02}"
            v["visual_id"] = code
            panel = scene(v["path"], v["box"])
            ImageDraw.Draw(panel).text((8, 162), code, fill="black")
            sheet.paste(panel, ((j % 4) * 254, (j // 4) * 182))
        sheet.save(OUT / f"visual_{start//12+1:02}.jpg", quality=91)
    ssda_train = [r for r in rows if r["partition"] == "train"]
    woo = [r for r in rows if r["source"] == "Woo"]
    tp = [v for v in development if v["hits"]["warm"]]
    fn = [v for v in development if not v["hits"]["warm"]]
    assert [len(ssda_train), len(woo), len(tp), len(fn)] == [537, 224, 44, 43]
    auc = {}
    for col in METRICS:
        pairs = [
            (v["desc"][col], int(not v["hits"]["warm"]))
            for v in development
            if v["desc"][col] is not None
        ]
        scores, labels = map(np.array, zip(*pairs))
        n = sum(labels)
        p = len(labels) - n
        auc[col] = float(
            (rankdata(scores)[labels == 1].sum() - n * (n + 1) / 2) / (n * p)
        )
    diagnostics = {}
    for v in development:
        group = diagnostics.setdefault(v["date"], Counter())
        for tag in hits:
            group[f"{tag}_TP"] += v["hits"][tag]
        group["GT"] += 1
    out = dict(
        distributions={
            name: stats(vals)
            for name, vals in (
                ("native_train", ssda_train),
                ("development", development),
                ("public_scratch", woo),
                ("warm_TP", tp),
                ("warm_FN", fn),
            )
        },
        dated=diagnostics,
        AUC_higher_feature_indicates_miss=auc,
        warm_fn_recoveries={m: sum(v["hits"][m] for v in fn) for m in hits},
        historical_cases={v["M"]: v["hits"] for v in development if v["M"]},
    )
    atomic_json(OUT / "regions.json", rows)
    atomic_json(OUT / "descriptive_summary.json", out)
    return out


def summarize_visual():
    rows = read(OUT / "regions.json")
    dev = [v for v in rows if v["partition"] == "dev"]
    native = [v for v in rows if v["partition"] == "train"]
    public = [v for v in rows if v["source"] == "Woo"]
    annotations = ROOT / "configs/ssda_scratch_visual_codes_20261118.json"
    tags = read(annotations)
    shapes = list(tags)[1:6]
    assigned = [i for k in shapes for i in tags[k]]
    assert sorted(assigned) == list(range(1, 88))
    sizes = np.array([v["desc"]["side_768"] for v in native])
    q1, q3 = np.quantile(sizes, [0.25, 0.75])
    table = defaultdict(Counter)
    by_gt = []
    model_gain = Counter()
    causes = Counter()

    def bin_size(x):
        return (
            "below_train_Q25"
            if x < q1
            else "above_train_Q75" if x > q3 else "train_IQR"
        )

    for v in dev:
        n = int(v["visual_id"][1:])
        shape = next(k for k in shapes if n in tags[k])
        outcome = "TP" if v["hits"]["warm"] else "FN"
        visible = (
            "subtle_or_unclear"
            if n in tags["subtle_or_unclear_appearance"]
            else "distinct"
        )
        edge = (
            "silhouette_in_detail"
            if n in tags["fruit_silhouette_visible_in_enlarged_context"]
            else "silhouette_outside_detail"
        )
        gap = v["desc"]["surface_neighbor_gap"]
        close = (
            "near_another_native_box"
            if gap is not None and gap <= 0.5
            else "no_close_native_box"
        )
        vals = (shape, visible, edge, close, bin_size(v["desc"]["side_768"]))
        for k in vals:
            table[k][outcome] += 1
        hits = v["hits"]
        if not hits["warm"]:
            if v["best_score_at_50"] is not None:
                miss = (
                    "low_score"
                    if v["best_score_at_50"] < 0.40
                    else "one_to_one_competition"
                )
            elif v["best_warm_same_iou"] >= 0.20:
                miss = "partial_localization"
            else:
                miss = "weak_or_missing_geometry"
            causes[miss] += 1
            combo = (
                "+".join(k for k in ("Small", "replay", "RCNN") if hits[k]) or "none"
            )
            model_gain[combo] += 1
        else:
            miss = None
        by_gt.append(
            dict(
                visual_id=v["visual_id"],
                image=v["image"],
                index=v["index"],
                original_M=v["M"],
                source_shape=shape,
                visibility=visible,
                crop_edge=edge,
                other_box=close,
                hit=hits,
                warm_miss=miss,
                max_class0_iou=v["best_warm_same_iou"],
                max_score_at_50=v["best_score_at_50"],
            )
        )
    out = dict(
        reading_source_sha=sha(annotations),
        descriptive_shape_groups=table,
        matched_by_others_within_warm_FN=model_gain,
        warm_miss_diagnostic=causes,
        train_derived_scale_boundaries_768=[q1, q3],
        scale_distributions={
            name: dict(Counter(bin_size(v["desc"]["side_768"]) for v in subset))
            for name, subset in (
                ("SSDA_train", native),
                ("SSDA_dev", dev),
                ("public_Woo", public),
            )
        },
        locations={
            name: [
                float(np.median([v["desc"][axis] for v in group]))
                for axis in ("x_fraction", "y_fraction")
            ]
            for name, group in (
                ("SSDA_train", native),
                ("SSDA_dev", dev),
                ("public_Woo", public),
            )
        },
        by_gt=sorted(by_gt, key=lambda q: q["visual_id"]),
    )
    atomic_json(OUT / "visual_audit.json", out)
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--visual-summary", action="store_true")
    a = p.parse_args()
    cv2.setNumThreads(1)
    if a.visual_summary:
        result = summarize_visual()
        print({k: v for k, v in result.items() if k != "by_gt"})
    else:
        result = build()
        print(result["dated"], result["warm_fn_recoveries"])
