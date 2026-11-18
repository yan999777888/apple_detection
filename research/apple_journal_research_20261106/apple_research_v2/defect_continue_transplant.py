"""Training-only illuminance-matched transfer of annotated SSDA surface defects."""

import hashlib
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

from class0_patch_refinement import ROOT, atomic_json, load_image
from defect9485_p2 import samples

OUT = ROOT / "experiments/defect_continue_rcnn"
SIDE = 704


def split_train():
    train = samples("train")
    # Group cameras together for nearby timestamps. There are no verified fruit IDs.
    def group(row):
        t = row["capture_filename_tokens"]
        return t["date"] + "_" + t["time"][:4]

    groups = sorted({group(row) for row in train},
                    key=lambda s: hashlib.sha256(("ssda-rcnn-" + s).encode()).digest())
    reserve = set(groups[:max(1, round(len(groups) * 0.17))])
    fit = [r for r in train if group(r) not in reserve]
    calibration = [r for r in train if group(r) in reserve]
    assert len(fit) + len(calibration) == 596 and not ({group(r) for r in fit} & reserve)
    return fit, calibration


@lru_cache(maxsize=64)
def image(path):
    return cv2.resize(load_image(path), (SIDE, SIDE), interpolation=cv2.INTER_AREA)


def labels(row):
    out = []
    for v in row["instances"]:
        x, y, w, h = np.array(v["xywh_normalized"]) * SIDE
        out.append([x - w / 2, y - h / 2, x + w / 2, y + h / 2, v["class_id"]])
    a = np.asarray(out, np.float32).reshape(-1, 5)
    a[:, :4] = np.clip(a[:, :4], 0, SIDE)
    return a


def eligible(patch):
    # Real fruit skin is coloured and non-black, including in the corners of the patch.
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    return np.mean((hsv[:, :, 2] > 34) & (hsv[:, :, 1] > 18)) >= 0.985


def intersects(outer, others):
    a = np.maximum(outer[:2], others[:, :2])
    b = np.minimum(outer[2:], others[:, 2:4])
    return bool(np.any((b[:, 0] > a[:, 0]) & (b[:, 1] > a[:, 1])))


def transplant(im, gt, donor, dg, idx, rng):
    x1, y1, x2, y2, cls = dg[idx]
    w, h = x2 - x1, y2 - y1
    if min(w, h) < 5 or max(w, h) > 0.22 * SIDE:
        return None
    pad = max(4, round(max(w, h) * 0.2))
    bounds = np.array([np.floor(x1) - pad, np.floor(y1) - pad,
                       np.ceil(x2) + pad, np.ceil(y2) + pad], dtype=int)
    if bounds.min() < 0 or bounds.max() > SIDE or intersects(bounds, np.delete(dg, idx, axis=0)):
        return None
    l, t, r, b = bounds
    source = donor[t:b, l:r]
    if not eligible(source):
        return None
    hh, ww = source.shape[:2]
    yy, xx = np.mgrid[:hh, :ww]
    edge = np.minimum(np.minimum(xx, ww - 1 - xx), np.minimum(yy, hh - 1 - yy))
    ring = edge < max(2, pad // 2)
    ramp = np.minimum(edge / pad, 1)
    alpha = (ramp * ramp * (3 - 2 * ramp))[:, :, None]
    for _ in range(28):
        u, v = rng.integers([0, 0], [SIDE - ww + 1, SIDE - hh + 1])
        outer = np.array([u, v, u + ww, v + hh])
        target = im[v:v + hh, u:u + ww]
        if intersects(outer, gt) or not eligible(target):
            continue
        destination_colour = np.median(target[ring], axis=0)
        source_colour = np.median(source[ring], axis=0)
        shift = destination_colour - source_colour
        if np.abs(shift).max() > 65 or np.std(target[ring].mean(1)) > 26:
            continue
        result = im.copy()
        result[v:v + hh, u:u + ww] = np.clip(
            (source.astype(np.float32) + shift) * alpha + target * (1 - alpha), 0, 255
        ).round().astype(np.uint8)
        inserted = np.array([[x1 - l + u, y1 - t + v, x2 - l + u, y2 - t + v, cls]],
                            dtype=np.float32)
        return result, np.concatenate((gt, inserted)), outer, bounds
    return None


def sample(rows, index, rng, probability):
    row = rows[index]
    im, gt = image(row["rgb_path"]), labels(row)
    if rng.random() >= probability:
        return im, gt, None
    target_class = rng.integers(2)
    for _ in range(6):
        j = rng.integers(len(rows))
        if j == index:
            continue
        donor_row = rows[j]
        dg = labels(donor_row)
        candidates = np.flatnonzero(dg[:, 4] == target_class)
        if not len(candidates):
            continue
        obj = int(rng.choice(candidates))
        ret = transplant(im, gt, image(donor_row["rgb_path"]), dg, obj, rng)
        if ret is not None:
            added, ann, dest, src = ret
            return added, ann, (donor_row["image_id"], row["image_id"], obj,
                                src.tolist(), dest.tolist())
    return im, gt, None


def audit():
    fit, calibration = split_train()
    rng = np.random.default_rng(9418)
    counts, strips = [0, 0], []
    for i, row in enumerate(fit):
        im, gt, rec = sample(fit, i, rng, 0.5)
        orig = labels(row)
        assert np.array_equal(orig, gt[:len(orig)]) and np.isfinite(gt).all()
        if rec is None:
            continue
        cls = int(gt[-1, -1])
        counts[cls] += 1
        l, t, r, b = rec[4]
        assert not intersects(np.array(rec[4]), orig)
        if len(strips) < 10:
            def patch(frame):
                a = frame[max(0, t-12):min(SIDE, b+12), max(0, l-12):min(SIDE, r+12)]
                return cv2.resize(a, (132,132), interpolation=cv2.INTER_LINEAR)
            strips.append(np.hstack((patch(image(row["rgb_path"])), patch(im))))
    OUT.mkdir(parents=True, exist_ok=True)
    encoded = cv2.imencode(".jpg", np.vstack([np.hstack(strips[i:i+2]) for i in range(0,10,2)]))[1]
    (OUT / "transplant_audit.jpg").write_bytes(encoded.tobytes())
    audit_info = dict(training_images=len(fit), calibration_images=len(calibration),
                      fit_gt=sum(len(r["instances"]) for r in fit),
                      calibration_gt=sum(len(r["instances"]) for r in calibration),
                      inserted_by_class=counts, untouched_original_box_coordinates=True,
                      donor_and_recipient_only_from_fit=True)
    atomic_json(OUT / "transplant_audit.json", audit_info)
    print(audit_info)


if __name__ == "__main__":
    audit()
