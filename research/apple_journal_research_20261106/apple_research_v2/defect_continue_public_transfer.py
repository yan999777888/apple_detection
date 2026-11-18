"""Build a training-only library of audited public lesion polygons and transfer one per fruit."""

from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from class0_patch_refinement import ROOT, read, sha

INTAKE = ROOT / "experiments/defect_public_intake"
SSDA = ROOT / "apple_research_v2/datasets/ssda_date_v2/manifest.json"
PROBABILITY = 0.35


@dataclass
class Donor:
    group: str
    key: str
    cls: int
    rgb: np.ndarray
    mask: np.ndarray
    skin: np.ndarray


def skin_map(rgb):
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    return (hsv[:, :, 2] > 34) & (hsv[:, :, 1] > 18)


def child(folder, name):
    path = (folder / name).resolve()
    if not path.is_relative_to(folder.resolve()):
        raise ValueError(f"File outside expected source folder: {name}")
    return path


@lru_cache(maxsize=2)
def full_rgb(path, expected):
    if sha(path) != expected:
        raise ValueError(f"Source image changed: {path.name}")
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"))


def select_snapshot(intake=INTAKE, name=None):
    intake = Path(intake)
    if name is None:
        return child(intake, read(intake / "active.json")["snapshot"])
    if not name or Path(name).name != name or name in (".", ".."):
        raise ValueError("Expected one snapshot directory name")
    return child(intake / "snapshots", name)


def load_library(intake=INTAKE, name=None):
    snap = select_snapshot(intake, name)
    receipt = read(snap / "receipt.json")
    manifest = snap / "positives.json"
    if (
        sha(manifest) != receipt["manifest_sha256"]
        or sha(SSDA) != receipt["SSDA_manifest_sha256"]
    ):
        raise ValueError("Training data manifest checksum mismatch")
    specs = {s["archive_sha256"]: s for s in receipt["source_specs"]}
    decisions = {d["key"]: d for d in read(snap / "image_decisions.json")}
    groups = defaultdict(lambda: defaultdict(list))
    rejected = Counter()
    mask_hashes = []
    for row in read(manifest):
        if row["raw_split"] != "train":
            rejected["source outside train"] += 1
            continue
        if (
            not row["mask_supervision"]
            or not row.get("mask")
            or row["geometry"] != "polygon"
        ):
            rejected["no original segmented polygon"] += 1
            continue
        spec = specs[row["archive_sha256"]]
        cls = row["target_class"]
        decision = decisions[row["source_id"]]
        if (
            not spec["license_evidence_url"]
            or not spec["record_url"]
            or row["license"] != spec["license"]
            or spec["category_map"].get(row["category_name"]) != cls
            or row["role"] != "training_only_local_positive"
            or row["background_supervision"] is not False
            or decision["decision"] != "export positive crops"
            or decision["group_id"] != row["group_id"]
        ):
            raise ValueError(
                "Unapproved category, source, supervision or deduplication record"
            )
        if decision["raw_split"] != "train":
            rejected["source image outside official training partition"] += 1
            continue
        crop = child(snap, row["crop"])
        if sha(crop) != row["crop_sha256"]:
            raise ValueError(f"Corrupt positive crop: {crop.name}")
        with Image.open(crop) as im:
            rgb = np.asarray(im.convert("RGB"))
        mask_path = child(snap, row["mask"])
        with Image.open(mask_path) as im:
            mask = np.asarray(im.convert("L")) > 0
        height, width = rgb.shape[:2]
        reference = Image.new("L", (width, height))
        draw = ImageDraw.Draw(reference)
        for polygon in row["polygon_in_crop"]:
            draw.polygon([tuple(p) for p in polygon], fill=255)
        if mask.shape != (height, width) or not np.array_equal(
            mask, np.asarray(reference) > 0
        ):
            raise ValueError(f"Mask disagrees with original polygon: {crop.name}")
        if not mask.any() or min(height, width) < 3:
            rejected["empty or unresolved polygon"] += 1
            continue
        full = full_rgb(child(snap, row["full_image"]), decision["sha256"])
        x0, y0, x1, y1 = row["crop_xyxy"]
        if not np.array_equal(rgb, full[y0:y1, x0:x1]):
            raise ValueError(f"Crop disagrees with source: {crop.name}")
        # Source skin colour comes from the original image around the polygon. It is
        # used as a colour reference only; none of those unlabelled pixels are copied.
        pad = max(3, min(14, round(min(width, height) * 0.18)))
        left, top = max(0, x0 - pad), max(0, y0 - pad)
        around = full[top : y1 + pad, left : x1 + pad]
        inside = np.zeros(around.shape[:2], np.uint8)
        inside[y0 - top : y1 - top, x0 - left : x1 - left] = mask
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * pad + 1, 2 * pad + 1)
        )
        annulus = cv2.dilate(inside, kernel).astype(bool) & ~(inside.astype(bool))
        skin = annulus & skin_map(around)
        if skin.sum() < 20 or skin.sum() < 0.85 * annulus.sum():
            rejected["insufficient skin around source lesion"] += 1
            continue
        reference_colour = np.median(around[skin], axis=0)
        ys, xs = np.nonzero(mask)
        l, t, r, b = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
        donor = Donor(
            row["group_id"],
            f"{row['source_id']}#{row['annotation']}",
            cls,
            rgb[t:b, l:r].copy(),
            mask[t:b, l:r].copy(),
            reference_colour,
        )
        groups[cls][donor.group].append(donor)
        mask_hashes.append(sha(mask_path))
    if not groups:
        raise ValueError(
            "No audited train-split lesion polygons with usable source skin"
        )
    bank = {
        c: [items for _, items in sorted(by_group.items())]
        for c, by_group in sorted(groups.items())
    }
    return bank, dict(
        snapshot=snap.name,
        manifest_sha256=receipt["manifest_sha256"],
        source_archives=sorted(specs),
        accepted_polygons_by_class={c: sum(map(len, g)) for c, g in bank.items()},
        groups_by_class={c: len(g) for c, g in bank.items()},
        excluded=dict(rejected),
        unique_accepted_mask_hashes=len(set(mask_hashes)),
    )


def empirical_sizes(labels, side):
    flat = np.concatenate(labels)
    return {c: flat[flat[:, 0] == c, 3:5] * side for c in (0, 1)}


def pick_size(donor, sizes, rng):
    h, w = donor.mask.shape
    samples = sizes[donor.cls]
    if not len(samples):
        return None
    # Draw a lesion of a size observed in the original training class. Preserve the
    # donor aspect ratio; compare to host GT with the most similar aspect ratios.
    score = np.abs(np.log(samples[:, 0] / samples[:, 1]) - np.log(w / h))
    nearest = np.argsort(score, kind="stable")[: min(40, len(samples))]
    sw, sh = samples[rng.choice(nearest)]
    scale = np.sqrt(float(sw * sh) / (w * h))
    width, height = max(1, round(w * scale)), max(1, round(h * scale))
    if min(width, height) < 4 or max(width, height) > 224 or scale > 3:
        return None
    return width, height


def transfer(im, labels, donor, sizes, rng):
    size = pick_size(donor, sizes, rng)
    if size is None:
        return None
    w, h = size
    canvas_h, canvas_w = im.shape[:2]
    if min(canvas_w - w, canvas_h - h) < 10:
        return None
    weights = donor.mask.astype(np.float32)
    interpolation = cv2.INTER_AREA if w < donor.mask.shape[1] else cv2.INTER_LINEAR
    sampled_weight = cv2.resize(weights, size, interpolation=interpolation)
    premultiplied = cv2.resize(
        donor.rgb * weights[:, :, None], size, interpolation=interpolation
    )
    pixels = premultiplied / np.maximum(sampled_weight[:, :, None], 1e-6)
    mask = cv2.resize(
        donor.mask.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST
    ).astype(bool)
    mask &= sampled_weight > 0
    if mask.sum() < 8:
        return None
    by, bx = np.nonzero(mask)
    # Inner feather only. Never introduce a donor-background halo.
    padded = np.pad(mask.astype(np.uint8), 1)
    distance = cv2.distanceTransform(padded, cv2.DIST_L2, 3)[1:-1, 1:-1]
    feather = min(6, max(1.5, 0.12 * min(w, h)))
    alpha = np.minimum(distance / feather, 1.0)[:, :, None]
    wh = np.array([canvas_w, canvas_h], dtype=np.float32)
    lo = (labels[:, 1:3] - labels[:, 3:5] / 2) * wh
    hi = (labels[:, 1:3] + labels[:, 3:5] / 2) * wh
    margin = max(4, round(0.12 * max(w, h)))
    if 2 * margin + max(w, h) >= min(canvas_h, canvas_w):
        return None
    for _ in range(28):
        x = int(rng.integers(margin, canvas_w - w - margin + 1))
        y = int(rng.integers(margin, canvas_h - h - margin + 1))
        if np.any(
            np.all(hi > [x - margin, y - margin], axis=1)
            & np.all(lo < [x + w + margin, y + h + margin], axis=1)
        ):
            continue
        zone = im[y - margin : y + h + margin, x - margin : x + w + margin]
        if skin_map(zone).mean() < 0.995:
            continue
        target = im[y : y + h, x : x + w]
        ring = np.ones(zone.shape[:2], bool)
        ring[margin : margin + h, margin : margin + w] = False
        context = zone[ring]
        colour = np.median(context, axis=0)
        shift = colour - donor.skin
        if max(abs(shift)) > 68 or np.std(context.mean(1)) > 24:
            continue
        # Relative per-channel illumination preserves dark lesion cores. Additive
        # white balance on saturated donor skins can turn nearly black pits blue.
        gain = (colour + 12.0) / (donor.skin + 12.0)
        if gain.min() < 0.65 or gain.max() > 1.55:
            continue
        # A source orchard and the SSDA workbench use different lighting. Transfer
        # lesion relief in perceptual lightness; preserve local host skin colour
        # and limit the colour residual, particularly on bright scar halos.
        src_lab = cv2.cvtColor((pixels / 255.0).astype(np.float32), cv2.COLOR_RGB2LAB)
        skin_lab = cv2.cvtColor(
            (donor.skin / 255.0).astype(np.float32).reshape(1, 1, 3), cv2.COLOR_RGB2LAB
        )[0, 0]
        host_lab = cv2.cvtColor(target.astype(np.float32) / 255.0, cv2.COLOR_RGB2LAB)
        delta = src_lab - skin_lab
        combined = host_lab.copy()
        combined[:, :, 0] = np.clip(
            host_lab[:, :, 0] + np.clip(delta[:, :, 0], -43, 23), 0, 100
        )
        # Dark pits have substantially lower chroma than illuminated red skin.
        dark_scale = np.clip(
            (combined[:, :, 0] + 8) / (host_lab[:, :, 0] + 8), 0.25, 1.0
        )[:, :, None]
        combined[:, :, 1:] = host_lab[:, :, 1:] * dark_scale + 0.25 * np.clip(
            delta[:, :, 1:], -12, 12
        )
        adjusted = cv2.cvtColor(combined, cv2.COLOR_LAB2RGB) * 255.0
        changed = adjusted * alpha + target * (1 - alpha)
        out = im.copy()
        out[y : y + h, x : x + w] = np.rint(changed).astype(np.uint8)
        l, t, r, b = (
            x + int(bx.min()),
            y + int(by.min()),
            x + int(bx.max()) + 1,
            y + int(by.max()) + 1,
        )
        add = np.array(
            [
                [
                    donor.cls,
                    (l + r) / (2 * canvas_w),
                    (t + b) / (2 * canvas_h),
                    (r - l) / canvas_w,
                    (b - t) / canvas_h,
                ]
            ],
            dtype=np.float32,
        )
        return (
            out,
            np.concatenate((labels, add)),
            dict(
                donor=donor.key,
                group=donor.group,
                target_class=donor.cls,
                box=[l, t, r, b],
                reference_rgb=colour.tolist(),
            ),
        )
    return None


def draw_transfer(im, labels, bank, sizes, rng):
    if rng.random() < PROBABILITY:
        classes = sorted(bank)
        c = int(rng.choice(classes))
        for _ in range(6):
            group = bank[c][rng.integers(len(bank[c]))]
            donor = group[rng.integers(len(group))]
            result = transfer(im, labels, donor, sizes, rng)
            if result is not None:
                return result
    return im, labels, None
