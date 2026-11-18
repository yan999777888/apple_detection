"""Re-create the frozen, visually reviewed gray-tile-session training anchors."""
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "datasets/harvard_grade_normal_anchors_v1"
META = ROOT / "datasets/apple_matched_quality_v1/MR4SXI.json"
HDF = ROOT / "apple_research_v2/sources/additional_attribute_sources/hdf_real_only/raw/fresh"
# Examined all 32 single-fruit views side-by-side. Exclude unclear dark blemishes
# and additional views from apparently the same pose. The remaining poses still
# share a session and possibly physical fruit, so all belong to ONE training group.
APPROVED = [2, 3, 4, 8, 10, 12, 17, 20, 23, 24, 30, 32]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fingerprints(image):
    small = cv2.resize(np.asarray(image.convert("RGB")), (32, 32), interpolation=cv2.INTER_AREA)
    variants = np.stack([np.rot90(v, n) for v in (small, small[:, ::-1]) for n in range(4)])
    return variants


def fruit_crop(path):
    rgb = np.array(Image.open(path).convert("RGB"))
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    colorful = np.uint8((hsv[:, :, 1] >= 65) & (hsv[:, :, 2] >= 40)) * 255
    joined = cv2.morphologyEx(colorful, cv2.MORPH_CLOSE, np.ones((13, 13), np.uint8))
    contours, _ = cv2.findContours(joined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    x, y, width, height = cv2.boundingRect(max(contours, key=cv2.contourArea))
    padding = round(.10 * max(width, height))
    box = [max(0, x-padding), max(0, y-padding), min(rgb.shape[1], x+width+padding), min(rgb.shape[0], y+height+padding)]
    return Image.fromarray(rgb).crop(box), box


def main():
    data = json.loads(META.read_text(encoding="utf-8"))
    assert data["license"]["rightsIdentifier"] == "CC0-1.0"
    files = {item["dataFile"]["filename"]: item["dataFile"] for item in data["files"]}
    hdf_paths = sorted(HDF.glob("*.jpg"))
    assert len(hdf_paths) == 2500
    hdf_hash = set()
    hdf_views = []
    for p in hdf_paths:
        hdf_hash.add(sha(p))
        with Image.open(p) as im:
            hdf_views.append(fingerprints(im)[0])
    hdf_views = np.stack(hdf_views).astype(np.int16)
    crops = BASE / "crops"
    crops.mkdir(parents=True, exist_ok=True)
    samples = []
    for number in APPROVED:
        name = f"Anjali_Apples_Good_{number:05d}.jpg"
        path = BASE / "originals" / name
        assert hashlib.md5(path.read_bytes()).hexdigest() == files[name]["md5"]
        crop, box = fruit_crop(path)
        out = crops / f"{number:05d}.png"
        crop.save(out)
        assert not {sha(path), sha(out)} & hdf_hash
        smallest = float("inf")
        for transformed in fingerprints(crop):
            # Color and shape screen against all previous HDF fresh photos.
            d = np.abs(hdf_views - transformed).mean(axis=(1, 2, 3))
            smallest = min(smallest, float(d.min()))
        assert smallest > 12, (name, smallest)
        samples.append(dict(path=out.relative_to(ROOT).as_posix(), sha256=sha(out),
                            source="Harvard_Dataverse_Good_Apple_CC0", source_member=name,
                            original_path=path.relative_to(ROOT).as_posix(), source_sha256=sha(path),
                            source_md5=files[name]["md5"], dataverse_file_id=files[name]["id"],
                            grade=0, split="train", group_id="Harvard_Anjali_gray_floor_session",
                            condition_id="Harvard_Anjali_gray_floor_session", crop_xyxy=box,
                            nearest_hdf_fresh_mean_absolute_RGB_32_D4=round(smallest,3)))
    manifest = dict(doi="10.7910/DVN/MR4SXI", author="Anjali Sharma", license="CC0-1.0",
                    label_basis="Author good class plus visual screening for evidently intact visible skin. No commercial inspection of unseen sides.",
                    sample_unit="Views of potentially repeated fruits. One training-only floor session. No independent per-fruit count inferred.",
                    overlap_screen="Original and crop SHA-256, minimum mean absolute RGB at 32x32 across rotations/reflections vs all 2500 HDF fresh. Conservative source/scene exclusion overrides image-level splits.",
                    counts={"train":{"0":len(samples)}}, samples=samples)
    (BASE/"manifest.json").write_text(json.dumps(manifest, ensure_ascii=False,indent=2)+"\n",encoding="utf-8")


if __name__ == "__main__":
    main()
