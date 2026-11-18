"""Prepare anonymous, train-only OOF lesion review cases and a private provenance key."""

import hashlib
import math
import os
import random
import shutil
import tempfile
import uuid
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

from audit_defect_causes_20261112 import match
from class0_patch_refinement import ROOT, atomic_json, read, sha
from sources.additional_attribute_sources.audit_hdf_perceptual import OPS, phash, dhash

MANIFEST = ROOT / "apple_research_v2/datasets/ssda_date_v2/manifest.json"
OOF = ROOT / "experiments/defect_algorithm_steps_20261112_v2/oof_v2"
OUT = ROOT / "experiments/ssda_blind_defects_v1"


def size(box):
    l, t, r, b = box
    return np.array([r - l, b - t])


def native_gt(row):
    h, w = row["native_hw"]
    labels = []
    for v in row["instances"]:
        x, y, bw, bh = v["xywh_normalized"]
        labels.append(
            dict(
                box=[
                    (x - bw / 2) * w,
                    (y - bh / 2) * h,
                    (x + bw / 2) * w,
                    (y + bh / 2) * h,
                ],
                **{"class": v["class_id"]},
            )
        )
    return labels


def real_oof():
    samples = read(MANIFEST)["samples"]
    train = {r["image_id"]: r for r in samples if r["partition"] == "train"}
    protected = {r["rgb_sha256"] for r in samples if r["partition"] != "train"}
    assert len(train) == 596 and sum(r["partition"] == "dev" for r in samples) == 113
    assert not ({r["rgb_sha256"] for r in train.values()} & protected)
    candidates, controls, provenance, covered = [], [], [], set()
    for fold in range(3):
        directory = OOF / f"fold{fold}"
        split = read(directory / "membership.json")
        fit, held = set(split["train"]), set(split["heldout"])
        assert fit.isdisjoint(held) and fit | held == set(train)
        assert {train[k]["same_second_group"] for k in fit}.isdisjoint(
            {train[k]["same_second_group"] for k in held}
        )
        doc = read(directory / "predictions.json")
        run = ROOT / "runs" / f"defect_oof3_fold{fold}_e40_20261112_v1"
        cp = read(run / "final_selection.json")["primary"]
        assert (
            sha(Path(cp["path"])) == doc["checkpoint_sha256"] == cp["sha256"]
            and cp["epoch"] == 40
        )
        assert set(r["image_id"] for r in doc["rows"]) == held and not covered & held
        assert len(doc["rows"]) == len(held)
        covered |= held
        provenance.append(
            dict(
                fold=fold,
                fit=len(fit),
                heldout=len(held),
                sha_checkpoint=cp["sha256"],
                sha_membership=sha(directory / "membership.json"),
                sha_predictions=sha(directory / "predictions.json"),
            )
        )
        for row in doc["rows"]:
            key = row["image_id"]
            gt = native_gt(train[key])
            assert row["fold"] == fold and len(gt) == len(row["GT"])
            for g, h in zip(gt, row["GT"]):
                assert g["class"] == h["class"] and np.allclose(
                    g["box"], h["box"], rtol=0, atol=1e-3
                )
            for i, g in enumerate(gt):
                controls.append(
                    dict(
                        image=key,
                        box=g["box"],
                        cls=g["class"],
                        gt_index=i,
                        fold=fold,
                        source="native positive",
                    )
                )
            pairs, overlap = match(row["pred"], gt)
            for i, p in enumerate(row["pred"]):
                box = p[:4]
                maximum = max(overlap[i], default=0.0)
                x, y = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
                inside = any(
                    l - 2 <= x <= r + 2 and t - 2 <= y <= b + 2
                    for l, t, r, b in (g["box"] for g in gt)
                )
                if i in pairs or p[4] < 0.45 or maximum >= 0.10 or inside:
                    continue
                candidates.append(
                    dict(
                        image=key,
                        box=box,
                        cls=int(p[5]),
                        confidence=p[4],
                        max_gt_iou=float(maximum),
                        pred_index=i,
                        fold=fold,
                        source="unmatched OOF candidate",
                    )
                )
    assert covered == set(train)
    return train, candidates, controls, provenance


class UniquePhotos:
    """Exact, capture-time and close full-frame mirrored/rotated photo exclusion."""

    def __init__(self, train):
        self.train = train
        self.keys = set()
        self.groups = set()
        self.hashes = set()
        self.reference = []
        self.cache = {}

    def fresh(self, key):
        row = self.train[key]
        if (
            key in self.keys
            or row["same_second_group"] in self.groups
            or row["rgb_sha256"] in self.hashes
        ):
            return False
        if key not in self.cache:
            with Image.open(row["rgb_path"]) as im:
                gray = im.convert("L")
                self.cache[key] = [
                    (phash(g), dhash(g))
                    for g in [gray if op is None else gray.transpose(op) for op in OPS]
                ]
        return not any(
            (a ^ c).bit_count() <= 7 and (b ^ d).bit_count() <= 8
            for a, b in self.cache[key]
            for c, d in self.reference
        )

    def add(self, key):
        assert self.fresh(key)
        row = self.train[key]
        self.keys.add(key)
        self.groups.add(row["same_second_group"])
        self.hashes.add(row["rgb_sha256"])
        self.reference.append(self.cache[key][0])


def select(train, candidates, controls):
    used = UniquePhotos(train)
    picked = []
    for cls in (0, 1):
        chosen = [r for r in candidates if r["cls"] == cls]
        chosen.sort(key=lambda r: (-r["confidence"], r["image"], r["pred_index"]))
        n = 0
        for r in chosen:
            if used.fresh(r["image"]):
                used.add(r["image"])
                picked.append(r)
                n += 1
                if n == 12:
                    break
        if n != 12:
            raise ValueError(
                f"Only {n} unique high-score candidates available for class {cls}"
            )
    for cls in (0, 1):
        cases = [
            v
            for v in picked
            if v["cls"] == cls and v["source"] == "unmatched OOF candidate"
        ]
        for index in (0, 2, 5, 8, 11):
            ref = cases[index]
            cam = train[ref["image"]]["capture_filename_tokens"]["camera"]
            base = size(ref["box"])
            options = []
            for r in controls:
                if (
                    r["cls"] != cls
                    or train[r["image"]]["capture_filename_tokens"]["camera"] != cam
                ):
                    continue
                delta = np.abs(np.log(size(r["box"]) / base))
                if delta.max() <= math.log(1.30):
                    options.append((float(delta.max() + 0.25 * delta.sum()), r))
            for _, r in sorted(
                options, key=lambda x: (x[0], x[1]["image"], x[1]["gt_index"])
            ):
                if used.fresh(r["image"]):
                    used.add(r["image"])
                    picked.append(
                        dict(
                            r,
                            matched_candidate=[ref["image"], ref["pred_index"]],
                            relative_size=(size(r["box"]) / base).tolist(),
                        )
                    )
                    break
            else:
                raise ValueError("No same-camera, size-matched control")
    assert len(picked) == 34 and len(used.groups) == 34
    return picked


def build():
    if OUT.exists():
        raise FileExistsError(f"Review packet already present: {OUT}")
    train, candidates, controls, folds = real_oof()
    chosen = select(train, candidates, controls)
    random.SystemRandom().shuffle(chosen)
    # Publish once so a partial run cannot leave an apparently ready review bundle.
    staging = Path(tempfile.mkdtemp(prefix="blind-review-", dir=OUT.parent))
    try:
        public = staging / "盲审材料"
        private = staging / "组织者留存"
        images = public / "images"
        images.mkdir(parents=True)
        private.mkdir()
        subjects = []
        key_data = []
        for item in chosen:
            ident = uuid.uuid4().hex[:12]
            raw = train[item["image"]]
            if (
                sha(Path(raw["rgb_path"])) != raw["rgb_sha256"]
                or sha(Path(raw["label_path"])) != raw["label_sha256"]
            ):
                raise ValueError("Frozen source changed")
            with Image.open(raw["rgb_path"]) as im:
                frame = Image.fromarray(np.asarray(im.convert("RGB")))
            l, t, r, b = item["box"]
            cx, cy = round((l + r) / 2), round((t + b) / 2)
            side = 256
            left = int(np.clip(cx - side // 2, 0, frame.width - side))
            top = int(np.clip(cy - side // 2, 0, frame.height - side))
            crop = frame.crop((left, top, left + side, top + side))
            frame.save(images / f"{ident}_whole.png")
            crop.save(images / f"{ident}_local.png")
            subjects.append(
                dict(
                    id=ident,
                    whole=f"images/{ident}_whole.png",
                    local=f"images/{ident}_local.png",
                    center=[cx, cy],
                    local_center=[cx - left, cy - top],
                )
            )
            key_data.append(
                dict(
                    id=ident,
                    **item,
                    raw_sha=raw["rgb_sha256"],
                    camera=raw["capture_filename_tokens"]["camera"],
                    capture_group=raw["same_second_group"],
                    crop_xyxy=[left, top, left + side, top + side],
                )
            )
        manifest = public / "cases.json"
        atomic_json(manifest, dict(schema=1, cases=subjects))
        packet_sha = sha(manifest)
        atomic_json(
            private / "source_key.json",
            dict(
                case_manifest_sha=packet_sha,
                ssda_manifest_sha=sha(MANIFEST),
                role="reference labels and OOF provenance; candidate diagnoses unknown, never training annotations",
                train_only=True,
                dev_cases=0,
                folds=folds,
                cases=key_data,
            ),
        )
        source = Path(__file__).with_name("defect_blind_review_app.py")
        shutil.copyfile(source, public / "review.py")
        python = ROOT.parent / "apple_detection/envs/local_rgbd_v1/Scripts/pythonw.exe"
        assert python.is_file()
        relative = os.path.relpath(python, public)
        (public / "开始盲审.cmd").write_bytes(
            (
                '@echo off\r\nset "PYTHONDONTWRITEBYTECODE=1"\r\nstart "" "%~dp0'
                + relative
                + '" -B "%~dp0review.py"\r\n'
            ).encode("ascii")
        )
        (public / "阅读说明.txt").write_text(
            "双击“开始盲审”。首次填写请自选匿名代号，再次打开输入同一代号可接续。\n"
            "看整果位置和未经修饰的局部原图，判断定位标记中心。四项任选一项。"
            "痕迹含义不清或无法区分时选“看不清／无法判断”。可前后翻页修改。每次选择自动保存在本文件夹；"
            "点击“导出已填结果”会在本文件夹的“导出结果”中保存 CSV 和 JSON，同时显示完整路径。\n"
            "独立复核时给每位审阅人单独一份盲审材料，请在各自提交前独立填写。\n",
            encoding="utf-8-sig",
        )
        os.rename(staging, OUT)
    finally:
        if staging.exists():
            # Only ever clean the fresh staging directory created above.
            assert (
                staging.resolve().parent == OUT.parent.resolve()
                and staging.name.startswith("blind-review-")
            )
            shutil.rmtree(staging)
    print("train-only cases", Counter((v["source"], v["cls"]) for v in chosen), OUT)


if __name__ == "__main__":
    build()
