"""CPU-only, lossless PApple train/development raster preparation.

Uses the existing apple_seg.data polygon grouping and rasterizer. The sealed
test split is not opened. Disconnected visible fragments remain one instance;
no contour conversion, hull, hole filling, GT repair, or prediction is used.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import sys

REPO = Path(__file__).resolve().parents[1]
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
sys.path.insert(0, str(REPO / "src"))

import numpy as np
from PIL import Image
from scipy.ndimage import label
from apple_seg.data import DataContractError, PAppleDataset, _rasterize, sha256_file

SCHEMA = "local_apple_native_raster_manifest_v1"
FG_CONNECTIVITY = np.ones((3, 3), dtype=np.uint8)
BG_CONNECTIVITY = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def mask_box(mask):
    y, x = np.nonzero(mask)
    if not len(x):
        raise DataContractError("An annotated instance must not become an empty mask")
    return [int(x.min()), int(y.min()), int(x.max()) + 1, int(y.max()) + 1]


def topology(mask):
    """Count 8-connected foreground and 4-connected enclosed background."""
    box = mask_box(mask)
    x1, y1, x2, y2 = box
    local = np.asarray(mask[y1:y2, x1:x2], dtype=bool)
    _, components = label(local, structure=FG_CONNECTIVITY)
    background, _ = label(~np.pad(local, 1, constant_values=False), structure=BG_CONNECTIVITY)
    exterior = background[0, 0]
    holes = len(set(np.unique(background).tolist()) - {0, int(exterior)})
    return {"components_8": int(components), "holes_4": int(holes), "bbox_xyxy": box}


def apple_identity(image_id, instance_id):
    # Same acquisition-year and apple_ID mapping as the author's diameter code.
    tokens = image_id.split("_")
    capture = tokens[2] if len(tokens) >= 3 else ""
    if not capture or capture[0] not in "2367":
        raise DataContractError(f"Unknown PApple acquisition year: {image_id}")
    year = "2018" if capture[0] in "23" else "2020"
    return f"{year}_01_{int(instance_id):03d}", f"{year}_{capture}"


def visibility_stratum(ratio):
    if ratio > 1:
        return "annotation_visible_area_exceeds_amodal"
    if ratio < .4:
        return "heavy_visible_lt_0.40"
    if ratio < .7:
        return "moderate_visible_0.40_to_0.70"
    if ratio < .9:
        return "light_visible_0.70_to_0.90"
    return "mostly_visible_0.90_to_1.00"


def instance_statistics(v, a, instance_id, image_id, v_polygons, a_polygons):
    vt, at = topology(v), topology(a)
    visible_area, amodal_area = int(v.sum()), int(a.sum())
    ratio = visible_area / amodal_area
    identity, capture = apple_identity(image_id, instance_id)
    h, w = v.shape
    v640, a640 = visible_area * (640 / max(h, w)) ** 2, amodal_area * (640 / max(h, w)) ** 2
    return {"instance_id": str(instance_id), "apple_identity": identity, "capture_group": capture,
            "visible_area_pixels": visible_area, "amodal_area_pixels": amodal_area,
            "visibility_V_area_over_A_area": ratio, "visibility_stratum": visibility_stratum(ratio),
            "visible_outside_amodal_pixels": int((v.astype(bool) & ~a.astype(bool)).sum()),
            "V_area_exceeds_A": visible_area > amodal_area,
            "visible_topology": vt, "amodal_topology": at,
            "visible_source_polygon_count": len(v_polygons), "amodal_source_polygon_count": len(a_polygons),
            "visible_area_at_letterbox640_estimate": v640, "amodal_area_at_letterbox640_estimate": a640,
            "visible_small_under_32sq_at_640": v640 < 32 ** 2,
            "amodal_small_under_32sq_at_640": a640 < 32 ** 2,
            "touches_frame": bool(v[0].any() or v[-1].any() or v[:, 0].any() or v[:, -1].any())}


def encode_masks(visible, amodal, instance_ids):
    if visible.shape != amodal.shape or visible.ndim != 3 or len(instance_ids) != len(visible):
        raise DataContractError("V/A must have the same N,H,W and ordered instance IDs")
    if not (np.isin(visible, [0, 1]).all() and np.isin(amodal, [0, 1]).all()):
        raise DataContractError("Masks must be binary; do not silently binarize")
    _, height, width = visible.shape
    return {"visible_packed": np.packbits(visible, axis=-1, bitorder="little"),
            "amodal_packed": np.packbits(amodal, axis=-1, bitorder="little"),
            "original_hw": np.array([height, width], dtype=np.int32),
            "instance_ids": np.asarray(instance_ids, dtype=str),
            "boxes_visible_xyxy": np.asarray([mask_box(m) for m in visible], dtype=np.float32).reshape(-1, 4),
            "boxes_amodal_xyxy": np.asarray([mask_box(m) for m in amodal], dtype=np.float32).reshape(-1, 4),
            "labels": np.zeros(len(instance_ids), dtype=np.int64)}


def export_sample(sample, split, cache_root):
    ids = sample["instance_ids"]
    if set(sample["visible"]) != set(sample["amodal"]):
        raise DataContractError(f"Unpaired V/A IDs: {sample['filename']}; no target silently dropped")
    with Image.open(sample["rgb_path"]) as image:
        width, height = image.size
    visible = np.empty((len(ids), height, width), dtype=np.uint8)
    amodal = np.empty_like(visible)
    stats = []
    for i, aid in enumerate(ids):
        vp, ap = sample["visible"][aid], sample["amodal"][aid]
        visible[i] = _rasterize(vp, width, height)
        amodal[i] = _rasterize(ap, width, height)
        stats.append(instance_statistics(visible[i], amodal[i], aid, sample["filename"], vp, ap))
    arrays = encode_masks(visible, amodal, ids)
    destination = cache_root / "masks" / split / (Path(sample["filename"]).stem + ".npz")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        with np.load(destination, allow_pickle=False) as old:
            if set(old.files) != set(arrays) or any(not np.array_equal(old[key], value) for key, value in arrays.items()):
                raise DataContractError(f"Existing cache differs from source raster: {destination}")
    else:
        temporary = destination.with_suffix(".tmp")
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **arrays)
        temporary.replace(destination)
    return {"split": split, "image_id": sample["filename"], "rgb_path": str(sample["rgb_path"].resolve()),
            "mask_path": str(destination.resolve()), "rgb_sha256": sha256_file(sample["rgb_path"]),
            "mask_sha256": sha256_file(destination), "original_hw": [height, width],
            "instance_count": len(ids), "empty": not ids, "instances": stats}


def summary(rows):
    instances = [instance for row in rows for instance in row["instances"]]
    return {"images": len(rows), "instances": len(instances), "empty_images": sum(row["empty"] for row in rows),
            "visibility_strata": dict(Counter(x["visibility_stratum"] for x in instances)),
            "visible_components_histogram": dict(Counter(x["visible_topology"]["components_8"] for x in instances)),
            "visible_holes_histogram": dict(Counter(x["visible_topology"]["holes_4"] for x in instances)),
            "amodal_components_histogram": dict(Counter(x["amodal_topology"]["components_8"] for x in instances)),
            "amodal_holes_histogram": dict(Counter(x["amodal_topology"]["holes_4"] for x in instances)),
            "V_not_subset_A_instances": sum(x["visible_outside_amodal_pixels"] > 0 for x in instances),
            "V_area_exceeds_A_instances": sum(x["V_area_exceeds_A"] for x in instances),
            "small_visible_at_640": sum(x["visible_small_under_32sq_at_640"] for x in instances),
            "small_amodal_at_640": sum(x["amodal_small_under_32sq_at_640"] for x in instances),
            "multiple_visible_components_instances": sum(x["visible_topology"]["components_8"] > 1 for x in instances),
            "unique_apple_identities": len({x["apple_identity"] for x in instances}),
            "unique_capture_groups_with_GT": len({x["capture_group"] for x in instances})}


def select_diagnostic(rows, count=16, seed=20261104):
    """Select by original GT only; independent of model outputs and errors."""
    if count < 6 or len(rows) < count:
        raise ValueError("Diagnostic selection requires enough images and >=6 slots")
    rng = random.Random(seed)
    pool = list(sorted(rows, key=lambda row: row["image_id"]))
    rng.shuffle(pool)
    predicates = [
        ("empty", lambda r: r["empty"]),
        ("heavy", lambda r: any(x["visibility_stratum"] == "heavy_visible_lt_0.40" for x in r["instances"])),
        ("multiple_components", lambda r: any(x["visible_topology"]["components_8"] > 1 for x in r["instances"])),
        ("V_area_exceeds_A", lambda r: any(x["V_area_exceeds_A"] for x in r["instances"])),
        ("small_visible", lambda r: any(x["visible_small_under_32sq_at_640"] for x in r["instances"])),
        ("moderate", lambda r: any(x["visibility_stratum"] == "moderate_visible_0.40_to_0.70" for x in r["instances"])),
    ]
    selected, reasons, unavailable = [], {}, []
    for name, predicate in predicates:
        candidates = [row for row in pool if predicate(row)]
        if not candidates:
            unavailable.append(name)
            continue
        candidate = next((r for r in candidates if r["image_id"] not in reasons), candidates[0])
        key = candidate["image_id"]
        if key not in reasons:
            selected.append(candidate)
            reasons[key] = []
        reasons[key].append(name)
    for row in pool:
        if len(selected) == count:
            break
        if row["image_id"] not in reasons:
            selected.append(row)
            reasons[row["image_id"]] = ["seeded_GT_independent_remainder"]
    required = {name for name, pred in predicates[:3] if any(pred(row) for row in rows)}
    if not required.issubset({reason for values in reasons.values() for reason in values}):
        raise AssertionError("Required smoke cases were not selected")
    return selected, {"seed": seed, "selection_uses_predictions": False, "reasons": reasons,
                      "unavailable_categories": unavailable, "purpose": "engineering smoke/diagnosis only; not precision acceptance"}


def split_overlap(train, val):
    def index(rows, field):
        result = defaultdict(set)
        for row in rows:
            for instance in row["instances"]:
                result[instance[field]].add(row["image_id"])
        return result
    result = {}
    for field in ("apple_identity", "capture_group"):
        left, right = index(train, field), index(val, field)
        common = sorted(set(left) & set(right))
        result[field] = {"train_unique": len(left), "val_unique": len(right), "overlap_unique": len(common),
                         "shared": [{"key": key, "train_images": sorted(left[key]), "val_images": sorted(right[key])} for key in common]}
    trgb = defaultdict(list)
    for row in train:
        trgb[row["rgb_sha256"]].append(row["image_id"])
    result["identical_RGB_cross_split"] = [{"sha256": row["rgb_sha256"], "train_images": trgb[row["rgb_sha256"]],
                                            "val_image": row["image_id"]} for row in val if row["rgb_sha256"] in trgb]
    return result


def manifest(split, rows, source, selection=None):
    value = {"schema": SCHEMA, "split": split, "role": "train" if split == "train" else "development",
             "created_utc": utc_now(), "source": source, "classes": {"0": "apple"},
             "raster_contract": {"packed_bitorder": "little", "packed_axis": -1,
                 "decode": "np.unpackbits(packed, axis=-1, count=original_hw[1], bitorder='little')",
                 "native_masks": True, "one_ID_one_instance": True, "bbox_convention": "half-open pixel xyxy",
                 "visible_boxes_for_visible_baseline": True, "V_clipped_to_A": False,
                 "hull_or_contour_reconstruction": False, "empty_images_retained": True},
             "statistics": summary(rows), "samples": rows}
    if selection:
        value["diagnostic_selection"] = selection
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=REPO.parent / "apple_vision/local_amodal/data/papple/data")
    parser.add_argument("--cache-root", type=Path, default=REPO / "cache/datasets/local_apple_v1")
    parser.add_argument("--report-root", type=Path, default=REPO / "reports/local_train_20261104/data_v1")
    args = parser.parse_args(argv)
    for location in (args.source_root, args.cache_root, args.report_root):
        if location.resolve().drive.upper() != "F:":
            raise ValueError(f"This preparation is restricted to F drive: {location}")
    import torch
    torch.set_num_threads(2)
    if torch.cuda.is_initialized():
        raise RuntimeError("CPU preparation must not initialize CUDA")
    args.report_root.mkdir(parents=True, exist_ok=True)
    args.cache_root.mkdir(parents=True, exist_ok=True)
    status_path = args.report_root / "pipeline_status.json"
    datasets, rows_by_split = {}, {}
    try:
        for split in ("train", "val"):
            dataset = PAppleDataset(args.source_root, split=split, input_size=640, augment=False)
            datasets[split] = dataset
            expected = {"train": (2304, 8579, 14), "val": (814, 3449, 13)}[split]
            meta = dataset.metadata()
            actual = (len(dataset), meta["GT_instances"], meta["empty_GT_images"])
            if actual != expected:
                raise DataContractError(f"Official split contract differs: {split} {actual} != {expected}")
            rows = []
            for i, sample in enumerate(dataset.samples):
                rows.append(export_sample(sample, split, args.cache_root))
                if (i + 1) % 50 == 0 or i + 1 == len(dataset):
                    progress = {"phase": "rasterizing", "split": split, "completed": i + 1,
                                "total": len(dataset), "updated_utc": utc_now(), "GPU_used": False}
                    write_json(status_path, progress)
                    print(json.dumps(progress), flush=True)
            rows_by_split[split] = rows
            source = {"data_root": str(dataset.root), "annotation_paths": {k: str(v) for k, v in dataset.annotation_paths.items()},
                      "annotation_sha256": dataset.annotation_sha256, "rasterizer": "apple_seg.data._rasterize",
                      "rasterizer_file_sha256": sha256_file(REPO / "src/apple_seg/data.py"),
                      "split_policy": "official supplied train/val retained; val is development, not independent test"}
            full = manifest(split, rows, source)
            if split == "val":
                expected_strata = {"heavy_visible_lt_0.40": 1514, "moderate_visible_0.40_to_0.70": 1079,
                                   "light_visible_0.70_to_0.90": 627, "mostly_visible_0.90_to_1.00": 215,
                                   "annotation_visible_area_exceeds_amodal": 14}
                if full["statistics"]["visibility_strata"] != expected_strata:
                    raise DataContractError("Original strong-model VAL visibility strata differ")
            write_json(args.cache_root / f"{split}_manifest.json", full)
            chosen, reasons = select_diagnostic(rows)
            diagnostic = manifest(split, chosen, source, reasons)
            name = "smoke_train16.json" if split == "train" else "diagnostic_val16.json"
            write_json(args.report_root / name, diagnostic)
        overlap = split_overlap(rows_by_split["train"], rows_by_split["val"])
        write_json(args.report_root / "split_overlap.json", overlap)
        anomalies = [{"split": split, "image_id": row["image_id"], **inst}
                     for split, rows in rows_by_split.items() for row in rows for inst in row["instances"]
                     if inst["visible_outside_amodal_pixels"] > 0]
        write_json(args.report_root / "annotation_containment_audit.json", {
            "GT_modified": False, "instances_retained": True, "criterion": "any visible raster pixel outside A",
            "area_exceeds_A_is_separate_subset": True, "instances": anomalies})
        report = {"schema": "local_apple_training_data_audit_v1", "completed_utc": utc_now(),
                  "GPU_used": False, "predictions_used": False, "original_data_modified": False,
                  "script_sha256": sha256_file(__file__),
                  "splits": {split: summary(rows) for split, rows in rows_by_split.items()},
                  "overlap_summary": {key: {k: v for k, v in values.items() if k != "shared"}
                                      for key, values in overlap.items() if isinstance(values, dict)},
                  "identical_RGB_cross_split": len(overlap["identical_RGB_cross_split"]),
                  "test": {"accessed": False, "known_images_from_previous_inventory": 807,
                           "policy": "sealed; not opened, hashed, sampled or used in this preparation"},
                  "validation_role": "development; repeated tuning and shared identities prevent independent-test claim",
                  "masks": "original native raster packed losslessly; polygon union by apple_ID; no hull/repair/splitting",
                  "visibility_definition": "Original strong-model strata for V/A area: heavy<0.40; moderate[0.40,0.70); light[0.70,0.90); mostly[0.90,1.00]; >1 anomaly",
                  "rasterizer_matches_original_strong_runtime": sha256_file(REPO / "src/apple_seg/data.py") == sha256_file(REPO / "strong_runtime/src/apple_seg/data.py"),
                  "small_definition": "area at uniform letterbox640 scale <32^2 pixels; descriptive, not exclusion",
                  "holes_definition": "8-connected foreground with 4-connected background holes, no filling",
                  "outputs": {"train": str(args.cache_root / "train_manifest.json"),
                              "val": str(args.cache_root / "val_manifest.json"),
                              "smoke_train16": str(args.report_root / "smoke_train16.json"),
                              "diagnostic_val16": str(args.report_root / "diagnostic_val16.json")}}
        write_json(args.report_root / "data_summary.json", report)
        lines = ["# 局部苹果分割训练数据审计", "", "官方 train/val 划分保留；验证集仅作 development，不称独立测试集。test 807 张本阶段未打开。", "",
                 "| 划分 | 图片 | 实例 | 空图 | 重遮挡实例 | 多分片实例 | V面积>A实例 | V存在A外像素实例 |",
                 "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for split, stats in report["splits"].items():
            lines.append(f"| {split} | {stats['images']} | {stats['instances']} | {stats['empty_images']} | {stats['visibility_strata'].get('heavy_visible_lt_0.40',0)} | {stats['multiple_visible_components_instances']} | {stats['V_area_exceeds_A_instances']} | {stats['V_not_subset_A_instances']} |")
        lines += ["", f"train/val 共有果实身份数：{overlap['apple_identity']['overlap_unique']}；共有原始拍摄编号数：{overlap['capture_group']['overlap_unique']}；相同 RGB SHA 的跨划分验证图数：{len(overlap['identical_RGB_cross_split'])}。",
                  "", "使用原 apple_seg.data 按 apple_ID 合并原始多边形并栅格化。原尺寸 V/A 二值掩膜以 little-endian packbits 无损保存，一个苹果只有一个实例；没有取最大轮廓、凸包、填洞或将 V 裁剪到 A。原 GT 中的包含/面积异常全部保留并逐项记录。",
                  "", "固定 smoke_train16 与 diagnostic_val16 仅按原 GT 分层加固定随机种子选择，包含空图、重遮挡和多分片；没有参考预测。用于工程烟测，不能用作 Precision 验收。",
                  "", "遮挡分层沿用原强模型：V/A<0.40为重遮挡；[0.40,0.70)为中遮挡；[0.70,0.90)为轻遮挡；[0.90,1.00]为基本可见；>1独立异常组，原VAL的14个异常全部保留。复用栅格器源码与strong_runtime相同哈希。",
                  "", "小目标定义为等比例缩放到640后估计面积<32²；拓扑采用8连通前景/4连通背景洞。数据并非末端 D405 真实近景时序，后续需要独立采集验收。"]
        (args.report_root / "data_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        write_json(status_path, {"phase": "completed", "updated_utc": utc_now(), "GPU_used": False,
                                 "summary": str(args.report_root / "data_summary.json")})
        print(json.dumps({"completed": str(args.report_root / "data_summary.json")}), flush=True)
    except Exception as error:
        write_json(status_path, {"phase": "failed", "updated_utc": utc_now(), "error": repr(error), "GPU_used": False})
        raise


if __name__ == "__main__":
    main()
