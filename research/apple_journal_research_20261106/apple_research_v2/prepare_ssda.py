"""Verify and prepare the CC-BY-4.0 SSDA v1 archive without executing repo code.

Preserves original train/val directories; training lists are rebuilt by capture
date. The new dev split is date-held-out, NOT verified independent fruit data.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
import re
import stat
import sys
import zipfile

sys.dont_write_bytecode = True
EXPECTED_BYTES = 160596573
EXPECTED_MD5 = "86ad2aad9cbe1bdba8b85814a2cfe8c8"
EXPECTED_SHA256 = "9d98ee39e3c7dffccdc42ab11606d88002b35be316c44da1ed08d99830f9f0fc"
TRAIN_DATES = {"20211104"}
DEV_DATES = {"20211025", "20211027"}
FILE_PATTERN = re.compile(r"(?P<camera>\d+)_(?P<date>\d{8})_(?P<time>\d{6})_(?P<millisecond>\d{3})")


def digest(path, algorithm="sha256"):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, algorithm).hexdigest()


def selected_path(info):
    """Only literal image/label files under one release root may be extracted."""
    value = info.filename
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or "\\" in value or ":" in value:
        raise ValueError(f"Unsafe archive member: {value}")
    if stat.S_ISLNK(info.external_attr >> 16):
        raise ValueError(f"Symlink archive member: {value}")
    if info.is_dir():
        return None
    parts = path.parts
    if len(parts) != 4 or parts[1] not in {"images", "labels"} or parts[2] not in {"train", "val"}:
        return None
    suffix = ".jpg" if parts[1] == "images" else ".txt"
    if not parts[3].endswith(suffix) or not FILE_PATTERN.fullmatch(Path(parts[3]).stem):
        raise ValueError(f"Unexpected dataset filename: {value}")
    if not 0 < info.file_size <= 10 * 1024 * 1024:
        raise ValueError(f"Unexpected member size: {value}")
    return PurePosixPath(*parts[1:])


def parse_labels(text, identity, permit_out_of_bounds=False):
    boxes = []
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 5:
            raise ValueError(f"Expected class,cx,cy,w,h at {identity}:{line_number}")
        values = [float(x) for x in fields]
        if not all(math.isfinite(x) for x in values) or values[0] not in (0, 1):
            raise ValueError(f"Unknown class or nonfinite value: {identity}:{line_number}")
        cls, x, y, w, h = values
        if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1):
            raise ValueError(f"Invalid normalized bounding box: {identity}:{line_number}")
        out_of_bounds = min(x-w/2, y-h/2) < 0 or max(x+w/2, y+h/2) > 1
        if out_of_bounds and not permit_out_of_bounds:
            raise ValueError(f"Bounding box extends outside image: {identity}:{line_number}")
        boxes.append({"class_id": int(cls), "xywh_normalized": [x, y, w, h],
                      "source_line": line_number, "source_out_of_bounds": out_of_bounds})
    if not boxes:
        raise ValueError(f"Unexpected empty source label: {identity}")
    return boxes


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def prepare(archive, output, metadata_path):
    from PIL import Image
    archive, output, metadata_path = archive.resolve(), output.resolve(), metadata_path.resolve()
    if output.exists():
        raise ValueError("Use a new output directory; existing evidence will not be overwritten")
    if archive.suffix.lower() != ".zip" or archive.stat().st_size != EXPECTED_BYTES:
        raise ValueError("Archive is incomplete or has unexpected byte size")
    actual_md5, actual_sha = digest(archive, "md5"), digest(archive)
    if actual_md5 != EXPECTED_MD5 or actual_sha != EXPECTED_SHA256:
        raise ValueError("SSDA release archive checksum mismatch")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8-sig"))
    if metadata["zenodo"]["metadata"]["license"]["id"] != "cc-by-4.0":
        raise ValueError("Expected original Zenodo CC BY 4.0 license receipt")
    tree = {x["path"]: x for x in metadata["github_tree"]["tree"] if x["type"] == "blob"}
    if metadata["github_tree"]["truncated"]:
        raise ValueError("Need complete release tree metadata")
    report = {"schema": "ssda_preparation_v2", "phase": "auditing", "created_utc": datetime.now(timezone.utc).isoformat(),
              "archive": str(archive), "archive_bytes": EXPECTED_BYTES, "archive_md5": actual_md5,
              "archive_sha256": actual_sha, "metadata_sha256": digest(metadata_path), "script_sha256": digest(__file__),
              "license": "CC BY 4.0", "license_source": "https://zenodo.org/api/records/10957715",
              "numeric_classes": {"0": "class_0", "1": "class_1"}, "semantic_mapping_verified": False,
              "train_dates": sorted(TRAIN_DATES), "development_dates": sorted(DEV_DATES),
              "fruit_identity_known": False, "independent_fruit_test_claimed": False, "test_partition_created": False,
              "repository_code_executed": False, "downloaded_cache_files_loaded": False,
              "original_data_modified": False, "maturity_or_grade_labels_generated": False}
    output.mkdir(parents=True, exist_ok=False)
    try:
        with zipfile.ZipFile(archive) as z:
            entries = {}
            for info in z.infolist():
                rel = selected_path(info)
                if rel is not None:
                    key = str(rel)
                    if key in entries:
                        raise ValueError(f"Duplicate archive output: {key}")
                    entries[key] = info
            expected_keys = {k for k in tree if re.fullmatch(r"(?:images|labels)/(?:train|val)/[^/]+\.(?:jpg|txt)", k)}
            if set(entries) != expected_keys or len(entries) != 1420:
                raise ValueError("Image/label member inventory differs from pinned Git release")
            rows, original_counts, date_counts, class_counts, image_shapes = [], Counter(), Counter(), defaultdict(Counter), Counter()
            timestamp_splits = defaultdict(set)
            source_image_shas = defaultdict(list)
            file_receipts, invalid_annotations, quarantine_groups = [], [], set()
            for key in sorted(k for k in entries if k.startswith("images/")):
                image_rel = PurePosixPath(key)
                label_key = str(PurePosixPath("labels", *image_rel.parts[1:]).with_suffix(".txt"))
                raw = {}
                for member_key in (key, label_key):
                    data = z.read(entries[member_key])  # zipfile verifies member CRC on read.
                    blob_sha = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
                    if blob_sha != tree[member_key]["sha"] or len(data) != tree[member_key]["size"]:
                        raise ValueError(f"Git blob mismatch: {member_key}")
                    file_hash = hashlib.sha256(data).hexdigest()
                    destination = output / "original" / Path(*PurePosixPath(member_key).parts)
                    if not destination.resolve().is_relative_to(output):
                        raise ValueError("Extraction escaped output directory")
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    destination.write_bytes(data)
                    file_receipts.append({"path": member_key, "sha256": file_hash, "git_blob_sha1": blob_sha, "bytes": len(data)})
                    raw[member_key] = data
                with Image.open(io.BytesIO(raw[key])) as image:
                    image.verify()
                with Image.open(io.BytesIO(raw[key])) as image:
                    image.load()
                    width, height, mode = image.width, image.height, image.mode
                if mode != "RGB" or width < 1 or height < 1:
                    raise ValueError(f"Unexpected image mode or dimensions: {key}")
                boxes = parse_labels(raw[label_key].decode("utf-8-sig"), label_key, permit_out_of_bounds=True)
                tokens = FILE_PATTERN.fullmatch(image_rel.stem).groupdict()
                date = tokens["date"]
                if date not in TRAIN_DATES | DEV_DATES:
                    raise ValueError(f"Unspecified acquisition date: {date}")
                partition = "train" if date in TRAIN_DATES else "dev"
                original_split = image_rel.parts[1]
                image_sha = hashlib.sha256(raw[key]).hexdigest()
                original_counts[original_split] += 1
                date_counts[date] += 1
                image_shapes[f"{width}x{height}"] += 1
                class_counts[f"original_{original_split}"].update(box["class_id"] for box in boxes)
                source_image_shas[image_sha].append(partition)
                timestamp_splits[date + "_" + tokens["time"]].add(original_split)
                same_second_group = date + "_" + tokens["time"]
                for box in boxes:
                    if box["source_out_of_bounds"]:
                        cx, cy, bw, bh = box["xywh_normalized"]
                        l, t, r, b = cx-bw/2, cy-bh/2, cx+bw/2, cy+bh/2
                        invalid_annotations.append({"label_file": label_key, "source_line": box["source_line"],
                            "class_id": box["class_id"], "xywh_normalized": box["xywh_normalized"],
                            "xyxy_normalized": [l,t,r,b], "native_wh": [width,height],
                            "overflow_pixels_ltrb": [max(0,-l)*width,max(0,-t)*height,max(0,r-1)*width,max(0,b-1)*height],
                            "same_second_group": same_second_group})
                        quarantine_groups.add(same_second_group)
                rows.append({"image_id": image_rel.name, "original_split": original_split, "partition": partition,
                    "requested_partition": partition, "same_second_group": same_second_group,
                    "rgb_path": str((output / "original" / Path(*image_rel.parts)).resolve()),
                    "label_path": str((output / "original" / Path(*PurePosixPath(label_key).parts)).resolve()),
                    "rgb_sha256": image_sha, "label_sha256": hashlib.sha256(raw[label_key]).hexdigest(),
                    "native_hw": [height, width], "capture_filename_tokens": tokens,
                    "split_group": date, "split_group_kind": "acquisition_date_only",
                    "fruit_id": None, "sequence_id": None, "instances": boxes})
        if dict(original_counts) != {"train": 568, "val": 142}:
            raise ValueError("Unexpected original split count")
        excluded = []
        for row in rows:
            row["training_eligible"] = row["same_second_group"] not in quarantine_groups
            if not row["training_eligible"]:
                row["partition"] = "quarantine"
                row["exclusion_reason"] = "same-second acquisition group contains out-of-bounds source annotation; entire image and group excluded without clipping"
                excluded.append(row)
            class_counts[row["partition"]].update(box["class_id"] for box in row["instances"])
        for partition in ("train", "dev"):
            if set(class_counts[partition]) != {0, 1}:
                raise ValueError(f"Both numeric classes must be represented in {partition}")
        shared_hashes = [h for h, parts in source_image_shas.items() if len(set(parts)) > 1]
        if shared_hashes:
            raise ValueError("Identical image bytes cross the new date split")
        split_dir = output / "splits"
        split_dir.mkdir()
        for partition in ("train", "dev"):
            names = [Path(r["rgb_path"]).as_posix() for r in rows if r["partition"] == partition]
            (split_dir / f"{partition}.txt").write_text("\n".join(names) + "\n", encoding="utf-8")
        yaml = ("# Date-held-out development only; physical fruit independence unknown.\n"
                f"path: {json.dumps(output.as_posix(), ensure_ascii=False)}\n"
                "train: splits/train.txt\nval: splits/dev.txt\n"
                "names:\n  0: class_0\n  1: class_1\n")
        (output / "data.yaml").write_text(yaml, encoding="utf-8")
        included = [r for r in rows if r["training_eligible"]]
        positive_overflows = [v for a in invalid_annotations for v in a["overflow_pixels_ltrb"] if v > 0]
        report.update(phase="complete", images=len(rows), included_images=len(included), image_dimensions=dict(image_shapes),
            partitions=dict(Counter(r["partition"] for r in rows)), original_split_counts=dict(original_counts),
            capture_dates=dict(date_counts), class_instance_counts={k:dict(v) for k,v in class_counts.items()},
            total_instances=sum(len(r["instances"]) for r in included),
            source_total_instances=sum(len(r["instances"]) for r in rows), empty_labels=0,
            invalid_annotations=invalid_annotations, quarantine_groups=sorted(quarantine_groups),
            quarantined_images=len(excluded), quarantined_instances=sum(len(r["instances"]) for r in excluded),
            excluded_image_ids=[r["image_id"] for r in excluded],
            overflow_pixels_min=min(positive_overflows, default=0), overflow_pixels_max=max(positive_overflows, default=0),
            invalid_annotation_handling="Unchanged original files retained; whole image and same-second acquisition group excluded from training/development/ROI manifests. No clipping or label removal.",
            exact_image_duplicates_cross_new_split=len(shared_hashes),
            official_cross_split_same_second_groups=sum(len(v)>1 for v in timestamp_splits.values()),
            verified_git_blob_files=len(file_receipts), verified_crc_files=len(file_receipts),
            warnings=["Class names intentionally remain numeric: original release contains no class-name mapping file.",
                      "Calendar-day separation is verified; physical fruit independence across days is not established.",
                      "The dev split is not an independent orchard or robot deployment test.",
                      "All source images have defect boxes; no normal fruit class or commercial grade is defined.",
                      "Source images are indoor harvested apples on black backgrounds, not orchard images."],
            completed_utc=datetime.now(timezone.utc).isoformat())
        dump(output / "manifest.json", {"schema":"ssda_date_manifest_v2", "report":"audit.json", "samples":rows})
        attribute_records = []
        for row in rows:
            if row["partition"] != "train":
                continue
            height, width = row["native_hw"]
            tokens = row["capture_filename_tokens"]
            for instance_index, box in enumerate(row["instances"]):
                cx, cy, w, h = box["xywh_normalized"]
                attribute_records.append({"image_path":row["rgb_path"],
                    "bbox_xyxy":[(cx-w/2)*width,(cy-h/2)*height,(cx+w/2)*width,(cy+h/2)*height],
                    "label":box["class_id"], "source_image_id":row["image_id"],
                    "source_instance_index":instance_index,
                    "group_id":tokens["date"]+"_"+tokens["time"], "group_kind":"filename_same_second_proxy",
                    "split":"train", "image_sha256":row["rgb_sha256"], "annotation_sha256":row["label_sha256"]})
        dump(output / "attribute_train_manifest.json", {
            "schema":"attribute_region_training_v1", "class_names":["ssda_class_0","ssda_class_1"],
            "label_semantics":"annotated defect-region type; NOT whole-fruit quality or healthy-vs-defective",
            "split_policy":{"train_dates":sorted(TRAIN_DATES), "independent":False,
                            "fruit_IDs":"unknown", "validation_records_included":False,
                            "quarantined_same_second_groups":sorted(quarantine_groups)},
            "bbox_definition":"float pixel xyxy, from unchanged normalized YOLO source boxes",
            "records":attribute_records})
        dump(output / "file_sha256.json", file_receipts)
        dump(output / "audit.json", report)
        attribution = ("Source: Nguyen, Bui Ngoc Han. Surface Subtle Defects Apple (SSDA), v1.0.0 (2024).\n"
                       "DOI: https://doi.org/10.5281/zenodo.10957715\n"
                       "License: Creative Commons Attribution 4.0 International (CC BY 4.0).\n"
                       "License terms: https://creativecommons.org/licenses/by/4.0/\n"
                       "License verified in original Zenodo record API metadata.\n"
                       "Original image/label bytes are unchanged. Changes: new acquisition-date train/dev lists,\n"
                       "exclusion of the entire same-second group containing an out-of-bounds annotation,\n"
                       "machine-readable audit and generic numeric class names. No maturity, normal or grade labels added.\n")
        (output / "ATTRIBUTION.txt").write_text(attribution, encoding="utf-8")
        return report
    except BaseException as error:
        report.update(phase="failed", error=str(error))
        dump(output / "audit.json", report)
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--archive", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--metadata", type=Path, default=Path(__file__).with_name("source_ssda.json"))
    a = p.parse_args()
    result = prepare(a.archive, a.output, a.metadata)
    print(json.dumps({k:result[k] for k in ("phase","images","partitions","image_dimensions","class_instance_counts","total_instances","verified_git_blob_files")},ensure_ascii=False))


if __name__ == "__main__":
    main()
