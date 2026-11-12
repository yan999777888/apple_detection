"""Prepare a documented attribute-crop development baseline from Mendeley v4.

Reads only original-resolution TRAIN/VAL annotation and images. TEST filenames
are used solely to exclude their parent families. No TEST image/JSON is read by
this script. Prior root inspection read one TEST JSON schema example; therefore
this project does not claim the test annotations were never opened.
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
import sys
from zipfile import ZipFile

sys.dont_write_bytecode = True
ROOT_IN_ZIP = "dataset-20260508"
EXPECTED_SHA = "049591afd4fc3529aedbd31ef9119f5ec0601ebcb8cbbd4a25b7e883fde147ce"
CLASS_NAMES = ["immature apple", "semi-mature apple", "mature apple"]
AUGMENTATION = re.compile(r"_(?:brightness|noise)$", re.IGNORECASE)


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")


def family(filename):
    stem = PurePosixPath(filename).stem
    while AUGMENTATION.search(stem):
        stem = AUGMENTATION.sub("", stem)
    return stem.casefold()


def is_augmentation(filename):
    return bool(AUGMENTATION.search(PurePosixPath(filename).stem))


def safe_name(name):
    p = PurePosixPath(name)
    if p.is_absolute() or ".." in p.parts or "\\" in name or ":" in name:
        raise ValueError(f"Unsafe archive member: {name}")
    return p


def inspect_region(region, width, height):
    """Read original points; README defines polyline as a closed polygon.

    Explicit endpoint duplication is not required. Image edge coordinates equal
    to width/height are valid under the recorded half-open crop-edge convention.
    No vertex, endpoint, label, or original annotation is repaired or changed.
    """
    attrs, shape = region.get("region_attributes", {}), region.get("shape_attributes", {})
    label = attrs.get("name")
    reasons = []
    if label not in CLASS_NAMES:
        reasons.append("unknown_original_label")
    kind = shape.get("name")
    if kind not in ("polyline", "polygon"):
        reasons.append("unsupported_shape")
    x, y = shape.get("all_points_x", []), shape.get("all_points_y", [])
    numeric = (isinstance(x, list) and isinstance(y, list) and len(x)==len(y) and len(x)>=3
               and all(isinstance(v, (int,float)) and not isinstance(v,bool) and math.isfinite(v) for v in x+y))
    if not numeric:
        return None, sorted(set(reasons+["invalid_vertices"])), {"label_original": label, "shape_name": kind}
    closed = x[0]==x[-1] and y[0]==y[-1]
    gap = math.hypot(x[0]-x[-1], y[0]-y[-1])
    pts = list(zip(x,y))
    if len(set(pts)) < 3:
        reasons.append("less_than_three_distinct_vertices")
    if any(not 0 <= xx <= width or not 0 <= yy <= height for xx,yy in pts):
        reasons.append("vertex_outside_native_image")
    area = abs(sum(x[i]*y[(i+1)%len(x)]-x[(i+1)%len(x)]*y[i] for i in range(len(x))))*.5
    if area <= 0:
        reasons.append("zero_polygon_area")
    info = {"label_original": label, "shape_name": kind, "explicitly_closed": closed,
            "endpoint_gap_px": gap, "polygon_area_px2": area,
            "original_attributes": attrs, "vertex_count": len(x),
            "closure_interpretation":"explicit" if closed else "implicit_per_source_readme",
            "vertex_on_right_or_bottom_edge":any(xx == width or yy == height for xx,yy in pts)}
    # Preserve original extrema as half-open continuous image-edge coordinates.
    # Training may floor the lower and ceil the upper crop edges, but cannot
    # alter source annotation vertices or manufacture additional labels.
    box = [min(x), min(y), max(x), max(y)]
    if box[2] <= box[0] or box[3] <= box[1]:
        reasons.append("empty_box")
    if reasons:
        return None, sorted(set(reasons)), info
    return {"bbox_xyxy": box, "label": CLASS_NAMES.index(label), **info}, [], info


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--archive", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--metadata", type=Path, default=Path(__file__).parent/"sources/mendeley_v4_metadata.json")
    args = ap.parse_args()
    archive, output = args.archive.resolve(), args.output.resolve()
    if output.exists():
        ap.error("Output exists; choose a new directory to preserve previous evidence")
    digest = sha_file(archive)
    if digest != EXPECTED_SHA:
        raise ValueError("Archive SHA256 differs from official version 4 file metadata")
    metadata = json.loads(args.metadata.read_text(encoding="utf-8-sig"))
    if (metadata.get("id"), metadata.get("version"), metadata.get("licence",{}).get("short_name")) != ("gfcmdbvw65",4,"CC BY 4.0"):
        raise ValueError("Wrong dataset version/license metadata")
    from PIL import Image
    report = {"schema":"mendeley_documented_preparation_v2", "created_utc":datetime.now(timezone.utc).isoformat(),
        "archive":str(archive), "archive_sha256":digest, "archive_bytes":archive.stat().st_size,
        "preparation_source_sha256":sha_file(__file__), "metadata_sha256":sha_file(args.metadata),
        "test_content_read_by_this_script":False, "test_used_for_training_or_evaluation":False,
        "prior_test_access":"Root previously inspected one TEST annotation schema example; not used to train/evaluate or select thresholds",
        "class_names":CLASS_NAMES, "label_semantics":"Author-supplied visible color developmental stages; not measured physiological ripeness or commercial grade",
        "mask_semantics":"Not asserted visible V ground truth; author describes occluded boundary annotation",
        "bbox_semantics":"Unchanged original coordinate extrema interpreted as half-open continuous image-edge crop bounds; [0,width] and [0,height] edge coordinates allowed; no ROI prediction and no mask claim",
        "closure_policy":"README states polyline actually represents closed polygon; both repeated-endpoint and implicit closure accepted, recorded separately; no vertices added or changed",
        "split_policy":{"source":"published original-resolution train/val", "remove_precomputed_augmentations":True,
            "test_families_excluded_by_filename_only":True, "train_val_crossing_families":"excluded from both sets",
            "cross_split_exact_image_hashes":"excluded from both sets", "independent_fruit_or_scene_verified":False,
            "role":"engineering/development only", "limitation":"Filename families and exact image hashes cannot establish fruit/tree/scene independence; no EXIF or fruit IDs supplied"},
        "license":metadata["licence"], "source_doi":metadata["doi"], "counts":{}, "images":[], "excluded":[]}
    with ZipFile(archive) as z:
        names = z.namelist()
        if len(set(names)) != len(names):
            raise ValueError("Duplicate archive member names")
        for name in names:
            safe_name(name)
        readme_raw = z.read(f"{ROOT_IN_ZIP}/ReadME.txt")
        readme_text = readme_raw.decode("utf-8-sig")
        if "Actually represents closed polygon" not in readme_text:
            raise ValueError("Source README no longer establishes polygon semantics")
        report["source_readme_sha256"] = hashlib.sha256(readme_raw).hexdigest()
        originals = {split: {PurePosixPath(n).name:n for n in names if PurePosixPath(n).parent.as_posix()==f"{ROOT_IN_ZIP}/{split}" and n.lower().endswith(".jpg")} for split in ("train","val","test")}
        test_families = {family(n) for n in originals["test"]}
        all_split_families = {s:{family(n) for n in originals[s]} for s in ("train","val")}
        crossing = all_split_families["train"] & all_split_families["val"]
        report["test_filename_count"] = len(originals["test"])
        report["test_filename_family_count"] = len(test_families)
        report["train_val_crossing_families"] = sorted(crossing)
        records, annotations = {}, {}
        for split in ("train","val"):
            raw = z.read(f"{ROOT_IN_ZIP}/{split}.json")
            annotations[split] = raw
            data = json.loads(raw.decode("utf-8-sig"))
            records[split] = list(data.values())
            filenames = [d.get("filename") for d in records[split]]
            if len(set(filenames)) != len(filenames):
                raise ValueError(f"Duplicate annotation filenames in {split}")
            if set(filenames) != set(originals[split]):
                raise ValueError(f"Image/annotation filename mismatch in {split}")
        candidates = []
        for split in ("train","val"):
            stats = Counter(annotation_images=len(records[split]))
            labels, attributes, shapes, inspected_closures = Counter(), Counter(), Counter(), Counter()
            for record in records[split]:
                filename = record["filename"]
                if PurePosixPath(filename).name != filename:
                    raise ValueError("Annotation filename is not a basename")
                regions = record["regions"]
                if isinstance(regions,dict): regions=list(regions.values())
                stats["annotation_regions"] += len(regions)
                for region in regions:
                    labels[str(region.get("region_attributes",{}).get("name"))]+=1
                    attributes.update(region.get("region_attributes",{}).keys())
                    shapes[str(region.get("shape_attributes",{}).get("name"))]+=1
                reason = ("precomputed_augmentation" if is_augmentation(filename) else
                          "family_touches_test" if family(filename) in test_families else
                          "family_crosses_train_val" if family(filename) in crossing else None)
                if reason:
                    stats["excluded_"+reason]+=1
                    report["excluded"].append({"split":split,"filename":filename,"group_id":family(filename),"reason":reason})
                    continue
                raw = z.read(originals[split][filename])
                image_sha = hashlib.sha256(raw).hexdigest()
                with Image.open(io.BytesIO(raw)) as image:
                    image.load()
                    width,height=image.size
                    mode=image.mode
                if record.get("size") != len(raw):
                    stats["annotation_byte_size_mismatch"]+=1
                valid=[]
                for index,region in enumerate(regions):
                    parsed,reasons,info=inspect_region(region,width,height)
                    if "closure_interpretation" in info:
                        inspected_closures[info["closure_interpretation"]]+=1
                    if info.get("vertex_on_right_or_bottom_edge"):
                        stats["regions_with_right_or_bottom_edge_vertices"]+=1
                    if reasons:
                        stats["rejected_regions"]+=1
                        for reason in reasons:stats["region_reason_"+reason]+=1
                        report["excluded"].append({"split":split,"filename":filename,"region_index":index,"reasons":reasons,**info})
                    else:valid.append({"region_index":index,**parsed})
                if not valid:
                    stats["images_without_valid_regions"]+=1
                candidates.append({"split":split,"filename":filename,"group_id":family(filename),"source_member":originals[split][filename],
                                   "width":width,"height":height,"mode":mode,"image_sha256":image_sha,"regions":valid})
            report["counts"][split]={**dict(stats),"original_label_counts":dict(labels),"attribute_field_counts":dict(attributes),"shape_counts":dict(shapes),"inspected_original_region_closure_counts":dict(inspected_closures)}
        hashes=defaultdict(set)
        for row in candidates:hashes[row["image_sha256"]].add(row["split"])
        bad_hashes={h for h,splits in hashes.items() if len(splits)>1}
        report["cross_split_exact_image_hashes"]=sorted(bad_hashes)
        output.mkdir(parents=True,exist_ok=False)
        (output/"annotations").mkdir()
        for split,raw in annotations.items():(output/"annotations"/(split+".json")).write_bytes(raw)
        (output/"ReadME.txt").write_bytes(readme_raw)
        manifests={"train":[],"val":[]}
        for row in candidates:
            if row["image_sha256"] in bad_hashes:
                report["excluded"].append({"split":row["split"],"filename":row["filename"],"reason":"cross_split_exact_image_hash"})
                report["counts"][row["split"]]["excluded_cross_split_exact_image_hash"]=report["counts"][row["split"]].get("excluded_cross_split_exact_image_hash",0)+1
                continue
            if not row["regions"]:continue
            target=output/"images"/row["split"]/row["filename"]
            target.parent.mkdir(parents=True,exist_ok=True)
            raw=z.read(row["source_member"])
            target.write_bytes(raw)
            if sha_file(target)!=row["image_sha256"]:raise RuntimeError("Extracted image hash mismatch")
            report["images"].append({k:v for k,v in row.items() if k!="regions"})
            for region in row["regions"]:
                manifests[row["split"]].append({"sample_id":f"{row['split']}/{row['filename']}#{region['region_index']}",
                    "image_path":str(target.resolve()),"bbox_xyxy":region["bbox_xyxy"],"label":region["label"],
                    "label_original":region["label_original"],"source_image_id":row["filename"],
                    "group_id":"mendeley_gfcmdbvw65:"+row["group_id"],"split":row["split"],
                    "source_region_index":region["region_index"],"image_sha256":row["image_sha256"],
                    "native_hw":[row["height"],row["width"]],"polygon_explicitly_closed":region["explicitly_closed"],
                    "annotation_sha256":hashlib.sha256(annotations[row["split"]]).hexdigest(),
                    "closure_interpretation":region["closure_interpretation"]})
        for split,rows in manifests.items():
            report["counts"][split].update(accepted_instances=len(rows),accepted_images=len({r["image_path"] for r in rows}),
                accepted_class_counts=dict(Counter(r["label_original"] for r in rows)),
                accepted_groups=len({r["group_id"] for r in rows}),
                accepted_closure_counts=dict(Counter(r["closure_interpretation"] for r in rows)))
            dump(output/(split+"_manifest.json"),{"schema":"apple_attribute_crop_manifest_v1","split":split,"role":"engineering_training" if split=="train" else "development_not_independent",
                "class_names":CLASS_NAMES,"label_semantics":report["label_semantics"],"split_policy":report["split_policy"],
                "source_archive_sha256":digest,"samples":rows})
    if sha_file(archive)!=digest:raise RuntimeError("Source archive changed")
    report["source_archive_unchanged"]=True
    dump(output/"data_audit.json",report)
    dump(output/"ATTRIBUTION.json",{"authors":["Dandan Wang","Bo Wang"],"title":metadata["name"],"version":4,"doi":metadata["doi"],
        "source_url":"https://data.mendeley.com/datasets/gfcmdbvw65/4","license":metadata["licence"],
        "modifications":"Selected original-resolution train/val; removed precomputed augmentations, conflicting filename families and cross-split exact duplicate images; excluded invalid coordinates or degenerate shapes; retained implicit polygon closure per source README and derived bbox extrema without changing vertices or original labels."})
    print(json.dumps({"output":str(output),"counts":report["counts"],"crossing_families":len(crossing),"test_families":len(test_families),"cross_hashes":len(bad_hashes)},ensure_ascii=False))


if __name__ == "__main__":
    main()
