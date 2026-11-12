"""Convert audited Mendeley v4 author polygons to stage-aware YOLO segmentation.

Input polygons/labels retain author semantics. Dataset development splits are
not independently identified fruits, trees or scenes. No test content is read.
All outputs are new files under the requested F-drive research workspace.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys

sys.dont_write_bytecode = True
CLASSES = ["immature apple", "semi-mature apple", "mature apple"]


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def within(path, root):
    path, root = Path(path).resolve(), Path(root).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Path escapes research workspace: {path}")
    return path


def require(ok, message):
    if not ok:
        raise ValueError(message)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prepared", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--relocation-marker", type=Path, required=True)
    p.add_argument("--verify-ultralytics", action="store_true")
    args = p.parse_args()
    project = Path(__file__).resolve().parent
    workspace = project.parent
    require(workspace.drive.upper() == "F:", "F-drive workspace is required")
    prepared = within(args.prepared, project)
    output = within(args.output, project)
    marker = within(args.relocation_marker, workspace)
    require(marker.is_file(), "Relocation marker missing; preparation must wait")
    require(not output.exists(), "Output exists; refusing overwrite")
    source_audit_path = prepared/"data_audit.json"
    audit = json.loads(source_audit_path.read_text(encoding="utf-8-sig"))
    require(audit["schema"] == "mendeley_documented_preparation_v2", "Only documented v2 accepted")
    require(audit["source_archive_unchanged"] is True, "Source archive audit not verified")
    require(audit["license"]["short_name"] == "CC BY 4.0", "Unexpected source license")
    require(audit["class_names"] == CLASSES, "Class order mismatch")
    input_hashes = {str(marker):sha(marker), str(source_audit_path):sha(source_audit_path)}
    rows_by_split = {}
    source_docs = {}
    image_groups, image_hashes = {}, {}
    for split in ("train", "val"):
        manifest_path = prepared/(split+"_manifest.json")
        document = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
        require(document["class_names"] == CLASSES, "Manifest class order mismatch")
        require(document["split"] == split, "Unexpected manifest split")
        rows = document["samples"]
        require(rows and all(r["split"] == split for r in rows), "Only stated train/val rows permitted")
        annotation = prepared/"annotations"/(split+".json")
        annotation_sha = sha(annotation)
        input_hashes[str(manifest_path)] = sha(manifest_path)
        input_hashes[str(annotation)] = annotation_sha
        for row in rows:
            require(row["annotation_sha256"] == annotation_sha, "Annotation source changed")
        docs = json.loads(annotation.read_text(encoding="utf-8-sig"))
        source_docs[split] = {v["filename"]:v for v in docs.values()}
        require(len(source_docs[split]) == len(docs), "Duplicate source annotation filenames")
        rows_by_split[split] = rows
        image_groups[split] = {r["group_id"] for r in rows}
        image_hashes[split] = {r["image_sha256"] for r in rows}
    require(not image_groups["train"] & image_groups["val"], "Cross-split source filename families")
    require(not image_hashes["train"] & image_hashes["val"], "Cross-split exact image hashes")
    output.mkdir(parents=True, exist_ok=False)
    (output/"source").mkdir()
    for split in ("train", "val"):
        shutil.copyfile(prepared/"annotations"/(split+".json"), output/"source"/(split+".json"))
    for name in ("ReadME.txt", "ATTRIBUTION.json"):
        source = prepared/name
        if source.exists():
            shutil.copyfile(source, output/"source"/name)
            input_hashes[str(source)] = sha(source)
    from PIL import Image
    conversion = {"schema":"mendeley_stage_segmentation_v1", "created_utc":datetime.now(timezone.utc).isoformat(),
        "source_preparation":str(prepared), "source_preparation_audit_sha256":sha(source_audit_path),
        "converter_sha256":sha(__file__), "source_archive_sha256":audit["archive_sha256"],
        "class_names":CLASSES, "label_semantics":audit["label_semantics"],
        "mask_semantics":audit["mask_semantics"], "split_policy":dict(audit["split_policy"], independent_group_holdout=False),
        "source_license":audit["license"], "source_doi":audit["source_doi"],
        "coordinate_policy":"Original point order, repeated vertices and endpoints preserved; x/W,y/H normalized with 17 significant digits; implicit closure follows source README; no clipping, vertex repair or synthetic labels",
        "evaluation_role":"Published-source development validation only; no independent fruit/tree/scene generalization established",
        "test_content_read_by_converter":False, "prior_test_access":audit["prior_test_access"],
        "counts":{}, "images":[], "source_input_hashes":input_hashes}
    for split, rows in rows_by_split.items():
        grouped = defaultdict(list)
        for row in rows:
            grouped[row["source_image_id"]].append(row)
        image_dir, label_dir = output/"images"/split, output/"labels"/split
        image_dir.mkdir(parents=True); label_dir.mkdir(parents=True)
        count, closure = Counter(), Counter()
        for filename, objects in grouped.items():
            require(Path(filename).name == filename and filename.lower().endswith(".jpg"), "Unsafe image filename")
            image_source = prepared/"images"/split/filename
            require(sha(image_source) == objects[0]["image_sha256"], "Source image SHA256 mismatch")
            with Image.open(image_source) as im:
                width, height = im.size
                im.verify()
            regions = source_docs[split][filename]["regions"]
            if isinstance(regions, dict):
                regions = list(regions.values())
            require({r["source_region_index"] for r in objects} == set(range(len(regions))),
                    "A retained image has omitted original instances")
            lines, source_records = [], []
            for row in sorted(objects, key=lambda r:r["source_region_index"]):
                index = row["source_region_index"]
                region = regions[index]
                shape = region["shape_attributes"]
                label = region["region_attributes"]["name"]
                require(label == row["label_original"] == CLASSES[row["label"]], "Original class mismatch")
                require(row["native_hw"] == [height,width], "Native image dimensions mismatch")
                require(shape["name"] in ("polyline", "polygon"), "Unexpected region type")
                x, y = shape["all_points_x"], shape["all_points_y"]
                require(len(x) == len(y) and len(x) >= 3, "Invalid polygon point count")
                require(all(isinstance(v,(int,float)) and not isinstance(v,bool) and math.isfinite(v) for v in x+y),
                        "Nonfinite or nonnumeric vertex")
                require(all(0 <= xx <= width and 0 <= yy <= height for xx,yy in zip(x,y)), "Out-of-bounds source vertex")
                require(len(set(zip(x,y))) >= 3, "Degenerate polygon")
                coords = [v for xx,yy in zip(x,y) for v in (xx/width,yy/height)]
                tokens = [format(v,".17g") for v in coords]
                restored = [float(s) for s in tokens]
                require(all(0 <= v <= 1 for v in restored), "Normalized coordinate out of bounds")
                require(max(abs(a-b) for a,b in zip(restored,coords)) <= 1e-15, "Normalization round-trip failed")
                lines.append(str(row["label"])+" "+" ".join(tokens))
                explicit = x[0] == x[-1] and y[0] == y[-1]
                closure["explicit" if explicit else "implicit_per_source_readme"] += 1
                count[label] += 1
                source_records.append({"source_region_index":index,"label_id":row["label"],"label_original":label,
                    "source_shape_name":shape["name"],"source_vertices_xy":[list(pt) for pt in zip(x,y)],
                    "explicitly_closed":explicit, "source_sample_id":row["sample_id"]})
            image_target = image_dir/filename
            label_target = label_dir/(Path(filename).stem+".txt")
            shutil.copyfile(image_source,image_target)
            with label_target.open("x",encoding="utf-8",newline="\n") as stream:
                stream.write("\n".join(lines)+"\n")
            require(sha(image_target) == objects[0]["image_sha256"], "Copied image changed")
            conversion["images"].append({"split":split,"image_path":str(image_target),"label_path":str(label_target),
                "source_image_path":str(image_source),"source_image_id":filename,"group_id":objects[0]["group_id"],
                "image_sha256":sha(image_target),"label_sha256":sha(label_target),"native_hw":[height,width],
                "annotation_path":str(output/"source"/(split+".json")),"annotation_sha256":objects[0]["annotation_sha256"],
                "instances":source_records})
        conversion["counts"][split] = {"images":len(grouped),"instances":sum(count.values()),
            "class_counts":dict(count),"closure_counts":dict(closure)}
    # JSON quoting is valid YAML and preserves Unicode Windows paths.
    yaml_text = "path: "+json.dumps(str(output).replace("\\","/"),ensure_ascii=False)+"\ntrain: images/train\nval: images/val\nnames:\n"
    for i,label in enumerate(CLASSES):
        yaml_text += f"  {i}: "+json.dumps(label)+"\n"
    with (output/"data.yaml").open("x",encoding="utf-8") as stream:
        stream.write(yaml_text)
    if args.verify_ultralytics:
        runtime = output/"verification_runtime"
        for name in ("tmp","yolo","matplotlib","torch"):
            (runtime/name).mkdir(parents=True,exist_ok=True)
        os.environ.update(PYTHONDONTWRITEBYTECODE="1",CUDA_VISIBLE_DEVICES="",YOLO_AUTOINSTALL="False",
            YOLO_CONFIG_DIR=str(runtime/"yolo"),MPLCONFIGDIR=str(runtime/"matplotlib"),
            TORCH_HOME=str(runtime/"torch"),TMP=str(runtime/"tmp"),TEMP=str(runtime/"tmp"))
        import ultralytics
        from ultralytics.data.utils import verify_image_label
        verified = Counter(); messages = []
        for entry in conversion["images"]:
            result = verify_image_label((entry["image_path"],entry["label_path"],"",False,3,0,0,False))
            file, labels, shape, segments, keypoints, missing, found, empty, corrupt, msg = result
            require(file is not None and found == 1 and not (missing or empty or corrupt),
                    f"Ultralytics rejected image or label: {msg}")
            require(len(labels) == len(segments) == len(entry["instances"]),
                    f"Ultralytics changed instance count: {entry['image_path']} {msg}")
            require([int(v) for v in labels[:,0]] == [r["label_id"] for r in entry["instances"]],
                    "Loader altered label order or classes")
            require(sha(entry["image_path"]) == entry["image_sha256"], "Loader modified image bytes")
            verified[entry["split"]+"_images"] += 1
            verified[entry["split"]+"_instances"] += len(segments)
            if msg: messages.append(msg)
        conversion["loader_verification"] = {"ultralytics_version":ultralytics.__version__,
            "method":"ultralytics.data.utils.verify_image_label for every exported image and label",
            "counts":dict(verified),"messages":messages,"GPU_used":False,"instance_counts_preserved":True}
    else:
        conversion["loader_verification"] = {"executed":False}
    for path,digest in input_hashes.items():
        require(sha(path) == digest, f"Source input changed during conversion: {path}")
    conversion["source_inputs_unchanged"] = True
    conversion["data_yaml_sha256"] = sha(output/"data.yaml")
    save(output/"conversion_audit.json",conversion)
    save(output/"DATA_READY.json",{"data_yaml":str(output/"data.yaml"),"audit_sha256":sha(output/"conversion_audit.json"),
        "counts":conversion["counts"],"loader_verification":conversion["loader_verification"],
        "independent_group_holdout":False})
    print(json.dumps({"output":str(output),"counts":conversion["counts"],
        "loader_verification":conversion["loader_verification"]},ensure_ascii=False))


if __name__ == "__main__":
    main()
