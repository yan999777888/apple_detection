"""Inventory existing apple research data without opening raw test data or GPU."""
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import re

ROOT = Path(r"F:\研究生毕业设计\模块一定位补偿以及预测")
OUT = Path(__file__).parent / "local_inventory.json"


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def provenance(path):
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest(path)}


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def main():
    vision = ROOT / "apple_vision"
    manifest_path = vision / "reports/local_v2/manifest.json"
    prepared = vision / "data/prepared/apple_local_v2"
    # Historical mixed report necessarily contains all splits. Immediately drop
    # sealed rows; never open any image/label path referenced by a test record.
    rows = [r for r in read_json(manifest_path) if r.get("split") in {"train", "val"}]
    sources = []
    for source in sorted({r["source"] for r in rows}):
        selected = [r for r in rows if r["source"] == source]
        per_split = {}
        for split in ("train", "val"):
            these = [r for r in selected if r["split"] == split]
            originals = [r for r in these if r["augmentation"] == "none"]
            labels = Counter()
            shape_counts = Counter()
            label_files = []
            malformed = []
            for row in these:
                label_path = prepared / "labels" / split / (Path(row["file"]).stem + ".txt")
                if any(part.lower() == "test" for part in label_path.parts):
                    raise RuntimeError("Refusing test label path")
                for lineno, line in enumerate(label_path.read_text(encoding="utf-8-sig").splitlines(), 1):
                    tokens = line.split()
                    if not tokens:
                        continue
                    shape_counts[len(tokens)] += 1
                    labels[tokens[0]] += 1
                    if len(tokens) != 5:
                        malformed.append({"path": str(label_path), "line": lineno, "tokens": len(tokens)})
                label_files.append(provenance(label_path))
            per_split[split] = {
                "prepared_images": len(these), "original_images": len(originals),
                "original_instances": sum(r["instances"] for r in originals),
                "prepared_instances": sum(r["instances"] for r in these),
                "stored_groups": len({r["group"] for r in originals}),
                "group_examples": sorted({r["group"] for r in originals})[:5],
                "class_id_counts": dict(labels), "label_token_count_histogram": dict(shape_counts),
                "label_count_matches_manifest": sum(labels.values()) == sum(r["instances"] for r in these),
                "malformed_labels": malformed,
                "labels_aggregate_sha256": hashlib.sha256(json.dumps(label_files, sort_keys=True).encode()).hexdigest(),
                "source_image_examples": [r["original_file"] for r in originals[:3]],
            }
        grouping_train = {r["group"] for r in selected if r["split"] == "train"}
        grouping_val = {r["group"] for r in selected if r["split"] == "val"}
        originals_train = {r["original_file"] for r in selected if r["split"] == "train"}
        originals_val = {r["original_file"] for r in selected if r["split"] == "val"}
        field_counts = Counter(k for r in selected for k in r)
        attr_pattern = re.compile(r"matur|ripe|quality|defect|grade|bruise|brix|firmness|fruit.?id|tree.?id|orchard|cultivar", re.I)
        sources.append({
            "source": source, "splits": per_split,
            "train_val_stored_group_overlap": sorted(grouping_train & grouping_val),
            "train_val_original_file_overlap": sorted(originals_train & originals_val),
            "field_coverage": dict(sorted(field_counts.items())),
            "attribute_or_identity_candidate_fields": [k for k in field_counts if attr_pattern.search(k)],
        })
    weights = [
        (ROOT / "yolo26n.pt", "COCO detector; filename/provenance not loaded"),
        (vision / "models/yolo26s.pt", "COCO detector; filename/provenance not loaded"),
        (vision / "models/yolo26m.pt", "COCO detector; filename/provenance not loaded"),
        (vision / "models/apple_yolo26s_best.pt", "apple-trained checkpoint; historical data exposure must be checked"),
        (ROOT / "apple_detection/weights/local_rgbd/yolo26s-seg.pt", "COCO segmentation runtime candidate"),
        (vision / "local_amodal/cache/torch/hub/checkpoints/maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth", "COCO Mask R-CNN with ResNet50 backbone"),
        (vision / "local_amodal/cache/torch/hub/checkpoints/maskrcnn_resnet50_fpn_v2_coco-73cbd019.pth", "COCO Mask R-CNN v2 with ResNet50 backbone"),
        (vision / "local_amodal/weights/query_dual_m2f_22c4a2f/model.safetensors", "query pretrained source; inspect adjacent config and provenance before reuse"),
    ]
    weight_info = []
    for path, role in weights:
        if path.exists():
            item = provenance(path)
            item["role"] = role
            item["loaded_or_gpu_tested_this_audit"] = False
            official = re.search(r"-([0-9a-f]{8})\.pth$", path.name)
            if official:
                item["sha_matches_official_filename_prefix"] = item["sha256"].startswith(official.group(1))
            weight_info.append(item)
    readme_paths = [ROOT / "数据集组成.png", ROOT / "新方案.txt", manifest_path,
                   vision / "scripts/prepare_local_mix.py", prepared / "data.yaml"]
    result = {
        "schema": "local_apple_research_inventory_v2",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "sources": [provenance(p) for p in readme_paths],
        "raw_test_image_label_or_GT_opened": False,
        "mixed_historical_manifest_read": True,
        "audit_scope_caveat": "Historical report contains test metadata. This script discards test records before label access/counting. One legacy report test row was displayed earlier during manual inventory; no raw test files, inference or metrics were used.",
        "GPU_used": False, "F_drive_modified": False,
        "training_inputs": sources, "local_weights": weight_info,
        "image_table_transcribed": {
            "MAD": {"train": 66, "val": 12, "test_as_printed_only": 27},
            "MSU": {"train": 875, "val": 164, "test_as_printed_only": 193},
            "LOCAL_apple": {"train": 91, "val": 21, "test_as_printed_only": 28},
            "LOCAL_shuju": {"train": 886, "val": 109, "test_as_printed_only": 120},
            "meaning": "Original image counts, not label categories; image was inspected visually",
        },
        "metadata_interpretation": {
            "LOCAL_apple": "120-second EXIF tag306 capture blocks; original validation retained. EXIF DateTime is an engineering proxy and not independently verified capture/fruit identity.",
            "LOCAL_shuju": "Without timestamps, 30-index filename blocks plus five-index cross-split exclusion. Not confirmed scene/tree/fruit identity.",
            "MSU": "Timestamp and camera suffix available from filenames; 120-second blocks and 10-second exclusion are recorded in earlier preparation. Both views grouped.",
            "MAD": "Only official split-level group is stored here; tree-level split claim comes from historical source audit, explicit per-image tree IDs are absent in this manifest.",
            "attribute_labels": "No maturity or visible-quality categories in inspected prepared train/val labels. Five-token rows encode class and bounding box only.",
        },
        "safe_next_work": [
            "Use unchanged train-only originals to choose scenes for human maturity/visible-quality annotation; boxes support crop extraction but do not supply attribute truth.",
            "Bind every crop, augmented image and duplicate view to its original image and stored scene group. Do not randomly split fruit crops.",
            "Audit/annotate repeated fruit and scene identities across original train/val before claiming fruit-independent classification performance; stored groups are only proxies.",
            "Existing COCO Mask R-CNN files permit extracting a ResNet50 backbone offline after CPU key/shape validation; YOLO pretrained backbones are also local. New classification heads remain untrained until labels exist.",
            "Avoid apple-trained weights for independence claims unless training exposure is documented; any new holdout must not have been seen by initialization.",
        ],
        "new_scheme_document_assessment": "新方案.txt proposes model architectures, contains no class annotation/schema. Claims of an ROI 92% ceiling, guaranteed 96%, and first-use novelty are unsupported by this audit and must not be treated as evidence.",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(OUT), "counts": [{"source": s["source"], "train_originals": s["splits"]["train"]["original_images"], "val_originals": s["splits"]["val"]["original_images"], "train_crop_candidates": s["splits"]["train"]["original_instances"], "class_ids": sorted(set(s["splits"]["train"]["class_id_counts"]) | set(s["splits"]["val"]["class_id_counts"])), "group_overlap": len(s["train_val_stored_group_overlap"])} for s in sources], "weight_files": len(weight_info)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
