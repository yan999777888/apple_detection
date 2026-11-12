"""Read-only train/val label and split audit. Never reads the sealed test set.

Uses only Python's standard library. Images, masks, weights and dataset files
are never modified; writes one JSON report to an explicitly supplied path.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path: Path):
    with path.open("r", encoding="utf-8-sig") as stream:
        return json.load(stream)


def records(mapping):
    return mapping.values() if isinstance(mapping, dict) else mapping


def grouping(samples: list[dict], use_capture: bool) -> dict:
    """Images linked through any shared fruit, optionally through capture IDs."""
    parent = list(range(len(samples)))

    def root(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left, right):
        parent[root(right)] = root(left)

    owner = {}
    for index, sample in enumerate(samples):
        tokens = set()
        for instance in sample["instances"]:
            if instance.get("apple_identity"):
                tokens.add(("fruit", instance["apple_identity"]))
            if use_capture and instance.get("capture_group"):
                tokens.add(("capture", instance["capture_group"]))
        for token in tokens:
            if token in owner:
                union(index, owner[token])
            else:
                owner[token] = index
    components = defaultdict(list)
    for index in range(len(samples)):
        components[root(index)].append(index)
    rows = []
    for indexes in components.values():
        members = [samples[index] for index in indexes]
        identities = {i["apple_identity"] for s in members for i in s["instances"]
                      if i.get("apple_identity")}
        captures = {i["capture_group"] for s in members for i in s["instances"]
                    if i.get("capture_group")}
        rows.append({
            "images": len(members),
            "instances": sum(len(s["instances"]) for s in members),
            "unique_fruits": len(identities),
            "unique_captures_with_GT": len(captures),
            "original_split_counts": dict(Counter(s["split"] for s in members)),
            "image_keys": sorted(f'{s["split"]}/{s["image_id"]}' for s in members),
            "fruit_identities": sorted(identities),
        })
    rows.sort(key=lambda x: (-x["images"], x["image_keys"][0]))
    return {
        "edges": "shared fruit identity" + (" or capture_group" if use_capture else ""),
        "component_count": len(rows),
        "largest_component_images": rows[0]["images"] if rows else 0,
        "largest_component_fraction": rows[0]["images"] / len(samples) if samples else 0,
        "components_crossing_existing_splits": sum(len(r["original_split_counts"]) > 1 for r in rows),
        "components": rows,
        "caveat": "Diagnostic only: no split created. Empty-GT images lack identity/capture metadata in these manifests and remain unlinked. Orchard/day/sequence groups still require source metadata.",
    }


def audit(project: Path) -> dict:
    paths = {split: project / "cache" / "datasets" / "local_apple_v1" / f"{split}_manifest.json"
             for split in ("train", "val")}
    manifests = {split: load(path) for split, path in paths.items()}
    source_paths = [paths["train"], paths["val"]]
    split_rows = {}
    identity_sets = {}
    capture_sets = {}
    image_hashes = {}
    all_samples = []
    for split, manifest in manifests.items():
        samples = manifest["samples"]
        if manifest.get("split") != split or any(s.get("split") != split for s in samples):
            raise ValueError(f"Manifest split mismatch: {paths[split]}")
        instances = [instance for sample in samples for instance in sample["instances"]]
        identity_sets[split] = {i["apple_identity"] for i in instances if i.get("apple_identity")}
        capture_sets[split] = {i["capture_group"] for i in instances if i.get("capture_group")}
        image_hashes[split] = {s["rgb_sha256"] for s in samples if s.get("rgb_sha256")}
        split_rows[split] = {
            "images": len(samples), "instances": len(instances),
            "empty_images": sum(not s["instances"] for s in samples),
            "classes": manifest.get("classes"),
            "unique_fruit_identities": len(identity_sets[split]),
            "unique_capture_groups_with_GT": len(capture_sets[split]),
            "instances_missing_fruit_identity": sum(not i.get("apple_identity") for i in instances),
            "visibility_strata": dict(Counter(i.get("visibility_stratum", "missing") for i in instances)),
            "instance_field_coverage": dict(sorted(Counter(k for i in instances for k in i).items())),
            "sample_field_coverage": dict(sorted(Counter(k for s in samples for k in s).items())),
            "manifest_statistics_match": all(manifest.get("statistics", {}).get(k) == v for k, v in {
                "images": len(samples), "instances": len(instances),
                "empty_images": sum(not s["instances"] for s in samples)}.items()),
        }
        all_samples.extend(samples)
    annotation_path = Path(manifests["train"]["source"]["annotation_paths"]["visible"])
    if "test" in {part.lower() for part in annotation_path.parts}:
        raise ValueError("Refusing a test annotation path")
    annotations = load(annotation_path)
    source_paths.append(annotation_path)
    region_fields = Counter()
    file_fields = Counter()
    region_count = 0
    for image in records(annotations):
        file_fields.update(image.get("file_attributes", {}).keys())
        for region in records(image.get("regions", {})):
            region_count += 1
            region_fields.update(region.get("region_attributes", {}).keys())
    attribute_pattern = re.compile(r"matur|ripe|quality|defect|grade|bruise|rot|sugar|brix|firmness|成熟|品质|缺陷|糖度|硬度", re.I)
    candidate_fields = sorted(k for k in set(region_fields) | set(file_fields)
                              if attribute_pattern.search(k))
    sources = [{"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha256(path)}
               for path in source_paths]
    return {
        "schema": "apple_data_readiness_audit_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "sources": sources,
        "test_opened": False,
        "dataset_modified": False,
        "read_scope": "train/val manifests and original TRAIN visible polygon JSON only; no image/mask bytes or test data read",
        "stored_RGB_hashes_verified_against_images": False,
        "splits": split_rows,
        "cross_split": {
            "shared_fruit_identities": sorted(identity_sets["train"] & identity_sets["val"]),
            "shared_fruit_count": len(identity_sets["train"] & identity_sets["val"]),
            "shared_capture_count_with_GT": len(capture_sets["train"] & capture_sets["val"]),
            "shared_stored_RGB_sha256_count": len(image_hashes["train"] & image_hashes["val"]),
        },
        "original_train_visible_annotations": {
            "images": len(annotations), "polygon_regions": region_count,
            "region_attribute_field_coverage": dict(region_fields),
            "file_attribute_field_coverage": dict(file_fields),
            "hash_matches_manifest": sources[-1]["sha256"] == manifests["train"]["source"]["annotation_sha256"]["visible"],
            "maturity_quality_candidate_fields": candidate_fields,
            "label_conclusion": "No maturity or visible-quality truth is present in inspected TRAIN attributes; only identity labels found." if set(region_fields) == {"apple_ID"} and not file_fields else "Inspect fields and label definitions manually; field names do not establish valid truth.",
        },
        "grouping_fruit_only": grouping(all_samples, False),
        "grouping_fruit_and_capture": grouping(all_samples, True),
        "recommendations": [
            "Keep historical validation scores labeled development; shared fruit identities prevent an independent-fruit generalization claim.",
            "Create a separate versioned split using fruit/sequence/orchard/day groups; retain whole connected image components together. Do not randomly repartition adjacent frames or crops.",
            "Do not evaluate old apple-trained weights as unbiased on a newly reassigned holdout that they already saw; use fresh eligible initialization or independently collected unseen data.",
            "Collect cultivar, orchard, tree, date, fruit-ID, sequence-ID, timestamps and camera intrinsics; bind all views of a fruit to one split.",
            "Specify maturity ground truth from an agreed cultivar-specific measurement/label protocol; visible quality labels require reviewed defect type/extent and unknown/unassessable states. Color alone is not a maturity truth label.",
            "Keep public segmentation supervision separate from missing attribute supervision; unknown attributes must be masked in loss and never converted to healthy/ripe labels.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    project = args.project_root.resolve()
    output = args.output.resolve()
    if output == project or project in output.parents:
        parser.error("Output must be outside the audited project")
    if output.suffix.lower() != ".json":
        parser.error("Output must be a JSON file")
    if output.exists():
        parser.error("Output exists; choose a new report path to preserve evidence")
    result = audit(project)
    result["audit_script"] = {"path": str(Path(__file__).resolve()), "sha256": sha256(Path(__file__))}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "shared_fruit_count": result["cross_split"]["shared_fruit_count"],
                      "train": {k: result["splits"]["train"][k] for k in ("images", "instances", "empty_images")},
                      "val": {k: result["splits"]["val"][k] for k in ("images", "instances", "empty_images")},
                      "original_train_fields": result["original_train_visible_annotations"]["region_attribute_field_coverage"],
                      "grouping_fruit_only": {k: result["grouping_fruit_only"][k] for k in ("component_count", "largest_component_images", "components_crossing_existing_splits")},
                      "grouping_fruit_and_capture": {k: result["grouping_fruit_and_capture"][k] for k in ("component_count", "largest_component_images", "components_crossing_existing_splits")}}, ensure_ascii=False))


if __name__ == "__main__":
    main()
