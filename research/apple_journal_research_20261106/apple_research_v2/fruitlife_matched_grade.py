"""Freeze a physical-fruit split, then append appearance-supervised training faces to Woo."""
import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

from fruitlife_apple_source import LABELS, OUT, ROOT, save, sha

SPLIT = OUT / "physical_id_split.json"
DEST = ROOT / "datasets/apple_matched_fruitlife_woo_v1"
WOO = ROOT / "datasets/apple_matched_quality_v1/manifest.json"
SALT = "FruitLife apple physical ID holdout v1"


def freeze(path, content):
    if path.exists() and json.loads(path.read_bytes()) != content:
        raise ValueError(f"Frozen data changed: {path}")
    if not path.exists():
        save(path, (json.dumps(content, ensure_ascii=False, indent=2) + "\n").encode())


def split():
    with (OUT / LABELS).open(encoding="utf-8-sig", newline="") as f:
        rows = [row for row in csv.DictReader(f) if row["class"] == "apple"]
    assignment = {}
    for storage in sorted({r["storage"] for r in rows}):
        ids = sorted({r["specimen_id"] for r in rows if r["storage"] == storage},
                     key=lambda i: hashlib.sha256((SALT + "|" + i).encode()).hexdigest())
        if len(ids) != 2:
            raise ValueError("Unexpected number of physical specimens in a storage condition")
        assignment.update({ids[0]: "train", ids[1]: "external_test"})
    freeze(SPLIT, dict(record="10.5281/zenodo.23589223", license="CC BY 4.0", id_salt=SALT,
           split="For each storage, sort SHA256(salt plus pipe plus specimen_id), first training, last external test.",
           assignment=assignment,
           source_labels="All author consensus stages retained, with original source stage and whether blind relabeled. No age-based labels.",
           new_model_predictions_seen=False,
           crop="Original author fruit mask bounding box expanded by 10% each side identical to existing localization context; no card/ruler normalization based on model predictions.",
           input_manifest_sha256=sha((OUT / LABELS).read_bytes()),
           first_pass="Train grades only with visually supported grade mapping explicitly identified as auxiliary. Report external all stages at fruit and image level. Never use external results for tuning or hyperparameter selection."))
    return assignment


def merge():
    assignment = split()
    samples = json.loads((OUT / "apple_manifest.json").read_bytes())["samples"]
    woo = json.loads(WOO.read_bytes())["samples"]
    if len({r["sha256"] for r in samples}) != len(samples):
        raise ValueError("Exact duplicate across auxiliary apple crops")
    afruit = json.loads((ROOT / "datasets/afruitdb_apples_v1/manifest.json").read_bytes())["samples"]
    hdf = json.loads((ROOT / "diagnostics/hdf_peripheral_v1_s20261106/frozen_split_manifest.json").read_bytes())["representatives"]
    old_hashes = {r["sha256"] for r in (afruit + woo + hdf)}
    train, external = [], []
    for r in samples:
        if not r["crop_method"].startswith("author shadow-free fruit mask;"):
            raise ValueError("The measurement scale/card must be removed in all frames")
        if sha((ROOT / r["path"]).read_bytes()) != r["sha256"]:
            raise ValueError("Unexpected crop bytes")
        if {r["source_sha256"], r["sha256"]} & old_hashes:
            raise ValueError("Found an exact overlap with an existing data source")
        stage = r["source_stage"]
        r = dict(r, split=assignment[r["specimen_id"]], condition_id="FruitLife_" + r["storage"],
                 source="FruitLife_v1.5_CC_BY_4.0", capture_key=r["time"], source_doi="10.5281/zenodo.23589223")
        if r["split"] == "external_test":
            external.append(r)
        elif stage in (0, 1):
            train.append(dict(r, grade=stage, label_basis=("Author stage 0, smooth glossy original-colour skin" if stage==0 else
                                    "Auxiliary author stage 1: gloss loss or browning with smooth round shape. Weak visible mild storage-change proxy.")))
    if any(x["group_id"] in {y["group_id"] for y in train} for x in external):
        raise ValueError("A physical specimen appears on both sides of the split")
    if len(train)!=30 or len(external)!=180 or len({r["group_id"] for r in external})!=3:
        raise ValueError("Unexpected number of samples or specimens")
    freeze(DEST / "manifest.json", dict(samples=woo + train, old_manifest_sha256=sha(WOO.read_bytes()),
           fruit_source_sha256=sha((OUT / "apple_manifest.json").read_bytes()), split_sha256=sha(SPLIT.read_bytes()),
           scope="Training only. Woo visible-face proxies plus FruitLife stages 0/1; the latter refer to fresh vs mild storage change on the same specimen.",
           photo_duplicates="SHA-256 sources/crops checked against AFruitDB, Woo crops, HDF representatives. Physical IDs separated by storage."))
    freeze(DEST / "external_test.json", dict(samples=external, split_sha256=sha(SPLIT.read_bytes()),
           stage_semantics={"0":"smooth and glossy, original colour", "1":"some gloss loss or browning, still round and smooth",
                            "2":"fine wrinkles, may be fully brown","3":"strong shrinkage or collapse","4":"mostly collapsed or deeply wrinkled"},
           stage_basis="Author FruitLife v1.5 src/docs species-specific apple visual rubric and consensus manifest. Stage 2 is retained as its own ordinal category.",
           scope="One previously unused physical fruit in each of three storage conditions; report all stages and per-physical-fruit counts. Whole fruit benchmark only; not a mechanical or insect injury dataset."))
    audit=dict(license="CC BY 4.0",doi="10.5281/zenodo.23589223", total_frames=len(samples),
               training_frames=len(train),external_frames=len(external),woo_frames=len(woo),source_hash_overlap=0,
               fruit_assignment=assignment,training_stage_hist=dict(sorted(Counter(str(r["source_stage"]) for r in train).items())),
               external_stage_hist=dict(sorted(Counter(str(r["source_stage"]) for r in external).items())),
               preserved_woo_sha256=sha(WOO.read_bytes()),manifest_sha256=sha((DEST/"manifest.json").read_bytes()),
               external_sha256=sha((DEST/"external_test.json").read_bytes()))
    freeze(DEST / "source_audit.json",audit)
    print(audit,flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("action",choices=["split","merge"])
    {"split":split,"merge":merge}[p.parse_args().action]()
