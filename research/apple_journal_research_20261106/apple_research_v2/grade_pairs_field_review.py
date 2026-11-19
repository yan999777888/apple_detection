"""Tabulate original and candidate decisions on identical fruit after the field lock."""
import argparse
from pathlib import Path

from afruit_grade_train import read, atomic
from afruit_grade_model import ROOT, file_sha


OUT = ROOT / "experiments/grade_pairs_20261119"
REFERENCES = {
    "woo": ROOT / "experiments/grade_matched_review_v1",
    "native320": ROOT / "experiments/grade_error49_v1",
}


def by_key(rows, key):
    indexed = {r[key]: r for r in rows}
    if len(rows) != len(indexed):
        raise ValueError(f"Repeated field sample identifier {key}")
    return indexed


def compare(candidate_field, candidate_external):
    lock = read(OUT / "field_lock.json")
    policy = read(OUT / "protocol.json")
    if lock["protocol_sha"] != file_sha(OUT / "protocol.json"):
        raise ValueError("Registration changed since selection")
    sig = lock["sha256"]
    if read(candidate_field / "receipt.json")["weight_sha256"] != sig:
        raise ValueError("Unexpected AF checkpoint")
    if read(candidate_external / "results.json")["models"]["candidate"] != sig:
        raise ValueError("Unexpected external checkpoint")
    af = by_key(read(candidate_field / "paired_predictions.json"), "path")
    hdf = by_key([r for r in read(candidate_external / "predictions.json") if r["panel"] == "HDF"], "sha256")
    if len(af) != 110 or len(hdf) != 269:
        raise ValueError("Historical panel denominators changed")
    changes = {}
    for name, path in REFERENCES.items():
        if read(path / "field/receipt.json")["weight_sha256"] != policy["reference"][name]:
            raise ValueError(f"Unexpected AF {name} reference")
        if read(path / "external/results.json")["models"]["candidate"] != policy["reference"][name]:
            raise ValueError(f"Unexpected HDF {name} reference")
        prev = by_key(read(path / "field/paired_predictions.json"), "path")
        ext = by_key([r for r in read(path / "external/predictions.json") if r["panel"] == "HDF"], "sha256")
        if set(prev) != set(af) or set(ext) != set(hdf):
            raise ValueError("Fruit identities do not match")
        ac, hc = [], []
        for key, current in af.items():
            before = prev[key]
            if current["truth"] != before["truth"] or current["fruit_count"] != before["fruit_count"]:
                raise ValueError(f"Grade truth or automatic instance count changed for {key}")
            a, b = before["linked_grade"], current["linked_grade"]
            if a != b:
                ac.append(dict(path=key, truth=current["truth"], previous=a, current=b))
        for key, current in hdf.items():
            before = ext[key]
            if current["target"] != before["target"]:
                raise ValueError(f"HDF source label changed for {key}")
            if before["candidate"] != current["candidate"]:
                hc.append(dict(sha256=key, path=current["path"], target=current["target"],
                               previous=before["candidate"], current=current["candidate"]))
        changes[name] = dict(af=ac, hdf=hc)
    a = read(candidate_field / "regression.json")["auto"]["confusion"]
    totals = read(candidate_external / "results.json")["panels"]["HDF"]["candidate"]["totals"]
    h = {key.lower(): val for key, val in totals.items()}
    fresh, rot = [next(val for key, val in h.items() if word in key) for word in ("fresh", "rot")]
    if [sum(row) for row in a] != [56, 37, 17] or [sum(fresh), sum(rot)] != [69, 200]:
        raise ValueError("Field label counts changed")
    g = policy["field_gates"]
    checks = dict(af_mild=a[1][1] >= g["AF_mild_correct_min"],
                  af_good=a[0][0] >= g["AF_good_correct_min"],
                  af_severe_release=a[2][0] <= g["AF_severe_to_good_max"],
                  af_severe_correct=a[2][2] >= g["AF_severe_correct_min"],
                  hdf_fresh_to_severe=fresh[2] <= g["HDF_source_fresh_to_severe_max"],
                  hdf_rot_to_good=rot[0] <= g["HDF_source_rot_to_good_max"])
    atomic(OUT / "field_review.json", dict(weights_sha256=sig, af=a, hdf=totals,
           historical_accuracy_checks=checks, changes=changes,
           scope="AF visible fruit grades; HDF binary source labels only. Runtime and independent panel reported separately."))
    print("Historical checks", checks, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("field", type=Path)
    parser.add_argument("external", type=Path)
    args = parser.parse_args()
    compare(args.field, args.external)
