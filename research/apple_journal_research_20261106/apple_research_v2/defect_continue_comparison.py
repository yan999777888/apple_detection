"""Compare frozen detectors on identical GTs and quantify candidate and score limitations."""

import argparse
from pathlib import Path

import numpy as np

from audit_defect_target9485 import PHASE, ROOT, oracle
from class0_patch_refinement import atomic_json, read, sha
from compare_defect_target9485 import indexed, inspect, load_candidates, delta

REFERENCE = ROOT / "experiments/defect9485_set_deploy/surface_only_predictions.json"


def report(fixed_path, low_path, speed_path, reference):
    canonical = indexed(read(PHASE))
    assert len(canonical) == 113 and sum(len(r["GT"]) for r in canonical.values()) == 230
    fixed = load_candidates(fixed_path, canonical)
    low = load_candidates(low_path, canonical)
    base = load_candidates(reference, canonical)
    stats, hits, fp_types, _ = inspect(fixed, canonical)
    _, ref_hits, _, _ = inspect(base, canonical)
    envelope = inspect(low, canonical)[3]["classwise_overall"]
    coverage = [0, 0]
    agnostic = 0
    for sid, row in canonical.items():
        for i in oracle(low[sid], row["GT"]):
            coverage[row["GT"][i]["class"]] += 1
        agnostic += len(oracle(low[sid], row["GT"], same_class=False))
    cases = read(ROOT/"experiments/defect_continue_teacher/error_ledger.json")["original_class0_misses"]
    detail = {x["code"]: (x["image"], x["index"]) in hits for x in cases}
    assert len(detail) == 29
    s = read(speed_path)
    timing = s.get("timing", s)
    assert timing["n"] == 600 and np.isclose(timing["serial_fps"], 1000 / timing["mean_ms"])
    paths = [PHASE, fixed_path, low_path, speed_path, reference]
    return dict(fixed=stats, fp_geometry=fp_types, fixed_delta_vs_reference=delta(hits, ref_hits, canonical),
                recovered_of_29_initial_scratch_misses=detail,
                oracle_low_score_coverage=dict(per_class=coverage, ignoring_class=agnostic),
                retrospective_classwise_threshold_envelope=envelope, dual_task_speed=timing,
                input_sha256={str(p.relative_to(ROOT)):sha(p) for p in paths})


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for arg in ("fixed", "candidates", "speed", "output"):
        p.add_argument("--"+arg, required=True, type=Path)
    p.add_argument("--reference", type=Path, default=REFERENCE)
    a = p.parse_args()
    result = report(a.fixed.resolve(), a.candidates.resolve(), a.speed.resolve(), a.reference.resolve())
    dst = a.output.resolve()
    if not dst.is_relative_to(ROOT/"experiments") or str(dst.relative_to(ROOT)) in result["input_sha256"]:
        raise ValueError("Output must be a separate project experiment file")
    atomic_json(dst, result)
    print(result["fixed"]["total"], result["retrospective_classwise_threshold_envelope"], flush=True)
