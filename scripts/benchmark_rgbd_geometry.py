"""CPU synthetic geometry check or strict rectified-RGBD manifest evaluation.

No model inference, camera motion, data download, or training is performed.
Real inputs without a complete calibrated K are refused rather than guessing a
crop principal point. Synthetic results are not real fruit localization results.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from apple_local_rgbd.boundaries import candidate_outer_edges
from apple_local_rgbd.geometry import SphereConfig, fit_sphere, points_from_depth, rays_from_pixels
from apple_local_rgbd.synthetic import render_sphere, tangent_rays


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_array(path):
    path = Path(path)
    if path.suffix.lower() == ".npy":
        return np.load(path, allow_pickle=False)
    with Image.open(path) as image:
        array = np.array(image)
    if array.ndim != 2:
        raise ValueError("Each mask/depth file must be a scalar H,W array")
    return array


def read_real_record(record):
    required = ("image_id", "depth_path", "visible_mask_path", "intrinsics", "intrinsics_provenance",
                "depth_scale_m", "depth_value_kind", "alignment_provenance")
    if any(key not in record for key in required):
        raise ValueError("Record needs paired/aligned Z-depth, visible mask, scale and complete calibrated K provenance")
    if record["depth_value_kind"] != "z" or record.get("distortion_model") != "rectified_pinhole":
        raise ValueError("Only declared rectified optical-Z depth is supported")
    k = record["intrinsics"]
    if any(name not in k for name in ("fx", "fy", "cx", "cy")) or not record["intrinsics_provenance"]:
        raise ValueError("Full native/crop K is required; focal-only data cannot supply a guessed principal point")
    if not record["alignment_provenance"] or float(record["depth_scale_m"]) <= 0:
        raise ValueError("Positive explicit depth scale and alignment provenance required")
    depth = load_array(record["depth_path"]).astype(np.float64)*record["depth_scale_m"]
    v = load_array(record["visible_mask_path"])
    v = v > (127 if v.max(initial=0) > 1 else .5)
    a = None
    if record.get("amodal_mask_path"):
        a = load_array(record["amodal_mask_path"])
        a = a > (127 if a.max(initial=0) > 1 else .5)
    if depth.shape != v.shape or (a is not None and a.shape != v.shape):
        raise ValueError("RGB-D/native mask dimensions differ; alignment cannot be inferred from resizing")
    if record.get("center_gt_m") is not None and not record.get("center_gt_provenance"):
        raise ValueError("A true 3-D centre requires independent ground-truth provenance")
    if record.get("diameter_gt_m") is not None and not record.get("diameter_gt_provenance"):
        raise ValueError("Diameter GT requires provenance and unit declaration")
    return dict(depth_m=depth, visible=v, amodal=a, intrinsics=k)


def run_case(identifier, sample, cfg, *, prior=None, amodal_quality=.25, ablation_prior=None):
    points = points_from_depth(sample["depth_m"], sample["visible"], sample["intrinsics"])
    edges = candidate_outer_edges(sample["visible"], sample.get("amodal"), amodal_quality=amodal_quality,
                                  depth_m=sample["depth_m"])
    rays = rays_from_pixels(edges["xy"], sample["intrinsics"])
    rows = []
    variants = [(name, name, prior) for name in ("ransac_refined", "depth_huber", "joint_huber")]
    if ablation_prior is not None:
        variants.append(("joint_huber", "joint_huber_external_prior", ablation_prior))
    for method, label, selected_prior in variants:
        result = fit_sphere(points, method=method, contour_rays=rays, contour_weights=edges["weights"],
                            radius_prior=selected_prior, config=cfg)
        row = {"case_id": identifier, **result, "method": label, "solver_method": method, "candidate_edge_count": len(rays),
               "boundary_flags": edges["flags"], "center_error_mm": None, "diameter_error_mm": None,
               "diameter_absolute_error_mm": None}
        # Ground truth is accessed only after the fitting call, never passed in.
        if result["center_m"] is not None:
            if sample.get("center_gt_m") is not None:
                row["center_error_mm"] = float(np.linalg.norm(np.array(result["center_m"])-sample["center_gt_m"])*1000)
            diameter_truth = sample.get("diameter_gt_m")
            if diameter_truth is None and sample.get("radius_gt_m") is not None:
                diameter_truth = 2*sample["radius_gt_m"]
            if diameter_truth is not None:
                row["diameter_error_mm"] = float((2*result["radius_m"]-diameter_truth)*1000)
                row["diameter_absolute_error_mm"] = abs(row["diameter_error_mm"])
        rows.append(row)
    return rows


def summarize(rows):
    summary = {}
    for method in dict.fromkeys(row["method"] for row in rows):
        selected = [row for row in rows if row["method"] == method]
        computed = [row for row in selected if row["status"] == "computed"]
        accepted = [row for row in computed if row["accepted"]]
        value = dict(cases=len(selected), computed=len(computed), accepted=len(accepted),
                     rejected=len(selected)-len(accepted),
                     mean_elapsed_ms=float(np.mean([r["elapsed_ms"] for r in selected])),
                     p95_elapsed_ms=float(np.percentile([r["elapsed_ms"] for r in selected], 95)))
        for scope, subset in (("all_computed", computed), ("accepted", accepted)):
            for key in ("center_error_mm", "diameter_absolute_error_mm"):
                numbers = [r[key] for r in subset if r[key] is not None]
                value[scope+"_"+key+"_mean"] = float(np.mean(numbers)) if numbers else None
                value[scope+"_"+key+"_p95"] = float(np.percentile(numbers, 95)) if numbers else None
        summary[method] = value
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--synthetic", action="store_true")
    source.add_argument("--manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max-records", type=int, default=0)
    args = parser.parse_args(argv)
    if args.output.resolve().drive.upper() != "F:":
        raise ValueError("All results must remain on F:")
    if args.output.exists():
        raise FileExistsError("Use a fresh result path; prior evidence is not overwritten")
    if args.repeats < 1:
        raise ValueError("At least one synthetic repeat required")
    cfg = SphereConfig()
    rows, cases, input_hashes = [], [], {}
    oracle_sanity = None
    illustrative_prior = dict(mean_m=.045, std_m=.015, provenance_scope="external",
                              source="synthetic protocol illustrative external prior, not measured/trained")
    if args.synthetic:
        for occlusion in ("none", "half", "heavy", "tiny_cap"):
            for repeat in range(args.repeats):
                sample = render_sphere(seed=17+repeat, occlusion=occlusion,
                                       depth_noise_m=.001, outlier_fraction=.05)
                identifier = f"{occlusion}_{repeat:02d}"
                rows.extend(run_case(identifier, sample, cfg, ablation_prior=illustrative_prior))
                cases.append(dict(case_id=identifier, occlusion=occlusion,
                                  visible_fraction=float(sample["visible"].sum()/sample["amodal"].sum()),
                                  source="analytic RGB-D sphere; perfect synthetic V/A masks, not model predictions",
                                  depth_noise_sigma_m=.001, outlier_fraction=.05))
        # One explicit oracle checks whether raster contour quantization matters.
        # Its rays are computed from synthetic truth and are NOT a deployment result.
        sample = render_sphere(seed=17, occlusion="none", depth_noise_m=.001, outlier_fraction=.05)
        points = points_from_depth(sample["depth_m"], sample["visible"], sample["intrinsics"])
        oracle_rays = tangent_rays(sample["center_gt_m"], sample["radius_gt_m"], count=160)
        oracle = fit_sphere(points, method="joint_huber", contour_rays=oracle_rays,
                            contour_weights=np.full(len(oracle_rays), .25), config=cfg)
        oracle_sanity = dict(source="exact analytic tangent rays constructed from synthetic centre/radius truth",
                            center_or_diameter_truth_used_to_construct_input=True,
                            excluded_from_benchmark_summary=True,
                            comparison_case_id="none_00", result=oracle,
                            center_error_mm=float(np.linalg.norm(np.array(oracle["center_m"])-sample["center_gt_m"])*1000)
                            if oracle["center_m"] is not None else None,
                            limitation="One oracle case diagnoses discretization sensitivity; it does not establish an improved real contour extractor")
    else:
        payload = json.loads(args.manifest.read_text(encoding="utf-8-sig"))
        if payload.get("schema") != "apple_rgbd_geometry_manifest_v1":
            raise ValueError("Unsupported real manifest schema")
        input_hashes[str(args.manifest.resolve())] = sha(args.manifest)
        records = payload["records"][:args.max_records or None]
        for record in records:
            sample = read_real_record(record)
            for key in ("center_gt_m", "diameter_gt_m"):
                sample[key] = record.get(key)
            for key in ("depth_path", "visible_mask_path", "amodal_mask_path"):
                if record.get(key):
                    input_hashes[record[key]] = sha(record[key])
            rows.extend(run_case(record["image_id"], sample, cfg, prior=record.get("radius_prior"),
                                 amodal_quality=float(record.get("amodal_quality", 0))))
            cases.append({key: value for key, value in record.items() if key not in ("center_gt_m", "diameter_gt_m")})
    files = [Path(__file__), ROOT/"src/apple_local_rgbd/geometry.py", ROOT/"src/apple_local_rgbd/boundaries.py",
             ROOT/"src/apple_local_rgbd/synthetic.py"]
    report = dict(schema="apple_rgbd_geometry_cpu_benchmark_v1", completed_utc=datetime.now(timezone.utc).isoformat(),
                  source_mode="synthetic" if args.synthetic else "real_manifest", config=asdict(cfg),
                  GPU_used=False, training_performed=False, benchmark_center_or_diameter_truth_passed_to_fit=False,
                  synthetic_masks_are_analytic_ground_truth=bool(args.synthetic),
                  synthetic_size_prior_ablation=illustrative_prior if args.synthetic else None,
                  analytic_contour_oracle_sanity=oracle_sanity,
                  sphere_magsac_plusplus=dict(status="not_implemented", included_in_comparison=False),
                  real_fruit_center_accuracy_claimed=False if args.synthetic else any(r["center_error_mm"] is not None for r in rows),
                  source_sha256={str(p):sha(p) for p in files}, input_sha256=input_hashes,
                  cases=cases, results=rows, summary=summarize(rows),
                  limitations=["Synthetic sphere noise/occlusion are not real D405 or PApple accuracy evidence",
                               "V/A edge agreement is only a weak candidate, not verified true contour",
                               "Joint solver is bounded Huber TRF, not classic LM or MAGSAC++",
                               "Depth-only conditioning gates are not calibrated uncertainty probabilities",
                               "No full visual model or camera latency is included in CPU geometry timing",
                               "Repeated noisy realizations measure synthetic sensitivity, not physical temporal jitter",
                               "joint_huber has no radius prior; joint_huber_external_prior adds an illustrative independent distribution",
                               "Analytic-contour oracle deliberately uses synthetic truth and is excluded from benchmark summary"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(dict(output=str(args.output.resolve()), summary=report["summary"]), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
