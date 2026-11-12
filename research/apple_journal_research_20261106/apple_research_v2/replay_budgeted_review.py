"""CPU-only fixed-budget replay of frozen paired image predictions.

No model imports, inference, fitting, threshold selection or test-set access.
Costs are OFFLINE sums of saved native prediction calls, not deployed latency.
"""
from __future__ import annotations
import argparse
import csv
from datetime import datetime, timezone
from decimal import Decimal, ROUND_FLOOR
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys

sys.dont_write_bytecode = True
FRACTIONS = ("0", "0.1", "0.25", "0.5", "1")
SEEDS = (20261106, 20261107, 20261108, 20261109, 20261110)
PAIR_SOURCE_SHA = "317f72febb7ab154fa759fa3281ab9057321b6b78fb04faf6045a5d7a4c939bc"
PROTOCOL_SHA = "44df99d13c594ca5fbdd4a7b94cf8d64f57626ca63841ec5da2fe81c8ed2a57e"
COUNT_KEYS = ("support", "TP", "FP", "FN")
ROOT = Path(__file__).resolve().parents[1]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write("\n")


def action_count(n, fraction):
    value = Decimal(str(fraction))
    require(n >= 0 and 0 <= value <= 1, "Invalid action-count budget")
    return int((Decimal(n) * value).to_integral_value(rounding=ROUND_FLOOR))


def gain(row):
    return row["resolutions"]["640"]["box_matching"]["FP_plus_FN"] - row["resolutions"]["960"]["box_matching"]["FP_plus_FN"]


def rank_ids(rows, policy, seed=None):
    if policy == "coarse_min_confidence":
        ranked = sorted(rows, key=lambda r: (r["coarse_min_confidence"], r["image_id"]))
    elif policy == "random":
        require(seed in SEEDS, "Unfrozen random seed")
        ranked = sorted(rows, key=lambda r: (hashlib.sha256((str(seed)+"\n"+r["image_id"]).encode("utf-8")).hexdigest(), r["image_id"]))
    elif policy == "GT_oracle_positive_gain":
        ranked = sorted((r for r in rows if gain(r) > 0), key=lambda r: (-gain(r), r["image_id"]))
    else:
        raise ValueError("Unknown ranking policy")
    return [r["image_id"] for r in ranked]


def add_rates(counts):
    counts["FP_plus_FN"] = counts["FP"] + counts["FN"]
    counts["precision"] = counts["TP"] / (counts["TP"]+counts["FP"]) if counts["TP"]+counts["FP"] else None
    counts["recall"] = counts["TP"] / counts["support"] if counts["support"] else None
    return counts


def aggregate(rows, selected, names, direct=False):
    per_class = {str(k): {"class_name": v, **{x: 0 for x in COUNT_KEYS}} for k, v in names.items()}
    for row in rows:
        size = "960" if direct or row["image_id"] in selected else "640"
        current = row["resolutions"][size]["box_matching"]
        for cid in per_class:
            for key in COUNT_KEYS:
                per_class[cid][key] += current["per_class"][cid][key]
    total = {key: sum(c[key] for c in per_class.values()) for key in COUNT_KEYS}
    total = add_rates(total)
    total["per_class"] = {cid: add_rates(value) for cid, value in per_class.items()}
    return total


def nearest_rank(values, quantile):
    require(bool(values) and 0 < quantile <= 1, "Invalid nearest-rank input")
    return sorted(values)[max(0, math.ceil(quantile*len(values))-1)]


def evaluate(rows, names, policy, selected_ids, fraction=None, seed=None, direct=False):
    selected = set(selected_ids)
    require(len(selected) == len(selected_ids), "Duplicate selected ID")
    require(selected <= {r["image_id"] for r in rows}, "Unknown selected ID")
    base_risk = sum(r["resolutions"]["640"]["box_matching"]["FP_plus_FN"] for r in rows)
    budget = None if fraction is None else action_count(len(rows), fraction)
    if budget is not None:
        require(len(selected) <= budget, "Action budget violated")
    records, costs, signs = [], [], {"positive": 0, "zero": 0, "negative": 0}
    selected_gain = 0
    for row in rows:
        is_selected = row["image_id"] in selected
        use_high = direct or is_selected
        size = "960" if use_high else "640"
        coarse = row["resolutions"]["640"]["native_prediction_call_ms"]
        high = row["resolutions"]["960"]["native_prediction_call_ms"]
        cost = high if direct else coarse + (high if is_selected else 0)
        delta = gain(row)
        if is_selected:
            signs["positive" if delta > 0 else "negative" if delta < 0 else "zero"] += 1
            selected_gain += delta
        costs.append(cost)
        records.append({
            "image_id": row["image_id"], "known_group_id": row.get("known_group_id"),
            "selected_rerun": is_selected, "output_imgsz": int(size),
            "coarse_min_confidence": row["coarse_min_confidence"],
            "replay_cost_ms": cost, "saved_640_ms": coarse, "saved_960_ms": high,
            "source_gain_640_minus_960": delta,
            "output_counts": {key: row["resolutions"][size]["box_matching"][key] for key in (*COUNT_KEYS, "FP_plus_FN")},
            "output_per_class": row["resolutions"][size]["box_matching"]["per_class"],
        })
    counts = aggregate(rows, selected, names, direct=direct)
    report = {
        "policy": policy, "fraction": fraction, "action_budget_K": budget,
        "random_seed": seed, "uses_GT_to_select": policy == "GT_oracle_positive_gain",
        "selection_deployable": policy != "GT_oracle_positive_gain",
        "policy_is_offline_dataset_ranking": fraction is not None,
        "sequential_rerun_actions": len(selected), "output_960_images": len(rows) if direct else len(selected),
        "selected_image_ids_in_rank_order": selected_ids, "selected_gain_sign_counts": signs,
        "selected_rerun_risk_reduction": selected_gain,
        "risk_reduction_from_640": base_risk-counts["FP_plus_FN"], "counts": counts,
        "offline_replay_cost_ms": {
            "sum": sum(costs), "mean": sum(costs)/len(costs),
            "p50_nearest_rank": nearest_rank(costs, .5), "p95_nearest_rank": nearest_rank(costs, .95),
            "p99_nearest_rank": nearest_rank(costs, .99),
            "scope": "Saved 960 call only for direct_960; otherwise saved 640 + selected saved 960. No gate/scheduling overhead or actual sequential re-execution.",
        },
    }
    return report, records


def load_source(pair_dir):
    pair_dir = Path(pair_dir).resolve()
    require(pair_dir.is_relative_to(ROOT/"experiments"/"resolution_pairs"), "Source must be in the authorized paired development-results directory")
    require(pair_dir.name in ("stage_v1", "ssda_v1"), "Only frozen v1 development pair inputs are supported")
    frozen = ROOT/"apple_research_v2"/"evaluate_resolution_pairs.py"
    protocol = ROOT/"experiments"/"protocols"/"fixed_budget_replay_v1.md"
    require(sha(frozen) == PAIR_SOURCE_SHA, "Frozen pair matching source changed")
    require(sha(protocol) == PROTOCOL_SHA, "Frozen replay protocol changed")
    spec = importlib.util.spec_from_file_location("frozen_pair_matcher", frozen)
    matcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(matcher)
    receipt = {str(frozen): sha(frozen), str(protocol): sha(protocol)}

    def read(path, expected=None):
        path = Path(path).resolve()
        require(path.is_relative_to(pair_dir), "Prediction path escapes source directory")
        digest = sha(path)
        require(expected is None or digest == expected, "Source SHA mismatch: "+str(path))
        require(str(path) not in receipt or receipt[str(path)] == digest, "Source changed while reading")
        receipt[str(path)] = digest
        return json.loads(path.read_text(encoding="utf-8-sig"))

    summary = read(pair_dir/"summary.json")
    preflight = read(pair_dir/"preflight.json", summary["preflight_sha256"])
    rows = read(pair_dir/"paired_images.json")
    require(summary["schema"] == "resolution_pairs_v1", "Unexpected source schema")
    require(preflight["source_sha256"] == PAIR_SOURCE_SHA, "Source pairs used an unexpected script")
    require(summary["source_inputs_unchanged"] is True, "Source pair inputs changed")
    require(summary["test_read"] is False and preflight["test_read"] is False, "Unexpected test access in source")
    require(summary["protocol"]["confidence_inclusive"] == .25, "Unexpected confidence")
    require(summary["protocol"]["box_IoU_inclusive"] == .5 and summary["protocol"]["class_aware"] is True, "Unexpected matcher")
    names = {str(k): v for k, v in summary["class_names"].items()}
    require(rows and len(rows) == summary["images"], "Image count mismatch")
    require(len({r["image_id"] for r in rows}) == len(rows), "Duplicate image IDs")
    for row in rows:
        gt_reference = None
        for size in ("640", "960"):
            resolution = row["resolutions"][size]
            detail = read(pair_dir/resolution["prediction_file"], resolution["prediction_file_sha256"])
            require(detail["image_id"] == row["image_id"] and detail["imgsz"] == int(size), "Prediction identity mismatch")
            require(detail["weights_sha256"] == summary["weights_sha256"], "Weight identity mismatch")
            require(detail["full_view"] is True, "Expected full-view output")
            require(detail["native_prediction_call_ms"] == resolution["native_prediction_call_ms"], "Cost mismatch")
            require(math.isfinite(resolution["native_prediction_call_ms"]) and resolution["native_prediction_call_ms"] > 0, "Invalid cost")
            ground_truth = detail["ground_truth"]
            current_gt = canonical(ground_truth)
            require(gt_reference is None or current_gt == gt_reference, "GT differs across scales")
            gt_reference = current_gt
            for target in ground_truth + detail["predictions"]:
                require(str(target["class_id"]) in names and target["class_name"] == names[str(target["class_id"])], "Class name/id mismatch")
            recomputed = matcher.match_boxes(detail["predictions"], ground_truth, names)
            # JSON-normalize integer dictionary keys for exact reproducible comparison.
            recomputed = json.loads(json.dumps(recomputed))
            require(canonical(recomputed) == canonical(detail["box_matching"]) == canonical(resolution["box_matching"]), "Recomputed matching differs")
            require(set(recomputed["per_class"]) == set(names), "Missing class count")
            if size == "640":
                scores = [p["score"] for p in detail["predictions"]]
                require(all(math.isfinite(s) and .25 <= s <= 1 for s in scores), "Unexpected coarse score filtering")
                row["coarse_min_confidence"] = min(scores) if scores else 0.0
                row["coarse_prediction_count"] = len(scores)
        require(row["delta"]["risk_reduction_640_minus_960"] == gain(row), "Saved gain mismatch")
        for cid in names:
            require(row["resolutions"]["640"]["box_matching"]["per_class"][cid]["support"] == row["resolutions"]["960"]["box_matching"]["per_class"][cid]["support"], "Changed class denominator")
    rows.sort(key=lambda r: r["image_id"])
    for size in ("640", "960"):
        computed = aggregate(rows, set(), names, direct=size == "960")
        require(canonical(computed) == canonical(summary["counts"][size]), "Aggregate mismatch vs source summary")
    return rows, names, summary, receipt


def run(pair_dir, output):
    rows, names, source, receipt = load_source(pair_dir)
    output = Path(output).resolve()
    require(ROOT.drive.lower() == "f:", "Replay must reside on F")
    require((ROOT/"relocation_complete.json").is_file(), "Relocation receipt required")
    require(output.is_relative_to(ROOT/"experiments"/"budget_replay"), "Output must be under F research experiments/budget_replay")
    require(not output.exists(), "Refuse to overwrite existing result")
    output.mkdir(parents=True, exist_ok=False)
    policies = []
    details = {}
    for name, selected, direct in (
        ("baseline_640", [], False), ("direct_960", [], True),
        ("rerun_all", [r["image_id"] for r in rows], False),
    ):
        report, records = evaluate(rows, names, name, selected, direct=direct)
        policies.append(report)
        details[name] = records
    rankings = {"coarse_min_confidence": rank_ids(rows, "coarse_min_confidence"),
                "GT_oracle_positive_gain": rank_ids(rows, "GT_oracle_positive_gain")}
    for seed in SEEDS:
        rankings["random_"+str(seed)] = rank_ids(rows, "random", seed)
    for policy_key, ranking in rankings.items():
        policy = "random" if policy_key.startswith("random_") else policy_key
        seed = int(policy_key.split("_")[1]) if policy == "random" else None
        for fraction in FRACTIONS:
            selected = ranking[:action_count(len(rows), fraction)]
            report, records = evaluate(rows, names, policy, selected, fraction, seed)
            run_id = policy_key+"_f"+fraction
            report["run_id"] = run_id
            policies.append(report)
            details[run_id] = records
    random_summaries = []
    for fraction in FRACTIONS:
        group = [p for p in policies if p["policy"] == "random" and p["fraction"] == fraction]
        random_summaries.append({
            "fraction": fraction, "action_budget_K": action_count(len(rows), fraction),
            "seeds": list(SEEDS), "runs": len(group),
            "risk_FP_plus_FN_mean": sum(p["counts"]["FP_plus_FN"] for p in group)/len(group),
            "risk_FP_plus_FN_min": min(p["counts"]["FP_plus_FN"] for p in group),
            "risk_FP_plus_FN_max": max(p["counts"]["FP_plus_FN"] for p in group),
            "offline_cost_ms_mean_over_seeds": sum(p["offline_replay_cost_ms"]["mean"] for p in group)/len(group),
            "note": "Descriptive variation across fixed deterministic permutations; not a confidence interval.",
        })
    unchanged = all(sha(path) == expected for path, expected in receipt.items())
    require(unchanged, "Source file changed during replay")
    summary = {
        "schema": "fixed_budget_confidence_replay_v1", "created_utc": datetime.now(timezone.utc).isoformat(),
        "kind": source["kind"], "source_pair_directory": str(Path(pair_dir).resolve()),
        "weights_sha256": source["weights_sha256"], "class_names": names, "images": len(rows),
        "script_sha256": sha(__file__), "protocol_sha256": PROTOCOL_SHA,
        "fractions": list(FRACTIONS), "random_seeds": list(SEEDS), "action_budget": "floor(N*f)",
        "coarse_score": "minimum retained 640 score; zero when no predictions; ascending; image_id ties",
        "matching_protocol": source["protocol"], "source_semantics": source["semantics"],
        "source_timing_scope": source["timing_scope"], "source_timing_excludes": source["timing_excludes"],
        "GPU_used": False, "inference_performed": False, "model_fitted": False,
        "test_read": False, "source_inputs_unchanged": unchanged,
        "no_FPS_estimate_computed": True, "no_gate_overhead_measured": True,
        "policies": policies, "random_fixed_seed_summary": random_summaries,
        "limitations": [
            "Locked descriptive development analysis after original paired results were observed; not independent final evaluation.",
            "Dataset-wide ranking action counts do not define an online streaming deadline or equal-latency budgets.",
            "Offline latency sums omit actual gate/scheduling overhead and execution-state changes; no deployed latency or FPS claim.",
            "GT oracle is nondeployable and at-most-K; its class outcomes and cost need not equal other policies.",
            "Stage box error uses original author polygon envelopes, not a mask metric or physiological ripeness.",
            "SSDA class names remain source numeric names; GT support counts defect boxes, not whole fruits.",
            "Existing known groups do not establish physical-fruit or orchard independence; no learned method or novelty established.",
            "Semi-mature class has only three development GT instances; no reliable class generalization claim.",
        ],
    }
    write_json(output/"summary.json", summary)
    write_json(output/"per_policy_images.json", details)
    write_json(output/"rankings.json", rankings)
    write_json(output/"source_sha256_receipt.json", receipt)
    fields = ["run_id", "policy", "fraction", "seed", "K", "rerun_actions", "output_960_images", "class_id", "class_name",
              "support", "TP", "FP", "FN", "FP_plus_FN", "precision", "recall", "risk_reduction_total", "offline_cost_mean_ms", "offline_cost_sum_ms"]
    with (output/"policy_metrics.csv").open("x", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for p in policies:
            for cid, counts in [("ALL", p["counts"]), *p["counts"]["per_class"].items()]:
                writer.writerow({
                    "run_id": p.get("run_id", p["policy"]), "policy": p["policy"], "fraction": p["fraction"],
                    "seed": p["random_seed"], "K": p["action_budget_K"], "rerun_actions": p["sequential_rerun_actions"],
                    "output_960_images": p["output_960_images"], "class_id": cid, "class_name": counts.get("class_name", "ALL"),
                    **{key: counts[key] for key in (*COUNT_KEYS, "FP_plus_FN", "precision", "recall")},
                    "risk_reduction_total": p["risk_reduction_from_640"],
                    "offline_cost_mean_ms": p["offline_replay_cost_ms"]["mean"], "offline_cost_sum_ms": p["offline_replay_cost_ms"]["sum"],
                })
    print(json.dumps({"output": str(output), "images": len(rows), "policy_runs": len(policies),
                      "source_inputs_unchanged": unchanged, "GPU_used": False}, ensure_ascii=False))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.pairs, args.output)

