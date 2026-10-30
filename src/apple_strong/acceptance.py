"""CPU-only acceptance of strong C0/C1 nodes with independent metric replay."""
from __future__ import annotations
from collections import Counter
from pathlib import Path
import hashlib
import importlib.util
import json
import math
import os

REPO = Path(__file__).absolute().parents[2]
RUNTIME = REPO / "strong_runtime"
BASE_SHA = "0fe257c5bfe115a77ac7deac830af479e4c2b0ad6ff69e609901b24c7980fa9b"
SMALL_SHA = "e0c1f659c2509d7a01d617e12cb83c673768b440ea5b4b96cb4f763e4207e7e3"
PROVENANCE_SHA = "fcd96d08c75e1663c83efecb8eb321bbc10220dc74314c8227f1f8565a2ccb3a"
METRICS_SHA = "1b0a7e385940fd0c6160cc08b75ddaf2e1f43f3ea00f5bd37d547fea6b08010d"
SCOPE = (814, 3449, 13)
FUSION = dict(iou_threshold=.45, score_mode="mean", box_mode="score_weighted",
              coverage_floor=1., view_weights=[.65, .65, .35, .35],
              consensus_groups=[0, 0, 1, 1], consensus_bonus=1.10, isolation_penalty=.90)
STRATA = {"heavy_visible_lt_0.40":1514, "moderate_visible_0.40_to_0.70":1079,
          "light_visible_0.70_to_0.90":627, "mostly_visible_0.90_to_1.00":215,
          "annotation_visible_area_exceeds_amodal":14}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _read(path):
    with Path(path).open(encoding="utf-8-sig") as stream:
        return json.load(stream)


def _sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _inventory():
    core = RUNTIME / "src/apple_seg"
    required = ("od_model.py", "od_inference.py", "od_evaluate.py", "od_data.py", "data.py")
    _require(all((core / name).is_file() for name in required), "Incomplete strong runtime")
    files = [REPO / "scripts/reproduce_strong_baseline.py", *sorted(core.glob("*.py")),
             RUNTIME / "tmp/flip_tta.py", RUNTIME / "reference/local_amodal/scripts/metrics.py"]
    return {str(path.relative_to(REPO)):_sha(path) for path in files}


def _source(value, inventory, label):
    _require(isinstance(value, dict) and value == inventory, label + " source inventory differs")


def _metrics():
    path = RUNTIME / "reference/local_amodal/scripts/metrics.py"
    _require(_sha(path) == METRICS_SHA, "Frozen matching implementation differs")
    spec = importlib.util.spec_from_file_location("_strong_acceptance_exact_metrics", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _provenance(path):
    _require(_sha(path) == PROVENANCE_SHA, "Original provenance SHA differs")
    value = _read(path)
    scope = value["scope"]
    _require(value.get("phase") == "complete" and value.get("passed") is True
             and scope.get("split") == "val", "Incomplete original provenance")
    images, truth = value["image_metadata"], value["ground_truth_instances"]
    _require((len(images), len(truth), scope.get("empty_GT_images")) == SCOPE
             and (scope.get("images"), scope.get("GT_instances")) == SCOPE[:2], "Original scope differs")
    names = [r["image_id"] for r in images]
    _require(all(isinstance(n,str) and n for n in names) and len(set(names)) == 814,
             "Original image identities are empty or duplicated")
    by_image = {n:[] for n in names}
    for row in truth:
        _require(row["image_id"] in by_image, "GT references unknown image")
        by_image[row["image_id"]].append(row)
    for index, image in enumerate(images):
        rows = by_image[image["image_id"]]
        _require(image["image_index"] == index and image["gt_count"] == len(rows)
                 and [r["gt_index"] for r in rows] == list(range(len(rows))), "Original GT order differs")
        _require(image["apple_ids_in_target_order"] == [r["apple_id"] for r in rows],
                 "Original apple identity differs")
    _require(sum(not r for r in by_image.values()) == 13, "Empty image count differs")
    _require(Counter(r["occlusion_stratum"] for r in truth) == STRATA, "Occlusion groups differ")
    return {"names":names, "truth":by_image}


def _records(path, contract):
    import numpy as np
    values = _read(path)
    _require(isinstance(values,list) and len(values) == 814, "Predictions must contain full 814 frames")
    names = [r.get("image_id") for r in values]
    _require(names == contract["names"] and len(set(names)) == 814, "Prediction image identities/order differ")
    for row in values:
        expected = [g["box"] for g in contract["truth"][row["image_id"]]]
        _require(row.get("gt",{}).get("boxes") == expected, "Predicted-record GT boxes differ")
        boxes = np.asarray(row["pred"]["boxes"],dtype=np.float64).reshape(-1,4)
        scores = np.asarray(row["pred"]["scores"],dtype=np.float64)
        _require(scores.shape == (len(boxes),) and np.isfinite(boxes).all() and np.isfinite(scores).all()
                 and ((scores >= 0) & (scores <= 1)).all(), "Prediction arrays are invalid")
        _require((boxes[:,2:] > boxes[:,:2]).all(), "Prediction contains non-positive box")
    return values


def _best(records, metrics):
    calibrated = metrics.calibrate_threshold(records,.5,return_curve=True)
    eligible = [r for r in calibrated["curve"] if r["recall"] >= .90]
    return max(eligible,key=lambda r:(r["precision"],r["recall"],r["f1"],r["confidence"])) if eligible else None


def _same_metric(declared, actual, label):
    _require(isinstance(declared,dict) and isinstance(actual,dict), label + " lacks attainable R90 workpoint")
    for key in ("tp","fp","fn","gt_count","prediction_count"):
        _require(type(declared.get(key)) is int and declared[key] == actual[key], label + " count differs: " + key)
    for key in ("precision","recall","f1","confidence","iou_threshold"):
        value = declared.get(key)
        _require(isinstance(value,(int,float)) and not isinstance(value,bool)
                 and math.isfinite(value) and abs(value-actual[key]) <= 1e-12, label + " value differs: " + key)


def _weight(meta, expected, cache):
    checkpoint = Path(meta["checkpoint"])
    _require(checkpoint.is_file(), "Checkpoint referenced by inference is absent")
    declared = meta.get("checkpoint_sha256")
    _require(isinstance(declared,str) and len(declared) == 64
             and all(c in "0123456789abcdef" for c in declared), "Invalid checkpoint SHA")
    if expected is not None:
        _require(declared == expected, "Wrong delivered backbone checkpoint")
    sidecar = checkpoint.with_suffix(checkpoint.suffix + ".sha256")
    _require(sidecar.read_text(encoding="ascii").strip() == declared, "Checkpoint sidecar differs")
    stat = checkpoint.stat()
    key = (str(checkpoint.absolute()),stat.st_size,stat.st_mtime_ns)
    actual = cache.get(key)
    if actual is None:
        actual = _sha(checkpoint); cache[key] = actual
    _require(actual == declared, "Actual checkpoint bytes differ from prediction metadata")
    return actual


def _metadata(directory, model, inventory, cache, expected=None, candidate=False):
    value = _read(directory / model / "prediction_metadata.json")
    _require(value.get("phase") == "complete" and value.get("model") == model, "Incomplete model inference")
    _require((value.get("images"), value.get("GT"), value.get("empty")) == SCOPE, "Model view scope differs")
    _require(value.get("candidate_mode") is candidate and value.get("from_fresh_weight_inference") is True
             and value.get("source_before_after_identical") is True and value.get("FP32") is True,
             "Model view was not fully reproduced with the frozen FP32 protocol")
    _require(value.get("short_side") == 1024 and value.get("max_size") == 1280, "Inference scale changed")
    _require(value.get("provenance_sha256") == PROVENANCE_SHA, "Model provenance differs")
    restore = value.get("strict_restore",{})
    _require(restore.get("strict") is True and restore.get("all_tensors_equal_to_checkpoint") is True
             and restore.get("missing_keys") == [] and restore.get("unexpected_keys") == []
             and type(restore.get("tensor_count")) is int and restore["tensor_count"] > 0,
             "Complete strict tensor restoration was not recorded")
    _source(value.get("source_sha256"),inventory,model)
    digest = _weight(value,expected,cache)
    return value,digest


def _heavy(records,confidence,contract,metrics):
    tp = 0
    for r in records:
        match = metrics.match_image(r["gt"]["boxes"],r["pred"]["boxes"],r["pred"]["scores"],.5,confidence)
        recalled = {int(i) for i in match["gt_indices"] if i >= 0}
        tp += sum(g["gt_index"] in recalled for g in contract["truth"][r["image_id"]]
                  if g["occlusion_stratum"] == "heavy_visible_lt_0.40")
    return {"GT":1514,"TP":tp,"recall":tp/1514}


def _masks(directory,records,view_best,contract,metrics):
    path = directory / "base/mask_metrics_at_recall90.json"
    declared = _read(path)
    _require(abs(declared["confidence"]-view_best["confidence"]) <= 1e-12, "Base mask threshold differs")
    diagnostics_path = directory / "base/mask_pair_diagnostics.json"
    diagnostics = _read(diagnostics_path)
    _require(isinstance(diagnostics,list) and len(diagnostics) == 814
             and [r["image_id"] for r in diagnostics] == contract["names"], "Base mask diagnostics scope differs")
    tp = {"visible":0,"amodal":0}; count = 0
    for row,record in zip(diagnostics,records,strict=True):
        gt = contract["truth"][row["image_id"]]
        _require(row["gt_count"] == len(gt) and row["apple_ids_in_target_order"] == [r["apple_id"] for r in gt],
                 "Base mask GT identity differs")
        scores = record["pred"]["scores"]
        _require(row["scores"] == scores, "Mask/box score alignment differs")
        selected = [s >= view_best["confidence"] for s in scores]
        count += sum(selected)
        for branch in tp:
            assignments = row[branch+"_mask_gt_indices_by_prediction"]
            _require(len(assignments) == len(scores)
                     and all(type(i) is int and -1 <= i < len(gt) for i in assignments), "Invalid mask matching identities")
            positives = [i for i in assignments if i >= 0]
            _require(len(positives) == len(set(positives)), "Mask assignments are not one-to-one")
            tp[branch] += sum(flag and i >= 0 for flag,i in zip(selected,assignments,strict=True))
    result = {}
    for branch in tp:
        point = metrics._counts(tp[branch],count-tp[branch],3449,view_best["confidence"],.5)
        _same_metric(declared["metrics"][branch],point,"Base "+branch+" mask")
        result[branch] = point
    return {"scope":"Base original view only; independently matched masks at its own R90 BOX threshold",
            "metrics":result,"diagnostics_sha256":_sha(diagnostics_path),"report_sha256":_sha(path)}


def _assess(directory,contract,metrics,inventory,cache,baseline=None):
    directory = Path(directory).absolute()
    candidate = baseline is not None
    summary = _read(directory / "fusion_summary.json")
    _require(summary.get("phase") == "complete" and (summary.get("images"),summary.get("GT"),summary.get("empty")) == SCOPE,
             "Fusion scope is not complete original VAL")
    _require(summary.get("from_fresh_weight_inference") is True and summary.get("candidate_mode") is candidate,
             "Fusion is not the expected fresh baseline/candidate mode")
    _source(summary.get("source_sha256"),inventory,"fusion")
    _require(summary.get("fusion") == FUSION, "Fixed four-view fusion settings changed")
    base_meta,base_sha = _metadata(directory,"base",inventory,cache,BASE_SHA if not candidate else None,candidate)
    small_meta,small_sha = _metadata(directory,"small",inventory,cache,SMALL_SHA,False)
    layers = {}
    hashes = {}
    for model in ("base","small"):
        for view in ("original","hflip"):
            rel = model+"/"+view+"_predictions.json"
            path = directory / rel
            layers[rel] = _records(path,contract)
            hashes[rel] = _sha(path)
            declared = (base_meta if model == "base" else small_meta)["view_metrics"][view]["at_recall90"]
            _same_metric(declared,_best(layers[rel],metrics),model+" "+view)
    if candidate:
        for rel in ("small/original_predictions.json","small/hflip_predictions.json"):
            _require(hashes[rel] == baseline["view_prediction_sha256"][rel], "Frozen Small cache changed")
    fused_path = directory / "fused_predictions.json"
    fused = _records(fused_path,contract)
    best = _best(fused,metrics)
    if best is None:
        _require(summary.get("at_recall90") is None, "Unattainable R90 was extrapolated")
    else:
        _same_metric(summary.get("at_recall90"),best,"fusion R90")
    fixed = metrics.evaluate_at_threshold(fused,.75,.5)
    _same_metric(summary.get("at_confidence_0_75"),fixed,"fusion confidence0.75")
    _require((fixed["images"],fixed["gt_count"],fixed["empty_gt_images"]) == SCOPE,"Replayed scope differs")
    masks = _masks(directory,layers["base/original_predictions.json"],base_meta["view_metrics"]["original"]["at_recall90"],contract,metrics)
    return {"directory":str(directory),"candidate_sha256":base_sha,"small_checkpoint_sha256":small_sha,
            "at_recall90":best,"at_confidence_0_75":fixed,"heavy":_heavy(fused,best["confidence"],contract,metrics) if best else None,
            "base_original_masks":masks,"view_prediction_sha256":hashes,
            "fused_predictions_sha256":_sha(fused_path),"fusion_summary_sha256":_sha(directory/"fusion_summary.json"),
            "base_metadata_sha256":_sha(directory/"base/prediction_metadata.json"),
            "small_metadata_sha256":_sha(directory/"small/prediction_metadata.json")}


def _baseline_gate(directory,assessment,inventory):
    local = Path(directory) / "baseline_gate.json"
    path = local if local.is_file() else REPO / "reports/strong_c0c1_20261030/baseline_gate.json"
    value = _read(path)
    _require(value.get("schema") == "strong_full_four_view_gate_v1" and value.get("passed") is True,
             "Original strong baseline has not passed")
    _require(os.path.normcase(os.path.abspath(value["baseline_directory"])) == os.path.normcase(os.path.abspath(directory)), "Baseline gate binds another directory")
    _require((value.get("images"),value.get("GT"),value.get("empty")) == SCOPE
             and value.get("base_checkpoint_sha256") == BASE_SHA and value.get("small_checkpoint_sha256") == SMALL_SHA,
             "Baseline gate model/scope differs")
    _source(value.get("source_sha256"),inventory,"baseline gate")
    fixed = assessment["at_confidence_0_75"]
    _same_metric(value.get("fusion"),fixed,"baseline gate")
    _same_metric(value.get("at_recall90"),assessment["at_recall90"],"baseline R90 gate")
    _require((fixed["tp"],fixed["fp"],fixed["fn"]) == (3105,183,344), "Published baseline count has not reproduced")
    artifacts = value.get("artifact_sha256")
    if artifacts is not None:
        _require(isinstance(artifacts,dict) and artifacts, "Baseline artifact SHA inventory is empty")
        normalized = {Path(name).as_posix():digest for name,digest in artifacts.items()}
        required = {"base/original_predictions.json", "base/hflip_predictions.json",
                    "small/original_predictions.json", "small/hflip_predictions.json",
                    "base/prediction_metadata.json", "small/prediction_metadata.json",
                    "fused_predictions.json", "base/mask_metrics_at_recall90.json"}
        _require(required <= set(normalized), "Baseline artifact SHA inventory is incomplete")
        absolute_root = os.path.normcase(os.path.abspath(directory))
        for name,digest in normalized.items():
            target = Path(directory) / name
            _require(not Path(name).is_absolute() and os.path.normcase(os.path.abspath(target)).startswith(absolute_root+os.sep),
                     "Baseline artifact path escaped directory")
            _require(_sha(target) == digest, "Baseline artifact SHA differs: " + name)
    return {"path":str(path),"sha256":_sha(path)}


def plan_experiment_gates(baseline_dir,c0_dir,c1_dir,provenance_path):
    """Permit both matched arms iff one independently qualifies; never promotes deployment."""
    inventory = _inventory()
    contract = _provenance(Path(provenance_path))
    metrics = _metrics()
    cache = {}
    baseline = _assess(baseline_dir,contract,metrics,inventory,cache)
    baseline_gate = _baseline_gate(baseline_dir,baseline,inventory)
    arms = {}
    for name,directory in (("C0",c0_dir),("C1",c1_dir)):
        assessment = _assess(directory,contract,metrics,inventory,cache,baseline)
        reasons = []
        workpoint = assessment["at_recall90"]
        if workpoint is None:
            reasons.append("Full fused VAL cannot reach Recall >= 90%")
        elif workpoint["precision"] + 1e-12 < baseline["at_recall90"]["precision"] + .002:
            reasons.append("Fusion Precision improvement is below 0.2 percentage points")
        if workpoint is not None and workpoint["recall"] < .90:
            reasons.append("Fusion Recall is below 90%")
        if assessment["heavy"] is not None and assessment["heavy"]["recall"] + 1e-12 < baseline["heavy"]["recall"] - .005:
            reasons.append("Fusion heavy-occlusion BOX recall falls by more than 0.5 percentage points")
        for branch in ("visible","amodal"):
            if assessment["base_original_masks"]["metrics"][branch]["f1"] + 1e-12 < baseline["base_original_masks"]["metrics"][branch]["f1"] - .005:
                reasons.append("Base original "+branch+" mask F1 falls by more than 0.5 percentage points")
        arms[name] = {"arm":name,"passed":False,"candidate_eligible":not reasons,
                      "candidate_sha256":assessment["candidate_sha256"],"reasons":reasons,
                      "continuation_role":"stopped","assessment":assessment}
    experiment_continue = any(a["candidate_eligible"] for a in arms.values())
    for arm in arms.values():
        arm["passed"] = experiment_continue
        if experiment_continue:
            arm["continuation_role"] = "eligible_candidate_same_budget" if arm["candidate_eligible"] else "matched_control_only_not_promoted"
    return {"schema":"strong_c0c1_acceptance_plan_v1","experiment_continue":experiment_continue,
            "baseline":baseline,"baseline_gate":baseline_gate,"arms":arms,
            "rule":{"minimum_precision_gain":.002,"minimum_recall":.90,"maximum_heavy_recall_drop":.005,
                    "maximum_Base_original_mask_F1_drop":.005},
            "source_sha256":inventory,"acceptance_source_sha256":_sha(Path(__file__)),
            "original_baseline_remains_protected":True,"deployment_promoted":False}
