"""UNEXECUTED candidate: one forward/image, fixed box-quality powers 0,.5,1.

No Torch/model imports at module scope. Future model execution requires the
six-epoch pinned trainer to have completed/exited, exclusive project locks and
no other model work plus a pinned desktop-identity baseline. Power0 must exactly reproduce saved validation;
any numerical difference rejects comparisons rather than relaxing tolerance.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import random
import re
import subprocess
import sys


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RUN = ROOT/"runs/context_dual_train01_20261021_2012"
ARCHIVES = (ROOT/"reports/context_dual_epoch_archive01_20261021",
            ROOT/"reports/context_dual_epoch_archive02_20261021")
BASELINE = ROOT/"reports/shuju_v2_original_val_single02"
CANDIDATE = ROOT/"temp/aggressive_amodal_candidate_20261021"
SCRIPTS = ROOT/"scripts"
SCHEMA = "apple_v2_context_dual_evidence_mixed_supervision_v1"
POWERS = (0., .5, 1.)
PINNED = {
    ROOT/"temp/context_dual_epoch_archive_20261021/watch_archive.py": "35187f308597f70e3d46515ff6d9c52257b807f213b9b971ed3965f089fff71c",
    ROOT/"temp/context_dual_epoch_archive_20261021/watch_archive_retry.py": "fc9eec5c638653e45540b785a807cffa7240d2f47be80024ed5c462e04e8e46d",
    CANDIDATE/"train_context_dual.py": "03190910ca208c63d691a2609979ff5c2ec1b08159c6f563026fac0fab48acf5",
    CANDIDATE/"context_dual_evidence_model.py": "e87cecf4ebc487e0393a2178ea177c3cf4628d33c01b7e5e805c0b9eaf6ab90b",
    ROOT/"temp/aggressive_data_candidate_20261021/real_mask_occlusion.py": "c4c212de0e22f3b1a11ad116ec564ef4eb80da720756bd00ce80dd4c639f05c7",
    SCRIPTS/"v2_dual_mask_model.py": "d0bed506c0182db2e472a274e322e3fa17b808d2076392b8de98a13b4f535fe8",
    SCRIPTS/"amodal_model.py": "d0d5ed20eed877ffeeea9a4bd3b4f7dc88f872bb1838271b2b59b8fa4609076c",
    SCRIPTS/"dataset.py": "4d8fe0f1ac98e3b54dd4f029675ad240b621766140b259a537e99290ace9a0a0",
    SCRIPTS/"metrics.py": "1b0a7e385940fd0c6160cc08b75ddaf2e1f43f3ea00f5bd37d547fea6b08010d",
    SCRIPTS/"runtime.py": "a21eba633ba1d08cfe07b7695186d811f2e69ac560ce441a236ffce7b93fb64c",
    SCRIPTS/"train_refinement.py": "94dc54a5212b1e825295f1085bf6c6ce89dffaa10b331220afa08c20b573fc8f",
    SCRIPTS/"refinement_config.py": "20b31460a36b93d5816358c1f40169ed081d78577b095eb7033fc59cddecd364",
    SCRIPTS/"atomic_io.py": "3b5ca14efc337ce3f511f60f1829c14c24e5ab6fc9772714e051bf265e645380",
    RUN/"run_config.json": "8823b611cf7bd269cad92ae82427146808923b08e889ca292b9effe19146cc41",
    RUN/"architecture.json": "73617210a542ca56cb2a38bffa3cd9c0ac870c8066cf1a31cfeaf1f0830d9980",
    BASELINE/"evaluation_metadata.json": "dc0e6b4f87ef9565f7a70ee8011bd871f9e8cccf51110098eed403446caeccc7",
    Path("F:/anaconda/envs/falldet/Lib/site-packages/torchvision/models/detection/roi_heads.py"): "0fa89fccc19faa5aaa9dc984a9e3dc29684a4a944dbd5353dd65b82cbef7fc8b",
    Path("F:/anaconda/envs/falldet/Lib/site-packages/torchvision/models/detection/transform.py"): "26ab6a7fdb8ba34b8112b689d690da3014ae00b639fef0c42e72c241c3b0c26d",
    Path("F:/anaconda/envs/falldet/Lib/site-packages/torchvision/models/detection/generalized_rcnn.py"): "ee8bed850c7c6848baa315e8bd2a8eec3ed06a7a76cb37751c75c041b412dd39",
}

WDDM_GUARD = HERE/'wddm_process_guard.py'
WDDM_GUARD_SHA = '67b3d86f751149f45f0989ebd78ab30abd1dda34aa3df597d4c5e6aac5c45ebe'
WDDM_BASELINE = ROOT/'reports/wddm_desktop_gpu_baseline_20261021_01.json'
WDDM_BASELINE_SHA = '0ebeff5fbf560055c6e3f0c0cbfedb3b4a0ee951c91f817e4bdbb249a8318896'
PINNED.update({WDDM_GUARD: WDDM_GUARD_SHA, WDDM_BASELINE: WDDM_BASELINE_SHA})



def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    path = Path(path).resolve()
    if any(re.search(r"(^|[_-])test($|[_-])", item.lower()) for item in path.parts):
        raise ValueError("Test access is forbidden")
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write(path, value):
    path = Path(path)
    temporary = path.with_name(path.name+".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    from atomic_io import replace_with_retry
    replace_with_retry(temporary, path, max_attempts=8)


def check_pins():
    actual = {str(path): sha(path) for path in PINNED}
    for path, expected in PINNED.items():
        if actual[str(path)] != expected:
            raise ValueError(f"Pinned source/config changed: {path}")
    return actual


def metadata(args):
    pins = check_pins()
    checkpoint = args.checkpoint.resolve(strict=True)
    if checkpoint.suffix != ".pt" or not re.fullmatch(r"[0-9a-fA-F]{64}", args.checkpoint_sha256):
        raise ValueError("A complete SHA256 of an own .pt checkpoint is mandatory")
    reference = args.reference.resolve(strict=True)
    if not reference.is_relative_to(RUN/"validation") or not re.fullmatch(r"epoch_\d{3}", reference.name):
        raise ValueError("Reference must be an exact completed epoch in the pinned context run")
    if checkpoint.parent == RUN and checkpoint.name in ("best.pt", "last.pt"):
        archive_manifest = None
    elif any(checkpoint.is_relative_to(root) for root in ARCHIVES) and checkpoint.name == "checkpoint.pt":
        archive_manifest = read(checkpoint.parent/"archive_manifest.json")
        if (archive_manifest.get("schema") != "context_dual_completed_epoch_file_archive_v1" or
                archive_manifest.get("phase") != "complete" or Path(archive_manifest["source_run"]).resolve() != RUN or
                archive_manifest["files"]["checkpoint.pt"]["sha256"] != args.checkpoint_sha256.lower()):
            raise ValueError("Invalid own-context checkpoint archive provenance")
    else:
        raise ValueError("Unknown checkpoint path; only pinned run best/last or its verified epoch archives allowed")
    config = read(RUN/"run_config.json")
    if config["schema"] != SCHEMA or config["quality_power"] != 0. or config["min_size"] != 1024 or config["max_size"] != 1280:
        raise ValueError("Unexpected context run configuration")
    summary, raw, original = read(reference/"summary.json"), read(reference/"raw_predictions.json"), read(BASELINE/"raw_predictions.json")
    expected = {str(r["image_id"]): r for r in original}
    if len(raw) != 120 or len(expected) != 120 or len({r["image_id"] for r in raw}) != 120 or sum(len(r["gt"]["boxes"]) for r in raw) != 1571:
        raise ValueError("Complete original 120-image/1571-GT validation required")
    if {r["image_id"]: r["gt"]["boxes"] for r in raw} != {key: r["gt"]["boxes"] for key, r in expected.items()}:
        raise ValueError("Reference GT differs from original baseline coordinates/order")
    policy = summary["policy"]
    required = {"split": "val", "images": 120, "gt": 1571, "input": "external_max1280_BILINEAR_then_model",
                "nms_iou": .5, "raw_score": .001, "detections_per_img": 100, "quality_power": 0.,
                "mask_GT_used": False, "test_used": False}
    if policy != required or summary["summary"]["images"] != 120 or summary["summary"]["gt_count"] != 1571:
        raise ValueError("Reference evaluation policy differs from frozen031 evaluator")
    snapshot = {**pins, str(reference/"summary.json"): sha(reference/"summary.json"),
                str(reference/"raw_predictions.json"): sha(reference/"raw_predictions.json"),
                str(BASELINE/"raw_predictions.json"): sha(BASELINE/"raw_predictions.json"),
                str(Path(__file__)): sha(__file__)}
    if archive_manifest is not None:
        snapshot[str(checkpoint.parent/"archive_manifest.json")] = sha(checkpoint.parent/"archive_manifest.json")
    return checkpoint, reference, config, summary, raw, expected, snapshot


def is_live(psutil, pid, create_time=None):
    try:
        process = psutil.Process(int(pid))
        return process.is_running() and (create_time is None or abs(process.create_time()-float(create_time)) < .001)
    except psutil.NoSuchProcess:
        return False


def no_other_work(psutil):
    if platform.node().upper() != "YAN" or Path(sys.executable).resolve() != Path("F:/anaconda/envs/falldet/python.exe").resolve():
        raise RuntimeError("Only authorized YAN host and pinned F: falldet Python are allowed")
    # Require actual completion, not just disappearance of the old GPU process.
    state = read(RUN/"status.json")
    if state.get("phase") != "complete" or state.get("epoch") != 6:
        raise RuntimeError("Pinned six-epoch context trainer has not completed")
    if is_live(psutil, 53748, 1792584803.285806):
        raise RuntimeError("Pinned trainer PID is still alive")
    if is_live(psutil, 12480, 1792584800.7768795):
        raise RuntimeError("Pinned original training supervisor PID is still alive")
    current = ROOT/"reports/current_job.json"
    if current.is_file():
        job = read(current)
        child = job.get("child") or {}
        if child.get("pid") != os.getpid() and child.get("pid") and is_live(psutil, child["pid"], child.get("create_time")):
            raise RuntimeError("An existing supervised compute child is still alive")
    for process in psutil.process_iter(["pid", "name", "cmdline"]):
        if process.pid == os.getpid():
            continue
        try:
            command = process.info["cmdline"] or []
            scripts = [Path(item).name.lower() for item in command if item.lower().endswith(".py")]
            if any(re.search(r"(^|_)(train|training|eval|evaluate|inference|predict|smoke|verify)(_|\.)", name) for name in scripts):
                raise RuntimeError(f"Another potentially GPU-using script is alive: PID {process.pid}, {scripts}")
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            if (process.info.get("name") or "").lower().startswith("python"):
                raise RuntimeError("Cannot inspect another Python process; concurrency check fails closed")
    # WDDM C+G includes normal desktop activity on this inspected host.
    # Keep every original model/PID/command check above, but accept only exact
    # identities in the immutable root-reviewed desktop baseline.
    if sha(WDDM_GUARD) != WDDM_GUARD_SHA:
        raise RuntimeError("WDDM guard source changed")
    spec = importlib.util.spec_from_file_location("fixed_quality_wddm_identity_gate", WDDM_GUARD)
    guard = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(guard)
    inventory = guard.inspect_desktop_baseline(psutil, WDDM_BASELINE, WDDM_BASELINE_SHA)
    if inventory["known_processes_exited_after_snapshot"]:
        for process in psutil.process_iter(["pid", "name", "cmdline"]):
            if process.pid == os.getpid():
                continue
            try:
                command = process.info["cmdline"] or []
                scripts = [Path(item).name.lower() for item in command if item.lower().endswith(".py")]
                if any(re.search(r"(^|_)(train|training|eval|evaluate|inference|predict|smoke|verify)(_|\.)", name) for name in scripts):
                    raise RuntimeError(f"Another potentially GPU-using script is alive: PID {process.pid}, {scripts}")
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                if (process.info.get("name") or "").lower().startswith("python"):
                    raise RuntimeError("Cannot inspect another Python process; concurrency check fails closed")
        inventory["original_model_command_rescan_after_snapshot_exit"] = True
    else:
        inventory["original_model_command_rescan_after_snapshot_exit"] = False
    return {"trainer_complete_epoch": 6, "trainer_PID_exited": True,
            "original_training_supervisor_PID_exited": True,
            "GPU_inventory_policy": "no_other_model_work_plus_exact_reviewed_desktop_baseline",
            "GPU_idle_claim": False, "WDDM_inventory": inventory}



class Capture:
    """Module hooks only; never replace or edit frozen postprocess/forward."""
    def __init__(self, roi):
        self.values, self.handles = {}, []
        self.handles.append(roi.register_forward_pre_hook(self.roi_input))
        for name in ("box_predictor", "refinement_classifier", "refinement_box", "box_quality"):
            module = getattr(roi, name)
            self.handles.append(module.register_forward_hook(self.output_hook(name)))

    def roi_input(self, module, args):
        if "roi_input" in self.values:
            raise RuntimeError("RoIHeads unexpectedly called more than once for an image")
        if len(args) > 3 and args[3] is not None:
            raise RuntimeError("GT must never enter the prediction forward")
        self.values["roi_input"] = ([proposal.detach().clone() for proposal in args[1]], list(args[2]))

    def output_hook(self, name):
        def hook(module, args, result):
            if name in self.values:
                raise RuntimeError(f"Module called multiple times: {name}")
            self.values[name] = tuple(x.detach().clone() for x in result) if isinstance(result, tuple) else result.detach().clone()
        return hook

    def clear(self):
        self.values.clear()

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.clear()


def postprocess_power(torch, model, captured, power, input_hw):
    if power not in POWERS:
        raise ValueError("Only preregistered powers 0,.5,1 are allowed")
    proposals, image_shapes = captured["roi_input"]
    base_logits, base_regression = captured["box_predictor"]
    # The additions retain the captured AMP dtype, exactly like the frozen core.
    final_logits = (base_logits+captured["refinement_classifier"]).float()
    final_regression = (base_regression+captured["refinement_box"]).float()
    quality = captured["box_quality"].flatten().float()
    if not all(bool(torch.isfinite(value).all()) for value in (final_logits, final_regression, quality)):
        raise ValueError("Nonfinite captured postprocess inputs")
    if power > 0:
        probability = final_logits.softmax(dim=-1)[:, 1]
        adjusted = (probability*quality.sigmoid().pow(power)).clamp(1e-7, 1-1e-7)
        final_logits = torch.stack((torch.log1p(-adjusted), adjusted.log()), dim=1)
    boxes, scores, labels = model.roi_heads.postprocess_detections(final_logits, final_regression, proposals, image_shapes)
    result = [{"boxes": b, "scores": s, "labels": label} for b, s, label in zip(boxes, scores, labels)]
    return model.transform.postprocess(result, image_shapes, [input_hw])[0]


def raw_record(np, output, sample, input_wh):
    width, height = sample["width"], sample["height"]
    boxes = output["boxes"].detach().cpu().numpy().astype(np.float64)
    boxes[:, [0, 2]] *= width/input_wh[0]
    boxes[:, [1, 3]] *= height/input_wh[1]
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, width)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, height)
    return {"image_id": sample["filename"], "gt": {"boxes": sample["boxes"].tolist()},
            "pred": {"boxes": boxes.tolist(), "scores": output["scores"].detach().cpu().tolist()}}


def mismatch_detail(np, actual, expected):
    result = {"exact_match": actual == expected, "image_id": actual["image_id"]}
    for key in ("boxes", "scores"):
        a, b = np.asarray(actual["pred"][key], dtype=np.float64), np.asarray(expected["pred"][key], dtype=np.float64)
        result[key] = {"actual_shape": list(a.shape), "expected_shape": list(b.shape)}
        if a.shape == b.shape:
            result[key].update(unequal_elements=int(np.count_nonzero(a != b)), max_absolute_difference=float(np.max(np.abs(a-b))) if a.size else 0.)
    return result


def goal_frontier(calibration):
    curve = calibration["curve"]
    best = lambda points, key: max(points, key=key) if points else None
    key = lambda p: (p["min_precision_recall"], p["f1"], p["confidence"])
    p98 = [p for p in curve if p["precision"] >= .98]
    r95 = [p for p in curve if p["recall"] >= .95]
    return {"P95_R95_exists": any(p["precision"] >= .95 and p["recall"] >= .95 for p in curve),
            "P98_R95_exists": any(p["precision"] >= .98 and p["recall"] >= .95 for p in curve),
            "maxmin": best(curve, key),
            "max_R_when_P98": best(p98, lambda p: (p["recall"], p["precision"], p["confidence"])),
            "max_P_when_R95": best(r95, lambda p: (p["precision"], p["recall"], p["confidence"])),
            "scope": "Saved candidates after fixed NMS/cap/.001, not whole-model upper bound"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("smoke", "full"))
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--smoke-report", type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(HERE) or output.exists():
        parser.error("Use a new output directory inside this independent candidate directory")
    # psutil/locks are CPU only; model/Torch imports occur strictly below guards.
    import psutil
    guard = no_other_work(psutil)
    checkpoint, reference, config, saved_summary, saved_raw, original, snapshot = metadata(args)
    sys.path[:0] = [str(CANDIDATE), str(SCRIPTS)]
    from train_refinement import TrainingLock
    with TrainingLock(SCRIPTS/"training.lock", output, psutil):
        guard = no_other_work(psutil)
        actual_sha = sha(checkpoint)
        if actual_sha != args.checkpoint_sha256.lower():
            raise ValueError("Checkpoint SHA mismatch")
        snapshot[str(checkpoint)] = actual_sha
        identity = {"checkpoint_sha256": actual_sha, "reference": str(reference),
                    "reference_summary_sha256": snapshot[str(reference/"summary.json")],
                    "reference_raw_sha256": snapshot[str(reference/"raw_predictions.json")],
                    "script_sha256": sha(__file__), "powers": list(POWERS), "source_pins": check_pins()}
        if args.mode == "full":
            if args.smoke_report is None:
                raise ValueError("Full comparison requires this exact candidate's successful first-image smoke")
            smoke = read(args.smoke_report)
            if smoke.get("phase") != "smoke_passed" or smoke.get("identity") != identity or not smoke.get("power0_bitexact"):
                raise ValueError("Smoke report does not match exact checkpoint/reference/code")
        output.mkdir(parents=True, exist_ok=False)
        report = {"phase": "initializing", "model_execution_started": False, "identity": identity, "guard": guard,
                  "mode": args.mode, "test_read": False, "mask_evaluation": False, "source_sha256_before": snapshot}
        write(output/"report.json", report)
        try:
            from runtime import authorized_guard
            torch = authorized_guard()
            report["torch_imported"] = True
            import numpy as np
            from PIL import Image
            from metrics import calibrate_threshold, evaluate_at_threshold
            from train_context_dual import BoxOnlyDataset
            from context_dual_evidence_model import build_model, checkpoint_model_config
            random.seed(config["seed"])
            np.random.seed(config["seed"])
            torch.manual_seed(config["seed"])
            torch.cuda.manual_seed_all(config["seed"])
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            torch.set_num_threads(4)
            data = torch.load(checkpoint, map_location="cpu", weights_only=False)
            if sha(checkpoint) != actual_sha:
                raise ValueError("Checkpoint changed during load")
            expected_epoch = int(reference.name.split("_")[1])
            if (data.get("architecture_metadata", {}).get("schema") != SCHEMA or data.get("run_config") != config or
                    data.get("epoch") != expected_epoch or data.get("validation") != saved_summary):
                raise ValueError("Checkpoint internal epoch/schema/config/validation does not match reference")
            if json.loads(json.dumps(data["architecture_metadata"])) != read(RUN/"architecture.json"):
                raise ValueError("Checkpoint architecture/provenance differs from pinned run metadata")
            model_config = data["model_config"]
            if (model_config.get("pretrained") is not False or model_config.get("min_size") != 1024 or
                    model_config.get("max_size") != 1280 or model_config.get("detections_per_img") != 100 or
                    model_config.get("box_quality_power") != 0. or model_config.get("guidance_mode") != "dual_mask"):
                raise ValueError("Unknown or altered model configuration")
            model = build_model(**model_config)
            if checkpoint_model_config(model) != model_config:
                raise ValueError("Builder configuration round-trip failed")
            if data["architecture_metadata"].get("candidate") != model.architecture_metadata["candidate"]:
                raise ValueError("Checkpoint architecture candidate settings mismatch")
            model.load_state_dict(data["state_dict"], strict=True)
            del data
            model = model.to("cuda:0").eval()
            report["model_loaded"] = True
            write(output/"report.json", report)
            model.roi_heads.return_masks = False
            if model.roi_heads.bypass_refinement or model.roi_heads.box_quality_power != 0.:
                raise ValueError("Unexpected core inference mode")
            if (model.roi_heads.nms_thresh, model.roi_heads.score_thresh, model.roi_heads.detections_per_img) != (.5, .001, 100):
                raise ValueError("NMS/raw score/cap differ from reference")
            dataset = BoxOnlyDataset(Path(config["shuju_root"]), "val")
            if len(dataset) != 120 or sum(len(s["boxes"]) for s in dataset.samples) != 1571:
                raise ValueError("Original VAL120/1571 changed")
            frozen = read(BASELINE/"evaluation_metadata.json")["frozen_inputs_sha256"]
            for sample in dataset.samples:
                if sample["filename"] not in original or sample["boxes"].tolist() != original[sample["filename"]]["gt"]["boxes"]:
                    raise ValueError("Current VAL GT/order differs from original baseline")
                for path in (sample["path"], sample["label"]):
                    digest = sha(path)
                    if frozen.get(str(path)) != digest:
                        raise ValueError(f"VAL image/label bytes differ from frozen baseline: {path}")
                    snapshot[str(path)] = digest
            # Verify reference metric arithmetic before running any image.
            reference_check = calibrate_threshold(saved_raw)
            if reference_check["summary"] != saved_summary["summary"] or reference_check["confidence"] != saved_summary["confidence"]:
                raise ValueError("Reference raw and saved summary/calibration disagree")
            expected_by_id = {r["image_id"]: r for r in saved_raw}
            records = {power: [] for power in POWERS}
            capture = Capture(model.roi_heads)
            report.update(phase="running", model_execution_started=True, checkpoint_epoch=expected_epoch,
                          model_config=model_config, images_planned=1 if args.mode == "smoke" else 120,
                          bitexact_images=0, source_sha256_before=snapshot)
            write(output/"report.json", report)
            try:
                for index, sample in enumerate(dataset.samples[:report["images_planned"]]):
                    capture.clear()
                    with Image.open(sample["path"]) as source:
                        rgb = source.convert("RGB")
                    width, height = rgb.size
                    scale = min(1., 1280/max(width, height))
                    input_wh = (max(1, round(width*scale)), max(1, round(height*scale)))
                    if rgb.size != input_wh:
                        rgb = rgb.resize(input_wh, Image.Resampling.BILINEAR)
                    array = np.asarray(rgb, dtype=np.uint8).transpose(2, 0, 1).copy()
                    image = torch.from_numpy(array).to(device="cuda:0", dtype=torch.float32).div_(255)
                    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
                        direct = model([image])[0]  # Exactly ONE network forward for all powers.
                        zero = postprocess_power(torch, model, capture.values, 0., tuple(image.shape[-2:]))
                        for key in ("boxes", "scores", "labels"):
                            if not torch.equal(direct[key], zero[key]):
                                report["mismatch"] = {"image_id": sample["filename"], "stage": "hook_reconstruction", "field": key}
                                raise ValueError("Power0 reconstructed postprocess differs from direct model output; no tolerance relaxation")
                        record = raw_record(np, zero, sample, input_wh)
                        if record != expected_by_id[sample["filename"]]:
                            report["mismatch"] = mismatch_detail(np, record, expected_by_id[sample["filename"]])
                            raise ValueError("Power0 is not bitexact to stored raw validation; formal comparison refused")
                        records[0.].append(record)
                        for power in POWERS[1:]:
                            rescored = postprocess_power(torch, model, capture.values, power, tuple(image.shape[-2:]))
                            records[power].append(raw_record(np, rescored, sample, input_wh))
                            del rescored
                    report["bitexact_images"] = index+1
                    write(output/"progress.json", {"image": index+1, "total": report["images_planned"], "power0_bitexact": True})
                    print(json.dumps({"event": "quality_fixed_power_progress", "image": index+1, "total": report["images_planned"]}), flush=True)
                    del image, array, rgb, direct, zero
                    capture.clear()
            finally:
                capture.close()
            after = {path: sha(Path(path)) for path in snapshot}
            if snapshot != after:
                raise ValueError("Source/checkpoint/VAL bytes changed during evaluation")
            report.update(power0_bitexact=True, source_sha256_after=after,
                          source_before_after_identical=True, mask_performance_claim=False)
            if args.mode == "smoke":
                report.update(phase="smoke_passed", comparison_permitted=False,
                              scope="first_original_VAL_image_only; no quality metric or achievement claim")
            else:
                results = {}
                reference_confidence = saved_summary["confidence"]
                for power in POWERS:
                    if len(records[power]) != 120 or sum(len(r["gt"]["boxes"]) for r in records[power]) != 1571:
                        raise ValueError("Full120/1571 scope not retained")
                    calibrated = calibrate_threshold(records[power], return_curve=True)
                    if power == 0. and (records[power] != saved_raw or calibrated["summary"] != saved_summary["summary"]):
                        raise ValueError("Full power0 raw/summary not exactly reproduced")
                    results[str(power)] = {"validation_selected_global": calibrated["summary"],
                                           "at_original_power0_confidence": evaluate_at_threshold(records[power], reference_confidence),
                                           "goals": goal_frontier(calibrated)}
                    sub = output/f"power_{power:g}"
                    sub.mkdir()
                    write(sub/"raw_predictions.json", records[power])
                    write(sub/"summary.json", results[str(power)])
                    write(sub/"threshold_curve.json", calibrated["curve"])
                report.update(phase="complete_fullval", scope="fullval_120_images_1571_GT", results=results,
                              interpretation="Fixed three-power validation ablation using identical proposals and regressed boxes; score changes can alter thresholding/NMS/cap. Not independent test evidence or mask quality.")
            report["completed_utc"] = datetime.now(timezone.utc).isoformat()
            write(output/"report.json", report)
            return 0
        except BaseException as error:
            report.update(phase="refused_or_failed", comparison_permitted=False, error=repr(error))
            write(output/"report.json", report)
            raise


if __name__ == "__main__":
    raise SystemExit(main())
