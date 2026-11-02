"""CPU-only numeric audit of the actual legacy RGB preprocessing contract.

No model is loaded; CUDA is hidden and its initialization is checked. These
results establish neither GPU timing nor GPU bitwise or detection equivalence.
"""
from datetime import datetime, timezone
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
for key, sub in (("TEMP", "tmp"), ("TMP", "tmp"), ("TORCH_HOME", "cache/torch"),
                 ("CUDA_CACHE_PATH", "cache/cuda"), ("XDG_CACHE_HOME", "cache")):
    os.environ[key] = str(REPO / sub)
sys.path[:0] = [str(REPO), str(REPO / "strong_runtime/src")]

import numpy as np
from PIL import Image, __version__ as PIL_VERSION
import torch
import torch.nn.functional as F
import torchvision
from apple_seg.od_data import legacy_rgb_tensor, NORMALIZATION_MEAN, NORMALIZATION_STD


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def metrics(reference, candidate, mean, std):
    assert reference.device.type == candidate.device.type == "cpu"
    delta = (candidate - reference).abs()
    pixel_delta = delta * std * 255
    reference_u8 = ((reference * std + mean) * 255).round().clamp(0, 255).to(torch.int16)
    candidate_u8 = ((candidate * std + mean) * 255).round().clamp(0, 255).to(torch.int16)
    quantized_delta = (candidate_u8 - reference_u8).abs()
    return {
        "channel_values": reference.numel(), "rgb_pixels": reference.shape[-2] * reference.shape[-1],
        "normalized_max_abs": delta.max().item(), "normalized_mean_abs": delta.double().mean().item(),
        "normalized_nonzero_values": torch.count_nonzero(delta).item(),
        "normalized_abs_gt_1e_6_values": torch.count_nonzero(delta > 1e-6).item(),
        "equivalent_0_255_max_abs": pixel_delta.max().item(),
        "equivalent_0_255_mean_abs": pixel_delta.double().mean().item(),
        "equivalent_0_255_abs_gt_1_values": torch.count_nonzero(pixel_delta > 1).item(),
        "round_clamp_u8_max_abs": quantized_delta.max().item(),
        "round_clamp_u8_changed_channel_values": torch.count_nonzero(quantized_delta).item(),
        "round_clamp_u8_changed_rgb_pixels": torch.count_nonzero(quantized_delta.any(0)).item(),
    }


def aggregate(rows):
    count = sum(row["channel_values"] for row in rows)
    pixels = sum(row["rgb_pixels"] for row in rows)
    result = {"channel_values": count, "rgb_pixels": pixels}
    for key in rows[0]:
        if key in result:
            continue
        if "max_abs" in key:
            result[key] = max(row[key] for row in rows)
        elif "mean_abs" in key:
            result[key] = sum(row[key] * row["channel_values"] for row in rows) / count
        else:
            result[key] = sum(row[key] for row in rows)
    result["round_clamp_u8_changed_rgb_pixel_percent"] = 100 * result["round_clamp_u8_changed_rgb_pixels"] / pixels
    result["normalized_nonzero_channel_percent"] = 100 * result["normalized_nonzero_values"] / count
    return result


def main():
    assert REPO.drive.upper() == "F:"
    assert not torch.cuda.is_initialized()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    config_path = REPO / "configs/strong_baseline_4060.json"
    manifest_path = REPO / "configs/premask_audit_samples_20261102.json"
    benchmark_path = REPO / "reports/premask_20261102/benchmark1024/benchmark_results.json"
    output = REPO / "reports/premask_stage2_20261102/preprocess_cpu.json"
    config = json.loads(config_path.read_text(encoding="utf-8-sig"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    benchmark = json.loads(benchmark_path.read_text(encoding="utf-8-sig"))
    root = (config_path.parent / config["path_base"]).resolve()
    papple = (root / config["paths"]["papple_root"]).resolve()
    sample_rows = manifest["samples"][:8]
    assert [r["image_id"] for r in sample_rows] == [r["image_id"] for r in benchmark["samples"]]
    installed_sources = [Path(importlib.import_module(name).__file__) for name in
        ("torchvision.ops.roi_align", "torchvision.ops.poolers", "torchvision._autograd_registrations")]
    files = [Path(__file__), config_path, manifest_path, benchmark_path,
             REPO / "strong_runtime/src/apple_seg/od_data.py", REPO / "deployment/strong_fast.py",
             REPO / "deployment/pre_mask.py", REPO / "scripts/benchmark_premask.py", *installed_sources]
    hashes = {str(path): sha(path) for path in files}
    result = {"schema": "premask_stage2_cpu_preprocess_numeric_audit_v1",
              "created_utc": datetime.now(timezone.utc).isoformat(),
              "device": "CPU only", "torch": torch.__version__, "torchvision": torchvision.__version__,
              "pillow": PIL_VERSION, "python": sys.version, "threads": torch.get_num_threads(),
              "torch_path": torch.__file__, "torchvision_path": torchvision.__file__,
              "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
              "cuda_initialized_before": torch.cuda.is_initialized(),
              "runtime_defaults": {"cudnn_benchmark": torch.backends.cudnn.benchmark,
                                   "cudnn_deterministic": torch.backends.cudnn.deterministic,
                                   "deterministic_algorithms": torch.are_deterministic_algorithms_enabled()},
              "short_side": 1024, "max_size": 1280, "image_count": 8,
              "ground_truth_loaded": False, "model_loaded": False,
              "reference": "Actual legacy_rgb_tensor: uint8->float32/255->normalize->Torch bilinear scale_factor,recompute=True,align_corners=False,antialias=False",
              "PIL_is_not_the_reference": True,
              "gpu_timing_claim": None, "gpu_bitwise_equivalence_claim": None, "detection_accuracy_claim": None,
              "metric_note": "Normalized abs differences; pixel-equivalent units use std*255. Rounded/clamped uint8 comparisons are diagnostic and are not input to the model.",
              "source_sha256": hashes, "samples": [], "comparisons": {},
              "current_benchmark_stages": {key: {name: value[name] for name in
                  ("summary", "synchronized_ms_mean", "roi_synchronized_ms_mean", "mask_candidates")}
                  for key, value in benchmark["cases"].items()},
              "roi_align_dispatch_table": torch._C._dispatch_dump_table("torchvision::roi_align")}
    mean = torch.tensor(NORMALIZATION_MEAN, dtype=torch.float32)[:, None, None]
    std = torch.tensor(NORMALIZATION_STD, dtype=torch.float32)[:, None, None]
    with torch.inference_mode():
        for row in sample_rows:
            path = papple / "images/val" / row["image_id"]
            assert path.resolve().drive.upper() == "F:"
            with Image.open(path) as source:
                rgb = source.convert("RGB")
            sample = {"image_id": row["image_id"], "RGB_path": str(path), "RGB_sha256": sha(path),
                      "native_hw": [rgb.height, rgb.width], "comparisons": {}}
            expected = next(item for item in benchmark["samples"] if item["image_id"] == row["image_id"])
            assert sample["RGB_sha256"] == expected["RGB_sha256"]
            reference = legacy_rgb_tensor(rgb, 1024, 1280)
            sample["output_hw"] = list(reference.shape[-2:])
            raw = torch.from_numpy(np.ascontiguousarray(np.asarray(rgb).transpose(2, 0, 1))).float().div_(255.)
            normalized = (raw - mean) / std
            scale = min(1024 / min(rgb.height, rgb.width), 1280 / max(rgb.height, rgb.width))
            def interpolate(value, antialias):
                return F.interpolate(value[None], scale_factor=scale, mode="bilinear", align_corners=False,
                                     recompute_scale_factor=True, antialias=antialias)[0].contiguous()
            pil_resized = rgb.resize((reference.shape[2], reference.shape[1]), Image.Resampling.BILINEAR)
            pil_raw = torch.from_numpy(np.ascontiguousarray(np.asarray(pil_resized).transpose(2, 0, 1))).float().div_(255.)
            variants = {
                "torch_normalize_first_antialias_false": interpolate(normalized, False),
                "torch_normalize_first_antialias_true": interpolate(normalized, True),
                "torch_resize_first_antialias_false": (interpolate(raw, False) - mean) / std,
                "torch_resize_first_antialias_true": (interpolate(raw, True) - mean) / std,
                "PIL_uint8_bilinear_then_normalize": (pil_raw - mean) / std,
            }
            for name, candidate in variants.items():
                sample["comparisons"]["legacy_vs_" + name] = metrics(reference, candidate, mean, std)
            pil_normalized = variants["PIL_uint8_bilinear_then_normalize"]
            for name in ("torch_resize_first_antialias_false", "torch_resize_first_antialias_true"):
                sample["comparisons"]["PIL_vs_" + name] = metrics(pil_normalized, variants[name], mean, std)
            result["samples"].append(sample)
            print("CPU_PREPROCESS " + row["image_id"], flush=True)
    for key in result["samples"][0]["comparisons"]:
        result["comparisons"][key] = aggregate([row["comparisons"][key] for row in result["samples"]])
    assert {str(path): sha(path) for path in files} == hashes
    assert all(sha(row["RGB_path"]) == row["RGB_sha256"] for row in result["samples"])
    assert not torch.cuda.is_initialized()
    result.update(cuda_initialized_after=False, source_and_images_unchanged=True,
                  completed_utc=datetime.now(timezone.utc).isoformat())
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps(result["comparisons"], ensure_ascii=False, indent=2))
    print("SAVED " + str(output), flush=True)


if __name__ == "__main__":
    main()
