"""Read-only checkpoint smoke/latency replay, never a camera or accuracy test.

Uses the project's exact strict loader and native-mask predictor. A new output
directory is mandatory. The original repository, checkpoints and data are read
only; private scratch and reports are placed inside the chosen output directory.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
import time


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=5)
    args = parser.parse_args()
    if not 1 <= args.rounds <= 10 or not 1 <= args.warmup <= 20:
        parser.error("Bounded diagnostic: rounds 1..10, warmup 1..20")
    repo, output, weights = args.repo.resolve(), args.output.resolve(), args.weights.resolve()
    if output == repo or repo in output.parents:
        parser.error("Choose an isolated output directory outside the original repository")
    output.mkdir(parents=True, exist_ok=False)
    scratch = output / "scratch"
    scratch.mkdir()
    for key in ("YOLO_CONFIG_DIR", "TORCH_HOME", "HF_HOME", "XDG_CACHE_HOME", "CUDA_CACHE_PATH", "MPLCONFIGDIR", "TEMP", "TMP"):
        os.environ[key] = str(scratch)
    os.environ.update(PYTHONDONTWRITEBYTECODE="1", HF_HUB_OFFLINE="1", YOLO_AUTOINSTALL="false", YOLO_OFFLINE="true")
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(repo / "src"))
    import cv2
    import numpy as np
    import torch
    from apple_local_rgbd.local_evaluation import make_predictor, predict_native, read_bgr

    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    torch.backends.cudnn.benchmark = False
    if not torch.cuda.is_available():
        raise RuntimeError("This diagnostic requires the verified CUDA runtime")
    manifest_path = repo / "cache/datasets/local_apple_v1/val_manifest.json"
    sample_path = repo / "configs/premask_audit_samples_20261102.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    if manifest["split"] != "val" or manifest["role"] != "development":
        raise ValueError("Only the existing development VAL is allowed; no test access")
    sample_ids = [r["image_id"] for r in json.loads(sample_path.read_text(encoding="utf-8-sig"))["samples"][:8]]
    lookup = {r["image_id"]: r for r in manifest["samples"]}
    rows = [lookup[x] for x in sample_ids]
    images = [read_bgr(r) for r in rows]  # SHA checks outside the timed region.
    weight_sha = sha(weights)
    predictor, apple = make_predictor(weights=weights, kind="local", imgsz=640,
                                     conf=.25, max_det=64, output=output)
    if not predictor.model.fp16 or apple != 0:
        raise RuntimeError("Expected FP16 and exact one-class apple mapping")

    def decode(row):
        bgr = cv2.imdecode(np.fromfile(row["rgb_path"], dtype=np.uint8), cv2.IMREAD_COLOR)
        if bgr is None or list(bgr.shape[:2]) != row["original_hw"]:
            raise ValueError("Failed native-image decode")
        return bgr

    def signature(boxes, masks):
        h = hashlib.sha256()
        for array in (boxes, masks):
            h.update(str((array.shape, str(array.dtype))).encode("ascii"))
            h.update(array.tobytes())
        return h.hexdigest()

    def stats(values):
        a = np.asarray(values, dtype=float)
        return dict(mean_ms=float(a.mean()), p50_ms=float(np.percentile(a, 50)),
                    p95_ms=float(np.percentile(a, 95)))

    baseline = {}
    cases = []
    for mode in ("predecoded_native_output", "disk_decode_native_output_json"):
        for i in range(args.warmup):
            predict_native(predictor, images[i % len(images)])
        frame_rows = []
        torch.cuda.synchronize()
        wall_start = time.perf_counter()
        for repeat in range(args.rounds):
            for row, cached in zip(rows, images):
                started = time.perf_counter()
                bgr = cached if mode.startswith("predecoded") else decode(row)
                decoded = time.perf_counter()
                boxes, masks, inner_ms = predict_native(predictor, bgr)
                result_at = time.perf_counter()
                payload = None
                if not mode.startswith("predecoded"):
                    # Native masks remain available in CPU memory; JSON carries boxes
                    # and shape only. No display, image/video/mask writing or network.
                    payload = json.dumps({"image_id": row["image_id"], "boxes": boxes.tolist(),
                                          "mask_shape": list(masks.shape)}, allow_nan=False)
                published = time.perf_counter()
                current = signature(boxes, masks)  # Excluded from per-frame latency.
                baseline.setdefault(row["image_id"], current)
                if current != baseline[row["image_id"]]:
                    raise RuntimeError("Prediction differs across repetitions or replay modes")
                frame_rows.append(dict(round=repeat + 1, image_id=row["image_id"],
                    decode_ms=(decoded-started)*1000, native_predict_call_ms=(result_at-decoded)*1000,
                    original_inner_ms=inner_ms, serialization_ms=(published-result_at)*1000,
                    service_ms=(published-started)*1000, instances=len(boxes),
                    payload_bytes=len(payload.encode("utf-8")) if payload else 0,
                    prediction_sha256=current))
        wall_seconds = time.perf_counter() - wall_start
        columns = {k: stats([r[k] for r in frame_rows]) for k in
                   ("decode_ms", "native_predict_call_ms", "original_inner_ms", "serialization_ms", "service_ms")}
        cases.append(dict(mode=mode, frames=frame_rows, summary=columns,
            active_service_fps=1000/columns["service_ms"]["mean_ms"],
            replay_wall_throughput_fps=len(frame_rows)/wall_seconds,
            replay_wall_seconds=wall_seconds,
            wall_includes_audit_hashes=True, camera_to_publish_ms=None))
    result = dict(schema="apple_bounded_rgb_replay_v1", created_utc=datetime.now(timezone.utc).isoformat(),
        scope="8 fixed development RGB images, repeated; not camera, independent test, accuracy or RGBD benchmark",
        weights=str(weights), weights_sha256=weight_sha, weights_unchanged=sha(weights)==weight_sha,
        manifest_sha256=sha(manifest_path), sample_manifest_sha256=sha(sample_path),
        source_sha256={str(p): sha(p) for p in [Path(__file__), repo/"src/apple_local_rgbd/local_training.py",
                                             repo/"src/apple_local_rgbd/local_evaluation.py"]},
        environment=dict(python=sys.version, platform=platform.platform(), gpu=torch.cuda.get_device_name(),
            gpu_total_bytes=torch.cuda.get_device_properties(0).total_memory, torch_threads=torch.get_num_threads(),
            cv2_threads=cv2.getNumThreads(), versions={k: importlib.metadata.version(k)
            for k in ("torch", "torchvision", "ultralytics", "numpy", "opencv-python", "scipy")}),
        config=dict(imgsz=640, conf=.25, max_det=64, batch=1, precision="FP16", retina_masks=True,
                    warmup_per_mode=args.warmup, rounds_per_mode=args.rounds), cases=cases,
        output_invariance="All replay repetitions/modes have byte-identical boxes and masks; not compared with previous hardware runs",
        training_performed=False, test_split_used=False, camera_opened=False,
        includes=["preprocess", "GPU inference", "native masks and boxes to CPU", "CUDA synchronization"],
        excludes=["camera exposure/acquisition/queue", "depth", "3D geometry", "association", "classification", "display", "robot communication"],
        end_to_end_camera_fps=None, real_4fps_bottleneck_identified=False)
    (output/"replay_report.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"report": str(output/"replay_report.json"), "weights_unchanged": result["weights_unchanged"],
                      "cases": [{k:v for k,v in c.items() if k != "frames"} for c in cases]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
