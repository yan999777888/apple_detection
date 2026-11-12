"""Synchronized YOLO whole-frame inference pipeline timing, without GT labels.

Scope: predecoded BGR frame -> predict preprocessing/H2D/forward/postprocessing
-> all boxes and segmentation masks materialized as independent CPU NumPy arrays.
Excludes camera/decode/tracking/fusion/robot I/O; not end-to-end robot FPS.
Multiple checkpoints run serially on the same whole frame with outputs unfused:
the result is joint computation cost, not validated perception quality.
No torch/Ultralytics import or GPU operation occurs during import/check-only.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import statistics
import sys
import time

sys.dont_write_bytecode = True
EXPECTED_ULTRALYTICS = "8.4.115"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, document):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(document, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def reject_test(path):
    for part in Path(path).parts:
        tokens = re.split(r"[^a-z0-9]+", part.casefold())
        require(not {"test", "tests", "testing"} & set(tokens),
                f"Test-labelled paths are forbidden: {path}")


def f_path(value, base=None):
    path = Path(value)
    if not path.is_absolute():
        path = (base or Path.cwd())/path
    path = path.resolve()
    require(path.drive.upper() == "F:", f"All inputs and outputs must remain on F: {path}")
    reject_test(path)
    return path


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--weights", type=Path, nargs="+", required=True,
                   help="Existing local .pt detection/segmentation checkpoints; all run serially per frame")
    p.add_argument("--data", type=Path, required=True, help="Dataset YAML; only its val entry is read")
    p.add_argument("--output", type=Path, required=True, help="Fresh output directory under this F-drive project")
    p.add_argument("--num-frames", type=int, default=40)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--warmup", type=int, default=5, help="Unmeasured full serial pipelines before timing")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=.25)
    p.add_argument("--iou", type=float, default=.7)
    p.add_argument("--max-det", type=int, default=300)
    p.add_argument("--device", default="0", help="One CUDA ordinal, e.g. 0 or cuda:0")
    p.add_argument("--seed", type=int, default=20261106)
    p.add_argument("--retina-masks", action=argparse.BooleanOptionalAction, default=True,
                   help="Default restores all segmentation masks to source H,W; --no-retina-masks measures predictor-resolution masks")
    p.add_argument("--check-only", action="store_true",
                   help="CPU file/hash/decode validation only; no torch/Ultralytics import, no files written")
    return p


def val_images(data_path):
    import yaml
    document = yaml.safe_load(data_path.read_text(encoding="utf-8-sig"))
    require(isinstance(document, dict) and "val" in document, "Dataset YAML needs val")
    root = f_path(document.get("path") or ".", data_path.parent)
    entries = document["val"] if isinstance(document["val"], list) else [document["val"]]
    require(entries and all(isinstance(x, str) and x.strip() for x in entries), "Unsupported val entry")
    found = []
    for value in entries:
        entry = f_path(value, root)
        require(entry.exists(), f"Missing val entry: {entry}")
        if entry.is_dir():
            found.extend(x.resolve() for x in entry.rglob("*") if x.is_file() and x.suffix.lower() in IMAGE_SUFFIXES)
        elif entry.suffix.lower() == ".txt":
            for line in entry.read_text(encoding="utf-8-sig").splitlines():
                if line.strip():
                    found.append(f_path(line.strip(), entry.parent))
        else:
            require(entry.suffix.lower() in IMAGE_SUFFIXES, f"Unsupported val file: {entry}")
            found.append(entry)
    require(found, "No validation images")
    for image in found:
        f_path(image)
        require(image.is_file() and image.suffix.lower() in IMAGE_SUFFIXES, f"Invalid val image: {image}")
    require(len(set(found)) == len(found), "Duplicate image paths in val definition")
    return sorted(found, key=lambda x:str(x).casefold())


def preflight(args):
    project = Path(__file__).resolve().parent
    require(project.drive.upper() == "F:", "Script must run from the F-drive project")
    output = f_path(args.output)
    require(output.is_relative_to(project), "Output must stay under the F-drive project")
    require(not output.exists(), "Output exists; refusing overwrite")
    require(1 <= args.num_frames <= 10000 and 1 <= args.repeats <= 100, "Invalid frames/repeats")
    require(1 <= args.warmup <= 1000, "At least one warmup pipeline required")
    require(args.imgsz > 0 and args.imgsz % 32 == 0, "imgsz must be a positive multiple of 32")
    require(math.isfinite(args.conf) and 0 < args.conf < 1, "Invalid conf")
    require(math.isfinite(args.iou) and 0 < args.iou <= 1, "Invalid IoU")
    require(1 <= args.max_det <= 10000, "Invalid max_det")
    match = re.fullmatch(r"(?:cuda:)?(\d+)", args.device)
    require(match is not None, "One CUDA ordinal is required")
    args.device = str(int(match.group(1)))
    data = f_path(args.data)
    require(data.is_file() and data.suffix.lower() in (".yaml", ".yml"), "Dataset YAML missing")
    weights = [f_path(x) for x in args.weights]
    require(len(set(weights)) == len(weights), "Duplicate checkpoint path")
    for weight in weights:
        require(weight.is_file() and weight.suffix.lower() == ".pt", f"Existing local .pt checkpoint required: {weight}")
    files = val_images(data)
    require(len(files) >= args.num_frames, f"val has {len(files)} images; requested {args.num_frames}")
    selected = random.Random(args.seed).sample(files, args.num_frames)
    import cv2
    import numpy as np
    frames, image_records = [], []
    for path in selected:
        encoded = path.read_bytes()
        image = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR)
        require(image is not None and image.ndim == 3 and image.shape[2] == 3, f"BGR decode failed: {path}")
        require(image.dtype == np.uint8 and image.flags.c_contiguous, "Unexpected predecoded BGR format")
        frames.append(image)
        image_records.append({"path":str(path),"image_sha256":hashlib.sha256(encoded).hexdigest(),
            "source_hw":[int(image.shape[0]),int(image.shape[1])],"decoded_bgr_sha256":hashlib.sha256(memoryview(image)).hexdigest()})
    manifest = {"data_yaml":str(data),"data_yaml_sha256":sha(data),"split":"val","available_val_images":len(files),
        "selection":"Uniform sample without replacement from sorted val image paths; no GT read",
        "selection_seed":args.seed,"images":image_records,
        "models":[{"path":str(w),"sha256":sha(w),"bytes":w.stat().st_size} for w in weights]}
    return project,output,weights,frames,manifest


def percentile(values, fraction):
    ordered = sorted(values)
    pos = (len(ordered)-1)*fraction
    low,high = math.floor(pos),math.ceil(pos)
    return ordered[low]+(ordered[high]-ordered[low])*(pos-low)


def summarize(timings):
    require(timings and all(math.isfinite(x) and x > 0 for x in timings), "Invalid timing samples")
    mean = statistics.mean(timings)
    return {"n_pipeline_frames":len(timings),"mean_ms":mean,"p50_ms":percentile(timings,.5),
        "p95_ms":percentile(timings,.95),"fps_1000_over_mean_ms":1000.0/mean,
        "fps_scope":"Predecoded-frame serial model pipeline including every output mask/box CPU materialization; not robot end-to-end FPS"}


def configure_runtime(output,project):
    runtime=output/"runtime"
    for name in ("tmp","yolo","matplotlib","torch","hf","cache","cuda","torch_extensions","pip"):
        (runtime/name).mkdir(parents=True,exist_ok=True)
    os.environ.update(PYTHONDONTWRITEBYTECODE="1",YOLO_AUTOINSTALL="False",YOLO_OFFLINE="true",
        HF_HUB_OFFLINE="1",TRANSFORMERS_OFFLINE="1",HF_HOME=str(runtime/"hf"),
        YOLO_CONFIG_DIR=str(runtime/"yolo"),MPLCONFIGDIR=str(runtime/"matplotlib"),
        TORCH_HOME=str(runtime/"torch"),XDG_CACHE_HOME=str(runtime/"cache"),
        CUDA_CACHE_PATH=str(runtime/"cuda"),TORCH_EXTENSIONS_DIR=str(runtime/"torch_extensions"),PIP_CACHE_DIR=str(runtime/"pip"),
        TMP=str(runtime/"tmp"),TEMP=str(runtime/"tmp"))
    os.chdir(project)


def serial_pipeline(models, frame, predict_args, np):
    """Timed caller retains every independent CPU NumPy array until stop time."""
    materialized=[]
    for model in models:
        results=model.predict(source=frame,stream=False,**predict_args)
        require(isinstance(results,list) and len(results)==1, "Expected exactly one frame result")
        result=results[0]
        require(result.boxes is not None, "Detection/segmentation result is missing boxes")
        boxes=np.array(result.boxes.data.detach().cpu().numpy(),copy=True,order="C")
        if result.masks is None:
            require(model.task=="detect" or len(boxes)==0, "Segment model returned detected boxes without masks")
            masks=np.empty((0,0,0),dtype=np.uint8)
        else:
            masks=np.array(result.masks.data.detach().cpu().numpy(),copy=True,order="C")
            require(masks.ndim==3 and masks.shape[0]==len(boxes), "Masks/boxes count mismatch")
        materialized.append((boxes,masks))
    return materialized


def output_metadata(materialized):
    return [{"detections":len(boxes),"boxes_shape":list(boxes.shape),"masks_shape":list(masks.shape),
             "boxes_dtype":str(boxes.dtype),"masks_dtype":str(masks.dtype),
             "cpu_output_bytes":int(boxes.nbytes+masks.nbytes)}
            for boxes,masks in materialized]


def run(args):
    project,output,weights,frames,manifest=preflight(args)
    if args.check_only:
        print(json.dumps({"phase":"CPU_preflight_passed","torch_imported":"torch" in sys.modules,
            "GPU_used":False,"files_written":False,"inputs":manifest},ensure_ascii=False))
        return
    output.mkdir(parents=True,exist_ok=False)
    configure_runtime(output,project)
    report={"schema":"yolo_predecoded_whole_frame_pipeline_benchmark_v1","started_utc":now(),
        "benchmark_source_sha256":sha(__file__),"inputs":manifest,"ground_truth_labels_read":False,
        "quality_evaluated":False,"is_robot_end_to_end_measurement":False,
        "models_run_serially_per_frame":len(weights),"outputs_fused":False,
        "multi_model_semantics":"Serial full-frame computation with independent model outputs; no validated fusion, routing, joint accuracy or Small+Base equivalence",
        "included":"YOLO.predict source setup, preprocess, host-to-device, forward, postprocess, all boxes and all mask planes copied to CPU NumPy, final CUDA sync",
        "excluded":["image file access and decoding","camera capture","model loading and warmup","tracking","cross-model fusion or routing","mask polygonization","robot control","sorting and box placement","report serialization"],
        "raw_timings":[]}
    try:
        import numpy as np
        import torch
        import ultralytics
        from ultralytics import YOLO
        require(ultralytics.__version__==EXPECTED_ULTRALYTICS,"Unexpected Ultralytics version; audit code before changing version")
        require(torch.cuda.is_available(),"CUDA required for this synchronized GPU benchmark")
        device=torch.device("cuda:"+args.device)
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
        torch.backends.cudnn.benchmark=False
        torch.set_float32_matmul_precision("highest")
        models=[]
        for path in weights:
            model=YOLO(str(path))
            require(model.task in ("detect","segment"),f"Checkpoint must be detect/segment: {path}")
            model.to(device)
            model.model.float().eval()
            models.append(model)
        predict_args={"device":args.device,"imgsz":args.imgsz,"batch":1,"quantize":32,
            "conf":args.conf,"iou":args.iou,"max_det":args.max_det,"rect":False,
            "retina_masks":args.retina_masks,"augment":False,"verbose":False,
            "save":False,"save_txt":False,"save_conf":False,"save_crop":False,"show":False,
            "project":str(output/"predict"),"name":"no_saved_predictions"}
        report["configuration"]={"warmup_pipeline_frames":args.warmup,"distinct_frames":args.num_frames,
            "repeats":args.repeats,"predict":predict_args,"precision":"FP32 weights and inputs; TF32 disabled",
            "synchronize":"torch.cuda.synchronize(device) before start and after all CPU outputs",
            "clock":"time.perf_counter_ns","mask_coordinates":"original image H,W" if args.retina_masks else "predictor preprocessed mask planes (not restored to original H,W)",
            "model_speed_fields_used":False}
        properties=torch.cuda.get_device_properties(device)
        report["environment"]={"ultralytics":ultralytics.__version__,"torch":torch.__version__,
            "torch_cuda":torch.version.cuda,"cudnn":torch.backends.cudnn.version(),"cuda_ordinal":args.device,
            "gpu_name":properties.name,"gpu_total_bytes":properties.total_memory,
            "cuda_visible_devices":os.environ.get("CUDA_VISIBLE_DEVICES"),
            "other_gpu_workloads_verified_absent":False}
        with torch.inference_mode():
            for i in range(args.warmup):
                discarded=serial_pipeline(models,frames[i%len(frames)],predict_args,np)
                torch.cuda.synchronize(device)
                del discarded
            for model in models:
                require(not model.predictor.model.fp16,"Predictor unexpectedly uses FP16")
                require(all(p.dtype==torch.float32 for p in model.model.parameters() if p.is_floating_point()),
                        "Checkpoint parameters not all FP32")
            report["model_runtime"]=[{"path":str(w),"task":m.task,"class_names":m.names,
                "predictor_rect":m.predictor.args.rect,"predictor_imgsz":m.predictor.args.imgsz,
                "predictor_quantize":m.predictor.args.quantize,"backend_fp16":bool(m.predictor.model.fp16),"end2end_head":bool(getattr(m.model,"end2end",False))}
                for w,m in zip(weights,models)]
            torch.cuda.reset_peak_memory_stats(device)
            for repeat in range(args.repeats):
                for index,frame in enumerate(frames):
                    torch.cuda.synchronize(device)
                    start=time.perf_counter_ns()
                    outputs=serial_pipeline(models,frame,predict_args,np)
                    torch.cuda.synchronize(device)
                    elapsed_ns=time.perf_counter_ns()-start
                    record={"repeat":repeat,"frame_index":index,"elapsed_ns":elapsed_ns,
                        "elapsed_ms":elapsed_ns/1_000_000,"models":output_metadata(outputs)}
                    report["raw_timings"].append(record)
                    del outputs
                print(json.dumps({"phase":"repeat_completed","repeat":repeat+1,
                    "summary":summarize([x["elapsed_ms"] for x in report["raw_timings"] if x["repeat"]==repeat])}),flush=True)
        report["summary"]=summarize([x["elapsed_ms"] for x in report["raw_timings"]])
        report["per_repeat"]=[summarize([x["elapsed_ms"] for x in report["raw_timings"] if x["repeat"]==r])
                              for r in range(args.repeats)]
        report["peak_gpu_allocated_bytes"]=torch.cuda.max_memory_allocated(device)
        report["peak_gpu_reserved_bytes"]=torch.cuda.max_memory_reserved(device)
        require(sha(manifest["data_yaml"])==manifest["data_yaml_sha256"],"Data YAML changed during benchmark")
        for row,frame in zip(manifest["images"],frames):
            require(sha(row["path"])==row["image_sha256"],"Source image changed during benchmark")
            require(hashlib.sha256(memoryview(frame)).hexdigest()==row["decoded_bgr_sha256"],
                    "Model mutated its original decoded BGR input")
        for row in manifest["models"]:
            require(sha(row["path"])==row["sha256"],"Checkpoint changed during benchmark")
        report["inputs_unchanged"]=True
        report["completed_utc"]=now()
        write_json(output/"benchmark.json",report)
        print(json.dumps({"phase":"completed","report":str(output/"benchmark.json"),"summary":report["summary"]},ensure_ascii=False))
    except Exception as error:
        report["failed_utc"]=now()
        report["error"]={"type":type(error).__name__,"message":str(error)}
        write_json(output/"failure.json",report)
        raise


if __name__=="__main__":
    run(parser().parse_args())
