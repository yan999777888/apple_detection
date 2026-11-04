"""Evaluate native visible masks of a local checkpoint on PApple development VAL."""
from pathlib import Path
import argparse
import json
import os
import sys


def main():
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=repo / "cache/datasets/local_apple_v1/val_manifest.json")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--kind", choices=("local", "migrated_initial", "coco"), default="local")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf-floor", type=float, default=.001)
    parser.add_argument("--fixed-threshold", type=float, default=.25)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--limit", type=int, help="Explicit first-N diagnostic, never reported as full validation")
    parser.add_argument("--benchmark-only", action="store_true")
    parser.add_argument("--run-benchmark", action="store_true")
    parser.add_argument("--benchmark-sizes", nargs="+", type=int, default=[640, 512])
    parser.add_argument("--save-mask-images", type=int, default=8)
    args = parser.parse_args()
    if repo.drive.upper() != "F:" or Path(sys.executable).drive.upper() != "F:":
        parser.error("Repository and environment must be on F:")
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    os.environ["YOLO_AUTOINSTALL"] = "false"
    for name, suffix in {"YOLO_CONFIG_DIR": "cache/ultralytics_local_training", "TORCH_HOME": "cache/torch",
                         "HF_HOME": "cache/huggingface", "TEMP": "tmp/local_rgbd", "TMP": "tmp/local_rgbd",
                         "MPLCONFIGDIR": "cache/matplotlib_local_rgbd", "CUDA_CACHE_PATH": "cache/cuda"}.items():
        destination = repo / suffix
        destination.mkdir(parents=True, exist_ok=True)
        os.environ[name] = str(destination)
    sys.path.insert(0, str(repo / "src"))
    from apple_local_rgbd.local_evaluation import run_evaluation
    print(json.dumps(run_evaluation(repo=repo, **vars(args)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
