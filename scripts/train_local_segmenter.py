"""Bounded native-raster V baseline smoke; no automatic long training."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys


def main() -> int:
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True, help="Official TRAIN raster manifest; VAL/test refused")
    parser.add_argument("--weights", type=Path, default=repo / "weights/local_rgbd/yolo26s-seg.pt")
    parser.add_argument("--output", type=Path, required=True, help="New empty directory on F:")
    parser.add_argument("--steps", type=int, default=12, help="Bounded optimizer steps, 1..32; default 12")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--mask-ratio", type=int, choices=(1, 2, 4), default=2)
    parser.add_argument("--seed", type=int, default=20261104)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--amp-init-scale", type=float, default=1024.0,
                        help="Finite positive GradScaler initial scale; default 1024")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--inspect-data-only", action="store_true", help="CPU raster/GT audit without model or CUDA calls")
    args = parser.parse_args()
    if repo.drive.upper() != "F:":
        parser.error("Repository, environment and outputs must reside on F:")
    # Configure caches before third-party imports, including package __init__.
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    os.environ["YOLO_AUTOINSTALL"] = "false"
    for key, suffix in {"YOLO_CONFIG_DIR": "cache/ultralytics_local_training", "TORCH_HOME": "cache/torch",
                        "HF_HOME": "cache/huggingface", "CUDA_CACHE_PATH": "cache/cuda",
                        "XDG_CACHE_HOME": "cache", "TEMP": "tmp", "TMP": "tmp"}.items():
        directory = repo / suffix
        directory.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(directory)
    sys.path.insert(0, str(repo / "src"))
    from apple_local_rgbd.local_training import run_smoke
    result = run_smoke(repo=repo, **vars(args))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
