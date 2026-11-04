"""Bounded complete-train PApple V model; stages stop at explicit epoch nodes."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys


def main() -> int:
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--weights", type=Path, default=repo / "weights/local_rgbd/yolo26s-seg.pt")
    parser.add_argument("--output", type=Path, required=True, help="New segment directory on F:")
    parser.add_argument("--epochs", type=int, default=10, help="Total fixed schedule, unchanged across resume")
    parser.add_argument("--until-epoch", type=int, help="Stop at completed epoch; zero saves initialization only")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--mask-ratio", type=int, choices=(1, 2, 4), default=2)
    parser.add_argument("--seed", type=int, default=20261104)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--amp-init-scale", type=float, default=1024)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--preflight-report", type=Path)
    parser.add_argument("--prepared-cache", type=Path, help="Optional F-drive lossless prepared-tensor cache")
    parser.add_argument("--resume", type=Path, help="Formal epoch-boundary checkpoint only")
    parser.add_argument("--max-updates", type=int, help="Independent 1..32-update engineering gate; not resumable")
    parser.add_argument("--log-interval", type=int, default=20)
    args = parser.parse_args()
    if repo.drive.upper() != "F:" or Path(sys.executable).drive.upper() != "F:":
        parser.error("Repository and environment must be on F:")
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    os.environ["YOLO_AUTOINSTALL"] = "false"
    for key, suffix in {"YOLO_CONFIG_DIR": "cache/ultralytics_local_training", "TORCH_HOME": "cache/torch",
        "HF_HOME": "cache/huggingface", "CUDA_CACHE_PATH": "cache/cuda", "XDG_CACHE_HOME": "cache",
        "MPLCONFIGDIR": "cache/matplotlib_local_rgbd", "TORCH_EXTENSIONS_DIR": "cache/torch_extensions",
        "TEMP": "tmp/local_rgbd", "TMP": "tmp/local_rgbd"}.items():
        location = repo / suffix
        location.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(location)
    sys.path.insert(0, str(repo / "src"))
    from apple_local_rgbd.formal_training import run_formal
    result = run_formal(repo=repo, **vars(args))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
