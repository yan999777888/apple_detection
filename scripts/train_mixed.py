"""Portable mixed-data pilot entry; source paths are resolved from config base."""
from pathlib import Path
import argparse
from datetime import datetime, timezone
from importlib.metadata import version, PackageNotFoundError
import json
import os
import platform
import sys
import traceback

sys.path.insert(0, str(Path(__file__).absolute().parents[1] / "src"))


class Tee:
    def __init__(self, console, log):
        self.console, self.log = console, log

    def write(self, text):
        self.console.write(text)
        self.log.write(text)
        self.log.flush()

    def flush(self):
        self.console.flush()
        self.log.flush()

    def __getattr__(self, name):
        return getattr(self.console, name)


def main():
    parser = argparse.ArgumentParser(description="Real PApple/MinneApple four-image mixed updates")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--smoke", action="store_true", help="One real mixed update by default; no full VAL or formal warm start")
    parser.add_argument("--max-updates", type=int)
    parser.add_argument("--input-size", type=int)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume", nargs="?", const=True, default=None, type=Path)
    args = parser.parse_args()
    from apple_seg.training import atomic_json, load_config, run_training
    cfg = load_config(args.config, output=args.output, input_size=args.input_size,
                      max_updates=args.max_updates, smoke=args.smoke)
    out = cfg["output"]
    if out.exists() and any(out.iterdir()) and args.resume is None:
        raise FileExistsError("Use a fresh output directory or --resume")
    out.mkdir(parents=True, exist_ok=True)
    with (out / "runtime.log").open("a" if args.resume is not None else "x", encoding="utf-8") as log:
        stdout, stderr = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = Tee(stdout, log), Tee(stderr, log)
        try:
            versions = {}
            for name in ("torch", "transformers", "numpy", "scipy", "Pillow"):
                try:
                    versions[name] = version(name)
                except PackageNotFoundError:
                    versions[name] = "not_installed"
            print(json.dumps({"event": "launch", "utc": datetime.now(timezone.utc).isoformat(),
                              "python": sys.executable, "python_version": platform.python_version(),
                              "versions": versions, "device": args.device, "smoke": args.smoke,
                              "config": str(cfg["path"]), "output": str(out),
                              "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES")}), flush=True)
            summary = run_training(args.config, device=args.device, smoke=args.smoke, max_updates=args.max_updates,
                                   input_size=args.input_size, output=args.output, resume=args.resume)
            print(json.dumps({key: summary[key] for key in ("phase", "updates", "output_directory", "full_validation_executed")}), flush=True)
            return 130 if summary["phase"] == "interrupted" else 0
        except BaseException as error:
            traceback.print_exc()
            status_path = out / "training_status.json"
            status = json.loads(status_path.read_text(encoding="utf-8")) if status_path.exists() else {}
            if status.get("phase") != "failed":
                atomic_json(status_path, {"phase": "interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                            "failure_type": type(error).__name__, "failure": str(error),
                            "last_status": status})
            raise
        finally:
            sys.stdout, sys.stderr = stdout, stderr


if __name__ == "__main__":
    sys.exit(main())
