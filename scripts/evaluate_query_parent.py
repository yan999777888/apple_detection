"""Recover Q0 on the complete frozen PApple VAL, without training."""
from pathlib import Path
import argparse
import json
import sys
import time

ROOT = Path(__file__).absolute().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import torch
    from apple_seg.model import DualMaskModel
    from apple_seg.data import PAppleDataset
    from apple_seg.evaluate import evaluate_papple, _atomic_json
    config_path = args.config.absolute()
    config = json.loads(config_path.read_text(encoding="utf-8-sig"))
    base = (config_path.parent / config["path_base"]).absolute()
    paths = {k: (base / v).absolute() for k, v in config["paths"].items()}
    output = args.output.absolute()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Q0 requires a fresh output directory")
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(config["training"]["cpu_threads"])
    torch.set_num_interop_threads(1)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    print("Q0 strict parent load; no optimizer or training", flush=True)
    model, parent = DualMaskModel.from_checkpoint(paths["parent_checkpoint"], config["parent_sha256"])
    model.requires_grad_(False).eval().to("cuda")
    dataset = PAppleDataset(paths["papple_root"], split="val", input_size=config["training"]["input_size"], augment=False)
    started = time.monotonic()
    result = evaluate_papple(model, dataset, torch.device("cuda"), output, paths["papple_provenance"])
    assert result["phase"] == "complete" and result["model_selection_eligible"]
    assert (result["summary"]["images"], result["summary"]["gt_count"], result["summary"]["empty_gt_images"]) == (814, 3449, 13)
    _atomic_json(output / "q0_run.json", {"phase": "complete", "elapsed_seconds": time.monotonic() - started,
        "parent_sha256": config["parent_sha256"], "strict_restore": parent["strict_model_load"],
        "input_size": config["training"]["input_size"], "training_performed": False,
        "summary": result["summary"]})
    print(json.dumps({"phase": "complete", "Q0": result["summary"]}), flush=True)


if __name__ == "__main__":
    main()
