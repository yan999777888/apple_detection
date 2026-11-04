"""Score one OD checkpoint on the complete original PApple VAL split.

The trainer's own validation directory is the reference protocol; this script
only re-runs it for a checkpoint whose recorded score came from a tensor set
that was not persisted (EMA before the artifact fix).
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).absolute().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402

from apple_seg.od_evaluate import evaluate_od  # noqa: E402
from apple_seg.od_model import build_od_amodal_detector  # noqa: E402
from apple_seg.od_training import load_checkpoint  # noqa: E402
from apple_seg.data import PAppleDataset  # noqa: E402
from apple_seg.od_data import ODInputDataset  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--provenance", default="assets/papple/provenance.json")
    parser.add_argument("--short-side", type=int, default=1024)
    parser.add_argument("--max-size", type=int, default=1280)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    payload = load_checkpoint(Path(args.checkpoint))
    print("weights_are_ema", payload.get("weights_are_ema", False), flush=True)
    model = build_od_amodal_detector(weights=None, **payload["constructor_config"])
    model.load_state_dict(payload["model"], strict=True)
    device = torch.device(args.device)
    model.to(device).eval()

    dataset = ODInputDataset(PAppleDataset(ROOT / "assets/papple", "val", 32),
                             short_sides=(args.short_side,), max_size=args.max_size)
    print("images", len(dataset), flush=True)
    with torch.inference_mode():
        result = evaluate_od(model, dataset, Path(args.out), device=device,
                             provenance_path=ROOT / args.provenance,
                             defer_mask_metrics=True)
    summary = result["summary"]
    print(json.dumps({"phase": result["phase"], "eligible": result["model_selection_eligible"],
                      "confidence": result["confidence"], "score": result["score"],
                      "precision": summary["precision"], "recall": summary["recall"],
                      "tp": summary["tp"], "fp": summary["fp"], "fn": summary["fn"],
                      "images": summary["images"], "gt": summary["gt_count"]}, indent=1), flush=True)


if __name__ == "__main__":
    main()
