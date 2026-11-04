"""Validate one existing checkpoint into a fresh full-VAL directory.

The training loop can only validate the model it is holding. Measuring the same
weights at a different evaluation resolution needs this small entry point, which
mirrors ``od_training.validate`` exactly: 814 images, 3449 GT, 13 empty frames,
the deferred mask metrics and the original provenance contract.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from apple_seg import od_training as training  # noqa: E402
from apple_seg.data import PAppleDataset  # noqa: E402
from apple_seg.od_data import ODInputDataset  # noqa: E402
from apple_seg.od_evaluate import evaluate_od  # noqa: E402
from apple_seg.od_model import build_od_amodal_detector  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--short-side', type=int, required=True)
    parser.add_argument('--max-size', type=int, required=True)
    parser.add_argument('--update-label', default='rescore')
    parser.add_argument('--out-dir', default=None)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--defer-mask-metrics', type=int, default=1)
    args = parser.parse_args()

    config = training.load_od_config(Path(args.config))
    payload = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    model = build_od_amodal_detector(weights=None, **payload['constructor_config'])
    model.load_state_dict(payload['model'], strict=True)
    device = torch.device(args.device)
    model.to(device).eval()

    run_dir = Path(args.checkpoint).resolve().parents[1]
    stamp = time.strftime('%Y%m%d_%H%M%S')
    out = Path(args.out_dir) if args.out_dir else (
        run_dir / 'validation' / f'update_{payload["updates"]:06d}_side{args.short_side}_{stamp}')
    out.mkdir(parents=True, exist_ok=True)
    dataset = ODInputDataset(PAppleDataset(ROOT / 'assets/papple', 'val', 32),
                             short_sides=(args.short_side,), max_size=args.max_size)
    print(f'checkpoint {Path(args.checkpoint).name} updates={payload["updates"]} '
          f'short_side={args.short_side} max_size={args.max_size} -> {out}', flush=True)
    evaluated = evaluate_od(model, dataset, out, device=device,
                            provenance_path=config['paths']['papple_provenance'],
                            defer_mask_metrics=bool(args.defer_mask_metrics))
    summary = evaluated['summary']
    block = dict(directory=str(out), updates=payload['updates'], short_side=args.short_side,
                 max_size=args.max_size, confidence=evaluated['confidence'],
                 score=evaluated['score'], precision=summary['precision'], recall=summary['recall'],
                 tp=summary['tp'], fp=summary['fp'], fn=summary['fn'],
                 images=summary['images'], gt_count=summary['gt_count'],
                 goal_95_met=summary['goal_95_met'],
                 eligible=evaluated['model_selection_eligible'], phase=evaluated['phase'])
    print(json.dumps(block, indent=1), flush=True)
    (out / 'resolution_summary.json').write_text(json.dumps(block, indent=1), encoding='utf-8')


if __name__ == '__main__':
    main()
