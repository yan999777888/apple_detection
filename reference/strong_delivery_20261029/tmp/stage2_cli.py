"""CLI for the second-stage crop verifier: cache, train, rescore.

Usage (training env, GPU for train, CPU for the rest):

  python -B tmp/stage2_cli.py cache
  python -B tmp/stage2_cli.py train --epochs 10 --run stage2_crop_v1
  python -B tmp/stage2_cli.py eval --checkpoint runs/stage2_crop/stage2_crop_v1_best.pt \
      --predictions runs/od_verifier_plain_joint8_11/validation/update_006912_...
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'tmp'))

import stage2_crop as s2  # noqa: E402
from apple_seg.data import PAppleDataset  # noqa: E402

CACHE = ROOT / 'tmp/stage2_cache'
RUNS = ROOT / 'runs/stage2_crop'
DEV_IMAGES = 300


def cmd_cache(args):
    started = time.time()
    for split in ('train', 'val'):
        manifest = s2.cache_images(ROOT / 'assets/papple/images' / split, CACHE / split,
                                   short_side=args.short_side, quality=args.quality,
                                   worker=lambda i, n: print(f'  {split} {i}/{n}', flush=True))
        sizes = [m['width'] * m['height'] for m in manifest.values()]
        print(f'{split}: {len(manifest)} images, mean cached pixels {int(np.mean(sizes))}', flush=True)
    print(f'cache done in {time.time() - started:.0f}s')


def load_annotations(split):
    """Annotation boxes and V/A ratios, cached once because rasterising 2304 mask
    frames takes minutes and the trainer needs them every run."""
    path = CACHE / f'{split}_annotations.json'
    if path.is_file():
        payload = json.loads(path.read_text(encoding='utf-8'))
        return {name: dict(boxes=np.asarray(row['boxes'], dtype='float32').reshape(-1, 4),
                           ratios=np.asarray(row['ratios'], dtype='float32'), hw=tuple(row['hw']),
                           image_id=row['image_id']) for name, row in payload.items()}
    index = s2.annotation_index(PAppleDataset(ROOT / 'assets/papple', split, 32))
    CACHE.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({name: dict(boxes=row['boxes'].tolist(), ratios=row['ratios'].tolist(),
                                           hw=list(row['hw']), image_id=row['image_id'])
                                for name, row in index.items()}), encoding='utf-8')
    return index


def _split_names(annotations):
    names = sorted(annotations)
    dev = {name for index, name in enumerate(names) if index % 8 == 0}
    return [n for n in names if n not in dev], sorted(dev), names


def list_collate(batch):
    """Module level on purpose: spawn-based workers must be able to pickle it."""
    return batch


def _flatten(batch):
    tight = torch.cat([item[0] for item in batch])
    context = torch.cat([item[1] for item in batch])
    scalars = torch.cat([item[2] for item in batch])
    labels = torch.cat([item[3] for item in batch])
    weights = torch.cat([item[4] for item in batch])
    return tight, context, scalars, labels, weights


def cmd_train(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    annotations = load_annotations('train')
    train_names, dev_names, all_names = _split_names(annotations)
    positives = sum(len(annotations[n]['boxes']) for n in train_names)
    heavy = sum(float((annotations[n]['ratios'] < s2.HEAVY_RATIO).sum()) for n in train_names)
    print(f'TRAIN images {len(train_names)} ({positives} GT, {heavy:.0f} heavy) '
          f'dev {len(dev_names)}', flush=True)
    dataset = s2.Stage2TrainDataset(CACHE / 'train', annotations, train_names, seed=args.seed)
    # Each image yields a variable number of crops, so collation stays a list
    # and the crop tensors are flattened after moving to the device.
    loader = DataLoader(dataset, batch_size=args.batch, shuffle=True, num_workers=args.workers,
                        drop_last=True, persistent_workers=args.workers > 0,
                        collate_fn=list_collate)
    dev = s2.Stage2TrainDataset(CACHE / 'train', annotations, dev_names, seed=args.seed + 1,
                                flip=False)
    dev_loader = DataLoader(dev, batch_size=32, shuffle=False, num_workers=args.workers,
                            collate_fn=list_collate)
    model = s2.CropVerifier(pretrained=args.pretrained).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps = max(1, len(loader) * args.epochs)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, total_steps=steps,
                                                    pct_start=.15)
    # bfloat16 needs no loss scaling; keep the loop dtype-simple.
    RUNS.mkdir(parents=True, exist_ok=True)
    history, best = [], None
    for epoch in range(args.epochs):
        dataset.set_epoch(epoch)
        model.train()
        running = 0.
        started = time.time()
        for step, batch in enumerate(loader):
            tight, context, scalars, labels, weights = (t.to(device, non_blocking=True)
                                                        for t in _flatten(batch))
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=device.type == 'cuda' and bool(args.amp)):
                logits = model(tight, context, scalars)
                loss = s2.focal_bce(logits.float(), labels, weights, gamma=args.gamma)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
            scheduler.step()
            running += float(loss.detach())
        model.eval()
        scores, targets = [], []
        with torch.no_grad():
            for batch in dev_loader:
                tight, context, scalars, labels, _ = (t.to(device) for t in _flatten(batch))
                with torch.autocast('cuda', dtype=torch.bfloat16, enabled=device.type == 'cuda' and bool(args.amp)):
                    logits = model(tight, context, scalars)
                scores.append(logits.float().sigmoid().cpu().numpy())
                targets.append(labels.cpu().numpy())
        scores, targets = np.concatenate(scores), np.concatenate(targets)
        auc = _auc(scores, targets)
        row = dict(epoch=epoch, loss=running / max(len(loader), 1), dev_auc=round(float(auc), 5),
                   seconds=round(time.time() - started, 1))
        history.append(row)
        print(json.dumps(row), flush=True)
        if best is None or auc > best['dev_auc']:
            best = dict(row)
            torch.save(dict(schema='stage2_crop_v1', model=model.state_dict(),
                            epoch=epoch, dev_auc=float(auc), args=vars(args)), RUNS / f'{args.run}_best.pt')
    (RUNS / f'{args.run}_history.json').write_text(json.dumps(
        dict(best=best, history=history, train_images=len(train_names),
             dev_images=len(dev_names), positives=positives, heavy=heavy), indent=1), encoding='utf-8')
    print('best', json.dumps(best), flush=True)


def _auc(scores, labels):
    order = np.argsort(scores)
    ranks = np.empty(len(scores), dtype='float64')
    ranks[order] = np.arange(1, len(scores) + 1)
    positive = labels >= .5
    n_pos, n_neg = int(positive.sum()), int((~positive).sum())
    if not n_pos or not n_neg:
        return float('nan')
    return (ranks[positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def _prefix(record, scores):
    """Descending-score order prefix: cumulative one-to-one TP and the sorted scores."""
    gt = np.asarray(record['gt'], dtype='float64').reshape(-1, 4)
    boxes = np.asarray(record['boxes'], dtype='float64').reshape(-1, 4)
    order = np.array(sorted(range(len(scores)), key=lambda i: (-scores[i], i)), dtype='int64')
    matrix = s2.iou_matrix(boxes[order] if len(order) else np.zeros((0, 4)), gt)
    used, prefix = set(), np.zeros(len(order) + 1, dtype='int64')
    for row in range(len(order)):
        candidates = [(matrix[row, j], j) for j in range(matrix.shape[1])
                      if j not in used and matrix[row, j] >= .5]
        if candidates:
            used.add(max(candidates)[1])
            prefix[row + 1] = prefix[row] + 1
        else:
            prefix[row + 1] = prefix[row]
    return np.sort(scores)[::-1], prefix, len(gt)


def _sweep_fast(prefixes, thresholds):
    best = None
    for threshold in thresholds:
        tp = fp = total = 0
        for sorted_scores, prefix, gt_count in prefixes:
            k = int(np.searchsorted(-sorted_scores, -threshold, side='left'))
            tp += int(prefix[k])
            fp += k - int(prefix[k])
            total += gt_count
        precision = tp / max(tp + fp, 1)
        recall = tp / max(total, 1)
        score = min(precision, recall)
        if best is None or score > best['score']:
            best = dict(threshold=float(threshold), precision=precision, recall=recall,
                        score=score, tp=tp, fp=fp, fn=total - tp)
    return best


def cmd_eval(args):
    directory = Path(args.predictions)
    if not directory.is_absolute():
        directory = (ROOT / directory)
    directory = directory.resolve()
    if not (directory / 'raw_predictions.json').is_file():
        # Accept a run directory as well as one validation update directory.
        directory = next((directory / 'validation').glob(f'update_{args.update}_*')).resolve()
    raw = json.loads((directory / 'raw_predictions.json').read_text(encoding='utf-8'))
    summary = json.loads((directory / 'summary.json').read_text(encoding='utf-8'))
    manifest = s2.read_manifest(CACHE / 'val')
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    model = s2.CropVerifier(pretrained=None)
    model.load_state_dict(checkpoint['model'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device).eval()
    print(f"checkpoint epoch {checkpoint['epoch']} dev_auc {checkpoint['dev_auc']:.4f}; "
          f"{len(raw)} VAL images", flush=True)
    started = time.time()
    records = []
    for index, entry in enumerate(raw):
        name = entry['image_id']
        meta = manifest[name]
        scale = meta['scale']
        boxes = np.asarray(entry['pred']['boxes'], dtype='float64').reshape(-1, 4)
        base = np.asarray(entry['pred']['scores'], dtype='float64')
        if not len(boxes):
            records.append(dict(image=name, boxes=boxes, base=base, stage2=base, gt=entry['gt']['boxes']))
            continue
        with s2.Image.open(CACHE / 'val' / meta['file']) as handle:
            image = handle.convert('RGB')
        tiles = [[], []]
        scalars = []
        for box in boxes:
            view = box * scale
            tight, context = s2.pair_tensor(image, view)
            tiles[0].append(tight)
            tiles[1].append(context)
            scalars.append(s2.box_scalars(view, float(meta['width']), float(meta['height'])))
        tight = torch.stack(tiles[0])
        context = torch.stack(tiles[1])
        scalar = torch.tensor(scalars, dtype=torch.float32)
        outputs = []
        for start in range(0, len(tight), args.chunk):
            with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16,
                                                 enabled=device.type == 'cuda'):
                logits = model(tight[start:start + args.chunk].to(device),
                               context[start:start + args.chunk].to(device),
                               scalar[start:start + args.chunk].to(device))
            outputs.append(logits.float().sigmoid().cpu().numpy())
        stage2 = np.concatenate(outputs)
        records.append(dict(image=name, boxes=boxes, base=base, stage2=stage2, gt=entry['gt']['boxes']))
        if index % 100 == 0:
            print(f'  {index}/{len(raw)}', flush=True)
    print(f'rescore done in {time.time() - started:.0f}s', flush=True)

    RUNS.mkdir(parents=True, exist_ok=True)
    # Persist the per-box scores first: the rescoring pass is the expensive part
    # and every later analysis must be able to reuse it.
    scores_path = RUNS / f"{Path(args.checkpoint).stem}_{directory.name[:16]}_scores.json"
    scores_path.write_text(json.dumps({record['image']: dict(
        boxes=[[round(float(v), 2) for v in box] for box in record['boxes']],
        base=[round(float(v), 6) for v in record['base']],
        stage2=[round(float(v), 6) for v in record['stage2']],
        gt=[[round(float(v), 2) for v in box] for box in
            (np.asarray(record['gt']).reshape(-1, 4).tolist() if len(np.asarray(record['gt'])) else [])]
    ) for record in records}), encoding='utf-8')
    print('wrote', scores_path, flush=True)

    pool = np.concatenate([r['base'] for r in records])
    thresholds = np.unique(np.quantile(pool, np.linspace(.55, .9999, 500)))
    report = dict(directory=directory.name, checkpoint=str(args.checkpoint),
                  scores_path=str(scores_path),
                  dev_auc=checkpoint['dev_auc'], epoch=checkpoint['epoch'],
                  candidate_pool=int(len(pool)),
                  repo_published=dict(confidence=summary['confidence'], precision=summary['summary']['precision'],
                                      recall=summary['summary']['recall'],
                                      tp=summary['summary']['tp'], fp=summary['summary']['fp'],
                                      fn=summary['summary']['fn']))
    for key in ('base', 'stage2'):
        prefixes = [_prefix(record, np.asarray(record[key], dtype='float64')) for record in records]
        report[key] = _sweep_fast(prefixes, thresholds)
        report[key]['feasible_fp_limit'] = int(report[key]['tp'] * (1 - .95) / .95)
    fused = []
    for record in records:
        fused.append(dict(boxes=record['boxes'], gt=record['gt'],
                          fused=np.sqrt(np.clip(record['base'], 1e-9, 1) * np.clip(record['stage2'], 1e-9, 1))))
    prefixes = [_prefix(record, record['fused']) for record in fused]
    report['fused_geometric_mean'] = _sweep_fast(prefixes, thresholds)
    out = RUNS / f"{Path(args.checkpoint).stem}_{directory.name[:16]}_rescore.json"
    out.write_text(json.dumps(report, indent=1), encoding='utf-8')
    print(json.dumps({k: v for k, v in report.items() if k not in ('checkpoint',)}, indent=1))
    print('wrote', out)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='command', required=True)
    cache = sub.add_parser('cache')
    cache.add_argument('--short-side', type=int, default=1024)
    cache.add_argument('--quality', type=int, default=92)
    cache.set_defaults(func=cmd_cache)

    train = sub.add_parser('train')
    train.add_argument('--run', default='stage2_crop_v1')
    train.add_argument('--epochs', type=int, default=10)
    train.add_argument('--batch', type=int, default=24, help='images per batch, crops vary')
    train.add_argument('--workers', type=int, default=8)
    train.add_argument('--lr', type=float, default=3e-4)
    train.add_argument('--weight-decay', type=float, default=1e-4)
    train.add_argument('--gamma', type=float, default=1.0)
    train.add_argument('--seed', type=int, default=17)
    train.add_argument('--pretrained', default='IMAGENET1K_V1')
    train.add_argument('--amp', type=int, default=1, choices=(0, 1))
    train.set_defaults(func=cmd_train)

    evaluate = sub.add_parser('eval')
    evaluate.add_argument('--checkpoint', required=True)
    evaluate.add_argument('--predictions', required=True)
    evaluate.add_argument('--update', default='006912')
    evaluate.add_argument('--chunk', type=int, default=256)
    evaluate.set_defaults(func=cmd_eval)
    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
