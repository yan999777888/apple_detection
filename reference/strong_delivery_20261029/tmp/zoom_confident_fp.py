"""Zoom in on the boxes both scorers call apples but the annotation calls background.

If a meaningful share of these really are apples, then the 95% precision target is
limited by the annotation, not by the model, and that has to be said explicitly.
"""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tmp'))
sys.path.insert(0, str(ROOT / 'src'))

import stage2_crop as s2  # noqa: E402

CACHE = ROOT / 'tmp/stage2_cache/val'
TILE = 150
COLUMNS = 6


def main():
    scores = json.loads((ROOT / 'runs/stage2_crop/stage2_crop_v1_best_update_006912_17_scores.json')
                        .read_text(encoding='utf-8'))
    manifest = s2.read_manifest(CACHE)
    picks = []
    for name, row in scores.items():
        boxes = np.asarray(row['boxes'], dtype='float64').reshape(-1, 4)
        gt = np.asarray(row['gt'], dtype='float64').reshape(-1, 4)
        if not len(boxes):
            continue
        matrix = s2.iou_matrix(boxes, gt) if len(gt) else np.zeros((len(boxes), 1))
        overlap = matrix.max(1) if len(gt) else np.zeros(len(boxes))
        base, stage2 = np.asarray(row['base']), np.asarray(row['stage2'])
        for index in np.nonzero((overlap <= 0) & (base >= .9) & (stage2 >= .9))[0]:
            picks.append((float(base[index]), name, boxes[index].tolist(), float(stage2[index])))
    picks.sort(reverse=True)
    print(f'{len(picks)} boxes with zero GT overlap that both scorers call apple (base>=.9, stage2>=.9)')
    picks = picks[:24]
    lines = max(1, (len(picks) + COLUMNS - 1) // COLUMNS)
    sheet = Image.new('RGB', (COLUMNS * TILE, lines * (TILE + 18)), (255, 255, 255))
    draw = ImageDraw.Draw(sheet)
    cache = {}
    for slot, (base, name, box, stage2) in enumerate(picks):
        meta = manifest[name]
        if name not in cache:
            with Image.open(CACHE / meta['file']) as handle:
                cache[name] = handle.convert('RGB').copy()
        tile = s2.crop_window(cache[name], np.asarray(box) * meta['scale'], 1.0, TILE)
        column, line = slot % COLUMNS, slot // COLUMNS
        sheet.paste(tile, (column * TILE, line * (TILE + 18)))
        draw.text((column * TILE + 2, line * (TILE + 18) + TILE + 3),
                  f'b{base:.2f}/s{stage2:.2f}', fill=(0, 0, 0))
    out = ROOT / 'reports/val_error_crops/zero_overlap_confident_006912.png'
    sheet.save(out)
    print('wrote', out)


if __name__ == '__main__':
    main()
