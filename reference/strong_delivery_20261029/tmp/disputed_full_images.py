"""Draw the whole image with GT boxes and the disputed boxes, to settle whether a
"zero-overlap false positive" is really unlabelled fruit or really a leaf."""
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
SIDE = 520


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
        hits = np.nonzero((overlap <= 0) & (base >= .95) & (stage2 >= .95))[0]
        for index in hits:
            picks.append((float(base[index]), name, index))
    picks.sort(reverse=True)
    print(f'{len(picks)} disputed boxes at base>=.95 and stage2>=.95')
    chosen, seen = [], set()
    for _, name, index in picks:
        if name in seen:
            continue
        seen.add(name)
        chosen.append((name, index))
        if len(chosen) == 4:
            break
    sheet = Image.new('RGB', (SIDE * 2, SIDE * 2), (255, 255, 255))
    for slot, (name, index) in enumerate(chosen):
        payload = scores[name]
        boxes = np.asarray(payload['boxes'], dtype='float64').reshape(-1, 4)
        gt = np.asarray(payload['gt'], dtype='float64').reshape(-1, 4)
        meta = manifest[name]
        with Image.open(CACHE / meta['file']) as handle:
            image = handle.convert('RGB').copy()
        draw = ImageDraw.Draw(image)
        for box in gt:
            draw.rectangle([box[0] * meta['scale'], box[1] * meta['scale'],
                            box[2] * meta['scale'], box[3] * meta['scale']], outline=(0, 255, 0), width=3)
        box = boxes[index]
        draw.rectangle([box[0] * meta['scale'], box[1] * meta['scale'],
                        box[2] * meta['scale'], box[3] * meta['scale']], outline=(255, 0, 0), width=3)
        draw.text((6, 6), f'{name} base={payload["base"][index]:.3f} s2={payload["stage2"][index]:.3f}',
                  fill=(255, 255, 0))
        sheet.paste(image.resize((SIDE, SIDE), Image.BILINEAR),
                    ((slot % 2) * SIDE, (slot // 2) * SIDE))
    out = ROOT / 'reports/val_error_crops/disputed_full_images_006912.png'
    sheet.save(out)
    print('wrote', out, [name for name, _ in chosen])


if __name__ == '__main__':
    main()
