"""Second-stage crop verifier for the PApple detector (experimental).

Motivation, measured on the published 6912-point full VAL of
`od_verifier_plain_joint8_11`:

* 201 false positives with exactly zero GT overlap carry base scores of
  0.94-0.997, and their crops are plain foliage. The first-stage score is
  saturated and therefore almost uninformative among surviving candidates.
* 193 false negatives already have a box at IoU >= 0.5 against a real apple but
  score 0.05-0.42. Their crops are heavily occluded apples.

Both errors are ranking errors over an already adequate candidate pool: the
saved raw predictions hold 105070 post-NMS boxes (20836 at base score >= 0.01)
for 3449 GT, so nothing needs to be re-proposed. This module therefore never
touches the detector, the RPN or the NMS: it re-scores the *same* boxes from
image crops and lets a threshold sweep decide the operating point.

Design rules that keep the experiment honest:

* TRAIN crops come from TRAIN annotations only; VAL pixels are read once, at
  rescoring time, and never influence training or model selection.
* Positives are jittered GT boxes; negatives are background boxes sampled away
  from every GT plus deliberate partial-overlap boxes (the measured 164
  partial-overlap false positives).
* Only geometry that is available identically at train and eval time enters the
  scalar branch, so there is no feature that exists in one place and not the
  other.
* Two crops per box: the tight box and a 2.4x context window, so a leaf cluster
  that sits between apples can be judged by its surroundings.
"""
import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch.utils.data import Dataset

CROP = 128
TIGHT_SCALE = 1.0
CONTEXT_SCALE = 2.4
SCALARS = 4
HEAVY_RATIO = .4
BACKGROUND_IOU = .05
PARTIAL_IOU = (.15, .45)


# --------------------------------------------------------------------------- #
# image cache
# --------------------------------------------------------------------------- #
def cache_images(source_dir, cache_dir, short_side=1024, quality=92, worker=None):
    """Write a resized JPEG copy of every image and return name -> scale."""
    source_dir, cache_dir = Path(source_dir), Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    names = sorted(p.name for p in source_dir.iterdir()
                   if p.suffix.lower() in ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))
    manifest = {}
    for index, name in enumerate(names):
        target = cache_dir / (Path(name).stem + '.jpg')
        with Image.open(source_dir / name) as image:
            image = image.convert('RGB')
            width, height = image.size
            scale = short_side / min(width, height)
            size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
            if not target.is_file() or os.path.getsize(target) == 0:
                image.resize(size, Image.BICUBIC).save(target, quality=quality, optimize=True)
        manifest[name] = dict(file=target.name, width=size[0], height=size[1], scale=scale)
        if worker is not None and index % 200 == 0:
            worker(index, len(names))
    (cache_dir / 'manifest.json').write_text(
        json.dumps(dict(source=str(source_dir), short_side=short_side, quality=quality,
                        images=manifest), indent=1), encoding='utf-8')
    return manifest


def read_manifest(cache_dir):
    payload = json.loads((Path(cache_dir) / 'manifest.json').read_text(encoding='utf-8'))
    return payload['images']


# --------------------------------------------------------------------------- #
# annotations (TRAIN only)
# --------------------------------------------------------------------------- #
def mask_boxes(masks):
    """xyxy of every instance mask, in the mask's own pixel grid."""
    boxes = []
    for mask in masks:
        rows = torch.nonzero(mask.any(dim=1), as_tuple=False)
        cols = torch.nonzero(mask.any(dim=0), as_tuple=False)
        if not len(rows) or not len(cols):
            boxes.append([0., 0., 0., 0.])
            continue
        boxes.append([float(cols[0]), float(rows[0]), float(cols[-1] + 1), float(rows[-1] + 1)])
    return torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4)


def annotation_index(dataset):
    """name -> dict(boxes, ratios, hw) from an unaugmented native split."""
    index = {}
    for item in dataset:
        target = item['target']
        source = target['source']
        name = source['image_id']
        visible, amodal = target['visible_masks'], target['amodal_masks']
        boxes = mask_boxes(amodal if amodal is not None else visible)
        if amodal is not None:
            total = amodal.sum(dim=(-1, -2)).clamp_min(1)
            ratios = (visible.sum(dim=(-1, -2)) / total).clamp(0, 1)
        else:
            ratios = torch.ones(len(boxes))
        index[name] = dict(boxes=boxes.numpy().astype('float32'), ratios=ratios.numpy().astype('float32'),
                           hw=tuple(int(v) for v in target['original_hw']),
                           image_id=source.get('original_image_id', name))
    return index


def iou_1_to_n(box, boxes):
    if not len(boxes):
        return np.zeros(0, dtype='float32')
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = max((box[2] - box[0]) * (box[3] - box[1]), 1e-9)
    other = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union = area + other - inter
    return (inter / np.maximum(union, 1e-9)).astype('float32')


# --------------------------------------------------------------------------- #
# cropping
# --------------------------------------------------------------------------- #
def crop_window(image, box, scale, size=CROP):
    """Square window around ``box`` blown up by ``scale``, clamped to ``image``."""
    left, top, right, bottom = box
    cx, cy = (left + right) / 2., (top + bottom) / 2.
    half = max(right - left, bottom - top) * scale / 2.
    half = max(half, 2.)
    x1, y1 = int(round(cx - half)), int(round(cy - half))
    x2, y2 = int(round(cx + half)), int(round(cy + half))
    pad = max(0, -x1, -y1, x2 - image.width, y2 - image.height)
    if pad:
        canvas = Image.new('RGB', (image.width + 2 * pad, image.height + 2 * pad), (114, 114, 114))
        canvas.paste(image, (pad, pad))
        image = canvas
        x1, y1, x2, y2 = x1 + pad, y1 + pad, x2 + pad, y2 + pad
    return image.crop((x1, y1, x2, y2)).resize((size, size), Image.BILINEAR)


def box_scalars(box, width, height):
    w = max(box[2] - box[0], 1.)
    h = max(box[3] - box[1], 1.)
    return [math.log(w / width) / 2., math.log(h / height) / 2.,
            math.log(w / h) / 2., math.sqrt(w * h / (width * height)) * 2. - 1.]


def to_tensor(image):
    array = np.array(image, dtype='uint8', copy=True)
    return torch.from_numpy(array).permute(2, 0, 1).float().div_(255.)


MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def pair_tensor(image, box, size=CROP):
    tight = (to_tensor(crop_window(image, box, TIGHT_SCALE, size)) - MEAN) / STD
    context = (to_tensor(crop_window(image, box, CONTEXT_SCALE, size)) - MEAN) / STD
    return tight, context


# --------------------------------------------------------------------------- #
# TRAIN dataset
# --------------------------------------------------------------------------- #
@dataclass
class SamplePlan:
    positives: int
    background: int
    partial: int


class Stage2TrainDataset(Dataset):
    """One item = one image; crops are drawn deterministically from ``epoch``."""

    def __init__(self, cache_dir, annotations, names, *, plan=SamplePlan(1.0, 1.0, .6),
                 seed=0, size=CROP, flip=True):
        self.cache_dir = Path(cache_dir)
        self.manifest = read_manifest(self.cache_dir)
        self.annotations = annotations
        self.names = [name for name in names if name in self.manifest
                      and name in annotations and len(annotations[name]['boxes'])]
        self.plan, self.seed, self.size, self.flip = plan, seed, size, flip
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __len__(self):
        return len(self.names)

    def _sample_box(self, boxes, size_log, rng, want):
        width, height, mean_side = size_log
        for _ in range(60):
            w = float(np.exp(rng.normal(0., .25))) * mean_side
            h = w * float(np.exp(rng.normal(0., .15)))
            w, h = max(6., min(w, width - 2)), max(6., min(h, height - 2))
            x = rng.uniform(0, width - w)
            y = rng.uniform(0, height - h)
            box = np.array([x, y, x + w, y + h], dtype='float32')
            best = float(iou_1_to_n(box, boxes).max()) if len(boxes) else 0.
            if want == 'background' and best < BACKGROUND_IOU:
                return box
            if want == 'partial' and PARTIAL_IOU[0] <= best <= PARTIAL_IOU[1]:
                return box
        return None

    def __getitem__(self, index):
        name = self.names[index]
        meta = self.manifest[name]
        info = self.annotations[name]
        rng = np.random.RandomState((self.seed + index * 7919 + self.epoch * 104729) % (2 ** 31))
        scale = meta['scale']
        width, height = float(meta['width']), float(meta['height'])
        with Image.open(self.cache_dir / meta['file']) as handle:
            image = handle.convert('RGB')
        boxes = info['boxes'] * scale
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        mean_side = float(np.sqrt(np.median(areas))) if len(areas) else 24.
        size_log = (width, height, mean_side)
        tiles, contexts, scalars, labels, weights = [], [], [], [], []
        for box, ratio in zip(boxes, info['ratios']):
            jitter = np.array([rng.uniform(-.12, .12), rng.uniform(-.12, .12)] * 2) * np.array(
                [box[2] - box[0], box[3] - box[1]] * 2)
            view = (box + jitter).astype('float32')
            tight, context = pair_tensor(image, view, self.size)
            tiles.append(tight), contexts.append(context)
            scalars.append(box_scalars(view, width, height))
            labels.append(1.)
            # Heavily occluded apples are the recall bottleneck; weight them up.
            weights.append(1. + 1.5 * (1. if ratio < HEAVY_RATIO else 0.))
        counts = self.plan
        for kind, share in (('background', counts.background), ('partial', counts.partial)):
            for _ in range(int(round(share * max(len(boxes), 1)))):
                box = self._sample_box(boxes, size_log, rng, kind)
                if box is None:
                    continue
                tight, context = pair_tensor(image, box, self.size)
                tiles.append(tight), contexts.append(context)
                scalars.append(box_scalars(box, width, height))
                labels.append(0.)
                weights.append(1.5 if kind == 'partial' else 1.)
        tight = torch.stack(tiles)
        context = torch.stack(contexts)
        if self.flip and rng.rand() < .5:
            tight, context = tight.flip(-1), context.flip(-1)
        return (tight, context, torch.tensor(scalars, dtype=torch.float32),
                torch.tensor(labels, dtype=torch.float32), torch.tensor(weights, dtype=torch.float32))


# --------------------------------------------------------------------------- #
# model
# --------------------------------------------------------------------------- #
class CropVerifier(nn.Module):
    """ResNet18 over (tight, context) crops plus four geometry scalars."""

    def __init__(self, *, pretrained=None, dropout=.1):
        super().__init__()
        from torchvision.models import resnet18
        backbone = resnet18(weights=pretrained)
        stem = backbone.conv1
        weight = stem.weight.data
        fused = torch.cat((weight, weight), dim=1) / 2.
        self.stem = nn.Conv2d(6, stem.out_channels, kernel_size=stem.kernel_size,
                              stride=stem.stride, padding=stem.padding, bias=False)
        with torch.no_grad():
            self.stem.weight.copy_(fused)
        self.bn1, self.relu, self.maxpool = backbone.bn1, backbone.relu, backbone.maxpool
        self.layer1, self.layer2 = backbone.layer1, backbone.layer2
        self.layer3, self.layer4 = backbone.layer3, backbone.layer4
        self.pool = nn.AdaptiveAvgPool2d(1)
        features = backbone.fc.in_features
        self.head = nn.Sequential(
            nn.Linear(features + SCALARS, 256), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(256, 64), nn.GELU(), nn.Linear(64, 1))

    def forward(self, tight, context, scalars):
        x = torch.cat((tight, context), dim=1)
        x = self.maxpool(self.relu(self.bn1(self.stem(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x).flatten(1)
        return self.head(torch.cat((x, scalars), dim=1)).squeeze(1)


def focal_bce(logits, labels, weights, gamma=1.0):
    bce = F.binary_cross_entropy_with_logits(logits, labels, reduction='none')
    probability = logits.sigmoid()
    modulation = (probability - labels).abs().pow(gamma)
    return (bce * modulation * weights).sum() / weights.sum().clamp_min(1e-6)


# --------------------------------------------------------------------------- #
# evaluation helpers
# --------------------------------------------------------------------------- #
def iou_matrix(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 1e-9, inter / np.maximum(union, 1e-9), 0.)


def match_counts(boxes, scores, gt, threshold):
    """Same rule as the repo's full-VAL evaluation: descending score, one-to-one, IoU>=.5."""
    order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
    selected = [i for i in order if scores[i] >= threshold]
    matrix = iou_matrix(np.asarray(boxes)[selected] if selected else np.zeros((0, 4)), gt)
    used, tp = set(), 0
    for row, index in enumerate(selected):
        candidates = [(matrix[row, j], j) for j in range(matrix.shape[1])
                      if j not in used and matrix[row, j] >= .5]
        if candidates:
            used.add(max(candidates)[1])
            tp += 1
    fp = len(selected) - tp
    return tp, fp, len(gt) - tp


def sweep(records, score_key, thresholds=None):
    """Best min(P,R) over a global threshold, exactly like the repo's selection."""
    all_scores = np.concatenate([np.asarray(r[score_key]) for r in records
                                 if len(np.asarray(r[score_key]))] or [np.zeros(1)])
    if thresholds is None:
        candidates = np.unique(np.quantile(all_scores, np.linspace(.5, .9995, 400)))
        thresholds = [float(t) for t in candidates]
    best = None
    for threshold in thresholds:
        tp = fp = total = 0
        for record in records:
            t, f, _ = match_counts(record['boxes'], np.asarray(record[score_key]),
                                   np.asarray(record['gt'], float).reshape(-1, 4), threshold)
            tp, fp, total = tp + t, fp + f, total + len(record['gt'])
        precision = tp / max(tp + fp, 1)
        recall = tp / max(total, 1)
        score = min(precision, recall)
        if best is None or score > best['score']:
            best = dict(threshold=threshold, precision=precision, recall=recall, score=score,
                        tp=tp, fp=fp, fn=total - tp)
    return best
