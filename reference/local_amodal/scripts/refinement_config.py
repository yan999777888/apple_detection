"""CPU-only safeguards and paired data augmentation for resolution experiments."""
from __future__ import annotations

import hashlib
import math


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def validate_initial_checkpoint(checkpoint, signature, data, schema):
    """Require an own completed-run checkpoint with unchanged train/val data."""
    config = checkpoint["run_config"]
    if checkpoint["architecture_metadata"]["schema"] != schema or config["schema"] != schema:
        raise ValueError("Initial architecture does not match this runner")
    if config["dataset_signature"] != signature or config["data"] != str(data):
        raise ValueError("Initial checkpoint train/val data differs; no cross-split warm start")
    if checkpoint["model_config"].get("pretrained") is not False:
        raise ValueError("Initial checkpoint must be an own download-free model")
    if checkpoint["validation"]["inference"]["dataset_split"] != "val":
        raise ValueError("Expected a validation-selected checkpoint, never test-selected")
    if (config.get('test_evaluation') != 'not_run'
            or checkpoint['validation']['summary']['images'] != config['val_images']):
        raise ValueError('Initial weights must use complete validation without test tuning')
    if int(checkpoint["epoch"]) < 1:
        raise ValueError("Initial checkpoint has no completed training epoch")
    model_config = dict(checkpoint["model_config"])
    if set(model_config) != {'pretrained', 'min_size', 'max_size', 'detections_per_img'}:
        raise ValueError('Unexpected model configuration fields')
    if checkpoint['architecture_metadata'].get('box_nms_thresh') != 0.5:
        raise ValueError('Initial NMS policy differs from the paired-resolution baseline')
    if checkpoint['architecture_metadata'].get('box_score_thresh') != 0.001:
        raise ValueError('Initial raw prediction threshold differs from baseline')
    return model_config


def validate_hyperparameters(args):
    if min(args.epochs, args.batch_size, args.accumulate, args.min_size, args.max_size) < 1:
        raise ValueError("Epochs, batch, accumulation and image sizes must be positive")
    if args.min_size > args.max_size or args.warmup_updates < 0:
        raise ValueError("Invalid size bounds or warmup")
    if not all(math.isfinite(x) and x > 0 for x in (args.head_lr, args.backbone_lr)):
        raise ValueError("Learning rates must be finite and positive")


def flip_for_sample(seed, epoch, filename):
    key = f"{int(seed)}\0{int(epoch)}\0{filename}".encode("utf-8")
    return bool(hashlib.sha256(key).digest()[0] & 1)


def fixed_flip_numpy(image, target):
    import numpy as np
    width = image.shape[1]
    result = dict(target)
    result["boxes"] = target["boxes"].copy()
    result["boxes"][:, [0, 2]] = width - target["boxes"][:, [2, 0]]
    for key in ("masks", "visible_masks"):
        result[key] = np.ascontiguousarray(target[key][:, :, ::-1])
    return np.ascontiguousarray(image[:, ::-1]), result


class MatchedFlipDataset:
    """Use identical per-image flips at each epoch across paired resolutions.

    The model's RNG and different anchor counts cannot change this decision.
    Dataset order uses the trainer's independent DataLoader generator as usual.
    """
    def __init__(self, base, seed):
        if base.augment:
            raise ValueError("The base dataset must not apply its own random flip")
        self.base, self.seed, self.epoch = base, int(seed), 0
        self.samples, self.root, self.split = base.samples, base.root, base.split

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        import numpy as np
        import torch
        image, target = self.base.load_sample(index)
        if flip_for_sample(self.seed, self.epoch, self.samples[index]["filename"]):
            image, target = fixed_flip_numpy(image, target)
        tensor = torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))).float().div_(255.0)
        return tensor, {key: torch.from_numpy(value) for key, value in target.items()}
