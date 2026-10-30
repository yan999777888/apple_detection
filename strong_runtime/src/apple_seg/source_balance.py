"""Source-aware classification loss; masks retain their original supervision."""
from copy import deepcopy
import hashlib
import math
from pathlib import Path
import re

import torch
from torch import nn


def source_sha256():
    with Path(__file__).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def validate_plan(weight, expected_sha256):
    if isinstance(weight, bool) or not isinstance(weight, (int, float)) or not math.isfinite(weight) or not 0 < weight <= 1:
        raise ValueError('minne_class_weight must be finite and in (0,1]')
    if not isinstance(expected_sha256, str) or re.fullmatch('[0-9a-f]{64}', expected_sha256) is None:
        raise ValueError('The exact classification adapter SHA256 is required in the recipe')
    if source_sha256() != expected_sha256:
        raise ValueError('Classification adapter source SHA256 differs from the pinned recipe')


class SourceBalancedCriterion(nn.Module):
    def __init__(self, base, weight, expected_sha256):
        super().__init__()
        validate_plan(weight, expected_sha256)
        self.base, self.weight, self.expected_sha256 = base, weight, expected_sha256
        self.configuration = {**deepcopy(base.configuration),
                              'source_class_loss_weights': {'PApple': 1.0, 'MinneApple': weight},
                              'classification_adapter_sha256': expected_sha256,
                              'source_mask_losses_changed': False}

    @property
    def last_assignment(self):
        return self.base.last_assignment

    def forward(self, outputs, targets, *, generator=None):
        # This math adapter is pinned in checkpoint identity.recipe and checked
        # on every call. Existing model/loss/data source guards remain intact.
        validate_plan(self.weight, self.expected_sha256)
        losses = self.base(outputs, targets, generator=generator)
        source = targets[0]['source']['dataset']
        if source == 'MinneApple':
            losses = {**losses, 'loss_class': losses['loss_class'] * self.weight}
        elif source != 'PApple':
            raise ValueError('Unsupported supervised source')
        return losses


def install_source_balance(weight, expected_sha256):
    validate_plan(weight, expected_sha256)
    from . import losses
    original = losses.build_annotation_aware_criterion
    if getattr(original, '_source_balance_plan', None) is not None:
        if original._source_balance_plan != (weight, expected_sha256):
            raise ValueError('Conflicting source balance plans in one process')
        return
    def build(*args, **kwargs):
        return SourceBalancedCriterion(original(*args, **kwargs), weight, expected_sha256)
    build._source_balance_plan = (weight, expected_sha256)
    losses.build_annotation_aware_criterion = build