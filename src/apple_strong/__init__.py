"""Strong-model-only experiment helpers; no model replacement or auto-training."""

from .consistency import gt_anchored_consistency, horizontal_flip_boxes

__all__ = ["gt_anchored_consistency", "horizontal_flip_boxes"]
