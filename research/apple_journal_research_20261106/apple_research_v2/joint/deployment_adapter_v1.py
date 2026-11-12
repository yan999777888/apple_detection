"""Dedicated, frozen FP32 worker; retains native preprocessing and postprocessing.

The caller transfers exclusive ownership of an eval-only model. Unlike the
training-time reference adapter, this worker does not restore training flags,
RNG or head caches after every frame. Never share it with a training thread.
"""
from __future__ import annotations

import threading
import numpy as np
import torch
from joint_eval_v1 import JointInferenceAdapter, TASK_TYPES


class FrozenDeploymentAdapter(JointInferenceAdapter):
    def __init__(self, model, **kwargs):
        if any(m.training for m in model.modules()):
            raise ValueError("Dedicated deployment model must already be eval-only")
        if any(p.requires_grad for p in model.parameters()):
            raise ValueError("Dedicated deployment parameters must be frozen")
        super().__init__(model, **kwargs)
        self._prediction_lock = threading.Lock()
        for view in self.views.values():
            view.set_head_attr(max_det=max(self.max_det, 300), agnostic_nms=False)

    @torch.inference_mode()
    def predict(self, images_bgr, *, paths=None, tasks=("stage", "defect"), cpu=True):
        tasks = self._tasks(tasks)
        images = [images_bgr] if isinstance(images_bgr, np.ndarray) else list(images_bgr)
        if not images or any(not isinstance(im, np.ndarray) or im.dtype != np.uint8
                             or im.ndim != 3 or im.shape[2] != 3 or min(im.shape[:2]) < 1
                             for im in images):
            raise ValueError("Expected BGR uint8 HWC images")
        paths = list(paths) if paths is not None else [f"frame_{i}" for i in range(len(images))]
        if len(paths) != len(images):
            raise ValueError("Path/image batch size mismatch")
        if self.model.training:
            raise RuntimeError("Deployment model was switched to training")
        with self._prediction_lock:
            tensor = self.predictors[tasks[0]].preprocess(images)
            if tuple(tensor.shape[1:]) != (3, 640, 640):
                raise AssertionError("Fixed-square 640 preprocessing required")
            raw = self.model.predict_both(tensor) if len(tasks) == 2 else {
                tasks[0]: self.model.predict_source(tensor, tasks[0])}
            output = {}
            for task in tasks:
                predictor = self.predictors[task]
                predictor.batch = (paths, images, [""] * len(images))
                results = predictor.postprocess(raw[task], tensor, images)
                if len(results) != len(images):
                    raise AssertionError("Postprocess batch count mismatch")
                output[task] = [r.cpu() for r in results] if cpu else results
            return output
