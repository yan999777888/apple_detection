"""Opt-in autocast for the frozen dual-task worker; FP32 weights stay intact.

Uses the same full-image 640 preprocessing, thresholds, native mask projection
and CPU Results as FrozenDeploymentAdapter. Numerical equivalence is NOT assumed.
Requires independent validation for each checkpoint.
"""
import torch
from deployment_adapter_v1 import FrozenDeploymentAdapter


class FrozenAMPDeploymentAdapter(FrozenDeploymentAdapter):
    def __init__(self, model, **kwargs):
        super().__init__(model, **kwargs)
        if self.device.type != 'cuda':
            raise ValueError('AMP deployment requires CUDA')

    def predict(self, *args, **kwargs):
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            return super().predict(*args, **kwargs)
