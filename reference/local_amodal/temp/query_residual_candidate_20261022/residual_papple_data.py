"""1024px PApple adapter: unchanged real masks plus original-box reference.

Import and configuration checks use only the standard library. The frozen
adapter still owns all decoding, normalization and original-GT verification.
"""
from copy import deepcopy
import hashlib
from pathlib import Path
from types import ModuleType

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OLD_ADAPTER = ROOT / "temp/query_dual_mask_candidate_20261021/query_papple_data.py"
OLD_ADAPTER_SHA256 = "1fbb142f4710454be90833bdacc3ff4841c9339ca1d5332897f743da36be96df"


def _old_adapter():
    source = OLD_ADAPTER.read_bytes()
    if hashlib.sha256(source).hexdigest() != OLD_ADAPTER_SHA256:
        raise ValueError("Frozen original PApple adapter changed")
    module = ModuleType("_residual_frozen_papple_adapter")
    module.__file__ = str(OLD_ADAPTER)
    exec(compile(source, str(OLD_ADAPTER), "exec"), module.__dict__)
    return module


_OLD = _old_adapter()
ANNOTATION_SHA256 = deepcopy(_OLD.ANNOTATION_SHA256)
DATASET_LOADER_SHA256 = _OLD.DATASET_LOADER_SHA256
ORIGINAL_SIZE = _OLD.ORIGINAL_SIZE
SUPPORTED_INPUT_SIZES = (1024,)
_source_paths = _OLD._source_paths
collate_residual_papple_batch = _OLD.collate_query_papple_batch
collate_query_papple_batch = collate_residual_papple_batch


def audit_adapter_config(*, split, input_size=1024, augment=False, batch_size=1, external_padding=False):
    if input_size != 1024 or isinstance(input_size, bool):
        raise ValueError("This residual trial uses exactly1024px whole-frame inputs")
    return _OLD.audit_adapter_config(split=split, input_size=input_size, augment=augment,
                                    batch_size=batch_size, external_padding=external_padding)


def build_residual_papple_dataset(root, split, *, input_size=1024, augment=False,
                                  batch_size=1, external_padding=False):
    """Lazy runtime factory; every original GT/mask/box/ID stays intact."""
    audit_adapter_config(split=split, input_size=input_size, augment=augment,
                         batch_size=batch_size, external_padding=external_padding)
    if hashlib.sha256(OLD_ADAPTER.read_bytes()).hexdigest() != OLD_ADAPTER_SHA256:
        raise ValueError("Frozen original PApple adapter changed before construction")
    dataset = _OLD.build_query_papple_dataset(root, split, input_size=input_size, augment=augment,
        batch_size=batch_size, external_padding=external_padding)

    class ResidualPAppleDataset:
        def __init__(self):
            self._base = dataset._base
            self.configuration = deepcopy(dataset.configuration)
            self.annotation_sha256 = deepcopy(dataset.annotation_sha256)

        def __len__(self):
            return len(dataset)

        def image_metadata(self, index):
            return dataset.image_metadata(index)

        def __getitem__(self, index):
            item = dataset[index]
            boxes = item["original_target"]["boxes"]
            import torch
            if (boxes.device.type != "cpu" or boxes.dtype != torch.float32
                    or tuple(boxes.shape) != (len(item["apple_ids"]), 4) or boxes.requires_grad):
                raise ValueError("Original boxes must be fixed CPU float32[N,4] exclusive XYXY")
            item["target"]["boxes"] = boxes
            if item["target"]["boxes"] is not item["original_target"]["boxes"]:
                raise AssertionError("Residual target must alias the untouched original boxes")
            return item

    return ResidualPAppleDataset()
