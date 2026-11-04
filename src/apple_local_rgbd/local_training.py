"""Native-raster, single-visible-mask YOLO26 training adapter.

This is an independent local-model baseline, not a replacement for Base + Small
and not an amodal model. No polygons, hulls, bridge lines or largest-component
operations occur here. One original instance ID remains one mask tensor.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset


MANIFEST_SCHEMA = "local_apple_native_raster_manifest_v1"
CHECKPOINT_SCHEMA = "local_apple_visible_yolo26_checkpoint_v1"
OFFICIAL_SHA256 = "3da1d83e31caec96f9300eb4064f4f62882c133c7c264d63dfe61a7c197837a4"


class TrainingContractError(ValueError):
    """Reject a contract violation instead of silently changing ground truth."""


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def require_f_path(path: str | Path) -> Path:
    result = Path(path).resolve()
    if result.drive.upper() != "F:":
        raise TrainingContractError(f"All runtime paths must be on F:, got {result}")
    return result


def configure_f_runtime(repo: str | Path) -> None:
    """Call before importing Ultralytics; never install or alter shared packages."""
    repo = require_f_path(repo)
    mapping = {
        "YOLO_CONFIG_DIR": repo / "cache/ultralytics_local_training",
        "TORCH_HOME": repo / "cache/torch",
        "HF_HOME": repo / "cache/huggingface",
        "XDG_CACHE_HOME": repo / "cache",
        "CUDA_CACHE_PATH": repo / "cache/cuda",
        "PIP_CACHE_DIR": repo / "cache/pip",
        "TEMP": repo / "tmp",
        "TMP": repo / "tmp",
    }
    for key, value in mapping.items():
        value.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(value)
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["YOLO_AUTOINSTALL"] = "false"


def tight_boxes(masks: np.ndarray) -> np.ndarray:
    """Pixel xyxy, right/bottom exclusive, encompassing every visible fragment."""
    boxes = np.empty((len(masks), 4), dtype=np.float32)
    for index, mask in enumerate(masks):
        yy, xx = np.nonzero(mask)
        if not len(xx):
            raise TrainingContractError(f"Empty visible mask at instance index {index}")
        boxes[index] = (xx.min(), yy.min(), xx.max() + 1, yy.max() + 1)
    return boxes


def decode_packed_masks(path: str | Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as data:
        required = {"visible_packed", "amodal_packed", "original_hw", "instance_ids", "labels",
                    "boxes_visible_xyxy", "boxes_amodal_xyxy"}
        if not required <= set(data.files):
            raise TrainingContractError(f"Missing packed raster keys: {required - set(data.files)}")
        height, width = [int(x) for x in data["original_hw"]]
        ids = [str(x) for x in data["instance_ids"]]
        n = len(ids)
        if len(set(ids)) != n or height < 1 or width < 1:
            raise TrainingContractError("Duplicate instance IDs or invalid native image size")
        result: dict[str, Any] = {"instance_ids": ids, "original_hw": [height, width]}
        for name in ("visible", "amodal"):
            packed = data[f"{name}_packed"]
            if packed.dtype != np.uint8 or packed.shape != (n, height, (width + 7) // 8):
                raise TrainingContractError(f"Invalid {name} packed shape/dtype: {packed.shape}/{packed.dtype}")
            result[name] = np.unpackbits(packed, axis=-1, count=width, bitorder="little")
        labels = data["labels"]
        if labels.shape != (n,) or np.any(labels != 0):
            raise TrainingContractError("This baseline accepts only one class, apple = 0")
        boxes = tight_boxes(result["visible"])
        if not np.array_equal(boxes, data["boxes_visible_xyxy"]):
            raise TrainingContractError("Stored V box differs from native V pixel tight box")
        result["boxes_visible_xyxy"] = boxes
        return result


def _resize_masks(masks: np.ndarray, new_hw: tuple[int, int]) -> np.ndarray:
    h, w = new_hw
    if not len(masks):
        return np.zeros((0, h, w), dtype=np.uint8)
    return np.stack([cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST) for mask in masks])


def _component_counts(masks: np.ndarray) -> list[int]:
    return [int(cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)[0] - 1) for mask in masks]


def prepare_raster_sample(rgb: np.ndarray, decoded: dict[str, Any], *, imgsz: int,
                          mask_ratio: int, metadata: dict[str, Any]) -> dict[str, Any]:
    if imgsz < 32 or imgsz % 32 or mask_ratio not in (1, 2, 4) or imgsz % mask_ratio:
        raise TrainingContractError("imgsz must be a multiple of 32; mask_ratio must be 1, 2 or 4")
    visible = decoded["visible"]
    ids = decoded["instance_ids"]
    h, w = decoded["original_hw"]
    if rgb.shape != (h, w, 3) or visible.shape != (len(ids), h, w):
        raise TrainingContractError(f"RGB/mask size mismatch for {metadata.get('image_id')}")
    native_area = visible.sum(axis=(1, 2), dtype=np.int64)
    if np.any(native_area == 0):
        raise TrainingContractError(f"Native empty V instance in {metadata.get('image_id')}")
    scale = min(imgsz / h, imgsz / w)
    nh, nw = max(1, round(h * scale)), max(1, round(w * scale))
    top, left = (imgsz - nh) // 2, (imgsz - nw) // 2
    image = np.full((imgsz, imgsz, 3), 114, dtype=np.uint8)
    image[top:top + nh, left:left + nw] = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_LINEAR)
    full_v = np.zeros((len(ids), imgsz, imgsz), dtype=np.uint8)
    full_v[:, top:top + nh, left:left + nw] = _resize_masks(visible, (nh, nw))
    full_area = full_v.sum(axis=(1, 2), dtype=np.int64)
    low_v = _resize_masks(full_v, (imgsz // mask_ratio, imgsz // mask_ratio))
    low_area = low_v.sum(axis=(1, 2), dtype=np.int64)
    lost = [ids[i] for i in range(len(ids)) if full_area[i] == 0 or low_area[i] == 0]
    if lost:
        raise TrainingContractError(
            f"Visible GT disappeared during resize/downsample: image={metadata.get('image_id')}, "
            f"ids={lost}, imgsz={imgsz}, mask_ratio={mask_ratio}; no GT was dropped")
    # BCEDiceLoss uses nearest interpolation to P3/8. Record loss of each tiny
    # instance in the auxiliary target, even when its instance target survives.
    semantic_area = _resize_masks(low_v, (imgsz // 8, imgsz // 8)).sum(axis=(1, 2), dtype=np.int64)
    semantic_lost = [ids[i] for i in range(len(ids)) if semantic_area[i] == 0]
    boxes = tight_boxes(full_v)
    xywh = np.empty_like(boxes)
    xywh[:, :2] = (boxes[:, :2] + boxes[:, 2:]) / (2 * imgsz)
    xywh[:, 2:] = (boxes[:, 2:] - boxes[:, :2]) / imgsz
    amodal_area = decoded["amodal"].sum(axis=(1, 2), dtype=np.int64)
    info = {**metadata, "instance_ids": ids, "native_visible_area": native_area.tolist(),
            "native_amodal_area_audit_only": amodal_area.tolist(), "resized_visible_area": full_area.tolist(),
            "loss_mask_visible_area": low_area.tolist(), "semantic_p3_visible_area": semantic_area.tolist(),
            "semantic_p3_vanished_ids": semantic_lost, "mask_ratio": mask_ratio,
            "visible_connected_components_8": {"native": _component_counts(visible),
                                               "resized": _component_counts(full_v),
                                               "loss_mask": _component_counts(low_v)},
            "component_count_note": "Resampling may merge or erase fragments; counts are audited, no hull or GT edits",
            "letterbox": {"native_hw": [h, w], "resized_hw": [nh, nw], "pad_top_left": [top, left]},
            "bbox_source": "tight resized visible raster, half-open xyxy", "amodal_used_in_loss": False}
    return {
        "img": torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))).float().div_(255),
        "masks": torch.from_numpy(low_v), "bboxes": torch.from_numpy(xywh),
        "cls": torch.zeros((len(ids), 1), dtype=torch.float32),
        # Single class index map; the stock loss separately builds foreground
        # presence from the per-instance V masks, so empty images remain zero.
        "sem_masks": torch.zeros((imgsz // mask_ratio, imgsz // mask_ratio), dtype=torch.int64),
        "metadata": info,
    }


class NativeVisibleDataset(Dataset):
    def __init__(self, manifest_path: str | Path, *, imgsz: int = 640, mask_ratio: int = 2,
                 verify_hashes: bool = True):
        self.manifest_path = Path(manifest_path).resolve()
        self.manifest_sha256 = sha256_file(self.manifest_path)
        self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if self.manifest.get("schema") != MANIFEST_SCHEMA:
            raise TrainingContractError("Unsupported manifest schema")
        if self.manifest.get("split") != "train" or self.manifest.get("role") != "train":
            raise TrainingContractError("Training accepts only official train split/role; VAL and test are forbidden")
        self.rows = self.manifest["samples"]
        if not self.rows or any(row.get("split") != "train" for row in self.rows):
            raise TrainingContractError("Empty manifest or non-train sample found")
        self.imgsz, self.mask_ratio, self.verify_hashes = imgsz, mask_ratio, verify_hashes
        self._verified: set[int] = set()

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        if self.verify_hashes and index not in self._verified:
            for path_key, hash_key in (("rgb_path", "rgb_sha256"), ("mask_path", "mask_sha256")):
                if sha256_file(row[path_key]) != row[hash_key]:
                    raise TrainingContractError(f"SHA256 mismatch: {row[path_key]}")
            self._verified.add(index)
        decoded = decode_packed_masks(row["mask_path"])
        if decoded["original_hw"] != row["original_hw"] or len(decoded["instance_ids"]) != row["instance_count"]:
            raise TrainingContractError(f"Manifest/NPZ disagreement: {row['image_id']}")
        with Image.open(row["rgb_path"]) as source:
            rgb = np.asarray(source.convert("RGB"))
        meta = {k: row[k] for k in ("image_id", "rgb_path", "mask_path", "rgb_sha256", "mask_sha256", "split")}
        return prepare_raster_sample(rgb, decoded, imgsz=self.imgsz, mask_ratio=self.mask_ratio, metadata=meta)

    def audit(self) -> dict[str, Any]:
        records, failures = [], []
        for index in range(len(self)):
            try:
                records.append(self[index]["metadata"])
            except TrainingContractError as error:
                failures.append({"image_id": self.rows[index]["image_id"], "error": str(error)})
        return {"manifest_path": str(self.manifest_path), "manifest_sha256": self.manifest_sha256,
                "image_count": len(self), "instance_count": sum(row["instance_count"] for row in self.rows),
                "mask_ratio": self.mask_ratio, "imgsz": self.imgsz, "records": records, "failures": failures,
                "semantic_p3_vanished_instance_count": sum(len(x["semantic_p3_vanished_ids"]) for x in records),
                "passed": not failures, "GT_dropped": 0, "A_in_loss": False}


def collate_native(samples: list[dict[str, Any]]) -> dict[str, Any]:
    return {"img": torch.stack([x["img"] for x in samples]),
            "masks": torch.cat([x["masks"] for x in samples], dim=0),
            "bboxes": torch.cat([x["bboxes"] for x in samples], dim=0),
            "cls": torch.cat([x["cls"] for x in samples], dim=0),
            "batch_idx": torch.cat([torch.full((len(x["cls"]),), i, dtype=torch.int64) for i, x in enumerate(samples)]),
            "sem_masks": torch.stack([x["sem_masks"] for x in samples]),
            "metadata": [x["metadata"] for x in samples]}


def transfer_apple_head(source: torch.nn.Module, target: torch.nn.Module) -> dict[str, Any]:
    """Use installed name-aware remap, then explicitly migrate semantic apple row."""
    source_names = source.names
    apple_indices = [int(k) for k, value in source_names.items() if str(value).strip().lower() == "apple"]
    if len(apple_indices) != 1:
        raise TrainingContractError("Official checkpoint must identify exactly one COCO apple class")
    apple = apple_indices[0]
    target.names = {0: "apple"}
    src = source.state_dict()
    target.load(source, verbose=False)
    dst = target.state_dict()
    copied, remapped, fresh = [], [], []
    semantic_prefix = None
    for name, module in target.named_modules():
        if name.endswith(".proto.semseg"):
            semantic_prefix = f"{name}.{len(module) - 1}."
    if semantic_prefix is None:
        raise TrainingContractError("Unfused Proto26 semantic training branch missing")
    with torch.no_grad():
        for key in dst:
            if key.startswith(semantic_prefix) and key in src and src[key].shape[1:] == dst[key].shape[1:]:
                dst[key].copy_(src[key][apple:apple + 1].to(dst[key].dtype))
    for key, value in dst.items():
        if key in src and value.shape == src[key].shape and torch.equal(value, src[key]):
            copied.append(key)
        elif (key in src and value.ndim and src[key].shape[0] == len(source_names)
              and value.shape[0] == 1 and value.shape[1:] == src[key].shape[1:]
              and (".cv3." in key or ".one2one_cv3." in key or key.startswith(semantic_prefix))):
            if not torch.equal(value, src[key][apple:apple + 1]):
                raise TrainingContractError(f"Expected apple-row transfer failed: {key}")
            remapped.append(key)
        else:
            fresh.append(key)
    cls_remapped = [key for key in remapped if ".cv3." in key or ".one2one_cv3." in key]
    if len(cls_remapped) != 12:
        raise TrainingContractError(f"Expected 6 class convs × weight/bias (12 rows), found {cls_remapped}")
    return {"source_class_count": len(source_names), "source_apple_class": apple, "target_class": 0,
            "target_names": {0: "apple"}, "exact_copied_tensors": copied, "apple_row_remapped_tensors": remapped,
            "fresh_or_structurally_changed_tensors": fresh, "source_full_head_strict_transfer": False,
            "method": "Ultralytics BaseModel.load name remap plus explicit Proto26 semantic apple row"}


def build_visible_model(weights: str | Path, *, epochs: int = 50) -> tuple[torch.nn.Module, dict[str, Any]]:
    from ultralytics import YOLO
    from ultralytics.cfg import get_cfg
    from ultralytics.nn.tasks import SegmentationModel

    weight_hash = sha256_file(weights)
    if weight_hash != OFFICIAL_SHA256:
        raise TrainingContractError(f"Unexpected official initialization hash: {weight_hash}")
    source = YOLO(str(weights), task="segment").model.cpu().float()
    if source.model[-1].__class__.__name__ != "Segment26" or not hasattr(source.model[-1], "cv4"):
        raise TrainingContractError("Expected complete, unfused YOLO26 segmentation training weights")
    model = SegmentationModel(cfg=copy.deepcopy(source.yaml), ch=3, nc=1, verbose=False)
    migration = transfer_apple_head(source, model)
    model.args = get_cfg(overrides={"task": "segment", "overlap_mask": False, "epochs": epochs})
    # Checkpoint loading may mark source parameters inference-only. New target
    # parameters must remain trainable (except the fixed DFL projection).
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(".dfl." not in name)
    migration["official_weights_path"] = str(Path(weights).resolve())
    migration["official_weights_sha256"] = weight_hash
    migration["trainable_parameter_count"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    migration["frozen_parameter_names"] = [n for n, p in model.named_parameters() if not p.requires_grad]
    return model, migration


def parameter_group(name: str) -> str:
    if ".one2one_cv2." in name:
        return "one2one_box"
    if ".one2one_cv3." in name:
        return "one2one_cls"
    if ".one2one_cv4." in name:
        return "one2one_mask_coeff"
    if ".proto.semseg." in name:
        return "semantic_aux"
    if ".proto." in name:
        return "mask_proto"
    if name.startswith("model.23.cv2."):
        return "one2many_box"
    if name.startswith("model.23.cv3."):
        return "one2many_cls"
    if name.startswith("model.23.cv4."):
        return "one2many_mask_coeff"
    return "backbone_neck"


def gradient_audit(model: torch.nn.Module) -> dict[str, Any]:
    groups: dict[str, dict[str, Any]] = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        group = groups.setdefault(parameter_group(name), {"trainable_tensors": 0, "grad_tensors": 0,
                                                          "nonzero_grad_tensors": 0, "sum_squared": 0.0})
        group["trainable_tensors"] += 1
        if parameter.grad is None:
            continue
        grad = parameter.grad.detach()
        if not bool(torch.isfinite(grad).all()):
            raise TrainingContractError(f"Nonfinite gradient in {name}")
        group["grad_tensors"] += 1
        squared = float(grad.float().square().sum())
        group["sum_squared"] += squared
        group["nonzero_grad_tensors"] += int(squared > 0)
    for group in groups.values():
        group["l2_norm"] = math.sqrt(group.pop("sum_squared"))
    return groups


def compute_e2e_loss(criterion: Any, predictions: Any, batch: dict[str, Any]):
    """Stock E2ELoss arithmetic, exposing both branches for the smoke audit."""
    parsed = criterion.one2many.parse_output(predictions)
    many, many_items = criterion.one2many.loss(parsed["one2many"], batch)
    one, one_items = criterion.one2one.loss(parsed["one2one"], batch)
    vector = many * criterion.o2m + one * criterion.o2o
    logs = {"one2many": {k: float(v) for k, v in many_items.items()},
            "one2one": {k: float(v) for k, v in one_items.items()},
            "o2m_weight": criterion.o2m, "o2o_weight": criterion.o2o,
            "weighted_loss_vector": vector.detach().float().cpu().tolist()}
    return vector.sum(), logs


def load_local_checkpoint(path: str | Path) -> torch.nn.Module:
    """Strict native checkpoint load; returned class is apple=0, never COCO47."""
    from ultralytics.cfg import get_cfg
    from ultralytics.nn.tasks import SegmentationModel
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("schema") != CHECKPOINT_SCHEMA:
        raise TrainingContractError("Not a local visible-segmentation checkpoint")
    model = SegmentationModel(cfg=checkpoint["model_yaml"], ch=3, nc=1, verbose=False)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.names = {0: "apple"}
    model.args = get_cfg(overrides=checkpoint["model_args"])
    return model


def prediction_tensors(output: Any, prefix: str = "output") -> dict[str, torch.Tensor]:
    """Flatten a native Segment26 prediction tree for a functional reload check."""
    if isinstance(output, torch.Tensor):
        return {prefix: output.detach().float().cpu()}
    flattened: dict[str, torch.Tensor] = {}
    if isinstance(output, dict):
        for key, value in output.items():
            flattened.update(prediction_tensors(value, f"{prefix}.{key}"))
    elif isinstance(output, (list, tuple)):
        for index, value in enumerate(output):
            flattened.update(prediction_tensors(value, f"{prefix}[{index}]"))
    return flattened


def run_smoke(*, repo: Path, manifest: Path, weights: Path, output: Path, steps: int = 12,
              batch_size: int = 2, imgsz: int = 640, mask_ratio: int = 2, seed: int = 20261104,
              lr: float = 1e-4, device: str = "cuda:0", inspect_data_only: bool = False,
              amp_init_scale: float = 1024.0) -> dict[str, Any]:
    """Bounded engineering smoke only: never turns into long training automatically."""
    if not 1 <= steps <= 32:
        raise TrainingContractError("This smoke entry permits 1..32 optimizer steps only")
    if batch_size < 1:
        raise TrainingContractError("batch_size must be positive")
    if not math.isfinite(amp_init_scale) or amp_init_scale <= 0:
        raise TrainingContractError("amp_init_scale must be finite and positive")
    configure_f_runtime(repo)
    manifest, weights, output = map(require_f_path, (manifest, weights, output))
    if output.exists() and any(output.iterdir()):
        raise TrainingContractError(f"Refusing to overwrite a nonempty run directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    dataset = NativeVisibleDataset(manifest, imgsz=imgsz, mask_ratio=mask_ratio)
    for row in dataset.rows:
        require_f_path(row["rgb_path"])
        require_f_path(row["mask_path"])
    audit = dataset.audit()
    write_json(output / "data_audit.json", audit)
    if not audit["passed"]:
        write_json(output / "status.json", {"state": "data_contract_failed", "details": "data_audit.json"})
        raise TrainingContractError("Data preflight failed; see data_audit.json. No model or GPU was started.")
    if inspect_data_only:
        write_json(output / "status.json", {"state": "data_audit_complete", "GPU_used": False})
        return audit
    selected_device = torch.device(device)
    if selected_device.type != "cuda" or not torch.cuda.is_available():
        raise TrainingContractError("Training smoke requires the root-controlled CUDA slot; CPU mode is audit only")
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    model, migration = build_visible_model(weights)
    write_json(output / "initialization.json", migration)
    model.to(selected_device).train()
    model.criterion = model.init_criterion()
    criterion = model.criterion
    if not all(hasattr(criterion, key) for key in ("one2many", "one2one", "o2m", "o2o")):
        raise TrainingContractError("Expected native YOLO26 dual-assignment E2ELoss")
    initial = {n: p.detach().cpu().clone() for n, p in model.named_parameters() if p.requires_grad}
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=True, init_scale=amp_init_scale)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False,
                        persistent_workers=False, collate_fn=collate_native,
                        generator=torch.Generator().manual_seed(seed), drop_last=False)
    config = {"mode": "bounded_smoke_not_accuracy_training", "steps": steps, "batch_size": batch_size,
              "imgsz": imgsz, "mask_ratio": mask_ratio, "seed": seed, "lr": lr, "optimizer": "AdamW",
              "weight_decay": 1e-4, "gradient_clip_norm": 10.0, "amp": True, "device": str(selected_device),
              "amp_init_scale": amp_init_scale, "nonfinite_gradient_policy": "fail immediately; never silently skip update",
              "workers": 0, "pin_memory": False, "augmentations": "letterbox only; no mosaic/mixup/hflip",
              "overlap_mask": False, "model_classes": {0: "apple"}, "amodal_in_loss": False,
              "semantic_target": "V union with class index 0; stock Proto26 semantic auxiliary loss",
              "manifest_sha256": dataset.manifest_sha256, "manifest_path": str(manifest),
              "official_weights_sha256": migration["official_weights_sha256"],
              "adapter_source_sha256": sha256_file(__file__), "torch_version": torch.__version__,
              "ultralytics_version": __import__("ultralytics").__version__,
              "cudnn_deterministic": True, "cudnn_benchmark": False,
              "e2e_epoch_schedule_total": 50, "GT_or_test_split_changed": False}
    write_json(output / "config.json", config)
    write_json(output / "status.json", {"state": "running", "completed_steps": 0})
    torch.cuda.reset_peak_memory_stats(selected_device)
    records, positive_groups = [], set()
    epoch, step = 0, 0
    start = time.perf_counter()
    try:
        while step < steps:
            for batch in loader:
                torch.cuda.synchronize(selected_device)
                tick = time.perf_counter()
                metadata = batch.pop("metadata")
                batch = {k: value.to(selected_device, non_blocking=False) for k, value in batch.items()}
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    predictions = model(batch["img"])
                    loss, loss_log = compute_e2e_loss(criterion, predictions, batch)
                if not bool(torch.isfinite(loss)):
                    raise TrainingContractError(f"Nonfinite loss at optimizer step {step + 1}")
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                try:
                    gradients = gradient_audit(model)
                except TrainingContractError as error:
                    write_json(output / "nonfinite_gradient.json", {
                        "attempted_step": step + 1, "completed_optimizer_updates": step,
                        "epoch_index": epoch, "total_loss": float(loss.detach()), **loss_log,
                        "amp_scale": scaler.get_scale(), "amp_init_scale": amp_init_scale,
                        "samples": metadata, "error": str(error),
                        "optimizer_update_performed_for_failed_step": False,
                        "policy": "Fail immediately; no silent AMP skipped step or changed loss/GT/LR"})
                    raise
                positive_groups.update(k for k, v in gradients.items() if v["nonzero_grad_tensors"])
                grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0, error_if_nonfinite=True))
                scale_before = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                if scaler.get_scale() < scale_before:
                    raise TrainingContractError("AMP skipped an optimizer step; smoke must not count skipped updates")
                torch.cuda.synchronize(selected_device)
                step += 1
                record = {"step": step, "epoch_index": epoch, "total_loss": float(loss.detach()), **loss_log,
                          "step_ms_including_gradient_audit_excluding_loader": (time.perf_counter() - tick) * 1000,
                          "grad_norm_before_clip": grad_norm, "amp_scale": scaler.get_scale(),
                          "peak_allocated_bytes": torch.cuda.max_memory_allocated(selected_device),
                          "peak_reserved_bytes": torch.cuda.max_memory_reserved(selected_device),
                          "gradient_groups": gradients, "samples": metadata}
                records.append(record)
                with (output / "steps.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                write_json(output / "status.json", {"state": "running", "completed_steps": step, "target_steps": steps})
                print(json.dumps({"step": step, "loss": record["total_loss"],
                                  "step_ms": record["step_ms_including_gradient_audit_excluding_loader"],
                                  "peak_allocated_MB": record["peak_allocated_bytes"] / 1e6}), flush=True)
                if step >= steps:
                    break
            epoch += 1
            criterion.update()
        required_groups = {"one2many_box", "one2many_cls", "one2many_mask_coeff", "one2one_box",
                           "one2one_cls", "one2one_mask_coeff", "mask_proto", "semantic_aux", "backbone_neck"}
        if not required_groups <= positive_groups:
            raise TrainingContractError(f"Missing nonzero branch gradients: {required_groups - positive_groups}")
        changed: dict[str, list[str]] = {}
        for name, parameter in model.named_parameters():
            if name in initial and not torch.equal(initial[name], parameter.detach().cpu()):
                changed.setdefault(parameter_group(name), []).append(name)
        if not required_groups <= set(changed):
            raise TrainingContractError(f"No actual parameter update in groups: {required_groups - set(changed)}")
        state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        checkpoint = {"schema": CHECKPOINT_SCHEMA, "state_dict": state, "model_yaml": copy.deepcopy(model.yaml),
                      "model_args": {"task": "segment", "overlap_mask": False, "epochs": 50},
                      "names": {0: "apple"}, "config": config, "initialization": migration,
                      "steps": step, "optimizer_state": optimizer.state_dict(), "scaler_state": scaler.state_dict(),
                      "metrics": None, "warning": "Smoke checkpoint, no accuracy claim or deployment promotion"}
        checkpoint_path = output / "smoke_last.pt"
        torch.save(checkpoint, checkpoint_path)
        reloaded = load_local_checkpoint(checkpoint_path)
        reload_equal = all(torch.equal(value, reloaded.state_dict()[key]) for key, value in state.items())
        if not reload_equal:
            raise TrainingContractError("Strict reload did not preserve checkpoint state tensors exactly")
        # One fixed TRAIN image, inference-only, no BN updates. Check native
        # checkpoint usability separately from any accuracy or speed benchmark.
        probe = dataset[0]
        probe_image = probe["img"].unsqueeze(0).to(selected_device)
        with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.float16):
            before = prediction_tensors(model.eval()(probe_image))
        model.cpu()  # never keep two full local models resident on the GPU
        reloaded.to(selected_device).eval()
        with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.float16):
            after = prediction_tensors(reloaded(probe_image))
        if not before or before.keys() != after.keys():
            raise TrainingContractError("Checkpoint inference output tree mismatch")
        inference_records = []
        for key, value in before.items():
            other = after[key]
            finite = bool(torch.isfinite(value).all() and torch.isfinite(other).all())
            agrees = value.shape == other.shape and torch.allclose(value, other, atol=1e-5, rtol=1e-5)
            if not finite or not agrees:
                raise TrainingContractError(f"Reload inference mismatch/nonfinite tensor: {key}")
            inference_records.append({"key": key, "shape": list(value.shape), "finite": finite,
                                      "max_absolute_difference": float((value - other).abs().max()) if value.numel() else 0.0})
        write_json(output / "checkpoint_inference_check.json", {
            "image": probe["metadata"], "mode": "eval + inference_mode + FP16 autocast", "GT_used_for_prediction": False,
            "all_outputs_finite_and_close": True, "atol": 1e-5, "rtol": 1e-5, "tensors": inference_records,
            "BN_updated": False, "accuracy_or_speed_measurement": False, "class_mapping": {0: "apple"}})
        summary = {"state": "smoke_passed", "steps_completed": step, "data_preflight_passed": True,
                   "finite_losses_and_gradients": True, "both_assignment_branches_received_gradients": True,
                   "changed_parameter_tensors_by_group": changed, "strict_reload_all_tensors_equal": reload_equal,
                   "reload_eval_forward_finite_and_close": True,
                   "exact_training_resume_supported": False,
                   "checkpoint_path": str(checkpoint_path), "checkpoint_sha256": sha256_file(checkpoint_path),
                   "trainable_parameter_count": migration["trainable_parameter_count"],
                   "peak_allocated_bytes": torch.cuda.max_memory_allocated(selected_device),
                   "wall_seconds_training_and_checkpoint": time.perf_counter() - start,
                   "accuracy_measured": False, "deployment_replacement": False,
                   "A_branch_trained": False, "D405_real_data_used": False,
                   "semantic_p3_vanished_instance_count": audit["semantic_p3_vanished_instance_count"]}
        write_json(output / "summary.json", summary)
        write_json(output / "status.json", {"state": "smoke_passed", "completed_steps": step})
        return summary
    except Exception as error:
        write_json(output / "status.json", {"state": "failed", "completed_steps": step,
                                           "error_type": type(error).__name__, "error": str(error)})
        raise
