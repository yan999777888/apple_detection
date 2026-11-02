"""Complete original VAL for one explicitly selected Stage 2 candidate.

Only scoped factories are replaced; the frozen evaluator still owns model
restoration, original GT/RGB auditing, native-coordinate output, complete PR
calibration and fusion. This entry point never uses the .738 deployment cut.
Four-view .35 results must be derived separately from these uncut predictions
and compared with their uncut control. Neither is automatically promoted.
"""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import evaluate_premask as premask

fast = premask.fast
CONFIG_KEYS = {"candidate_policy", "preprocess_device", "channels_last", "cudnn_benchmark",
               "cudnn_deterministic", "roi_align_precision"}
EXTRA_SOURCES = (*premask.EXTRA_SOURCES, "scripts/evaluate_stage2.py", "deployment/stage2.py")
PROTECTED_P90 = {"precision": 3105 / 3288, "recall": 3105 / 3449,
                 "TP": 3105, "FP": 183, "FN": 344}


def normalize_stage2_config(value):
    if not isinstance(value, dict) or set(value) != CONFIG_KEYS:
        raise ValueError("Stage2 config requires exactly: " + ", ".join(sorted(CONFIG_KEYS)))
    result = deepcopy(value)
    result["candidate_policy"] = premask.normalize_policy(value["candidate_policy"])
    if result["preprocess_device"] not in ("cpu", "cuda"):
        raise ValueError("preprocess_device must be cpu or cuda")
    if result["roi_align_precision"] not in ("fp32", "fp16"):
        raise ValueError("roi_align_precision must be fp32 or fp16")
    for key in ("channels_last", "cudnn_benchmark", "cudnn_deterministic"):
        if type(result[key]) is not bool:
            raise ValueError(key + " must be boolean")
    return result


def load_stage2_config(path):
    path = Path(path).resolve()
    if path.drive.upper() != "F:":
        raise ValueError("Stage2 config must remain on F:")
    before = fast.sha(path)
    config = normalize_stage2_config(json.loads(path.read_text(encoding="utf-8-sig"),
                                               object_pairs_hook=premask._no_duplicate_keys))
    if fast.sha(path) != before:
        raise RuntimeError("Stage2 config changed while reading")
    return config, {"path": str(path), "sha256": before}


def build_parser(original_builder=None):
    parser = (original_builder or fast.build_parser)()
    parser.description = __doc__
    parser.add_argument("--stage2-config", type=Path, required=True,
                        help="F: JSON binding candidate_policy, preprocess_device, channels_last, cuDNN and ROIAlign settings")
    parser.set_defaults(pre_nms_threshold=None, roi_chunk_size=64)
    return parser


def validate_args(args, config=None):
    premask.validate_args(args)
    if args.precision != "fp16" or args.roi_precision != "autocast":
        raise ValueError("Stage2 full VAL requires the screened FP16 autocast/ROI autocast setting")
    if (args.short_side, args.max_size, args.roi_chunk_size) != (1024, 1280, 64):
        raise ValueError("Stage2 full VAL preserves screened 1024/1280 and ROI chunk64")
    if config is not None:
        normalize_stage2_config(config)


def backend_state(torch):
    return {"cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled()}


@contextmanager
def scoped_backend(torch, config):
    original = backend_state(torch)
    try:
        torch.backends.cudnn.benchmark = config["cudnn_benchmark"]
        torch.backends.cudnn.deterministic = config["cudnn_deterministic"]
        # Match the separately measured screen; these are not additional knobs.
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = False
        yield backend_state(torch)
    finally:
        torch.backends.cudnn.benchmark = original["cudnn_benchmark"]
        torch.backends.cudnn.deterministic = original["cudnn_deterministic"]
        torch.backends.cudnn.allow_tf32 = original["cudnn_allow_tf32"]
        torch.backends.cuda.matmul.allow_tf32 = original["cuda_matmul_allow_tf32"]


def gpu_dataset_type(original_type, prepare_pixels, torch):
    """Use genuine native targets, with only raw RGB preprocessing moved.

The frozen loop accepts CHW pixels on CPU or GPU. Its original input_hw and
flip-back remain correct. Copying preprocessed pixels to CPU for a SHA is
deliberately outside its timer; full-VAL timing is not a deployment benchmark.
No GT value is passed to prepare_pixels or the inference adapter.
"""
    class GPUInputDataset(original_type):
        def __getitem__(self, index):
            from PIL import Image
            from apple_seg.od_data import LEGACY_RGB_PREPROCESS, resized_hw
            if self.augment or len(self.short_sides) != 1:
                raise ValueError("Stage2 GPU preprocessing only supports deterministic validation")
            target = self.native_dataset[index]["target"]
            source = dict(target["source"])
            if source.get("horizontal_flip", False):
                raise ValueError("Native annotation view must be unflipped")
            height, width = target["original_hw"]
            with Image.open(Path(source["rgb_path"])) as image:
                rgb = image.convert("RGB")
                crop = source.get("crop_xyxy", (0, 0, *rgb.size))
                if len(crop) != 4 or any(type(v) is not int for v in crop):
                    raise ValueError("Source crop must be an integer xyxy window")
                x1, y1, x2, y2 = crop
                if not (0 <= x1 < x2 <= rgb.width and 0 <= y1 < y2 <= rgb.height):
                    raise ValueError("Source crop is outside original RGB")
                rgb = rgb.crop(tuple(crop))
                if rgb.size != (width, height):
                    raise ValueError("Source RGB crop and native mask sizes differ")
                pixels = prepare_pixels(rgb, self.short_sides[0], self.max_size, "cuda",
                                        preprocess_device="cuda", channels_last=False)[0]
            new_h, new_w = pixels.shape[-2:]
            if (new_h, new_w) != resized_hw(height, width, self.short_sides[0], self.max_size):
                raise RuntimeError("Stage2 preprocessing changed the original resize dimensions")
            if pixels.dtype != torch.float32 or pixels.device.type != "cuda":
                raise RuntimeError("GPU preprocessing must return genuine CUDA FP32 normalized input")
            source.update(resize_hw=[new_h, new_w], horizontal_flip=False,
                          original_to_input_scale_xy=[new_w / width, new_h / height],
                          resize_policy="torchvision_GeneralizedRCNNTransform_float32_normalize_then_resize",
                          **LEGACY_RGB_PREPROCESS,
                          preprocessing_device="cuda", preprocessing_backend="deployment.stage2.prepare_pixels",
                          normalized_input_sha256=hashlib.sha256(
                              memoryview(pixels.detach().cpu().contiguous().numpy())).hexdigest())
            return {"pixel_values": pixels, "pixel_mask": torch.ones((new_h, new_w), dtype=torch.bool),
                    "target": {**target, "source": source}}
    return GPUInputDataset


@contextmanager
def patched_evaluator(args, *, adapter_type=None, adapter_module=None, data_module=None,
                      torch_module=None, prepare_pixels=None):
    config, binding = load_stage2_config(args.stage2_config)
    validate_args(args, config)
    checkpoints = premask.checkpoint_bindings(args.config)
    if adapter_type is None or prepare_pixels is None:
        from deployment.stage2 import Stage2DeploymentAdapter, prepare_pixels as real_prepare
        adapter_type = adapter_type or Stage2DeploymentAdapter
        prepare_pixels = prepare_pixels or real_prepare
    if adapter_module is None:
        import deployment.strong_fast as adapter_module
    if data_module is None:
        import apple_seg.od_data as data_module
    if torch_module is None:
        import torch as torch_module
    original = {name: getattr(fast, name) for name in
                ("build_parser", "deployment_spec", "source_inventory", "final_integrity")}
    old_adapter, old_dataset = adapter_module.FastDeploymentAdapter, data_module.ODInputDataset
    layout = torch_module.channels_last if config["channels_last"] else torch_module.contiguous_format

    class ConfiguredAdapter(adapter_type):
        def predict_views(self, pixels, pixel_mask=None):
            # Frozen paired cat/flip may create a different tensor layout.
            return super().predict_views(pixels.contiguous(memory_format=layout), pixel_mask)

    def factory(model, **kwargs):
        model.to(memory_format=layout)
        return ConfiguredAdapter(model, candidate_policy=deepcopy(config["candidate_policy"]),
                                 capture_candidates=False, profile_stages=False, profile_pools=False,
                                 roi_align_precision=config["roi_align_precision"], **kwargs)

    def inventory():
        result = original["source_inventory"]()
        for relative in EXTRA_SOURCES:
            path = REPO / relative
            result[str(path.relative_to(REPO))] = fast.sha(path)
        return result

    def spec(arguments):
        from deployment.candidate_policy import policy_description
        result = original["deployment_spec"](arguments)
        candidate = config["candidate_policy"]
        limited = any(v is not None for v in candidate.values())
        result.update(stage2_configuration=deepcopy(config), stage2_config_file=deepcopy(binding),
                      stage2_runtime_numerics=backend_state(torch_module),
                      candidate_policy=deepcopy(candidate),
                      candidate_policy_semantics=policy_description(**candidate),
                      candidate_selection_stage="after decoded valid boxes, before V/H/A mask ROI inference",
                      candidate_selection_uses_gt=False, candidate_audit_capture=False,
                      complete_original_candidate_set=not limited,
                      deployment_experiment="stage2_original_strong_full_val_v1",
                      original_weight_bindings=deepcopy(checkpoints), automatic_promotion=False,
                      protected_original_FP32_four_view_P90=deepcopy(PROTECTED_P90),
                      user_hard_abort_precision_floor=.94,
                      hard_abort_floor_is_promotion_threshold=False,
                      fusion_cutoff_035_evidence="Must be derived from these complete fresh four layers and re-fused; retain uncut comparison; equality not assumed")
        result["evaluation_policy"].update(
            curve_scope="All model-returned scores after configured pre-mask policy and original final-score/top-k/dual-mask-NMS; no additional final-score cutoff",
            candidate_budget_restricts_recall=limited, original_unfiltered_PR_claimed=False,
            full_PR_final_score_cutoff=None, threshold_0738_full_PR_forbidden=True)
        return result

    def integrity(metadata, sources):
        result = original["final_integrity"](metadata, sources)
        premask.assert_bindings_unchanged(binding, checkpoints)
        if backend_state(torch_module) != metadata["deployment"]["stage2_runtime_numerics"]:
            raise RuntimeError("Stage2 backend numerical flags changed during full VAL")
        result.update(stage2_configuration_before_after_identical=True,
                      original_weights_before_after_identical=True,
                      original_weight_bindings=deepcopy(checkpoints))
        return result

    with scoped_backend(torch_module, config):
        try:
            fast.build_parser = lambda: build_parser(original["build_parser"])
            fast.deployment_spec, fast.source_inventory, fast.final_integrity = spec, inventory, integrity
            adapter_module.FastDeploymentAdapter = factory
            if config["preprocess_device"] == "cuda":
                data_module.ODInputDataset = gpu_dataset_type(old_dataset, prepare_pixels, torch_module)
            yield {"stage2_configuration": config, "stage2_config_file": binding,
                   "original_weight_bindings": checkpoints}
        finally:
            adapter_module.FastDeploymentAdapter, data_module.ODInputDataset = old_adapter, old_dataset
            for name, function in original.items():
                setattr(fast, name, function)


def main(argv=None):
    args = build_parser().parse_args(argv)
    validate_args(args)
    with patched_evaluator(args):
        return fast.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
