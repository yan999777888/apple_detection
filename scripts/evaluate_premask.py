"""Full original VAL for an explicitly bound, pre-mask candidate policy.

This entry point reuses the audited evaluator without modifying its source.
Its temporary factory installs PreMaskDeploymentAdapter; GT stays exclusively
inside the evaluator. All surviving low-score outputs are retained for PR.
The original Base/Small checkpoints, data contract and fusion rules are fixed.
"""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
import json
import math
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import evaluate_fast_deployment as fast


POLICY_KEYS = ("score_threshold", "box_nms_iou", "topk")
EXTRA_SOURCES = ("scripts/evaluate_premask.py", "deployment/pre_mask.py",
                 "deployment/candidate_policy.py")


def normalize_policy(value):
    if not isinstance(value, dict) or set(value) - set(POLICY_KEYS):
        raise ValueError("Candidate policy must contain only score_threshold, box_nms_iou and topk")
    result = {key: value.get(key) for key in POLICY_KEYS}
    for key in ("score_threshold", "box_nms_iou"):
        number = result[key]
        if number is not None:
            if isinstance(number, bool) or not isinstance(number, (int, float)) or not math.isfinite(number) or not 0 <= number <= 1:
                raise ValueError(key + " must be null or a finite probability in [0,1]")
            result[key] = float(number)
    topk = result["topk"]
    if topk is not None and (type(topk) is not int or topk < 1):
        raise ValueError("topk must be null or a positive integer")
    return result


def _no_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate candidate-policy key: " + key)
        result[key] = value
    return result


def load_policy(path):
    path = Path(path).resolve()
    if path.drive.upper() != "F:":
        raise ValueError("Candidate policy file must remain on F:")
    before = fast.sha(path)
    value = normalize_policy(json.loads(path.read_text(encoding="utf-8-sig"), object_pairs_hook=_no_duplicate_keys))
    if fast.sha(path) != before:
        raise RuntimeError("Candidate policy changed while reading")
    return value, {"path": str(path), "sha256": before}


def build_parser(original_builder=None):
    parser = (original_builder or fast.build_parser)()
    parser.description = __doc__
    parser.add_argument("--candidate-policy", type=Path, required=True,
                        help="F: JSON with score_threshold, box_nms_iou, topk (null disables each)")
    parser.set_defaults(pre_nms_threshold=None, roi_chunk_size=64)
    return parser


def validate_args(args):
    # Keep this experiment about mask-before selection. Post-score truncation
    # would confound candidate audits and fusion, and could hide a failed R90.
    if args.pre_nms_threshold is not None or args.fixed_only:
        raise ValueError("Pre-mask full VAL retains all returned scores: use --pre-nms-threshold none and no --fixed-only")
    if args.skip_dead_amodal:
        raise ValueError("Original strong models require their genuine amodal refinement")
    fast.deployment_spec(args)


def checkpoint_bindings(config_path):
    path = Path(config_path).resolve()
    config = json.loads(path.read_text(encoding="utf-8-sig"))
    root = (path.parent / config["path_base"]).resolve()
    result = {}
    for name in ("base", "small"):
        checkpoint = (root / config["paths"][name + "_checkpoint"]).resolve()
        if checkpoint.drive.upper() != "F:":
            raise ValueError("Original checkpoint must remain on F:")
        digest = fast.sha(checkpoint)
        if digest != config[name + "_checkpoint_sha256"]:
            raise ValueError("Original " + name + " checkpoint hash differs")
        result[name] = {"path": str(checkpoint), "sha256": digest}
    return result


def assert_bindings_unchanged(policy_binding, checkpoints):
    if fast.sha(policy_binding["path"]) != policy_binding["sha256"]:
        raise RuntimeError("Candidate policy changed during full VAL")
    for name, binding in checkpoints.items():
        if fast.sha(binding["path"]) != binding["sha256"]:
            raise RuntimeError("Original " + name + " checkpoint changed during full VAL")


@contextmanager
def patched_evaluator(args, *, adapter_type=None, adapter_module=None):
    """Scoped factory substitution; no persisted edits to audited sources.

    Dependency injection is solely for CPU tests. Real execution imports the
    adapter class BEFORE replacing the parent's module-level factory name.
    """
    candidate, candidate_binding = load_policy(args.candidate_policy)
    checkpoints = checkpoint_bindings(args.config)
    if adapter_type is None:
        from deployment.pre_mask import PreMaskDeploymentAdapter
        adapter_type = PreMaskDeploymentAdapter
    if adapter_module is None:
        import deployment.strong_fast as adapter_module
    original = {name: getattr(fast, name) for name in
                ("build_parser", "deployment_spec", "source_inventory", "final_integrity")}
    old_adapter = adapter_module.FastDeploymentAdapter

    def factory(model, **kwargs):
        return adapter_type(model, candidate_policy=deepcopy(candidate),
                            capture_candidates=False, **kwargs)

    def inventory():
        result = original["source_inventory"]()
        for relative in EXTRA_SOURCES:
            path = REPO / relative
            result[str(path.relative_to(REPO))] = fast.sha(path)
        return result

    def spec(arguments):
        from deployment.candidate_policy import policy_description
        result = original["deployment_spec"](arguments)
        result.update(candidate_policy=deepcopy(candidate), candidate_policy_file=deepcopy(candidate_binding),
                      candidate_policy_semantics=policy_description(**candidate),
                      candidate_selection_stage="after decoded valid boxes, before V/H/A mask ROI inference",
                      candidate_selection_uses_gt=False, candidate_audit_capture=False,
                      complete_original_candidate_set=not any(v is not None for v in candidate.values()),
                      deployment_experiment="pre_mask_candidate_selection_v1",
                      automatic_promotion=False)
        result["evaluation_policy"].update(
            curve_scope=("All model-returned candidates after pre-mask selection and original final-score/top-k/dual-mask-NMS; no extra score truncation"),
            candidate_budget_restricts_recall=any(v is not None for v in candidate.values()),
            original_unfiltered_PR_claimed=False,
        )
        return result

    def integrity(metadata, sources):
        result = original["final_integrity"](metadata, sources)
        assert_bindings_unchanged(candidate_binding, checkpoints)
        result.update(candidate_policy_before_after_identical=True,
                      original_weights_before_after_identical=True,
                      original_weight_bindings=deepcopy(checkpoints))
        return result

    try:
        fast.build_parser = lambda: build_parser(original["build_parser"])
        fast.deployment_spec = spec
        fast.source_inventory = inventory
        fast.final_integrity = integrity
        adapter_module.FastDeploymentAdapter = factory
        yield {"candidate_policy": candidate, "candidate_policy_file": candidate_binding,
               "original_weight_bindings": checkpoints}
    finally:
        adapter_module.FastDeploymentAdapter = old_adapter
        for name, function in original.items():
            setattr(fast, name, function)


def main(argv=None):
    args = build_parser().parse_args(argv)
    validate_args(args)
    with patched_evaluator(args):
        return fast.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
