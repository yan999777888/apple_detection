"""CPU-only protocol tests; never import the GPU runtime or read weights."""
import importlib.util
from pathlib import Path

import pytest


path = Path(__file__).resolve().parents[1] / "scripts/benchmark_premask.py"
spec = importlib.util.spec_from_file_location("premask_benchmark_under_test", path)
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)


def arguments(*values):
    return bench.validate_args(bench.build_parser().parse_args(["--output", "unused", *values]))


def test_default_protocol():
    args = arguments()
    assert (args.count, args.rounds, args.warmup, args.diagnostic_count) == (8, 2, 3, 3)
    assert args.models == ["base", "small"]
    assert args.policies == ["none", "k64", "nms080_k64", "score003_nms080_k64"]
    assert (args.short_side, args.max_size, args.crop_edge, args.precision) == (1024, 1280, None, "fp16")


def test_center_512_mode_is_a_real_crop():
    args = arguments("--short-side", "512", "--crop-edge", "512")
    assert args.max_size == 512
    assert bench.center_crop_box(1300, 1300, args.crop_edge) == (394, 394, 906, 906)
    assert bench.center_crop_box(601, 511, 511) == (45, 0, 556, 511)
    with pytest.raises(ValueError):
        bench.center_crop_box(511, 1300, 512)


def test_small_smoke_cli():
    args = arguments("--models", "small", "--policies", "k64", "--count", "1",
                     "--rounds", "1", "--warmup", "0", "--diagnostic-count", "0")
    assert args.models == ["small"] and args.policies == ["k64"] and args.count == 1


@pytest.mark.parametrize("values", [
    ("--count", "0"), ("--rounds", "0"), ("--warmup", "-1"), ("--roi-chunk", "0"),
    ("--short-side", "1024", "--max-size", "512"), ("--crop-edge", "0"),
    ("--diagnostic-count", "9"), ("--models", "base", "base"),
    ("--policies", "none", "none"),
])
def test_invalid_protocol_rejected(values):
    with pytest.raises(ValueError):
        arguments(*values)


def test_latency_uses_source_frame_mean_and_interpolated_p95():
    result = bench.latency_summary([400., 100., 300., 200.])
    assert result["mean_ms"] == 250.
    assert result["p95_ms"] == 385.
    assert result["source_frames_per_second"] == 4.
    assert result["n_source_frames"] == 4


@pytest.mark.parametrize("values", [[], [0.], [-1.], [float("nan")], [float("inf")]])
def test_invalid_latency_rejected(values):
    with pytest.raises(ValueError):
        bench.latency_summary(values)


def test_policies_do_not_hide_postmask_filtering():
    assert bench.POLICIES["none"] == {}
    assert bench.POLICIES["k64"] == {"topk": 64}
    assert bench.POLICIES["nms080_k64"] == {"box_nms_iou": .8, "topk": 64}
    assert bench.POLICIES["score003_nms080_k64"] == {"score_threshold": .03, "box_nms_iou": .8, "topk": 64}
