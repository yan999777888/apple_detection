"""Independent geometric checks, including failures that should not look confident."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from apple_local_rgbd.boundaries import candidate_outer_edges
from apple_local_rgbd.geometry import SphereConfig, fit_sphere, points_from_depth, tangent_residual
from apple_local_rgbd.synthetic import render_sphere, tangent_rays


def test_off_axis_true_sphere_rays_are_tangent():
    center, radius = np.array([.14, -.06, .36]), .045
    rays = tangent_rays(center, radius)
    assert np.max(np.abs(tangent_residual(center, radius, rays))) < 1e-12
    assert (rays @ center > 0).all()
    # A wrong radius cannot satisfy exactly the same perspective silhouette.
    assert np.min(np.abs(tangent_residual(center, radius+.01, rays))) > .0099


@pytest.mark.parametrize("method", ["ransac_refined", "depth_huber", "joint_huber"])
def test_noiseless_off_axis_center_is_recovered_without_gt_prior(method):
    sample = render_sphere(center=(.06, -.015, .32), depth_noise_m=0, outlier_fraction=0, occlusion="half")
    points = points_from_depth(sample["depth_m"], sample["visible"], sample["intrinsics"])
    result = fit_sphere(points, method=method, contour_rays=tangent_rays(sample["center_gt_m"], sample["radius_gt_m"]))
    assert result["accepted"]
    assert np.linalg.norm(np.array(result["center_m"])-sample["center_gt_m"]) < 1e-6
    assert abs(result["radius_m"]-sample["radius_gt_m"]) < 1e-6
    assert not result["used_size_prior"]


def test_occluder_depth_never_enters_visible_point_cloud():
    sample = render_sphere(occlusion="half", depth_noise_m=0, outlier_fraction=0)
    assert (sample["depth_m"][sample["amodal"] & ~sample["visible"]] == .18).all()
    points = points_from_depth(sample["depth_m"], sample["visible"], sample["intrinsics"])
    assert points[:, 2].min() > .25
    assert len(points) == sample["visible"].sum()


def test_tiny_visible_cap_is_not_accepted_even_when_perfectly_fit():
    sample = render_sphere(occlusion="tiny_cap", depth_noise_m=0, outlier_fraction=0)
    result = fit_sphere(points_from_depth(sample["depth_m"], sample["visible"], sample["intrinsics"]), method="joint_huber")
    assert not result["accepted"]
    assert "small_visible_cap_center_radius_ambiguity" in result["flags"]


def test_partial_contour_excludes_synthetic_leaf_cut():
    sample = render_sphere(occlusion="half", depth_noise_m=0, outlier_fraction=0)
    edges = candidate_outer_edges(sample["visible"], sample["amodal"], amodal_quality=.25)
    assert len(edges["xy"]) > 0
    assert edges["occlusion_boundary_mask"].sum() > 20
    assert not (edges["candidate_mask"] & edges["occlusion_boundary_mask"]).any()
    assert "not_true_edge" in edges["provenance"]


def test_no_amodal_or_no_quality_disables_contour():
    mask = np.ones((30, 30), bool)
    assert len(candidate_outer_edges(mask)["xy"]) == 0
    assert len(candidate_outer_edges(mask, mask)["xy"]) == 0


def test_test_truth_cannot_be_used_as_prior():
    sample = render_sphere()
    points = points_from_depth(sample["depth_m"], sample["visible"], sample["intrinsics"])
    with pytest.raises(ValueError, match="independent"):
        fit_sphere(points, method="joint_huber", radius_prior=dict(mean_m=.04, std_m=.001,
                                                                  provenance_scope="test", source="diameter GT"))


def test_empty_depth_has_explicit_failure():
    result = fit_sphere(np.empty((0, 3)))
    assert result["center_m"] is None and not result["accepted"]
    assert result["flags"] == ["insufficient_valid_depth"]


def test_fixed_seed_is_reproducible():
    sample = render_sphere(occlusion="heavy")
    points = points_from_depth(sample["depth_m"], sample["visible"], sample["intrinsics"])
    a, b = fit_sphere(points), fit_sphere(points)
    assert a["center_m"] == b["center_m"] and a["flags"] == b["flags"]


def test_no_fake_magsac_sphere_backend():
    with pytest.raises(ValueError, match="MAGSAC"):
        fit_sphere(np.zeros((50, 3)), method="magsac")


def test_focal_only_real_data_is_rejected_before_files_are_opened():
    path = Path(__file__).resolve().parents[1] / "scripts/benchmark_rgbd_geometry.py"
    spec = importlib.util.spec_from_file_location("geometry_cli", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    record = dict(image_id="focal_only", depth_path="unopened.npy", visible_mask_path="unopened.png",
                  intrinsics={"fx":5805.34,"fy":5805.34}, intrinsics_provenance="author focal only",
                  alignment_provenance="author paired", depth_scale_m=1, depth_value_kind="z",
                  distortion_model="rectified_pinhole")
    with pytest.raises(ValueError, match="principal point"):
        module.read_real_record(record)
