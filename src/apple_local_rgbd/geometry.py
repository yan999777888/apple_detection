"""Auditable sphere baselines in metres, with explicit degeneracy diagnostics.

RANSAC is implemented here for spheres. No OpenCV homography/fundamental-matrix
MAGSAC call is substituted for a sphere solver. The bounded robust solver uses
SciPy's trust-region reflective method, not Levenberg-Marquardt.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import time

import numpy as np
from scipy.optimize import least_squares


@dataclass(frozen=True)
class SphereConfig:
    seed: int = 4729
    max_points: int = 1500
    min_points: int = 30
    ransac_trials: int = 192
    inlier_threshold_m: float = 0.004
    depth_sigma_m: float = 0.002
    radius_min_m: float = 0.005
    radius_max_m: float = 0.150
    max_nfev: int = 100
    contour_sigma_rad: float = 0.003
    contour_weight: float = 1.0
    minimum_normal_dispersion: float = 0.005
    maximum_depth_jacobian_condition: float = 1.0e6

    def validate(self):
        if self.min_points < 4 or self.max_points < self.min_points or self.ransac_trials < 1:
            raise ValueError("Invalid bounded sample budget")
        if not 0 < self.radius_min_m < self.radius_max_m:
            raise ValueError("Radius bounds must be positive metres")
        if min(self.inlier_threshold_m, self.depth_sigma_m, self.contour_sigma_rad) <= 0:
            raise ValueError("Noise scales must be positive")
        if self.max_nfev < 1 or self.contour_weight < 0:
            raise ValueError("Invalid optimization budget/weight")


def _intrinsics(intrinsics):
    values = np.array([intrinsics[k] for k in ("fx", "fy", "cx", "cy")], dtype=np.float64)
    if not np.isfinite(values).all() or min(values[:2]) <= 0:
        raise ValueError("Finite rectified pinhole intrinsics are required")
    return values


def rays_from_pixels(xy, intrinsics):
    """Unit rays for rectified pixel coordinates (array indices are centres)."""
    fx, fy, cx, cy = _intrinsics(intrinsics)
    xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
    rays = np.column_stack(((xy[:, 0] - cx) / fx, (xy[:, 1] - cy) / fy, np.ones(len(xy))))
    return rays / np.linalg.norm(rays, axis=1, keepdims=True)


def points_from_depth(depth_m, visible_mask, intrinsics):
    """Back-project only valid V pixels. Depth means optical Z, not ray range."""
    depth = np.asarray(depth_m, dtype=np.float64)
    visible = np.asarray(visible_mask, dtype=bool)
    if depth.ndim != 2 or depth.shape != visible.shape:
        raise ValueError("Aligned native depth and V masks must have identical H,W")
    fx, fy, cx, cy = _intrinsics(intrinsics)
    valid = visible & np.isfinite(depth) & (depth > 0)
    yy, xx = np.nonzero(valid)
    z = depth[yy, xx]
    return np.column_stack(((xx-cx)*z/fx, (yy-cy)*z/fy, z))


def tangent_residual(center, radius, unit_rays):
    """Exact perspective-sphere tangent-ray residual, in metres.

    A silhouette ray has distance r to the 3-D centre: ||c cross n|| = r.
    This does not assume a circular pixel silhouette or its centre equals Kc.
    The caller additionally checks forward-facing geometry.
    """
    return np.linalg.norm(np.cross(np.asarray(center), unit_rays), axis=1) - radius


def _radial(parameters, points):
    return np.linalg.norm(points - parameters[:3], axis=1) - parameters[3]


def _four_point_sphere(points):
    a = 2 * (points[1:] - points[0])
    if np.linalg.cond(a) > 1e9:
        return None
    try:
        center = np.linalg.solve(a, (points[1:]**2).sum(1) - (points[0]**2).sum())
    except np.linalg.LinAlgError:
        return None
    radius = np.linalg.norm(points[0] - center)
    result = np.r_[center, radius]
    return result if np.isfinite(result).all() else None


def _initialize(points, cfg):
    rng = np.random.default_rng(cfg.seed)
    best, best_key = None, (-1, -np.inf)
    for _ in range(cfg.ransac_trials):
        proposal = _four_point_sphere(points[rng.choice(len(points), 4, replace=False)])
        if (proposal is None or not cfg.radius_min_m <= proposal[3] <= cfg.radius_max_m
                or proposal[2] <= proposal[3]):
            continue
        errors = np.abs(_radial(proposal, points))
        inside = errors <= cfg.inlier_threshold_m
        count = int(inside.sum())
        key = (count, -float(np.median(errors[inside])) if count else -np.inf)
        if key > best_key:
            best, best_key = proposal, key
    return best


def _diagnostics(parameters, points, cfg):
    errors = _radial(parameters, points)
    inside = np.abs(errors) <= cfg.inlier_threshold_m
    support = points[inside]
    flags = []
    if len(support) < cfg.min_points:
        flags.append("insufficient_inlier_support")
    normals = support - parameters[:3]
    lengths = np.linalg.norm(normals, axis=1)
    normals = normals[lengths > 1e-12] / lengths[lengths > 1e-12, None]
    dispersion = float(1 - np.linalg.norm(normals.mean(0))) if len(normals) else 0.0
    jac = np.column_stack((-normals, -np.ones(len(normals))))
    singular = np.linalg.svd(jac, compute_uv=False) if len(normals) >= 4 else np.zeros(4)
    condition = float(singular[0] / singular[-1]) if singular[-1] > 1e-12 else float("inf")
    if dispersion < cfg.minimum_normal_dispersion:
        flags.append("small_visible_cap_center_radius_ambiguity")
    if condition > cfg.maximum_depth_jacobian_condition:
        flags.append("ill_conditioned_depth_geometry")
    if parameters[2] <= parameters[3]:
        flags.append("sphere_not_wholly_in_front_of_camera")
    # Depth cameras observe the camera-facing hemisphere of an opaque sphere.
    facing = np.einsum("ij,ij->i", support, support - parameters[:3]) <= 0
    facing_fraction = float(facing.mean()) if len(facing) else 0.0
    if facing_fraction < .9:
        flags.append("front_surface_inconsistency")
    if min(parameters[3]-cfg.radius_min_m, cfg.radius_max_m-parameters[3]) < 1e-5:
        flags.append("radius_at_bound")
    return dict(inliers=int(inside.sum()), inlier_fraction=float(inside.mean()),
                radial_median_abs_m=float(np.median(np.abs(errors))),
                radial_inlier_rmse_m=float(np.sqrt(np.mean(errors[inside]**2))) if inside.any() else None,
                normal_dispersion=dispersion,
                depth_jacobian_condition=condition if np.isfinite(condition) else None,
                camera_facing_fraction=facing_fraction, flags=flags)


def fit_sphere(points, *, method="ransac_refined", contour_rays=None, contour_weights=None,
               radius_prior=None, config=None):
    """Run one fixed-budget baseline; truths never enter this API.

    radius_prior, if supplied, is {mean_m, std_m, provenance_scope, source}; only
    independent 'train' or 'external' provenance is accepted. A test instance's
    measured diameter must never be used as this prior.
    """
    started = time.perf_counter()
    cfg = config or SphereConfig()
    cfg.validate()
    if method not in ("ransac_refined", "depth_huber", "joint_huber"):
        raise ValueError("Supported: ransac_refined, depth_huber, joint_huber; sphere MAGSAC++ is not implemented")
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("Expected points[N,3] in metres")
    points = points[np.isfinite(points).all(1) & (points[:, 2] > 0)]
    input_count = len(points)
    if len(points) > cfg.max_points:
        chosen = np.sort(np.random.default_rng(cfg.seed).choice(len(points), cfg.max_points, replace=False))
        points = points[chosen]
    base = dict(method=method, input_points=input_count, fitted_points=len(points), config=asdict(cfg),
                center_m=None, radius_m=None, accepted=False, status="failed", flags=[],
                solver="scipy_least_squares_TRF", center_covariance_claimed=False,
                training_performed=False, used_contour_points=0, used_size_prior=False)

    def finish(**updates):
        return {**base, **updates, "elapsed_ms": (time.perf_counter()-started)*1000}

    if len(points) < cfg.min_points:
        return finish(flags=["insufficient_valid_depth"])
    if radius_prior is not None:
        if (radius_prior.get("provenance_scope") not in ("train", "external")
                or not radius_prior.get("source") or radius_prior.get("std_m", 0) <= 0
                or radius_prior.get("mean_m", 0) <= 0
                or not np.isfinite([radius_prior.get("std_m", 0), radius_prior.get("mean_m", 0)]).all()):
            raise ValueError("Size prior needs independent train/external provenance and positive mean/std")
    initial = _initialize(points, cfg)
    if initial is None:
        return finish(flags=["no_valid_sphere_initialization"])
    original_inliers = np.abs(_radial(initial, points)) <= cfg.inlier_threshold_m
    if int(original_inliers.sum()) < cfg.min_points:
        return finish(flags=["insufficient_initial_inliers"])
    lower = [-np.inf, -np.inf, 1e-6, cfg.radius_min_m]
    upper = [np.inf, np.inf, np.inf, cfg.radius_max_m]
    # Identical RANSAC and inlier polish initializes all ablations.
    polish = least_squares(lambda p: _radial(p, points[original_inliers]), initial,
                           bounds=(lower, upper), method="trf", max_nfev=cfg.max_nfev)
    initial = polish.x
    rays = np.empty((0, 3)) if contour_rays is None else np.asarray(contour_rays, dtype=np.float64).reshape(-1, 3)
    weights = np.ones(len(rays)) if contour_weights is None else np.asarray(contour_weights, dtype=np.float64)
    if weights.shape != (len(rays),) or not np.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("Nonnegative finite per-ray weights required")
    if len(rays) and (not np.isfinite(rays).all() or not np.allclose(np.linalg.norm(rays, axis=1), 1, atol=1e-6)):
        raise ValueError("Contour rays must be finite unit directions")
    if method != "joint_huber":
        rays, weights, radius_prior = np.empty((0, 3)), np.empty(0), None

    def residual(p):
        terms = [_radial(p, points) / cfg.depth_sigma_m]
        if len(rays):
            # The angular pixel uncertainty becomes a metric tangent uncertainty.
            scale_m = max(np.linalg.norm(p[:3]), .01) * cfg.contour_sigma_rad
            terms.append(np.sqrt(weights*cfg.contour_weight) * tangent_residual(p[:3], p[3], rays) / scale_m)
        if radius_prior is not None:
            terms.append(np.array([(p[3]-radius_prior["mean_m"]) / radius_prior["std_m"]]))
        return np.concatenate(terms)

    if method == "ransac_refined":
        parameters, converged, nfev = initial, bool(polish.success), int(polish.nfev)
    else:
        result = least_squares(residual, initial, bounds=(lower, upper), loss="huber", f_scale=1,
                               method="trf", max_nfev=cfg.max_nfev)
        parameters, converged, nfev = result.x, bool(result.success), int(result.nfev)
    diagnostics = _diagnostics(parameters, points, cfg)
    flags = diagnostics.pop("flags")
    if not converged:
        flags.append("optimization_budget_exhausted")
    if len(rays) and (rays @ parameters[:3] <= 0).any():
        flags.append("contour_ray_behind_camera")
    return finish(status="computed", accepted=not flags, flags=flags,
                  center_m=parameters[:3].tolist(), radius_m=float(parameters[3]),
                  initial_parameters=initial.tolist(), diagnostics=diagnostics, nfev=nfev,
                  used_contour_points=len(rays), used_size_prior=radius_prior is not None,
                  size_prior_provenance=radius_prior,
                  reliability_note="Diagnostics are gates, not calibrated confidence intervals; rejection does not prove no fruit exists")
