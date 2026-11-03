"""V/A agreement proposes outer edges; it does not establish edge ground truth."""
from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion, distance_transform_edt, minimum_filter


def candidate_outer_edges(visible, amodal=None, *, amodal_quality=0.0, relative_tolerance=.025,
                          maximum_points=160, depth_m=None, occluder_gap_m=.01):
    """Return weak contour candidates in native pixel coordinates.

    The caller explicitly supplies an uncalibrated A-quality weight. The default
    disables the constraint. A/V predictions can agree and still both be wrong.
    No fixed two-pixel rule, ground-truth-edge label, or 1-ms speed claim is made.
    """
    visible = np.asarray(visible)
    if visible.ndim != 2 or not np.isfinite(visible).all():
        raise ValueError("V must be a finite H,W probability/binary mask")
    v = visible >= .5
    empty = dict(xy=np.empty((0, 2)), weights=np.empty(0),
                 candidate_mask=np.zeros(v.shape, bool), occlusion_boundary_mask=np.zeros(v.shape, bool),
                 provenance="predicted_V_A_agreement_candidate_not_true_edge", flags=[])
    if amodal is None or amodal_quality <= 0:
        return {**empty, "flags": ["amodal_missing_or_constraint_disabled"]}
    if not 0 < amodal_quality <= 1 or relative_tolerance <= 0 or maximum_points < 1:
        raise ValueError("Invalid candidate quality/tolerance/budget")
    amodal = np.asarray(amodal)
    if amodal.shape != v.shape or not np.isfinite(amodal).all():
        raise ValueError("V/A must share finite native coordinates")
    a = amodal >= .5
    if not a.any() or not v.any():
        return {**empty, "flags": ["empty_V_or_A"]}
    flags = []
    outside_fraction = float((v & ~a).sum() / v.sum())
    if outside_fraction > .05:
        return {**empty, "flags": ["V_outside_A_inconsistency"], "V_outside_A_fraction": outside_fraction}
    v_edge = v & ~binary_erosion(v)
    a_edge = a & ~binary_erosion(a)
    tolerance = max(1., relative_tolerance * np.sqrt(float(a.sum()) / np.pi))
    distance = distance_transform_edt(~a_edge)
    hidden = a & ~v
    occluded = v_edge & binary_dilation(hidden) & (distance > tolerance)
    candidates = v_edge & (distance <= tolerance)
    candidates[[0, -1], :] = False
    candidates[:, [0, -1]] = False
    if float(hidden.sum()) / max(1, int(a.sum())) < .01:
        flags.append("A_nearly_equals_V_does_not_verify_absence_of_occlusion")
        amodal_quality = min(amodal_quality, .5)
    if depth_m is not None:
        depth = np.asarray(depth_m, dtype=np.float64)
        if depth.shape != v.shape:
            raise ValueError("Depth must share native V/A shape")
        valid = np.isfinite(depth) & (depth > 0)
        near_outside = minimum_filter(np.where(valid & ~v, depth, np.inf), size=5, mode="constant", cval=np.inf)
        foreground_occluder = valid & np.isfinite(near_outside) & (near_outside < depth-occluder_gap_m)
        candidates &= valid & ~foreground_occluder
        occluded |= v_edge & foreground_occluder
    yy, xx = np.nonzero(candidates)
    if len(xx) > maximum_points:
        # Deterministic angular spread instead of keeping only one raster edge.
        angle = np.arctan2(yy-yy.mean(), xx-xx.mean())
        order = np.argsort(angle, kind="stable")
        chosen = order[np.linspace(0, len(order)-1, maximum_points).astype(int)]
        yy, xx = yy[chosen], xx[chosen]
    weights = amodal_quality * np.exp(-.5*(distance[yy, xx]/tolerance)**2)
    return {**empty, "xy": np.column_stack((xx, yy)).astype(float), "weights": weights,
            "candidate_mask": candidates, "occlusion_boundary_mask": occluded,
            "flags": flags, "tolerance_pixels": tolerance, "V_outside_A_fraction": outside_fraction,
            "amodal_quality_weight": amodal_quality}
