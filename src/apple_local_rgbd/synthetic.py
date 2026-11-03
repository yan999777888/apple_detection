"""Independent ray/sphere rendering for geometry checks; not orchard evidence."""
from __future__ import annotations

import numpy as np


def tangent_rays(center, radius, count=160):
    center = np.asarray(center, dtype=float)
    distance = np.linalg.norm(center)
    if distance <= radius:
        raise ValueError("Camera must be outside synthetic sphere")
    axis = center/distance
    helper = np.array([1., 0., 0.]) if abs(axis[0]) < .8 else np.array([0., 1., 0.])
    u = np.cross(axis, helper)
    u /= np.linalg.norm(u)
    v = np.cross(axis, u)
    angle = np.arange(count)*2*np.pi/count
    alpha = np.arcsin(radius/distance)
    return np.cos(alpha)*axis + np.sin(alpha)*(np.cos(angle)[:, None]*u + np.sin(angle)[:, None]*v)


def render_sphere(center=(.06, -.015, .32), radius=.04, *, seed=17, height=192, width=256,
                  depth_noise_m=.001, outlier_fraction=.05, occlusion="none"):
    center = np.asarray(center, dtype=np.float64)
    intrinsics = dict(fx=300., fy=315., cx=(width-1)/2, cy=(height-1)/2)
    yy, xx = np.mgrid[:height, :width]
    direction = np.stack(((xx-intrinsics["cx"])/intrinsics["fx"],
                          (yy-intrinsics["cy"])/intrinsics["fy"], np.ones_like(xx)), axis=-1)
    aa = (direction**2).sum(-1)
    bb = direction @ center
    discriminant = bb**2-aa*(center@center-radius**2)
    a = (discriminant >= 0) & (bb > 0)
    depth = np.zeros((height, width), dtype=np.float64)
    depth[a] = (bb[a]-np.sqrt(discriminant[a]))/aa[a]
    v = a.copy()
    projected = np.array([intrinsics["fx"]*center[0]/center[2]+intrinsics["cx"],
                          intrinsics["fy"]*center[1]/center[2]+intrinsics["cy"]])
    if occlusion == "half":
        v &= xx < projected[0]
    elif occlusion == "heavy":
        v &= xx < projected[0]-12
    elif occlusion == "tiny_cap":
        v &= (xx-projected[0])**2 + (yy-projected[1])**2 <= 5**2
    elif occlusion != "none":
        raise ValueError("Unknown synthetic occlusion")
    rng = np.random.default_rng(seed)
    depth[v] += rng.normal(0, depth_noise_m, int(v.sum()))
    indices = np.flatnonzero(v)
    count = int(len(indices)*outlier_fraction)
    if count:
        corrupted = rng.choice(indices, count, replace=False)
        depth.flat[corrupted] += rng.uniform(-.025, .025, count)
    # Hidden depth deliberately represents the foreground occluder, not fruit.
    depth[a & ~v] = .18
    return dict(depth_m=depth, visible=v, amodal=a, intrinsics=intrinsics,
                center_gt_m=center, radius_gt_m=radius,
                ground_truth_origin="analytic ray/sphere intersection, synthetic only",
                occlusion=occlusion)
