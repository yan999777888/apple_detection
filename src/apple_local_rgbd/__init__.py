"""CPU-only experimental geometry; independent of the protected vision models."""

from .geometry import SphereConfig, fit_sphere, points_from_depth, rays_from_pixels

__all__ = ["SphereConfig", "fit_sphere", "points_from_depth", "rays_from_pixels"]
