# Copyright (c) 2026 vecnode. All rights reserved.
"""3D / depth warping (Marigold, pytorch3d)."""

from .depth_backends import DepthBackend, MarigoldDepthBackend
from .warp import (
    get_spherical_projection,
    transform_image_3d,
)

__all__ = [
    "DepthBackend",
    "MarigoldDepthBackend",
    "get_spherical_projection",
    "transform_image_3d",
]
