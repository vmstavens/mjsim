"""Helpers for converting between spatialmath and mink SE3 objects."""

import numpy as np
import spatialmath as sm
import spatialmath.base as smb
from mink import SE3


def sm_to_smx(T: sm.SE3) -> SE3:
    """Convert a SpatialMath SE3 to a mink SE3.

    Args:
        T: SpatialMath pose.

    Returns:
        mink SE3 pose with wxyz quaternion ordering and xyz translation.
    """
    t = T.t
    q = smb.r2q(T.R)
    qt = np.concatenate([q, t])
    return SE3(wxyz_xyz=qt)


def smx_to_sm(T: SE3) -> sm.SE3:
    """Convert a mink SE3 to a SpatialMath SE3."""
    t = T.translation()
    R = T.rotation().as_matrix()
    return sm.SE3.Rt(R=R, t=t)
