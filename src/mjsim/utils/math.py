import numpy as np


def quat_normalize(quat: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Normalize a NumPy quaternion in ``[w, x, y, z]`` order."""
    quat = np.asarray(quat, dtype=np.float64)
    norm = np.linalg.norm(quat)
    if norm <= eps:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return quat / norm


def rotvec_to_quat(rotvec: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Convert a NumPy rotation vector to a quaternion in ``[w, x, y, z]`` order."""
    rotvec = np.asarray(rotvec, dtype=np.float64)
    angle = np.linalg.norm(rotvec)
    if angle <= eps:
        return quat_normalize(np.concatenate([[1.0], 0.5 * rotvec]))

    axis = rotvec / angle
    half = 0.5 * angle
    return quat_normalize(np.concatenate([[np.cos(half)], axis * np.sin(half)]))
