"""
Quick grasping helper demo: build a tiny two-contact scenario and inspect the
grasp matrix and friction cone approximation.

This example is self-contained (NumPy only) and does not require MuJoCo.
"""

import numpy as np

from mjsim.utils import grasping


def main() -> None:
    # Two opposing contacts on a small cube centered at the origin.
    contact_points = np.array(
        [
            [0.05, 0.0, 0.0],  # +X face
            [-0.05, 0.0, 0.0],  # -X face
        ]
    )
    # Contact frames aligned with world; normals point along +Z by convention.
    contact_frames = np.stack([np.eye(3), np.eye(3)])
    p_world_object = np.zeros(3)

    # Friction coefficients and torsional friction radii (b) per contact.
    mu = np.array([0.6, 0.6])
    b = np.array([0.01, 0.01])
    ng = 4  # friction cone linearization slices

    G = grasping.compute_G(contact_points, contact_frames, p_world_object)
    F = grasping.compute_F(mu, contact_frames)
    S = grasping.compute_S(mu, b, ng)

    print("Grasp matrix G shape:", G.shape, "full rank?", grasping.rank_condition(G))
    print("Friction inequality matrix F shape:", F.shape)
    print("Friction cone approximation S shape:", S.shape)


if __name__ == "__main__":
    main()
