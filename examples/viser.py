import mujoco
import viser
from mjviser import Viewer
from robot_descriptions import g1_mj_description

from mjsim.utils.mjs import empty_scene

import mujoco as mj


def load_humanoid() -> tuple[mujoco.MjModel, mujoco.MjData]:
    """Load a MuJoCo Menagerie humanoid into a small scene.

    ``robot_descriptions`` exposes cached MuJoCo Menagerie assets. The G1 model is
    a Unitree humanoid from Menagerie:

        robot_descriptions.g1_mj_description.MJCF_PATH

    The raw Menagerie robot XML does not include a ground plane. If it is loaded
    directly, the humanoid simply falls through empty space. This example attaches
    the robot into ``empty_scene()``, which adds the floor, light, skybox, and
    ground material used by the other examples in this repo.
    """
    scene = empty_scene(
        sim_name="g1_viser",
        memory="100M",
        statistic_center=[0, 0, 0.8],
        statistic_extent=2.0,
        floor_size=[5, 5, 0.05],
    )

    humanoid = mujoco.MjSpec.from_file(g1_mj_description.MJCF_PATH)
    scene.attach(humanoid, prefix="g1/", frame=scene.worldbody.add_frame())

    model = scene.compile()
    data = mujoco.MjData(model)

    if model.nkey > 0:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    else:
        data.qpos[:] = model.qpos0

    mujoco.mj_forward(model, data)
    return model, data


def main() -> None:
    model, data = load_humanoid()
    server = viser.ViserServer()

    def reset(model: mujoco.MjModel, data: mujoco.MjData) -> None:
        if model.nkey > 0:
            mujoco.mj_resetDataKeyframe(model, data, 0)
        else:
            mujoco.mj_resetData(model, data)
            data.qpos[:] = model.qpos0
        mujoco.mj_forward(model, data)

    print(f"Loaded MuJoCo Menagerie humanoid: {g1_mj_description.MJCF_PATH}")
    print("Viser server running at http://localhost:8080")

    viewer = Viewer(model, data, reset_fn=reset, server=server)
    viewer.run()


if __name__ == "__main__":
    main()
