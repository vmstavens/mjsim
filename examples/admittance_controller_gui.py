import argparse
import sys
from pathlib import Path

import mujoco as mj

EXAMPLES_DIR = Path(__file__).resolve().parent
removed_paths = [
    path
    for path in sys.path
    if Path(path or ".").resolve() == EXAMPLES_DIR
]
sys.path[:] = [
    path
    for path in sys.path
    if Path(path or ".").resolve() != EXAMPLES_DIR
]
import viser  # noqa: E402
from mjviser import Viewer  # noqa: E402

sys.path[:0] = removed_paths
from admittance_controller import Sim  # noqa: E402


def reset_sim(sim: Sim) -> None:
    mj.mj_resetDataKeyframe(sim.model, sim.data, sim.model.keyframe("ur/home").id)
    mj.mj_forward(sim.model, sim.data)

    sim.force_enabled = False
    sim.admittance_offset[:] = 0.0
    sim.admittance_velocity[:] = 0.0
    sim.admittance_rotvec[:] = 0.0
    sim.admittance_angular_velocity[:] = 0.0
    sim.q_target = sim.ur.q.copy()
    sim.home_target_pos = sim.data.site_xpos[sim.actual_site_id].copy()
    sim.home_target_quat = sim.site_quat(sim.actual_site_id)
    sim.target_quat = sim.home_target_quat.copy()
    sim.ur.set_ctrl(sim.q_target)
    sim.update_target_frame()
    sim.update_pusher_control()


def add_float_control(
    server: viser.ViserServer,
    sim: Sim,
    label: str,
    attr: str,
    minimum: float,
    maximum: float,
    step: float,
):
    handle = server.gui.add_slider(
        label,
        min=minimum,
        max=maximum,
        step=step,
        initial_value=float(getattr(sim, attr)),
    )

    @handle.on_update
    def _(_event) -> None:
        setattr(sim, attr, float(handle.value))

    return handle


def add_bool_control(server: viser.ViserServer, sim: Sim, label: str, attr: str):
    handle = server.gui.add_checkbox(label, initial_value=bool(getattr(sim, attr)))

    @handle.on_update
    def _(_event) -> None:
        setattr(sim, attr, bool(handle.value))
        if attr == "force_enabled":
            sim.update_pusher_control()

    return handle


def add_gui(server: viser.ViserServer, sim: Sim):
    server.gui.set_panel_label("Admittance Controller")
    server.gui.add_markdown(
        "Tune the 6D admittance controller live. "
        "Green is the actual flange frame; magenta is the compliant target."
    )

    with server.gui.add_folder("Pusher", expand_by_default=True):
        pusher_enabled = add_bool_control(
            server,
            sim,
            "Enable physical pusher",
            "force_enabled",
        )
        add_float_control(
            server,
            sim,
            "Pusher penetration offset [m]",
            "pusher_penetration_offset",
            0.035,
            0.075,
            0.001,
        )

    with server.gui.add_folder("Translational Admittance", expand_by_default=True):
        add_float_control(server, sim, "Mass [kg]", "admittance_mass", 0.5, 30.0, 0.5)
        add_float_control(
            server,
            sim,
            "Damping [N s/m]",
            "admittance_damping",
            10.0,
            600.0,
            5.0,
        )
        add_float_control(
            server,
            sim,
            "Stiffness [N/m]",
            "admittance_stiffness",
            100.0,
            10000.0,
            50.0,
        )
        add_float_control(
            server,
            sim,
            "Max target offset [m]",
            "max_target_offset",
            0.005,
            0.25,
            0.005,
        )

    with server.gui.add_folder("Rotational Admittance", expand_by_default=True):
        add_float_control(
            server,
            sim,
            "Inertia [kg m^2]",
            "admittance_rot_inertia",
            0.02,
            3.0,
            0.02,
        )
        add_float_control(
            server,
            sim,
            "Rot damping [N m s/rad]",
            "admittance_rot_damping",
            0.2,
            40.0,
            0.2,
        )
        add_float_control(
            server,
            sim,
            "Rot stiffness [N m/rad]",
            "admittance_rot_stiffness",
            1.0,
            200.0,
            1.0,
        )
        add_float_control(
            server,
            sim,
            "Max target rotvec [rad]",
            "max_target_rotvec",
            0.02,
            1.0,
            0.01,
        )

    with server.gui.add_folder("Wrench Filtering", expand_by_default=False):
        add_float_control(
            server,
            sim,
            "Force deadband [N]",
            "force_deadband",
            0.0,
            50.0,
            0.5,
        )
        add_float_control(
            server,
            sim,
            "Force limit [N]",
            "force_limit",
            5.0,
            1000.0,
            5.0,
        )
        add_float_control(
            server,
            sim,
            "Torque deadband [N m]",
            "torque_deadband",
            0.0,
            2.0,
            0.01,
        )
        add_float_control(
            server,
            sim,
            "Torque limit [N m]",
            "torque_limit",
            0.1,
            50.0,
            0.1,
        )

    with server.gui.add_folder("IK Tracking", expand_by_default=False):
        add_float_control(server, sim, "Position gain", "ik_pos_gain", 0.0, 0.5, 0.005)
        add_float_control(
            server,
            sim,
            "Orientation gain",
            "ik_ori_gain",
            0.0,
            0.5,
            0.005,
        )
        add_float_control(server, sim, "Damping", "ik_damping", 0.001, 0.2, 0.001)
        add_float_control(server, sim, "Force sign", "force_sign", -1.0, 1.0, 2.0)
        add_float_control(server, sim, "Torque sign", "torque_sign", -1.0, 1.0, 2.0)

    readout = server.gui.add_text("State", "", multiline=True, disabled=True)
    reset_button = server.gui.add_button("Reset simulation", color="gray")

    @reset_button.on_click
    def _(_event) -> None:
        reset_sim(sim)
        pusher_enabled.value = sim.force_enabled

    return readout


def format_readout(sim: Sim) -> str:
    force = sim.force_world_frame()
    torque = sim.torque_world_frame()
    return (
        f"force_enabled: {sim.force_enabled}\n"
        f"|F_world|: {(force @ force) ** 0.5: .3f} N\n"
        f"|T_world|: {(torque @ torque) ** 0.5: .3f} N m\n"
        f"target_offset: {sim.admittance_offset}\n"
        f"target_rotvec: {sim.admittance_rotvec}\n"
        f"pusher_z: {sim.pusher_position(): .4f} m"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    sim = Sim()
    server = viser.ViserServer(port=args.port)
    readout = add_gui(server, sim)
    step_count = 0

    def step(_model: mj.MjModel, _data: mj.MjData) -> None:
        nonlocal step_count
        sim.update_pusher_control()
        sim.step_admittance_controller()
        mj.mj_step(sim.model, sim.data)

        step_count += 1
        if step_count % 25 == 0:
            readout.value = format_readout(sim)

    def reset(_model: mj.MjModel, _data: mj.MjData) -> None:
        reset_sim(sim)

    print(f"Viser admittance controller GUI: http://localhost:{args.port}")
    viewer = Viewer(sim.model, sim.data, step_fn=step, reset_fn=reset, server=server)
    viewer.run()


if __name__ == "__main__":
    main()
