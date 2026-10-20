"""Interactive conveyor-belt simulation.

This is a polished Python adaptation of the conveyor example in
https://github.com/felixokolo/MuJoCo_tutorials.  MuJoCo has no native moving-
surface geom, so the belt is modeled as a fixed contact surface.  A marker on
the driven roller measures the belt's tangential velocity; while the parcel is
touching the belt, a velocity actuator applies that measured velocity to the
parcel.  End-zone touch sensors reverse the conveyor automatically.

Run ``uv run python examples/conveyor_belt.py`` to open the viewer.  Press
Space to reverse the conveyor and Backspace to reset the parcel.  For a smoke
test, run with ``--headless``.
"""

from __future__ import annotations

import argparse
import time

import mujoco as mj
import mujoco.viewer


MODEL_XML = r"""
<mujoco model="conveyor_belt_demo">
  <compiler angle="degree"/>
  <option timestep="0.002" integrator="implicitfast"/>
  <statistic center="0 0 0.22" extent="1.25"/>

  <visual>
    <headlight ambient="0.35 0.35 0.35" diffuse="0.7 0.7 0.7"/>
    <rgba haze="0.12 0.16 0.20 1"/>
    <global azimuth="115" elevation="-24"/>
  </visual>

  <asset>
    <texture name="floor_grid" type="2d" builtin="checker"
             rgb1="0.16 0.19 0.22" rgb2="0.22 0.25 0.28"
             width="256" height="256"/>
    <material name="floor_mat" texture="floor_grid" texrepeat="5 5"
              reflectance="0.08"/>
    <material name="belt_mat" rgba="0.055 0.065 0.075 1"
              specular="0.15" shininess="0.25"/>
    <material name="metal_mat" rgba="0.25 0.29 0.34 1"
              metallic="0.65" roughness="0.35"/>
  </asset>

  <default>
    <geom solref="0.01 1" solimp="0.9 0.95 0.001"/>
    <default class="roller">
      <geom type="cylinder" size="0.075 0.34" material="metal_mat"
            contype="0" conaffinity="0"/>
      <joint type="hinge" axis="0 0 1" damping="0.02"/>
    </default>
    <default class="frame">
      <geom type="box" material="metal_mat" contype="0" conaffinity="0"/>
    </default>
  </default>

  <worldbody>
    <light pos="0 -1 2.5" dir="0 0.35 -1" directional="true"/>
    <geom name="floor" type="plane" size="3 3 0.05" material="floor_mat"/>

    <!-- The black belt is a fixed collision surface. -->
    <body name="conveyor" pos="0 0 0">
      <geom name="belt" type="box" pos="0 0 0.27" size="1 0.30 0.025"
            material="belt_mat" friction="0.7 0.01 0.0001"/>

      <!-- Edge rails and a simple supporting frame. -->
      <geom class="frame" pos="0 0.335 0.25" size="1.08 0.025 0.075"/>
      <geom class="frame" pos="0 -0.335 0.25" size="1.08 0.025 0.075"/>
      <geom class="frame" pos="-0.78 0 0.12" size="0.035 0.30 0.12"/>
      <geom class="frame" pos="0.78 0 0.12" size="0.035 0.30 0.12"/>

      <body name="left_roller" pos="-0.93 0 0.25" euler="90 0 0"
            childclass="roller">
        <joint name="left_roller_joint"/>
        <geom name="left_roller_geom"/>
      </body>

      <body name="right_roller" pos="0.93 0 0.25" euler="90 0 0"
            childclass="roller">
        <joint name="right_roller_joint"/>
        <geom name="right_roller_geom"/>
        <!-- Its red marker makes rotation visible and measures tangent speed. -->
        <geom type="box" size="0.012 0.012 0.34" pos="0.063 0 0"
              rgba="0.95 0.12 0.08 1" contype="0" conaffinity="0"/>
        <site name="roller_speed_site" pos="0 0.075 0" size="0.008"
              rgba="1 0.2 0.1 1"/>
      </body>

      <!-- Touch sensors report contacts whose contact point lies in a site. -->
      <site name="belt_touch_site" type="box" pos="0 0 0.301"
            size="0.96 0.295 0.035" rgba="0 0 0 0"/>
      <site name="left_end_site" type="box" pos="-0.78 0 0.305"
            size="0.035 0.295 0.04" rgba="1 0.72 0.08 0.28"/>
      <site name="right_end_site" type="box" pos="0.78 0 0.305"
            size="0.035 0.295 0.04" rgba="1 0.72 0.08 0.28"/>
    </body>

    <body name="parcel" pos="0 0 0.80">
      <freejoint name="parcel_joint"/>
      <geom name="parcel_geom" type="box" size="0.09 0.075 0.075"
            mass="0.35" rgba="0.90 0.28 0.08 1"
            friction="0.7 0.01 0.0001"/>
      <geom type="box" pos="0 0 0.076" size="0.092 0.012 0.003"
            rgba="1 0.78 0.18 1" contype="0" conaffinity="0"/>
    </body>
  </worldbody>

  <!-- One driven roller; the equality constraint synchronizes the other. -->
  <equality>
    <joint joint1="right_roller_joint" joint2="left_roller_joint"/>
  </equality>

  <actuator>
    <velocity name="roller_velocity" joint="right_roller_joint"
              kv="8" ctrlrange="-30 30"/>
    <!-- A free-joint transmission with this gear acts only along world X. -->
    <velocity name="parcel_velocity" joint="parcel_joint"
              gear="1 0 0 0 0 0" kv="15" ctrlrange="-2 2"/>
  </actuator>

  <sensor>
    <velocimeter name="belt_velocity" site="roller_speed_site"/>
    <touch name="belt_touch" site="belt_touch_site"/>
    <touch name="left_end" site="left_end_site"/>
    <touch name="right_end" site="right_end_site"/>
  </sensor>
</mujoco>
"""


class Sim:
    """Sensor-driven conveyor controller and viewer."""

    def __init__(self, speed: float = 0.45) -> None:
        if speed <= 0:
            raise ValueError("speed must be positive")

        self.speed = speed
        self.direction = 1.0
        self.m = mj.MjModel.from_xml_string(MODEL_XML)
        self.d = mj.MjData(self.m)

        self.roller_actuator_id = self.m.actuator("roller_velocity").id
        self.parcel_actuator_id = self.m.actuator("parcel_velocity").id
        self.parcel_qpos_adr = self.m.jnt_qposadr[self.m.joint("parcel_joint").id]
        self.sensor_adrs = {
            name: self.m.sensor_adr[self.m.sensor(name).id]
            for name in ("belt_velocity", "belt_touch", "left_end", "right_end")
        }
        self.reset()

    def sensor(self, name: str) -> float:
        """Return a scalar sensor, or the X component of a vector sensor."""
        return float(self.d.sensordata[self.sensor_adrs[name]])

    def reset(self) -> None:
        """Reset the parcel and start the conveyor toward positive X."""
        mj.mj_resetData(self.m, self.d)
        self.direction = 1.0
        self._set_roller_control()
        mj.mj_forward(self.m, self.d)

    def reverse(self) -> None:
        self.direction *= -1.0

    def _set_roller_control(self) -> None:
        roller_radius = 0.075
        # With the roller's orientation, negative angular velocity moves +X.
        self.d.ctrl[self.roller_actuator_id] = (
            -self.direction * self.speed / roller_radius
        )

    def _control(self) -> None:
        # End zones select a direction rather than toggling it every time step.
        if self.sensor("right_end") > 0.0:
            self.direction = -1.0
        elif self.sensor("left_end") > 0.0:
            self.direction = 1.0
        self._set_roller_control()

        # This is the tutorial's key idea: copy measured roller tangent speed
        # to the parcel only while contact with the belt is detected.
        parcel_speed = 0.0
        if self.sensor("belt_touch") > 0.0:
            parcel_speed = self.sensor("belt_velocity")
        self.d.ctrl[self.parcel_actuator_id] = parcel_speed

    def step(self) -> None:
        self._control()
        mj.mj_step(self.m, self.d)

        # Recover gracefully if the parcel is dragged out of the scene.
        parcel_pos = self.d.qpos[self.parcel_qpos_adr : self.parcel_qpos_adr + 3]
        if parcel_pos[2] < -0.2 or abs(parcel_pos[1]) > 1.0:
            self.reset()

    def key_callback(self, key: int) -> None:
        if key == 32:  # Space
            self.reverse()
        elif key == 259:  # Backspace (GLFW_KEY_BACKSPACE)
            self.reset()

    def run_headless(self, steps: int) -> None:
        direction_changes = 0
        old_direction = self.direction
        for _ in range(steps):
            self.step()
            if self.direction != old_direction:
                direction_changes += 1
                old_direction = self.direction

        parcel_pos = self.d.qpos[self.parcel_qpos_adr : self.parcel_qpos_adr + 3]
        print(
            "conveyor belt: "
            f"time={self.d.time:.2f}s, x={parcel_pos[0]:.3f}m, "
            f"belt_vx={self.sensor('belt_velocity'):.3f}m/s, "
            f"touch={self.sensor('belt_touch'):.3f}N, "
            f"reversals={direction_changes}"
        )

    def run_viewer(self) -> None:
        print("Space: reverse | Backspace: reset | Close viewer: exit")
        with mujoco.viewer.launch_passive(
            self.m, self.d, key_callback=self.key_callback
        ) as viewer:
            viewer.cam.azimuth = 115
            viewer.cam.elevation = -24
            viewer.cam.lookat = [0.0, 0.0, 0.25]
            viewer.cam.distance = 2.4

            while viewer.is_running():
                step_start = time.perf_counter()
                self.step()
                viewer.sync()

                remaining = self.m.opt.timestep - (time.perf_counter() - step_start)
                if remaining > 0.0:
                    time.sleep(remaining)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--speed", type=float, default=0.45, help="belt speed in m/s")
    args = parser.parse_args()

    sim = Sim(speed=args.speed)
    if args.headless:
        sim.run_headless(args.steps)
    else:
        sim.run_viewer()


if __name__ == "__main__":
    main()
