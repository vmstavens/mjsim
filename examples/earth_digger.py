import mujoco as mj

import mjsim as ms


class Sim(ms.BaseSim):
    def __init__(self):
        self._model, self._data = self._init()

    def _init(self) -> tuple[mj.MjModel, mj.MjData]:

        _XML = """
            <mujoco model="earth-digger">
                <compiler angle="radian" autolimits="true"/>
                <option timestep=".002" integrator="implicitfast" cone="elliptic" gravity="0 0 -9.81"/>

                <!-- Slider-crank transmissions are rendered as hydraulic cylinders. -->
                <visual>
                    <scale slidercrank=".08"/>
                    <rgba slidercrank="0.16 0.31 0.72 1" crankbroken="0.7 0.1 0.1 1"/>
                    <global azimuth="135" elevation="-25"/>
                </visual>

                <default>
                    <geom density="350" friction="1.1 .03 .003" solref=".012 1" solimp=".9 .95 .001" contype="0" conaffinity="0"/>
                    <joint damping="3" armature=".03" frictionloss=".15"/>
                    <position ctrllimited="true" kp="8000" kv="500" forcelimited="true" forcerange="-16000 16000"/>
                    <default class="pin">
                        <geom type="cylinder" size=".014" fromto="0 -.07 0 0 .07 0" mass=".06" rgba=".07 .07 .08 1"/>
                    </default>
                    <default class="link">
                        <geom type="capsule" size=".026" rgba=".93 .62 .12 1"/>
                    </default>
                    <default class="hydraulic_base">
                        <site type="cylinder" size=".012 .045" rgba=".88 .91 .96 1"/>
                    </default>
                    <default class="hydraulic_rod">
                        <site size=".016" rgba=".06 .14 .30 1"/>
                    </default>
                </default>

                <worldbody>
                    <light pos="0 -1.6 2.2" dir="0 1 -1"/>
                    <camera name="overview" pos=".15 -2.2 .8" xyaxes="1 0 0 0 .35 .94"/>

                    <geom name="ground" type="plane" size="1.8 1.2 .02" contype="1" conaffinity="1" rgba=".34 .27 .18 1"/>
                    <geom name="trench" type="box" pos=".55 0 -.02" size=".28 .28 .018" contype="1" conaffinity="1" rgba=".17 .12 .08 1"/>
                    <geom name="spoil_pile_1" type="ellipsoid" pos=".18 -.35 .045" size=".22 .11 .055" contype="1" conaffinity="1" rgba=".46 .34 .20 1"/>
                    <geom name="spoil_pile_2" type="ellipsoid" pos=".55 .31 .035" size=".17 .09 .045" contype="1" conaffinity="1" rgba=".41 .29 .17 1"/>

                    <body name="undercarriage" pos="-.42 0 .075">
                        <geom name="left_track" type="box" pos="0 -.13 0" size=".31 .045 .045" mass="2.2" rgba=".07 .07 .07 1"/>
                        <geom name="right_track" type="box" pos="0 .13 0" size=".31 .045 .045" mass="2.2" rgba=".07 .07 .07 1"/>
                        <geom name="track_frame" type="box" pos="0 0 .035" size=".24 .16 .035" mass="1.4" rgba=".18 .19 .20 1"/>

                        <body name="house" pos=".02 0 .095">
                            <joint name="slew" type="hinge" axis="0 0 1" range="-.9 .9" damping="8" armature=".08" frictionloss=".8"/>
                            <geom name="turntable" type="cylinder" pos="0 0 0" size=".15 .035" mass="2.0" rgba=".13 .14 .15 1"/>
                            <geom name="cab" type="box" pos="-.055 .045 .105" size=".085 .085 .085" mass=".7" rgba=".95 .67 .14 1"/>
                            <geom name="cab_window" type="box" pos="-.09 .048 .13" size=".007 .087 .038" rgba=".15 .28 .36 .75"/>
                            <geom name="counterweight" type="box" pos="-.16 0 .07" size=".075 .13 .065" mass="3.5" rgba=".13 .13 .14 1"/>
                            <geom name="engine_cover" type="box" pos=".02 -.02 .075" size=".12 .11 .055" mass="1.4" rgba=".88 .55 .10 1"/>

                            <site name="boom_cylinder_base" class="hydraulic_base" pos="-.07 0 .03" zaxis="1 0 .35"/>

                            <body name="boom" pos=".10 0 .11">
                                <joint name="boom_hinge" type="hinge" axis="0 1 0" range="-.95 .65" damping="5" armature=".05" frictionloss=".25"/>
                                <geom class="pin"/>
                                <geom name="boom_left" class="link" fromto="0 -.035 0 .52 -.035 .22" mass="1.3"/>
                                <geom name="boom_right" class="link" fromto="0 .035 0 .52 .035 .22" mass="1.3"/>
                                <site name="boom_cylinder_rod" class="hydraulic_rod" pos=".24 0 .005"/>
                                <site name="stick_cylinder_base" class="hydraulic_base" pos=".19 0 .17" zaxis=".8 0 .45"/>

                                <body name="stick" pos=".52 0 .22">
                                    <joint name="stick_hinge" type="hinge" axis="0 1 0" range="-1.55 .45" damping="4" armature=".04" frictionloss=".2"/>
                                    <geom class="pin"/>
                                    <geom name="stick_link" class="link" fromto="0 0 0 .42 0 -.30" mass="1.5" rgba=".88 .54 .08 1"/>
                                    <site name="stick_cylinder_rod" class="hydraulic_rod" pos=".18 0 -.075"/>
                                    <site name="bucket_cylinder_base" class="hydraulic_base" pos=".22 0 -.08" zaxis=".5 0 1"/>

                                    <body name="bucket" pos=".42 0 -.30">
                                        <joint name="bucket_hinge" type="hinge" axis="0 1 0" range="-1.45 .65" damping="2.5" armature=".02" frictionloss=".12"/>
                                        <geom class="pin"/>
                                        <site name="bucket_cylinder_rod" class="hydraulic_rod" pos="-.055 0 .06"/>
                                        <geom name="bucket_back" type="box" pos=".025 0 -.018" size=".065 .105 .018" mass=".22" contype="1" conaffinity="1" rgba=".17 .17 .18 1"/>
                                        <geom name="bucket_lip" type="box" pos=".105 0 -.065" size=".045 .11 .014" mass=".18" contype="1" conaffinity="1" rgba=".11 .11 .12 1"/>
                                        <geom name="bucket_left_side" type="box" pos=".047 -.108 -.035" size=".065 .010 .048" mass=".08" contype="1" conaffinity="1" rgba=".14 .14 .15 1"/>
                                        <geom name="bucket_right_side" type="box" pos=".047 .108 -.035" size=".065 .010 .048" mass=".08" contype="1" conaffinity="1" rgba=".14 .14 .15 1"/>
                                        <geom name="tooth_1" type="box" pos=".155 -.065 -.082" size=".023 .013 .011" mass=".025" contype="1" conaffinity="1" rgba=".06 .06 .07 1"/>
                                        <geom name="tooth_2" type="box" pos=".158 0 -.083" size=".025 .013 .011" mass=".025" contype="1" conaffinity="1" rgba=".06 .06 .07 1"/>
                                        <geom name="tooth_3" type="box" pos=".155 .065 -.082" size=".023 .013 .011" mass=".025" contype="1" conaffinity="1" rgba=".06 .06 .07 1"/>
                                    </body>
                                </body>
                            </body>
                        </body>
                    </body>
                </worldbody>

                <actuator>
                    <position name="slew_position" joint="slew" ctrlrange="-.75 .75" forcerange="-350 350" kp="160" kv="35"/>
                    <!-- Signed slider-crank lengths: these ranges are tuned from the compiled linkage. -->
                    <position name="boom_lift" cranksite="boom_cylinder_base" slidersite="boom_cylinder_rod" cranklength=".20" ctrlrange="-.22 .13"/>
                    <position name="stick_curl" cranksite="stick_cylinder_rod" slidersite="stick_cylinder_base" cranklength=".18" ctrlrange=".28 .50"/>
                    <position name="bucket_curl" cranksite="bucket_cylinder_rod" slidersite="bucket_cylinder_base" cranklength=".12" ctrlrange="-.18 -.04"/>
                </actuator>
            </mujoco>
        """

        scene = mj.MjSpec.from_string(_XML)

        m = scene.compile()
        d = mj.MjData(m)

        d.qpos[m.joint("slew").qposadr[0]] = 0.0
        d.qpos[m.joint("boom_hinge").qposadr[0]] = 0.15
        d.qpos[m.joint("stick_hinge").qposadr[0]] = -0.95
        d.qpos[m.joint("bucket_hinge").qposadr[0]] = -0.55
        mj.mj_forward(m, d)
        d.ctrl[m.actuator("slew_position").id] = d.qpos[m.joint("slew").qposadr[0]]
        d.ctrl[m.actuator("boom_lift").id] = d.actuator_length[
            m.actuator("boom_lift").id
        ]
        d.ctrl[m.actuator("stick_curl").id] = d.actuator_length[
            m.actuator("stick_curl").id
        ]
        d.ctrl[m.actuator("bucket_curl").id] = d.actuator_length[
            m.actuator("bucket_curl").id
        ]

        return m, d

    @property
    def model(self):
        return self._model

    @property
    def data(self):
        return self._data

    @ms.thread
    def see_me_run(self, ss: ms.SimSync):
        while True:
            ss.step()


if __name__ == "__main__":
    sim = Sim()

    sim.run()
