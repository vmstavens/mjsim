_XMLS = {
    "bags": """

<mujoco model="Bag">
  <compiler angle="radian" meshdir=""/>
  <option timestep="0.0001" jacobian="sparse" integrator="implicitfast" solver="Newton"
          iterations="2000" tolerance="1e-4"/>
  <statistic extent="2.5" center="0 0 1"/>

  <visual>
    <map force="0.1" zfar="30"/>
    <global offwidth="2560" offheight="1440" elevation="-20" azimuth="120"/>
  </visual>

  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0"
             width="512" height="512"/>
    <texture name="texplane" type="2d" builtin="checker" rgb1=".2 .3 .4" rgb2=".1 0.15 0.2"
             width="512" height="512" mark="cross" markrgb=".8 .8 .8"/>
    <material name="matplane" reflectance="0.3" texture="texplane" texrepeat="10 10"
              texuniform="true"/>
    <model name="humanoid" file="examples/assets/humanoid.xml"/>
  </asset>

  <worldbody>
    <geom name="ground" type="plane" size="0 0 1" material="matplane" condim="1"/>

    <!-- Both lights are directional. A positional light inside the bag's closed volume (as in
         model/flex/scene.xml, which puts one at z=2) blacks out the cloth around it. -->
    <light directional="true" diffuse=".7 .7 .7" specular=".1 .1 .1" pos="0 0 4" dir="0 0 -1"/>
    <light directional="true" diffuse=".4 .4 .4" specular=".1 .1 .1" pos="3 -3 4" dir="-.6 .6 -.6"/>

    <!-- The bag, scaled up so the humanoid fits and held open by pinning the ring of vertices
         around its mouth. It hangs well above the floor, so the cloth catches the humanoid. -->
    <body name="bag_container" euler="1.57 0 0" pos="0 0 1.5">
      <flexcomp name="bag" type="mesh" file="examples/assets/bag.obj" dim="2" scale="3.0 3.09 3.0"
                radius="0.003" mass="5" rgba="0.6 0.6 0.62 0.6">
        <elasticity young="3e6" poisson="0.3" thickness="0.002" elastic2d="stretch" damping="5e-3"/>
        <pin id="193 194 195 196 197 198 199 200 201 202 203 204 205 206 207 208 209 210 211 212
                 213 214 215 216 239 240 241 242 243 244 245 246 247 248 249 250
                 251 252 253 254 255 256 257 258 259 260 267 268 269 270 271 272
                 645 646 647 648 649 650 651 652 653 654 655 656 657 658 659 660
                 661 662 663 664 665 666 667 668 671 672 673 675 677 678 679 682
                 687 688 689 690 711 712 713 714 715 716 717 718 719 720 721 722
                 723 724 725 726 727 728 729 730 755 756 758 761 762 764 766 770
                 772 773 803 804 805 806 813 815 819 820 821 822 1003 1005 1006
                 1009 1010 1013 1014 1015 1017 1019 1021 1024 1025 1051 1052
                 1054 1056 1059 1061 1063 1065 1066 1069 1070"/>
        <!-- priority=1: the flex parameters win outright instead of averaging with the
             humanoid's softer solref/solimp, which let limbs sink ~35mm into the cloth. -->
        <contact selfcollide="none" contype="1" conaffinity="1" priority="1"
                 solref="0.004 1" solimp="0.99 0.999 0.001"/>
      </flexcomp>
    </body>

    <!-- The humanoid, dropped above the bag's mouth; its torso sits at z=1.282 in its
         own model. It is the stock 40kg humanoid, hence the heavy, stiff cloth. -->
    <frame pos="0 0 1.618">
      <attach model="humanoid" body="torso" prefix="humanoid_"/>
    </frame>
  </worldbody>
</mujoco>
""",
    "drape": """
<mujoco model="Drape">
  <include file="examples/assets/scene.xml"/>

  <!-- Three cloths dropped over a sphere. All contacts are passive (cloth-cloth, self, and
       cloth-vs-static sphere) with stiffness carried implicitly by the effective metric. -->

  <!-- The metric solve for qacc_smooth is iterative and its blocks do not see the vertex-to-vertex
       coupling that contact introduces, so a contact-rich scene like this one needs a larger
       iteration budget than the default to converge it. -->
  <option timestep="0.0002" solver="Newton" tolerance="1e-6" iterations="400" integrator="implicitfast"/>

  <size memory="50M"/>

  <worldbody>
    <geom name="ball" type="sphere" size=".3" pos="0 0 .3" rgba=".45 .45 .5 1"/>

    <flexcomp type="grid" count="13 13 1" spacing=".055 .055 .055" pos="0 0 .68"
              radius=".004" mass=".25" name="cloth1" dim="2" rgba=".85 .35 .25 1">
      <contact selfcollide="auto" passive="true" solref="0.01 1" solimp=".95 .99 .0001"/>
      <elasticity young="2e4" poisson=".2" thickness="1e-3" elastic2d="both" damping="1e-2"/>
    </flexcomp>

    <flexcomp type="grid" count="13 13 1" spacing=".055 .055 .055" pos=".06 -.04 .78"
              radius=".004" mass=".25" name="cloth2" dim="2" rgba=".25 .55 .8 1">
      <contact selfcollide="auto" passive="true" solref="0.01 1" solimp=".95 .99 .0001"/>
      <elasticity young="2e4" poisson=".2" thickness="1e-3" elastic2d="both" damping="1e-2"/>
    </flexcomp>

    <flexcomp type="grid" count="13 13 1" spacing=".055 .055 .055" pos="-.05 .05 .88"
              radius=".004" mass=".25" name="cloth3" dim="2" rgba=".95 .8 .3 1">
      <contact selfcollide="auto" passive="true" solref="0.01 1" solimp=".95 .99 .0001"/>
      <elasticity young="2e4" poisson=".2" thickness="1e-3" elastic2d="both" damping="1e-2"/>
    </flexcomp>
  </worldbody>
</mujoco>
""",
}
import mujoco as mj

import mjsim as ms


class Sim(ms.BaseSim):
    def __init__(self):
        self._model, self._data = self._init()

    def _init(self) -> tuple[mj.MjModel, mj.MjData]:

        scene = mj.MjSpec.from_string(_XMLS["drape"])
        # scene = mj.MjSpec.from_string(_XMLS["bags"])
        m = scene.compile()
        return m, mj.MjData(m)

    @property
    def model(self):
        return self._model

    @property
    def data(self):
        return self._data

    def keyboard_callback(self, key):
        pass

    @ms.thread
    def see_me_run(self, ss: ms.SimSync):
        while True:
            ss.step()


if __name__ == "__main__":
    sim = Sim()

    sim.run()
