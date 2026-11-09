"""CPU batching with live state views and deterministic scalar-loop validation.

The batch API arrived in MuJoCo 3.17 and ships with MuJoCo 4's core bindings.
This example needs no GPU, graphics context, or external assets.
"""

import argparse

import mujoco as mj
import numpy as np
from mujoco.batch import Batch


def run(nsim: int = 32, nstep: int = 200, nthread: int = 4) -> np.ndarray:
    if nsim < 1 or nstep < 1 or nthread < 1:
        raise ValueError("nsim, nstep, and nthread must be positive")
    model = mj.MjModel.from_xml_string("""
      <mujoco>
        <worldbody><body>
          <joint name="hinge" axis="0 1 0" damping="0.1"/>
          <geom type="capsule" size="0.02" fromto="0 0 0 0 0 -0.3"/>
        </body></worldbody>
        <actuator><motor joint="hinge"/></actuator>
      </mujoco>
    """)
    batch = Batch(model, nsim=nsim, nthread=nthread)
    initial = np.linspace(-0.5, 0.5, nsim)
    controls = np.linspace(-0.02, 0.02, nsim)
    # Named views write directly into each simulation's integration state.
    batch.joint("hinge").qpos[:, 0] = initial
    batch.bind("ctrl")[:, 0] = controls
    batch.step(nstep=nstep)
    result = batch.bind("qpos").copy()
    for i in range(nsim):
        data = mj.MjData(model)
        data.qpos[0] = initial[i]
        data.ctrl[0] = controls[i]
        mj.mj_step(model, data, nstep=nstep)
        np.testing.assert_array_equal(result[i], data.qpos)
    # Derived fields are computed at the beginning of the last step, just as
    # with mj_step. Explicit forward() refreshes them at the resulting state.
    batch.forward()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nsim", type=int, default=32)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    result = run(args.nsim, args.steps, args.threads)
    print(f"{len(result)} batched simulations matched the scalar loop exactly.")


if __name__ == "__main__":
    main()
