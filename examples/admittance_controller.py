from admittance_controller_gui import Sim, use_motor_actuators

__all__ = ["Sim", "use_motor_actuators"]


if __name__ == "__main__":
    sim = Sim()
    print(
        "UR admittance demo. Press space to push the flange; green is actual, "
        "magenta is compliant target."
    )
    sim.run()
