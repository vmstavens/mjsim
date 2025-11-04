# mj_sim

A repository to demonstrate, implement, test and learn control in MuJoCo.


### Getting Repository
To get the repository clone it into your directory of choice

```bash
git clone git@gitlab.sdu.dk:sdurobotics/teaching/mj_sim.git
```
then go into the repository by
```bash
cd mj_sim
```

## Installing Dependencies

After activating the virtual environment, you can install the project dependencies using pip. `mj_sim` uses [poetry](https://python-poetry.org/docs/) for managing dependencies and can thus be installed using
```bash
pip install poetry
```
Once poetry is installed in your virtual environment, install the project dependencies using 
```bash
poetry install
```
in the root of the project. Poetry will then install the project's base dependencies.

> [!WARNING]
> In case your poetry installation seems to run forever, kill the process and run the following 
> ```bash
> export PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring
> ```
> and try again.

Some modules requires more dependencies which can be installed in the following manner

 - **Learning** 
    ```bash
    poetry install --with learning
    ```
 - **Real**
    ```bash
    poetry install --with real
    ```
 - **All**
    ```bash
    poetry install --all-extras
    ```

# Real Time

Real time simulations can be found in [`sims/`](/sims/) and is run in the following manner as an example

```bash
python -m sims.opspace_ur5e  # Linux
mjpython -m sims.opspace_ur5e # MacOS
Python -m sims.opspace_ur5e   # Windows
```

A demo of this can be seen below

![here](/public/docs/real_time_sim_demo.gif)


# Learning

The learning uses JAX, Brax and MuJoCo for performing Deep Reinforcement Learning. See more [here](/learning/).

# Docs

A static website is generatd using [`pdoc`](https://pdoc.dev/) and can be found in `public/`. Access the documentation through 
```bash
<your-webbrowser-of-choise> public/index.html # e.g. firefox public/index.html
```
or online through the link found [here](https://sdurobotics.pages.sdu.dk/teaching/mj_sim/).

In case of generating pdoc documentation fails, remember to enable pdoc subprocess execution i.e.

```bash
export PDOC_ALLOW_EXEC=1
```

# VSCODE Compatibility

In case you are using [vscode](https://code.visualstudio.com/download) native type hinting is not a default for the MuJoCo or `ur_rtde` bindings (at the writing of this README.md). To add type hinting look here [here](https://github.com/google-deepmind/mujoco/issues/1292) (`pybind11-stubgen` is installed with the [`pyproject.toml`](pyproject.toml) when you install the dependencies through poetry) i.e. 
For MuJoCo:
```bash
pybind11-stubgen mujoco -o ./typings/
```
For `ur_thde`:
```bash
pybind11-stubgen rtde_control -o ./typings/
```
```bash
pybind11-stubgen rtde_receive -o ./typings/
```
```bash
pybind11-stubgen rtde_io -o ./typings/
```
For more documentation see [here](https://github.com/sizmailov/pybind11-stubgen).


Generate then the `.pyi` files for vscode to find your OMPL types and functions (stubgen is installed together with [mypy](https://github.com/python/mypy))
```bash
stubgen -p ompl -o typings
``` 

# Planning (OMPL)

[OMPL](https://ompl.kavrakilab.org/) is the library we use for planning and can be installed using the precompiled binary wheels from [here](https://github.com/ompl/ompl/releases/tag/prerelease). Download the wheel corresponding to your platform and install it using
```bash
pip install ompl-1.6.0-cp312-cp312-manylinux_2_28_x86_64.whl # replace the file w. your file
```

# License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

# Cite this work
To cite this work using `bibtex` please use the following
```bibtex
@misc{staven2024mjsim,
  author       = {Staven, Victor M},
  title        = {mj\_sim},
  version      = {3.0.0},
  year         = {2024},
  howpublished = {\url{https://gitlab.sdu.dk/sdurobotics/teaching/mj_sim}},
  note         = {A repository to demonstrate, implement, test and learn control in MuJoCo.},
}
```
