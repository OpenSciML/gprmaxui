# GPR-MAX UI

![Library Logo](https://raw.githubusercontent.com/OpenSciML/gprmaxui/main/images/logo.png)

GprMax is open-source software that simulates electromagnetic wave propagation. It solves Maxwell’s equations in 3D using the Finite-Difference Time-Domain (FDTD) method. Although it was designed initially for modeling Ground Penetrating Radar (GPR), it can also be used to model electromagnetic wave propagation for many other applications.  GprMax-UI enhances this functionality by providing a high-level API for executing GprMax models, along with tools for visualization, analysis, and result interpretation.

The following video have been created using gprmaxui:

[![Watch the video](https://img.youtube.com/vi/oKURUSD32Ts/hqdefault.jpg)](https://www.youtube.com/watch?v=oKURUSD32Ts&ab_channel=HenryRuiz)[![Watch the video](https://img.youtube.com/vi/8RjslPXEv0Y/hqdefault.jpg)](https://www.youtube.com/watch?v=8RjslPXEv0Y&ab_channel=HenryRuiz)


## Prerequisites

- [Python 3.10+](https://www.python.org/downloads/)
- [gprMax](https://docs.gprmax.com/en/latest/)

## Install Pycuda
    
```bash
sudo apt install build-essential clang
sudo apt install libstdc++-12-dev
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
uv add pycuda --optional gpu
```

## Install gprMax

```bash
git clone https://github.com/gprMax/gprMax.git
sudo apt install libgomp1
sudo apt install libomp-dev
python setup.py build
python setup.py develop --no-deps
```

## Installation gprMaxUI

```bash
pip install gprmaxui
```

## Build the documentation

```bash
mkdocs build
mkdocs serve -a localhost:8000
```


## Usage

```Python
from pathlib import Path
from src.gprmaxui import *

# Create a GPRMax model
model = GprMaxModel(
    title="B scan from a single target buried in a dielectric sand-space",
    output_folder=Path("output"),
    domain_size=DomainSize(x=0.2, y=0.2, z=0.002),
    domain_resolution=DomainResolution(dx=0.002, dy=0.002, dz=0.002),
    time_window=TimeWindow(twt=3e-9),
)

# Register model materials
model.register_materials(
    Material(
        id="half_space", permittivity=6, conductivity=0, permeability=1, color="red"
    )
)

# Register model sources
tx_rx_sep = 2e-2
model.set_source(
    TxRxPair(
        tx=Tx(
            waveform=Waveform(wave_family="ricker", amplitude=1.0, frequency=1.5e9),
            source=HertzianDipole(polarization="z", x=0.03, y=0.15, z=0.0),
        ),
        rx=Rx(x=0.03 + tx_rx_sep, y=0.15, z=0.0),
        src_steps=SrcSteps(dx=0.002, dy=0.0, dz=0.0),
        rx_steps=RxSteps(dx=0.002, dy=0.0, dz=0.0),
    )
)

# add model geometries
box = DomainBox(
    x_min=0.0,
    y_min=0.0,
    z_min=0.0,
    x_max=0.2,
    y_max=0.145,
    z_max=0.002,
    material="half_space",
)
model.add_geometry(box)

cx = model.domain_size.x / 2
sphere = DomainSphere(cx=cx, cy=0.1, cz=0.0, radius=0.005, material="pec")
model.add_geometry(sphere)

print(model)
model.run(n="auto", geometry=True, snapshots=True)

```
