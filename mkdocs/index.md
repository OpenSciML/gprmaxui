# GPR-MAX UI

GprMax-UI provides a high-level API to run gprMax models along with a set of functions for visualization, analysis and interpreting the results. 

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
    title="B scan from a single target buried in a dielectric half-space",
    output_folder=Path("output"),
    domain_size=DomainSize(x=0.2, y=0.2, z=0.002),
    domain_resolution=DomainResolution(dx=0.002, dy=0.002, dz=0.002),
    time_window=TimeWindow(twt=3e-9),
)
# Register model materials
model.register_materials(
    Material(id="half_space", permittivity=6, conductivity=0, permeability=1)
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

cx = box.center().x
cy = box.center().y
cz = box.center().z
sphere = DomainSphere(cx=cx, cy=cy, cz=cz, radius=0.005, material="pec")
model.add_geometry(sphere)

# Register model sources
tx_rx_sep = 2e-2
tx = Tx(
    waveform=Waveform(wave_family="ricker", amplitude=1.0, frequency=1.5e9),
    source=HertzianDipole(polarization="z", x=0.03, y=0.15, z=0.0),
)
rx = Rx(x=tx.source.x + tx_rx_sep, y=0.15, z=0.0)

model.set_source(
    TxRxPair(
        tx=tx,
        rx=rx,
        src_steps=SrcSteps(dx=0.002, dy=0.0, dz=0.0),
        rx_steps=RxSteps(dx=0.002, dy=0.0, dz=0.0),
    )
)

model.run(n="auto", geometry=True, snapshots=True)
model.plot_data()
```
