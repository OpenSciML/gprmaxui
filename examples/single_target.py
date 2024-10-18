from gprmaxui.commands import *
from gprmaxui import GprMaxModel
from gprmaxui.utils import make_images_grid

if __name__ == "__main__":
    # Create a GPRMax model
    model = GprMaxModel(
        title="B scan from a single target buried in a dielectric half-space",
        output_folder=Path("output").absolute(),
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

    model.run(n="auto", geometry=False, snapshots=False, gpu=[0,1])
    model.plot_data()
    model.plot_geometry()
    model.plot_snapshot(trace_idx=60, iteration_idx=300)
    model.save_video("test.mp4", fps=25, figsize=(6, 10))

    data_dict = model.data()
    for rx_component, data in data_dict.items():
        data_arr, dt = data
        print(data_arr.shape)

    captures = []
    for i in range(1, 500, 80):
        snapshot_image = model.plot_snapshot(
            trace_idx=35, iteration_idx=i, return_image=True
        )
        captures.append(snapshot_image)
    print(len(captures))
    output_image = make_images_grid(captures, num_cols=4)
    output_image.show()
