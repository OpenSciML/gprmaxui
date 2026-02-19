from gprmaxui.commands import *
from gprmaxui import GprMaxModel

if __name__ == "__main__":
    # Create a GPRMax model
    model = GprMaxModel(
        title="B scan from multiple root system images",
        output_folder=Path("output"),
        domain_size=DomainSize(x=0.6, y=0.3, z=0.002),
        domain_resolution=DomainResolution(dx=0.002, dy=0.002, dz=0.002),
        time_window=TimeWindow(twt=512),
    )

    # Register model materials
    model.register_materials(
        Material(id="half_space", permittivity=6, conductivity=0, permeability=1),
        Material(id="sand", permittivity=3.0, conductivity=0.01, permeability=1),
        Material(id="object", permittivity=1.0, conductivity=0.01, permeability=1),
    )

    # Register model sources
    tx_rx_sep = 2e-2
    model.set_source(
        TxRxPair(
            tx=Tx(
                waveform=Waveform(wave_family="ricker", amplitude=1.0, frequency=1.8e9),
                source=HertzianDipole(polarization="z", x=0.02, y=0.21, z=0.0),
            ),
            rx=Rx(x=0.02 + tx_rx_sep, y=0.21, z=0.0),
            src_steps=SrcSteps(dx=0.002, dy=0.0, dz=0.0),
            rx_steps=RxSteps(dx=0.002, dy=0.0, dz=0.0),
        )
    )

    box = DomainBox(
        x_min=0.0,
        y_min=0.0,
        z_min=0.0,
        y_max=0.2,  # depth
        x_max=model.domain_size.x,
        z_max=model.domain_size.z,
        material="sand",
    )
    model.add_geometry(box)

    model.add_geometry(
        GeometryObjectsRead(
            filename="root_images/root1.h5",
            materials_filename="root_images/root1.txt",
            x=0.1,
            y=0.0 + 0.1,
            z=0.0
        )
    )

    model.add_geometry(
        GeometryObjectsRead(
            filename="root_images/root2.h5",
            materials_filename="root_images/root2.txt",
            x=0.25,
            y=0.0 + 0.14,
            z=0.0
        )
    )

    model.add_geometry(
        GeometryObjectsRead(
            filename="root_images/root3.h5",
            materials_filename="root_images/root3.txt",
            x=0.4,
            y=0.0 + 0.12,
            z=0.0
        )
    )

    print(model)
    model.plot_geometry()
    #
    model.run(n="auto", geometry=True, snapshots=True)
    model.plot_data()
    model.plot_snapshot(trace_idx=60, iteration_idx=300)
    model.save_video("test.mp4", fps=25, figsize=(6, 10), cmap="jet")
