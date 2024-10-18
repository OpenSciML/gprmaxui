from src.gprmaxui import *


def add_targets_cluster(
    n: int,
    cluster_domain: DomainBox,
    cluster_spacing: float,
    targets_material: str,
    targets_radius: tuple,
):
    """
    Add a cluster of targets to the model.
    """
    clusters_points = []
    while True:
        x = np.random.uniform(low=cluster_domain.x_min, high=cluster_domain.x_max)
        y = np.random.uniform(low=cluster_domain.y_min, high=cluster_domain.y_max)
        z = np.random.uniform(low=cluster_domain.z_min, high=cluster_domain.z_max)
        target_radius = np.random.uniform(low=targets_radius[0], high=targets_radius[1])
        point_candidate = DomainPoint(x=x, y=y, z=z)
        if all(
            point_candidate.distance(point) > cluster_spacing
            for point in clusters_points
        ):
            clusters_points.append(point_candidate)
        if len(clusters_points) == n:
            break

    for point in clusters_points:
        sphere = DomainSphere(
            cx=point.x,
            cy=point.y,
            cz=point.z,
            radius=target_radius,
            material=targets_material,
        )
        model.add_geometry(sphere)


if __name__ == "__main__":
    # Create a GPRMax model
    model = GprMaxModel(
        title="B scan from a single target buried in a dielectric sand-space",
        output_folder=Path("output"),
        domain_size=DomainSize(x=0.3, y=0.2, z=0.002),
        domain_resolution=DomainResolution(dx=0.002, dy=0.002, dz=0.002),
        time_window=TimeWindow(twt=512),
    )

    # Register model materials
    model.register_materials(
        Material(id="half_space", permittivity=6, conductivity=0, permeability=1),
        Material(id="sand", permittivity=3.0, conductivity=0.01, permeability=1),
        Material(id="potato", permittivity=1.0, conductivity=0.01, permeability=1),
    )

    # Register model sources
    tx_rx_sep = 2e-2
    model.set_source(
        TxRxPair(
            tx=Tx(
                waveform=Waveform(wave_family="ricker", amplitude=1.0, frequency=1.8e9),
                source=HertzianDipole(polarization="z", x=0.03, y=0.15, z=0.0),
            ),
            rx=Rx(x=0.03 + tx_rx_sep, y=0.15, z=0.0),
            src_steps=SrcSteps(dx=0.002, dy=0.0, dz=0.0),
            rx_steps=RxSteps(dx=0.002, dy=0.0, dz=0.0),
        )
    )

    box = DomainBox(
        x_min=0.0,
        y_min=0.0,
        z_min=0.0,
        y_max=0.148,  # depth
        x_max=model.domain_size.x,
        z_max=model.domain_size.z,
        material="sand",
    )
    model.add_geometry(box)

    # add model geometries
    cluster_box = DomainBox.from_size(
        pos=box.center(),
        sz=DomainSize(x=0.1, y=0.1, z=model.domain_size.z),
        material="half_space",
    )

    # model.add_geometry(cluster_box)
    add_targets_cluster(
        n=6,
        cluster_domain=cluster_box,
        cluster_spacing=0.03,
        targets_material="potato",
        targets_radius=(0.005, 0.02),
    )

    model.plot_geometry()
    model.run(n="auto", geometry=True, snapshots=True, gpu=[0,1])
    model.plot_data()

    captures = []
    for i in range(1, 500, 80):
        snapshot_image = model.plot_snapshot(
            trace_idx=50, iteration_idx=i, return_image=True
        )
        captures.append(snapshot_image)
    print(len(captures))
    output_image = make_images_grid(captures, num_cols=4)
    output_image.show()

    model.save_video("test.mp4")
