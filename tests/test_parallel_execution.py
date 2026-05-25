import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

import gprmaxui.gprmax_model as gprmax_model
from gprmaxui import GprMaxModel
from gprmaxui.commands import (
    DomainBox,
    DomainResolution,
    DomainSize,
    HertzianDipole,
    Material,
    Rx,
    RxSteps,
    SrcSteps,
    TimeWindow,
    Tx,
    TxRxPair,
    Waveform,
)


def build_model(output_folder: Path) -> GprMaxModel:
    model = GprMaxModel(
        title="test",
        output_folder=output_folder,
        domain_size=DomainSize(x=0.1, y=0.1, z=0.01),
        domain_resolution=DomainResolution(dx=0.01, dy=0.01, dz=0.01),
        time_window=TimeWindow(twt=5),
    )
    model.register_materials(
        Material(id="sand", permittivity=3.0, conductivity=0.0, permeability=1.0)
    )
    model.set_source(
        TxRxPair(
            tx=Tx(
                waveform=Waveform(wave_family="ricker", amplitude=1.0, frequency=1.5e9),
                source=HertzianDipole(polarization="z", x=0.01, y=0.09, z=0.0),
            ),
            rx=Rx(x=0.03, y=0.09, z=0.0),
            src_steps=SrcSteps(dx=0.01, dy=0.0, dz=0.0),
            rx_steps=RxSteps(dx=0.01, dy=0.0, dz=0.0),
        )
    )
    model.add_geometry(
        DomainBox(
            x_min=0.0,
            y_min=0.0,
            z_min=0.0,
            x_max=0.1,
            y_max=0.08,
            z_max=0.01,
            material="sand",
        )
    )
    return model


def fake_gprmax_api(calls):
    package = types.ModuleType("gprMax")
    module = types.ModuleType("gprMax.gprMax")

    def api(inputfile, *args, **kwargs):
        calls.append((inputfile, args, kwargs))

    module.api = api
    return patch.dict(sys.modules, {"gprMax": package, "gprMax.gprMax": module})


def prepare_video_inputs(output_folder: Path, n_traces: int, n_iterations: int) -> None:
    for trace_idx in range(1, n_traces + 1):
        output_folder.joinpath(f"geometry{trace_idx}.vti").write_text("geometry")
        snapshot_folder = output_folder.joinpath(f"sim_snaps{trace_idx}")
        snapshot_folder.mkdir(parents=True, exist_ok=True)
        for iteration_idx in range(1, n_iterations + 1):
            snapshot_folder.joinpath(f"snapshot{iteration_idx}.vti").write_text(
                "snapshot"
            )


class FakeVideoWriter:
    instances = []

    def __init__(self):
        self.frames = []
        self.released = False
        FakeVideoWriter.instances.append(self)

    def open(self, output_file, fourcc, fps, cap_size, is_color):
        self.output_file = output_file
        self.fourcc = fourcc
        self.fps = fps
        self.cap_size = cap_size
        self.is_color = is_color
        return True

    def write(self, frame):
        self.frames.append(int(frame[0, 0, 0]))

    def release(self):
        self.released = True


def fake_render_video_frame(task):
    image = Image.new("RGB", (1, 1), (task.frame_index, 0, 0))
    image.save(task.frame_path)
    return task.frame_index, task.frame_path


class ParallelExecutionTests(unittest.TestCase):
    def test_snapshot_stride_emits_only_requested_snapshots(self):
        calls = []
        with tempfile.TemporaryDirectory() as tmpdir, fake_gprmax_api(calls):
            model = build_model(Path(tmpdir))
            model.run(
                n=3,
                geometry=True,
                snapshots=True,
                snapshot_stride=2,
                geometry_only=True,
            )

            sim_text = Path(tmpdir).joinpath("sim.in").read_text()
            snapshot_lines = [
                line for line in sim_text.splitlines() if line.startswith("#snapshot:")
            ]

        self.assertEqual(len(snapshot_lines), 2)
        self.assertTrue(any(line.endswith(" 1 snapshot1") for line in snapshot_lines))
        self.assertTrue(any(line.endswith(" 3 snapshot3") for line in snapshot_lines))
        self.assertFalse(any(" snapshot2" in line for line in snapshot_lines))
        self.assertFalse(any(" snapshot4" in line for line in snapshot_lines))

    def test_run_writes_num_threads_and_passes_parallel_api_options(self):
        calls = []
        with tempfile.TemporaryDirectory() as tmpdir, fake_gprmax_api(calls):
            model = build_model(Path(tmpdir))
            model.run(
                n=4,
                geometry_only=True,
                num_threads=8,
                mpi="auto",
                gpu=[0, 1],
                geometry_fixed=True,
            )

            sim_text = Path(tmpdir).joinpath("sim.in").read_text()

        self.assertIn("#num_threads: 8", sim_text)
        self.assertEqual(len(calls), 1)
        _, _, api_kwargs = calls[0]
        self.assertEqual(api_kwargs["n"], 4)
        self.assertEqual(api_kwargs["mpi"], 3)
        self.assertEqual(api_kwargs["gpu"], [0, 1])
        self.assertTrue(api_kwargs["geometry_fixed"])
        self.assertTrue(api_kwargs["geometry_only"])

    def test_video_frame_indices_are_stable_and_ordered(self):
        self.assertEqual(
            gprmax_model._video_frame_indices(n_traces=2, n_iterations=5, frame_step=2),
            [
                (0, 0, 0),
                (1, 0, 2),
                (2, 0, 4),
                (3, 1, 0),
                (4, 1, 2),
                (5, 1, 4),
            ],
        )

    def test_save_video_writes_fake_rendered_frames_in_order_and_cleans_temp(self):
        FakeVideoWriter.instances = []
        with tempfile.TemporaryDirectory() as tmpdir:
            output_folder = Path(tmpdir).joinpath("output")
            output_folder.mkdir()
            temp_parent = Path(tmpdir).joinpath("frames")
            prepare_video_inputs(output_folder, n_traces=2, n_iterations=3)
            model = build_model(output_folder)
            model.data = lambda rx=1: {
                "Ez": (np.arange(6, dtype=np.float32).reshape(3, 2), 1e-9)
            }

            with (
                patch.object(
                    gprmax_model,
                    "_render_video_frame",
                    side_effect=fake_render_video_frame,
                ),
                patch.object(gprmax_model.cv2, "VideoWriter", FakeVideoWriter),
                patch.object(gprmax_model.cv2, "VideoWriter_fourcc", return_value=0),
            ):
                model.save_video(
                    output_folder.joinpath("test.mp4"),
                    rx_component="Ez",
                    frame_step=1,
                    workers=1,
                    temp_dir=temp_parent,
                )

            self.assertEqual(FakeVideoWriter.instances[0].frames, [0, 1, 2, 3, 4, 5])
            self.assertTrue(FakeVideoWriter.instances[0].released)
            self.assertEqual(list(temp_parent.iterdir()), [])

    def test_save_video_cleans_temp_after_render_failure(self):
        def failing_renderer(task):
            raise RuntimeError("render failed")

        with tempfile.TemporaryDirectory() as tmpdir:
            output_folder = Path(tmpdir).joinpath("output")
            output_folder.mkdir()
            temp_parent = Path(tmpdir).joinpath("frames")
            prepare_video_inputs(output_folder, n_traces=1, n_iterations=1)
            model = build_model(output_folder)
            model.data = lambda rx=1: {"Ez": (np.ones((1, 1), dtype=np.float32), 1e-9)}

            with patch.object(
                gprmax_model, "_render_video_frame", side_effect=failing_renderer
            ):
                with self.assertRaisesRegex(RuntimeError, "render failed"):
                    model.save_video(
                        output_folder.joinpath("test.mp4"),
                        rx_component="Ez",
                        frame_step=1,
                        workers=1,
                        temp_dir=temp_parent,
                    )

            self.assertEqual(list(temp_parent.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
