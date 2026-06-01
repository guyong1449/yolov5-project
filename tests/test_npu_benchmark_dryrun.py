import importlib.util
import os
import shutil
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest import mock

import torch

from utils import general
from utils.general import Profile
from utils.env_config import get_video_dir

REPO_ROOT = Path(__file__).resolve().parents[1]
RUN_WITH_LOG_PATH = REPO_ROOT / "scripts" / "run_with_log.py"


def _load_run_with_log_module():
    spec = importlib.util.spec_from_file_location("run_with_log", RUN_WITH_LOG_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class ProfileTimingTests(unittest.TestCase):
    def test_profile_uses_shared_time_sync(self):
        with mock.patch("utils.torch_utils.time_sync", side_effect=[10.0, 10.5]) as mocked:
            prof = Profile()
            with prof:
                pass

        self.assertEqual(mocked.call_count, 2)
        self.assertAlmostEqual(prof.dt, 0.5)
        self.assertAlmostEqual(prof.t, 0.5)


class DetectDryRunTimingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.run_with_log = _load_run_with_log_module()

    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp(prefix="npu_dryrun_"))
        self.detect_stub = self.tmpdir / "detect.py"
        self.detect_stub.write_text(
            textwrap.dedent(
                """
                import time

                print("Loaded names: {0: 'drone'}", flush=True)
                time.sleep(0.01)
                print("Speed: 1.0ms pre-process, 2.0ms inference, 3.0ms NMS per image at shape (1, 3, 640, 640)", flush=True)
                """
            ).strip()
            + "\n",
            encoding="utf-8",
        )
        self.log_dir = REPO_ROOT / "runs" / "logs" / "_pytest_npu_benchmark"
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_run_with_log_writes_detect_timing_sidecar(self):
        log_file = self.log_dir / "npu_detect_dryrun.log"
        md_file = self.log_dir / "npu_detect_dryrun.md"
        timing_file = self.log_dir / "npu_detect_dryrun_inference_time.txt"
        for path in (log_file, md_file, timing_file):
            if path.exists():
                path.unlink()

        code = self.run_with_log.run_with_log(
            [
                sys.executable,
                str(self.detect_stub),
                "--device",
                "npu:0",
                "--project",
                "runs/detect",
                "--name",
                "npu_video20_r1",
            ],
            log_file=log_file,
            md_file=md_file,
            realtime=False,
            cwd=REPO_ROOT,
        )

        self.assertEqual(code, 0)
        self.assertTrue(timing_file.exists())
        timing_text = timing_file.read_text(encoding="utf-8")
        self.assertIn("run_name=npu_video20_r1", timing_text)
        self.assertIn("timing_mode=synced_speed", timing_text)
        self.assertIn("wall_clock_seconds=", timing_text)
        self.assertIn("speed_line=Speed: 1.0ms pre-process, 2.0ms inference, 3.0ms NMS per image at shape (1, 3, 640, 640)", timing_text)

    def test_script_wrapper_is_treated_as_detect_timing_source(self):
        self.assertTrue(self.run_with_log._is_detect_command(["bash", "tools/npu_video_benchmark.sh"]))

        old_device = self.run_with_log.os.environ.get("DEVICE")
        self.run_with_log.os.environ["DEVICE"] = "npu:0"
        try:
            self.assertEqual(self.run_with_log._timing_mode(["bash", "tools/npu_video_benchmark.sh"]), "synced_speed")
        finally:
            if old_device is None:
                self.run_with_log.os.environ.pop("DEVICE", None)
            else:
                self.run_with_log.os.environ["DEVICE"] = old_device


class ParallelDetectDryRunSummaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.run_with_log = _load_run_with_log_module()

    def setUp(self):
        self.video_src = os.path.join(get_video_dir(), "video20.mp4")
        self.tmpdir = Path(tempfile.mkdtemp(prefix="npu_parallel_detect_"))
        self.detect_stub = self.tmpdir / "detect.py"
        self.detect_stub.write_text(
            textwrap.dedent(
                f"""
                print("INFER init: rank=0 local_rank=0 world_size=4 device=npu:0 source={self.video_src} shard_mode=mod", flush=True)
                print("INFER init: rank=1 local_rank=1 world_size=4 device=npu:1 source={self.video_src} shard_mode=mod", flush=True)
                print("INFER init: rank=2 local_rank=2 world_size=4 device=npu:2 source={self.video_src} shard_mode=mod", flush=True)
                print("INFER init: rank=3 local_rank=3 world_size=4 device=npu:3 source={self.video_src} shard_mode=mod", flush=True)
                print("INFER done: rank=0 processed_frames=1041", flush=True)
                print("INFER done: rank=1 processed_frames=1041", flush=True)
                print("INFER done: rank=2 processed_frames=1041", flush=True)
                print("INFER done: rank=3 processed_frames=1042", flush=True)
                print("INFER aggregate: world_size=4 rank_frame_counts=0:1041,1:1041,2:1041,3:1042 aggregate_frames=4165 parallel_infer_confirmed=true", flush=True)
                print("Speed: 1.0ms pre-process, 2.0ms inference, 3.0ms NMS per image at shape (1, 3, 640, 640)", flush=True)
                """
            ).strip()
            + "\n",
            encoding="utf-8",
        )
        self.log_dir = REPO_ROOT / "runs" / "logs" / "_pytest_npu_benchmark"
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_run_with_log_writes_parallel_inference_summary(self):
        log_file = self.log_dir / "npu_parallel_detect.log"
        md_file = self.log_dir / "npu_parallel_detect.md"
        summary_file = self.log_dir / "npu_parallel_detect_parallel_inference_summary.txt"
        for path in (log_file, md_file, summary_file):
            if path.exists():
                path.unlink()

        code = self.run_with_log.run_with_log(
            [
                sys.executable,
                "-m",
                "torch.distributed.run",
                "--nproc_per_node",
                "4",
                str(self.detect_stub),
                "--device",
                "npu:0,1,2,3",
                "--source",
                self.video_src,
                "--ddp-infer",
                "--project",
                "runs/detect",
                "--name",
                "npu_video20_ddp_r1",
            ],
            log_file=log_file,
            md_file=md_file,
            realtime=False,
            cwd=REPO_ROOT,
        )

        self.assertEqual(code, 0)
        self.assertTrue(summary_file.exists())
        summary = summary_file.read_text(encoding="utf-8")
        self.assertIn("run_name=npu_video20_ddp_r1", summary)
        self.assertIn("world_size=4", summary)
        self.assertIn(f"source={self.video_src}", summary)
        self.assertIn("rank_done_counts=0:1041,1:1041,2:1041,3:1042", summary)
        self.assertIn("aggregate_frames=4165", summary)
        self.assertIn("parallel_infer_confirmed=true", summary)

    def test_script_wrapper_is_treated_as_parallel_detect_source(self):
        self.assertTrue(self.run_with_log._is_detect_command(["bash", "tools/npu_ddp_detect_benchmark.sh"]))

    def test_run_with_log_writes_batch_buffer_parallel_summary(self):
        log_file = self.log_dir / "npu_parallel_batch_buffer.log"
        md_file = self.log_dir / "npu_parallel_batch_buffer.md"
        summary_file = self.log_dir / "npu_parallel_batch_buffer_parallel_inference_summary.txt"
        for path in (log_file, md_file, summary_file):
            if path.exists():
                path.unlink()

        batch_stub = self.tmpdir / "batch_buffer_case" / "detect.py"
        batch_stub.parent.mkdir(parents=True, exist_ok=True)
        batch_stub.write_text(
            textwrap.dedent(
                f"""
                print("INFER init: rank=0 local_rank=0 world_size=4 device=npu:0 source={self.video_src} shard_mode=mod infer_mode=batch_buffer buffer_size=3", flush=True)
                print("INFER init: rank=1 local_rank=1 world_size=4 device=npu:1 source={self.video_src} shard_mode=mod infer_mode=batch_buffer buffer_size=3", flush=True)
                print("INFER init: rank=2 local_rank=2 world_size=4 device=npu:2 source={self.video_src} shard_mode=mod infer_mode=batch_buffer buffer_size=3", flush=True)
                print("INFER init: rank=3 local_rank=3 world_size=4 device=npu:3 source={self.video_src} shard_mode=mod infer_mode=batch_buffer buffer_size=3", flush=True)
                print("INFER done: rank=0 processed_frames=2 processed_batches=2 infer_mode=batch_buffer buffer_size=3", flush=True)
                print("INFER done: rank=1 processed_frames=2 processed_batches=2 infer_mode=batch_buffer buffer_size=3", flush=True)
                print("INFER done: rank=2 processed_frames=1 processed_batches=2 infer_mode=batch_buffer buffer_size=3", flush=True)
                print("INFER done: rank=3 processed_frames=0 processed_batches=2 infer_mode=batch_buffer buffer_size=3", flush=True)
                print("INFER aggregate: world_size=4 rank_frame_counts=0:2,1:2,2:1,3:0 aggregate_frames=5 infer_mode=batch_buffer buffer_size=3 batch_count=2 tail_batch_size=2 parallel_infer_confirmed=true", flush=True)
                print("Speed: 1.0ms pre-process, 2.0ms inference, 3.0ms NMS per image at shape (1, 3, 640, 640)", flush=True)
                """
            ).strip()
            + "\n",
            encoding="utf-8",
        )

        code = self.run_with_log.run_with_log(
            [
                sys.executable,
                "-m",
                "torch.distributed.run",
                "--nproc_per_node",
                "4",
                str(batch_stub),
                "--device",
                "npu:0,1,2,3",
                "--source",
                self.video_src,
                "--ddp-infer",
                "--batch-buffer",
                "--buffer-size",
                "3",
                "--project",
                "runs/detect",
                "--name",
                "npu_video20_batch_buffer_r1",
            ],
            log_file=log_file,
            md_file=md_file,
            realtime=False,
            cwd=REPO_ROOT,
        )

        self.assertEqual(code, 0)
        self.assertTrue(summary_file.exists())
        summary = summary_file.read_text(encoding="utf-8")
        self.assertIn("infer_mode=batch_buffer", summary)
        self.assertIn("buffer_size=3", summary)
        self.assertIn("rank_batch_counts=0:2,1:2,2:2,3:2", summary)
        self.assertIn("batch_count=2", summary)
        self.assertIn("tail_batch_size=2", summary)


class TrainDryRunSummaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.run_with_log = _load_run_with_log_module()

    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp(prefix="npu_train_dryrun_"))
        self.train_stub = self.tmpdir / "train.py"
        self.train_stub.write_text(
            textwrap.dedent(
                """
                print("DDP init: rank=0 local_rank=0 world_size=4 backend=hccl device=npu:0", flush=True)
                print("DDP init: rank=1 local_rank=1 world_size=4 backend=hccl device=npu:1", flush=True)
                print("DDP init: rank=2 local_rank=2 world_size=4 backend=hccl device=npu:2", flush=True)
                print("DDP init: rank=3 local_rank=3 world_size=4 backend=hccl device=npu:3", flush=True)
                print("3 epochs completed in 0.001 hours.", flush=True)
                """
            ).strip()
            + "\n",
            encoding="utf-8",
        )
        self.log_dir = REPO_ROOT / "runs" / "logs" / "_pytest_npu_benchmark"
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_run_with_log_writes_training_ddp_summary(self):
        log_file = self.log_dir / "npu_train_dryrun.log"
        md_file = self.log_dir / "npu_train_dryrun.md"
        summary_file = self.log_dir / "npu_train_dryrun_ddp_training_summary.txt"
        for path in (log_file, md_file, summary_file):
            if path.exists():
                path.unlink()

        code = self.run_with_log.run_with_log(
            [
                sys.executable,
                "-m",
                "torch.distributed.run",
                "--nproc_per_node",
                "4",
                str(self.train_stub),
                "--device",
                "npu:0,1,2,3",
                "--project",
                "runs/train",
                "--name",
                "panel_smoke10_ddp_r1",
            ],
            log_file=log_file,
            md_file=md_file,
            realtime=False,
            cwd=REPO_ROOT,
        )

        self.assertEqual(code, 0)
        self.assertTrue(summary_file.exists())
        summary = summary_file.read_text(encoding="utf-8")
        self.assertIn("run_name=panel_smoke10_ddp_r1", summary)
        self.assertIn("world_size=4", summary)
        self.assertIn("backend=hccl", summary)
        self.assertIn("device=npu:0,1,2,3", summary)
        self.assertIn("ddp_confirmed=true", summary)
        self.assertIn("epochs_completed=3 epochs completed in 0.001 hours.", summary)

    def test_script_wrapper_is_treated_as_train_summary_source(self):
        self.assertTrue(self.run_with_log._is_train_command(["bash", "tools/npu_ddp_training_benchmark.sh"]))


class NpuNmsFallbackTests(unittest.TestCase):
    def test_non_max_suppression_moves_npu_nms_to_cpu(self):
        pred = torch.zeros((1, 2, 6), dtype=torch.float32)
        pred[0, 0, :4] = torch.tensor([10.0, 10.0, 4.0, 4.0])
        pred[0, 0, 4] = 0.9
        pred[0, 0, 5] = 0.8
        pred[0, 1, :4] = torch.tensor([10.5, 10.5, 4.0, 4.0])
        pred[0, 1, 4] = 0.8
        pred[0, 1, 5] = 0.7

        class FakeTensor:
            def __init__(self, tensor):
                self._tensor = tensor
                self.device = type("Device", (), {"type": "npu"})()
                self.shape = tensor.shape

            def __getitem__(self, item):
                return self._tensor[item]

            def __getattr__(self, name):
                return getattr(self._tensor, name)

            def cpu(self):
                return self._tensor

        fake_pred = FakeTensor(pred)

        captured = {}

        def fake_nms(boxes, scores, iou_thres):
            captured["boxes_device"] = boxes.device.type
            captured["scores_device"] = scores.device.type
            return torch.tensor([0], dtype=torch.long)

        real_zeros = torch.zeros

        def fake_zeros(*args, **kwargs):
            kwargs.pop("device", None)
            return real_zeros(*args, **kwargs)

        with mock.patch("utils.general.torchvision.ops.nms", side_effect=fake_nms), \
             mock.patch("utils.general.torch.zeros", side_effect=fake_zeros):
            output = general.non_max_suppression(fake_pred, conf_thres=0.25, iou_thres=0.45)

        self.assertEqual(captured["boxes_device"], "cpu")
        self.assertEqual(captured["scores_device"], "cpu")
        self.assertEqual(len(output), 1)
        self.assertEqual(output[0].shape[0], 1)


if __name__ == "__main__":
    unittest.main()
