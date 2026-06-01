import sys
import unittest
from argparse import Namespace
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import train  # noqa: E402
import detect  # noqa: E402
import val  # noqa: E402
from utils import metrics  # noqa: E402
from utils import torch_utils  # noqa: E402


class SelectDeviceNpuTests(unittest.TestCase):
    def test_select_device_accepts_multi_npu_request(self):
        with mock.patch("utils.torch_utils._npu_is_available", return_value=True), \
             mock.patch("utils.torch_utils.torch.npu.device_count", return_value=4), \
             mock.patch("utils.torch_utils.torch.npu.get_device_name", side_effect=lambda idx: f"Ascend-{idx}"), \
             mock.patch("utils.torch_utils.LOGGER.info") as info_log, \
             mock.patch("utils.torch_utils.torch.device", side_effect=lambda arg: arg):
            device = torch_utils.select_device("npu:0,1,2,3", batch_size=8)

        self.assertEqual(device, "npu:0")
        logged = "\n".join(call.args[0] for call in info_log.call_args_list)
        for idx in range(4):
            self.assertIn(f"NPU:{idx} (Ascend-{idx})", logged)

    def test_torch_distributed_zero_first_uses_generic_barrier(self):
        with mock.patch("utils.torch_utils.dist.is_available", return_value=True), \
             mock.patch("utils.torch_utils.dist.is_initialized", return_value=True), \
             mock.patch("utils.torch_utils.dist.barrier") as barrier:
            with torch_utils.torch_distributed_zero_first(1):
                pass

        barrier.assert_called_once_with()


class DetectParallelInferenceTests(unittest.TestCase):
    def test_parse_opt_defaults_to_four_card_npu(self):
        with mock.patch.object(sys, "argv", ["detect.py"]):
            opt = detect.parse_opt()
        self.assertEqual(opt.device, "npu:0,1,2,3")

    def test_frame_mod_sharding_covers_all_ranks(self):
        frames = list(range(1, 9))
        shards = {
            rank: [frame for frame in frames if detect.should_process_frame(frame, ddp_infer=True, rank=rank, world_size=4)]
            for rank in range(4)
        }

        for rank in range(4):
            self.assertTrue(shards[rank], f"rank {rank} should receive at least one frame")
        covered = sorted(frame for shard in shards.values() for frame in shard)
        self.assertEqual(covered, frames)


class TrainDistributedInitTests(unittest.TestCase):
    def _opt(self, device="npu:0,1,2,3"):
        return Namespace(image_weights=False, evolve=False, batch_size=8, device=device)

    def test_initialize_distributed_training_uses_hccl_for_npu(self):
        with mock.patch.object(train, "LOCAL_RANK", 2), \
             mock.patch.object(train, "RANK", 2), \
             mock.patch.object(train, "WORLD_SIZE", 4), \
             mock.patch("train.torch.npu.is_available", return_value=True), \
             mock.patch("train.dist.is_hccl_available", return_value=True, create=True), \
             mock.patch("train.torch.npu.device_count", return_value=4), \
             mock.patch("train.torch.npu.set_device") as set_device, \
             mock.patch("train.torch.device", side_effect=lambda kind, idx=None: f"{kind}:{idx}" if idx is not None else kind), \
             mock.patch("train.dist.init_process_group") as init_pg, \
             mock.patch("train.LOGGER.info") as info_log:
            resolved = train.initialize_distributed_training(self._opt(), device="npu:0")

        self.assertEqual(resolved, "npu:2")
        set_device.assert_called_once_with(2)
        init_pg.assert_called_once_with(backend="hccl")
        self.assertIn("backend=hccl", info_log.call_args[0][0])


class MetricsNpuFallbackTests(unittest.TestCase):
    def test_box_iou_moves_mixed_npu_tensors_to_cpu(self):
        class FakeTensor:
            def __init__(self, tensor, device_type):
                self._tensor = tensor
                self.device = type("Device", (), {"type": device_type})()

            def __getattr__(self, name):
                return getattr(self._tensor, name)

            def cpu(self):
                return self._tensor

        box1 = FakeTensor(train.torch.tensor([[0.0, 0.0, 2.0, 2.0]]), "npu")
        box2 = FakeTensor(train.torch.tensor([[1.0, 1.0, 3.0, 3.0]]), "cpu")
        iou = metrics.box_iou(box1, box2)
        self.assertAlmostEqual(float(iou[0, 0]), 1.0 / 7.0, places=6)

    def test_val_process_batch_handles_npu_cpu_mixed_tensors(self):
        class FakeTensor:
            def __init__(self, tensor, device_type):
                self._tensor = tensor
                self.device = type("Device", (), {"type": device_type})()
                self.shape = tensor.shape

            def __getitem__(self, item):
                return self._tensor[item]

            def __getattr__(self, name):
                return getattr(self._tensor, name)

            def cpu(self):
                return self._tensor

        detections = FakeTensor(train.torch.tensor([[0.0, 0.0, 2.0, 2.0, 0.9, 1.0]]), "npu")
        labels = FakeTensor(train.torch.tensor([[1.0, 0.0, 0.0, 2.0, 2.0]]), "cpu")
        with mock.patch("val.torch.tensor", side_effect=lambda data, dtype=None, device=None: train.torch.as_tensor(data, dtype=dtype)):
            correct = val.process_batch(detections, labels, train.torch.tensor([0.5]))
        self.assertTrue(bool(correct[0, 0]))


if __name__ == "__main__":
    unittest.main()
