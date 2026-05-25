import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.general import strip_optimizer  # noqa: E402
from utils.metrics import compute_ap  # noqa: E402


class TestMetricsNumpyCompat(unittest.TestCase):
    def test_compute_ap_supports_current_numpy(self):
        recall = np.array([0.25, 0.5, 0.75, 1.0], dtype=float)
        precision = np.array([1.0, 0.8, 0.6, 0.5], dtype=float)

        ap, mpre, mrec = compute_ap(recall, precision)

        self.assertIsInstance(ap, float)
        self.assertGreaterEqual(ap, 0.0)
        self.assertLessEqual(ap, 1.0)
        self.assertEqual(mpre.shape[0], precision.shape[0] + 2)
        self.assertEqual(mrec.shape[0], recall.shape[0] + 2)

    def test_strip_optimizer_supports_current_torch_load_defaults(self):
        tmp = Path(tempfile.mkdtemp())
        try:
            checkpoint = tmp / 'checkpoint.pt'
            stripped = tmp / 'stripped.pt'
            model = torch.nn.Linear(2, 1)
            torch.save(
                {
                    'model': model,
                    'optimizer': {'state': {'np': np.array([1.0, 2.0], dtype=np.float32)}},
                    'best_fitness': 0.5,
                    'ema': None,
                    'updates': 1,
                    'epoch': 0,
                },
                checkpoint,
            )

            strip_optimizer(checkpoint, stripped)

            self.assertTrue(stripped.is_file())
            loaded = torch.load(stripped, map_location='cpu', weights_only=False)
            self.assertIsNone(loaded['optimizer'])
            self.assertEqual(loaded['epoch'], -1)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == '__main__':
    unittest.main()
