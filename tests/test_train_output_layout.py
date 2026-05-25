"""Tests for train output layout helpers."""

import shutil
import unittest
from pathlib import Path

import pandas as pd

import train
from utils.plots import plot_results

REPO_ROOT = Path(__file__).resolve().parents[1]


class TrainOutputLayoutTests(unittest.TestCase):
    def test_train_image_dir_appends_images_folder(self):
        save_dir = REPO_ROOT / "runs" / "train" / "_pytest_output_layout"
        self.assertEqual(train.train_image_dir(save_dir), save_dir / "images")

    def test_plot_results_can_write_png_to_explicit_output_dir(self):
        save_dir = REPO_ROOT / "runs" / "train" / "_pytest_plot_results"
        image_dir = save_dir / "images"
        if save_dir.exists():
            shutil.rmtree(save_dir)
        image_dir.mkdir(parents=True, exist_ok=True)

        csv_path = save_dir / "results.csv"
        df = pd.DataFrame(
            {
                "epoch": [0, 1],
                "train/box_loss": [1.0, 0.9],
                "train/obj_loss": [1.0, 0.8],
                "train/cls_loss": [1.0, 0.7],
                "metrics/precision": [0.1, 0.2],
                "metrics/recall": [0.1, 0.2],
                "metrics/mAP_0.5": [0.1, 0.2],
                "metrics/mAP_0.5:0.95": [0.1, 0.2],
                "val/box_loss": [1.0, 0.9],
                "val/obj_loss": [1.0, 0.8],
                "val/cls_loss": [1.0, 0.7],
                "x/lr0": [0.01, 0.009],
                "x/lr1": [0.01, 0.009],
                "x/lr2": [0.01, 0.009],
            }
        )
        df.to_csv(csv_path, index=False)

        plot_results(file=csv_path, output_dir=image_dir)

        self.assertTrue((image_dir / "results.png").exists())
        self.assertFalse((save_dir / "results.png").exists())


if __name__ == "__main__":
    unittest.main()
