import csv
import shutil
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import autoresearch_phase1  # noqa: E402


class TestAutoResearchPhase1(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_extract_map50_from_results_csv(self):
        results_csv = self.tmp / "results.csv"
        with results_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["epoch", "metrics/mAP_0.5"])
            writer.writeheader()
            writer.writerow({"epoch": "0", "metrics/mAP_0.5": "0.501"})
            writer.writerow({"epoch": "1", "metrics/mAP_0.5": "0.612"})

        self.assertEqual(autoresearch_phase1.extract_map50(results_csv), 0.612)

    def test_build_train_command_uses_explicit_args(self):
        custom_python = "/opt/conda/envs/yolo/bin/python"
        command, save_dir = autoresearch_phase1.build_train_command(
            stage="smoke",
            candidate_id="cand_a",
            python_executable=Path(custom_python),
            data_path=ROOT / "data" / "dataAirVis.yaml",
            weights_path=ROOT / "checkpoint" / "yolov5_best.pt",
            device="0",
            seed=0,
            cli_overrides={"batch_size": 4, "imgsz": 640, "optimizer": "AdamW", "cos_lr": True, "workers": 2},
            hyp_path=ROOT / "data" / "hyps" / "hyp.scratch-low.yaml",
        )

        self.assertEqual(command[0], custom_python)
        self.assertIn("--optimizer", command)
        self.assertIn("AdamW", command)
        self.assertIn("--cos-lr", command)
        self.assertIn("--hyp", command)
        self.assertTrue(str(save_dir).endswith("runs\\autoresearch\\smoke\\cand_a") or str(save_dir).endswith("runs/autoresearch/smoke/cand_a"))

    def test_write_generated_hyp_merges_overrides(self):
        base_hyp = self.tmp / "base.yaml"
        base_hyp.write_text("lr0: 0.01\nmosaic: 1.0\n", encoding="utf-8")
        original_dir = autoresearch_phase1.GENERATED_HYP_DIR
        autoresearch_phase1.GENERATED_HYP_DIR = self.tmp / "generated"
        try:
            spec = autoresearch_phase1.CandidateSpec(
                candidate_id="cand_b",
                description="demo",
                cli_overrides={},
                hyp_overrides={"lr0": 0.02},
                base_hyp=base_hyp,
            )
            generated = autoresearch_phase1.write_generated_hyp(spec, "sprint")
            self.assertIsNotNone(generated)
            payload = autoresearch_phase1.load_yaml(generated)
            self.assertEqual(payload["lr0"], 0.02)
            self.assertEqual(payload["mosaic"], 1.0)
        finally:
            autoresearch_phase1.GENERATED_HYP_DIR = original_dir

    def test_decide_result_prefers_improvement(self):
        self.assertEqual(
            autoresearch_phase1.decide_result("sprint", 0.61, 0.60, True),
            ("success", "keep"),
        )
        self.assertEqual(
            autoresearch_phase1.decide_result("sprint", 0.59, 0.60, True),
            ("success", "discard"),
        )
        self.assertEqual(
            autoresearch_phase1.decide_result("smoke", 0.10, 0.60, True),
            ("success", "keep"),
        )
        self.assertEqual(
            autoresearch_phase1.decide_result("sprint", None, 0.60, False),
            ("crash", "discard"),
        )


if __name__ == "__main__":
    unittest.main()
