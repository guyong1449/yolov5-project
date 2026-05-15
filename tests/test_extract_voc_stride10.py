import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import extract_voc_stride10  # noqa: E402


class TestExtractVocStride10(unittest.TestCase):
    def test_build_detect_command_includes_stride10_defaults(self):
        args = extract_voc_stride10.parse_args(
            [
                "--weights",
                "checkpoint/yolov5_best.pt",
                "--source",
                "F:/1/video/1.mp4",
                "--voc-root",
                "F:/1/voc",
                "--data-yaml",
                "data/dataAirVis.yaml",
            ]
        )

        command = extract_voc_stride10.build_detect_command(args)

        expected = [
            sys.executable,
            str(ROOT / "detect.py"),
            "--weights",
            "checkpoint/yolov5_best.pt",
            "--source",
            "F:/1/video/1.mp4",
            "--voc-root",
            "F:/1/voc",
            "--data",
            "data/dataAirVis.yaml",
            "--imgsz",
            "640",
            "--device",
            "0",
            "--project",
            str(ROOT / "runs" / "detect"),
            "--name",
            "voc_stride10",
            "--vid-stride",
            "10",
            "--save-img-frames",
            "--nosave",
        ]
        self.assertEqual(command, expected)

    def test_build_detect_command_supports_explicit_overrides(self):
        args = extract_voc_stride10.parse_args(
            [
                "--weights",
                "weights.pt",
                "--source",
                "dataset",
                "--voc-root",
                "voc",
                "--data-yaml",
                "data/custom.yaml",
                "--imgsz",
                "1280",
                "720",
                "--device",
                "cpu",
                "--project",
                "runs/custom",
                "--name",
                "manual",
                "--python-exe",
                "D:/Miniconda3/python.exe",
                "--vid-stride",
                "3",
            ]
        )

        command = extract_voc_stride10.build_detect_command(args)

        self.assertEqual(command[0], "D:/Miniconda3/python.exe")
        self.assertIn("--imgsz", command)
        self.assertEqual(
            command[command.index("--imgsz") + 1: command.index("--imgsz") + 3],
            ["1280", "720"],
        )
        self.assertEqual(
            command[-4:],
            ["--vid-stride", "3", "--save-img-frames", "--nosave"],
        )

    def test_parse_args_rejects_vid_stride_less_than_one(self):
        for invalid_value in ("0", "-1"):
            with self.subTest(invalid_value=invalid_value):
                with self.assertRaises(SystemExit):
                    extract_voc_stride10.parse_args(
                        [
                            "--weights",
                            "weights.pt",
                            "--source",
                            "source.mp4",
                            "--voc-root",
                            "voc",
                            "--data-yaml",
                            "data.yaml",
                            "--vid-stride",
                            invalid_value,
                        ]
                    )


if __name__ == "__main__":
    unittest.main()
