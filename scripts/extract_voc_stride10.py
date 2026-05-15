import argparse
import subprocess
import sys
from pathlib import Path
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROJECT = ROOT / "runs" / "detect"
DEFAULT_NAME = "voc_stride10"


def positive_int(value: str) -> int:
    """Parse a CLI integer that must be at least 1."""
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("vid-stride must be >= 1")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for the VOC extraction wrapper."""
    parser = argparse.ArgumentParser(
        description="Run detect.py for VOC-style frame/XML extraction with stride defaults."
    )
    parser.add_argument("--weights", required=True, help="Path to model weights.")
    parser.add_argument("--source", required=True, help="Input source for detect.py.")
    parser.add_argument("--voc-root", required=True, help="VOC output root directory.")
    parser.add_argument("--data-yaml", required=True, help="Dataset YAML passed to detect.py --data.")
    parser.add_argument(
        "--imgsz",
        nargs="+",
        type=int,
        default=[640],
        help="Inference image size passed through to detect.py.",
    )
    parser.add_argument("--device", default="0", help="Inference device, e.g. 0 or cpu.")
    parser.add_argument("--project", default=str(DEFAULT_PROJECT), help="detect.py --project value.")
    parser.add_argument("--name", default=DEFAULT_NAME, help="detect.py --name value.")
    parser.add_argument(
        "--python-exe",
        default=sys.executable,
        help="Python executable used to launch detect.py.",
    )
    parser.add_argument(
        "--vid-stride",
        type=positive_int,
        default=10,
        help="Video frame stride passed to detect.py.",
    )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse wrapper CLI arguments."""
    return build_parser().parse_args(argv)


def build_detect_command(args: argparse.Namespace) -> list[str]:
    """Build the exact subprocess command list for detect.py."""
    command = [
        str(args.python_exe),
        str(ROOT / "detect.py"),
        "--weights",
        str(args.weights),
        "--source",
        str(args.source),
        "--voc-root",
        str(args.voc_root),
        "--data",
        str(args.data_yaml),
        "--imgsz",
        *[str(size) for size in args.imgsz],
        "--device",
        str(args.device),
        "--project",
        str(args.project),
        "--name",
        str(args.name),
        "--vid-stride",
        str(args.vid_stride),
        "--save-img-frames",
        "--nosave",
    ]
    return command


def main(argv: Sequence[str] | None = None) -> int:
    """Parse arguments, build the detect.py command, and execute it."""
    args = parse_args(argv)
    command = build_detect_command(args)
    subprocess.run(command, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
