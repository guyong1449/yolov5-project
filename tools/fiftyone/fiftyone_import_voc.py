from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Import a VOC image dataset into FiftyOne and optionally launch the app."
    )
    parser.add_argument("--name", required=True, help="FiftyOne dataset name")
    parser.add_argument("--data-dir", required=True, type=Path, help="Directory containing source images")
    parser.add_argument("--labels-dir", required=True, type=Path, help="Directory containing VOC XML files")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete an existing FiftyOne dataset with the same name before import.",
    )
    parser.add_argument(
        "--label-field",
        default="ground_truth",
        help="Destination label field name used during import.",
    )
    parser.add_argument(
        "--no-app",
        action="store_true",
        help="Skip launching the FiftyOne App after import.",
    )
    parser.add_argument(
        "--wait",
        action="store_true",
        help="Block the process and keep the session open after launching the app.",
    )
    return parser


def validate_voc_layout(data_dir: Path, labels_dir: Path) -> tuple[Path, Path]:
    data_dir = Path(data_dir).resolve()
    labels_dir = Path(labels_dir).resolve()

    if not data_dir.is_dir():
        raise FileNotFoundError(f"VOC data directory not found: {data_dir}")
    if not labels_dir.is_dir():
        raise FileNotFoundError(f"VOC labels directory not found: {labels_dir}")

    image_count = sum(1 for path in data_dir.iterdir() if path.is_file())
    xml_count = sum(1 for path in labels_dir.glob("*.xml"))
    if image_count == 0:
        raise ValueError(f"No images found in: {data_dir}")
    if xml_count == 0:
        raise ValueError(f"No XML files found in: {labels_dir}")

    return data_dir, labels_dir


def import_voc_dataset(
    name: str,
    data_dir: Path,
    labels_dir: Path,
    *,
    overwrite: bool = False,
    label_field: str = "ground_truth",
):
    import fiftyone as fo

    data_dir, labels_dir = validate_voc_layout(data_dir, labels_dir)

    if overwrite and fo.dataset_exists(name):
        fo.delete_dataset(name)
    elif fo.dataset_exists(name):
        raise ValueError(f"FiftyOne dataset already exists: {name}. Use --overwrite to replace it.")

    dataset = fo.Dataset(name, persistent=True)
    dataset.add_dir(
        dataset_type=fo.types.VOCDetectionDataset,
        data_path=str(data_dir),
        labels_path=str(labels_dir),
        label_field=label_field,
    )
    return dataset


def launch_dataset_app(dataset, *, launch_app: bool = True):
    if not launch_app:
        return None

    import fiftyone as fo

    return fo.launch_app(dataset)


def main() -> None:
    args = build_parser().parse_args()
    dataset = import_voc_dataset(
        args.name,
        args.data_dir,
        args.labels_dir,
        overwrite=args.overwrite,
        label_field=args.label_field,
    )
    session = launch_dataset_app(
        dataset,
        launch_app=not args.no_app,
    )

    print(f"dataset_name={dataset.name}")
    print(f"samples_count={len(dataset)}")
    if session is not None:
        print(f"session_url={getattr(session, 'url', None)}")
        if args.wait:
            session.wait()


if __name__ == "__main__":
    main()
