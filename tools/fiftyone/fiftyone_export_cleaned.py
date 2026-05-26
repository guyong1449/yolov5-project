from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export a cleaned VOC dataset from a FiftyOne dataset or view."
    )
    parser.add_argument("--dataset-name", required=True, help="Existing FiftyOne dataset name")
    parser.add_argument("--export-dir", required=True, type=Path, help="Destination directory for the exported VOC data")
    parser.add_argument(
        "--label-field",
        default="ground_truth",
        help="Detection field to export in VOC format.",
    )
    parser.add_argument(
        "--exclude-tag",
        action="append",
        default=[],
        help="Exclude samples that have this tag. Can be provided multiple times.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting an existing export directory.",
    )
    return parser


def build_export_view(dataset, exclude_tags: list[str]):
    if not exclude_tags:
        return dataset
    return dataset.match_tags(exclude_tags, bool=False, all=False)


def export_cleaned_dataset(
    dataset_name: str,
    export_dir: Path,
    *,
    label_field: str = "ground_truth",
    exclude_tags: list[str] | None = None,
    overwrite: bool = False,
):
    import fiftyone as fo

    export_dir = Path(export_dir).resolve()
    if export_dir.exists() and any(export_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"Export directory already exists and is not empty: {export_dir}")

    if not fo.dataset_exists(dataset_name):
        raise ValueError(f"FiftyOne dataset not found: {dataset_name}")

    dataset = fo.load_dataset(dataset_name)
    view = build_export_view(dataset, exclude_tags or [])
    view.export(
        export_dir=str(export_dir),
        dataset_type=fo.types.VOCDetectionDataset,
        label_field=label_field,
        export_media=True,
        overwrite=overwrite,
    )
    return view


def main() -> None:
    args = build_parser().parse_args()
    view = export_cleaned_dataset(
        args.dataset_name,
        args.export_dir,
        label_field=args.label_field,
        exclude_tags=args.exclude_tag,
        overwrite=args.overwrite,
    )
    print(f"export_dir={Path(args.export_dir).resolve()}")
    print(f"samples_exported={len(view)}")


if __name__ == "__main__":
    main()
