from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.label_tools import convert_voc_xml_dir_to_fiftyone


def build_parser():
    import argparse

    parser = argparse.ArgumentParser(
        description="Normalize VOC XML annotations for FiftyOne VOCDetectionDataset import."
    )
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--data-yaml", required=True, type=Path)
    parser.add_argument("--source-subdir", default="annotations")
    parser.add_argument("--output-subdir", default="fiftyone_labels")
    parser.add_argument(
        "--layout",
        choices=("labels_only", "fiftyone_voc"),
        default="fiftyone_voc",
        help="labels_only writes <output-subdir>/; fiftyone_voc also creates fiftyone_voc/data+labels.",
    )
    parser.add_argument("--no-overwrite", action="store_true")
    parser.add_argument(
        "--validate-classes",
        action="store_true",
        help="Reject XML classes that are not listed in data.yaml.",
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    stats = convert_voc_xml_dir_to_fiftyone(
        args.dataset_root,
        args.data_yaml,
        source_subdir=args.source_subdir,
        output_subdir=args.output_subdir,
        layout=args.layout,
        overwrite=not args.no_overwrite,
        validate_classes=args.validate_classes,
    )
    for key, value in stats.items():
        print(f"{key}={value}")


if __name__ == "__main__":
    main()
