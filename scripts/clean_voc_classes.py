from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.label_tools import clean_voc_xml_dir_classes


def build_parser():
    import argparse

    parser = argparse.ArgumentParser(
        description="Keep only classes listed in data.yaml and drop other VOC XML objects."
    )
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--data-yaml", required=True, type=Path)
    parser.add_argument("--annotations-subdir", default="annotations")
    parser.add_argument("--backup", action="store_true")
    parser.add_argument("--backup-suffix", default=".xmlbak")
    parser.add_argument(
        "--remove-empty",
        action="store_true",
        help="Delete XML files that contain zero objects after class filtering.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    stats = clean_voc_xml_dir_classes(
        args.dataset_root,
        args.data_yaml,
        annotations_subdir=args.annotations_subdir,
        backup=args.backup,
        backup_suffix=args.backup_suffix,
        remove_empty=args.remove_empty,
        dry_run=args.dry_run,
    )
    for key, value in stats.items():
        print(f"{key}={value}")


if __name__ == "__main__":
    main()
