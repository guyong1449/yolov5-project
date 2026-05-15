from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.label_tools import build_parser, convert_voc_xml_dir_to_yolo


def main():
    parser = build_parser()
    args = parser.parse_args()
    if args.command != "voc-xml-to-yolo":
        parser.error("This wrapper only supports the 'voc-xml-to-yolo' command.")

    stats = convert_voc_xml_dir_to_yolo(
        args.dataset_root,
        args.data_yaml,
        backup=args.backup,
        backup_suffix=args.backup_suffix,
    )
    for key, value in stats.items():
        print(f"{key}={value}")


if __name__ == "__main__":
    main()
