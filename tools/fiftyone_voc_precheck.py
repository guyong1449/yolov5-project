from __future__ import annotations

import argparse
import csv
import struct
import xml.etree.ElementTree as ET
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.label_tools import _load_names


REPORT_COLUMNS = (
    "image_path",
    "xml_path",
    "issue_type",
    "detail",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run prechecks on a VOC image dataset and export an issue report CSV."
    )
    parser.add_argument("--data-dir", required=True, type=Path, help="Directory containing source images")
    parser.add_argument("--labels-dir", required=True, type=Path, help="Directory containing VOC XML files")
    parser.add_argument("--out-csv", required=True, type=Path, help="Output CSV path")
    parser.add_argument(
        "--data-yaml",
        type=Path,
        help="Optional dataset YAML used to validate class names against the names list.",
    )
    parser.add_argument(
        "--allowed-class",
        action="append",
        default=[],
        help="Additional allowed class name. Can be provided multiple times.",
    )
    return parser


def _add_issue(rows: list[dict[str, str]], image_path: Path | None, xml_path: Path | None, issue_type: str, detail: str) -> None:
    rows.append(
        {
            "image_path": str(image_path.resolve()) if image_path else "",
            "xml_path": str(xml_path.resolve()) if xml_path else "",
            "issue_type": issue_type,
            "detail": detail,
        }
    )


def _read_png_size(image_path: Path) -> tuple[int, int]:
    with image_path.open("rb") as fh:
        signature = fh.read(8)
        if signature != b"\x89PNG\r\n\x1a\n":
            raise ValueError(f"Unsupported PNG signature: {image_path}")
        header_length = fh.read(4)
        chunk_type = fh.read(4)
        if len(header_length) != 4 or chunk_type != b"IHDR":
            raise ValueError(f"Missing PNG IHDR chunk: {image_path}")
        width, height = struct.unpack(">II", fh.read(8))
    return int(width), int(height)


def _read_image_size(image_path: Path) -> tuple[int, int]:
    suffix = image_path.suffix.lower()
    if suffix == ".png":
        return _read_png_size(image_path)

    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError(
            f"Pillow is required to inspect image size for {image_path}. Install pillow in the active environment."
        ) from exc

    with Image.open(image_path) as img:
        return int(img.width), int(img.height)


def load_allowed_classes(data_yaml: Path | None, cli_allowed_classes: list[str]) -> set[str] | None:
    allowed = set(cli_allowed_classes)
    if data_yaml is not None:
        allowed.update(_load_names(Path(data_yaml)).keys())
    return allowed or None


def scan_voc_dataset(
    data_dir: Path,
    labels_dir: Path,
    *,
    allowed_classes: set[str] | None = None,
) -> list[dict[str, str]]:
    data_dir = Path(data_dir).resolve()
    labels_dir = Path(labels_dir).resolve()
    rows: list[dict[str, str]] = []

    if not data_dir.is_dir():
        raise FileNotFoundError(f"VOC data directory not found: {data_dir}")
    if not labels_dir.is_dir():
        raise FileNotFoundError(f"VOC labels directory not found: {labels_dir}")

    image_map = {path.stem: path for path in data_dir.iterdir() if path.is_file()}
    xml_map = {path.stem: path for path in labels_dir.glob("*.xml")}

    for stem in sorted(set(image_map) - set(xml_map)):
        _add_issue(rows, image_map[stem], None, "missing_xml", "Image has no matching XML annotation.")
    for stem in sorted(set(xml_map) - set(image_map)):
        _add_issue(rows, None, xml_map[stem], "missing_image", "XML has no matching source image.")

    for stem in sorted(set(image_map) & set(xml_map)):
        image_path = image_map[stem]
        xml_path = xml_map[stem]

        try:
            image_width, image_height = _read_image_size(image_path)
        except Exception as exc:  # pragma: no cover - kept broad for robust reporting
            _add_issue(rows, image_path, xml_path, "image_read_error", str(exc))
            continue

        try:
            root = ET.parse(xml_path).getroot()
        except ET.ParseError as exc:
            _add_issue(rows, image_path, xml_path, "xml_parse_error", str(exc))
            continue

        size_node = root.find("size")
        if size_node is None:
            _add_issue(rows, image_path, xml_path, "missing_size", "Annotation is missing the <size> node.")
            continue

        xml_width = int(float(size_node.findtext("width", default="0")))
        xml_height = int(float(size_node.findtext("height", default="0")))
        if (xml_width, xml_height) != (image_width, image_height):
            _add_issue(
                rows,
                image_path,
                xml_path,
                "image_size_mismatch",
                f"image=({image_width}, {image_height}) xml=({xml_width}, {xml_height})",
            )

        objects = root.findall("object")
        if not objects:
            _add_issue(rows, image_path, xml_path, "no_objects", "Annotation contains zero <object> entries.")
            continue

        for idx, obj in enumerate(objects, start=1):
            class_name = (obj.findtext("name") or "").strip()
            if not class_name:
                _add_issue(rows, image_path, xml_path, "empty_class_name", f"object_index={idx}")
                continue
            if allowed_classes is not None and class_name not in allowed_classes:
                _add_issue(rows, image_path, xml_path, "unknown_class", f"object_index={idx} class={class_name}")

            bndbox = obj.find("bndbox")
            if bndbox is None:
                _add_issue(rows, image_path, xml_path, "missing_bbox", f"object_index={idx}")
                continue

            try:
                xmin = int(float(bndbox.findtext("xmin", default="0")))
                ymin = int(float(bndbox.findtext("ymin", default="0")))
                xmax = int(float(bndbox.findtext("xmax", default="0")))
                ymax = int(float(bndbox.findtext("ymax", default="0")))
            except ValueError as exc:
                _add_issue(rows, image_path, xml_path, "invalid_bbox", f"object_index={idx} parse_error={exc}")
                continue

            if xmax <= xmin or ymax <= ymin:
                _add_issue(
                    rows,
                    image_path,
                    xml_path,
                    "invalid_bbox",
                    f"object_index={idx} bbox=({xmin}, {ymin}, {xmax}, {ymax})",
                )
                continue

            if xmin < 0 or ymin < 0 or xmax > image_width or ymax > image_height:
                _add_issue(
                    rows,
                    image_path,
                    xml_path,
                    "bbox_out_of_bounds",
                    f"object_index={idx} bbox=({xmin}, {ymin}, {xmax}, {ymax}) image=({image_width}, {image_height})",
                )

    return rows


def write_report_csv(rows: list[dict[str, str]], out_csv: Path) -> Path:
    out_csv = Path(out_csv).resolve()
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=REPORT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return out_csv


def main() -> None:
    args = build_parser().parse_args()
    allowed_classes = load_allowed_classes(args.data_yaml, args.allowed_class)
    rows = scan_voc_dataset(
        args.data_dir,
        args.labels_dir,
        allowed_classes=allowed_classes,
    )
    out_csv = write_report_csv(rows, args.out_csv)
    print(f"issues_found={len(rows)}")
    print(f"out_csv={out_csv}")


if __name__ == "__main__":
    main()
