import argparse
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import yaml


def _fmt_float(value: float) -> str:
    return f"{value:.6f}".rstrip("0").rstrip(".")


def _load_names(data_yaml: Path) -> list[str]:
    data = yaml.safe_load(Path(data_yaml).read_text(encoding="utf-8")) or {}
    names = data.get("names") or []
    if not names:
        raise ValueError(f"names missing in {data_yaml}")
    return list(names)


def _image_map(images_dir: Path) -> dict[str, Path]:
    image_paths = {}
    for path in sorted(images_dir.iterdir()):
        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
            image_paths[path.stem] = path
    return image_paths


def trim_label_conf_columns(labels_dir: Path, backup_suffix: str = ".bak") -> dict[str, int]:
    labels_dir = Path(labels_dir)
    files_changed = 0
    lines_trimmed = 0
    for path in sorted(labels_dir.glob("*.txt")):
        original_lines = path.read_text(encoding="utf-8").splitlines()
        updated_lines = []
        changed = False
        for line in original_lines:
            parts = line.split()
            if len(parts) > 5:
                updated_lines.append(" ".join(parts[:5]))
                lines_trimmed += 1
                changed = True
            else:
                updated_lines.append(line)
        if changed:
            shutil.copy2(path, Path(f"{path}{backup_suffix}"))
            path.write_text("\n".join(updated_lines) + "\n", encoding="utf-8")
            files_changed += 1
    return {"files_changed": files_changed, "lines_trimmed": lines_trimmed}


def convert_voc_xml_dir_to_yolo(
    dataset_root: Path,
    data_yaml: Path,
    *,
    backup: bool = False,
    backup_suffix: str = ".xmlbak",
) -> dict[str, int]:
    dataset_root = Path(dataset_root)
    images_dir = dataset_root / "images"
    annotations_dir = dataset_root / "annotations"
    labels_dir = dataset_root / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)

    names = _load_names(data_yaml)
    class_map = {name: idx for idx, name in enumerate(names)}
    images = _image_map(images_dir)

    xml_files_seen = 0
    labels_written = 0
    objects_written = 0
    skipped_xml_without_image = 0

    for xml_path in sorted(annotations_dir.glob("*.xml")):
        xml_files_seen += 1
        root = ET.parse(xml_path).getroot()
        stem = xml_path.stem
        image_path = images.get(stem)
        if image_path is None:
            filename = root.findtext("filename", default="")
            if filename:
                image_path = images.get(Path(filename).stem)
        if image_path is None:
            skipped_xml_without_image += 1
            continue

        size = root.find("size")
        width = float(size.findtext("width", default="0")) if size is not None else 0.0
        height = float(size.findtext("height", default="0")) if size is not None else 0.0
        if width <= 0 or height <= 0:
            raise ValueError(f"invalid image size in {xml_path}")

        rows = []
        for obj in root.findall("object"):
            class_name = (obj.findtext("name", default="") or "").strip()
            if class_name not in class_map:
                raise ValueError(f"unknown class {class_name!r} in {xml_path}")
            bbox = obj.find("bndbox")
            if bbox is None:
                continue
            xmin = float(bbox.findtext("xmin", default="0"))
            ymin = float(bbox.findtext("ymin", default="0"))
            xmax = float(bbox.findtext("xmax", default="0"))
            ymax = float(bbox.findtext("ymax", default="0"))
            x_center = ((xmin + xmax) / 2.0) / width
            y_center = ((ymin + ymax) / 2.0) / height
            box_w = (xmax - xmin) / width
            box_h = (ymax - ymin) / height
            rows.append(
                f"{class_map[class_name]} "
                f"{_fmt_float(x_center)} {_fmt_float(y_center)} {_fmt_float(box_w)} {_fmt_float(box_h)}"
            )
            objects_written += 1

        label_path = labels_dir / f"{image_path.stem}.txt"
        label_path.write_text(("\n".join(rows) + "\n") if rows else "", encoding="utf-8")
        labels_written += 1
        if backup:
            shutil.copy2(xml_path, Path(f"{xml_path}{backup_suffix}"))

    return {
        "xml_files_seen": xml_files_seen,
        "labels_written": labels_written,
        "objects_written": objects_written,
        "skipped_xml_without_image": skipped_xml_without_image,
    }


def sample_voc_frames(source_root: Path, output_root: Path, keep_every: int, offset: int = 0) -> dict[str, int]:
    source_root = Path(source_root)
    output_root = Path(output_root)
    src_images = source_root / "images"
    src_annotations = source_root / "annotations"
    dst_images = output_root / "images"
    dst_annotations = output_root / "annotations"
    dst_images.mkdir(parents=True, exist_ok=True)
    dst_annotations.mkdir(parents=True, exist_ok=True)

    frames_seen = 0
    frames_kept = 0
    for index, image_path in enumerate(sorted(src_images.iterdir())):
        if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
            continue
        frames_seen += 1
        if (index - offset) % keep_every != 0:
            continue
        xml_path = src_annotations / f"{image_path.stem}.xml"
        if not xml_path.exists():
            continue
        shutil.copy2(image_path, dst_images / image_path.name)
        shutil.copy2(xml_path, dst_annotations / xml_path.name)
        frames_kept += 1
    return {"frames_seen": frames_seen, "frames_kept": frames_kept}


def build_training_lists_and_yaml(dataset_root: Path, data_yaml: Path, val_keys: list[str]) -> dict[str, int]:
    dataset_root = Path(dataset_root)
    images_dir = dataset_root / "images"
    labels_dir = dataset_root / "labels"
    names = _load_names(data_yaml)
    train_lines = []
    val_lines = []

    for image_path in sorted(images_dir.iterdir()):
        if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
            continue
        if not (labels_dir / f"{image_path.stem}.txt").exists():
            continue
        rel = image_path.relative_to(dataset_root).as_posix()
        target = val_lines if any(image_path.stem.startswith(key) for key in val_keys) else train_lines
        target.append(rel)

    (dataset_root / "train.txt").write_text("\n".join(train_lines) + ("\n" if train_lines else ""), encoding="utf-8")
    (dataset_root / "val.txt").write_text("\n".join(val_lines) + ("\n" if val_lines else ""), encoding="utf-8")
    yaml_payload = {
        "path": dataset_root.as_posix(),
        "train": "train.txt",
        "val": "val.txt",
        "nc": len(names),
        "names": names,
    }
    (dataset_root / "data.yaml").write_text(yaml.safe_dump(yaml_payload, sort_keys=False), encoding="utf-8")
    return {"train_images": len(train_lines), "val_images": len(val_lines)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="VOC/XML dataset helpers for YOLO training.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    yolo_parser = subparsers.add_parser("voc-xml-to-yolo")
    yolo_parser.add_argument("--dataset-root", type=Path, required=True)
    yolo_parser.add_argument("--data-yaml", type=Path, required=True)
    yolo_parser.add_argument("--backup", action="store_true")
    yolo_parser.add_argument("--backup-suffix", default=".xmlbak")
    return parser
