import argparse
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import yaml


def _fmt_float(value: float) -> str:
    return f"{value:.6f}".rstrip("0").rstrip(".")


_format_float = _fmt_float


def _load_names(data_yaml: Path) -> list[str]:
    data = yaml.safe_load(Path(data_yaml).read_text(encoding="utf-8")) or {}
    names = data.get("names") or []
    if isinstance(names, dict):
        return [str(name) for _, name in sorted(((int(idx), name) for idx, name in names.items()), key=lambda item: item[0])]
    if isinstance(names, list) and names:
        return [str(name) for name in names]
    raise ValueError(f"names missing in {data_yaml}")


def _load_names_map(data_yaml: Path) -> dict[str, int]:
    data = yaml.safe_load(Path(data_yaml).read_text(encoding="utf-8")) or {}
    names = data.get("names") or []
    if isinstance(names, dict):
        return {str(name): int(idx) for name, idx in names.items()}
    if isinstance(names, list) and names:
        return {str(name): idx for idx, name in enumerate(names)}
    raise ValueError(f"names missing in {data_yaml}")


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


def _write_fiftyone_voc_xml(
    image_path: Path,
    image_width: int,
    image_height: int,
    detections: list[tuple[int, int, int, int, str]],
    folder_name: str = "data",
    depth: int = 3,
) -> ET.ElementTree:
    """Build a PASCAL VOC XML tree compatible with FiftyOne import."""
    annotation = ET.Element("annotation")

    folder = ET.SubElement(annotation, "folder")
    folder.text = folder_name

    filename = ET.SubElement(annotation, "filename")
    filename.text = image_path.name

    path = ET.SubElement(annotation, "path")
    path.text = str(image_path.resolve())

    source = ET.SubElement(annotation, "source")
    database = ET.SubElement(source, "database")
    database.text = "Unknown"

    size = ET.SubElement(annotation, "size")
    ET.SubElement(size, "width").text = str(int(image_width))
    ET.SubElement(size, "height").text = str(int(image_height))
    ET.SubElement(size, "depth").text = str(int(depth))

    segmented = ET.SubElement(annotation, "segmented")
    segmented.text = "0"

    for xmin, ymin, xmax, ymax, class_name in detections:
        obj = ET.SubElement(annotation, "object")
        ET.SubElement(obj, "name").text = class_name
        ET.SubElement(obj, "pose").text = "Unspecified"
        ET.SubElement(obj, "truncated").text = "0"
        ET.SubElement(obj, "difficult").text = "0"
        ET.SubElement(obj, "occluded").text = "0"
        bndbox = ET.SubElement(obj, "bndbox")
        ET.SubElement(bndbox, "xmin").text = str(int(xmin))
        ET.SubElement(bndbox, "ymin").text = str(int(ymin))
        ET.SubElement(bndbox, "xmax").text = str(int(xmax))
        ET.SubElement(bndbox, "ymax").text = str(int(ymax))

    tree = ET.ElementTree(annotation)
    if hasattr(ET, "indent"):
        ET.indent(tree, space="  ")
    return tree


def _parse_voc_detections(
    root: ET.Element,
    xml_path: Path,
    class_names: dict[str, int] | None = None,
    *,
    validate_classes: bool = True,
) -> list[tuple[int, int, int, int, str]]:
    size = root.find("size")
    if size is None:
        raise ValueError(f"Missing <size> in {xml_path}")

    detections = []
    for obj in root.findall("object"):
        class_name = obj.findtext("name")
        if not class_name:
            raise ValueError(f"Missing <name> in {xml_path}")
        if validate_classes and class_names is not None and class_name not in class_names:
            raise KeyError(f"Class '{class_name}' from {xml_path} not found in dataset names")

        bndbox = obj.find("bndbox")
        if bndbox is None:
            raise ValueError(f"Missing <bndbox> in {xml_path}")
        xmin = int(float(bndbox.findtext("xmin", default="0")))
        ymin = int(float(bndbox.findtext("ymin", default="0")))
        xmax = int(float(bndbox.findtext("xmax", default="0")))
        ymax = int(float(bndbox.findtext("ymax", default="0")))
        if xmax <= xmin or ymax <= ymin:
            raise ValueError(f"Invalid box in {xml_path}: {(xmin, ymin, xmax, ymax)}")
        detections.append((xmin, ymin, xmax, ymax, class_name))
    return detections


def clean_voc_xml_dir_classes(
    dataset_root,
    data_yaml,
    annotations_subdir="annotations",
    backup=False,
    backup_suffix=".xmlbak",
    remove_empty=False,
    dry_run=False,
):
    """Keep only classes listed in data.yaml and drop other VOC objects."""
    dataset_root = Path(dataset_root)
    data_yaml = Path(data_yaml)
    allowed_classes = set(_load_names_map(data_yaml).keys())
    annotations_dir = dataset_root / annotations_subdir
    images_dir = dataset_root / "images"

    stats = {
        "xml_files_seen": 0,
        "xml_files_changed": 0,
        "xml_files_unchanged": 0,
        "xml_files_removed": 0,
        "objects_before": 0,
        "objects_after": 0,
        "objects_dropped": 0,
        "dropped_classes": set(),
        "allowed_classes": sorted(allowed_classes),
        "dry_run": dry_run,
    }
    dropped_classes: set[str] = set()

    for xml_path in sorted(annotations_dir.glob("*.xml")):
        stats["xml_files_seen"] += 1
        tree = ET.parse(xml_path)
        root = tree.getroot()

        size = root.find("size")
        if size is None:
            raise ValueError(f"Missing <size> in {xml_path}")
        width = int(float(size.findtext("width", default="0")))
        height = int(float(size.findtext("height", default="0")))
        depth = int(float(size.findtext("depth", default="3")))
        if width <= 0 or height <= 0:
            raise ValueError(f"Invalid image size in {xml_path}: width={width}, height={height}")

        filename = root.findtext("filename", default=f"{xml_path.stem}.jpg")
        image_path = images_dir / Path(filename).name
        if not image_path.is_file():
            raise FileNotFoundError(f"Image not found for {xml_path}: {image_path}")

        detections = _parse_voc_detections(root, xml_path, allowed_classes, validate_classes=False)
        stats["objects_before"] += len(detections)
        kept = [box for box in detections if box[4] in allowed_classes]
        for *_, class_name in detections:
            if class_name not in allowed_classes:
                dropped_classes.add(class_name)
        stats["objects_after"] += len(kept)
        stats["objects_dropped"] += len(detections) - len(kept)

        if len(kept) == len(detections):
            stats["xml_files_unchanged"] += 1
            continue

        stats["xml_files_changed"] += 1
        if dry_run:
            continue

        if len(kept) == 0 and remove_empty:
            if backup:
                backup_path = xml_path.with_name(xml_path.name + backup_suffix)
                if not backup_path.exists():
                    shutil.copy2(xml_path, backup_path)
            xml_path.unlink()
            stats["xml_files_removed"] += 1
            continue

        if backup:
            backup_path = xml_path.with_name(xml_path.name + backup_suffix)
            if not backup_path.exists():
                shutil.copy2(xml_path, backup_path)

        cleaned = _write_fiftyone_voc_xml(
            image_path=image_path,
            image_width=width,
            image_height=height,
            detections=kept,
            folder_name=root.findtext("folder", default="images") or "images",
            depth=depth,
        )
        cleaned.write(str(xml_path), encoding="utf-8", xml_declaration=True)

    stats["dropped_classes"] = sorted(dropped_classes)
    return stats


def convert_voc_xml_dir_to_fiftyone(
    dataset_root,
    data_yaml,
    source_subdir="annotations",
    output_subdir="fiftyone_labels",
    layout="labels_only",
    overwrite=True,
    validate_classes=False,
):
    """Normalize VOC XML files for FiftyOne VOCDetectionDataset import."""
    dataset_root = Path(dataset_root)
    data_yaml = Path(data_yaml)
    class_names = _load_names_map(data_yaml) if validate_classes else None
    source_dir = dataset_root / source_subdir
    images_dir = dataset_root / "images"
    labels_dir = dataset_root / output_subdir

    stats = {
        "xml_files_seen": 0,
        "xml_files_written": 0,
        "xml_files_skipped": 0,
        "objects_written": 0,
        "missing_images": 0,
        "layout": layout,
        "labels_dir": str(labels_dir),
        "unknown_classes": [],
    }
    unknown_classes: set[str] = set()
    expected_classes = set(_load_names_map(data_yaml).keys())

    if layout == "fiftyone_voc":
        fiftyone_root = dataset_root / "fiftyone_voc"
        data_dir = fiftyone_root / "data"
        labels_dir = fiftyone_root / "labels"
        data_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)
        stats["fiftyone_root"] = str(fiftyone_root)
        stats["data_dir"] = str(data_dir)
        stats["labels_dir"] = str(labels_dir)
    elif layout == "labels_only":
        labels_dir.mkdir(parents=True, exist_ok=True)
    else:
        raise ValueError(f"Unsupported layout: {layout}")

    for xml_path in sorted(source_dir.glob("*.xml")):
        stats["xml_files_seen"] += 1
        tree = ET.parse(xml_path)
        root = tree.getroot()

        size = root.find("size")
        if size is None:
            raise ValueError(f"Missing <size> in {xml_path}")
        width = int(float(size.findtext("width", default="0")))
        height = int(float(size.findtext("height", default="0")))
        depth = int(float(size.findtext("depth", default="3")))
        if width <= 0 or height <= 0:
            raise ValueError(f"Invalid image size in {xml_path}: width={width}, height={height}")

        filename = root.findtext("filename", default=f"{xml_path.stem}.jpg")
        image_path = images_dir / Path(filename).name
        if not image_path.is_file():
            stats["missing_images"] += 1
            raise FileNotFoundError(f"Image not found for {xml_path}: {image_path}")

        detections = _parse_voc_detections(
            root,
            xml_path,
            class_names,
            validate_classes=validate_classes,
        )
        if not validate_classes:
            for *_, class_name in detections:
                if class_name not in expected_classes:
                    unknown_classes.add(class_name)
        stats["objects_written"] += len(detections)

        out_xml = labels_dir / xml_path.name
        if out_xml.exists() and not overwrite:
            stats["xml_files_skipped"] += 1
            continue

        folder_name = "data" if layout == "fiftyone_voc" else "images"
        normalized = _write_fiftyone_voc_xml(
            image_path=image_path,
            image_width=width,
            image_height=height,
            detections=detections,
            folder_name=folder_name,
            depth=depth,
        )
        normalized.write(str(out_xml), encoding="utf-8", xml_declaration=True)
        stats["xml_files_written"] += 1

        if layout == "fiftyone_voc":
            linked_image = data_dir / image_path.name
            if not linked_image.exists():
                try:
                    linked_image.hardlink_to(image_path)
                except OSError:
                    shutil.copy2(image_path, linked_image)

    stats["unknown_classes"] = sorted(unknown_classes)
    return stats


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
