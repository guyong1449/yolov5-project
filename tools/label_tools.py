import argparse
import re
import shutil
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

import yaml


def _format_float(value: float) -> str:
    return f"{value:.6f}".rstrip("0").rstrip(".")


def _load_names(data_yaml: Path) -> dict[str, int]:
    data = yaml.safe_load(data_yaml.read_text(encoding="utf-8")) or {}
    names = data.get("names")
    if isinstance(names, dict):
        return {str(name): int(idx) for idx, name in names.items()}
    if isinstance(names, list):
        return {str(name): idx for idx, name in enumerate(names)}
    raise ValueError(f"'names' not found or unsupported in {data_yaml}")


def _load_names_list_and_nc(data_yaml: Path) -> tuple[list[str], int]:
    data = yaml.safe_load(data_yaml.read_text(encoding="utf-8")) or {}
    names = data.get("names")
    if isinstance(names, dict):
        ordered = [str(name) for _, name in sorted(((int(idx), name) for idx, name in names.items()), key=lambda item: item[0])]
    elif isinstance(names, list):
        ordered = [str(name) for name in names]
    else:
        raise ValueError(f"'names' not found or unsupported in {data_yaml}")
    nc = int(data.get("nc", len(ordered)))
    if nc != len(ordered):
        raise ValueError(f"'nc' does not match names count in {data_yaml}: nc={nc}, names={len(ordered)}")
    return ordered, nc


def _split_video_key_and_frame(stem: str) -> tuple[str, int]:
    match = re.match(r"^(?P<video_key>.+)_frame(?P<frame_id>\d+)$", stem)
    if match is None:
        return stem, 0
    return match.group("video_key"), int(match.group("frame_id"))


def trim_label_conf_columns(labels_dir, backup_suffix=".bak"):
    labels_dir = Path(labels_dir)
    stats = {
        "files_seen": 0,
        "files_changed": 0,
        "lines_seen": 0,
        "lines_trimmed": 0,
        "backup_files_created": 0,
    }

    for txt_path in sorted(labels_dir.rglob("*.txt")):
        stats["files_seen"] += 1
        original = txt_path.read_text(encoding="utf-8").splitlines()
        updated = []
        changed = False

        for line in original:
            stripped = line.strip()
            if not stripped:
                updated.append("")
                continue

            stats["lines_seen"] += 1
            parts = stripped.split()
            if len(parts) == 6:
                updated.append(" ".join(parts[:5]))
                stats["lines_trimmed"] += 1
                changed = True
            else:
                updated.append(" ".join(parts))

        if changed:
            backup_path = txt_path.with_name(txt_path.name + backup_suffix)
            if not backup_path.exists():
                shutil.copy2(txt_path, backup_path)
                stats["backup_files_created"] += 1
            txt_path.write_text("\n".join(updated) + "\n", encoding="utf-8")
            stats["files_changed"] += 1

    return stats


def _write_fiftyone_voc_xml(
    image_path: Path,
    image_width: int,
    image_height: int,
    detections: list[tuple[int, int, int, int, str]],
    folder_name: str = "data",
    depth: int = 3,
) -> ET.ElementTree:
    """Build a PASCAL VOC XML tree that matches FiftyOne's VOCDetectionDataset schema."""
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
    """Parse object boxes from an existing VOC XML root."""
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
    """Keep only classes listed in ``data.yaml`` and drop all other VOC objects.

    Args:
        dataset_root: Dataset root containing ``images/`` and annotation XML files.
        data_yaml: Dataset YAML whose ``names`` define the allowed class set.
        annotations_subdir: Directory holding VOC XML files to clean in place.
        backup: When True, copy each modified XML to ``<name><backup_suffix>`` first.
        backup_suffix: Suffix appended to XML filenames for backups.
        remove_empty: Delete XML files that have zero objects after filtering.
        dry_run: Compute statistics without writing changes.

    Returns:
        Statistics dictionary with cleaning counts.
    """
    dataset_root = Path(dataset_root)
    data_yaml = Path(data_yaml)
    allowed_classes = set(_load_names(data_yaml).keys())
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
    """Normalize VOC XML files for FiftyOne VOCDetectionDataset import.

    Args:
        dataset_root: Dataset root containing `images/` and source XML annotations.
        data_yaml: Dataset YAML with class `names` used for validation.
        source_subdir: Directory name holding source VOC XML files.
        output_subdir: Output directory name for normalized XML files.
        layout: ``labels_only`` writes only XML labels; ``fiftyone_voc`` also
            creates ``fiftyone_voc/data`` hardlinks and ``fiftyone_voc/labels``.
        overwrite: Replace existing converted XML files when True.
        validate_classes: When True, reject XML class names outside ``data_yaml``.

    Returns:
        Statistics dictionary with conversion counts.
    """
    dataset_root = Path(dataset_root)
    data_yaml = Path(data_yaml)
    class_names = _load_names(data_yaml) if validate_classes else None
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
    expected_classes = set(_load_names(data_yaml).keys())

    if layout == "fiftyone_voc":
        fiftyone_root = dataset_root / "fiftyone_voc"
        data_dir = fiftyone_root / "data"
        labels_dir = fiftyone_root / "labels"
        data_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)
        stats["fiftyone_root"] = str(fiftyone_root)
        stats["data_dir"] = str(data_dir)
        stats["labels_dir"] = str(labels_dir)
    else:
        labels_dir.mkdir(parents=True, exist_ok=True)

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


def convert_voc_xml_dir_to_yolo(dataset_root, data_yaml, backup=False, backup_suffix=".xmlbak"):
    dataset_root = Path(dataset_root)
    data_yaml = Path(data_yaml)
    annotations_dir = dataset_root / "annotations"
    labels_dir = dataset_root / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)

    name_to_id = _load_names(data_yaml)
    stats = {
        "xml_files_seen": 0,
        "labels_written": 0,
        "objects_written": 0,
        "backup_files_created": 0,
    }

    for xml_path in sorted(annotations_dir.glob("*.xml")):
        stats["xml_files_seen"] += 1
        tree = ET.parse(xml_path)
        root = tree.getroot()

        size = root.find("size")
        if size is None:
            raise ValueError(f"Missing <size> in {xml_path}")
        width = float(size.findtext("width", default="0"))
        height = float(size.findtext("height", default="0"))
        if width <= 0 or height <= 0:
            raise ValueError(f"Invalid image size in {xml_path}: width={width}, height={height}")

        filename = root.findtext("filename", default=f"{xml_path.stem}.jpg")
        label_path = labels_dir / f"{Path(filename).stem}.txt"
        yolo_lines = []

        for obj in root.findall("object"):
            class_name = obj.findtext("name")
            if class_name not in name_to_id:
                raise KeyError(f"Class '{class_name}' from {xml_path} not found in {data_yaml}")
            bndbox = obj.find("bndbox")
            if bndbox is None:
                raise ValueError(f"Missing <bndbox> in {xml_path}")

            xmin = float(bndbox.findtext("xmin", default="0"))
            ymin = float(bndbox.findtext("ymin", default="0"))
            xmax = float(bndbox.findtext("xmax", default="0"))
            ymax = float(bndbox.findtext("ymax", default="0"))
            if xmax < xmin or ymax < ymin:
                raise ValueError(f"Invalid box in {xml_path}: {(xmin, ymin, xmax, ymax)}")

            x_center = ((xmin + xmax) / 2.0) / width
            y_center = ((ymin + ymax) / 2.0) / height
            box_width = (xmax - xmin) / width
            box_height = (ymax - ymin) / height

            yolo_lines.append(
                " ".join(
                    [
                        str(name_to_id[class_name]),
                        _format_float(x_center),
                        _format_float(y_center),
                        _format_float(box_width),
                        _format_float(box_height),
                    ]
                )
            )
            stats["objects_written"] += 1

        if backup and label_path.exists():
            backup_path = label_path.with_name(label_path.name + backup_suffix)
            if not backup_path.exists():
                shutil.copy2(label_path, backup_path)
                stats["backup_files_created"] += 1

        label_path.write_text(("\n".join(yolo_lines) + "\n") if yolo_lines else "", encoding="utf-8")
        stats["labels_written"] += 1

    return stats


def sample_voc_frames(dataset_root, output_root, keep_every=3, offset=0):
    """Copy one frame out of each fixed-size window into a new VOC-style dataset root.

    Args:
        dataset_root: Source VOC-style root containing `images/` and `annotations/`.
        output_root: Destination root to create.
        keep_every: Keep one frame for every N ordered frames per video key.
        offset: Zero-based index within each N-frame window to keep.

    Returns:
        A stats dictionary describing the sampled result.

    Raises:
        FileNotFoundError: If source images or annotations directory is missing.
        ValueError: If keep_every or offset are invalid.
    """
    dataset_root = Path(dataset_root)
    output_root = Path(output_root)
    source_images = dataset_root / "images"
    source_annotations = dataset_root / "annotations"
    target_images = output_root / "images"
    target_annotations = output_root / "annotations"

    if not source_images.is_dir():
        raise FileNotFoundError(f"Images directory not found: {source_images}")
    if not source_annotations.is_dir():
        raise FileNotFoundError(f"Annotations directory not found: {source_annotations}")
    if keep_every <= 0:
        raise ValueError(f"keep_every must be >= 1, got {keep_every}")
    if offset < 0 or offset >= keep_every:
        raise ValueError(f"offset must satisfy 0 <= offset < keep_every, got offset={offset}, keep_every={keep_every}")

    target_images.mkdir(parents=True, exist_ok=True)
    target_annotations.mkdir(parents=True, exist_ok=True)

    grouped_stems = defaultdict(list)
    for xml_path in sorted(source_annotations.glob("*.xml")):
        image_path = source_images / f"{xml_path.stem}.jpg"
        if not image_path.is_file():
            continue
        video_key, frame_id = _split_video_key_and_frame(xml_path.stem)
        grouped_stems[video_key].append((frame_id, xml_path.stem))

    stats = {
        "groups_seen": len(grouped_stems),
        "frames_seen": 0,
        "frames_kept": 0,
        "images_copied": 0,
        "annotations_copied": 0,
        "frames_missing_image": 0,
    }

    for video_key in sorted(grouped_stems):
        frames = sorted(grouped_stems[video_key], key=lambda item: (item[0], item[1]))
        for index, (_, stem) in enumerate(frames):
            stats["frames_seen"] += 1
            if index % keep_every != offset:
                continue
            image_path = source_images / f"{stem}.jpg"
            xml_path = source_annotations / f"{stem}.xml"
            if not image_path.is_file():
                stats["frames_missing_image"] += 1
                continue
            shutil.copy2(image_path, target_images / image_path.name)
            shutil.copy2(xml_path, target_annotations / xml_path.name)
            stats["frames_kept"] += 1
            stats["images_copied"] += 1
            stats["annotations_copied"] += 1

    return stats


def build_training_lists_and_yaml(dataset_root, data_yaml, val_keys, train_txt_name="train.txt", val_txt_name="val.txt"):
    """Build train/val image lists and a dataset yaml from a trainable dataset root."""
    dataset_root = Path(dataset_root)
    data_yaml = Path(data_yaml)
    images_dir = dataset_root / "images"
    labels_dir = dataset_root / "labels"
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    if not labels_dir.is_dir():
        raise FileNotFoundError(f"Labels directory not found: {labels_dir}")

    val_key_set = {str(key) for key in val_keys}
    if not val_key_set:
        raise ValueError("val_keys must not be empty")

    train_lines = []
    val_lines = []
    skipped_missing_label = 0

    for image_path in sorted(images_dir.glob("*.jpg")):
        label_path = labels_dir / f"{image_path.stem}.txt"
        if not label_path.is_file():
            skipped_missing_label += 1
            continue
        video_key, _ = _split_video_key_and_frame(image_path.stem)
        line = image_path.resolve().as_posix()
        if video_key in val_key_set:
            val_lines.append(line)
        else:
            train_lines.append(line)

    if not train_lines:
        raise ValueError("No training images were assigned; check val_keys and labels directory.")
    if not val_lines:
        raise ValueError("No validation images were assigned; check val_keys and labels directory.")

    (dataset_root / train_txt_name).write_text("\n".join(train_lines) + "\n", encoding="utf-8")
    (dataset_root / val_txt_name).write_text("\n".join(val_lines) + "\n", encoding="utf-8")

    names, nc = _load_names_list_and_nc(data_yaml)
    dataset_yaml = {
        "path": dataset_root.resolve().as_posix(),
        "train": train_txt_name,
        "val": val_txt_name,
        "nc": nc,
        "names": names,
    }
    (dataset_root / "data.yaml").write_text(yaml.safe_dump(dataset_yaml, sort_keys=False, allow_unicode=True), encoding="utf-8")

    return {
        "train_images": len(train_lines),
        "val_images": len(val_lines),
        "skipped_missing_label": skipped_missing_label,
        "val_keys": ",".join(sorted(val_key_set)),
    }


def build_trainable_voc_dataset(dataset_root, output_root, data_yaml, keep_every=3, offset=0, val_keys=None):
    """Sample a VOC dataset, rebuild YOLO labels, and write train/val manifests."""
    if val_keys is None:
        val_keys = []
    sample_stats = sample_voc_frames(dataset_root, output_root, keep_every=keep_every, offset=offset)
    convert_stats = convert_voc_xml_dir_to_yolo(output_root, data_yaml, backup=False)
    split_stats = build_training_lists_and_yaml(output_root, data_yaml, val_keys=val_keys)
    merged = {}
    merged.update({f"sample_{key}": value for key, value in sample_stats.items()})
    merged.update({f"convert_{key}": value for key, value in convert_stats.items()})
    merged.update({f"split_{key}": value for key, value in split_stats.items()})
    return merged


def build_parser():
    parser = argparse.ArgumentParser(description="Offline label utilities for this YOLOv5 workspace.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    trim_parser = subparsers.add_parser("trim-conf", help="Trim 6-column YOLO txt labels to standard 5 columns.")
    trim_parser.add_argument("--labels-dir", required=True, type=Path)
    trim_parser.add_argument("--backup-suffix", default=".bak")

    xml_parser = subparsers.add_parser("voc-xml-to-yolo", help="Convert VOC XML annotations into YOLO txt labels.")
    xml_parser.add_argument("--dataset-root", required=True, type=Path)
    xml_parser.add_argument("--data-yaml", required=True, type=Path)
    xml_parser.add_argument("--backup", action="store_true")
    xml_parser.add_argument("--backup-suffix", default=".xmlbak")

    fiftyone_parser = subparsers.add_parser(
        "voc-xml-to-fiftyone",
        help="Normalize VOC XML annotations for FiftyOne VOCDetectionDataset import.",
    )
    fiftyone_parser.add_argument("--dataset-root", required=True, type=Path)
    fiftyone_parser.add_argument("--data-yaml", required=True, type=Path)
    fiftyone_parser.add_argument("--source-subdir", default="annotations")
    fiftyone_parser.add_argument("--output-subdir", default="fiftyone_labels")
    fiftyone_parser.add_argument(
        "--layout",
        choices=("labels_only", "fiftyone_voc"),
        default="fiftyone_voc",
        help="labels_only writes <output-subdir>/; fiftyone_voc also creates fiftyone_voc/data+labels.",
    )
    fiftyone_parser.add_argument("--no-overwrite", action="store_true")
    fiftyone_parser.add_argument(
        "--validate-classes",
        action="store_true",
        help="Reject XML classes that are not listed in data.yaml.",
    )

    clean_parser = subparsers.add_parser(
        "clean-voc-classes",
        help="Keep only classes from data.yaml and drop other VOC XML objects.",
    )
    clean_parser.add_argument("--dataset-root", required=True, type=Path)
    clean_parser.add_argument("--data-yaml", required=True, type=Path)
    clean_parser.add_argument("--annotations-subdir", default="annotations")
    clean_parser.add_argument("--backup", action="store_true")
    clean_parser.add_argument("--backup-suffix", default=".xmlbak")
    clean_parser.add_argument(
        "--remove-empty",
        action="store_true",
        help="Delete XML files that contain zero objects after class filtering.",
    )
    clean_parser.add_argument("--dry-run", action="store_true")

    sample_parser = subparsers.add_parser(
        "sample-voc-frames",
        help="Copy one out of every N video frames into a new VOC-style dataset root.",
    )
    sample_parser.add_argument("--dataset-root", required=True, type=Path)
    sample_parser.add_argument("--output-root", required=True, type=Path)
    sample_parser.add_argument("--keep-every", type=int, default=3)
    sample_parser.add_argument("--offset", type=int, default=0)

    build_parser_obj = subparsers.add_parser(
        "build-trainable-voc",
        help="Sample VOC frames, rebuild YOLO labels, and write train.txt/val.txt/data.yaml.",
    )
    build_parser_obj.add_argument("--dataset-root", required=True, type=Path)
    build_parser_obj.add_argument("--output-root", required=True, type=Path)
    build_parser_obj.add_argument("--data-yaml", required=True, type=Path)
    build_parser_obj.add_argument("--keep-every", type=int, default=3)
    build_parser_obj.add_argument("--offset", type=int, default=0)
    build_parser_obj.add_argument("--val-keys", nargs="+", required=True)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "trim-conf":
        stats = trim_label_conf_columns(args.labels_dir, backup_suffix=args.backup_suffix)
    elif args.command == "voc-xml-to-yolo":
        stats = convert_voc_xml_dir_to_yolo(
            args.dataset_root,
            args.data_yaml,
            backup=args.backup,
            backup_suffix=args.backup_suffix,
        )
    elif args.command == "voc-xml-to-fiftyone":
        stats = convert_voc_xml_dir_to_fiftyone(
            args.dataset_root,
            args.data_yaml,
            source_subdir=args.source_subdir,
            output_subdir=args.output_subdir,
            layout=args.layout,
            overwrite=not args.no_overwrite,
            validate_classes=args.validate_classes,
        )
    elif args.command == "clean-voc-classes":
        stats = clean_voc_xml_dir_classes(
            args.dataset_root,
            args.data_yaml,
            annotations_subdir=args.annotations_subdir,
            backup=args.backup,
            backup_suffix=args.backup_suffix,
            remove_empty=args.remove_empty,
            dry_run=args.dry_run,
        )
    elif args.command == "sample-voc-frames":
        stats = sample_voc_frames(
            args.dataset_root,
            args.output_root,
            keep_every=args.keep_every,
            offset=args.offset,
        )
    else:
        stats = build_trainable_voc_dataset(
            args.dataset_root,
            args.output_root,
            args.data_yaml,
            keep_every=args.keep_every,
            offset=args.offset,
            val_keys=args.val_keys,
        )

    for key, value in stats.items():
        print(f"{key}={value}")


if __name__ == "__main__":
    main()
