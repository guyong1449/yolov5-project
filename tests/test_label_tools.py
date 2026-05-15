import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.label_tools import (  # noqa: E402
    build_training_lists_and_yaml,
    convert_voc_xml_dir_to_yolo,
    sample_voc_frames,
    trim_label_conf_columns,
)


class TestLabelTools(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_trim_label_conf_columns_creates_backup_and_keeps_first_five_columns(self):
        labels_dir = self.tmp / "labels"
        labels_dir.mkdir()
        src = labels_dir / "frame001.txt"
        src.write_text(
            "1 0.5 0.4 0.2 0.1 0.91\n"
            "2 0.3 0.2 0.1 0.1 0.80\n"
            "0 0.1 0.1 0.1 0.1\n",
            encoding="utf-8",
        )

        stats = trim_label_conf_columns(labels_dir, backup_suffix=".bak")

        self.assertEqual(stats["files_changed"], 1)
        self.assertEqual(stats["lines_trimmed"], 2)
        self.assertTrue((labels_dir / "frame001.txt.bak").is_file())
        self.assertEqual(
            src.read_text(encoding="utf-8"),
            "1 0.5 0.4 0.2 0.1\n"
            "2 0.3 0.2 0.1 0.1\n"
            "0 0.1 0.1 0.1 0.1\n",
        )

    def test_convert_voc_xml_dir_to_yolo_writes_label_next_to_matching_image_name(self):
        dataset_root = self.tmp / "voc"
        images_dir = dataset_root / "images"
        annotations_dir = dataset_root / "annotations"
        labels_dir = dataset_root / "labels"
        images_dir.mkdir(parents=True)
        annotations_dir.mkdir()
        images_dir.joinpath("sample.jpg").write_bytes(b"jpg")
        annotations_dir.joinpath("sample.xml").write_text(
            """<?xml version='1.0' encoding='utf-8'?>
<annotation>
  <filename>sample.jpg</filename>
  <size><width>100</width><height>200</height><depth>3</depth></size>
  <object>
    <name>drone</name>
    <bndbox><xmin>10</xmin><ymin>20</ymin><xmax>30</xmax><ymax>60</ymax></bndbox>
  </object>
</annotation>
""",
            encoding="utf-8",
        )
        data_yaml = self.tmp / "data.yaml"
        data_yaml.write_text(yaml.safe_dump({"names": ["bird", "drone"]}, sort_keys=False), encoding="utf-8")

        stats = convert_voc_xml_dir_to_yolo(dataset_root, data_yaml, backup=False)

        self.assertEqual(stats["xml_files_seen"], 1)
        self.assertEqual(stats["labels_written"], 1)
        self.assertTrue(labels_dir.joinpath("sample.txt").is_file())
        self.assertEqual(
            labels_dir.joinpath("sample.txt").read_text(encoding="utf-8").strip(),
            "1 0.2 0.2 0.2 0.2",
        )

    def test_sample_voc_frames_keeps_every_third_frame_and_pairs_xml(self):
        source_root = self.tmp / "source"
        source_images = source_root / "images"
        source_annotations = source_root / "annotations"
        source_images.mkdir(parents=True)
        source_annotations.mkdir()

        for frame_id in [1, 2, 3, 4, 5, 6]:
            stem = f"video12_mp4_frame{frame_id:06d}"
            source_images.joinpath(f"{stem}.jpg").write_bytes(b"jpg")
            source_annotations.joinpath(f"{stem}.xml").write_text(
                f"""<?xml version='1.0' encoding='utf-8'?>
<annotation>
  <filename>{stem}.jpg</filename>
  <size><width>100</width><height>100</height><depth>3</depth></size>
</annotation>
""",
                encoding="utf-8",
            )

        output_root = self.tmp / "sampled"
        stats = sample_voc_frames(source_root, output_root, keep_every=3, offset=0)

        kept_images = sorted(p.name for p in (output_root / "images").glob("*.jpg"))
        kept_xml = sorted(p.name for p in (output_root / "annotations").glob("*.xml"))

        self.assertEqual(stats["frames_seen"], 6)
        self.assertEqual(stats["frames_kept"], 2)
        self.assertEqual(
            kept_images,
            ["video12_mp4_frame000001.jpg", "video12_mp4_frame000004.jpg"],
        )
        self.assertEqual(
            kept_xml,
            ["video12_mp4_frame000001.xml", "video12_mp4_frame000004.xml"],
        )

    def test_build_training_lists_and_yaml_splits_by_video_key(self):
        dataset_root = self.tmp / "trainable"
        images_dir = dataset_root / "images"
        labels_dir = dataset_root / "labels"
        images_dir.mkdir(parents=True)
        labels_dir.mkdir()
        for stem in [
            "video12_mp4_frame000001",
            "video12_mp4_frame000004",
            "video13_mp4_frame000001",
        ]:
            images_dir.joinpath(f"{stem}.jpg").write_bytes(b"jpg")
            labels_dir.joinpath(f"{stem}.txt").write_text("2 0.5 0.5 0.1 0.1\n", encoding="utf-8")

        source_yaml = self.tmp / "data.yaml"
        source_yaml.write_text(
            yaml.safe_dump({"nc": 2, "names": ["bird", "drone"]}, sort_keys=False),
            encoding="utf-8",
        )

        stats = build_training_lists_and_yaml(
            dataset_root=dataset_root,
            data_yaml=source_yaml,
            val_keys=["video13_mp4"],
        )

        train_lines = (dataset_root / "train.txt").read_text(encoding="utf-8").splitlines()
        val_lines = (dataset_root / "val.txt").read_text(encoding="utf-8").splitlines()
        dataset_yaml = yaml.safe_load((dataset_root / "data.yaml").read_text(encoding="utf-8"))

        self.assertEqual(stats["train_images"], 2)
        self.assertEqual(stats["val_images"], 1)
        self.assertEqual(len(train_lines), 2)
        self.assertEqual(len(val_lines), 1)
        self.assertTrue(all(line.endswith(".jpg") for line in train_lines + val_lines))
        self.assertEqual(dataset_yaml["path"], dataset_root.as_posix())
        self.assertEqual(dataset_yaml["train"], "train.txt")
        self.assertEqual(dataset_yaml["val"], "val.txt")
        self.assertEqual(dataset_yaml["names"], ["bird", "drone"])


if __name__ == "__main__":
    unittest.main()
