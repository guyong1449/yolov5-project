"""Tests for FiftyOne-compatible VOC XML normalization."""

import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

from tools.label_tools import (
    _write_fiftyone_voc_xml,
    clean_voc_xml_dir_classes,
    convert_voc_xml_dir_to_fiftyone,
)


class TestVocXmlToFiftyone(unittest.TestCase):
    def test_write_fiftyone_voc_xml_includes_path_and_occluded(self):
        with tempfile.TemporaryDirectory() as tmp:
            image_path = Path(tmp) / "sample.jpg"
            image_path.write_bytes(b"fake")

            tree = _write_fiftyone_voc_xml(
                image_path=image_path,
                image_width=640,
                image_height=480,
                detections=[(10, 20, 30, 40, "bird")],
            )
            root = tree.getroot()
            self.assertEqual(root.findtext("path"), str(image_path.resolve()))
            obj = root.find("object")
            self.assertIsNotNone(obj)
            self.assertEqual(obj.findtext("occluded"), "0")

    def test_convert_voc_xml_dir_to_fiftyone_writes_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "images").mkdir()
            (root / "annotations").mkdir()
            (root / "data.yaml").write_text(
                "nc: 1\nnames:\n  - bird\n",
                encoding="utf-8",
            )

            image_path = root / "images" / "frame000001.jpg"
            image_path.write_bytes(b"fake")
            ann = root / "annotations" / "frame000001.xml"
            ann.write_text(
                (
                    "<?xml version='1.0' encoding='utf-8'?>"
                    "<annotation>"
                    "<filename>frame000001.jpg</filename>"
                    "<size><width>640</width><height>480</height><depth>3</depth></size>"
                    "<object><name>bird</name><bndbox>"
                    "<xmin>10</xmin><ymin>20</ymin><xmax>30</xmax><ymax>40</ymax>"
                    "</bndbox></object>"
                    "</annotation>"
                ),
                encoding="utf-8",
            )

            stats = convert_voc_xml_dir_to_fiftyone(
                root,
                root / "data.yaml",
                layout="labels_only",
                output_subdir="fiftyone_labels",
            )
            out_xml = root / "fiftyone_labels" / "frame000001.xml"
            self.assertTrue(out_xml.is_file())
            self.assertEqual(stats["xml_files_written"], 1)
            parsed = ET.parse(out_xml).getroot()
            self.assertIsNotNone(parsed.find("path"))
            self.assertEqual(parsed.find("object").findtext("occluded"), "0")


    def test_clean_voc_xml_dir_classes_drops_unknown_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "images").mkdir()
            (root / "annotations").mkdir()
            (root / "data.yaml").write_text(
                "nc: 2\nnames:\n  - bird\n  - drone\n",
                encoding="utf-8",
            )
            image_path = root / "images" / "frame000001.jpg"
            image_path.write_bytes(b"fake")
            ann = root / "annotations" / "frame000001.xml"
            ann.write_text(
                (
                    "<?xml version='1.0' encoding='utf-8'?>"
                    "<annotation>"
                    "<filename>frame000001.jpg</filename>"
                    "<size><width>640</width><height>480</height><depth>3</depth></size>"
                    "<object><name>bird</name><bndbox>"
                    "<xmin>10</xmin><ymin>20</ymin><xmax>30</xmax><ymax>40</ymax>"
                    "</bndbox></object>"
                    "<object><name>balloon</name><bndbox>"
                    "<xmin>50</xmin><ymin>60</ymin><xmax>70</xmax><ymax>80</ymax>"
                    "</bndbox></object>"
                    "</annotation>"
                ),
                encoding="utf-8",
            )

            stats = clean_voc_xml_dir_classes(root, root / "data.yaml")
            root_xml = ET.parse(ann).getroot()
            names = [obj.findtext("name") for obj in root_xml.findall("object")]
            self.assertEqual(names, ["bird"])
            self.assertEqual(stats["objects_dropped"], 1)


if __name__ == "__main__":
    unittest.main()
