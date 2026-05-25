import csv
import tempfile
import unittest
from pathlib import Path


PNG_1X1 = (
    b"\x89PNG\r\n\x1a\n"
    b"\x00\x00\x00\rIHDR"
    b"\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00"
    b"\x90wS\xde"
    b"\x00\x00\x00\x0cIDATx\x9cc\xf8\xcf\xc0\x00\x00\x03\x01\x01\x00"
    b"\xc9\xfe\x92\xef"
    b"\x00\x00\x00\x00IEND\xaeB`\x82"
)


class _FakeDataset:
    def __init__(self):
        self.calls = []

    def match_tags(self, tags, bool=None, all=False):
        self.calls.append((tags, bool, all))
        return "filtered-view"


class TestFiftyOneTools(unittest.TestCase):
    def test_validate_voc_layout_accepts_existing_dirs(self):
        from tools.fiftyone_import_voc import validate_voc_layout

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_dir = root / "data"
            labels_dir = root / "labels"
            data_dir.mkdir()
            labels_dir.mkdir()
            (data_dir / "frame001.jpg").write_bytes(b"fake")
            (labels_dir / "frame001.xml").write_text("<annotation />", encoding="utf-8")

            resolved_data, resolved_labels = validate_voc_layout(data_dir, labels_dir)

            self.assertEqual(resolved_data, data_dir.resolve())
            self.assertEqual(resolved_labels, labels_dir.resolve())

    def test_validate_voc_layout_rejects_missing_dir(self):
        from tools.fiftyone_import_voc import validate_voc_layout

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_dir = root / "data"
            labels_dir = root / "labels"
            data_dir.mkdir()

            with self.assertRaises(FileNotFoundError):
                validate_voc_layout(data_dir, labels_dir)

    def test_scan_voc_dataset_reports_unknown_class_and_invalid_bbox(self):
        from tools.fiftyone_voc_precheck import scan_voc_dataset

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_dir = root / "data"
            labels_dir = root / "labels"
            data_dir.mkdir()
            labels_dir.mkdir()

            image_path = data_dir / "frame001.png"
            image_path.write_bytes(PNG_1X1)
            (labels_dir / "frame001.xml").write_text(
                (
                    "<annotation>"
                    "<filename>frame001.png</filename>"
                    "<size><width>2</width><height>3</height><depth>3</depth></size>"
                    "<object><name>alien</name><bndbox>"
                    "<xmin>4</xmin><ymin>1</ymin><xmax>2</xmax><ymax>1</ymax>"
                    "</bndbox></object>"
                    "</annotation>"
                ),
                encoding="utf-8",
            )

            rows = scan_voc_dataset(
                data_dir,
                labels_dir,
                allowed_classes={"bird"},
            )
            issue_types = {row["issue_type"] for row in rows}

            self.assertIn("image_size_mismatch", issue_types)
            self.assertIn("unknown_class", issue_types)
            self.assertIn("invalid_bbox", issue_types)

    def test_write_report_csv_writes_header_for_empty_rows(self):
        from tools.fiftyone_voc_precheck import REPORT_COLUMNS, write_report_csv

        with tempfile.TemporaryDirectory() as tmp:
            out_csv = Path(tmp) / "report.csv"
            write_report_csv([], out_csv)

            with out_csv.open("r", encoding="utf-8-sig", newline="") as fh:
                reader = csv.reader(fh)
                header = next(reader)
                remaining = list(reader)

            self.assertEqual(header, list(REPORT_COLUMNS))
            self.assertEqual(remaining, [])

    def test_build_export_view_uses_match_tags_to_exclude_samples(self):
        from tools.fiftyone_export_cleaned import build_export_view

        dataset = _FakeDataset()

        view = build_export_view(dataset, ["drop", "bad_case"])

        self.assertEqual(view, "filtered-view")
        self.assertEqual(dataset.calls, [(["drop", "bad_case"], False, False)])


if __name__ == "__main__":
    unittest.main()
