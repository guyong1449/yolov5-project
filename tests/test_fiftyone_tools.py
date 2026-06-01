import csv
import json
import os
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

import sys
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.env_config import get_dataset_dir  # noqa: E402


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
    def test_start_fiftyone_ps1_exists_and_references_expected_paths(self):
        script_path = REPO_ROOT / "tools" / "fiftyone" / "start_fiftyone_voc.ps1"

        self.assertTrue(script_path.is_file(), f"Missing script: {script_path}")

        contents = script_path.read_text(encoding="utf-8")

        self.assertIn("python", contents)
        self.assertIn("tools\\fiftyone\\fiftyone_import_voc.py", contents)
        expected_data = os.path.join(get_dataset_dir(), "fiftyone_voc", "data")
        expected_labels = os.path.join(get_dataset_dir(), "fiftyone_voc", "labels")
        self.assertIn(expected_data, contents)
        self.assertIn(expected_labels, contents)
        self.assertIn("--wait", contents)

    def test_start_fiftyone_bat_exists_and_invokes_ps1(self):
        script_path = REPO_ROOT / "tools" / "fiftyone" / "start_fiftyone_voc.bat"

        self.assertTrue(script_path.is_file(), f"Missing script: {script_path}")

        contents = script_path.read_text(encoding="utf-8")

        self.assertIn("start_fiftyone_voc.ps1", contents)
        self.assertIn("powershell", contents.lower())

    def test_validate_voc_layout_accepts_existing_dirs(self):
        from tools.fiftyone.fiftyone_import_voc import validate_voc_layout

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
        from tools.fiftyone.fiftyone_import_voc import validate_voc_layout

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_dir = root / "data"
            labels_dir = root / "labels"
            data_dir.mkdir()

            with self.assertRaises(FileNotFoundError):
                validate_voc_layout(data_dir, labels_dir)

    def test_scan_voc_dataset_reports_unknown_class_and_invalid_bbox(self):
        from tools.fiftyone.fiftyone_voc_precheck import scan_voc_dataset

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
        from tools.fiftyone.fiftyone_voc_precheck import REPORT_COLUMNS, write_report_csv

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
        from tools.fiftyone.fiftyone_export_cleaned import build_export_view

        dataset = _FakeDataset()

        view = build_export_view(dataset, ["drop", "bad_case"])

        self.assertEqual(view, "filtered-view")
        self.assertEqual(dataset.calls, [(["drop", "bad_case"], False, False)])

    def test_similarity_parser_uses_expected_defaults(self):
        from tools.fiftyone.fiftyone_compute_similarity import build_parser

        args = build_parser().parse_args(["--dataset-name", "test1_stride10_voc"])

        self.assertEqual(args.dataset_name, "test1_stride10_voc")
        self.assertEqual(args.model, "clip-vit-base32-torch")
        self.assertEqual(args.brain_key, "clip_vit_base32_sim")
        self.assertFalse(args.overwrite)

    def test_deduplicate_parser_uses_expected_defaults(self):
        from tools.fiftyone.fiftyone_deduplicate_dataset import build_parser

        export_dir = os.path.join(get_dataset_dir(), "fiftyone_voc_deduped")
        args = build_parser().parse_args(
            [
                "--dataset-name",
                "test1_stride10_voc",
                "--export-dir",
                export_dir,
            ]
        )

        self.assertEqual(args.dataset_name, "test1_stride10_voc")
        self.assertEqual(args.label_field, "ground_truth")
        self.assertEqual(args.exact_mode, "deduplicate")
        self.assertEqual(args.approx_brain_key, "clip_vit_base32_sim")
        self.assertEqual(args.approx_threshold, 0.12)
        self.assertEqual(args.approx_group_keep_ratio, 0.3)

    def test_pipeline_parser_uses_expected_defaults(self):
        from tools.fiftyone.fiftyone_run_full_dedup_pipeline import build_parser

        voc_root = os.path.join(get_dataset_dir(), "fiftyone_voc")
        export_dir = os.path.join(get_dataset_dir(), "fiftyone_voc_deduped")
        args = build_parser().parse_args(
            [
                "--dataset-name",
                "test1_stride10_voc",
                "--voc-root",
                voc_root,
                "--export-dir",
                export_dir,
            ]
        )

        self.assertEqual(args.model, "clip-vit-base32-torch")
        self.assertEqual(args.brain_key, "clip_vit_base32_sim")
        self.assertEqual(args.report_dir, Path(os.path.join(get_dataset_dir(), "fiftyone_voc", "dedup_reports")))
        self.assertEqual(args.label_field, "ground_truth")

    def test_compute_keep_count_uses_ceiling_and_keeps_at_least_one(self):
        from tools.fiftyone.fiftyone_deduplicate_dataset import compute_keep_count

        self.assertEqual(compute_keep_count(10, 0.3), 3)
        self.assertEqual(compute_keep_count(2, 0.3), 1)
        self.assertEqual(compute_keep_count(3, 0.01), 1)

    def test_select_group_ids_for_removal_keeps_sorted_prefix(self):
        from tools.fiftyone.fiftyone_deduplicate_dataset import select_group_ids_for_removal

        group_rows = [
            {"sample_id": "c", "filepath": "/tmp/img/c.jpg"},
            {"sample_id": "a", "filepath": "/tmp/img/a.jpg"},
            {"sample_id": "b", "filepath": "/tmp/img/b.jpg"},
            {"sample_id": "d", "filepath": "/tmp/img/d.jpg"},
        ]

        kept_ids, removed_ids = select_group_ids_for_removal(group_rows, 0.3)

        self.assertEqual(kept_ids, ["a", "b"])
        self.assertEqual(removed_ids, ["c", "d"])

    def test_write_dedup_csv_writes_header_for_empty_rows(self):
        from tools.fiftyone.fiftyone_dedup_report import DEDUP_REPORT_COLUMNS, write_dedup_csv

        with tempfile.TemporaryDirectory() as tmp:
            out_csv = Path(tmp) / "exact_duplicates.csv"
            write_dedup_csv([], out_csv)

            with out_csv.open("r", encoding="utf-8-sig", newline="") as fh:
                reader = csv.reader(fh)
                header = next(reader)
                remaining = list(reader)

            self.assertEqual(header, list(DEDUP_REPORT_COLUMNS))
            self.assertEqual(remaining, [])

    def test_write_json_report_persists_expected_payload(self):
        from tools.fiftyone.fiftyone_dedup_report import write_json_report

        with tempfile.TemporaryDirectory() as tmp:
            out_json = Path(tmp) / "summary.json"
            payload = {"removed": 12, "kept": 5}
            write_json_report(payload, out_json)

            loaded = json.loads(out_json.read_text(encoding="utf-8"))

            self.assertEqual(loaded, payload)

    def test_get_report_data_dir_returns_report_data_child(self):
        from tools.fiftyone.fiftyone_dedup_report import get_report_data_dir

        report_dir = Path(os.path.join(get_dataset_dir(), "fiftyone_voc", "dedup_reports"))

        self.assertEqual(
            get_report_data_dir(report_dir),
            report_dir.resolve() / "report_data",
        )

    def test_write_reports_only_writes_six_csv_json_files_under_report_data(self):
        from tools.fiftyone.fiftyone_deduplicate_dataset import write_reports

        with tempfile.TemporaryDirectory() as tmp:
            report_dir = Path(tmp) / "dedup_reports"
            report_dir.mkdir(parents=True)
            (report_dir / "exact_duplicates.csv").write_text("legacy", encoding="utf-8")
            (report_dir / "dedup_summary.json").write_text("{}", encoding="utf-8")
            write_reports(
                report_dir,
                exact_keep_rows=[],
                exact_remove_rows=[],
                approx_keep_rows=[],
                approx_remove_rows=[],
                approx_group_payload={"groups": []},
                summary_payload={"dataset_name": "demo"},
            )

            report_data_dir = report_dir / "report_data"
            files = sorted(path.name for path in report_data_dir.iterdir())

            self.assertEqual(
                files,
                [
                    "approx_duplicate_groups.json",
                    "approx_duplicates.csv",
                    "dedup_summary.json",
                    "deleted_samples.csv",
                    "exact_duplicates.csv",
                    "kept_representatives.csv",
                ],
            )
            self.assertFalse((report_data_dir / "exact_original_backup_manifest.csv").exists())
            self.assertFalse((report_data_dir / "exact_original_backup_summary.json").exists())
            self.assertFalse((report_dir / "exact_duplicates.csv").exists())
            self.assertFalse((report_dir / "dedup_summary.json").exists())

    def test_build_dedup_summary_contains_ratio_fields(self):
        from tools.fiftyone.fiftyone_deduplicate_dataset import build_dedup_summary

        summary = build_dedup_summary(
            dataset_name="demo",
            initial_samples=100,
            exact_duplicate_samples=12,
            exact_removed=10,
            approx_duplicate_samples=20,
            approx_removed=15,
            final_samples=75,
            export_dir=Path("/tmp/export"),
            report_dir=Path("/tmp/reports"),
            backup_dir=Path("/tmp/backup"),
            approx_brain_key="clip_vit_base32_sim",
            approx_threshold=0.12,
            approx_group_keep_ratio=0.3,
        )

        self.assertEqual(summary["total_removed"], 25)
        self.assertAlmostEqual(summary["exact_removed_ratio"], 0.10)
        self.assertAlmostEqual(summary["approx_removed_ratio"], 0.15)
        self.assertAlmostEqual(summary["final_retained_ratio"], 0.75)

    def test_write_dedup_stats_csv_outputs_count_and_ratio_rows(self):
        from tools.fiftyone.fiftyone_deduplicate_dataset import write_dedup_stats_csv

        with tempfile.TemporaryDirectory() as tmp:
            report_dir = Path(tmp) / "dedup_reports"
            summary = {
                "initial_samples": 100,
                "exact_removed": 10,
                "approx_removed": 15,
                "final_samples": 75,
                "exact_removed_ratio": 0.10,
                "approx_removed_ratio": 0.15,
                "final_retained_ratio": 0.75,
            }
            out_csv = write_dedup_stats_csv(report_dir, summary)

            with out_csv.open("r", encoding="utf-8-sig", newline="") as fh:
                reader = csv.DictReader(fh)
                rows = list(reader)

            self.assertEqual(reader.fieldnames, ["metric", "count", "ratio"])
            self.assertEqual(rows[0]["metric"], "exact_removed")
            self.assertEqual(rows[0]["count"], "10")
            self.assertEqual(rows[0]["ratio"], "0.1")

    def test_move_exact_duplicates_to_backup_moves_image_and_xml(self):
        from tools.fiftyone.fiftyone_deduplicate_dataset import move_exact_duplicates_to_backup

        with tempfile.TemporaryDirectory() as tmp:
            voc_root = Path(tmp) / "fiftyone_voc"
            data_dir = voc_root / "data"
            labels_dir = voc_root / "labels"
            data_dir.mkdir(parents=True)
            labels_dir.mkdir(parents=True)
            image_path = data_dir / "frame001.jpg"
            xml_path = labels_dir / "frame001.xml"
            image_path.write_bytes(b"fake-image")
            xml_path.write_text("<annotation />", encoding="utf-8")

            rows = [
                {
                    "sample_id": "1",
                    "filepath": str(image_path),
                    "dedup_type": "exact",
                    "group_id": "hash1",
                    "group_size": 2,
                    "kept_or_removed": "removed",
                    "reason": "exact_duplicate_removed",
                    "brain_key": "clip_vit_base32_sim",
                    "threshold": None,
                    "group_keep_ratio": None,
                }
            ]

            backup_dir, updated_rows = move_exact_duplicates_to_backup(voc_root, rows)

            self.assertTrue((backup_dir / "data" / "frame001.jpg").is_file())
            self.assertTrue((backup_dir / "labels" / "frame001.xml").is_file())
            self.assertFalse(image_path.exists())
            self.assertFalse(xml_path.exists())
            self.assertEqual(updated_rows[0]["reason"], "exact_duplicate_removed_moved_to_backup")

    def test_move_exact_duplicates_to_backup_marks_missing_xml_in_reason(self):
        from tools.fiftyone.fiftyone_deduplicate_dataset import move_exact_duplicates_to_backup

        with tempfile.TemporaryDirectory() as tmp:
            voc_root = Path(tmp) / "fiftyone_voc"
            data_dir = voc_root / "data"
            labels_dir = voc_root / "labels"
            data_dir.mkdir(parents=True)
            labels_dir.mkdir(parents=True)
            image_path = data_dir / "frame002.jpg"
            image_path.write_bytes(b"fake-image")

            rows = [
                {
                    "sample_id": "2",
                    "filepath": str(image_path),
                    "dedup_type": "exact",
                    "group_id": "hash2",
                    "group_size": 2,
                    "kept_or_removed": "removed",
                    "reason": "exact_duplicate_removed",
                    "brain_key": "clip_vit_base32_sim",
                    "threshold": None,
                    "group_keep_ratio": None,
                }
            ]

            backup_dir, updated_rows = move_exact_duplicates_to_backup(voc_root, rows)

            self.assertTrue((backup_dir / "data" / "frame002.jpg").is_file())
            self.assertIn("xml_missing", updated_rows[0]["reason"])

    def test_get_chart_paths_returns_expected_png_paths(self):
        from tools.fiftyone.fiftyone_deduplicate_dataset import get_chart_paths

        report_dir = Path(os.path.join(get_dataset_dir(), "fiftyone_voc", "dedup_reports"))
        paths = get_chart_paths(report_dir)

        self.assertEqual(paths["donut"], report_dir.resolve() / "dedup_ratio_donut.png")
        self.assertEqual(paths["bar"], report_dir.resolve() / "dedup_count_bar.png")

    def test_build_work_dataset_name_uses_source_name_and_suffix(self):
        from tools.fiftyone.fiftyone_run_full_dedup_pipeline import build_work_dataset_name

        self.assertEqual(
            build_work_dataset_name("test1_stride10_voc"),
            "test1_stride10_voc__dedup_work",
        )

    def test_ensure_export_dir_rejects_non_empty_directory_without_overwrite(self):
        from tools.fiftyone.fiftyone_deduplicate_dataset import ensure_export_dir

        with tempfile.TemporaryDirectory() as tmp:
            export_dir = Path(tmp) / "voc_deduped"
            export_dir.mkdir()
            (export_dir / "sentinel.txt").write_text("occupied", encoding="utf-8")

            with self.assertRaises(FileExistsError):
                ensure_export_dir(export_dir, overwrite=False)


if __name__ == "__main__":
    unittest.main()
