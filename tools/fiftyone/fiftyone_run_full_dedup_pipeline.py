from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.env_config import get_dedup_reports_dir
from tools.fiftyone.fiftyone_compute_similarity import compute_similarity_run
from tools.fiftyone.fiftyone_import_voc import import_voc_dataset
from tools.fiftyone.fiftyone_deduplicate_dataset import (
    build_dedup_summary,
    deduplicate_approximate_duplicates,
    deduplicate_exact_duplicates,
    ensure_export_dir,
    export_dataset,
    move_exact_duplicates_to_backup,
    write_dedup_charts,
    write_dedup_stats_csv,
    write_reports,
)


def build_work_dataset_name(dataset_name: str) -> str:
    return f"{dataset_name}__dedup_work"


def ensure_source_dataset_ready(dataset_name: str, voc_root: Path, *, label_field: str) -> None:
    import fiftyone as fo

    if fo.dataset_exists(dataset_name):
        dataset = fo.load_dataset(dataset_name)
        if len(dataset) > 0:
            return
        fo.delete_dataset(dataset_name)

    import_voc_dataset(
        dataset_name,
        Path(voc_root) / "data",
        Path(voc_root) / "labels",
        overwrite=False,
        label_field=label_field,
    )


def create_work_dataset(dataset_name: str, *, overwrite: bool) -> str:
    import fiftyone as fo

    source_dataset = fo.load_dataset(dataset_name)
    work_dataset_name = build_work_dataset_name(dataset_name)
    if fo.dataset_exists(work_dataset_name):
        if not overwrite:
            raise ValueError(
                f"Working dataset already exists: {work_dataset_name}. Use --overwrite to replace it."
            )
        fo.delete_dataset(work_dataset_name)

    source_dataset.clone(work_dataset_name, persistent=False)
    return work_dataset_name


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the full FiftyOne similarity, deduplication, backup, reporting, and chart pipeline."
    )
    parser.add_argument("--dataset-name", required=True, help="Existing FiftyOne dataset name")
    parser.add_argument(
        "--model",
        default="clip-vit-base32-torch",
        help="Model zoo name used to generate embeddings.",
    )
    parser.add_argument(
        "--brain-key",
        default="clip_vit_base32_sim",
        help="Brain key used to store the similarity run.",
    )
    parser.add_argument("--voc-root", required=True, type=Path, help="Original VOC root containing data/ and labels/")
    parser.add_argument("--export-dir", required=True, type=Path, help="Destination VOC export directory")
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=Path(get_dedup_reports_dir()),
        help="Directory for CSV/JSON reports and charts.",
    )
    parser.add_argument(
        "--label-field",
        default="ground_truth",
        help="Detection field to export in VOC format.",
    )
    parser.add_argument(
        "--approx-threshold",
        type=float,
        default=0.12,
        help="Distance threshold used to form approximate duplicate groups.",
    )
    parser.add_argument(
        "--approx-group-keep-ratio",
        type=float,
        default=0.3,
        help="Per-group keep ratio for approximate duplicate groups.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting an existing non-empty export directory and replacing an existing brain run.",
    )
    return parser


def run_full_pipeline(
    dataset_name: str,
    *,
    model: str,
    brain_key: str,
    voc_root: Path,
    export_dir: Path,
    report_dir: Path,
    label_field: str,
    approx_threshold: float,
    approx_group_keep_ratio: float,
    overwrite: bool,
) -> dict[str, object]:
    import fiftyone as fo

    ensure_source_dataset_ready(dataset_name, voc_root, label_field=label_field)
    compute_similarity_run(
        dataset_name,
        model=model,
        brain_key=brain_key,
        overwrite=overwrite,
    )

    work_dataset_name = create_work_dataset(dataset_name, overwrite=overwrite)
    dataset = fo.load_dataset(work_dataset_name)
    initial_count = len(dataset)
    exact_removed_ids, exact_keep_rows, exact_remove_rows = deduplicate_exact_duplicates(
        dataset,
        brain_key=brain_key,
    )
    backup_dir, exact_remove_rows = move_exact_duplicates_to_backup(voc_root, exact_remove_rows)
    approx_removed_ids, approx_keep_rows, approx_remove_rows, approx_group_payload = deduplicate_approximate_duplicates(
        dataset,
        brain_key=brain_key,
        threshold=approx_threshold,
        keep_ratio=approx_group_keep_ratio,
    )

    ensure_export_dir(export_dir, overwrite=overwrite)
    export_dataset(
        dataset,
        export_dir,
        label_field=label_field,
        overwrite=overwrite,
    )

    summary = build_dedup_summary(
        dataset_name=dataset_name,
        initial_samples=initial_count,
        exact_duplicate_samples=len(exact_keep_rows) + len(exact_remove_rows),
        exact_removed=len(exact_removed_ids),
        approx_duplicate_samples=len(approx_keep_rows) + len(approx_remove_rows),
        approx_removed=len(approx_removed_ids),
        final_samples=len(dataset),
        export_dir=export_dir,
        report_dir=report_dir,
        backup_dir=backup_dir,
        approx_brain_key=brain_key,
        approx_threshold=approx_threshold,
        approx_group_keep_ratio=approx_group_keep_ratio,
    )
    write_reports(
        report_dir,
        exact_keep_rows=exact_keep_rows,
        exact_remove_rows=exact_remove_rows,
        approx_keep_rows=approx_keep_rows,
        approx_remove_rows=approx_remove_rows,
        approx_group_payload=approx_group_payload,
        summary_payload=summary,
    )
    write_dedup_stats_csv(report_dir, summary)
    write_dedup_charts(report_dir, summary)
    fo.delete_dataset(work_dataset_name)
    return summary


def main() -> None:
    args = build_parser().parse_args()
    summary = run_full_pipeline(
        args.dataset_name,
        model=args.model,
        brain_key=args.brain_key,
        voc_root=args.voc_root,
        export_dir=args.export_dir,
        report_dir=args.report_dir,
        label_field=args.label_field,
        approx_threshold=args.approx_threshold,
        approx_group_keep_ratio=args.approx_group_keep_ratio,
        overwrite=args.overwrite,
    )
    for key, value in summary.items():
        print(f"{key}={value}")


if __name__ == "__main__":
    main()
