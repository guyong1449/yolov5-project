from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
import shutil
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.env_config import get_dedup_reports_dir
from tools.fiftyone.fiftyone_dedup_report import get_report_data_dir, write_dedup_csv, write_json_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Deduplicate a FiftyOne dataset and export a cleaned VOC copy."
    )
    parser.add_argument("--dataset-name", required=True, help="Existing FiftyOne dataset name")
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
        "--exact-mode",
        choices=("deduplicate",),
        default="deduplicate",
        help="How to handle exact duplicates.",
    )
    parser.add_argument(
        "--approx-brain-key",
        default="clip_vit_base32_sim",
        help="Existing similarity brain key used to discover approximate duplicates.",
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
        help="Allow overwriting an existing non-empty export directory.",
    )
    return parser


def ensure_export_dir(export_dir: Path, *, overwrite: bool) -> Path:
    export_dir = Path(export_dir).resolve()
    if export_dir.exists() and any(export_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"Export directory already exists and is not empty: {export_dir}")
    export_dir.mkdir(parents=True, exist_ok=True)
    return export_dir


def compute_keep_count(group_size: int, keep_ratio: float) -> int:
    if group_size <= 0:
        raise ValueError(f"group_size must be positive, got {group_size}")
    if not 0 < keep_ratio <= 1:
        raise ValueError(f"keep_ratio must be within (0, 1], got {keep_ratio}")
    return max(1, int(math.ceil(group_size * keep_ratio)))


def select_group_ids_for_removal(
    group_rows: list[dict[str, object]],
    keep_ratio: float,
) -> tuple[list[str], list[str]]:
    sorted_rows = sorted(group_rows, key=lambda row: str(row["filepath"]).lower())
    keep_count = compute_keep_count(len(sorted_rows), keep_ratio)
    kept_rows = sorted_rows[:keep_count]
    removed_rows = sorted_rows[keep_count:]
    kept_ids = [str(row["sample_id"]) for row in kept_rows]
    removed_ids = [str(row["sample_id"]) for row in removed_rows]
    return kept_ids, removed_ids


def _sample_filepath(sample) -> str:
    return str(getattr(sample, "local_path", None) or sample.filepath)


def _safe_ratio(part: int, whole: int) -> float:
    if whole <= 0:
        return 0.0
    return part / whole


def get_chart_paths(report_dir: Path) -> dict[str, Path]:
    resolved = Path(report_dir).resolve()
    return {
        "donut": resolved / "dedup_ratio_donut.png",
        "bar": resolved / "dedup_count_bar.png",
    }


def _build_report_row(
    *,
    sample_id: str,
    filepath: str,
    dedup_type: str,
    group_id: str,
    group_size: int,
    kept_or_removed: str,
    reason: str,
    brain_key: str,
    threshold: float | None,
    group_keep_ratio: float | None,
) -> dict[str, object]:
    return {
        "sample_id": sample_id,
        "filepath": filepath,
        "dedup_type": dedup_type,
        "group_id": group_id,
        "group_size": group_size,
        "kept_or_removed": kept_or_removed,
        "reason": reason,
        "brain_key": brain_key,
        "threshold": threshold,
        "group_keep_ratio": group_keep_ratio,
    }


def _save_exact_duplicate_view(dataset, sample_ids: list[str]) -> None:
    if not sample_ids:
        return
    exact_dup_view = dataset.select(sample_ids)
    dataset.save_view("exact_dup_view", exact_dup_view, overwrite=True)


def _save_approx_duplicate_views(dataset, neighbors_map: dict[str, list[tuple[str, float]]]) -> None:
    if not neighbors_map:
        return

    duplicate_ids: list[str] = []
    group_field = "approx_dup_group_id"
    for rep_id, duplicates in neighbors_map.items():
        ids = [rep_id] + [dup_id for dup_id, _distance in duplicates]
        duplicate_ids.extend(ids)
        for sample in dataset.select(ids).iter_samples(autosave=True):
            sample[group_field] = rep_id

    approx_dup_view = dataset.select(duplicate_ids)
    dataset.save_view("approx_dup_view", approx_dup_view, overwrite=True)
    approx_dup_groups_view = approx_dup_view.group_by(group_field)
    dataset.save_view("approx_dup_groups_view", approx_dup_groups_view, overwrite=True)


def deduplicate_exact_duplicates(
    dataset,
    *,
    brain_key: str,
) -> tuple[list[str], list[dict[str, object]], list[dict[str, object]]]:
    from collections import defaultdict

    import fiftyone.core.utils as fou

    filehash_to_rows: dict[str, list[dict[str, object]]] = defaultdict(list)
    for sample in dataset.iter_samples(autosave=True):
        filehash = str(fou.compute_filehash(_sample_filepath(sample)))
        sample["filehash"] = filehash
        filehash_to_rows[filehash].append(
            {
                "sample_id": str(sample.id),
                "filepath": _sample_filepath(sample),
            }
        )

    keep_rows: list[dict[str, object]] = []
    remove_rows: list[dict[str, object]] = []
    remove_ids: list[str] = []
    exact_view_ids: list[str] = []

    for filehash, rows in sorted(filehash_to_rows.items()):
        if len(rows) <= 1:
            continue

        sorted_rows = sorted(rows, key=lambda row: str(row["filepath"]).lower())
        exact_view_ids.extend(str(row["sample_id"]) for row in sorted_rows)
        keep_row = sorted_rows[0]
        keep_rows.append(
            _build_report_row(
                sample_id=str(keep_row["sample_id"]),
                filepath=str(keep_row["filepath"]),
                dedup_type="exact",
                group_id=filehash,
                group_size=len(sorted_rows),
                kept_or_removed="kept",
                reason="exact_duplicate_representative",
                brain_key=brain_key,
                threshold=None,
                group_keep_ratio=None,
            )
        )
        for remove_row in sorted_rows[1:]:
            sample_id = str(remove_row["sample_id"])
            remove_ids.append(sample_id)
            remove_rows.append(
                _build_report_row(
                    sample_id=sample_id,
                    filepath=str(remove_row["filepath"]),
                    dedup_type="exact",
                    group_id=filehash,
                    group_size=len(sorted_rows),
                    kept_or_removed="removed",
                    reason="exact_duplicate_removed",
                    brain_key=brain_key,
                    threshold=None,
                    group_keep_ratio=None,
                )
            )

    _save_exact_duplicate_view(dataset, exact_view_ids)
    if remove_ids:
        dataset.delete_samples(remove_ids)

    return remove_ids, keep_rows, remove_rows


def deduplicate_approximate_duplicates(
    dataset,
    *,
    brain_key: str,
    threshold: float,
    keep_ratio: float,
) -> tuple[list[str], list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    index = dataset.load_brain_results(brain_key)
    index.find_duplicates(thresh=threshold)
    neighbors_map = index.neighbors_map or {}
    _save_approx_duplicate_views(dataset, neighbors_map)

    keep_rows: list[dict[str, object]] = []
    remove_rows: list[dict[str, object]] = []
    remove_ids: list[str] = []
    group_summaries: list[dict[str, object]] = []

    for rep_id, duplicates in sorted(neighbors_map.items()):
        ids = [rep_id] + [dup_id for dup_id, _distance in duplicates]
        filepath_map = {
            str(sample.id): _sample_filepath(sample)
            for sample in dataset.select(ids).iter_samples()
        }
        group_rows = [
            {
                "sample_id": sample_id,
                "filepath": filepath_map[sample_id],
            }
            for sample_id in ids
            if sample_id in filepath_map
        ]
        if not group_rows:
            continue

        kept_ids, removed_ids = select_group_ids_for_removal(group_rows, keep_ratio)
        group_id = str(rep_id)
        group_size = len(group_rows)
        group_summaries.append(
            {
                "group_id": group_id,
                "group_size": group_size,
                "kept_ids": kept_ids,
                "removed_ids": removed_ids,
                "threshold": threshold,
                "group_keep_ratio": keep_ratio,
            }
        )

        for sample_id in kept_ids:
            keep_rows.append(
                _build_report_row(
                    sample_id=sample_id,
                    filepath=filepath_map[sample_id],
                    dedup_type="approx",
                    group_id=group_id,
                    group_size=group_size,
                    kept_or_removed="kept",
                    reason="approx_duplicate_representative",
                    brain_key=brain_key,
                    threshold=threshold,
                    group_keep_ratio=keep_ratio,
                )
            )
        for sample_id in removed_ids:
            remove_ids.append(sample_id)
            remove_rows.append(
                _build_report_row(
                    sample_id=sample_id,
                    filepath=filepath_map[sample_id],
                    dedup_type="approx",
                    group_id=group_id,
                    group_size=group_size,
                    kept_or_removed="removed",
                    reason="approx_duplicate_removed_by_group_ratio",
                    brain_key=brain_key,
                    threshold=threshold,
                    group_keep_ratio=keep_ratio,
                )
            )

    if remove_ids:
        dataset.delete_samples(remove_ids)

    return remove_ids, keep_rows, remove_rows, {"groups": group_summaries}


def export_dataset(dataset, export_dir: Path, *, label_field: str, overwrite: bool) -> None:
    import fiftyone as fo

    dataset.export(
        export_dir=str(export_dir),
        dataset_type=fo.types.VOCDetectionDataset,
        label_field=label_field,
        export_media=True,
        overwrite=overwrite,
    )


def build_dedup_summary(
    *,
    dataset_name: str,
    initial_samples: int,
    exact_duplicate_samples: int,
    exact_removed: int,
    approx_duplicate_samples: int,
    approx_removed: int,
    final_samples: int,
    export_dir: Path,
    report_dir: Path,
    backup_dir: Path,
    approx_brain_key: str,
    approx_threshold: float,
    approx_group_keep_ratio: float,
) -> dict[str, object]:
    return {
        "dataset_name": dataset_name,
        "initial_samples": initial_samples,
        "exact_duplicate_samples": exact_duplicate_samples,
        "exact_removed": exact_removed,
        "approx_duplicate_samples": approx_duplicate_samples,
        "approx_removed": approx_removed,
        "final_samples": final_samples,
        "total_removed": exact_removed + approx_removed,
        "exact_removed_ratio": _safe_ratio(exact_removed, initial_samples),
        "approx_removed_ratio": _safe_ratio(approx_removed, initial_samples),
        "final_retained_ratio": _safe_ratio(final_samples, initial_samples),
        "export_dir": str(Path(export_dir).resolve()),
        "report_dir": str(Path(report_dir).resolve()),
        "backup_dir": str(Path(backup_dir).resolve()),
        "approx_brain_key": approx_brain_key,
        "approx_threshold": approx_threshold,
        "approx_group_keep_ratio": approx_group_keep_ratio,
    }


def write_reports(
    report_dir: Path,
    *,
    exact_keep_rows: list[dict[str, object]],
    exact_remove_rows: list[dict[str, object]],
    approx_keep_rows: list[dict[str, object]],
    approx_remove_rows: list[dict[str, object]],
    approx_group_payload: dict[str, object],
    summary_payload: dict[str, object],
) -> Path:
    report_dir = Path(report_dir).resolve()
    report_dir.mkdir(parents=True, exist_ok=True)
    for legacy_name in (
        "exact_duplicates.csv",
        "approx_duplicates.csv",
        "deleted_samples.csv",
        "kept_representatives.csv",
        "dedup_summary.json",
        "approx_duplicate_groups.json",
    ):
        legacy_path = report_dir / legacy_name
        if legacy_path.exists():
            legacy_path.unlink()

    report_data_dir = get_report_data_dir(report_dir)
    report_data_dir.mkdir(parents=True, exist_ok=True)

    write_dedup_csv(exact_keep_rows + exact_remove_rows, report_data_dir / "exact_duplicates.csv")
    write_dedup_csv(approx_keep_rows + approx_remove_rows, report_data_dir / "approx_duplicates.csv")
    write_dedup_csv(exact_remove_rows + approx_remove_rows, report_data_dir / "deleted_samples.csv")
    write_dedup_csv(exact_keep_rows + approx_keep_rows, report_data_dir / "kept_representatives.csv")
    write_json_report(summary_payload, report_data_dir / "dedup_summary.json")
    write_json_report(approx_group_payload, report_data_dir / "approx_duplicate_groups.json")
    return report_data_dir


def _xml_path_for_image(voc_root: Path, image_path: Path) -> Path:
    return Path(voc_root).resolve() / "labels" / f"{image_path.stem}.xml"


def move_exact_duplicates_to_backup(
    voc_root: Path,
    exact_remove_rows: list[dict[str, object]],
) -> tuple[Path, list[dict[str, object]]]:
    voc_root = Path(voc_root).resolve()
    backup_dir = voc_root / "backup_removed_exact"
    backup_data_dir = backup_dir / "data"
    backup_labels_dir = backup_dir / "labels"
    backup_data_dir.mkdir(parents=True, exist_ok=True)
    backup_labels_dir.mkdir(parents=True, exist_ok=True)

    updated_rows: list[dict[str, object]] = []
    for row in exact_remove_rows:
        row_copy = dict(row)
        image_path = Path(str(row_copy["filepath"])).resolve()
        xml_path = _xml_path_for_image(voc_root, image_path)

        if image_path.is_file():
            shutil.move(str(image_path), str(backup_data_dir / image_path.name))

        if xml_path.is_file():
            shutil.move(str(xml_path), str(backup_labels_dir / xml_path.name))
            row_copy["reason"] = "exact_duplicate_removed_moved_to_backup"
        else:
            row_copy["reason"] = "exact_duplicate_removed_moved_to_backup_xml_missing"

        updated_rows.append(row_copy)

    return backup_dir, updated_rows


def write_dedup_stats_csv(report_dir: Path, summary_payload: dict[str, object]) -> Path:
    report_dir = Path(report_dir).resolve()
    report_dir.mkdir(parents=True, exist_ok=True)
    out_csv = report_dir / "dedup_stats.csv"
    rows = [
        {
            "metric": "exact_removed",
            "count": summary_payload["exact_removed"],
            "ratio": summary_payload["exact_removed_ratio"],
        },
        {
            "metric": "approx_removed",
            "count": summary_payload["approx_removed"],
            "ratio": summary_payload["approx_removed_ratio"],
        },
        {
            "metric": "retained",
            "count": summary_payload["final_samples"],
            "ratio": summary_payload["final_retained_ratio"],
        },
        {
            "metric": "initial_samples",
            "count": summary_payload["initial_samples"],
            "ratio": 1.0,
        },
    ]
    with out_csv.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=("metric", "count", "ratio"))
        writer.writeheader()
        writer.writerows(rows)
    return out_csv


def write_dedup_charts(report_dir: Path, summary_payload: dict[str, object]) -> dict[str, Path]:
    import matplotlib.pyplot as plt

    paths = get_chart_paths(report_dir)
    labels = ["exact_removed", "approx_removed", "retained"]
    sizes = [
        summary_payload["exact_removed"],
        summary_payload["approx_removed"],
        summary_payload["final_samples"],
    ]
    colors = ["#d95f02", "#7570b3", "#1b9e77"]

    fig, ax = plt.subplots(figsize=(6, 6))
    wedges, _ = ax.pie(sizes, colors=colors, startangle=90, wedgeprops={"width": 0.45})
    ax.legend(wedges, labels, loc="center left", bbox_to_anchor=(1, 0.5))
    ax.set_title("Dedup Ratio")
    fig.tight_layout()
    fig.savefig(paths["donut"], dpi=160)
    plt.close(fig)

    bar_labels = ["initial_samples", "exact_removed", "approx_removed", "final_samples"]
    bar_values = [
        summary_payload["initial_samples"],
        summary_payload["exact_removed"],
        summary_payload["approx_removed"],
        summary_payload["final_samples"],
    ]
    bar_colors = ["#4c78a8", "#d95f02", "#7570b3", "#1b9e77"]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(bar_labels, bar_values, color=bar_colors)
    ax.set_title("Dedup Counts")
    ax.set_ylabel("Samples")
    fig.tight_layout()
    fig.savefig(paths["bar"], dpi=160)
    plt.close(fig)

    return paths


def deduplicate_dataset(
    dataset_name: str,
    export_dir: Path,
    *,
    report_dir: Path,
    label_field: str = "ground_truth",
    exact_mode: str = "deduplicate",
    approx_brain_key: str = "clip_vit_base32_sim",
    approx_threshold: float = 0.12,
    approx_group_keep_ratio: float = 0.3,
    overwrite: bool = False,
) -> dict[str, object]:
    import fiftyone as fo

    if exact_mode != "deduplicate":
        raise ValueError(f"Unsupported exact_mode: {exact_mode}")
    if not fo.dataset_exists(dataset_name):
        raise ValueError(f"FiftyOne dataset not found: {dataset_name}")

    export_dir = ensure_export_dir(export_dir, overwrite=overwrite)
    dataset = fo.load_dataset(dataset_name)

    if approx_brain_key not in dataset.list_brain_runs():
        raise ValueError(
            f"Similarity brain run not found: {approx_brain_key}. Run tools/fiftyone/fiftyone_compute_similarity.py first."
        )

    initial_count = len(dataset)
    exact_removed_ids, exact_keep_rows, exact_remove_rows = deduplicate_exact_duplicates(
        dataset,
        brain_key=approx_brain_key,
    )
    approx_removed_ids, approx_keep_rows, approx_remove_rows, approx_group_payload = deduplicate_approximate_duplicates(
        dataset,
        brain_key=approx_brain_key,
        threshold=approx_threshold,
        keep_ratio=approx_group_keep_ratio,
    )

    export_dataset(
        dataset,
        export_dir,
        label_field=label_field,
        overwrite=overwrite,
    )

    backup_dir = Path(report_dir).resolve().parent / "backup_removed_exact"
    summary_payload = build_dedup_summary(
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
        approx_brain_key=approx_brain_key,
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
        summary_payload=summary_payload,
    )
    return summary_payload


def main() -> None:
    args = build_parser().parse_args()
    summary = deduplicate_dataset(
        args.dataset_name,
        args.export_dir,
        report_dir=args.report_dir,
        label_field=args.label_field,
        exact_mode=args.exact_mode,
        approx_brain_key=args.approx_brain_key,
        approx_threshold=args.approx_threshold,
        approx_group_keep_ratio=args.approx_group_keep_ratio,
        overwrite=args.overwrite,
    )
    for key, value in summary.items():
        print(f"{key}={value}")


if __name__ == "__main__":
    main()
