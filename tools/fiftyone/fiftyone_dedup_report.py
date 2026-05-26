from __future__ import annotations

import csv
import json
from pathlib import Path


DEDUP_REPORT_COLUMNS = (
    "sample_id",
    "filepath",
    "dedup_type",
    "group_id",
    "group_size",
    "kept_or_removed",
    "reason",
    "brain_key",
    "threshold",
    "group_keep_ratio",
)


def get_report_data_dir(report_dir: Path) -> Path:
    return Path(report_dir).resolve() / "report_data"


def ensure_parent_dir(path: Path) -> Path:
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def write_dedup_csv(rows: list[dict[str, object]], out_csv: Path) -> Path:
    out_csv = ensure_parent_dir(out_csv)
    with out_csv.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=DEDUP_REPORT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return out_csv


def write_json_report(payload: object, out_json: Path) -> Path:
    out_json = ensure_parent_dir(out_json)
    out_json.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return out_json
