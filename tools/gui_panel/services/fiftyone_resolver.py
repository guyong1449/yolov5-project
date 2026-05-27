from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class ResolvedDatasetLayout:
    dataset_root: Path
    data_dir: Path
    labels_dir: Path
    layout: str


def resolve_dataset_root(dataset_root: str | Path) -> ResolvedDatasetLayout:
    root = Path(dataset_root).expanduser().resolve()
    candidates = [
        ("fiftyone_voc", root / "fiftyone_voc" / "data", root / "fiftyone_voc" / "labels"),
        ("voc_root", root / "images", root / "annotations"),
    ]
    for layout, data_dir, labels_dir in candidates:
        if data_dir.is_dir() and labels_dir.is_dir():
            return ResolvedDatasetLayout(
                dataset_root=root,
                data_dir=data_dir.resolve(),
                labels_dir=labels_dir.resolve(),
                layout=layout,
            )
    raise FileNotFoundError(
        "Unable to resolve FiftyOne dataset layout. Expected either "
        "<dataset_root>/fiftyone_voc/data + labels or <dataset_root>/images + annotations."
    )
