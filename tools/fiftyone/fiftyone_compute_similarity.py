from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute and persist a FiftyOne similarity run for an existing dataset."
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
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete an existing brain run with the same key before recomputing.",
    )
    return parser


def compute_similarity_run(
    dataset_name: str,
    *,
    model: str = "clip-vit-base32-torch",
    brain_key: str = "clip_vit_base32_sim",
    overwrite: bool = False,
):
    import fiftyone as fo
    import fiftyone.brain as fob

    if not fo.dataset_exists(dataset_name):
        raise ValueError(f"FiftyOne dataset not found: {dataset_name}")

    dataset = fo.load_dataset(dataset_name)
    if brain_key in dataset.list_brain_runs():
        if not overwrite:
            raise ValueError(
                f"Similarity brain run already exists: {brain_key}. Use --overwrite to replace it."
            )
        dataset.delete_brain_run(brain_key)

    fob.compute_similarity(
        dataset,
        model=model,
        brain_key=brain_key,
    )
    dataset.reload()
    return dataset


def main() -> None:
    args = build_parser().parse_args()
    dataset = compute_similarity_run(
        args.dataset_name,
        model=args.model,
        brain_key=args.brain_key,
        overwrite=args.overwrite,
    )
    print(f"dataset_name={dataset.name}")
    print(f"samples_count={len(dataset)}")
    print(f"model={args.model}")
    print(f"brain_key={args.brain_key}")
    print(f"brain_runs={','.join(dataset.list_brain_runs())}")


if __name__ == "__main__":
    main()
