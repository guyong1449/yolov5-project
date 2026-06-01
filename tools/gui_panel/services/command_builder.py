from __future__ import annotations

import shlex
import sys
from pathlib import Path
from typing import Any

from utils.env_config import get_hyp_file
from tools.gui_panel.schemas import ValidationResponse
from tools.gui_panel.services.fiftyone_resolver import resolve_dataset_root
from tools.gui_panel.task_specs import TASK_SPECS

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_TRAIN_HYP_CANDIDATES = [
    REPO_ROOT / get_hyp_file(),
]


def build_command_preview(task_type: str, values: dict[str, Any]) -> list[str]:
    normalized, _, _ = normalize_values(task_type, values, validate_paths=False)
    command, _ = _build_command(task_type, normalized)
    return command


def validate_task(task_type: str, values: dict[str, Any], *, validate_paths: bool = True) -> ValidationResponse:
    normalized, errors, metadata = normalize_values(task_type, values, validate_paths=validate_paths)
    command, build_metadata = _build_command(task_type, normalized)
    metadata.update(build_metadata)
    return ValidationResponse(
        ok=not errors,
        command=_command_to_text(command),
        argv=command,
        normalized_values=normalized,
        errors=errors,
        metadata=metadata,
    )


def normalize_values(task_type: str, values: dict[str, Any], *, validate_paths: bool) -> tuple[dict[str, Any], list[str], dict[str, Any]]:
    spec = TASK_SPECS[task_type]
    normalized = dict(spec.defaults)
    normalized.update(values or {})
    errors: list[str] = []
    metadata: dict[str, Any] = {}

    for field in spec.fields:
        value = normalized.get(field.name)
        if field.kind == "number" and value not in ("", None):
            if isinstance(value, bool):
                errors.append(f"{field.label} must be numeric")
            elif isinstance(value, str) and "." in value:
                normalized[field.name] = float(value)
            else:
                normalized[field.name] = int(value)
        elif field.kind == "checkbox":
            normalized[field.name] = bool(value)
        elif value is None:
            normalized[field.name] = ""
        elif isinstance(value, str):
            normalized[field.name] = value.strip()
        if field.required and normalized.get(field.name) in ("", None):
            errors.append(f"{field.label} is required")

    for field_name in ("data", "weights", "source", "hyp", "cfg", "voc_root", "project", "dataset_root", "data_dir", "labels_dir"):
        if field_name in normalized and isinstance(normalized[field_name], str):
            normalized[field_name] = normalized[field_name].strip()

    if validate_paths:
        path_fields = {
            "data": "file",
            "weights": "file",
            "source": "any",
            "hyp": "file",
            "cfg": "file",
            "voc_root": "directory_or_empty",
            "project": "directory_parent",
            "dataset_root": "directory",
            "data_dir": "directory",
            "labels_dir": "directory",
        }
        for name, kind in path_fields.items():
            raw = normalized.get(name)
            if not raw:
                continue
            path = Path(raw)
            if kind == "file" and not path.is_file():
                errors.append(f"{name} not found: {path}")
            elif kind == "directory" and not path.is_dir():
                errors.append(f"{name} not found: {path}")
            elif kind == "any" and not path.exists():
                errors.append(f"{name} not found: {path}")
            elif kind == "directory_or_empty" and raw and not path.exists():
                errors.append(f"{name} not found: {path}")
            elif kind == "directory_parent" and path.parent != path and not path.exists() and not path.parent.exists():
                errors.append(f"parent directory not found for {name}: {path.parent}")

    if task_type == "detect":
        if normalized.get("incremental_mp4") and not normalized.get("save_img_frames"):
            errors.append("incremental_mp4 requires save_img_frames")
        if normalized.get("incremental_mp4") and not normalized.get("voc_root"):
            errors.append("incremental_mp4 requires voc_root")

    if task_type == "fiftyone":
        mode = normalized.get("mode")
        if mode == "dataset_root_auto":
            if normalized.get("dataset_root"):
                try:
                    resolved = resolve_dataset_root(normalized["dataset_root"])
                    metadata["resolved_layout"] = resolved.layout
                    metadata["resolved_data_dir"] = str(resolved.data_dir)
                    metadata["resolved_labels_dir"] = str(resolved.labels_dir)
                except FileNotFoundError as exc:
                    if validate_paths:
                        errors.append(str(exc))
            elif validate_paths:
                errors.append("dataset_root is required")
        elif mode == "explicit_voc":
            if validate_paths and not normalized.get("data_dir"):
                errors.append("data_dir is required")
            if validate_paths and not normalized.get("labels_dir"):
                errors.append("labels_dir is required")
        else:
            errors.append(f"Unsupported FiftyOne mode: {mode}")

    return normalized, errors, metadata


def _build_command(task_type: str, values: dict[str, Any]) -> tuple[list[str], dict[str, Any]]:
    if task_type == "train":
        return _build_run_with_log_command("train.py", values, _train_args(values))
    if task_type == "detect":
        return _build_run_with_log_command("detect.py", values, _detect_args(values))
    if task_type == "val":
        return _build_run_with_log_command("val.py", values, _val_args(values))
    if task_type == "fiftyone":
        return _build_fiftyone_command(values)
    raise KeyError(f"Unsupported task type: {task_type}")


def _build_run_with_log_command(entrypoint: str, values: dict[str, Any], cli_args: list[str]) -> tuple[list[str], dict[str, Any]]:
    command = [
        sys.executable,
        "scripts/run_with_log.py",
        "--name",
        str(values["name"]),
        "--cwd",
        str(REPO_ROOT),
        "--",
        sys.executable,
        entrypoint,
        *cli_args,
    ]
    output_path = str((REPO_ROOT / values["project"] / values["name"]).resolve())
    metadata = {"output_path": output_path}
    return command, metadata


def _train_args(values: dict[str, Any]) -> list[str]:
    hyp_path = _resolve_train_hyp(values.get("hyp", ""))
    args = [
        "--data", values["data"],
        "--weights", values["weights"],
        "--project", values["project"],
        "--name", values["name"],
        "--device", str(values["device"]),
        "--epochs", str(values["epochs"]),
        "--batch-size", str(values["batch_size"]),
        "--imgsz", str(values["imgsz"]),
        "--workers", str(values["workers"]),
        "--seed", str(values["seed"]),
        "--patience", str(values["patience"]),
        "--optimizer", str(values["optimizer"]),
    ]
    if values.get("exist_ok"):
        args.append("--exist-ok")
    if values.get("resume"):
        args.append("--resume")
    if hyp_path:
        args.extend(["--hyp", hyp_path])
    if values.get("cfg"):
        args.extend(["--cfg", values["cfg"]])
    if values.get("amp_mode") == "on":
        args.append("--amp")
    elif values.get("amp_mode") == "off":
        args.append("--no-amp")
    args.extend(_parse_extra_args(values.get("extra_args", "")))
    return args


def _detect_args(values: dict[str, Any]) -> list[str]:
    args = [
        "--weights", values["weights"],
        "--source", values["source"],
        "--data", values["data"],
        "--project", values["project"],
        "--name", values["name"],
        "--device", str(values["device"]),
        "--imgsz", str(values["imgsz"]),
        "--conf-thres", str(values["conf_thres"]),
        "--iou-thres", str(values["iou_thres"]),
        "--vid-stride", str(values["vid_stride"]),
    ]
    for key, flag in (
        ("view_img", "--view-img"),
        ("save_txt", "--save-txt"),
        ("save_conf", "--save-conf"),
        ("nosave", "--nosave"),
        ("save_img_frames", "--save-img-frames"),
        ("incremental_mp4", "--incremental-mp4"),
        ("exist_ok", "--exist-ok"),
    ):
        if values.get(key):
            args.append(flag)
    if values.get("voc_root"):
        args.extend(["--voc-root", values["voc_root"]])
    if values.get("classes"):
        classes = [part.strip() for part in str(values["classes"]).split(",") if part.strip()]
        if classes:
            args.append("--classes")
            args.extend(classes)
    args.extend(_parse_extra_args(values.get("extra_args", "")))
    return args


def _val_args(values: dict[str, Any]) -> list[str]:
    args = [
        "--data", values["data"],
        "--weights", values["weights"],
        "--project", values["project"],
        "--name", values["name"],
        "--device", str(values["device"]),
        "--batch-size", str(values["batch_size"]),
        "--imgsz", str(values["imgsz"]),
        "--conf-thres", str(values["conf_thres"]),
        "--iou-thres", str(values["iou_thres"]),
        "--task", str(values["task"]),
        "--workers", str(values["workers"]),
    ]
    for key, flag in (
        ("save_txt", "--save-txt"),
        ("save_json", "--save-json"),
        ("half", "--half"),
        ("dnn", "--dnn"),
        ("exist_ok", "--exist-ok"),
    ):
        if values.get(key):
            args.append(flag)
    args.extend(_parse_extra_args(values.get("extra_args", "")))
    return args


def _build_fiftyone_command(values: dict[str, Any]) -> tuple[list[str], dict[str, Any]]:
    mode = values.get("mode")
    metadata: dict[str, Any] = {}
    if mode == "dataset_root_auto":
        resolved = resolve_dataset_root(values["dataset_root"])
        data_dir = str(resolved.data_dir)
        labels_dir = str(resolved.labels_dir)
        metadata["output_path"] = str(resolved.dataset_root)
        metadata["resolved_layout"] = resolved.layout
    else:
        data_dir = values["data_dir"]
        labels_dir = values["labels_dir"]
        metadata["output_path"] = str(Path(data_dir).resolve().parent)
    command = [
        sys.executable,
        "tools/fiftyone/fiftyone_import_voc.py",
        "--name", values["dataset_name"],
        "--data-dir", data_dir,
        "--labels-dir", labels_dir,
        "--label-field", values["label_field"],
    ]
    if values.get("overwrite"):
        command.append("--overwrite")
    if values.get("launch_app", True):
        command.append("--wait")
    else:
        command.append("--no-app")
    command.extend(_parse_extra_args(values.get("extra_args", "")))
    return command, metadata


def _parse_extra_args(raw: str) -> list[str]:
    text = str(raw or "").strip()
    return shlex.split(text) if text else []


def _command_to_text(command: list[str]) -> str:
    parts = [shlex.quote(part) for part in command]
    if not parts:
        return ""

    try:
        sep_idx = parts.index("--")
    except ValueError:
        sep_idx = None

    if sep_idx is not None and sep_idx + 2 < len(parts):
        # Line 1: python + run_with_log.py args + --
        line1 = " ".join(parts[: sep_idx + 1])

        # Line 2: sub-python + sub-script
        line2 = "  " + " ".join(parts[sep_idx + 1 : sep_idx + 3])

        remaining = parts[sep_idx + 3 :]
        lines = [line1 + " \\", line2 + " \\"]

        chunk_size = 6
        for i in range(0, len(remaining), chunk_size):
            chunk = remaining[i : i + chunk_size]
            line = "  " + " ".join(chunk)
            if i + chunk_size < len(remaining):
                line += " \\"
            lines.append(line)

        return "\n".join(lines)

    # Fallback: one arg per line
    result = parts[0]
    for part in parts[1:]:
        result += " \\\n  " + part
    return result


def _resolve_train_hyp(raw_value: str) -> str:
    text = str(raw_value or "").strip()
    if text:
        return text
    for candidate in DEFAULT_TRAIN_HYP_CANDIDATES:
        if candidate.is_file():
            try:
                return str(candidate.relative_to(REPO_ROOT)).replace("\\", "/")
            except ValueError:
                return str(candidate)
    return ""
