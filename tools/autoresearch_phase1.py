from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
RUNS_ROOT = ROOT / "runs" / "autoresearch"
SNAPSHOT_DIR = RUNS_ROOT / "snapshots"
GENERATED_HYP_DIR = SNAPSHOT_DIR / "generated_hyps"
LEADERBOARD_DIR = RUNS_ROOT / "leaderboard"
LEADERBOARD_TSV = LEADERBOARD_DIR / "history.tsv"
CHAMPION_YAML = LEADERBOARD_DIR / "champion.yaml"
BASELINE_YAML = LEADERBOARD_DIR / "baseline.yaml"
DEFAULT_PYTHON = Path(r"D:\Miniconda3\envs\yolo\python.exe")
FALLBACK_PYTHONS = (
    Path(r"D:\Miniconda3\python.exe"),
    Path(r"D:\Miniconda3\envs\py38\python.exe"),
)
DEFAULT_DATA = ROOT / "data" / "dataAirVis.yaml"
DEFAULT_WEIGHTS = ROOT / "checkpoint" / "yolov5_best.pt"
DEFAULT_HYP = ROOT / "data" / "hyps" / "hyp.scratch-low.yaml"
DEFAULT_DEVICE = "0"
DEFAULT_SEED = 0
BASELINE_EPOCHS = 10
SMOKE_EPOCHS = 1
SPRINT_EPOCHS = 10
METRIC_KEYS = ("metrics/mAP_0.5", "mAP_0.5", "metrics/mAP50(B)")


@dataclass
class CandidateSpec:
    candidate_id: str
    description: str
    cli_overrides: dict[str, Any]
    hyp_overrides: dict[str, Any]
    base_hyp: Path


@dataclass
class RunOutcome:
    status: str
    decision: str
    metric: float | None
    compared_against: float | None
    save_dir: Path
    snapshot_path: Path
    log_path: Path
    command: list[str]
    returncode: int | None


def stage_epochs(stage: str) -> int:
    mapping = {
        "baseline": BASELINE_EPOCHS,
        "smoke": SMOKE_EPOCHS,
        "sprint": SPRINT_EPOCHS,
    }
    if stage not in mapping:
        raise ValueError(f"Unsupported stage: {stage}")
    return mapping[stage]


def ensure_layout() -> None:
    for path in (
        RUNS_ROOT / "baseline",
        RUNS_ROOT / "smoke",
        RUNS_ROOT / "sprint",
        SNAPSHOT_DIR,
        GENERATED_HYP_DIR,
        LEADERBOARD_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)
    ensure_leaderboard_header()


def resolve_python_executable(requested: Path) -> Path:
    if requested.is_file():
        return requested
    for candidate in FALLBACK_PYTHONS:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"Requested python not found: {requested}. No fallback interpreter is available."
    )


def ensure_leaderboard_header() -> None:
    if LEADERBOARD_TSV.exists():
        return
    LEADERBOARD_TSV.write_text(
        "timestamp\tcandidate_id\tstage\tmetric\tstatus\tdecision\tbaseline_source\toutput_dir\tsnapshot\tdescription\n",
        encoding="utf-8",
    )


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in YAML file: {path}")
    return data


def save_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False, allow_unicode=True)


def load_candidate_spec(path: Path) -> CandidateSpec:
    data = load_yaml(path)
    candidate_id = data.get("candidate_id")
    if not candidate_id:
        raise ValueError(f"Missing candidate_id in spec: {path}")
    description = data.get("description", "").strip() or "candidate run"
    cli_overrides = data.get("cli_overrides") or {}
    hyp_overrides = data.get("hyp_overrides") or {}
    if not isinstance(cli_overrides, dict):
        raise ValueError("cli_overrides must be a mapping")
    if not isinstance(hyp_overrides, dict):
        raise ValueError("hyp_overrides must be a mapping")
    base_hyp = Path(data.get("base_hyp", DEFAULT_HYP))
    if not base_hyp.is_absolute():
        base_hyp = (ROOT / base_hyp).resolve()
    return CandidateSpec(
        candidate_id=str(candidate_id),
        description=description,
        cli_overrides=cli_overrides,
        hyp_overrides=hyp_overrides,
        base_hyp=base_hyp,
    )


def write_generated_hyp(spec: CandidateSpec, stage: str) -> Path | None:
    if not spec.hyp_overrides and spec.base_hyp.resolve() == DEFAULT_HYP.resolve():
        return None
    base = load_yaml(spec.base_hyp)
    base.update(spec.hyp_overrides)
    hyp_path = GENERATED_HYP_DIR / f"{spec.candidate_id}_{stage}.yaml"
    save_yaml(hyp_path, base)
    return hyp_path


def normalize_cli_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def build_train_command(
    *,
    stage: str,
    candidate_id: str,
    python_executable: Path,
    data_path: Path,
    weights_path: Path,
    device: str,
    seed: int,
    cli_overrides: dict[str, Any],
    hyp_path: Path | None,
) -> tuple[list[str], Path]:
    project_dir = RUNS_ROOT / stage
    save_dir = next_increment_path(project_dir / candidate_id)
    command = [
        str(python_executable),
        str(ROOT / "train.py"),
        "--data",
        str(data_path),
        "--weights",
        str(weights_path),
        "--epochs",
        str(stage_epochs(stage)),
        "--batch-size",
        str(cli_overrides.get("batch_size", 4)),
        "--imgsz",
        str(cli_overrides.get("imgsz", 640)),
        "--device",
        device,
        "--seed",
        str(seed),
        "--project",
        str(project_dir),
        "--name",
        save_dir.name,
        "--workers",
        str(cli_overrides.get("workers", 8)),
    ]
    if cli_overrides.get("optimizer"):
        command.extend(["--optimizer", str(cli_overrides["optimizer"])])
    if cli_overrides.get("patience") is not None:
        command.extend(["--patience", str(cli_overrides["patience"])])
    if cli_overrides.get("label_smoothing") is not None:
        command.extend(["--label-smoothing", str(cli_overrides["label_smoothing"])])
    if cli_overrides.get("freeze"):
        command.append("--freeze")
        command.extend(str(item) for item in cli_overrides["freeze"])
    for flag_key, flag_name in (
        ("cos_lr", "--cos-lr"),
        ("rect", "--rect"),
        ("cache", "--cache"),
        ("quad", "--quad"),
        ("noplots", "--noplots"),
        ("noval", "--noval"),
        ("nosave", "--nosave"),
        ("noautoanchor", "--noautoanchor"),
        ("multi_scale", "--multi-scale"),
        ("image_weights", "--image-weights"),
    ):
        value = cli_overrides.get(flag_key)
        if isinstance(value, bool):
            if value:
                command.append(flag_name)
        elif value is not None:
            command.extend([flag_name, normalize_cli_value(value)])
    if hyp_path is not None:
        command.extend(["--hyp", str(hyp_path)])
    return command, save_dir


def next_increment_path(path: Path) -> Path:
    if not path.exists():
        return path
    for index in range(2, 10000):
        candidate = Path(f"{path}{index}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Failed to allocate unique run path under {path}")


def extract_map50(results_csv: Path) -> float | None:
    if not results_csv.is_file():
        return None
    with results_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    if not rows:
        return None
    last_row = {key.strip(): value.strip() for key, value in rows[-1].items() if key}
    for key in METRIC_KEYS:
        if key in last_row and last_row[key]:
            return float(last_row[key])
    return None


def read_reference_metric() -> tuple[float | None, str]:
    if CHAMPION_YAML.is_file():
        champion = load_yaml(CHAMPION_YAML)
        metric = champion.get("metric")
        if metric is not None:
            return float(metric), str(champion.get("candidate_id", "champion"))
    if BASELINE_YAML.is_file():
        baseline = load_yaml(BASELINE_YAML)
        metric = baseline.get("metric")
        if metric is not None:
            return float(metric), str(baseline.get("candidate_id", "baseline"))
    return None, "none"


def decide_result(stage: str, metric: float | None, reference_metric: float | None, success: bool) -> tuple[str, str]:
    if not success:
        return "crash", "discard"
    if stage == "baseline":
        return "success", "keep"
    if stage == "smoke":
        return "success", "keep"
    if metric is None:
        return "crash", "discard"
    if reference_metric is None or metric > reference_metric:
        return "success", "keep"
    return "success", "discard"


def append_history(entry: dict[str, Any]) -> None:
    ensure_leaderboard_header()
    line = "\t".join(
        [
            str(entry["timestamp"]),
            str(entry["candidate_id"]),
            str(entry["stage"]),
            "" if entry["metric"] is None else f"{entry['metric']:.6f}",
            str(entry["status"]),
            str(entry["decision"]),
            str(entry["baseline_source"]),
            str(entry["output_dir"]),
            str(entry["snapshot"]),
            str(entry["description"]).replace("\t", " "),
        ]
    )
    with LEADERBOARD_TSV.open("a", encoding="utf-8") as handle:
        handle.write(f"{line}\n")


def update_reference_files(
    outcome: RunOutcome,
    *,
    candidate_id: str,
    description: str,
    baseline_source: str,
    stage: str,
) -> None:
    payload = {
        "candidate_id": candidate_id,
        "metric": outcome.metric,
        "status": outcome.status,
        "decision": outcome.decision,
        "save_dir": str(outcome.save_dir),
        "snapshot": str(outcome.snapshot_path),
        "description": description,
        "baseline_source": baseline_source,
        "updated_at": timestamp(),
    }
    if stage == "baseline":
        save_yaml(BASELINE_YAML, payload)
        save_yaml(CHAMPION_YAML, payload)
        return
    if stage == "sprint" and outcome.decision == "keep":
        save_yaml(CHAMPION_YAML, payload)


def timestamp() -> str:
    return datetime.now().isoformat(timespec="seconds")


def execute_stage(
    *,
    stage: str,
    spec: CandidateSpec,
    python_executable: Path,
    data_path: Path,
    weights_path: Path,
    device: str,
    seed: int,
) -> RunOutcome:
    ensure_layout()
    generated_hyp = write_generated_hyp(spec, stage)
    command, save_dir = build_train_command(
        stage=stage,
        candidate_id=spec.candidate_id,
        python_executable=python_executable,
        data_path=data_path,
        weights_path=weights_path,
        device=device,
        seed=seed,
        cli_overrides=spec.cli_overrides,
        hyp_path=generated_hyp,
    )
    log_path = save_dir / "run.log"
    reference_metric, baseline_source = read_reference_metric()
    snapshot_payload = {
        "candidate_id": spec.candidate_id,
        "description": spec.description,
        "stage": stage,
        "timestamp": timestamp(),
        "baseline_source": baseline_source,
        "compared_against": reference_metric,
        "data": str(data_path),
        "weights": str(weights_path),
        "base_hyp": str(spec.base_hyp),
        "generated_hyp": str(generated_hyp) if generated_hyp else None,
        "cli_overrides": spec.cli_overrides,
        "hyp_overrides": spec.hyp_overrides,
        "output_dir": str(save_dir),
        "log_path": str(log_path),
        "command": command,
        "status": "pending",
        "decision": "pending",
        "metric": None,
        "returncode": None,
    }
    snapshot_path = SNAPSHOT_DIR / f"{spec.candidate_id}_{stage}.yaml"
    save_yaml(snapshot_path, snapshot_payload)

    save_dir.mkdir(parents=True, exist_ok=True)
    returncode: int | None = None
    try:
        with log_path.open("w", encoding="utf-8") as handle:
            completed = subprocess.run(
                command,
                cwd=ROOT,
                stdout=handle,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
        returncode = completed.returncode
    except subprocess.TimeoutExpired:
        returncode = None

    metric = extract_map50(save_dir / "results.csv")
    success = returncode == 0 and metric is not None
    status, decision = decide_result(stage, metric, reference_metric, success)
    snapshot_payload.update(
        {
            "status": status,
            "decision": decision,
            "metric": metric,
            "returncode": returncode,
            "completed_at": timestamp(),
        }
    )
    save_yaml(snapshot_path, snapshot_payload)
    outcome = RunOutcome(
        status=status,
        decision=decision,
        metric=metric,
        compared_against=reference_metric,
        save_dir=save_dir,
        snapshot_path=snapshot_path,
        log_path=log_path,
        command=command,
        returncode=returncode,
    )
    append_history(
        {
            "timestamp": snapshot_payload["completed_at"],
            "candidate_id": spec.candidate_id,
            "stage": stage,
            "metric": metric,
            "status": status,
            "decision": decision,
            "baseline_source": baseline_source,
            "output_dir": save_dir,
            "snapshot": snapshot_path,
            "description": spec.description,
        }
    )
    update_reference_files(
        outcome,
        candidate_id=spec.candidate_id,
        description=spec.description,
        baseline_source=baseline_source,
        stage=stage,
    )
    return outcome


def make_baseline_spec() -> CandidateSpec:
    return CandidateSpec(
        candidate_id="baseline_10ep",
        description="manual baseline short run",
        cli_overrides={"batch_size": 4, "imgsz": 640, "workers": 8, "noplots": True},
        hyp_overrides={},
        base_hyp=DEFAULT_HYP,
    )


def print_outcome(outcome: RunOutcome) -> None:
    print(f"stage={outcome.snapshot_path.stem}")
    print(f"status={outcome.status}")
    print(f"decision={outcome.decision}")
    print(f"metric={'' if outcome.metric is None else f'{outcome.metric:.6f}'}")
    print(f"save_dir={outcome.save_dir}")
    print(f"snapshot={outcome.snapshot_path}")
    print(f"log={outcome.log_path}")


def run_candidate(args: argparse.Namespace) -> int:
    python_executable = resolve_python_executable(Path(args.python))
    spec = load_candidate_spec(Path(args.spec))
    smoke = execute_stage(
        stage="smoke",
        spec=spec,
        python_executable=python_executable,
        data_path=Path(args.data),
        weights_path=Path(args.weights),
        device=args.device,
        seed=args.seed,
    )
    print_outcome(smoke)
    if smoke.decision != "keep":
        return 1
    sprint = execute_stage(
        stage="sprint",
        spec=spec,
        python_executable=python_executable,
        data_path=Path(args.data),
        weights_path=Path(args.weights),
        device=args.device,
        seed=args.seed,
    )
    print_outcome(sprint)
    return 0 if sprint.decision in {"keep", "discard"} else 1


def run_baseline(args: argparse.Namespace) -> int:
    python_executable = resolve_python_executable(Path(args.python))
    outcome = execute_stage(
        stage="baseline",
        spec=make_baseline_spec(),
        python_executable=python_executable,
        data_path=Path(args.data),
        weights_path=Path(args.weights),
        device=args.device,
        seed=args.seed,
    )
    print_outcome(outcome)
    return 0 if outcome.decision == "keep" else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="YOLOv5 AutoResearch Phase 1 runner")
    parser.add_argument("--python", default=str(DEFAULT_PYTHON), help="Python executable inside the yolo env")
    parser.add_argument("--data", default=str(DEFAULT_DATA), help="Dataset YAML path")
    parser.add_argument("--weights", default=str(DEFAULT_WEIGHTS), help="Initial weights path")
    parser.add_argument("--device", default=DEFAULT_DEVICE, help="Training device")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Global seed")
    subparsers = parser.add_subparsers(dest="command", required=True)

    baseline = subparsers.add_parser("baseline", help="Run the fixed 10-epoch baseline")
    baseline.set_defaults(handler=run_baseline)

    candidate = subparsers.add_parser("candidate", help="Run one candidate through smoke and sprint")
    candidate.add_argument("--spec", required=True, help="Candidate spec YAML path")
    candidate.set_defaults(handler=run_candidate)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    resolved = resolve_python_executable(Path(args.python))
    if resolved != Path(args.python):
        print(f"warning=python_fallback requested={args.python} resolved={resolved}")
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
