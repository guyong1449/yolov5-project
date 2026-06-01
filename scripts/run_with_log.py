#!/usr/bin/env python3
"""Run a command while mirroring stdout/stderr to a log file and a Markdown view.

Adapted from Spectralmae ``scripts/run_with_log.py`` for this YOLOv5 repo.

Typical usage::

    python scripts/run_with_log.py --name smoke_train -- \\
        python train.py --data data/dataAirVis.yaml ...

The Markdown file is meant for in-editor viewing (Cursor/VS Code preview).
Raw text is also written to a ``.log`` sibling file.

Examples::

    python scripts/run_with_log.py --name val_check -- \\
        python val.py --data data/dataAirVis.yaml ...

    python scripts/run_with_log.py -l runs/logs/custom.log --append -- \\
        python detect.py --weights checkpoint/yolov5_best.pt --source bus.jpg
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import IO, Iterable, List, Optional, Sequence, Union

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOG_DIR = REPO_ROOT / "runs" / "logs"


def _command_line(command: Union[str, Sequence[str]]) -> str:
    if isinstance(command, str):
        return command
    return " ".join(shlex.quote(part) for part in command)


def _command_parts(command: Union[str, Sequence[str], None]) -> list[str]:
    if command is None:
        return []
    if isinstance(command, str):
        return shlex.split(command)
    return [str(part) for part in command]


def _option_value(parts: Sequence[str], option: str) -> Optional[str]:
    for index, part in enumerate(parts):
        if part == option and index + 1 < len(parts):
            return parts[index + 1]
        if part.startswith(f"{option}="):
            return part.split("=", 1)[1]
    return None


def _train_run_dir(command: Union[str, Sequence[str], None]) -> Optional[Path]:
    parts = _command_parts(command)
    train_entries = {"train.py", "npu_ddp_training_benchmark.sh"}
    if not any(Path(part).name.lower() in train_entries for part in parts):
        return None

    project = _option_value(parts, "--project") or os.environ.get("PROJECT_DIR")
    name = _option_value(parts, "--name") or os.environ.get("RUN_NAME")
    if not project or not name:
        return None

    project_path = Path(project)
    if not project_path.is_absolute():
        project_path = REPO_ROOT / project_path

    if project_path.resolve() != (REPO_ROOT / "runs" / "train").resolve():
        return None

    return project_path / name


def _is_train_command(command: Union[str, Sequence[str], None]) -> bool:
    train_entries = {"train.py", "npu_ddp_training_benchmark.sh"}
    return any(Path(part).name.lower() in train_entries for part in _command_parts(command))


def _is_detect_command(command: Union[str, Sequence[str], None]) -> bool:
    detect_entries = {"detect.py", "npu_video_benchmark.sh", "npu_ddp_detect_benchmark.sh"}
    return any(Path(part).name.lower() in detect_entries for part in _command_parts(command))


def _timing_mode(command: Union[str, Sequence[str], None]) -> str:
    device = (_option_value(_command_parts(command), "--device") or os.environ.get("DEVICE") or "").strip().lower()
    return "synced_speed" if device.startswith("npu") else "wall_clock_primary"


def _timing_log_path(log_path: Path) -> Path:
    return log_path.with_name(f"{log_path.stem}_inference_time.txt")


def _training_summary_log_path(log_path: Path) -> Path:
    return log_path.with_name(f"{log_path.stem}_ddp_training_summary.txt")


def _parallel_inference_summary_log_path(log_path: Path) -> Path:
    return log_path.with_name(f"{log_path.stem}_parallel_inference_summary.txt")


def _is_parallel_detect_command(command: Union[str, Sequence[str], None]) -> bool:
    parts = _command_parts(command)
    return _is_detect_command(command) and ("--ddp-infer" in parts or "npu_ddp_detect_benchmark.sh" in {Path(part).name.lower() for part in parts})


def _ddp_summary(command: Union[str, Sequence[str]], output_lines: Sequence[str]) -> list[str]:
    parts = _command_parts(command)
    device = (_option_value(parts, "--device") or os.environ.get("DEVICE") or "").strip()
    world_size = os.environ.get("WORLD_SIZE", "") or os.environ.get("NPROC_PER_NODE", "")
    if not world_size:
        for index, part in enumerate(parts):
            if part == "--nproc_per_node" and index + 1 < len(parts):
                world_size = parts[index + 1]
                break
            if part.startswith("--nproc_per_node="):
                world_size = part.split("=", 1)[1]
                break

    backend = ""
    rank_lines = []
    epochs_completed = ""
    for line in output_lines:
        if line.startswith("DDP init:"):
            rank_lines.append(line)
            if "backend=" in line and not backend:
                backend = line.split("backend=", 1)[1].split()[0]
            if "world_size=" in line and not world_size:
                world_size = line.split("world_size=", 1)[1].split()[0]
            if "device=" in line and not device:
                device = line.split("device=", 1)[1].split()[0]
        if "epochs completed in" in line:
            epochs_completed = line

    rank_ids = set()
    for line in rank_lines:
        if "local_rank=" in line:
            rank_ids.add(line.split("local_rank=", 1)[1].split()[0])

    expected_world_size = int(world_size) if world_size.isdigit() else None
    rank_coverage_ok = len(rank_ids) == expected_world_size if expected_world_size else bool(rank_lines)
    ddp_confirmed = bool(rank_lines and backend == "hccl" and epochs_completed)
    return [
        f"world_size={world_size or 'unknown'}",
        f"backend={backend or 'unknown'}",
        f"device={device or 'unknown'}",
        f"rank_lines={len(rank_lines)}",
        f"rank_coverage_ok={'true' if rank_coverage_ok else 'false'}",
        f"ddp_confirmed={'true' if ddp_confirmed else 'false'}",
        f"epochs_completed={epochs_completed}",
    ]


def _parallel_detect_summary(command: Union[str, Sequence[str]], output_lines: Sequence[str]) -> list[str]:
    parts = _command_parts(command)
    device = (_option_value(parts, "--device") or os.environ.get("DEVICE") or "").strip()
    source = (_option_value(parts, "--source") or os.environ.get("SOURCE_PATH") or "").strip()
    world_size = os.environ.get("WORLD_SIZE", "") or os.environ.get("NPROC_PER_NODE", "")
    if not world_size:
        for index, part in enumerate(parts):
            if part == "--nproc_per_node" and index + 1 < len(parts):
                world_size = parts[index + 1]
                break
            if part.startswith("--nproc_per_node="):
                world_size = part.split("=", 1)[1]
                break

    init_ranks = set()
    done_counts = {}
    done_batches = {}
    aggregate_frames = ""
    rank_frame_counts = ""
    speed_line = ""
    infer_mode = ""
    buffer_size = ""
    batch_count = ""
    tail_batch_size = ""
    for line in output_lines:
        if line.startswith("INFER init:"):
            if "rank=" in line:
                init_ranks.add(line.split("rank=", 1)[1].split()[0])
            if "device=" in line and not device:
                device = line.split("device=", 1)[1].split()[0]
            if "source=" in line and not source:
                source = line.split("source=", 1)[1].split()[0]
            if "world_size=" in line and not world_size:
                world_size = line.split("world_size=", 1)[1].split()[0]
            if "infer_mode=" in line and not infer_mode:
                infer_mode = line.split("infer_mode=", 1)[1].split()[0]
            if "buffer_size=" in line and not buffer_size:
                buffer_size = line.split("buffer_size=", 1)[1].split()[0]
        elif line.startswith("INFER done:") and "rank=" in line and "processed_frames=" in line:
            rank = line.split("rank=", 1)[1].split()[0]
            count = line.split("processed_frames=", 1)[1].split()[0]
            done_counts[rank] = count
            if "processed_batches=" in line:
                done_batches[rank] = line.split("processed_batches=", 1)[1].split()[0]
            if "infer_mode=" in line and not infer_mode:
                infer_mode = line.split("infer_mode=", 1)[1].split()[0]
            if "buffer_size=" in line and not buffer_size:
                buffer_size = line.split("buffer_size=", 1)[1].split()[0]
        elif line.startswith("INFER aggregate:"):
            if "rank_frame_counts=" in line:
                rank_frame_counts = line.split("rank_frame_counts=", 1)[1].split()[0]
            if "aggregate_frames=" in line:
                aggregate_frames = line.split("aggregate_frames=", 1)[1].split()[0]
            if "world_size=" in line and not world_size:
                world_size = line.split("world_size=", 1)[1].split()[0]
            if "infer_mode=" in line and not infer_mode:
                infer_mode = line.split("infer_mode=", 1)[1].split()[0]
            if "buffer_size=" in line and not buffer_size:
                buffer_size = line.split("buffer_size=", 1)[1].split()[0]
            if "batch_count=" in line:
                batch_count = line.split("batch_count=", 1)[1].split()[0]
            if "tail_batch_size=" in line:
                tail_batch_size = line.split("tail_batch_size=", 1)[1].split()[0]
        elif "Speed:" in line:
            speed_line = line

    expected_world_size = int(world_size) if world_size.isdigit() else None
    all_ranks_seen = len(init_ranks) == expected_world_size if expected_world_size else bool(init_ranks)
    nonzero_ranks = sorted(rank for rank, count in done_counts.items() if count.isdigit() and int(count) > 0)
    parsed_aggregate_counts = {}
    if rank_frame_counts:
        for pair in rank_frame_counts.split(','):
            if ':' not in pair:
                continue
            rank, count = pair.split(':', 1)
            parsed_aggregate_counts[rank] = count
    aggregate_nonzero_ranks = sorted(
        rank for rank, count in parsed_aggregate_counts.items() if count.isdigit() and int(count) > 0
    )
    all_nonzero = len(nonzero_ranks) == expected_world_size if expected_world_size else bool(nonzero_ranks)
    aggregate_all_nonzero = (
        len(aggregate_nonzero_ranks) == expected_world_size if expected_world_size else bool(aggregate_nonzero_ranks)
    )
    parallel_infer_confirmed = bool(aggregate_frames) and (
        (all_ranks_seen and all_nonzero) or aggregate_all_nonzero
    )
    ordered_counts = ",".join(f"{rank}:{done_counts[rank]}" for rank in sorted(done_counts, key=int))
    ordered_batches = ",".join(f"{rank}:{done_batches[rank]}" for rank in sorted(done_batches, key=int))

    return [
        f"world_size={world_size or 'unknown'}",
        f"device={device or 'unknown'}",
        f"source={source or 'unknown'}",
        f"infer_mode={infer_mode or 'mod'}",
        f"buffer_size={buffer_size or '0'}",
        f"rank_lines={len(init_ranks)}",
        f"rank_done_counts={ordered_counts}",
        f"rank_batch_counts={ordered_batches}",
        f"rank_frame_counts={rank_frame_counts or ordered_counts}",
        f"aggregate_frames={aggregate_frames or 'unknown'}",
        f"batch_count={batch_count or '0'}",
        f"tail_batch_size={tail_batch_size or '0'}",
        f"parallel_infer_confirmed={'true' if parallel_infer_confirmed else 'false'}",
        f"speed_line={speed_line}",
    ]


def _write_timing_sidecar(
    command: Union[str, Sequence[str]],
    *,
    log_path: Path,
    start_time: datetime,
    end_time: datetime,
    elapsed_seconds: float,
    output_lines: Sequence[str],
) -> None:
    if not _is_detect_command(command):
        return

    parts = _command_parts(command)
    run_name = _option_value(parts, "--name") or log_path.stem
    speed_line = next((line for line in reversed(output_lines) if "Speed:" in line), "")
    content = [
        f"run_name={run_name}",
        f"start_time={start_time:%Y-%m-%d %H:%M:%S}",
        f"end_time={end_time:%Y-%m-%d %H:%M:%S}",
        f"wall_clock_seconds={elapsed_seconds:.6f}",
        f"timing_mode={_timing_mode(command)}",
        f"speed_line={speed_line}",
    ]
    _timing_log_path(log_path).write_text("\n".join(content) + "\n", encoding="utf-8")


def _write_training_sidecar(
    command: Union[str, Sequence[str]],
    *,
    log_path: Path,
    start_time: datetime,
    end_time: datetime,
    elapsed_seconds: float,
    output_lines: Sequence[str],
) -> None:
    if not _is_train_command(command):
        return

    parts = _command_parts(command)
    run_name = _option_value(parts, "--name") or os.environ.get("RUN_NAME") or log_path.stem
    content = [
        f"run_name={run_name}",
        f"start_time={start_time:%Y-%m-%d %H:%M:%S}",
        f"end_time={end_time:%Y-%m-%d %H:%M:%S}",
        f"wall_clock_seconds={elapsed_seconds:.6f}",
    ]
    content.extend(_ddp_summary(command, output_lines))
    _training_summary_log_path(log_path).write_text("\n".join(content) + "\n", encoding="utf-8")


def _write_parallel_detect_sidecar(
    command: Union[str, Sequence[str]],
    *,
    log_path: Path,
    start_time: datetime,
    end_time: datetime,
    elapsed_seconds: float,
    output_lines: Sequence[str],
) -> None:
    if not _is_parallel_detect_command(command):
        return

    parts = _command_parts(command)
    run_name = _option_value(parts, "--name") or os.environ.get("RUN_NAME") or log_path.stem
    content = [
        f"run_name={run_name}",
        f"start_time={start_time:%Y-%m-%d %H:%M:%S}",
        f"end_time={end_time:%Y-%m-%d %H:%M:%S}",
        f"wall_clock_seconds={elapsed_seconds:.6f}",
    ]
    content.extend(_parallel_detect_summary(command, output_lines))
    _parallel_inference_summary_log_path(log_path).write_text("\n".join(content) + "\n", encoding="utf-8")


def _resolve_paths(
    log_file: Optional[Path],
    md_file: Optional[Path],
    name: Optional[str],
    *,
    command: Union[str, Sequence[str], None] = None,
) -> tuple[Path, Path]:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = name or f"run_{stamp}"
    task_dir = DEFAULT_LOG_DIR / f"{stem}_{stamp}"
    train_run_dir = _train_run_dir(command)

    if log_file is None:
        log_file = (train_run_dir / "run.log") if train_run_dir else (task_dir / "run.log")
    else:
        log_file = Path(log_file)
        if not log_file.is_absolute():
            log_file = REPO_ROOT / log_file

    if md_file is None:
        md_file = log_file.with_suffix(".md")
    else:
        md_file = Path(md_file)
        if not md_file.is_absolute():
            md_file = REPO_ROOT / md_file

    return log_file, md_file


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def _print_realtime_line(line: str) -> None:
    """Print a line to the active console without crashing on narrow encodings."""
    try:
        print(line)
        return
    except UnicodeEncodeError:
        pass

    encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
    safe_line = line.encode(encoding, errors="replace").decode(encoding, errors="replace")
    print(safe_line)


class MarkdownLogMirror:
    """Write plain log lines and keep a Markdown wrapper in sync."""

    def __init__(
        self,
        log_path: Path,
        md_path: Path,
        command: Union[str, Sequence[str]],
        *,
        append: bool = False,
        cwd: Optional[Path] = None,
    ) -> None:
        self.log_path = log_path
        self.md_path = md_path
        self.command = command
        self.append = append
        self.cwd = cwd
        self.start_time = datetime.now()
        self._lines: List[str] = []
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.md_path.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if append else "w"
        self._log_fp: IO[str] = open(self.log_path, mode, encoding="utf-8", newline="\n")
        if append and self.log_path.stat().st_size > 0:
            self._log_fp.write("\n")
        self._write_header()

    def _write_header(self) -> None:
        if not self.append:
            banner = "=" * 80
            self._log_fp.write(f"{banner}\n")
            self._log_fp.write(f"Command: {_command_line(self.command)}\n")
            if self.cwd is not None:
                self._log_fp.write(f"CWD: {self.cwd}\n")
            self._log_fp.write(f"Start: {self.start_time:%Y-%m-%d %H:%M:%S}\n")
            self._log_fp.write(f"{banner}\n\n")
            self._log_fp.flush()
        self._flush_markdown()

    def write_line(self, line: str) -> None:
        self._lines.append(line)
        self._log_fp.write(line + "\n")
        self._log_fp.flush()
        self._flush_markdown()

    def close(self, return_code: int, *, interrupted: bool = False) -> None:
        end_time = datetime.now()
        banner = "=" * 80
        self._log_fp.write(f"\n{banner}\n")
        self._log_fp.write(f"End: {end_time:%Y-%m-%d %H:%M:%S}\n")
        if interrupted:
            self._log_fp.write("Status: interrupted\n")
        self._log_fp.write(f"Return code: {return_code}\n")
        self._log_fp.write(f"{banner}\n")
        self._log_fp.flush()
        self._log_fp.close()
        self._flush_markdown(return_code=return_code, end_time=end_time, interrupted=interrupted)

    def _flush_markdown(
        self,
        *,
        return_code: Optional[int] = None,
        end_time: Optional[datetime] = None,
        interrupted: bool = False,
    ) -> None:
        status = "running"
        if interrupted:
            status = "interrupted"
        elif return_code is not None:
            status = "ok" if return_code == 0 else f"failed ({return_code})"

        body = "\n".join(self._lines)
        md_parts = [
            "# Terminal log",
            "",
            "| Field | Value |",
            "| --- | --- |",
            f"| Status | `{status}` |",
            f"| Command | `{_command_line(self.command)}` |",
            f"| CWD | `{self.cwd or Path.cwd()}` |",
            f"| Start | {self.start_time:%Y-%m-%d %H:%M:%S} |",
            f"| Log file | `{_rel(self.log_path)}` |",
            "",
            "> Open this file in the editor and keep it visible while the job runs.",
            "> Raw text is also mirrored to the ``.log`` file beside this Markdown file.",
            "",
            "## Output",
            "",
            "```text",
            body,
            "```",
            "",
        ]
        if return_code is not None:
            md_parts.extend(
                [
                    "## Run footer",
                    "",
                    f"- End: {end_time:%Y-%m-%d %H:%M:%S}" if end_time else "",
                    f"- Return code: `{return_code}`",
                    f"- Interrupted: `{interrupted}`",
                    "",
                ]
            )
        self.md_path.write_text("\n".join(md_parts), encoding="utf-8")


def run_with_log(
    command: Union[str, Sequence[str]],
    *,
    log_file: Optional[Path] = None,
    md_file: Optional[Path] = None,
    name: Optional[str] = None,
    append: bool = False,
    realtime: bool = True,
    cwd: Optional[Path] = None,
) -> int:
    """Execute *command* and mirror merged stdout/stderr to log + Markdown files."""
    log_path, md_path = _resolve_paths(log_file, md_file, name, command=command)
    workdir = Path(cwd) if cwd else None
    start_monotonic = time.perf_counter()
    mirror = MarkdownLogMirror(
        log_path,
        md_path,
        command,
        append=append,
        cwd=workdir,
    )

    print(f"[run_with_log] log:  {_rel(log_path)}")
    print(f"[run_with_log] view: {_rel(md_path)}")

    popen_kwargs = {
        "stdout": subprocess.PIPE,
        "stderr": subprocess.STDOUT,
        "universal_newlines": True,
        "bufsize": 1,
        "encoding": "utf-8",
        "errors": "replace",
    }
    if workdir is not None:
        popen_kwargs["cwd"] = str(workdir)

    if isinstance(command, str):
        process = subprocess.Popen(command, shell=True, **popen_kwargs)
    else:
        process = subprocess.Popen(list(command), shell=False, **popen_kwargs)

    try:
        assert process.stdout is not None
        for line in process.stdout:
            line = line.rstrip("\r\n")
            mirror.write_line(line)
            if realtime:
                _print_realtime_line(line)
        return_code = process.wait()
        process.stdout.close()
        mirror.close(return_code)
        _write_timing_sidecar(
            command,
            log_path=log_path,
            start_time=mirror.start_time,
            end_time=datetime.now(),
            elapsed_seconds=time.perf_counter() - start_monotonic,
            output_lines=mirror._lines,
        )
        _write_training_sidecar(
            command,
            log_path=log_path,
            start_time=mirror.start_time,
            end_time=datetime.now(),
            elapsed_seconds=time.perf_counter() - start_monotonic,
            output_lines=mirror._lines,
        )
        _write_parallel_detect_sidecar(
            command,
            log_path=log_path,
            start_time=mirror.start_time,
            end_time=datetime.now(),
            elapsed_seconds=time.perf_counter() - start_monotonic,
            output_lines=mirror._lines,
        )
        return return_code
    except KeyboardInterrupt:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        mirror.close(130, interrupted=True)
        _write_timing_sidecar(
            command,
            log_path=log_path,
            start_time=mirror.start_time,
            end_time=datetime.now(),
            elapsed_seconds=time.perf_counter() - start_monotonic,
            output_lines=mirror._lines,
        )
        _write_training_sidecar(
            command,
            log_path=log_path,
            start_time=mirror.start_time,
            end_time=datetime.now(),
            elapsed_seconds=time.perf_counter() - start_monotonic,
            output_lines=mirror._lines,
        )
        _write_parallel_detect_sidecar(
            command,
            log_path=log_path,
            start_time=mirror.start_time,
            end_time=datetime.now(),
            elapsed_seconds=time.perf_counter() - start_monotonic,
            output_lines=mirror._lines,
        )
        return 130


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a command and mirror terminal output to .log and .md files.",
    )
    parser.add_argument(
        "--log",
        "--log-file",
        "-l",
        type=str,
        default=None,
        help="Plain-text log path (default: runs/logs/<name>_<timestamp>/run.log).",
    )
    parser.add_argument(
        "--md",
        "--md-file",
        type=str,
        default=None,
        help="Markdown view path (default: same stem as --log with .md suffix).",
    )
    parser.add_argument(
        "--name",
        "-n",
        type=str,
        default=None,
        help="Run name used when --log is omitted (default: run_<timestamp>, grouped under one task folder).",
    )
    parser.add_argument("--append", "-a", action="store_true", help="Append to existing log files.")
    parser.add_argument("--no-realtime", action="store_true", help="Do not echo output to the terminal.")
    parser.add_argument(
        "--cwd",
        type=str,
        default=str(REPO_ROOT),
        help=f"Working directory for the child process (default: {REPO_ROOT}).",
    )
    parser.add_argument(
        "command",
        nargs=argparse.REMAINDER,
        help="Command after ``--`` separator, e.g. -- python train.py ...",
    )
    args = parser.parse_args(argv)
    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    return args


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    if not args.command:
        print("error: missing command; pass it after `--`", file=sys.stderr)
        return 2

    command: Union[str, Sequence[str]]
    if len(args.command) == 1 and " " in args.command[0]:
        command = args.command[0]
    else:
        command = args.command

    cwd = Path(args.cwd)
    code = run_with_log(
        command,
        log_file=Path(args.log) if args.log else None,
        md_file=Path(args.md) if args.md else None,
        name=args.name,
        append=args.append,
        realtime=not args.no_realtime,
        cwd=cwd,
    )
    return code


if __name__ == "__main__":
    raise SystemExit(main())
