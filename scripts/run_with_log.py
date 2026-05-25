#!/usr/bin/env python3
"""Run a command while mirroring stdout/stderr to a log file and a Markdown view.

Adapted from Spectralmae ``scripts/run_with_log.py`` for this YOLOv5 repo.

Typical usage::

    python scripts/run_with_log.py --name smoke_train -- \\
        D:/Miniconda3/python.exe train.py --data data/dataAirVis.yaml ...

The Markdown file is meant for in-editor viewing (Cursor/VS Code preview).
Raw text is also written to a ``.log`` sibling file.

Examples::

    python scripts/run_with_log.py --name val_check -- \\
        D:/Miniconda3/python.exe val.py --data data/dataAirVis.yaml ...

    python scripts/run_with_log.py -l runs/logs/custom.log --append -- \\
        python detect.py --weights checkpoint/yolov5_best.pt --source bus.jpg
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
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
    if not any(Path(part).name.lower() == "train.py" for part in parts):
        return None

    project = _option_value(parts, "--project")
    name = _option_value(parts, "--name")
    if not project or not name:
        return None

    project_path = Path(project)
    if not project_path.is_absolute():
        project_path = REPO_ROOT / project_path

    if project_path.resolve() != (REPO_ROOT / "runs" / "train").resolve():
        return None

    return project_path / name


def _resolve_paths(
    log_file: Optional[Path],
    md_file: Optional[Path],
    name: Optional[str],
    *,
    command: Union[str, Sequence[str], None] = None,
) -> tuple[Path, Path]:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = name or f"run_{stamp}"

    if log_file is None:
        train_run_dir = _train_run_dir(command)
        if train_run_dir is not None:
            log_file = train_run_dir / "run.log"
        else:
            log_file = DEFAULT_LOG_DIR / f"{stem}.log"
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
        return return_code
    except KeyboardInterrupt:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        mirror.close(130, interrupted=True)
        return 130


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a command and mirror terminal output to .log and .md files.",
    )
    parser.add_argument(
        "--log",
        "-l",
        type=str,
        default=None,
        help="Plain-text log path (default: train.py writes to runs/train/<name>/run.log, otherwise runs/logs/<name>.log).",
    )
    parser.add_argument(
        "--md",
        type=str,
        default=None,
        help="Markdown view path (default: same stem as --log with .md suffix).",
    )
    parser.add_argument(
        "--name",
        "-n",
        type=str,
        default=None,
        help="Run name used when --log is omitted (default: run_<timestamp>).",
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
