from __future__ import annotations

import os
import subprocess
import threading
from collections import deque
from pathlib import Path
from typing import Iterator

from tools.gui_panel.schemas import RunRequest, RuntimeStateModel, iso_now


class ProcessRunner:
    def __init__(self, *, log_limit: int = 400) -> None:
        self._lock = threading.RLock()
        self._condition = threading.Condition(self._lock)
        self._process: subprocess.Popen[str] | None = None
        self._thread: threading.Thread | None = None
        self._stop_requested = False
        self._event_id = 0
        self._recent_logs: deque[str] = deque(maxlen=log_limit)
        self._state = RuntimeStateModel()

    def runtime_state(self) -> RuntimeStateModel:
        with self._lock:
            payload = self._state.model_dump()
            payload["recent_logs"] = list(self._recent_logs)
            return RuntimeStateModel(**payload)

    def start(self, request: RunRequest) -> RuntimeStateModel:
        with self._lock:
            if self._process is not None and self._process.poll() is None:
                raise RuntimeError(f"Task already running: {self._state.active_task}")

            command = request.command_override or request.metadata.get("argv")
            if not command:
                raise RuntimeError("No command available to start")

            self._stop_requested = False
            self._recent_logs.clear()
            self._process = subprocess.Popen(
                list(command),
                cwd=str(Path(__file__).resolve().parents[3]),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )
            self._state = RuntimeStateModel(
                status="running",
                active_task=request.task_type,
                pid=self._process.pid,
                started_at=iso_now(),
                stop_requested=False,
                command_preview=request.metadata.get("command", ""),
                last_output_path=request.metadata.get("output_path", ""),
                recent_logs=[],
            )
            self._thread = threading.Thread(target=self._consume_output, name="gui-panel-runner", daemon=True)
            self._thread.start()
            self._condition.notify_all()
            return self.runtime_state()

    def stop(self) -> RuntimeStateModel:
        with self._lock:
            process = self._process
            thread = self._thread
            if process is None or process.poll() is not None:
                if self._state.status == "running":
                    self._state.status = "stopped"
                return self.runtime_state()
            self._stop_requested = True
            self._state.status = "stopping"
            self._state.stop_requested = True
            pid = process.pid

        if os.name == "nt":
            subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True, text=True)
        else:
            process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
        if thread is not None:
            thread.join(timeout=5)
        return self.runtime_state()

    def clear_logs(self) -> None:
        with self._lock:
            self._recent_logs.clear()
            self._event_id += 1
            self._condition.notify_all()

    def log_stream(self, *, since: int = 0, heartbeat_seconds: float = 15.0) -> Iterator[tuple[int, str | None]]:
        cursor = since
        while True:
            line: str | None = None
            with self._condition:
                has_event = self._condition.wait_for(lambda: self._event_id > cursor, timeout=heartbeat_seconds)
                if not has_event:
                    line = None
                if self._event_id > cursor:
                    cursor = self._event_id
                    line = self._recent_logs[-1] if self._recent_logs else ""
            yield cursor, line

    def _consume_output(self) -> None:
        assert self._process is not None
        process = self._process
        assert process.stdout is not None
        try:
            for raw_line in process.stdout:
                self._append_log(raw_line.rstrip("\r\n"))
            return_code = process.wait()
        finally:
            process.stdout.close()
        self._finalize(return_code)

    def _append_log(self, line: str) -> None:
        with self._condition:
            self._recent_logs.append(line)
            self._event_id += 1
            self._parse_result_line(line)
            self._condition.notify_all()

    def _parse_result_line(self, line: str) -> None:
        stripped = line.strip()
        if stripped.startswith("[run_with_log] log:"):
            self._state.last_log_path = stripped.split(":", 1)[1].strip()
        elif stripped.startswith("[run_with_log] view:"):
            self._state.last_view_path = stripped.split(":", 1)[1].strip()
        elif stripped.startswith("Results saved to"):
            self._state.last_output_path = stripped.split("Results saved to", 1)[1].strip()
        elif stripped.startswith("session_url="):
            self._state.last_session_url = stripped.split("=", 1)[1].strip()
        elif stripped.startswith("dataset_name="):
            self._state.last_dataset_name = stripped.split("=", 1)[1].strip()
        elif stripped.startswith("samples_count="):
            try:
                self._state.last_samples_count = int(stripped.split("=", 1)[1].strip())
            except ValueError:
                pass

    def _finalize(self, return_code: int) -> None:
        with self._condition:
            if self._stop_requested:
                status = "stopped"
            else:
                status = "succeeded" if return_code == 0 else "failed"
            self._state.status = status
            self._state.finished_at = iso_now()
            self._state.return_code = return_code
            self._state.stop_requested = self._stop_requested
            self._process = None
            self._thread = None
            self._event_id += 1
            self._condition.notify_all()
