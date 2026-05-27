from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


class SessionStore:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._state: dict[str, Any] = {
            "recent_values": {},
            "recent_commands": {},
            "recent_outputs": {},
            "recent_logs": {},
            "command_history": [],
        }
        self.load()

    def load(self) -> dict[str, Any]:
        if self.path.exists():
            self._state.update(json.loads(self.path.read_text(encoding="utf-8")))
        return self._state

    def save(self) -> None:
        self.path.write_text(json.dumps(self._state, indent=2, ensure_ascii=False), encoding="utf-8")

    @property
    def state(self) -> dict[str, Any]:
        return self._state

    def update_task(
        self,
        task_type: str,
        *,
        values: dict[str, Any] | None = None,
        command: str | None = None,
        output_path: str | None = None,
        log_path: str | None = None,
    ) -> None:
        if values is not None:
            self._state.setdefault("recent_values", {})[task_type] = values
        if command is not None:
            self._state.setdefault("recent_commands", {})[task_type] = command
        if output_path is not None:
            self._state.setdefault("recent_outputs", {})[task_type] = output_path
        if log_path is not None:
            self._state.setdefault("recent_logs", {})[task_type] = log_path
        self.save()

    def add_command_history(self, task_type: str, command: str) -> None:
        history = self._state.setdefault("command_history", [])
        history.insert(0, {
            "task_type": task_type,
            "command": command,
            "timestamp": time.time(),
        })
        # Keep only last 3
        self._state["command_history"] = history[:3]
        self.save()
