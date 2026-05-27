from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Optional

from pydantic import BaseModel, Field


ValidatorFunc = Callable[[dict[str, Any]], tuple[dict[str, Any], list[str], dict[str, Any]]]
CommandBuilderFunc = Callable[[dict[str, Any]], tuple[list[str], dict[str, Any]]]
ResultParserFunc = Callable[[str], dict[str, Any]]


@dataclass
class FieldSpec:
    name: str
    label: str
    kind: str
    group: str
    required: bool = False
    help_text: str = ""
    placeholder: str = ""
    step: str = ""
    choices: list[dict[str, str]] = field(default_factory=list)
    browsable: bool = False
    browse_kind: Optional[str] = None
    visible_when: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "label": self.label,
            "kind": self.kind,
            "group": self.group,
            "required": self.required,
            "help_text": self.help_text,
            "placeholder": self.placeholder,
            "step": self.step,
            "choices": self.choices,
            "browsable": self.browsable,
            "browse_kind": self.browse_kind,
            "visible_when": self.visible_when,
        }


@dataclass
class TaskSpec:
    task_type: str
    display_name: str
    description: str
    field_groups: list[dict[str, str]]
    fields: list[FieldSpec]
    defaults: dict[str, Any]
    validator: Optional[ValidatorFunc] = None
    command_builder: Optional[CommandBuilderFunc] = None
    result_parser: Optional[ResultParserFunc] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_type": self.task_type,
            "display_name": self.display_name,
            "description": self.description,
            "field_groups": self.field_groups,
            "fields": [field.to_dict() for field in self.fields],
            "defaults": self.defaults,
        }


class RunRequest(BaseModel):
    task_type: str
    values: dict[str, Any] = Field(default_factory=dict)
    command_override: Optional[list[str]] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ValidationResponse(BaseModel):
    ok: bool
    command: str
    argv: list[str] = Field(default_factory=list)
    normalized_values: dict[str, Any] = Field(default_factory=dict)
    errors: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class PathDialogRequest(BaseModel):
    kind: str = "file"
    title: str = "Select path"
    initial_path: str = ""


class OpenPathRequest(BaseModel):
    path: str


class ResolveLayoutRequest(BaseModel):
    dataset_root: str


class RuntimeStateModel(BaseModel):
    status: str = "idle"
    active_task: Optional[str] = None
    pid: Optional[int] = None
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    return_code: Optional[int] = None
    stop_requested: bool = False
    command_preview: str = ""
    last_log_path: str = ""
    last_view_path: str = ""
    last_output_path: str = ""
    last_session_url: str = ""
    last_dataset_name: str = ""
    last_samples_count: Optional[int] = None
    message: str = ""
    recent_logs: list[str] = Field(default_factory=list)
    recent_values: dict[str, dict[str, Any]] = Field(default_factory=dict)


def iso_now() -> str:
    return datetime.now().isoformat(timespec="seconds")
