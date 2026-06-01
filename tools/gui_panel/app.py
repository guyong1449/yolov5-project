from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from tools.gui_panel.schemas import OpenPathRequest, PathDialogRequest, ResolveLayoutRequest, RunRequest
from tools.gui_panel.services.command_builder import validate_task
from tools.gui_panel.services.fiftyone_resolver import resolve_dataset_root
from tools.gui_panel.services.path_dialog import select_path
from tools.gui_panel.services.process_runner import ProcessRunner
from tools.gui_panel.services.session_store import SessionStore
from tools.gui_panel.task_specs import TASK_SPECS
from utils.env_config import get_data_yaml, get_dataset_dir, get_device, get_output_dir, get_video_dir, get_weights

REPO_ROOT = Path(__file__).resolve().parents[2]
STATIC_DIR = Path(__file__).resolve().parent / "static"


def create_app(*, session_file: Path | None = None) -> FastAPI:
    app = FastAPI(title="YOLOv5 GUI Panel")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://127.0.0.1", "http://localhost"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    runner = ProcessRunner()
    store = SessionStore(session_file or REPO_ROOT / "runs" / "gui_panel" / "last_session.json")
    app.state.runner = runner
    app.state.store = store

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/api/task-definitions")
    def task_definitions() -> dict[str, Any]:
        return {
            "tasks": {name: spec.to_dict() for name, spec in TASK_SPECS.items()},
            "session": store.state,
            "command_history": store.state.get("command_history", []),
        }

    @app.get("/api/runtime-state")
    def runtime_state() -> dict[str, Any]:
        state = runner.runtime_state().model_dump()
        if state.get("active_task"):
            store.update_task(
                state["active_task"],
                output_path=state.get("last_output_path") or None,
                log_path=state.get("last_log_path") or None,
            )
        state["recent_values"] = store.state.get("recent_values", {})
        state["recent_outputs"] = store.state.get("recent_outputs", {})
        state["recent_logs_paths"] = store.state.get("recent_logs", {})
        state["command_history"] = store.state.get("command_history", [])
        return state

    @app.post("/api/tasks/{task_type}/preview")
    def preview_task(task_type: str, request: RunRequest) -> dict[str, Any]:
        _ensure_task(task_type)
        values = _merge_defaults(task_type, request.values)
        result = validate_task(task_type, values, validate_paths=False)
        return {"command": result.command, "argv": result.argv}

    @app.post("/api/tasks/{task_type}/validate")
    def validate_route(task_type: str, request: RunRequest) -> JSONResponse:
        _ensure_task(task_type)
        values = _merge_defaults(task_type, request.values)
        result = validate_task(task_type, values, validate_paths=True)
        store.update_task(task_type, values=result.normalized_values, command=result.command)
        return JSONResponse(result.model_dump())

    @app.post("/api/tasks/{task_type}/start")
    def start_task(task_type: str, request: RunRequest) -> dict[str, Any]:
        _ensure_task(task_type)
        values = _merge_defaults(task_type, request.values)
        result = validate_task(task_type, values, validate_paths=True)
        if not result.ok:
            raise HTTPException(status_code=400, detail=result.errors)
        run_request = RunRequest(
            task_type=task_type,
            values=result.normalized_values,
            metadata={
                "argv": result.argv,
                "command": result.command,
                "output_path": result.metadata.get("output_path", ""),
            },
        )
        try:
            state = runner.start(run_request)
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        store.update_task(
            task_type,
            values=result.normalized_values,
            command=result.command,
            output_path=result.metadata.get("output_path", ""),
        )
        store.add_command_history(task_type, result.command)
        return state.model_dump()

    @app.post("/api/tasks/stop")
    def stop_task() -> dict[str, Any]:
        return runner.stop().model_dump()

    @app.post("/api/logs/clear")
    def clear_logs() -> dict[str, str]:
        runner.clear_logs()
        return {"status": "ok"}

    @app.get("/api/logs/stream")
    def logs_stream() -> StreamingResponse:
        def event_stream():
            for event_id, line in runner.log_stream():
                if line is None:
                    yield ": keepalive\n\n"
                else:
                    yield f"id: {event_id}\ndata: {line}\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    @app.post("/api/dialog/select-path")
    def select_path_route(request: PathDialogRequest) -> dict[str, str]:
        try:
            return {"path": select_path(request.kind, request.title, request.initial_path)}
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.post("/api/fiftyone/resolve-layout")
    def resolve_layout(request: ResolveLayoutRequest) -> dict[str, str]:
        try:
            resolved = resolve_dataset_root(request.dataset_root)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {
            "dataset_root": str(resolved.dataset_root),
            "data_dir": str(resolved.data_dir),
            "labels_dir": str(resolved.labels_dir),
            "layout": resolved.layout,
        }

    @app.post("/api/open-path")
    def open_path(request: OpenPathRequest) -> dict[str, str]:
        path = Path(request.path)
        if not path.exists():
            raise HTTPException(status_code=404, detail=f"Path not found: {path}")
        os.startfile(str(path))  # type: ignore[attr-defined]
        return {"status": "ok"}

    @app.get("/api/example-commands")
    def example_commands() -> dict[str, Any]:
        weights = get_weights()
        video_dir = get_video_dir()
        data_yaml = get_data_yaml()
        dataset_dir = get_dataset_dir()
        detect_output = get_output_dir("detect")
        device = get_device()
        return {
            "train": [
                {
                    "label": "标准训练 (SGD, 70 epochs)",
                    "command": (
                        f"python scripts/run_with_log.py -- \\\n"
                        f"  python train.py \\\n"
                        f"  --data {dataset_dir}/data.yaml \\\n"
                        f"  --weights {weights} \\\n"
                        f"  --epochs 70 --batch-size 4 --imgsz 640 \\\n"
                        f"  --device {device} --seed 0 --workers 2 --patience 20 \\\n"
                        f"  --optimizer SGD --project {get_output_dir('train')} --name test1_stride10_sgd_70e"
                    ),
                },
            ],
            "detect": [
                {
                    "label": "VOC 抽帧导出 (vid-stride=10)",
                    "command": (
                        f"python scripts/run_with_log.py -- \\\n"
                        f"  python detect.py \\\n"
                        f"  --weights {weights} \\\n"
                        f"  --source \"{video_dir}\" \\\n"
                        f"  --data {data_yaml} \\\n"
                        f"  --imgsz 640 --device cpu \\\n"
                        f"  --project {detect_output} --name voc_stride10 \\\n"
                        f"  --voc-root {dataset_dir} \\\n"
                        f"  --vid-stride 10 --save-img-frames --nosave --incremental-mp4"
                    ),
                },
                {
                    "label": "封装脚本 (extract_voc_stride10)",
                    "command": (
                        f"python scripts/run_with_log.py -- \\\n"
                        f"  python scripts/extract_voc_stride10.py \\\n"
                        f"  --weights {weights} \\\n"
                        f"  --source \"{video_dir}\" \\\n"
                        f"  --voc-root {dataset_dir.replace('test1', 'test2')} \\\n"
                        f"  --data-yaml {data_yaml} \\\n"
                        f"  --device cpu"
                    ),
                },
            ],
            "val": [],
            "fiftyone": [
                {
                    "label": "FiftyOne 连续去重",
                    "command": (
                        f"python tools/fiftyone/fiftyone_run_full_dedup_pipeline.py \\\n"
                        f"  --dataset-name test1_stride10_voc \\\n"
                        f"  --model clip-vit-base32-torch \\\n"
                        f"  --brain-key clip_vit_base32_sim \\\n"
                        f"  --approx-threshold 0.12 --approx-group-keep-ratio 0.3 \\\n"
                        f"  --voc-root \"{dataset_dir}/fiftyone_voc\" \\\n"
                        f"  --export-dir \"{dataset_dir}/fiftyone_voc_deduped\" \\\n"
                        f"  --report-dir \"{dataset_dir}/fiftyone_voc/dedup_reports\" \\\n"
                        f"  --overwrite"
                    ),
                },
            ],
        }

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    return app


def _merge_defaults(task_type: str, values: dict[str, Any]) -> dict[str, Any]:
    merged = dict(TASK_SPECS[task_type].defaults)
    merged.update(values or {})
    return merged


def _ensure_task(task_type: str) -> None:
    if task_type not in TASK_SPECS:
        raise HTTPException(status_code=404, detail=f"Unknown task type: {task_type}")


app = create_app()


if __name__ == "__main__":
    uvicorn.run("tools.gui_panel.app:app", host="127.0.0.1", port=8752, reload=False)
