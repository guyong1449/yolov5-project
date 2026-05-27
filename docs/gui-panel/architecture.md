# GUI Panel Architecture

## 三层结构

- `UI`
  原生 `HTML/CSS/JS` 单页，负责任务卡、动态表单、命令预览和实时日志展示。
- `Control`
  `tools/gui_panel/app.py` 提供 FastAPI 接口；`task_specs.py` 管字段定义；`services/` 处理命令、状态、路径选择、布局识别。
- `Execution`
  复用 `train.py`、`detect.py`、`val.py`、`scripts/run_with_log.py`、`tools/fiftyone/fiftyone_import_voc.py`。

## 状态机

- 固定状态：`idle`、`running`、`stopping`、`succeeded`、`failed`、`stopped`
- `ProcessRunner` 是唯一状态源
- 任意时刻只允许一个活动任务

## 日志通路

- 子进程统一合并 `stdout/stderr`
- `ProcessRunner` 维护内存 ring buffer
- `/api/logs/stream` 通过 `SSE` 推送增量日志
- `run_with_log.py` 继续负责落盘 `run.log` 与 `run.md`

## 扩展护栏

- 新任务必须先补 `TaskSpec`
- 命令拼装只放在 `services/command_builder.py`
- 路径自动识别只放在专门 service
- 前端不写任务执行逻辑
