# GUI Panel Task Specs

## 当前任务

- `train`
  重点字段：`data`、`weights`、`project`、`name`、`device`、`epochs`、`batch_size`、`imgsz`
- `detect`
  重点字段：`source`、`voc_root`、`save_img_frames`、`incremental_mp4`
- `val`
  重点字段：`task`、`weights`、`data`、`batch_size`、`imgsz`
- `fiftyone`
  模式：`dataset_root_auto`、`explicit_voc`

## 字段组织

- 分组固定：`basic`、`output`、`advanced`、`extra`
- 每个字段由 `FieldSpec` 描述类型、必填、可见条件、浏览按钮
- 前端通过 `/api/task-definitions` 动态渲染表单，不硬编码单任务页面

## 命令构造原则

- `train/detect/val` 一律通过 `scripts/run_with_log.py`
- `fiftyone` 直接调用 `tools/fiftyone/fiftyone_import_voc.py`
- 所有命令都构造成参数数组，不拼接裸 shell 字符串
- `extra_args` 只作为补充，不覆盖显式主参数

## 新增任务时

1. 在 `task_specs.py` 新增 `TaskSpec`
2. 在 `command_builder.py` 增加构造逻辑
3. 如有特殊路径解析，新增独立 service
4. 补测试，再让前端自动消费定义
