# AGENTS.md

## 项目范围

- 当前主线是 YOLOv5 目标检测, 针对无人机、鸟类、飞行器等等小目标。
- 常用入口：`train.py`、`detect.py`、`val.py`、`export.py`
- 当前仓库内数据配置：`data/dataAirVis.yaml`
- 当前主权重：`checkpoint/yolov5_best.pt`

## 当前任务
1. 加入tracking增强连续目标跨帧检测
2. 加入适合yolo检测小目标的额外头

## 当前辅助流程

- FiftyOne 去重流程说明见 [docs/fiftyone-dedup-workflow.md](/abs/path/F:/1/yolov5-master/docs/fiftyone-dedup-workflow.md)
- `ssh 189` 本机代理中转见 [docs/189-mihomo-usage.md](docs/189-mihomo-usage.md)；GitHub SSH 排查见 [docs/189-github-ssh.md](docs/189-github-ssh.md)
- GUI 面板架构说明见 [docs/gui-panel/architecture.md](/abs/path/F:/1/yolov5-master/docs/gui-panel/architecture.md)
- GUI 面板任务规格见 [docs/gui-panel/task-specs.md](/abs/path/F:/1/yolov5-master/docs/gui-panel/task-specs.md)
- GUI 面板 FiftyOne 集成见 [docs/gui-panel/fiftyone-integration.md](/abs/path/F:/1/yolov5-master/docs/gui-panel/fiftyone-integration.md)
- GUI 面板启动方式见 [docs/gui-panel/startup.md](/abs/path/F:/1/yolov5-master/docs/gui-panel/startup.md)

## 红线

- 未经当前回合明确授权，不改 Python / Conda 环境。
- 未经当前回合明确授权，不执行 `conda create`、`conda remove`、`conda env remove`、`pip install`、`pip uninstall`。
- 未经确认，不删除、覆盖或移动 `runs/`、`checkpoint/`、任意 `*.pt`、数据集根目录、导出产物。

## 当前工作规则

- 训练、验证、推理命令优先显式传 `--data`、`--weights`、`--source`、`--device`、`--project`、`--name`、`--seed`。
- 任何数据集相关改动先核对 YAML 的 `train`、`val`、`nc`、`names`。
- 若使用 VOC/XML 流程，训练图片必须是原始帧，不得写入框、类别名、置信度或其他叠加信息。
- 若修改 `detect.py` 中的 VOC/XML、逐帧导出或增量视频逻辑，先说明是否影响现有 YOLOv5 行为。

## 当前目录约定

- `tools/fiftyone/`：FiftyOne 导入、相似度、去重、启动器
- `tools/ssh_189/`：`189` 反向代理隧道脚本
- `tools/label_tools.py`：VOC / YOLO / FiftyOne 标签处理
- `archive/`：已归档的官方示例、缓存、副本和训练压缩包

## 最小验证

- 单测：`pytest tests/`
- FiftyOne 相关单测：`python -m unittest tests.test_fiftyone_tools`
