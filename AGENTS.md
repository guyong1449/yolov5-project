# AGENTS.md

## 项目范围

- 当前主线是 YOLOv5 目标检测, 针对无人机、鸟类、飞行器等等小目标。
- 常用入口：`train.py`、`detect.py`、`val.py`、`export.py`
- 当前仓库内数据配置：`data/dataAirVis.yaml`
- 当前主权重：`checkpoint/yolov5_best.pt`

## 当前任务
1. 加入tracking增强连续目标跨帧检测
2. 加入适合yolo检测小目标的额外头

- 在 **4 个 NPU** 上启用并稳定运行 **DDP 分布式训练**。
- 目标包括：可重复启动、日志可追溯、失败可定位、单机多卡吞吐提升可量化。

## 工作要求

- 涉及训练、验证、推理命令时，优先显式传 `--data`、`--weights`、`--source`、`--device`、`--project`、`--name`、`--seed`。
- 涉及数据集改动前，先核对 YAML 中 `train`、`val`、`nc`、`names` 一致性。
- 涉及日志输出时，优先使用 `scripts/run_with_log.py`，并保持同一任务产物落在同一目录。
- 若修改训练或分布式相关逻辑，需说明对单卡流程的兼容性影响。

## 辅助流程

- FiftyOne 去重流程：`docs/fiftyone-dedup-workflow.md`。
- 网络与 SSH 相关：`docs/189-mihomo-usage.md`、`docs/189-github-ssh.md`。
- GUI 面板文档：`docs/gui-panel/architecture.md`、`docs/gui-panel/task-specs.md`、`docs/gui-panel/fiftyone-integration.md`、`docs/gui-panel/startup.md`。

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
- `tools/npu_ddp_detect_benchmark.sh`：4 卡并行 detect 包装脚本
- `docs/npu-detect-dataflow.md`：detect 单进程 / 四卡并行的数据通路说明
- `docs/npu-detect-run-commands.md`：NPU detect 运行命令与三轮实测口径
- `scripts/run_with_log.py`：统一日志、Markdown 与并行摘要落盘
- `tests/test_npu_benchmark_dryrun.py`、`tests/test_npu_ddp_support.py`：NPU detect/DDP 相关测试
- `archive/`：已归档的官方示例、缓存、副本和训练压缩包

## 最小验证

- 单测：`pytest tests/`
- FiftyOne 相关单测：`python -m unittest tests.test_fiftyone_tools`
