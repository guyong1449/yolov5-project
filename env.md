# 当前环境配置（Linux）

更新时间：2026-05-28  
主机：Linux  
仓库路径：`/root/workspace/repos/yolov5-project`  
默认说明语言：中文

## 1. 核心路径

- 仓库根目录：`/root/workspace/repos/yolov5-project`
- 数据目录：`/root/workspace/data`
- 运行输出目录：`/root/workspace/outputs/runs`
- 日志目录：`/root/workspace/repos/yolov5-project/runs/logs`

## 2. Python / Conda

- Conda 根目录（约定）：`/root/miniconda3`
- 常用环境：`f312`、`yolo312`
- 建议在仓库内统一使用：
  - `source /root/.bashrc`
  - `conda activate yolo312`
  - `python ...`

## 3. 常用命令约定

- 训练/推理/验证命令优先通过 `scripts/run_with_log.py` 包裹执行。
- `detect.py`、`train.py`、`val.py` 命令建议显式传：
  - `--data`
  - `--weights`
  - `--source`
  - `--device`
  - `--project`
  - `--name`

## 4. 路径风格要求

- 项目内文档、脚本默认使用 Linux 路径（`/root/...`）。
- 不再使用 Windows 风格盘符路径，统一采用 Linux 绝对路径。
