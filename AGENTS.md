# AGENTS.md

## 1. 项目基础信息

- 项目类型：目标检测（YOLOv5，PyTorch 单阶段检测）。
- 仓库形态：`yolov5-master` 风格，真实入口以 `train.py`、`detect.py`、`val.py`、`export.py` 为准。
- 技术栈：
  - Python
  - PyTorch + `torchvision`：版本以下列两处为准
    - 仓库 `requirements.txt`
    - 官方 PyTorch 安装矩阵与本机 CUDA 对应版本
  - `opencv-python`
  - `PyYAML`
  - 其他依赖以 `requirements.txt` 为准，不自行扩展“理论上可能需要”的库

### 硬件与运行预期

- 训练优先使用 NVIDIA GPU + CUDA/cuDNN。
- 仅 `detect.py` 推理时，CPU 可跑，但速度通常明显受限。
- 显存建议：
  - `detect.py` 单图/短视频推理：通常 4GB+ 显存即可起步，具体受 `--imgsz`、模型大小、视频分辨率影响。
  - `train.py` 训练：通常建议 8GB+ 显存，更大 batch 或更大输入尺寸需要更高显存。
- 若出现 CUDA OOM，先降 `--batch-size`，再降 `--imgsz`，最后考虑换更小权重或改设备。

### 核心约束

- 训练/验证尽量固定随机种子。
  - 本仓库 `train.py` 支持 `--seed`，默认显式传参，不依赖默认值。
- 数据集配置以 `data/*.yaml` 为准。
  - 本仓库已存在 `data/dataAirVis.yaml`。
  - 修改数据路径、类别数、类名时，先检查 YAML 中的 `train`、`val`、`nc`、`names`。
- 预训练权重来源必须写清：
  - 官方权重：如 `yolov5s.pt`
  - 用户自备权重：如 `checkpoint/yolov5_best.pt`
- 不假设脚本默认参数可靠。
  - 本仓库存在分支级本地定制，运行时优先显式传 `--data`、`--weights`、`--source`、`--device`、`--project`、`--name`。

## 2. 环境搭建与开发流程

### 环境创建

优先使用 `pip`：

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

若使用 conda，先创建环境，再按本机 CUDA 版本安装匹配的 PyTorch：

```bash
conda create -n yolov5 python=3.10 -y
conda activate yolov5
pip install -r requirements.txt
```

若需单独安装 PyTorch，请以官方安装矩阵为准，不要在 `AGENTS.md` 中写死某个 CUDA 轮子地址。

### 解释器与命令执行

- 本机默认 `python` 可能指向系统解释器，不默认等同于本项目可用解释器。
- 当前已验证可用于本仓库检查与测试的解释器路径：`D:\Miniconda3\python.exe`
- 当需要确认依赖、运行单元测试、或执行仓库脚本时，优先显式使用：

```bash
D:\Miniconda3\python.exe -m unittest tests.test_detect_source_behavior
```

- 若用户只写 `python ...`，先核对实际命中的解释器；不要假设系统默认 `python` 已具备 `torch`、`psutil`、`opencv-python` 等本项目依赖。

### 数据准备

自定义数据集只需对齐数据 YAML，不展开数据治理流程。最少保证：

- `train`：训练集图像列表或目录
- `val`：验证集图像列表或目录
- `nc`：类别数
- `names`：类别名列表，顺序与标签类别 ID 一致
- 若训练输入采用“原视频帧 + XML/VOC 标注”，则 `images/` 下图片必须是原始帧，禁止写入任何检测框、类别名、置信度、水印或调试叠加信息。
- 任何由检测预览图导出的“带框/带置信度 JPG”都不得作为训练数据集图片使用；这类历史数据默认判定为不可用，必须重新生成。

简短示例：

```yaml
train: D:/data/my_dataset/train.txt
val: D:/data/my_dataset/val.txt
nc: 3
names: [person, car, drone]
```

### 训练

推荐始终显式传关键参数，避免吃到分支默认值：

```bash
python train.py --data data/dataAirVis.yaml --weights checkpoint/yolov5_best.pt --epochs 100 --batch-size 16 --imgsz 640 --device 0 --seed 0 --project runs/train --name airvis_exp
```

参数约定：

- `--data`：数据 YAML
- `--weights`：初始权重；从头训练时可传空字符串并配合 `--cfg`
- `--epochs`：训练轮数
- `--batch-size`：总 batch size
- `--imgsz`：训练/验证输入尺寸
- `--device`：如 `0`、`0,1` 或 `cpu`
- `--seed`：训练随机种子
- `--project` / `--name`：输出目录

从头训练示例：

```bash
python train.py --data data/dataAirVis.yaml --weights "" --cfg models/yolov5s.yaml --epochs 100 --batch-size 16 --imgsz 640 --device 0 --seed 0 --project runs/train --name scratch_exp
```

### 推理

本仓库 `detect.py` 有本地定制参数，推理时必须显式传参：

```bash
python detect.py --weights checkpoint/yolov5_best.pt --source "F:/1/video/output" --imgsz 640 --device 0 --project runs/detect --name demo_exp
```

常用参数：

- `--source`：图片、视频、摄像头、`.txt` 路径列表，或显式 glob 模式
- 当前本仓库 `detect.py` 会对普通目录 `--source` 自动递归扫描，等效于把目录展开成 `目录/**/*.*` 后再交给 YOLOv5 过滤支持格式
- 若只想限定某一类文件，仍可显式传 glob，例如 `F:/1/video/output/**/*.mp4`
- `--weights`：推理权重
- `--imgsz`：推理输入尺寸
- `--device`：如 `0` 或 `cpu`
- `--project` / `--name`：输出目录
- `--save-txt` / `--save-csv` / `--save-crop`：按需输出附加结果
- `--save-img-frames` / `--voc-root` / `--incremental-mp4`：仅在明确需要本仓库 VOC/增量视频定制逻辑时使用

### 验证

```bash
python val.py --data data/dataAirVis.yaml --weights checkpoint/yolov5_best.pt --batch-size 16 --imgsz 640 --device 0 --project runs/val --name airvis_val
```

常用参数：

- `--data`：验证集 YAML
- `--weights`：待评估权重
- `--batch-size`：验证 batch
- `--imgsz`：验证尺寸
- `--device`：设备
- `--project` / `--name`：输出目录

### 导出

按部署需求显式指定导出格式：

```bash
python export.py --weights checkpoint/yolov5_best.pt --imgsz 640 640 --device 0 --include onnx
```

常用 `--include`：

- `torchscript`
- `onnx`
- `openvino`
- `engine`

## 3. 测试规范

### 当前仓库测试形态

- 已存在 `tests/`
- 已存在 `setup.cfg` 中的 pytest 配置
- 当前最直接的自动化测试入口可用：

```bash
pytest tests/
```

### 最小通过标准

- 无 traceback
- YAML 可正常读取
- 至少完成一次推理或验证前向
- 产物按预期落在 `runs/` 下，或测试脚本按预期通过

### 推荐测试顺序

1. 先跑现有单元测试：

```bash
pytest tests/
```

2. 再跑一次最小推理冒烟：

```bash
python detect.py --weights checkpoint/yolov5_best.pt --source "F:/1/video/output" --imgsz 640 --device 0 --project runs/detect --name smoke_detect
```

验收标准：

- `pytest tests/` 通过
- `detect.py` 无 traceback
- 输出目录 `runs/detect/smoke_detect` 成功生成
- 若启用保存图像/视频，能看到对应结果文件

### 可选 1 epoch 冒烟

仅在需要验证训练链路时使用：

```bash
python train.py --data data/dataAirVis.yaml --weights checkpoint/yolov5_best.pt --epochs 1 --batch-size 4 --imgsz 640 --device 0 --seed 0 --project runs/train --name smoke_train
```

验收标准：

- 训练能启动并完成 1 epoch
- `runs/train/smoke_train` 生成日志与权重目录
- 无数据 YAML 解析错误、无设备初始化错误

## 4. 代码风格规范

### Python 约定

- 新增函数优先补类型注解。
- docstring 全文统一使用 Google 风格。
- 显式传递 `device`、路径、阈值等关键参数，不隐藏在全局状态里。
- 错误处理优先给出可定位的信息，如权重路径、YAML 路径、source 路径。
- 不为了“更现代”而大改 Ultralytics 既有调用方式；优先最小差异。

### 优质示例

```python
from pathlib import Path
import torch


def load_image_tensor(image_path: str, device: torch.device) -> torch.Tensor:
    """Load an image path and move a BCHW tensor to the target device.

    Args:
        image_path: Input image file path.
        device: Target torch device.

    Returns:
        A float32 tensor on the requested device.

    Raises:
        FileNotFoundError: If the image path does not exist.
    """
    path = Path(image_path)
    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {path}")

    tensor = torch.zeros(1, 3, 640, 640, dtype=torch.float32)
    return tensor.to(device)
```

### 日志与可视化

- 若启用 `wandb` 或 TensorBoard，不在代码或文档中写死凭据。
- 仅使用环境变量或本机已配置环境。
- 不提交 API key、token、账号信息。

## 5. 操作边界与禁止行为

### 必须做

- 只使用仓库真实存在的入口脚本：
  - `train.py`
  - `detect.py`
  - `val.py`
  - `export.py`
- 修改训练/推理命令时，优先给出可复制的一整条命令。
- 任何涉及数据集的改动，先核对 `data/*.yaml`。
- 任何涉及结果目录的改动，先说明会写到哪个 `runs/...` 目录。
- 任何影响复现的改动，尽量保留显式参数，如 `--seed`、`--project`、`--name`。

### 先询问

- 删除、覆盖、移动以下内容前必须确认：
  - `runs/`
  - `weights/`
  - `checkpoint/`
  - 数据集根目录
  - 任意 `*.pt`
  - 导出产物，如 `.onnx`、`.engine`、`.xml`
- 修改以下核心逻辑前必须先说明风险，并标记“需人类审核”：
  - `models/yolo.py`
  - 检测 head
  - loss 计算
  - anchor / target assign
  - 导出图结构相关逻辑
- 若要改 `detect.py` 中本仓库自定义的 VOC/XML、逐帧导出、增量视频处理逻辑，先说明会不会影响原始 Ultralytics 行为。
- 若使用 `--save-img-frames` / `--voc-root` 生成训练数据集，导出的 `images/*.jpg` 只能保存原始视频帧；框和置信度仅允许存在于 XML 标注中，不允许回写到训练图片。

### 绝对禁止

- 编造不存在的脚本、配置、目录或命令参数。
- 把分布式训练、多机多卡、`torch.distributed.run` 写进本项目常规工作流。
- 在未确认的情况下覆盖用户自备权重或数据集。
- 未经确认直接清空 `runs/`、替换 `best.pt`、重写导出目录。
- 把检测预览图、带框图、带置信度标签图当作 VOC/YOLO 训练图片继续使用。
- 将科研论文式流程强塞进本仓库：
  - 数据治理大 checklist
  - 消融矩阵
  - 论文级图表流水线
- 因“代码整洁”对 YOLOv5 主干做大规模重构。

## 6. 路径边界

### 默认可写

- `runs/train/*`
- `runs/val/*`
- `runs/detect/*`
- 新建的临时日志目录
- 用户明确指定的导出目录

### 默认只读

- `weights/*.pt`
- `checkpoint/*.pt`
- 数据集根目录
- 已存在的导出模型产物
- `models/` 下核心网络定义
- `utils/` 下被训练/推理主流程直接依赖的稳定工具函数

### 覆盖前必须确认

- 同名 `runs/...` 实验目录且使用了 `--exist-ok`
- `best.pt`、`last.pt`
- 用户手工整理过的 `results.csv`
- VOC/XML 导出目录中的历史标注结果

## 7. 产物管理

- 训练产物默认在 `runs/train/<name>/`
  - 常见文件：`weights/best.pt`、`weights/last.pt`、`results.csv`
- 验证产物默认在 `runs/val/<name>/`
- 推理产物默认在 `runs/detect/<name>/`
- 若实验需要长期保留，优先改 `--name`，不要反复覆盖 `exp`
- 非必要不要清理历史 checkpoint；清理前先确认是否还要复现实验

## 8. 常见问题

### CUDA OOM

先降资源占用：

```bash
python train.py --data data/dataAirVis.yaml --weights checkpoint/yolov5_best.pt --epochs 100 --batch-size 8 --imgsz 512 --device 0 --seed 0 --project runs/train --name oom_retry
```

### OpenCV 读视频失败

先确认路径和视频后缀，再用最小命令复现：

```bash
python detect.py --weights checkpoint/yolov5_best.pt --source "F:/1/video/output" --imgsz 640 --device 0 --project runs/detect --name cv_check
```

### 本机 torch 与 `requirements.txt` 不一致

先看实际安装版本，不要只看 `requirements.txt`：

```bash
python -c "import torch, torchvision; print(torch.__version__, torchvision.__version__, torch.cuda.is_available())"
```

若版本不匹配，以本机 CUDA 和官方 PyTorch 安装矩阵重新安装，再回到本仓库命令验证。
