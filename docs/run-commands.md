# 运行命令（Linux + YOLO312）

## 基线约束（强制）

- 项目目录固定为：`/root/workspace/repos/yolov5-project`
- 视频文件固定为：`/root/workspace/data/videos/video20.mp4`
- 权重固定为：`/root/workspace/repos/yolov5-project/checkpoint/yolov5_best.pt`
- 输出目录固定为：`/root/workspace/outputs/runs`
- 所有修改、验证、benchmark 一律在 `yolo312` 环境执行


## 测试npu
```bash
cd /root/workspace/repos/yolov5-project
source /root/.bashrc
conda activate yolo312
python /opt/torch_env/npu_demo.py
python /opt/torch_env/npu_stress_all.py
```

```bash
scp -r /root/workspace/data/videos/video20.mp4 root@189:/root/workspace/repos/yolov5-project/checkpoint
```

## 当前推荐命令

基于 `runs/train/test1_stride10_sgd_70e3/opt.yaml` 对应配置，当前保留主线命令如下：

```bash
python -u -c "import torch, torch_npu; print(torch.__version__); print(torch.npu.is_available()); print(torch.npu.device_count())"
npu-smi info
```

预期：`torch.npu.is_available()` 为 `True`，且 `device_count() >= 1`。

- 本命令在当前代码下会把实际训练目录增量落到 `runs/train/test1_stride10_sgd_70e3/`
- `run_with_log.py` 对 `train.py` 已默认并入训练目录，不必再单独指定 `runs/logs/...`
- 如需避免继续递增目录，请只在明确允许覆盖时再加 `--exist-ok`

## VOC 抽帧导出（detect.py，`vid-stride=10`）

从 `/root/workspace/data/videos` 扫视频，每 10 帧取 1 帧，把**原始帧**写入 VOC `images/`、检测框写入 `annotations/`（与训练集 `test1_stride10` 同根目录，便于后续 LabelImg 复核与转 YOLO 标签）。

```bash
cd /root/workspace/repos/yolov5-project
source /root/.bashrc
conda activate yolo312

python scripts/run_with_log.py --name detect_voc_stride10 -- \
  python detect.py \
    --weights /root/workspace/repos/yolov5-project/checkpoint/yolov5_best.pt \
    --source /root/workspace/data/videos \
    --data /root/workspace/repos/yolov5-project/data/dataAirVis.yaml \
    --imgsz 640 \
    --device cpu \
    --project /root/workspace/outputs/runs/detect \
    --name voc_stride10 \
    --voc-root /root/workspace/data/labelimg/test1_stride10 \
    --vid-stride 10 \
    --save-img-frames \
    --nosave \
    --incremental-mp4 \
    --conf-thres 0.25 \
    --iou-thres 0.45
```

产出目录（`--voc-root` 下）：

- `/root/workspace/data/labelimg/test1_stride10/images/*.jpg`：无框原始帧
- `/root/workspace/data/labelimg/test1_stride10/annotations/*.xml`：PASCAL VOC 初始框（需人工复核后才可当训练真值）

保存策略：

- 仅当「stride 采样后的该帧」**至少有一个检测框**时才写入 JPG + XML；无目标的采样帧不落盘（省磁盘，适合伪标签流程）。

说明：

- **默认不弹窗预览**；需要 `cv2.imshow` 实时看画面时再显式加 `--view-img`
- `--source` 为目录时，本仓库会递归扫描其下支持的视频后缀；目录已存在则向同一 `images/`、`annotations/` **追加**写入
- `--nosave`：不在 `runs/detect/...` 再落带框预览图/视频，避免与 VOC 目录混淆
- `--incremental-mp4`：只转换状态文件中尚未记录的视频（状态见 `{voc-root}/.yolov5_mp4_convert_state.json`）；首次全量导出可去掉该参数
- `--voc-root` 应与 `--data` 指向的数据集根目录一致（与 `train.txt` 同级）；GUI 面板 Detect 任务默认已按此配置
- **GUI 面板**（`tools/gui_panel`）Detect 默认：`vid_stride=10`、`save_img_frames`、`incremental_mp4`、`nosave`、`name=voc_stride10`，`voc_root` 与 `data/dataAirVis.yaml` 对应数据集根一致

### conf / iou（全局，四类共用一组阈值）

- **`--conf-thres`**：置信度低于该值的框在 NMS 前丢弃。伪标签抽帧可先试 **0.25**；drone / ptarget 偏小易漏检时用 **0.20~0.25**；bird 误检多时用 **0.30~0.35**。不能在同一趟 detect 里为各类设不同 conf（除非分次跑 `--classes`）。
- **`--iou-thres`**：NMS 用。两框 IoU（交集面积 ÷ 并集面积）超过该阈值时，保留高 conf、去掉重叠的低 conf 框，用于去掉同一目标的重复框。默认 **0.45** 即可；密集小目标可试 **0.40~0.50**。

也可用封装脚本（默认 `--vid-stride 10`）：

```bash
cd /root/workspace/repos/yolov5-project
source /root/.bashrc
conda activate yolo312

python scripts/run_with_log.py --name extract_voc_stride10 -- \
  python scripts/extract_voc_stride10.py \
    --weights /root/workspace/repos/yolov5-project/checkpoint/yolov5_best.pt \
    --source /root/workspace/data/videos \
    --voc-root /root/workspace/data/labelimg/test2_stride10 \
    --data-yaml /root/workspace/repos/yolov5-project/data/dataAirVis.yaml \
    --device cpu
```

## FiftyOne 连续去重

完整说明见：

- [fiftyone-dedup-workflow.md](docs/fiftyone-dedup-workflow.md)

主命令：

```bash
cd /root/workspace/repos/yolov5-project
source /root/.bashrc
conda activate yolo312

python tools/fiftyone/fiftyone_run_full_dedup_pipeline.py \
  --dataset-name test1_stride10_voc \
  --model clip-vit-base32-torch \
  --brain-key clip_vit_base32_sim \
  --approx-threshold 0.12 \
  --approx-group-keep-ratio 0.3 \
  --voc-root /root/workspace/data/labelimg/test1_stride10/fiftyone_voc \
  --export-dir /root/workspace/data/labelimg/test1_stride10/fiftyone_voc_deduped \
  --report-dir /root/workspace/data/labelimg/test1_stride10/fiftyone_voc/dedup_reports \
  --overwrite
```

## Linux NPU 视频 benchmark

适用前提：

- 权重存在：`checkpoint/yolov5_best.pt`
- 数据配置存在：`data/dataAirVis.yaml`
- 测试视频放在 `/root/workspace/data/videos/`
- 已激活包含 `torch` 与 `torch_npu` 的环境

先建目录：

```bash
mkdir -p /root/workspace/data/videos
mkdir -p /root/workspace/outputs/runs
```

CPU 冒烟：

```bash
cd /root/workspace/repos/yolov5-project
source /root/.bashrc
conda activate yolo312

python scripts/run_with_log.py --name cpu_video20 -- \
  python detect.py \
    --weights /root/workspace/repos/yolov5-project/checkpoint/yolov5_best.pt \
    --source /root/workspace/data/videos/video20.mp4 \
    --data data/dataAirVis.yaml \
    --device cpu \
    --project /root/workspace/outputs/runs \
    --name cpu_video20
```

## NPU 正式组（video20）

```bash
cd /root/workspace/repos/yolov5-project
source /root/.bashrc
conda activate yolo312
export ASCEND_RT_VISIBLE_DEVICES=0

python scripts/run_with_log.py --name npu_video20 -- \
  python detect.py \
    --weights /root/workspace/repos/yolov5-project/checkpoint/yolov5_best.pt \
    --source /root/workspace/data/videos/video20.mp4 \
    --data data/dataAirVis.yaml \
    --device npu:0 \
    --project /root/workspace/outputs/runs \
    --name npu_video20
```

## NPU 组（脚本版）

```bash
cd /root/workspace/repos/yolov5-project
source /root/.bashrc
conda activate yolo312

python scripts/run_with_log.py --name npu_video20_script -- \
  env DEVICE=npu:0 \
      WEIGHTS=/root/workspace/repos/yolov5-project/checkpoint/yolov5_best.pt \
      SOURCE_PATH=/root/workspace/data/videos/video20.mp4 \
      PROJECT_DIR=/root/workspace/outputs/runs \
      RUN_NAME=npu_video20 \
      bash tools/npu_video_benchmark.sh
```

## 监控建议

```bash
watch -n 1 npu-smi info
```

## 结果检查

- CPU 输出：`/root/workspace/outputs/runs/cpu_video20`
- NPU 输出：`/root/workspace/outputs/runs/npu_video20`
- 对比指标：总耗时、有效 FPS、NPU AICore 与显存占用峰值
