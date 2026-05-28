# NPU Detect 运行命令（Linux + YOLO312）

## 基线约束（强制）

- 项目目录固定为：`/root/workspace/repos/yolov5-project`
- 权重固定为：`/root/workspace/repos/yolov5-project/checkpoint/yolov5_best.pt`
- 输入目录固定为：`/root/workspace/data/videos`
- 输出目录固定为：`/root/workspace/outputs/runs`
- 所有命令一律在 `yolo312` 环境执行
- 本文只记录 **NPU detect 推理**，不包含 CPU 对照 benchmark


## 运行前检查

```bash
cd /root/workspace/repos/yolov5-project
source /root/.bashrc
conda activate yolo312

python -u -c "import torch, torch_npu; print(torch.__version__); print(torch.npu.is_available()); print(torch.npu.device_count())"
npu-smi info
```
## 主命令：目录推理并保存带框结果

```bash
cd /root/workspace/repos/yolov5-project
source /root/.bashrc
conda activate yolo312
export ASCEND_RT_VISIBLE_DEVICES=0

python scripts/run_with_log.py --name vis_boxed_video -- \
  python detect.py \
    --weights /root/workspace/repos/yolov5-project/checkpoint/yolov5_best.pt \
    --source /root/workspace/data/videos \
    --data data/dataAirVis.yaml \
    --imgsz 640 \
    --device npu:0 \
    --project /root/workspace/outputs/runs \
    --name vis_boxed_video \
    --conf-thres 0.25 \
    --iou-thres 0.45
```

说明：

- `--source /root/workspace/data/videos` 表示对该目录下的视频或图像源做推理
- `--name vis_boxed_video` 表示输出目录为 `/root/workspace/outputs/runs/vis_boxed_video`
- 该命令会保存带框可视化结果
- 该命令不做 VOC/XML 导出
- 该命令不需要 `--voc-root`
- 该命令不需要 `--vid-stride`
- 该命令不需要 `--save-img-frames`
- 该命令不需要 `--nosave`

## 脚本版命令（与仓库现有封装保持一致）

```bash
cd /root/workspace/repos/yolov5-project
source /root/.bashrc
conda activate yolo312
export ASCEND_RT_VISIBLE_DEVICES=0

python scripts/run_with_log.py --name vis_boxed_video_script -- \
  env DEVICE=npu:0 \
      WEIGHTS=/root/workspace/repos/yolov5-project/checkpoint/yolov5_best.pt \
      DATA_YAML=/root/workspace/repos/yolov5-project/data/dataAirVis.yaml \
      SOURCE_PATH=/root/workspace/data/videos \
      PROJECT_DIR=/root/workspace/outputs/runs \
      RUN_NAME=vis_boxed_video \
      bash tools/npu_video_benchmark.sh
```

说明：

- `tools/npu_video_benchmark.sh` 本质上仍是调用 `detect.py`
- 适合后续统一封装日志或批量执行时复用
- 若只做单次推理，优先使用上面的 `run_with_log + detect.py` 主命令，便于直接观察参数

## 输出检查

预期输出目录：

- `/root/workspace/outputs/runs/vis_boxed_video`

重点检查：

- 目录已生成
- 检测后的图片或视频已保存
- 终端日志中没有 NPU 初始化失败、权重加载失败、数据配置缺失等报错
