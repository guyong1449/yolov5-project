# NPU Detect 运行命令（Linux + YOLO312）

## 基线约束

- 项目目录固定：`/root/workspace/repos/yolov5-project`
- 权重固定：`/root/workspace/repos/yolov5-project/checkpoint/yolov5_best.pt`
- 输入目录固定：`/root/workspace/data/videos`
- 输出目录固定：`/root/workspace/outputs/runs`
- 所有命令都在 `yolo312` 环境执行
- 默认暴露 4 张 NPU：`ASCEND_RT_VISIBLE_DEVICES=0,1,2,3`

## 运行前检查

```bash
cd /root/workspace/repos/yolov5-project
source /root/.bashrc
conda activate yolo312

python -u -c "import torch, torch_npu; print(torch.__version__); print(torch.npu.is_available()); print(torch.npu.device_count())"
npu-smi info
```

## 单进程 detect

`detect.py` 现在默认 `--device=npu:0,1,2,3`，但未开启 `--ddp-infer` 时仍是单进程推理，实际绑定首张卡。

```bash
cd /root/workspace/repos/yolov5-project
source /root/.bashrc
conda activate yolo312
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3

python scripts/run_with_log.py --name vis_boxed_video -- \
  python detect.py \
    --weights /root/workspace/repos/yolov5-project/checkpoint/yolov5_best.pt \
    --source /root/workspace/data/videos \
    --data /root/workspace/repos/yolov5-project/data/dataAirVis.yaml \
    --imgsz 640 \
    --project /root/workspace/outputs/runs \
    --name vis_boxed_video
```

## 四卡并行 detect

四卡并行模式通过 `torch.distributed.run` 启动 4 个进程，每个 rank 只处理满足 `frame_index % 4 == rank` 的帧。

- 使用脚本：`tools/npu_ddp_detect_benchmark.sh`
- 不输出视频，不输出图片，不输出 labels，只保留 `.log`、`.md`、`.txt`
- 真四卡证据来自日志中的：
  - `INFER init: rank=0..3`
  - `INFER done: rank=... processed_frames=...`
  - `INFER aggregate: world_size=4 ... parallel_infer_confirmed=true`

单次命令：

```bash
cd /root/workspace/repos/yolov5-project
source /root/.bashrc
conda activate yolo312
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3

env WEIGHTS=/root/workspace/repos/yolov5-project/checkpoint/yolov5_best.pt \
    DATA_YAML=/root/workspace/repos/yolov5-project/data/dataAirVis.yaml \
    SOURCE_PATH=/root/workspace/data/videos/video20.mp4 \
    PROJECT_DIR=/root/workspace/outputs/runs \
    RUN_NAME=npu_video20_ddp_once \
    MASTER_PORT=29540 \
    NPROC_PER_NODE=4 \
    DEVICE=npu:0,1,2,3 \
    python scripts/run_with_log.py \
      --name npu_video20_ddp_once \
      --log-file runs/logs/npu_video20_ddp/npu_video20_ddp_once/npu_video20_ddp_once.log \
      --md-file runs/logs/npu_video20_ddp/npu_video20_ddp_once/npu_video20_ddp_once.md \
      -- bash tools/npu_ddp_detect_benchmark.sh
```

## 三轮实测命令

第 1 轮：

```bash
cd /root/workspace/repos/yolov5-project
source /root/.bashrc
conda activate yolo312
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3

env WEIGHTS=/root/workspace/repos/yolov5-project/checkpoint/yolov5_best.pt \
    DATA_YAML=/root/workspace/repos/yolov5-project/data/dataAirVis.yaml \
    SOURCE_PATH=/root/workspace/data/videos/video20.mp4 \
    PROJECT_DIR=/root/workspace/outputs/runs \
    RUN_NAME=npu_video20_ddp_r1 \
    MASTER_PORT=29541 \
    NPROC_PER_NODE=4 \
    DEVICE=npu:0,1,2,3 \
    python scripts/run_with_log.py \
      --name npu_video20_ddp_r1 \
      --log-file runs/logs/npu_video20_ddp/npu_video20_ddp_r1/npu_video20_ddp_r1.log \
      --md-file runs/logs/npu_video20_ddp/npu_video20_ddp_r1/npu_video20_ddp_r1.md \
      -- bash tools/npu_ddp_detect_benchmark.sh
```

第 2 轮：

```bash
cd /root/workspace/repos/yolov5-project
source /root/.bashrc
conda activate yolo312
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3

env WEIGHTS=/root/workspace/repos/yolov5-project/checkpoint/yolov5_best.pt \
    DATA_YAML=/root/workspace/repos/yolov5-project/data/dataAirVis.yaml \
    SOURCE_PATH=/root/workspace/data/videos/video20.mp4 \
    PROJECT_DIR=/root/workspace/outputs/runs \
    RUN_NAME=npu_video20_ddp_r2 \
    MASTER_PORT=29542 \
    NPROC_PER_NODE=4 \
    DEVICE=npu:0,1,2,3 \
    python scripts/run_with_log.py \
      --name npu_video20_ddp_r2 \
      --log-file runs/logs/npu_video20_ddp/npu_video20_ddp_r2/npu_video20_ddp_r2.log \
      --md-file runs/logs/npu_video20_ddp/npu_video20_ddp_r2/npu_video20_ddp_r2.md \
      -- bash tools/npu_ddp_detect_benchmark.sh
```

第 3 轮：

```bash
cd /root/workspace/repos/yolov5-project
source /root/.bashrc
conda activate yolo312
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3

env WEIGHTS=/root/workspace/repos/yolov5-project/checkpoint/yolov5_best.pt \
    DATA_YAML=/root/workspace/repos/yolov5-project/data/dataAirVis.yaml \
    SOURCE_PATH=/root/workspace/data/videos/video20.mp4 \
    PROJECT_DIR=/root/workspace/outputs/runs \
    RUN_NAME=npu_video20_ddp_r3 \
    MASTER_PORT=29543 \
    NPROC_PER_NODE=4 \
    DEVICE=npu:0,1,2,3 \
    python scripts/run_with_log.py \
      --name npu_video20_ddp_r3 \
      --log-file runs/logs/npu_video20_ddp/npu_video20_ddp_r3/npu_video20_ddp_r3.log \
      --md-file runs/logs/npu_video20_ddp/npu_video20_ddp_r3/npu_video20_ddp_r3.md \
      -- bash tools/npu_ddp_detect_benchmark.sh
```

## 验收标准

每轮都必须满足：

- 返回码 `0`
- `log` 中存在 `INFER init:` 且覆盖 rank `0..3`
- `log` 中存在 `INFER done:` 且 4 个 rank 的 `processed_frames > 0`
- `parallel_inference_summary.txt` 中存在：
  - `world_size=4`
  - `aggregate_frames=...`
  - `parallel_infer_confirmed=true`
- 不产生视频输出文件

## 四卡批缓冲 detect（Phase 1）

批缓冲模式是对四卡并行 detect 的补充路径：

- 支持本地视频文件、目录、glob、txt 路径列表
- 解析后的 source 必须全部是视频文件，不能混入图片
- 必须和 `--ddp-infer` 一起使用
- 当前不支持 RTSP、摄像头、`.streams` 和截图模式
- 额外排队延迟上界约为 $$\frac{\text{buffer\_size} - 1}{\text{fps}} + \text{同步与聚合开销}$$

单次命令：

```bash
cd /root/workspace/repos/yolov5-project
source /root/.bashrc
conda activate yolo312
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3

python scripts/run_with_log.py --name npu_video20_batch_buffer_r1 -- \
  python -m torch.distributed.run --nproc_per_node=4 --master_port=29544 \
    detect.py \
      --weights /root/workspace/repos/yolov5-project/checkpoint/yolov5_best.pt \
      --source /root/workspace/data/videos/video20.mp4 \
      --data /root/workspace/repos/yolov5-project/data/dataAirVis.yaml \
      --imgsz 640 \
      --device npu:0,1,2,3 \
      --ddp-infer --batch-buffer --buffer-size 4 \
      --project /root/workspace/outputs/runs \
      --name npu_video20_batch_buffer_r1 \
      --conf-thres 0.25 --iou-thres 0.45 \
      --save-summary-only --nosave
```

期望日志特征：

- `INFER init: ... infer_mode=batch_buffer buffer_size=4`
- `INFER done: ... processed_batches=...`
- `INFER aggregate: ... infer_mode=batch_buffer buffer_size=4 batch_count=... tail_batch_size=... parallel_infer_confirmed=true`

## 在线流说明（Phase 2）

RTSP、摄像头和其它真实在线流还没有并入 `detect.py` 的批缓冲路径，后续需要单独实现 `dispatcher + workers + ordered sink` 结构。
