# NPU Detect 数据通路与流程

## 目的

这份文档描述当前仓库里 `detect.py` 的输入、分支、执行路径、日志与产物输出，重点覆盖两条实际可用路径：

- 单进程 NPU detect 推理
- 4 进程 NPU 并行 detect 推理（按帧分片）

当前实测输入固定为：

- 视频源：`/root/workspace/data/videos/video20.mp4`
- 权重：`/root/workspace/repos/yolov5-project/checkpoint/yolov5_best.pt`
- 类别配置：`/root/workspace/repos/yolov5-project/data/dataAirVis.yaml`

## 文件职责

- [detect.py](/root/workspace/repos/yolov5-project/detect.py)
  - 解析 detect CLI
  - 决定单进程还是并行推理
  - 负责视频读取、前向推理、NMS、日志、按需落盘
- [scripts/run_with_log.py](/root/workspace/repos/yolov5-project/scripts/run_with_log.py)
  - 捕获 stdout/stderr
  - 同步写入 `.log` 和 `.md`
  - 为并行 detect 生成 `*_parallel_inference_summary.txt`
- [tools/npu_ddp_detect_benchmark.sh](/root/workspace/repos/yolov5-project/tools/npu_ddp_detect_benchmark.sh)
  - 统一设置 4 卡环境变量
  - 通过 `torch.distributed.run` 启动 4 个 detect 进程
- [docs/npu-detect-run-commands.md](/root/workspace/repos/yolov5-project/docs/npu-detect-run-commands.md)
  - 运行命令手册

## 单进程路径

单进程路径由 `detect.py` 默认命令走通，不需要 `--ddp-infer`。

```text
CLI
  -> parse_opt()
  -> main(opt)
  -> run(...)
  -> select_device(opt.device)
  -> DetectMultiBackend(...)
  -> LoadImages/LoadStreams/LoadScreenshots
  -> model.warmup(...)
  -> for each frame/image
       -> preprocess
       -> model(im)
       -> non_max_suppression(...)
       -> draw/save/log
  -> Speed line / Results saved
```

单进程下，`detect.py` 的默认设备字符串是 `npu:0,1,2,3`，但 `select_device()` 仍会把它解析成首张可见卡作为实际执行设备。

## 四卡并行路径

四卡并行路径需要 `torch.distributed.run --nproc_per_node 4` 启动，并显式传 `--ddp-infer`。

### 启动阶段

```text
shell script
  -> torch.distributed.run (4 processes)
  -> detect.py --ddp-infer --frame-shard-mode mod --save-summary-only --nosave
```

每个 rank 在启动时都会读取环境变量：

- `LOCAL_RANK`
- `RANK`
- `WORLD_SIZE`

然后执行：

```text
initialize_parallel_inference()
  -> 验证 torch.npu 可用
  -> 验证 HCCL 可用
  -> dist.init_process_group(backend='hccl')
  -> torch.npu.set_device(LOCAL_RANK)
  -> 返回 torch.device('npu', LOCAL_RANK)
```

### 帧分片规则

当前并行推理采用固定的取模分片：

```text
frame_index % WORLD_SIZE == RANK
```

含义如下：

- rank 0 处理 `0, 4, 8, ...`
- rank 1 处理 `1, 5, 9, ...`
- rank 2 处理 `2, 6, 10, ...`
- rank 3 处理 `3, 7, 11, ...`

这样做的结果是：

- 四个进程都会真正前向推理
- 单个视频也能分摊到 4 卡
- 合并后覆盖原始完整帧序列

### rank 内部流程

每个 rank 的循环仍然沿用原有 detect 逻辑，只是先做分片判断：

```text
for each frame
  -> 计算当前 frame_index
  -> 若 frame_index % 4 != rank，则跳过
  -> preprocess
  -> model(im)
  -> non_max_suppression(...)
  -> 仅记录日志，不输出视频
```

每个 rank 会打印：

- `INFER init: rank=... local_rank=... world_size=4 ...`
- `INFER done: rank=... processed_frames=...`

rank 0 额外打印聚合结果：

- `INFER aggregate: world_size=4 rank_frame_counts=... aggregate_frames=4165 parallel_infer_confirmed=true`

## 数据路径

当前实测里，数据从命令行流入 detect 的路径如下：

```text
/root/workspace/data/videos/video20.mp4
  -> tools/npu_ddp_detect_benchmark.sh
  -> scripts/run_with_log.py
  -> detect.py --ddp-infer
  -> normalize_source_for_detect(source)
  -> LoadImages(source)
  -> OpenCV video reader
  -> per-frame shard decision
  -> model inference
  -> NMS
  -> rank0 aggregate summary
```

`data/dataAirVis.yaml` 在这里主要提供类别名与数据集元信息，不承担训练集/验证集分割职责。

## 产物路径

### 单进程 detect

默认产物会落到 `--project/--name` 对应目录，例如：

- `/root/workspace/outputs/runs/vis_boxed_video`

### 四卡并行 detect

三轮实测的日志与摘要固定落在：

- `runs/logs/npu_video20_ddp/npu_video20_ddp_r1/`
- `runs/logs/npu_video20_ddp/npu_video20_ddp_r2/`
- `runs/logs/npu_video20_ddp/npu_video20_ddp_r3/`

每轮目录包含：

- `<run>.log`
- `<run>.md`
- `<run>_parallel_inference_summary.txt`

## 验收信号

判断“真四卡并行推理”时，不看默认设备字符串本身，而看这些信号：

- `npu-smi info` 里同时有 4 个 `python` 进程
- 日志里有 `INFER init`
- 聚合摘要里有 `world_size=4`
- 聚合摘要里有 `rank_frame_counts=0:1041,1:1042,2:1041,3:1041`
- 聚合摘要里有 `aggregate_frames=4165`
- 聚合摘要里有 `parallel_infer_confirmed=true`

