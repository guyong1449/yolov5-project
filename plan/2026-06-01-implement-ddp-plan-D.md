---
name: DDP Batch Buffer Online Inference
overview: 在现有 DDP 帧级分片推理基础上，实现方案D（批缓冲）：帧到达 → 积攒 N 帧组成 micro-batch → 分发给 N 个 rank 并行推理 → 按原始顺序聚合结果输出。相比当前的静态分片，批缓冲模式更适合单路实时视频流场景。
todos:
  - id: "1"
    content: 新建 utils/frame_buffer.py：实现 FrameBuffer 和 DistributedBatchInference 类
    status: pending
  - id: "2"
    content: 修改 detect.py：新增 --batch-buffer/--buffer-size 参数
    status: pending
  - id: "3"
    content: 修改 detect.py：推理循环增加批缓冲模式分支
    status: pending
  - id: "4"
    content: 修改 scripts/run_with_log.py：解析批缓冲模式日志
    status: pending
  - id: "5"
    content: 运行 benchmark 验证批缓冲模式
    status: pending
  - id: "6"
    content: 更新文档记录启动命令和结果
    status: pending
isProject: false
---

## 方案D（批缓冲）详细设计

### 核心架构

```
帧到达 → FrameBuffer(积攒N帧) → micro-batch分发 → N个rank并行推理 → 按序聚合输出
```

### 与现有代码的关系

**现有实现**（[detect.py](detect.py) L314-328）：
- `--ddp-infer` 启用多进程推理
- `should_process_frame()` 按 `frame_index % world_size == rank` 静态分片
- 每个 rank 独立处理自己的帧，最后 `all_gather_object` 汇总计数

**方案D改动**：
- 新增 `--batch-buffer` 模式，替代静态分片
- rank 0 作为 dispatcher 收集帧，组成 micro-batch 后分发
- 所有 rank 参与每一批的推理（每 rank 处理 batch 中一帧）
- rank 0 负责按序聚合结果

### 实现步骤

#### 1. 新增 `FrameBuffer` 类（新建 `utils/frame_buffer.py`）

```python
class FrameBuffer:
    """积攒帧组成 micro-batch，支持按序输出。"""
    def __init__(self, batch_size: int):
        self.batch_size = batch_size
        self.buffer = []  # [(frame_idx, path, im_tensor, im0s, vid_cap, s)]
    
    def add(self, frame_idx, path, im, im0s, vid_cap, s) -> Optional[list]:
        """添加帧，满时返回 micro-batch，否则返回 None。"""
        self.buffer.append((frame_idx, path, im, im0s, vid_cap, s))
        if len(self.buffer) >= self.batch_size:
            batch = self.buffer[:]
            self.buffer.clear()
            return batch
        return None
    
    def flush(self) -> list:
        """返回剩余帧（视频结束时调用）。"""
        remaining = self.buffer[:]
        self.buffer.clear()
        return remaining
```

#### 2. 新增 `DistributedBatchInference` 类（同文件）

```python
class DistributedBatchInference:
    """管理 rank 间 micro-batch 分发与结果聚合。"""
    def __init__(self, world_size, rank, device):
        self.world_size = world_size
        self.rank = rank
        self.device = device
    
    def distribute_batch(self, batch):
        """rank 0 将 batch scatter 给所有 rank，其他 rank 接收。"""
        # 使用 dist.scatter，每 rank 收到 1 帧
        ...
    
    def gather_results(self, local_result):
        """各 rank 将推理结果 gather 回 rank 0。"""
        # 使用 dist.gather
        ...
    
    def aggregate_ordered(self, gathered_results, original_order):
        """rank 0 按原始帧序号重排结果。"""
        ...
```

#### 3. 修改 [detect.py](detect.py) 推理循环

**新增 CLI 参数**（L537-542 附近）：
```python
parser.add_argument('--batch-buffer', action='store_true',
                    help='启用批缓冲模式：积攒N帧后分发给多rank并行推理')
parser.add_argument('--buffer-size', type=int, default=0,
                    help='批缓冲大小，默认等于 world_size')
```

**修改 `run()` 函数**（L314-328 附近）：

```python
if batch_buffer:
    # 批缓冲模式
    buffer = FrameBuffer(buffer_size or WORLD_SIZE)
    batch_engine = DistributedBatchInference(WORLD_SIZE, RANK, device)
    
    for path, im, im0s, vid_cap, s in dataset:
        # 预处理
        im_tensor = preprocess(im, device, model.fp16)
        frame_idx = getattr(dataset, 'frame', 0)
        
        # 积攒帧
        batch = buffer.add(frame_idx, path, im_tensor, im0s, vid_cap, s)
        if batch is not None:
            # 分发 → 推理 → 聚合
            local_frame = batch_engine.distribute_batch(batch)
            pred = model(local_frame['tensor'][None])
            batch_engine.gather_results(pred)
    
    # 处理剩余帧
    remaining = buffer.flush()
    if remaining:
        batch_engine.process_remaining(remaining)
        
elif ddp_infer:
    # 现有静态分片逻辑（保持不变）
    ...
```

#### 4. 修改 [utils/torch_utils.py](utils/torch_utils.py)

`select_device()` 已支持多卡解析（L135-163），无需修改。

#### 5. 修改 [scripts/run_with_log.py](scripts/run_with_log.py)

在 `_parallel_detect_summary()` 中添加批缓冲模式的解析：
- 检测 `--batch-buffer` 参数
- 记录 buffer_size、batch 处理数量等指标

#### 6. 更新启动命令

```bash
# 批缓冲模式启动命令
python scripts/run_with_log.py --name npu_batch_buffer_r1 -- \
  torch.distributed.run --nproc_per_node=4 \
  python detect.py \
    --weights checkpoint/yolov5_best.pt \
    --source /root/workspace/data/videos/video20.mp4 \
    --data data/dataAirVis.yaml \
    --imgsz 640 \
    --device npu:0,1,2,3 \
    --ddp-infer --batch-buffer --buffer-size 4 \
    --project /root/workspace/outputs/runs \
    --name npu_batch_buffer_r1 \
    --conf-thres 0.25 --iou-thres 0.45 \
    --nosave
```

### 关键设计决策

| 决策点 | 选择 | 理由 |
|--------|------|------|
| 分发方式 | `dist.scatter` | 一次通信完成，延迟低 |
| 聚合方式 | `dist.gather` | rank 0 集中处理，简化逻辑 |
| 缓冲区位置 | rank 0 本地 | 避免跨 rank 共享状态 |
| 帧序号维护 | 原始 frame_idx 传递 | 确保输出顺序正确 |
| 剩余帧处理 | flush() 强制推理 | 避免视频末尾帧丢失 |

### 延迟分析

- 延迟 = buffer_size × 帧间隔
- 例：30fps 视频，buffer_size=4 → 延迟 ≈ 133ms
- 适合视频会议、直播标注等可容忍少量延迟的场景

### 验证要点

1. 输出帧序与输入一致（无乱序）
2. 所有 rank 参与每批推理（日志显示每 rank 处理帧数 > 0）
3. 总帧数与静态分片模式一致
4. 延迟符合 buffer_size × 帧间隔 预期
5. 视频末尾帧不丢失（flush 机制生效）

### 文件变更清单

| 文件 | 变更类型 | 说明 |
|------|----------|------|
| `utils/frame_buffer.py` | 新建 | FrameBuffer + DistributedBatchInference 类 |
| `detect.py` | 修改 | 新增 --batch-buffer/--buffer-size 参数，推理循环增加批缓冲分支 |
| `scripts/run_with_log.py` | 修改 | 解析批缓冲模式日志指标 |
| `docs/npu-detect-run-commands.md` | 修改 | 记录批缓冲模式启动命令和 benchmark 结果 |

### 依赖关系

```
utils/frame_buffer.py (新建)
    ↓
detect.py (修改) ← 需要 import FrameBuffer, DistributedBatchInference
    ↓
scripts/run_with_log.py (修改) ← 解析新增的日志格式
    ↓
benchmark 执行 → 验证 → 文档更新
```