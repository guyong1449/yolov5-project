# Detect 4-Card Parallel Inference Plan

## Summary

- 将 `detect.py` 的默认设备改为 `npu:0,1,2,3`
- 新增真实四卡并行推理模式，使用 `torch.distributed.run` 拉起 4 个 NPU 进程
- 单个 `video20.mp4` 采用按帧取模分片：`frame_index % world_size == rank`
- 新增并行推理摘要 `.txt`，确保能证明 rank 0-3 都实际参与并处理了非零帧数
- 三轮实测使用 `/root/workspace/data/videos/video20.mp4`，不输出视频

## Key Changes

- `detect.py`
  - 增加 `--ddp-infer`
  - 增加 `--frame-shard-mode mod`
  - 增加 `--save-summary-only`
  - 增加 `INFER init` / `INFER done` / `INFER aggregate` 日志
- `scripts/run_with_log.py`
  - 识别 `tools/npu_ddp_detect_benchmark.sh`
  - 生成 `<run>_parallel_inference_summary.txt`
- `tools/npu_ddp_detect_benchmark.sh`
  - 四卡并行 detect 包装脚本
- `docs/npu-detect-run-commands.md`
  - 更新为“单进程 detect + 四卡并行 detect”并存说明

## Test Plan

- `pytest tests/test_npu_benchmark_dryrun.py tests/test_npu_ddp_support.py tests/test_run_with_log.py`
- 验证 `detect.py` 默认设备为 `npu:0,1,2,3`
- 验证帧分片逻辑覆盖 rank 0-3 且合并后覆盖全集
- 验证并行推理摘要需要 4 个 rank 和非零帧数才判定成功
- 真实三轮 `/root/workspace/data/videos/video20.mp4` 四卡实测

## Acceptance

- 每轮返回码 `0`
- 每轮日志包含 rank 0-3 的 `INFER init`
- 每轮日志包含 rank 0-3 的 `INFER done`
- 每轮摘要包含 `world_size=4` 与 `parallel_infer_confirmed=true`
- 不输出视频文件
