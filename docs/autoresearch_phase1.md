# AutoResearch Phase 1

本地 Phase 1 只接入一个很薄的 YOLO 运行器，不修改训练核心逻辑。

## 固定前提

- YOLO 仓库：`F:\1\yolov5-master`
- AutoResearch 仓库：`F:\autoresearch`
- 首选 Python：`D:\Miniconda3\envs\yolo\python.exe`
- 当前机器回退 Python：`D:\Miniconda3\python.exe`
- 数据：`data/dataAirVis.yaml`
- 权重：`checkpoint/yolov5_best.pt`

## 基线

```powershell
& 'D:\Miniconda3\python.exe' tools\autoresearch_phase1.py baseline
```

如果当前机器上的 `yolo` 目录不是有效 conda 环境，运行器会自动回退到 `D:\Miniconda3\python.exe`。

基线产物会写到：

- `runs/autoresearch/baseline/`
- `runs/autoresearch/snapshots/`
- `runs/autoresearch/leaderboard/`

## 候选

先准备一个候选规范 YAML，例如 [`tools/autoresearch_phase1.example.yaml`](/F:/1/yolov5-master/tools/autoresearch_phase1.example.yaml)。

```powershell
& 'D:\Miniconda3\python.exe' tools\autoresearch_phase1.py candidate --spec tools\autoresearch_phase1.example.yaml
```

候选运行流程固定为：

1. `smoke`：1 epoch
2. `sprint`：10 epoch
3. 读取 `results.csv` 里的 `mAP@0.5`
4. 与 baseline 或当前 champion 比较
5. 写入 `history.tsv`、快照 YAML、champion YAML

## 快照和记录

- YAML 快照：`runs/autoresearch/snapshots/`
- 生成的临时 hyp：`runs/autoresearch/snapshots/generated_hyps/`
- 记录表：`runs/autoresearch/leaderboard/history.tsv`
- 当前冠军：`runs/autoresearch/leaderboard/champion.yaml`
