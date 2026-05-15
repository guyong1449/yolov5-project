# 数据集标注复核后的文件更新与工作流

本文面向当前数据集 `F:\1\labelimg\data\yolo_data_stride3`，说明你在 LabelImg 中人工复核、改框、补框、删框之后，哪些文件需要更新，哪些文件通常不需要更新，以及推荐的后续训练/验证工作流。

## 1. 当前数据集的角色划分

当前 `yolo_data_stride3` 目录下已有：

- `annotations/`
- `images/`
- `labels/`
- `train.txt`
- `val.txt`
- `data.yaml`

在这个仓库里，这些文件的职责不是一样的：

- `annotations/*.xml`
  - 这是你当前用 LabelImg 维护的人工标注主源。
  - 从现有目录和文件名看，你现在使用的是 `VOC XML` 格式。
- `labels/*.txt`
  - 这是 YOLOv5 训练直接读取的标签文件。
  - 每一行必须是标准 5 列格式：`class cx cy w h`
- `train.txt` / `val.txt`
  - 这是训练集和验证集的图片清单。
  - `train.py` 会通过这两个清单找到图片，再按图片名映射同名 `labels/*.txt`
- `data.yaml`
  - 这是数据集配置文件。
  - 它定义了数据根目录、`train/val` 清单位置、类别数 `nc`、类别名 `names`

关键结论：

- 你现在人工维护的事实来源是 `annotations/*.xml`
- 但 `train.py` 真正训练时吃的是 `images + labels/*.txt + data.yaml`
- 所以 XML 改完以后，必须把对应的 YOLO 标签重新生成出来

## 2. 哪些文件必须更新

### 2.1 必须更新：`labels/*.txt`

只要你改动了 `annotations/*.xml` 中的框或类别，`labels/*.txt` 就必须重建。

原因：

- YOLOv5 训练不会直接读取 XML
- 如果 XML 已经修正，但 `labels/*.txt` 还是旧内容，那么训练吃到的仍然是旧标签

当前仓库里，推荐使用现有工具从 XML 重建 YOLO 标签：

```powershell
D:\Miniconda3\python.exe F:\1\yolov5-master\tools\label_tools.py voc-xml-to-yolo `
  --dataset-root F:\1\labelimg\data\yolo_data_stride3 `
  --data-yaml F:\1\yolov5-master\data\dataAirVis.yaml
```

如果你希望覆盖旧 `labels/*.txt` 之前先做备份，可以加：

```powershell
D:\Miniconda3\python.exe F:\1\yolov5-master\tools\label_tools.py voc-xml-to-yolo `
  --dataset-root F:\1\labelimg\data\yolo_data_stride3 `
  --data-yaml F:\1\yolov5-master\data\dataAirVis.yaml `
  --backup `
  --backup-suffix .xmlbak
```

## 3. 哪些文件按情况更新

### 3.1 条件更新：`train.txt` / `val.txt`

下面这些情况，通常不需要重建 `train.txt` / `val.txt`：

- 只是改框的位置
- 只是补框或删框
- 只是修正类别
- 图片文件名和图片集合没有变化
- 训练集/验证集划分不变

下面这些情况，需要重建 `train.txt` / `val.txt`：

- 新增了图片
- 删除了图片
- 改了图片文件名
- 想重做 train/val 划分
- 想按新的视频键重新指定验证集

当前 `yolo_data_stride3` 的划分已经是按视频级做的，而不是随机按帧拆：

- `train.txt` 当前来自：`1`、`video12_mp4`、`video15_mp4`
- `val.txt` 当前来自：`video13_mp4`

这能避免“同一视频相邻帧同时出现在 train 和 val”的最严重污染，但仍不等于验证集一定足够有挑战。

### 3.2 条件更新：`data.yaml`

下面这些情况，通常不需要改 `data.yaml`：

- 只是改框
- 只是改标签内容
- `train.txt` / `val.txt` 文件名不变
- 类别总数和类别名不变

下面这些情况，需要改 `data.yaml`：

- 数据根目录变了
- `train.txt` / `val.txt` 文件名或位置变了
- `nc` 变了
- `names` 变了

## 4. 推荐工作流

对于当前 `F:\1\labelimg\data\yolo_data_stride3`，推荐工作流是：

1. 在 LabelImg 中复核 `annotations/*.xml`
2. 用仓库工具把 XML 重建成 `labels/*.txt`
3. 如果图片集合或划分变化了，再重建 `train.txt` / `val.txt`
4. 如果类别定义或路径配置变化了，再更新 `data.yaml`
5. 再进入训练或验证

最常见的实际场景是：

- 你只是在 XML 中改框
- 图片没变
- train/val 划分没变
- 类别体系没变

这种情况下，通常只需要：

1. 复核 XML
2. 重建 `labels/*.txt`
3. 直接训练/验证

## 5. 关于当前验证集的建议

当前 `val` 已按视频级拆分，这是正确方向，但还要区分两种验证目标：

- 开发验证集
  - 用于日常看训练有没有退化、有没有提升
  - 可以继续使用当前 `val.txt`
- 外部分布验证集或测试集
  - 用于判断模型在新视频、新场景、新轨迹条件下的泛化能力
  - 不建议和当前训练视频同源

原因是：

- 即使按视频级拆分，训练集和验证集仍可能来自非常相近的拍摄条件、目标尺度、背景和轨迹模式
- 这样得到的指标可能对“同分布效果”较敏感，但对“新视频泛化能力”判断不够强

因此更稳妥的做法是：

- 保留当前 `val.txt` 作为开发验证集
- 另外从新的原始视频域单独构建一份外部分布 `val/test`

例如你之前提到的 `F:\1\video\output\video1_3`，更适合用于：

- 先跑 `detect.py` 生成候选帧和初始框
- 再人工复核成新的 XML 真值
- 最后把它作为独立的新域 `val` 或 `test`

注意：

- `detect.py` 产出的预测结果不能直接当真值验证集
- 真正可用于训练或验证的数据，仍然需要人工确认后的标注

## 6. 训练前最小检查

在 XML 复核和标签重建完成后，建议最少检查以下几点：

- `annotations/*.xml` 与 `images/*.jpg` stem 对齐
- `labels/*.txt` 与 `images/*.jpg` stem 对齐
- `labels/*.txt` 每行是 5 列，而不是带 `conf` 的 6 列
- `train.txt` / `val.txt` 指向的图片都存在
- `data.yaml` 的 `nc` 与 `names` 一致

如果只是想先做一次最小链路体检，可以再跑一次轻量训练或验证冒烟，而不是直接开始长时间正式训练。

## 7. 一句话总结

你现在的人工标注主源是 `VOC XML`，而训练直接输入是 YOLO `txt`。因此，XML 改完以后，最关键的更新动作是重建 `labels/*.txt`；只有当图片集合、划分策略或类别配置发生变化时，才需要继续重建 `train.txt`、`val.txt` 或更新 `data.yaml`。
