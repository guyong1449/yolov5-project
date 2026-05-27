# FiftyOne 去重流程

适用场景：

- 已有 VOC 风格目录 `data/ + labels/`
- 需要导入 FiftyOne
- 需要做相似度计算、精确去重、近似去重、导出和报告

## 文件位置

- 总控脚本：[tools/fiftyone/fiftyone_run_full_dedup_pipeline.py](/abs/path/F:/1/yolov5-master/tools/fiftyone/fiftyone_run_full_dedup_pipeline.py)
- 导入脚本：[tools/fiftyone/fiftyone_import_voc.py](/abs/path/F:/1/yolov5-master/tools/fiftyone/fiftyone_import_voc.py)
- 相似度脚本：[tools/fiftyone/fiftyone_compute_similarity.py](/abs/path/F:/1/yolov5-master/tools/fiftyone/fiftyone_compute_similarity.py)
- 去重脚本：[tools/fiftyone/fiftyone_deduplicate_dataset.py](/abs/path/F:/1/yolov5-master/tools/fiftyone/fiftyone_deduplicate_dataset.py)
- 启动器：[tools/fiftyone/start_fiftyone_voc.ps1](/abs/path/F:/1/yolov5-master/tools/fiftyone/start_fiftyone_voc.ps1)

## 当前约定

- 源数据集名：`test1_stride10_voc`
- 原始 VOC 根目录：`F:\1\labelimg\data\test1_stride10\fiftyone_voc`
- 导出目录：`F:\1\labelimg\data\test1_stride10\fiftyone_voc_deduped`
- 报告目录：`F:\1\labelimg\data\test1_stride10\fiftyone_voc\dedup_reports`
- 精确重复备份目录：`F:\1\labelimg\data\test1_stride10\fiftyone_voc\backup_removed_exact`

## 环境

- 真实运行：`D:\Miniconda3\envs\f312\python.exe`
- 单测运行：`D:\Miniconda3\envs\yolo\Scripts\python.exe -m unittest tests.test_fiftyone_tools`

## 主命令

```powershell
D:\Miniconda3\envs\f312\python.exe tools\fiftyone\fiftyone_run_full_dedup_pipeline.py `
  --dataset-name test1_stride10_voc `
  --model clip-vit-base32-torch `
  --brain-key clip_vit_base32_sim `
  --approx-threshold 0.12 `
  --approx-group-keep-ratio 0.3 `
  --voc-root "F:\1\labelimg\data\test1_stride10\fiftyone_voc" `
  --export-dir "F:\1\labelimg\data\test1_stride10\fiftyone_voc_deduped" `
  --report-dir "F:\1\labelimg\data\test1_stride10\fiftyone_voc\dedup_reports" `
  --overwrite
```

流程：

1. 检查或重建源数据集。
2. 计算 `clip_vit_base32_sim` 相似度。
3. 克隆临时工作数据集。
4. 精确去重并把原始重复样本移动到 `backup_removed_exact`。
5. 近似去重。
6. 导出去重后的 VOC 数据集。
7. 写出 CSV、JSON 和图表。

## 分步命令

### 导入 VOC

```powershell
D:\Miniconda3\envs\f312\python.exe tools\fiftyone\fiftyone_import_voc.py `
  --name test1_stride10_voc `
  --data-dir "F:\1\labelimg\data\test1_stride10\fiftyone_voc\data" `
  --labels-dir "F:\1\labelimg\data\test1_stride10\fiftyone_voc\labels" `
  --overwrite
```

### 计算相似度

```powershell
D:\Miniconda3\envs\f312\python.exe tools\fiftyone\fiftyone_compute_similarity.py `
  --dataset-name test1_stride10_voc `
  --model clip-vit-base32-torch `
  --brain-key clip_vit_base32_sim `
  --overwrite
```

### 单独去重导出

```powershell
D:\Miniconda3\envs\f312\python.exe tools\fiftyone\fiftyone_deduplicate_dataset.py `
  --dataset-name test1_stride10_voc `
  --export-dir "F:\1\labelimg\data\test1_stride10\fiftyone_voc_deduped" `
  --report-dir "F:\1\labelimg\data\test1_stride10\fiftyone_voc\dedup_reports" `
  --label-field ground_truth `
  --exact-mode deduplicate `
  --approx-brain-key clip_vit_base32_sim `
  --approx-threshold 0.12 `
  --approx-group-keep-ratio 0.3 `
  --overwrite
```

## 验收

- `report_data/` 下存在报告文件
- `dedup_reports/` 顶层存在统计 CSV 和图表
- `fiftyone_voc_deduped/data` 与 `labels` 数量一致
- `backup_removed_exact/data` 与 `labels` 数量一致
