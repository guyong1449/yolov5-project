---
name: fiftyone-dedup-workflow
description: 用于当前仓库的 FiftyOne VOC 去重流程。用户提到 FiftyOne、VOC 去重、相似度去重、导入 FiftyOne、导出去重数据集、去重报告时必须使用。优先复用仓库内 tools/fiftyone/ 和 docs/fiftyone-dedup-workflow.md，而不是重新发明命令。
---

# FiftyOne Dedup Workflow

适用前先读：

- `docs/fiftyone-dedup-workflow.md`

执行规则：

1. 先确认用户是在处理 VOC 风格 `data/ + labels/` 数据集。
2. 优先使用 `tools/fiftyone/` 下现成脚本。
3. 运行命令时优先复用文档里的路径、数据集名和 brain key。
4. 若移动或改名脚本，先同步更新文档与测试。
5. 输出时给出可直接复制的完整命令。

常用入口：

- `tools/fiftyone/fiftyone_run_full_dedup_pipeline.py`
- `tools/fiftyone/fiftyone_import_voc.py`
- `tools/fiftyone/fiftyone_compute_similarity.py`
- `tools/fiftyone/fiftyone_deduplicate_dataset.py`
