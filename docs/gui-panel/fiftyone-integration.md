# GUI Panel FiftyOne Integration

## 两种输入模式

- `dataset_root_auto`
  面板只接收一个数据集根目录，后端自动识别目录布局
- `explicit_voc`
  面板显式接收 `data_dir` 和 `labels_dir`

## 自动识别优先级

1. `<dataset_root>/fiftyone_voc/data` + `<dataset_root>/fiftyone_voc/labels`
2. `<dataset_root>/images` + `<dataset_root>/annotations`

未命中任一布局时，后端返回结构化错误，前端只展示错误，不猜路径。

## 导入命令

- 使用当前启动面板的 Python 解释器
- 调用 `tools/fiftyone/fiftyone_import_voc.py`
- 可选 `--overwrite`
- `launch_app=false` 时追加 `--no-app`

## 结果提取

从导入输出中提取：

- `dataset_name=...`
- `samples_count=...`
- `session_url=...`

这些字段会回写到运行状态和任务卡摘要。
