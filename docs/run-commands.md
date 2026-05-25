# 运行命令

当前目录规则：

- 训练核心文件保留在 `runs/train/<run_name>/`
- 日志默认写入 `runs/train/<run_name>/run.log` 与 `run.md`
- 所有训练图片统一写入 `runs/train/<run_name>/images/`

## 当前推荐命令

基于 [runs/train/test1_stride10_sgd_70e3/opt.yaml](/abs/path/f:/1/yolov5-master/runs/train/test1_stride10_sgd_70e3/opt.yaml:1) 对应配置，当前保留主线命令如下：

```bash
cd /f/1/yolov5-master

D:/Miniconda3/python.exe scripts/run_with_log.py -- \
  D:/Miniconda3/python.exe train.py \
  --data F:/1/labelimg/data/test1_stride10/data.yaml \
  --weights checkpoint/yolov5_best.pt \
  --epochs 70 \
  --batch-size 4 \
  --imgsz 640 \
  --device 0 \
  --seed 0 \
  --workers 2 \
  --patience 20 \
  --optimizer SGD \
  --project runs/train \
  --name test1_stride10_sgd_70e
```

说明：

- 本命令在当前代码下会把实际训练目录增量落到 `runs/train/test1_stride10_sgd_70e3/`
- `run_with_log.py` 对 `train.py` 已默认并入训练目录，不必再单独指定 `runs/logs/...`
- 如需避免继续递增目录，请只在明确允许覆盖时再加 `--exist-ok`

## VOC 抽帧导出（detect.py，`vid-stride=10`）

从 `F:/1/video/output` 扫视频，每 10 帧取 1 帧，把**原始帧**写入 VOC `images/`、检测框写入 `annotations/`（与训练集 `test1_stride10` 同根目录，便于后续 LabelImg 复核与转 YOLO 标签）。

```bash
cd /f/1/yolov5-master

D:/Miniconda3/python.exe detect.py \
  --weights checkpoint/yolov5_best.pt \
  --source "F:/1/video/output" \
  --data F:/1/labelimg/data/test1_stride10/data.yaml \
  --imgsz 640 \
  --device 0 \
  --project runs/detect \
  --name voc_stride10 \
  --voc-root F:/1/labelimg/data/test1_stride10 \
  --vid-stride 10 \
  --save-img-frames \
  --nosave \
  --incremental-mp4
```

产出目录（`--voc-root` 下）：

- `F:/1/labelimg/data/test1_stride10/images/*.jpg`：无框原始帧
- `F:/1/labelimg/data/test1_stride10/annotations/*.xml`：PASCAL VOC 初始框（需人工复核后才可当训练真值）

说明：

- **默认不弹窗预览**；需要 `cv2.imshow` 实时看画面时再显式加 `--view-img`
- `--source` 为目录时，本仓库会递归扫描其下支持的视频后缀；目录已存在则向同一 `images/`、`annotations/` **追加**写入
- `--nosave`：不在 `runs/detect/...` 再落带框预览图/视频，避免与 VOC 目录混淆
- `--incremental-mp4`：只转换状态文件中尚未记录的视频（状态见 `{voc-root}/.yolov5_mp4_convert_state.json`）；首次全量导出可去掉该参数
- 也可用封装脚本（默认 `--vid-stride 10`）：

```bash
D:/Miniconda3/python.exe scripts/extract_voc_stride10.py \
  --weights checkpoint/yolov5_best.pt \
  --source "F:/1/video/output" \
  --voc-root F:/1/labelimg/data/test1_stride10 \
  --data-yaml F:/1/labelimg/data/test1_stride10/data.yaml \
  --device 0
```

