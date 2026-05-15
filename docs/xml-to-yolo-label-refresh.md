# XML 转 TXT 执行说明

当前仓库的 XML 转 YOLO TXT 脚本入口放在：

- [scripts/voc_xml_to_yolo.py](/F:/1/yolov5-master/scripts/voc_xml_to_yolo.py)

底层实现复用：

- [tools/label_tools.py](/F:/1/yolov5-master/tools/label_tools.py)

## 执行命令

```powershell
D:\Miniconda3\python.exe F:\1\yolov5-master\scripts\voc_xml_to_yolo.py voc-xml-to-yolo `
  --dataset-root F:\1\labelimg\data\yolo_data_stride3 `
  --data-yaml F:\1\yolov5-master\data\dataAirVis.yaml
```

## 参数说明

- `voc-xml-to-yolo`：调用 XML 转 YOLO TXT 子命令
- `--dataset-root`：VOC 风格数据根目录，目录下应至少包含 `annotations/`，输出会写到同级 `labels/`
- `--data-yaml`：类别映射来源，当前使用仓库里的 `dataAirVis.yaml`

## 可选参数

如果你希望在覆盖已有 `labels/*.txt` 前先备份，可以加：

```powershell
D:\Miniconda3\python.exe F:\1\yolov5-master\scripts\voc_xml_to_yolo.py voc-xml-to-yolo `
  --dataset-root F:\1\labelimg\data\yolo_data_stride3 `
  --data-yaml F:\1\yolov5-master\data\dataAirVis.yaml `
  --backup `
  --backup-suffix .xmlbak
```
