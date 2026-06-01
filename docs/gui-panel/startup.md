# GUI Panel Startup

## 环境

- 依赖安装到 `f312`
- 启动解释器固定使用 `python`（先 `conda activate f312`）

## 安装依赖

```bash
python -m pip install -r requirements.txt
```

## 启动面板

```bash
cd /root/workspace/repos/yolov5-project
python tools/gui_panel/start_gui_panel.py
```

默认地址：

- `http://127.0.0.1:8752/`

## 可选端口

```bash
python tools/gui_panel/start_gui_panel.py --port 8760
```

## 验证

```bash
python -m unittest tests.test_gui_panel tests.test_run_with_log
```
