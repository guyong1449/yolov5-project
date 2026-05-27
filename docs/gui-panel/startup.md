# GUI Panel Startup

## 环境

- 依赖安装到 `f312`
- 启动解释器固定使用 `D:\Miniconda3\envs\f312\python.exe`

## 安装依赖

```powershell
D:\Miniconda3\envs\f312\python.exe -m pip install -r requirements.txt
```

## 启动面板

```powershell
cd F:\1\yolov5-master
D:\Miniconda3\envs\f312\python.exe tools\gui_panel\start_gui_panel.py
```

默认地址：

- `http://127.0.0.1:8752/`

## 可选端口

```powershell
D:\Miniconda3\envs\f312\python.exe tools\gui_panel\start_gui_panel.py --port 8760
```

## 验证

```powershell
D:\Miniconda3\envs\f312\python.exe -m unittest tests.test_gui_panel tests.test_run_with_log
```
