# 当前环境配置

更新时间：2026-05-26  
主机：当前 Windows 机器  
仓库路径：`F:\1\yolov5-master`  
默认说明语言：中文

## 1. 当前命中路径总览

以下内容基于当前最终状态整理，优先记录“新开的 PowerShell / cmd 会实际命中什么”。

| 项目 | 当前命中路径 | 主安装目录 | 说明 |
| --- | --- | --- | --- |
| `git` | `C:\Program Files\Git\cmd\git.exe` | `C:\Program Files\Git` | 已迁回 `C:` |
| `python` | `C:\Users\admin\AppData\Local\Programs\Python\Python313\python.exe` | `C:\Users\admin\AppData\Local\Programs\Python\Python313` | 默认 `python` 仍是系统 Python |
| `py` | `C:\Users\admin\AppData\Local\Programs\Python\Launcher\py.exe` | `C:\Users\admin\AppData\Local\Programs\Python\Launcher` | Python Launcher |
| `conda` | `D:\Miniconda3\Scripts\conda.exe` | `D:\Miniconda3` | 已迁到 `D:` |
| `pip` | `C:\Users\admin\AppData\Local\Programs\Python\Python313\Scripts\pip.exe` | 跟随系统 Python | 默认 `pip` 仍指向系统 Python |
| `node` | `D:\Apps\Nodejs\node.exe` | `D:\Apps\Nodejs` | 默认新 shell 已优先命中 `D:` |
| `npm` | `D:\Apps\Nodejs\npm.cmd` | `D:\Apps\Nodejs` | 默认新 shell 已优先命中 `D:` |
| `npx` | `D:\Apps\Nodejs\npx.cmd` | `D:\Apps\Nodejs` | 默认新 shell 已优先命中 `D:` |

当前版本：

- `git version 2.54.0.windows.1`
- `Python 3.13.0`
- `conda 26.3.2`
- `pip 26.1.1`
- `node v24.15.0`
- `npm 11.12.1`
- `npx 11.12.1`

## 2. Conda / Python / pip

### 2.1 Miniconda

- `base` 主环境：`D:\Miniconda3`
- `conda` 程序：`D:\Miniconda3\Scripts\conda.exe`
- 包缓存目录：`D:\Miniconda3\pkgs`
- 默认环境目录：`D:\Miniconda3\envs`
- 用户配置文件：`C:\Users\admin\.condarc`

当前 `conda info` 关键结果：

- `base environment : D:\Miniconda3`
- `package cache : D:\Miniconda3\pkgs`
- `envs directories : D:\Miniconda3\envs`

当前 `conda env list`：

- `base -> D:\Miniconda3`
- `f312 -> D:\Miniconda3\envs\f312`（Python 3.12.13，torch 2.12.0+cu126，fiftyone 1.15.0）
- `py38 -> D:\Miniconda3\envs\py38`

### 2.2 系统 Python

- 安装目录：`C:\Users\admin\AppData\Local\Programs\Python\Python313`
- 当前默认 `python`：`C:\Users\admin\AppData\Local\Programs\Python\Python313\python.exe`
- 当前默认 `pip`：`C:\Users\admin\AppData\Local\Programs\Python\Python313\Scripts\pip.exe`

### 2.3 pip 配置

当前 `pip` 配置：

```ini
global.cache-dir='D:\\pip-cache'
global.index-url='https://mirrors.ustc.edu.cn/pypi/web/simple'
```

结论：

- `pip` 的缓存已固定到 `D:\pip-cache`
- 但默认 `pip` 仍属于系统 Python，而不是 `conda base`

## 3. Git

- 安装目录：`C:\Program Files\Git`
- 当前命中：`C:\Program Files\Git\cmd\git.exe`

当前主要配置来源：

- `C:\Program Files\Git\etc\gitconfig`
- `C:\Users\admin\.gitconfig`
- `F:\1\yolov5-master\.git\config`

当前用户信息：

- `user.name = guyong1449`
- `user.email = 2753856912@qq.com`

## 4. Node.js / npm / npx

### 4.1 当前有效安装

- Node.js 主目录：`D:\Apps\Nodejs`
- `node`：`D:\Apps\Nodejs\node.exe`
- `npm`：`D:\Apps\Nodejs\npm.cmd`
- `npx`：`D:\Apps\Nodejs\npx.cmd`

### 4.2 npm 数据目录

- 全局前缀：`D:\Apps\npm-global`
- 缓存目录：`D:\Apps\npm-cache`

### 4.3 当前生效方式

为保证新开的 shell 默认优先命中 `D:`，当前同时使用了两层用户级入口：

- PowerShell 启动文件：`C:\Users\admin\Documents\WindowsPowerShell\profile.ps1`
- cmd 用户级 AutoRun：`HKCU\Software\Microsoft\Command Processor\AutoRun`

它们都会把以下路径插到前面：

- `D:\Apps\Nodejs`
- `D:\Apps\npm-global`

## 5. 微信 / Tencent 数据目录

### 5.1 程序目录

- 微信程序目录：`D:\Program Files\Tencent\WeChat`
- 当前可见主程序：`D:\Program Files\Tencent\WeChat\Weixin.exe`

### 5.2 用户数据目录

当前已迁到 `D:` 的微信相关用户数据：

- `D:\WeChat Files`
- `D:\TencentRoaming\WeChat`
- `D:\TencentRoaming\xwechat`

当前保留在 `C:` 的是回链入口，不再是主存储位置：

- `C:\Users\admin\Documents\WeChat Files`
- `C:\Users\admin\AppData\Roaming\Tencent\WeChat`
- `C:\Users\admin\AppData\Roaming\Tencent\xwechat`

它们当前都是 Junction：

- `C:\Users\admin\Documents\WeChat Files -> D:\WeChat Files`
- `C:\Users\admin\AppData\Roaming\Tencent\WeChat -> D:\TencentRoaming\WeChat`
- `C:\Users\admin\AppData\Roaming\Tencent\xwechat -> D:\TencentRoaming\xwechat`

当前验证结果：

- 从 `C:` 原路径写入测试文件，会实际落到对应的 `D:` 目录
- 说明目录联接生效，数据重定向已经成立

## 6. 其他应用安装位置

| 应用 | 当前目录 | 说明 |
| --- | --- | --- |
| Claude Code | `D:\Apps\ClaudeCode` | 已迁到 `D:` |
| Obsidian | `D:\Apps\Obsidian` | 已迁到 `D:` |
| Git | `C:\Program Files\Git` | 已迁回 `C:` |
| Miniconda | `D:\Miniconda3` | 已迁到 `D:` |
| Python 3.13 | `C:\Users\admin\AppData\Local\Programs\Python\Python313` | 仍在 `C:` |
| PyCharm Community 2024.2.3 | `C:\Program Files\JetBrains\PyCharm Community Edition 2024.2.3` | 仍在 `C:` |
| VS Code | `C:\Users\admin\AppData\Local\Programs\Microsoft VS Code` | 仍在 `C:` |
| WeChat | `D:\Program Files\Tencent\WeChat` | 程序在 `D:`，数据也已主要落到 `D:` |

## 7. 当前 shell 相关配置

### 7.1 PowerShell

当前用户级执行策略：

- `CurrentUser = RemoteSigned`

当前用户级 PowerShell 启动文件：

- `C:\Users\admin\Documents\WindowsPowerShell\profile.ps1`

作用：

- 新开的 PowerShell 会优先把 `D:\Apps\Nodejs` 和 `D:\Apps\npm-global` 放到 PATH 前面

### 7.2 cmd

当前用户级 AutoRun：

- `HKCU\Software\Microsoft\Command Processor\AutoRun`

作用：

- 新开的 `cmd` 会优先把 `D:\Apps\Nodejs` 和 `D:\Apps\npm-global` 放到 PATH 前面
