# ssh 189 本机代理中转

适用场景：

- `189` 自身不能稳定访问外网
- 需要借用当前 Windows 本机的 ikuuu / Clash 兼容代理
- 需要临时给 `189` 执行订阅更新、浏览器访问或单条命令出网

## 文件位置

- 启动隧道：[tools/ssh_189/start-189-local-proxy-tunnel.bat](/abs/path/F:/1/yolov5-master/tools/ssh_189/start-189-local-proxy-tunnel.bat)
- 关闭隧道：[tools/ssh_189/stop-189-local-proxy-tunnel.bat](/abs/path/F:/1/yolov5-master/tools/ssh_189/stop-189-local-proxy-tunnel.bat)

## 当前链路

- 本机代理入口：`127.0.0.1:7890`
- 反向隧道目标：`189:127.0.0.1:17890`
- `189` 上使用的代理地址：`http://127.0.0.1:17890`

## 启动

直接双击：

- [start-189-local-proxy-tunnel.bat](/abs/path/F:/1/yolov5-master/tools/ssh_189/start-189-local-proxy-tunnel.bat)

或手工执行：

```powershell
ssh -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=3 -N -R 17890:127.0.0.1:7890 189
```

说明：

- 这个窗口必须持续保持开启
- 本机 `ikuuu` 或其他 Clash 兼容代理也必须保持运行

## 在 189 上使用

单条命令走代理：

```bash
curl --proxy http://127.0.0.1:17890 -I https://www.google.com
```

当前 shell 临时走代理：

```bash
export http_proxy=http://127.0.0.1:17890
export https_proxy=http://127.0.0.1:17890
```

取消：

```bash
unset http_proxy https_proxy
```

## 关闭

直接双击：

- [stop-189-local-proxy-tunnel.bat](/abs/path/F:/1/yolov5-master/tools/ssh_189/stop-189-local-proxy-tunnel.bat)

或直接关闭运行隧道的终端窗口。

## 常用操作

测试 Google：

```bash
curl --proxy http://127.0.0.1:17890 -I https://www.google.com
```

手动更新订阅：

```bash
ssh 189 "/usr/local/bin/mihomo-update.sh"
```

说明：

- 当前远端更新脚本已按 `SUBSCRIPTION_PROXY_URL='http://127.0.0.1:17890'` 使用这条链路
- 前提仍是本机代理和反向隧道在线
