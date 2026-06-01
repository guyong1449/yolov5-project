# ssh 189 本机代理中转

## 适用场景

- `189` 自身不能稳定访问外网
- 借用当前 Windows 本机的 ikuuu / Clash 兼容代理
- 给 `189` 做订阅更新、浏览器访问或 shell 命令出网

**限制**：依赖 Windows 代理与隧道在线；本机关代理或隧道断开时，`189` 无法独立出网。

## 链路

| 环节 | 地址 |
|------|------|
| 本机代理 | `127.0.0.1:7890` |
| 反向隧道（189 侧） | `127.0.0.1:17890` |
| 189 上使用的代理 URL | `http://127.0.0.1:17890` |

---

## 默认出网（三层都要满足）

### 1. Windows 侧

1. ikuuu / Clash 保持运行，HTTP 代理在 `127.0.0.1:7890`
2. 反向隧道保持运行（窗口勿关）：
   - 双击 `start-189-local-proxy-tunnel.bat`，或
   - ```powershell
     ssh -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=3 -N -R 17890:127.0.0.1:7890 189
     ```

长期稳定出网时，建议用独立隧道窗口，不要只靠交互式 `ssh 189` 会话里的转发。

### 2. 189 侧：登录 shell 默认走代理（一次性）

```bash
cat >> ~/.bashrc << 'EOF'

# 借用 Windows 本机代理（需隧道 + ikuuu 在线）
export http_proxy=http://127.0.0.1:17890
export https_proxy=http://127.0.0.1:17890
export HTTP_PROXY=http://127.0.0.1:17890
export HTTPS_PROXY=http://127.0.0.1:17890
export no_proxy=localhost,127.0.0.1,192.168.0.0/16,10.0.0.0/8
export NO_PROXY="$no_proxy"
EOF
```

重新登录或 `source ~/.bashrc` 后，`curl` / `wget` / `pip` 等默认走代理。

验证：

```bash
curl -I https://www.google.com
```

### 3. 可选：`ssh 189` 时自动建隧道

在 `~/.ssh/config` 的 `Host 189` 中增加（或单独建 `189-proxy` 别名）：

```sshconfig
  RemoteForward 17890 127.0.0.1:7890
  ExitOnForwardFailure yes
  ServerAliveInterval 30
  ServerAliveCountMax 3
```

SSH 断开时转发通常一并断开；与第 1 步独立隧道二选一或并用时，以独立隧道为准。

---

## 速查

| 目标 | 操作 |
|------|------|
| 189 命令默认出网 | `~/.bashrc` 设置 `http_proxy` / `https_proxy` |
| 189 能连上代理 | Windows 开 ikuuu + 保持隧道 |
| 取消当前 shell 代理 | `unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY` |
| 单条命令走代理 | `curl --proxy http://127.0.0.1:17890 -I https://www.google.com` |
| 更新 mihomo 订阅 | `ssh 189 "/usr/local/bin/mihomo-update.sh"` |

订阅脚本已配置 `SUBSCRIPTION_PROXY_URL='http://127.0.0.1:17890'`，前提仍是本机代理与隧道在线。

## 相关

- GitHub SSH（DNS / 密钥 / ProxyCommand）：[189-github-ssh.md](189-github-ssh.md)
