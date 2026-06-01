# 189 上 GitHub SSH 排查报告

## 结论

**不是 SSH 密钥问题。** 密钥已在 GitHub 生效；失败原因是 **DNS 解析失败 + SSH 默认不走 HTTP 代理**。

代理通路下认证成功示例：

```text
Hi guyong1449! You've successfully authenticated, but GitHub does not provide shell access.
```

前置：本机 ikuuu + 反向隧道在线，且 `189` 已配置 `https_proxy=http://127.0.0.1:17890`。详见 [189-mihomo-usage.md](189-mihomo-usage.md)。

---

## 一、如何区分：密钥 vs 网络/DNS

看 **报错原文**，不要只看 `curl` 能否上网。

| 报错 | 含义 | 与密钥有关 |
|------|------|------------|
| `Could not resolve hostname github.com` | DNS 失败，未连上 GitHub | 否 |
| `Network is unreachable` / `No route to host` | 路由/网关问题 | 否 |
| `Connection timed out` | 能解析但连不上（防火墙、无出口） | 否 |
| `Connection refused` | 能到主机，22 端口不通 | 否 |
| `Host key verification failed` | 主机指纹问题，未到验钥 | 否 |
| `Permission denied (publickey)` | 已连上 GitHub，密钥不被接受 | **是** |
| `Hi xxx! You've successfully authenticated` | 密钥正常 | 密钥 OK |

`Could not resolve hostname` 时 SSH 停在「查 github.com IP」，**未使用密钥**。

---

## 二、网关、DNS 与代理（简要）

```text
189 (192.168.2.189) ──► 网关 192.168.2.1 ──► 外网 ──► GitHub
```

| 概念 | 说明 |
|------|------|
| 默认路由 | 无外网地址时经 `default via 192.168.2.1` 转发；无 default 则直连公网/DNS 失败 |
| DNS | 如 `223.5.5.5`；无网关则访问不到 DNS，`getent hosts github.com` 失败 |
| HTTP 代理 | `.bashrc` 中 `https_proxy=127.0.0.1:17890`；`curl` 走代理可成功 |
| SSH（默认） | **不读** `http_proxy`；仍用系统 DNS + 直连 22 端口 |

| 工具 | 路径 |
|------|------|
| `curl`（已设代理） | 本机 → `17890` → 外网 |
| `ssh`（默认） | 本机 → 系统 DNS → 直连 22 |

---

## 三、排查步骤（按序）

```bash
# 1. DNS
getent hosts github.com
nslookup github.com

# 2. 是否仅靠代理
echo $https_proxy
curl -I https://github.com
env -u http_proxy -u https_proxy curl -I https://github.com

# 3. 密钥（仅在能连 GitHub 时才有意义）
ssh -T git@github.com

# 4. 默认网关
ip route | grep default
```

无 `default` 输出 → 直连 DNS/SSH 易失败；有代理时 `curl` 仍可能成功。

---

## 四、可行方案

### A. SSH 走 HTTP 代理（推荐，与 curl 一致）

编辑 `189` 上 `~/.ssh/config`：

```ssh-config
Host github.com
  HostName github.com
  User git
  ProxyCommand ncat --proxy 127.0.0.1:17890 --proxy-type http %h %p
```

验证：

```bash
ssh -T git@github.com
git clone git@github.com:用户名/仓库名.git
```

需已安装 `ncat`（nmap 包）。

### B. 补默认网关（有网管时）

```bash
ip route add default via 192.168.2.1 dev enp125s0f1
```

网关 IP、网卡名以现场为准。之后 DNS/`ssh` 可不依赖 ProxyCommand。

### C. 临时 hosts（应急，不推荐长期）

GitHub IP 会变；先经代理查 IP 再写入 `/etc/hosts`。

---

## 五、密钥 OK 后的 Git 用法

```bash
git config --global user.name "你的名字"
git config --global user.email "你的邮箱"

git clone git@github.com:用户名/仓库名.git
git add .
git commit -m "说明"
git push
```

---

## 六、状态速查

| 项目 | 当前情况 |
|------|----------|
| SSH 密钥 | OK（`guyong1449` 已认证） |
| 默认网关 | 常无 default，直连外网/DNS 不通 |
| 外网 | 主要靠 `127.0.0.1:17890` HTTP 代理 |
| `ssh` 失败主因 | 不走代理 + DNS 失败 |
| 建议下一步 | 配置 `~/.ssh/config` 的 `ProxyCommand`，或补默认路由 |

---

## 相关文档

- [189-mihomo-usage.md](189-mihomo-usage.md) — 隧道与 `https_proxy`
- [tools/ssh_189/](../tools/ssh_189/) — Windows 侧启动/关闭隧道脚本
