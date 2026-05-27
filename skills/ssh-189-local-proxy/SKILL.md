---
name: ssh-189-local-proxy
description: 用于当前仓库里 ssh 别名 189 借用本机代理出网的流程。用户提到 189、ikuuu、反向隧道、本机代理、中转订阅更新、让 189 访问外网时必须使用。优先复用 docs/189-mihomo-usage.md、docs/189-github-ssh.md 和 tools/ssh_189/，不要重新组织一套新步骤。
---

# SSH 189 Local Proxy

适用前先读：

- `docs/189-mihomo-usage.md`
- `docs/189-github-ssh.md`（GitHub SSH / DNS / 密钥区分）

执行规则：

1. 先确认本机代理入口仍是 `127.0.0.1:7890`。
2. 优先使用 `tools/ssh_189/` 下现成 bat 脚本启动和关闭隧道。
3. 明确告诉用户隧道窗口和本机代理都必须保持在线。
4. 在 189 上需要代理时，优先给出单条命令或 `export http_proxy/https_proxy` 的最小做法。
5. 若脚本位置变化，先同步更新文档。

常用入口：

- `tools/ssh_189/start-189-local-proxy-tunnel.bat`
- `tools/ssh_189/stop-189-local-proxy-tunnel.bat`
