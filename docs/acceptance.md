# 验收说明

## 按原题逐项核对

| 题目要求 | 核对结果 | 实现与证据 |
| --- | --- | --- |
| 从零实现，不依赖 Agent 框架 | 已实现 | 自研 `runtime.py`；依赖仅 HTTP、Schema、文件锁、环境文件等通用库 |
| 输入 → 模型决策 → 工具 → 继续或回答 | 已实现并测试 | `test_runtime.py`；真实 DeepSeek 工具链通过 |
| 至少三个工具且必须包含 todo | 已实现并测试 | calculator、search（题目允许 mock）、todo；`test_tools.py` 与 live 用例 |
| 名称、描述、参数 Schema 注册与自主调用 | 已实现并测试 | `ToolRegistry`、`tools`、`tool_choice=auto`；无关键词路由 |
| 解析思考、调用或最终答案 | 已实现并测试 | `parse_response` 解析可选 reasoning_content、tool_calls、content；默认非思考模式 |
| 同用户两个窗口 session 独立且可续聊 | 已实现并测试 | `(user_id,session_id)`、SQLite、文件锁；真实第二会话 list 为空且请求不含第一会话内容 |
| 最大轮次 | 已实现并测试 | 每个用户请求默认 8 次主循环决策；耗尽明确结束，不报告成功 |
| 持续对话记住状态、纯对话追问 | 已实现并测试 | 恢复历史和摘要；真实重启后代号/预算追问通过 |
| 带工具的追问 | 已实现并测试 | 真实新增 todo 后用“刚才那条”完成，数据库状态确认 |
| Context 放置策略与基础压缩 | 已实现并测试 | 摘要 + 最近完整交互 + 工具结果；200 轮离线用例与合成历史触发的真实摘要用例 |
| 基本异常处理 | 已实现并测试 | 超时/有限重试、参数错误回填、模型格式错误、取消与恢复 |
| 工具 trace 或执行日志 | 已实现并测试 | SQLite trace，`--trace` 或 `/trace` 查看调用与结果状态 |
| 构建测试用例 | 已完成 | 七个测试文件，包含单元、集成、真实 API 用例；数据均用临时目录 |
| 使用真实 LLM API | 已接入并实际验收 | 两个 DeepSeek live 用例通过，非 mock 模型 |
| GitHub 代码链接 | 已提供公开仓库 | [zhanpeng911-droid/Loomora](https://github.com/zhanpeng911-droid/Loomora) |
| README 的运行、系统设计、memory 召回说明 | 已完成 | README 对应章节包含召回时机、放置顺序和有损压缩边界 |
| AI Prompt 与问题解决记录 | 已完成 | `docs/ai-development-log.md`，标明 Prompt 摘要和真实验证边界 |
| 架构题每模块选一道，放在 docs | 已完成 | `docs/architecture-answers.md`，分别选择第 2、1、1、2、1 题 |
| 最小实现、适当注释 | 已完成 | Python CLI，无 Web 服务或现成 Agent 编排层；协议、压缩、锁及恢复处有注释 |

## 离线可重复检查

```powershell
uv sync --frozen
uv run pytest -q
uv run ruff check .
uv run ruff format --check .
```

测试数据写入 pytest 临时目录，不使用用户 `.loomora`。HTTP 模拟是协议测试，脚本化模型是 runtime 测试；它们不能证明真实模型的工具选择质量、摘要质量或服务可用性。

初版开发验证记录（2026-11-04）：`uv run pytest -q` 为 **171 passed, 1 skipped**，当时尚未配置密钥。`uv run ruff check .`、`uv run ruff format --check .` 和 `uv lock --check` 均通过；另检查了 CLI 帮助和缺少密钥时的可读错误提示。

同日补充真实验收：配置密钥后执行 `RUN_LIVE_TESTS=1` 的 `uv run pytest -q -m live`，结果 **2 passed, 171 deselected**（14.04 秒）。检查三个工具的实际成功结果、待办追问、第二会话的工具返回值与发送上下文，以及纯对话重启续聊和真实摘要。压缩测试的长历史是合成夹具，仅摘要和追问请求使用真实模型；不将它表述为数百轮真实对话。这是发布前的小型 API 验收记录。

逐项复核发现并修复“压缩后工具回填使请求再次超预算，循环直接提前结束”的问题，新增 6 项回归。修复后执行 `uv run pytest -q -m 'not live'`，结果 **177 passed, 2 deselected**；Ruff 检查、格式检查、锁文件检查均通过。这次回归明确排除了全部 live 用例，没有追加 DeepSeek 请求。单次用户请求最多一次模型摘要，后续预算收紧用本地摘录处理；当前交互自身放不下时仍明确停止。

## 真实 API 验收

1. 在项目根目录 `.env` 配置自己的 DeepSeek key、base URL 和模型。
2. 设置 `RUN_LIVE_TESTS=1`，执行 `uv run pytest -q -m live`。
3. 第一个用例要求模型调用三个工具并得到成功结果，通过追问完成待办。第二会话必须实际调用 todo.list 得到空列表，其模型请求不应包含第一会话内容。
4. 第二个用例检查纯对话重启续聊；写入合成历史触发真实摘要后，项目代号和预算应仍可正确回答。
5. 按 README 打开两个终端验证：同用户不同 session 的待办与历史不同；重启窗口一，原对话和状态仍可继续。
6. 输入纯对话追问和工具追问，查看 `/trace` 是否包含对应调用。用 `--max-steps 1` 发起需要工具的任务，确认明确显示轮次耗尽且工具结果保留。
7. 使用 `--context-bytes 8000` 连续对话以更早触发压缩，检查 `/trace` 中的压缩事件、近期指代和已知约束。语义质量需人工检查，不能仅凭请求长度判断。

若真实 API 返回 401/402/429 等错误，先核查密钥、余额或额度；不要把 API 故障改成假模型成功。真实模型可能不遵循工具指令，live 测试失败时应检查存储的工具结果与安全 trace，而不是放宽断言掩盖问题。

## 已知边界

- search 是静态 mock，天气工具不在本次范围；不会伪造实时天气。
- 会话 ID 是本地隔离键，不是身份认证；不支持分布式部署。
- 基础摘要有损，旧原文不会自动语义检索；原文可用 `/history` 查看。
- 工具全部同步；busy 请求明确拒绝，不自动排队。架构题中的事件 inbox/outbox 是扩展设计。
- 不自动撤销已完成的副作用。模型或进程中断时，可能存在需要核对的未知结果。
- 真实 API 通过的范围是上述两个小型用例，不代表所有输入均可可靠完成。
