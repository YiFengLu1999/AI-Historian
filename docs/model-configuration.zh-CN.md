# 底层模型配置指南

AI Historian 通过 OpenAI Chat Completions 协议比较多个模型厂商。仓库只提供根目录一个 `.env.example`：`AIH_CHAT_PROVIDER` 选择凭证组，`AIH_CHAT_MODEL` 明确记录本次运行的准确模型 ID，凭证和地址使用厂商原生变量。

## 快速开始

```bash
cp .env.example .env
# 编辑 .env：选择 provider 和模型，并填写对应厂商的 API key。
uv run python scripts/check_model_config.py --env-file .env
uv run python scripts/smoke_test_model.py --env-file .env
```

离线检查会验证 provider、模型 ID、URL、所选凭证和检索模式，并隐藏凭证内容。smoke test 会发送一次最小请求，验证 endpoint、模型 ID、system message 和 JSON object 输出。

加载配置后，所有正式入口读取同一套变量：

```bash
set -a
source .env
set +a

uv run aih examples/input --output runs/quickstart
uv run aih-fulltext examples/input --output runs/quickstart-fulltext
```

实验一、实验二和完整论文复现脚本也使用相同配置。

## Chat 配置契约

| `AIH_CHAT_PROVIDER` | API key | Base URL | 默认地址 |
| --- | --- | --- | --- |
| `openai` | `OPENAI_API_KEY` | `OPENAI_BASE_URL` | `https://api.openai.com/v1` |
| `anthropic` | `ANTHROPIC_API_KEY` | `ANTHROPIC_BASE_URL` | `https://api.anthropic.com/v1/` |
| `deepseek` | `DEEPSEEK_API_KEY` | `DEEPSEEK_BASE_URL` | `https://api.deepseek.com` |
| `gemini` | `GEMINI_API_KEY` | `GEMINI_BASE_URL` | `https://generativelanguage.googleapis.com/v1beta/openai/` |
| `dashscope` | `DASHSCOPE_API_KEY` | `DASHSCOPE_BASE_URL` | `https://dashscope.aliyuncs.com/compatible-mode/v1` |
| `compatible` | `AIH_COMPATIBLE_API_KEY` | `AIH_COMPATIBLE_BASE_URL` | 必须明确填写 |

解析规则是严格的：

1. `AIH_CHAT_PROVIDER` 和 `AIH_CHAT_MODEL` 都必须明确填写。
2. provider 只读取表格中同一行的 API key 和 base URL。
3. 缺少所选 provider 的凭证时立即报错；不会回退到其他厂商的 key。
4. `AIH_CHAT_MODEL` 不会根据 provider、endpoint 或 `/v1/models` 自动推断或替换。
5. 厂商默认地址可以通过对应的 `*_BASE_URL` 覆盖。`compatible` 必须同时填写自己的 key 和地址；本地服务可使用 `AIH_COMPATIBLE_API_KEY=EMPTY`。

厂商原生变量与官方 SDK、CLI 和部署平台可以直接互操作，参见 [OpenAI](https://platform.openai.com/docs/quickstart/make-your-first-api-request)、[Anthropic](https://platform.claude.com/docs/en/manage-claude/authentication)、[Gemini](https://ai.google.dev/gemini-api/docs/api-key) 和 [DeepSeek](https://api-docs.deepseek.com/) 文档。

## 模型资格清单

适合 AIH 的模型接入应具备：

1. OpenAI-compatible `chat.completions` 请求接口。
2. `system` 和 `user` 消息角色。
3. 稳定生成 JSON object，以支持多阶段结构化抽取。
4. 与证据包匹配的上下文容量，以及容纳详细证据对象的输出长度。
5. 良好的中文阅读能力；处理仓库所附历史语料时，文言文能力尤其重要。
6. 可记录的准确模型 ID 和采样参数，便于模型间公平比较。

AIH 默认请求 JSON object。Anthropic OpenAI compatibility 接口不支持该请求参数时，运行时会自动去掉参数并继续使用明确的 JSON 提示词；其他兼容 endpoint 若拒绝该参数，也会重试一次提示词模式。首次使用任何 endpoint 前仍应运行 smoke test。

## 全文 Agent 的 Embedding 配置

Chat 和 embedding 是两个独立 endpoint。全文 Agent 首次运行可以采用不需要 embedding 的 lexical retrieval：

```bash
AIH_CROSSDOC_SCOPE_SELECTOR=lexical
```

启用 embedding 或 hybrid retrieval 时，必须完整填写独立配置：

```bash
AIH_CROSSDOC_SCOPE_SELECTOR=hybrid
AIH_EMBED_API_KEY=...
AIH_EMBED_BASE_URL=https://your-embedding-endpoint.example/v1
AIH_EMBED_MODEL=your-embedding-model-id
```

`AIH_EMBED_*` 不会回退到聊天模型凭证，避免请求被发送到错误的服务或账户。

## 可复现运行记录

每次生成新结果时，应记录 provider、完整模型 ID 或 checkpoint、解析后的 endpoint、日期、temperature、token 上限、并发量、代码 commit 和配置来源。`run_manifest.json` 会记录非敏感配置；API key 始终保留在本地 `.env` 中，不进入 Git 或结果文件。
