# Base-model configuration

AI Historian compares models from multiple providers through the OpenAI Chat Completions protocol. The repository has one root `.env.example`: `AIH_CHAT_PROVIDER` selects a credential group, `AIH_CHAT_MODEL` records the exact model ID for the run, and credentials and endpoints use provider-native variables.

## Quick start

```bash
cp .env.example .env
# Edit .env: select a provider and model, then set that provider's API key.
uv run python scripts/check_model_config.py --env-file .env
uv run python scripts/smoke_test_model.py --env-file .env
```

The offline checker validates the provider, model ID, URL, selected credential, and retrieval mode without displaying secrets. The smoke test sends one minimal request to verify endpoint access, the model ID, system messages, and JSON-object output.

Load the configuration before using any formal entry point:

```bash
set -a
source .env
set +a

uv run aih examples/input --output runs/quickstart
uv run aih-fulltext examples/input --output runs/quickstart-fulltext
```

Experiment 1, Experiment 2, and the full paper-reproduction script use the same contract.

## Chat configuration contract

| `AIH_CHAT_PROVIDER` | API key | Base URL | Default endpoint |
| --- | --- | --- | --- |
| `openai` | `OPENAI_API_KEY` | `OPENAI_BASE_URL` | `https://api.openai.com/v1` |
| `anthropic` | `ANTHROPIC_API_KEY` | `ANTHROPIC_BASE_URL` | `https://api.anthropic.com/v1/` |
| `deepseek` | `DEEPSEEK_API_KEY` | `DEEPSEEK_BASE_URL` | `https://api.deepseek.com` |
| `gemini` | `GEMINI_API_KEY` | `GEMINI_BASE_URL` | `https://generativelanguage.googleapis.com/v1beta/openai/` |
| `dashscope` | `DASHSCOPE_API_KEY` | `DASHSCOPE_BASE_URL` | `https://dashscope.aliyuncs.com/compatible-mode/v1` |
| `compatible` | `AIH_COMPATIBLE_API_KEY` | `AIH_COMPATIBLE_BASE_URL` | Must be set explicitly |

Resolution is strict:

1. `AIH_CHAT_PROVIDER` and `AIH_CHAT_MODEL` must both be explicit.
2. A provider reads only the API key and base URL from its own table row.
3. A missing selected-provider credential is an error; AIH never falls back to another provider's key.
4. AIH never infers or replaces `AIH_CHAT_MODEL` from the provider, endpoint, or `/v1/models` response.
5. Provider defaults may be overridden through the corresponding `*_BASE_URL`. `compatible` requires its own key and URL; a local server may use `AIH_COMPATIBLE_API_KEY=EMPTY`.

Provider-native variables interoperate directly with official SDKs, CLIs, and deployment platforms. See the official [OpenAI](https://platform.openai.com/docs/quickstart/make-your-first-api-request), [Anthropic](https://platform.claude.com/docs/en/manage-claude/authentication), [Gemini](https://ai.google.dev/gemini-api/docs/api-key), and [DeepSeek](https://api-docs.deepseek.com/) documentation.

## Model qualification checklist

An effective AIH model route provides:

1. OpenAI-compatible `chat.completions` requests.
2. `system` and `user` message roles.
3. Reliable JSON-object generation for multi-stage structured extraction.
4. Sufficient context for the evidence packet and sufficient output tokens for detailed evidence objects.
5. Strong Chinese reading ability; Classical Chinese capability is valuable for the supplied historical corpus.
6. Stable model IDs and reproducible sampling settings for comparative experiments.

AIH requests JSON-object output by default. When Anthropic's OpenAI compatibility endpoint does not accept that request parameter, the runtime removes it and continues with an explicit JSON prompt. Other compatible endpoints that reject the parameter receive one prompt-only retry. Run the smoke test before using any endpoint for a formal run.

## Embeddings for the full-text Agent

Chat and embeddings are independent endpoints. The full-text Agent can start with lexical retrieval, which needs no embedding service:

```bash
AIH_CROSSDOC_SCOPE_SELECTOR=lexical
```

Embedding or hybrid retrieval requires a complete, separate configuration:

```bash
AIH_CROSSDOC_SCOPE_SELECTOR=hybrid
AIH_EMBED_API_KEY=...
AIH_EMBED_BASE_URL=https://your-embedding-endpoint.example/v1
AIH_EMBED_MODEL=your-embedding-model-id
```

`AIH_EMBED_*` never falls back to a chat credential, preventing requests from reaching the wrong service or account.

## Reproducible run records

For every new result, record the provider, full model ID or checkpoint, resolved endpoint, date, temperature, token limits, concurrency, source commit, and configuration source. `run_manifest.json` records non-secret configuration; API keys remain only in the local `.env` and never enter Git or result artifacts.
