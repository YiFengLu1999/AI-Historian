# AIH scalable full-text Agent

<p align="center">
  <img src="../../assets/branding/aih-mark.png" alt="AI Historian mark" width="118">
</p>

[中文说明](README.zh-CN.md) | **English**

OpenAI, DeepSeek, Qwen, Gemini, Claude, local inference servers, and compatible gateways share the [base-model configuration guide](../model-configuration.md).

This guide documents the platform-oriented AI Historian execution profile derived from the Agent system used with [Westlake Historian](https://westlakehistorian.com). It extends the evaluation pipeline to complete texts and multi-document collections while retaining the same A1–A9 semantic method, TimeBlock representation, source traceability, and evidence-verification rules.

![Scalable full-text AIH pipeline](../../assets/figures/fulltext-pipeline-overview.png)

## Runtime-scope design

The controlled experiments operate on bounded case packets, while a platform processes much longer documents. This implementation provides reliable and scalable context management through a runtime-scope layer that retrieves compact, anchor-aware `episode_packet` candidates, runs the core A8/10B verifier on each mini-scope, and merges accepted evidence back into the complete run.

![A8 runtime scope and evidence propagation](../../assets/figures/a8-runtime-scope-flow.png)

The adaptation manages context size while preserving the AIH method. See [the profile comparison](../profiles.md) for the exact boundary between this profile and the frozen experimental implementation.

## Pipeline

1. Steps 1–5 preprocess texts and run sentence-level agents A1–A4.
2. Steps 6–9 construct TimeBlocks, complete temporal markers, restore order, and classify temporal granularity.
3. A8/10A generates temporal markers for full-text TimeBlocks.
4. The runtime-scope runner retrieves cross-document candidates and builds `episode_packet` mini-scopes.
5. A8/10B verifies cross-text evidence inside each mini-scope; accepted evidence is deduplicated and merged into the full run.
6. A8/10C–10D stabilize anchors and propagate verified temporal constraints.
7. A9/Step 11 normalizes temporal ranges.
8. Steps 12–14 prepare source-linked application outputs and readable summaries.

### Internal temporal stages

The temporal core is split into explicit, inspectable stages. Step 10A generates temporal markers; 10B verifies quoted cross-document evidence inside each retrieved mini-scope; 10C stabilizes within-document anchors; and 10D propagates accepted boundary constraints through the temporal graph. Step 11 then normalizes ranges using the frozen lookup table first, deterministic regnal-year conversion second, and an LLM fallback only when necessary.

## Setup

From the repository root:

```bash
uv sync --locked
cp .env.example .env
```

Set `AIH_CHAT_PROVIDER`, `AIH_CHAT_MODEL`, and the selected provider's native API-key variable, then load the configuration:

```bash
set -a
source .env
set +a
```

## Input

Place related UTF-8 text files in one directory. Use lowercase kebab-case names:

```text
liu-bang-synthetic-biography.txt
```

Declare each file's `person` and `title` in the sibling `manifest.json` described in [Data format](../data-format.md). Blank lines delimit paragraphs. A built-in book catalog supplies stable metadata for supported historical collections; set `AIH_BOOK_CATALOG` to use a custom catalog. Unmatched files receive deterministic collection identifiers.

## Run

```bash
uv run aih-fulltext path/to/input_texts --output runs/my_collection
```

Cross-document runtime scoping is enabled by default. For within-document processing:

```bash
uv run aih-fulltext path/to/input_texts \
  --output runs/my_collection \
  --skip-crossdoc
```

Resume an interrupted run at a numbered stage:

```bash
uv run aih-fulltext path/to/input_texts \
  --output runs/my_collection \
  --resume-from 10
```

Useful runtime-scope controls include `AIH_CROSSDOC_SCOPE_SELECTOR`, `AIH_CROSSDOC_SCOPE_MAX_CASES`, `AIH_CROSSDOC_SCOPE_TOP_K_PER_PAIR`, and `AIH_CROSSDOC_SCOPE_CONTEXT_PAD`. Defaults reproduce the deployment-oriented `episode_packet` profile.

## Outputs

Every stage writes a new `sentence/`, `timeblock/`, or `sequence/` subdirectory under the selected run root. Runtime-scope reports and mini-scopes are retained for inspection. Step 14 creates source-linked integration data under `<run-root>/export/`. Every CLI run also writes `run_manifest.json` with input hashes, source commit, non-secret configuration, and completed stages.

## Package contents

This package contains the portable Agent source, reference time mappings, structure figures, runtime-scope implementation, and application-export interfaces.

The reported experiment scores are linked to the frozen evaluation profile described in [`../../experiments/`](../../experiments/).

## Feedback and contributions

See [`../../CONTRIBUTING.md`](../../CONTRIBUTING.md) for contribution guidance.

Reports on runtime-scope behavior, cross-document retrieval, temporal normalization, corpus integration, and reproducibility are especially welcome. Please open a GitHub issue with the command, execution profile, relevant paths, and a compact example. We are happy to discuss new historical collections and collaborate on improvements.
