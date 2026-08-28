# AI Historian

<p align="center">
  <img src="assets/branding/aih-logo.png" alt="AI Historian" width="720">
</p>

<p align="center">
  <strong>English</strong> · <a href="README.zh-CN.md">中文</a>
</p>

<p align="center">
  <a href="https://westlakehistorian.com">Platform</a> ·
  <a href="results.md">Results</a> ·
  <a href="experiments/human-baselines.md">Human baselines</a> ·
  <a href="docs/model-configuration.md">Model setup</a> ·
  <a href="data-licenses.md">Data terms</a> ·
  <a href="docs/fulltext/README.md">Full-text Agent</a> ·
  <a href="CITATION.cff">Citation</a>
</p>

<p align="center"><strong>Version 1.0.0</strong></p>

Official implementation and reproducibility materials for **“AI Historian: Helping historians organize and verify person-centred temporal clues from dispersed historical narratives.”**

AI Historian (AIH) converts dispersed biographical narratives into traceable person–time evidence. It extracts people and temporal expressions, constructs TimeBlocks, restores chronological order, verifies cross-document relations, and normalizes supported evidence into comparable temporal ranges.

![AIH organizes dispersed source sentences into person-centred timelines](assets/figures/aih-overview.png)

## Quick start

### 1. Install

Requirements: Python 3.10 or newer. Recomputing the frozen experiment metrics also uses Node.js 18 or newer.

```bash
uv sync --locked
```

`pyproject.toml` is the dependency and packaging source of truth; `uv.lock`
pins the complete environment. Install [uv](https://docs.astral.sh/uv/) first
if it is not already available. All commands below run inside that locked
environment through `uv run`.

### 2. Configure a base model

AIH uses one configuration interface across the evaluation Agent, both experiments, and the scalable full-text Agent.

```bash
cp .env.example .env
# Select a provider and model, then set that provider's API key.
uv run python scripts/check_model_config.py --env-file .env
```

Load the configuration:

```bash
set -a
source .env
set +a
```

The single template contains provider-native credential groups for OpenAI, Anthropic, DeepSeek, Gemini, DashScope, and generic OpenAI-compatible endpoints. See [Base-model configuration](docs/model-configuration.md) for strict selection rules, embedding settings, model qualification, and the optional endpoint smoke test.

### 3. Run a minimal example

The repository includes a small synthetic Chinese biography at
`examples/input/liu-bang-synthetic-biography.txt`. Its person and title are
declared separately in `examples/input/manifest.json`; filenames remain plain
lowercase kebab-case instead of acting as metadata. Blank lines delimit paragraphs.

```bash
sed -n '1,20p' examples/input/liu-bang-synthetic-biography.txt
sed -n '1,40p' examples/input/manifest.json
```

You can first run the deterministic local preprocessing stage to check file
discovery, paragraph splitting, sentence segmentation, and stable sentence
identifiers:

```bash
uv run aih examples/input \
  --output runs/quickstart-preprocess \
  --through-step 1

find runs/quickstart-preprocess -maxdepth 3 -type f | sort
```

After loading the model configuration from Step 2, run the complete
evaluation-aligned Agent:

```bash
uv run aih examples/input --output runs/quickstart
```

The run directory preserves the intermediate representation produced by every
stage. Sentence-level JSON is stored under `sentence/`, constructed TimeBlocks
under `timeblock/`, and chronological ordering artifacts under `sequence/`.
List the generated files with:

```bash
find runs/quickstart -maxdepth 3 -type f | sort
find runs/quickstart/timeblock/step11output \
  -maxdepth 1 -type f -name '*.json' | sort
```

To exercise evidence-verified alignment across several source files, place the
files in the same input directory and add `--cross-document`. To also generate
readable summaries after normalization, add `--summaries`:

```bash
uv run aih examples/input \
  --output runs/quickstart-crossdoc \
  --cross-document \
  --summaries
```

Run the scalable full-text Agent on the same input when working with complete
texts or larger multi-document collections:

```bash
uv run aih-fulltext examples/input --output runs/quickstart-fulltext
```

For a new collection, place lowercase kebab-case UTF-8 `.txt` files in one
directory and describe every file in `manifest.json`. Store source texts in a
dedicated corpus directory and reserve `runs/` for outputs. Choose a new output
directory for each configuration so that model and Agent variants remain
directly comparable. See [Data format](docs/data-format.md) for the input
manifest and the sentence and TimeBlock fields written by the pipeline.

## Choose an implementation

AIH provides two execution profiles built around the same A1–A9 semantic method:

| Profile | Entry point | Intended use | Cross-document scope |
| --- | --- | --- | --- |
| **Evaluation profile** | `uv run aih` | Paper-aligned experiments and compact research collections | Bounded case packets |
| **Scalable full-text profile** | `uv run aih-fulltext` | Complete texts, multi-document collections, and platform-oriented processing | Runtime retrieval creates compact, anchor-aware `episode_packet` scopes |

Both profiles live in the `ai_historian` package and preserve TimeBlocks, source provenance, cross-text evidence verification, and normalized temporal ranges. The reported paper metrics correspond to the frozen evaluation profile. See the [profile comparison](docs/profiles.md) and [full-text Agent guide](docs/fulltext/README.md) for stage-level details.

## Reproduce the paper artifacts

The repository includes frozen inputs, intermediate stages, model outputs, timing records, consensus predictions, and scoring programs. [results.md](results.md) is the central index for every reported result.
Paper reproduction is a repository workflow, so it is invoked through `scripts/reproduce_paper.py` and is not installed as a wheel console command.

Run every deterministic frozen-artifact check with one command:

```bash
uv run python scripts/reproduce_paper.py frozen
```

After configuring a model endpoint, regenerate Experiment 1 from its case
inputs through three independent AIH runs, row-level consensus, MicroIoU
scoring, both Experiment 2 model conditions, their consensus outputs, human
metrics, and the final comparison with:

```bash
uv run python scripts/reproduce_paper.py full --env-file .env
```

Recompute Experiment 1 MicroIoU from the frozen consensus predictions:

```bash
node experiments/experiment-1/evaluation/score_ai_prefill_variant.js
```

Rebuild the Experiment 2 human metrics and final comparison:

```bash
node experiments/experiment-2/code/build_human_accuracy.js
node experiments/experiment-2/code/build_strict_total_html.js
```

Run the deterministic preprocessing, TimeBlock, MicroIoU, temporal
normalization, cross-text constraint, and configuration-loading tests:

```bash
uv run python -m unittest discover -s tests -v
```

Exact input/output paths, metric definitions, and full-regeneration guidance are documented in [Reproducibility](docs/reproducibility-guide.md), [Experiment 1](experiments/experiment-1/README.md), and [Experiment 2](experiments/experiment-2/README.md).

## Key results

### Experiment 1: end-to-end temporal reconstruction

| Condition | MicroIoU | Cumulative elapsed time |
| --- | ---: | ---: |
| **AIH Agent** | **86.2%** | **13 min 56 s** |
| Human-only | 81.3% | 1 h 32 min |
| Direct LLM prompting | 17.1% | 2 min 28 s |

The comparison covers 244 eligible sentence-level temporal ranges across six *Shiji* cases.

The public [Experiment 1 human baseline package](experiments/experiment-1/results/human/) includes sentence-level Human-only and Human+AI responses, case scores, and timing records.

### Multi-model replication

| Underlying model | AIH Agent | Direct prompting |
| --- | ---: | ---: |
| DeepSeek-V4-Flash, non-thinking | 90.2% | 15.4% |
| GPT-5.6 SOL | 86.9% | 17.9% |
| Gemini 3.1 Pro | 88.6% | 14.6% |
| Claude Opus 5 | **90.7%** | 14.3% |
| Qwen 3.6 | 78.6% | 15.0% |

Values are descriptive macro-averages of the six case-level MicroIoU scores. The full source tables and model-specific artifacts are indexed in [results.md](results.md).

### Experiment 2: diagnostic reasoning

| Condition | Overall strict accuracy | Cross-text verification and alignment |
| --- | ---: | ---: |
| Human | 64.6% | 95.8% |
| Direct LLM prompting | 68.8% | 75.0% |
| **Structured LLM prompting** | **75.0%** | **91.7%** |

The public [Experiment 2 human baseline package](experiments/experiment-2/results/human/) includes pseudonymized responses, field-level correctness, participant/form/block summaries, and timing. A unified comparison entry point is available in [Public human baselines](experiments/human-baselines.md).

## Method

Sentence-level agents A1–A4 identify people, original temporal expressions, narrative functions, and interludes. TimeBlock-level agents A5–A9 complete temporal markers, restore chronological order, assess temporal granularity and anchors, verify cross-text evidence, and convert accepted evidence into normalized temporal constraints.

![AIH Agent architecture and TimeBlock-level reasoning](assets/figures/aih-agent-architecture.png)

See [Architecture](docs/system-architecture.md) for the full stage graph and [Data format](docs/data-format.md) for input, intermediate, and output schemas.

## Westlake Historian platform

**[Westlake Historian](https://westlakehistorian.com)** is the public research platform developed and maintained by our team and powered by AIH. It supports cross-text person–time evidence retrieval, source-level inspection, temporal-basis verification, revision records, and human-in-the-loop final decisions.

At the paper's analysis cut-off, the platform contained 25 collections, 75 processed texts, and 75 person-centred chronology tracks spanning Chinese, Japanese, Korean, and modern historical materials.

![Westlake Historian supports source-grounded evidence inspection and human revision](assets/figures/westlake-historian-platform.png)

## Repository layout

```text
AI-Historian/
├── src/ai_historian/          # Installable package and both execution profiles
│   ├── profiles/evaluation/   # Paper-evaluation A1–A9 profile
│   ├── profiles/scalable_fulltext/ # Runtime-scoped full-text profile
│   ├── pipeline/              # Shared runners, paths, logging, normalization
│   └── resources/             # Canonical bundled catalogs and time mappings
├── experiments/
│   ├── experiment-1/          # End-to-end reconstruction and MicroIoU
│   ├── experiment-2/          # Diagnostic reasoning evaluation
│   └── shared/human-study/    # Combined source export shared by both experiments
├── .env.example               # Shared provider-native model configuration template
├── scripts/                   # Configuration checks and reproduction entry point
├── pyproject.toml             # Package metadata and dependency source of truth
├── uv.lock                    # Reproducible universal dependency lock
├── examples/input/            # Minimal UTF-8 input collection
├── docs/                      # Architecture, formats, setup, reproducibility
├── assets/                    # Logos, architecture figures, result figures
├── data-licenses.md           # Data provenance, licensing, and participant safeguards
├── results.md                 # Frozen-result index
└── CITATION.cff               # Machine-readable citation metadata
```

Chinese source passages and prompt templates remain in their experimental form. Public documentation and repository navigation use English filenames, with Chinese guides provided alongside the main documents.

## Data and provenance

The release preserves original source sentences, stable identifiers, sentence-level intermediate annotations, TimeBlocks, ordering sequences, normalized ranges, and final predictions. Gold annotations and participant results are stored separately from generation inputs and enter the workflow during scoring. The source and reuse terms for translations, project annotations, pseudonymized participant responses, model outputs, figures, and software are defined in [Data sources and licenses](data-licenses.md).

The evaluation-aligned pipeline incorporates the frozen Agent stages used for the reported experiments. The scalable implementation is the portable release of the platform deployment implementation and packages the runtime-scope strategy used for full-text processing.

## Citation

Citation metadata is provided in [CITATION.cff](CITATION.cff). GitHub surfaces this file through its **Cite this repository** interface.

## License

Software is released under [Apache License 2.0](LICENSE.md). Project-authored
research data and documentation use CC BY 4.0, while modern translation
excerpts and participant-derived records follow the scoped terms in
[data-licenses.md](data-licenses.md).

## Community

Contribution guidance is available in [CONTRIBUTING.md](CONTRIBUTING.md).

Questions, reproducibility feedback, new historical-corpus cases, model-provider profiles, and implementation proposals are warmly welcome. Please open a GitHub issue with a concise description, relevant input or output paths, and the execution profile you used. We look forward to learning from and building with the digital-history, historical-research, and AI-agent communities.
