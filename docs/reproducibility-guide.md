# Reproducibility guide

## Environment

Use Python 3.10 or newer, Node.js 18 or newer, and uv. The project metadata and universal lock are the auditable environment definition:

```bash
uv sync --locked
```

Edit dependency ranges only in `pyproject.toml`, regenerate `uv.lock` with `uv lock`, and commit both files. CI rejects a stale lock with `uv sync --locked`.

The reproduction driver is intentionally repository-only: use `scripts/reproduce_paper.py` from a checkout. It is not installed as a wheel console command because it depends on the repository's `experiments/` assets.

## One-command frozen verification

Start with the supplied predictions and participant results:

```bash
uv run python scripts/reproduce_paper.py frozen
```

This command performs four deterministic operations:

1. recomputes Experiment 1 MicroIoU from the frozen row-level AIH consensus;
2. rebuilds Experiment 2 human strict-accuracy metrics;
3. rebuilds the Experiment 2 JSON/HTML condition comparison;
4. runs the preprocessing, TimeBlock, MicroIoU, temporal-normalization, cross-text-constraint, and configuration tests.

Fresh metric artifacts are written under each experiment's ignored `recomputed/` or generated-output location. The frozen files under `results/` remain the archival reference.

## One-command full regeneration

Select a provider profile, validate it, and run a small endpoint qualification:

```bash
cp .env.example .env
# Select the provider and exact model ID, then set that provider's API key.
uv run python scripts/check_model_config.py --env-file .env
uv run python scripts/smoke_test_model.py --env-file .env
```

Then regenerate the paper workflow:

```bash
uv run python scripts/reproduce_paper.py full --env-file .env
```

The command expands to the following source-to-score graph:

```text
Experiment 1 LaTeX case inputs
  -> standard Agent sentence packets
  -> six isolated AIH case pipelines x three independent runs
  -> row-level majority/interval-medoid consensus
  -> month-level MicroIoU scoring

Experiment 1 Direct LLM inputs
  -> direct baseline generation and Agent Step 11 normalization
  -> MicroIoU scoring

Experiment 2 forms and scoring references
  -> Direct LLM x three runs -> per-field consensus -> strict scoring
  -> Structured LLM x three runs -> per-question consensus -> strict scoring
  -> participant scoring -> final JSON/HTML comparison
```

New outputs are grouped under `runs/paper-reproduction/<timestamp>/`. A `reproduction_manifest.json` records the source commit, provider, exact model ID, repetition count, and output roots.

Useful controls include:

```bash
# Inspect every command and output path before model calls.
uv run python scripts/reproduce_paper.py full --env-file .env --dry-run

# Run isolated cases concurrently while keeping repetitions sequential.
uv run python scripts/reproduce_paper.py full --env-file .env --parallel-cases 2

# Reuse the archived Experiment 1 Direct LLM baseline.
uv run python scripts/reproduce_paper.py full --env-file .env --skip-experiment1-direct
```

The complete Experiment 1 AIH entry point is `experiments/experiment-1/code/run_experiment1_ai_only_consensus.py`. It can resume an output root with `--skip-existing-runs` after a transient provider failure.

## Consensus rules

Experiment 1 groups predictions by case, participant sheet, row number, and sentence identifier. An exact prediction signature with more than half of the runs is selected. A tie selects the observed interval medoid that minimizes total start/end month distance to the other runs; state disagreements receive a fixed penalty. Structural completeness and the prediction signature provide deterministic tie-breaking. Gold ranges do not enter consensus selection.

Experiment 2 applies majority voting independently to the scored fields. Ties retain the earliest run value. The consensus rows enter strict scoring only after voting is complete.

## Expected frozen results

- Experiment 1 contains 249 source sentences; 244 eligible sentence ranges enter MicroIoU scoring.
- Frozen AIH case MicroIoU values are 1.000, 1.000, 1.000, 1.000, 0.596, and 0.819 for H-C1 through H-C6.
- Experiment 2 overall strict accuracy is 64.6% for humans, 68.8% for direct prompting, and 75.0% for structured prompting.

## Run records and interpretation

Model calls vary with provider model revisions, service load, sampling behavior, and concurrency. Record the provider, full model identifier or checkpoint, endpoint type, execution date, token settings, concurrency, source commit, and environment-template name for each new result.

The included time-expression map and era table define the submitted temporal-normalization configuration. Chinese prompts and case texts preserve the evaluated task. Source and redistribution terms for translations, annotations, model outputs, and participant-derived records are defined in [`data-licenses.md`](../data-licenses.md).
