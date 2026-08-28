# Experiment 2: diagnostic temporal reasoning

This directory provides the frozen inputs, outputs, and scoring code for the diagnostic evaluation of intermediate temporal-reasoning judgements.

## Contents

- `inputs/forms/`: T1–T3 participant-facing questions and blank response sheets.
- `inputs/researcher/`: per-form answer keys, combined key/template, and the form manifest.
- `inputs/scoring/`: canonical answer JSON used for scoring.
- `inputs/responses/`: participant response JSON used to compute human performance.
- `code/`: reproducibility scripts.
- `results/human/`: public pseudonymized human baseline with raw-response link, field-level scoring, grouped metrics, and timing.
- `results/direct-llm/`: frozen final Direct LLM run.
- `results/structured-llm/`: frozen final Structured LLM run.
- `figure/`: final comparison HTML and its source metrics JSON.
- `outputs/`: new rerun outputs. This folder is created/updated by the scripts.

The complete input schema and the 48-question inventory are documented in [`inputs/README.md`](inputs/README.md).

## Conditions

- `Human`: participant results scored against the standard answers.
- `Direct LLM`: one direct prompt per Experiment 2 question, three runs, per-field majority vote.
- `Structured LLM`: human-visible question text, block-specific structured prompts, Block A second-pass review, three runs, per-field majority vote.

The final Experiment 2 comparison uses its self-contained question text, block-specific prompts, answer keys, and condition outputs.

## Reproduce

For new model runs, copy the repository-root `.env.example` to `.env`, fill the selected provider's credentials, or pass `ENV_FILE=/path/to/.env`.

```zsh
cd experiments/experiment-2

./code/run_direct_llm.sh
./code/run_structured_llm.sh
./code/rebuild_figure.sh
```

To recompute human metrics from raw participant responses:

```zsh
node code/build_human_accuracy.js
```

The final figure is:

- `figure/experiment-2-strict-total-comparison.html`

The primary metric is row-strict accuracy: a Block A row counts as correct when time-span, sink, and interlude are all correct; a Block B/C row counts as correct when the selected choice is correct.
