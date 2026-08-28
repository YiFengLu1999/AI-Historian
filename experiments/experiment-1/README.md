# Experiment 1: end-to-end temporal reconstruction

Experiment 1 compares human-only annotation, direct LLM prompting, and AIH Agent on six vernacular-Chinese *Shiji* case packets concerning Liu Bang, Xiang Yu, and Xiao He.

The primary metric is month-level MicroIoU over 244 eligible sentence-level ranges. Open prediction boundaries are clipped to the gold evaluation window for each case. The reported AIH prediction table is a row-level consensus across three independent model runs.

## Layout

- `inputs/cases/`: six canonical case packets (`H-C1`–`H-C6`) and their combined machine-readable collection.
- `inputs/latex-cases/`: canonical case forms consumed by the complete Agent regeneration entry point.
- `inputs/annotations/`: adjudicated gold ranges and human-only annotations.
- `inputs/config/`: cross-document scope, frozen time-string map, and Direct LLM rules.
- `inputs/manifest.json`: case counts, source texts, and canonical input paths.
- `results/`: frozen AIH intermediate/final outputs, consensus predictions, score summary, and timing summary.
- `results/multimodel/`: final five-model Agent/direct score summaries and the consolidated analysis table used by the paper.
- `direct-llm/`: direct-prompt baseline generation and postprocessing code.
- `direct-llm-results/`: frozen raw Direct LLM responses, canonicalized final rows, run summaries, and scores. Reproducible `agent_postprocess_workspace/` scratch files are excluded.
- `evaluation/`: deterministic scoring and diagnostic scripts.
- `code/`: case preparation and three-run AIH consensus orchestration.

## Existing final outputs

- Main AIH consensus score: `results/ai-variant-score-experiment-1-ai-only-final-api-consensus.json`
- Main AIH consensus rows: `results/generated_results_api_consensus_20260614_195059/`
- Direct-prompt baseline: `direct-llm-results/generated_results_direct_llm_20260615_190423/`
- Multi-model summary: `results/multimodel/analysis/experiment-1-multimodel-metrics.json`
- Public human baseline: [`results/human/`](results/human/), with sentence-level responses, case metrics, and timing

## Canonical inputs

The complete six-case input collection is `inputs/cases/experiment-1-cases.json`. Individual Agent-ready packets are stored as `inputs/cases/H-C1-sentence.json` through `H-C6-sentence.json`. See [`inputs/README.md`](inputs/README.md) for the data boundary and file roles.

## Recompute AIH metrics

From the repository root:

```bash
node experiments/experiment-1/evaluation/score_ai_prefill_variant.js
```

The scorer reads the frozen predictions and writes a new JSON summary under ignored `recomputed/`.

## Regenerate three AIH runs, consensus, and score

From the repository root, after loading a validated model configuration:

```bash
uv run python experiments/experiment-1/code/run_experiment1_ai_only_consensus.py \
  --repeats 3 \
  --env-file .env \
  --output-dir runs/experiment-1-full
```

This entry point converts the canonical case inputs into isolated Agent run
roots, executes all six cases three times, groups matching sentence rows,
selects exact majorities or an interval medoid for ties, writes the consensus
table, and invokes the MicroIoU scorer. Gold annotations enter only the final
scoring stage.

The repository-level command runs this workflow together with Experiment 2:

```bash
uv run python scripts/reproduce_paper.py full --env-file .env
```
