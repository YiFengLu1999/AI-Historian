# Experiment 1 human baselines

This directory provides the public, pseudonymized human-condition result package used for comparison with AIH Agent and Direct LLM prompting.

| Artifact | Contents |
| --- | --- |
| `human-baseline-summary.json` | Paper-level Human-only result, case metrics, participant assignment, and cumulative time |
| `human-only-ranges.csv` | All 249 Human-only sentence-level responses |
| `human-ai-ranges.csv` | All 249 Human+AI sentence-level responses retained for secondary analysis |
| `human-case-metrics.csv` | Human-only and Human+AI MicroIoU by case |
| `human-timing-by-case.csv` | Pseudonymous participant, condition, case, and elapsed time |

The primary paper comparison uses the Human-only condition: six participants independently completed one case each, producing 244 eligible scored ranges, 81.3% case-size-weighted MicroIoU, and 5,520 seconds of cumulative task time.

Participant identifiers are limited to `P1`–`P6`. The public tables contain task responses, case assignments, and timing records for methodological comparison.

Gold ranges are stored under `inputs/annotations/`, and the deterministic scorer is stored under `evaluation/`.
