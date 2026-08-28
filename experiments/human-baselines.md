# Public human baselines

AI Historian releases the pseudonymized human-condition responses, timing records, scoring references, and derived metrics used in both paper experiments. Participant identifiers use `P1`–`P6`; the public records focus on task responses and evaluation provenance.

## Comparison at a glance

| Experiment | Human condition | Human result | AI/model comparison | Public artifacts |
| --- | --- | ---: | --- | --- |
| Experiment 1 | Six participants, one Human-only case each | 81.3% MicroIoU; 5,520 s | AIH Agent: 86.2%; Direct LLM: 17.1% | [`experiment-1/results/human/`](experiment-1/results/human/) |
| Experiment 2 | Six participants, two completions per form | 64.6% strict accuracy (62/96); 8,796 s | Structured LLM: 75.0%; Direct LLM: 68.8% | [`experiment-2/results/human/`](experiment-2/results/human/) |

## Experiment 1

- Sentence-level Human-only responses: [`human-only-ranges.csv`](experiment-1/results/human/human-only-ranges.csv)
- Sentence-level Human+AI responses: [`human-ai-ranges.csv`](experiment-1/results/human/human-ai-ranges.csv)
- Case-level scores and timing: [`human-baseline-summary.json`](experiment-1/results/human/human-baseline-summary.json)
- Gold ranges: [`gold-iso-ranges.csv`](experiment-1/inputs/annotations/gold-iso-ranges.csv)
- Scoring implementation: [`score_ai_prefill_variant.js`](experiment-1/evaluation/score_ai_prefill_variant.js)

## Experiment 2

- Pseudonymized participant responses: [`experiment-2-participant-responses.json`](experiment-2/inputs/responses/experiment-2-participant-responses.json)
- Field-level scored results: [`experiment-2-human-accuracy-detail.csv`](experiment-2/results/human/experiment-2-human-accuracy-detail.csv)
- Human metric summary: [`experiment-2-human-accuracy-metrics.json`](experiment-2/results/human/experiment-2-human-accuracy-metrics.json)
- Standard answers: [`experiment-2-standard-answers.json`](experiment-2/inputs/scoring/experiment-2-standard-answers.json)
- Scoring implementation: [`build_human_accuracy.js`](experiment-2/code/build_human_accuracy.js)

The original combined study export used to derive the experiment-specific files is preserved under [`shared/human-study/`](shared/human-study/). This separation keeps the source record auditable while giving each experiment a direct, self-contained comparison package.
