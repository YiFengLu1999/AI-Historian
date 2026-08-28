# Experiment 2 human baseline

This directory contains the public, pseudonymized human-condition results for the 48-question diagnostic evaluation.

| Artifact | Contents |
| --- | --- |
| `experiment-2-human-accuracy-metrics.json` | Overall, participant-, form-, block-, field-, and timing-level metrics |
| `experiment-2-human-accuracy-summary.csv` | Compact comparison table for all reported human metrics |
| `experiment-2-human-accuracy-detail.csv` | Field-level answer, reference value, and correctness for every scored response |

Six participants completed one of T1–T3, with two participants per form. This produced 96 scored human rows and 192 scored components.

| Metric | Human result |
| --- | ---: |
| Overall strict accuracy | 64.6% (62/96) |
| Component accuracy | 77.6% (149/192) |
| Block A strict accuracy | 41.7% (20/48) |
| Block B strict accuracy | 79.2% (19/24) |
| Block C strict accuracy | 95.8% (23/24) |
| Cumulative elapsed time | 8,796 s |

Pseudonymized raw responses are available at `../../inputs/responses/experiment-2-participant-responses.json`; the reference answers are stored separately under `../../inputs/scoring/`.
