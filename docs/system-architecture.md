# AIH architecture

AIH separates temporal reconstruction into inspectable sentence-level and TimeBlock-level stages. The public stage numbers preserve the identifiers used by the research implementation, while A1–A9 denote the semantic-agent roles in the paper.

| Stage | Paper role | Main output |
| --- | --- | --- |
| 1 | Text preprocessing | Stable source, paragraph, sentence, chapter, and book identifiers |
| 2 | A1: person identification | Per-sentence person membership |
| 3 | A2: temporal-expression extraction | Original temporal information (OTI) |
| 4 | A3: narrative-function judgement | Event vs descriptive/background sentence |
| 5 | A4: temporal-shift detection | Retrospective/interlude labels |
| 6 | TimeBlock assembly | Adjacent sentences governed by a shared temporal context |
| 7 | A5: temporal-marker completion | Context-completed temporal information with a stated basis |
| 8 | A6: within-document ordering | Chronological TimeBlock sequence |
| 9–10c | A7: granularity, anchors, and within-document stabilization | Canonical temporal markers and anchor judgements |
| 10b | A8: cross-document evidence verification | Candidate relations, quotations, confidence, and transfer eligibility |
| 10d–11 | A9: boundary propagation and normalization | Evidence-constrained ISO ranges |
| 14 | Presentation layer | Readable summaries linked to TimeBlocks and source sentences |

Every stage writes a new directory instead of overwriting its input. A final range can therefore be traced backward through normalization, constraints, TimeBlocks, sentence annotations, and the original source text.

Cross-document transfer is opt-in in the evaluation-oriented `uv run aih` profile. In the scalable `uv run aih-fulltext` profile it is enabled by default and preceded by runtime `episode_packet` scoping. In both profiles, temporal transfer requires a verified relation type and supporting text. The temporal graph then converts accepted evidence into boundary constraints. See [profiles.md](profiles.md) for the execution-profile comparison.
