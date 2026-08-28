# Data sources, licenses, and responsible-use terms

This file defines the provenance and reuse terms for the research materials distributed with AI Historian. The repository-level [Apache License 2.0](LICENSE.md) applies to software; it does not automatically apply to every text or dataset stored beside the software.

## License summary

| Material class | Principal paths | Source and ownership | Release terms |
| --- | --- | --- | --- |
| Software, tests, and configuration templates | `src/ai_historian/`, `scripts/`, `tests/`, `configs/` | Original AI Historian software | Apache License 2.0 |
| Project-authored annotations and evaluation structures | `experiments/**/inputs/annotations/`, `experiments/**/inputs/config/`, answer keys, manifests, scoring references, and form structure | Created or adjudicated by the AI Historian research team | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |
| Project-authored documentation, synthetic examples, tables, and figures | `docs/`, `examples/`, repository-authored Markdown, `assets/`, and derived result summaries | Created by the AI Historian research team, except where a file credits another source | CC BY 4.0 |
| Model-generated outputs | `experiments/**/results/`, excluding human responses and third-party source text embedded in an output | Generated for this study with the model and configuration recorded by each run | CC BY 4.0 to the extent the research team can license the output; model-provider terms and rights in embedded source text continue to apply |
| Modern-Chinese *Shiji* translation excerpts | Experiment case packets, diagnostic forms, and response/scoring files where the source sentence or question reproduces an excerpt | Modern-Chinese translations of *Shiji*, chapters 7, 8, and 53; the underlying classical work is attributed to Sima Qian, while rights in a modern translation may belong to its translator or publisher | Research and reproducibility excerpts. These excerpts are excluded from the CC BY 4.0 grant. Reuse beyond quotation, verification, and replication requires permission from the relevant translation rights holder or an independently licensed source |
| Pseudonymized participant-derived records | `experiments/shared/human-study/`, `experiments/experiment-1/results/human/`, `experiments/experiment-2/inputs/responses/`, and `experiments/experiment-2/results/human/` | Task responses collected in the human evaluation and released under participant codes `P1`–`P6` | CC BY 4.0 for research verification and comparative analysis, subject to the participant-data safeguards below |

## Historical source material

The experiments use modern-Chinese vernacular excerpts corresponding to:

- *Shiji*, chapter 7, **Basic Annals of Xiang Yu** (`项羽本纪`);
- *Shiji*, chapter 8, **Basic Annals of Gaozu** (`高祖本纪`);
- *Shiji*, chapter 53, **Hereditary House of Chancellor Xiao** (`萧相国世家`).

Sentence identifiers preserve the source chapter and location used by the project. The repository distributes the excerpts required to inspect and reproduce the reported evaluation. The project does not represent that every modern translation is in the public domain, and the CC BY 4.0 grant for project-created data does not extend to translation wording supplied by a third party.

When redistributing an evaluation derivative, retain chapter attribution, sentence identifiers, this file, and any source notice attached to the relevant artifact. A user who substitutes another translation should record its edition, translator, publisher or host, access date, and reuse license.

## Project-created annotations and derived data

The CC BY 4.0 data grant covers the research team's original contribution to:

- adjudicated temporal ranges and human-condition range tables;
- time-expression maps, cross-document scopes, manifests, question structures, answer keys, and scoring references;
- pseudonymized and aggregated human-performance tables;
- model consensus tables, metric summaries, timing summaries, and analysis figures;
- synthetic demonstration text and repository-authored data documentation.

Attribution should name **AI Historian**, cite the associated paper through [`CITATION.cff`](CITATION.cff), link to this repository, identify the source commit, and state material modifications.

## Participant-derived data safeguards

The public participant package uses pseudonymous identifiers `P1`–`P6` and retains task responses, case or form assignment, condition labels, scoring fields, and elapsed time required for methodological comparison. The study team authorized release of this pseudonymized research package for reproducibility and comparative research.

Reuse of participant-derived records must preserve the following boundaries:

1. Use the records for research verification, methods development, education, or aggregate comparison.
2. Preserve pseudonymous identifiers and report results at the least identifying level compatible with the analysis.
3. Do not attempt re-identification, identity linkage, participant contact, or inference of sensitive personal traits.
4. Do not use the records to make decisions about an individual participant.
5. Apply institutional ethics review, data-protection requirements, and secure storage practices when a new study combines these records with other participant-level data.

Names, email addresses, telephone numbers, account handles, consent forms, recruitment records, and contact fields are outside the public research package. Those records remain under the study team's controlled research administration.

## Model outputs

Model outputs may reflect provider-specific service terms and may quote the input source. Record the provider, full model identifier or checkpoint, endpoint type, generation date, source commit, and configuration when publishing a derivative. Provider permission does not replace permission for third-party text reproduced inside an output.

## Corrections and rights inquiries

Open a repository issue for provenance corrections, attribution requests, or a rights-holder review. Include the exact path, the material in question, the asserted source or right, and a preferred resolution. Maintainers will review substantiated requests and update or restrict the affected artifact as appropriate.
