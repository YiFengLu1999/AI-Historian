# Experiment 2 inputs

This directory separates participant-facing forms, researcher materials, scoring references, and collected responses.

```text
inputs/
├── forms/
│   ├── T1/
│   ├── T2/
│   └── T3/                      # Questions (.md/.tex/.pdf) and blank response sheets
├── researcher/                  # Per-form keys, combined key/template, and manifest
├── scoring/
│   └── experiment-2-standard-answers.json
└── responses/
    └── experiment-2-participant-responses.json
```

Each form contains eight Block A questions, four Block B questions, and four Block C questions. Across T1–T3, the package contains 48 diagnostic questions covering Agents 2–6 and Agent 9.

The scoring and response JSON files contain T1–T3 records only. The original combined human-study export shared by both experiments is preserved under `experiments/shared/human-study/`.

LaTeX source is retained with each form. Participant-ready PDFs are stored beside the source, while compiler intermediates are excluded from the public input package.

Question text, answer keys, participant-derived records, and embedded
translation excerpts follow the category-specific terms in
[`data-licenses.md`](../../../data-licenses.md).
