# Experiment 1 inputs

This directory is the canonical, result-independent input package for Experiment 1.

```text
inputs/
├── cases/
│   ├── experiment-1-cases.json  # Combined six-case collection used by generation code
│   └── H-C1-sentence.json ... H-C6-sentence.json
├── annotations/
│   ├── gold-iso-ranges.csv      # Adjudicated ranges loaded during scoring
│   └── human-iso-ranges.csv     # Human-only condition annotations
├── config/
│   ├── cross-document-scope.json # Bounded cross-document scope for H-C5/H-C6
│   ├── time-string-iso-map.json  # Historical time-expression normalization map
│   └── direct-llm-rules.tex      # Frozen Direct LLM task rules
└── manifest.json                # Counts, sources, and canonical paths
```

The six case files contain 249 source sentences. The evaluation scorer uses 244 eligible rows after applying the paper's inclusion rules. `experiment-1-cases.json` combines the same six canonical case files into the schema consumed by the Direct LLM runner.

Input files remain separate from `results/`, which stores frozen generated artifacts and metrics.

The human range table is the Experiment 1 scoring representation derived from the combined study export preserved under `experiments/shared/human-study/`.

Translation excerpts, project-created annotations, and participant-derived
records follow the category-specific terms in
[`data-licenses.md`](../../../data-licenses.md).
