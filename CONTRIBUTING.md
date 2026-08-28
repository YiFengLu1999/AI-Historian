# Contributing to AI Historian

[中文](CONTRIBUTING.zh-CN.md) | **English**

Contributions from digital humanities, historical research, natural language processing, and agent-system engineering are warmly welcome.

### Useful contribution areas

- Reproducibility checks for the frozen Experiment 1 and Experiment 2 artifacts
- Runtime-scope retrieval and `episode_packet` ranking
- Cross-document evidence verification and temporal-constraint propagation
- Historical calendar conversion and temporal normalization
- Corpus adapters, metadata catalogs, and source-provenance handling
- Tests, documentation, examples, and usability improvements

### Issue reports

Please include:

1. the execution profile: `evaluation` (`aih`) or `scalable_fulltext` (`aih-fulltext`);
2. the command and relevant environment-variable names;
3. the input and output paths involved;
4. expected and observed behavior;
5. a compact, shareable example or artifact reference.

### Naming conventions

- Use lowercase `kebab-case` for reader-facing directories, documentation, curated inputs, figures, and final result artifacts.
- Keep conventional repository files such as `README.md`, `LICENSE.md`, `CITATION.cff`, and `CONTRIBUTING.md`, plus formal experiment identifiers such as `H-C1` and `T1`.
- Use `snake_case` for Python modules and scripts, other source-code filenames, JSON keys, schema identifiers, and runtime-generated working files.
- Preserve names inside frozen raw-output trees when they are part of the producing pipeline's reproducibility contract; do not treat those internal names as conventions for new public paths.

### Pull requests

Keep changes focused, document user-visible behavior, and add a deterministic test for reusable logic. For changes affecting reported metrics, include the source artifact, aggregation rule, and before/after values.

Thank you for helping make source-grounded AI research for history more transparent, reproducible, and useful. We look forward to your ideas and collaboration.
