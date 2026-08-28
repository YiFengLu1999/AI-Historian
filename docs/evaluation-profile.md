# Evaluation profile

The `ai_historian.profiles.evaluation` package combines the general AIH preprocessing and TimeBlock framework with the frozen semantic-agent implementation used for the final Experiment 1 analysis.

## Stage order

1. `step_01_text_preprocess.py`
2. `step_02_character_detection.py`
3. `step_03_time_info_extraction.py`
4. `step_04_description_detection.py`
5. `step_05_interlude_detection.py`
6. `step_06_timeblock_generation.py`
7. `step_07_timeblock_conversion.py`
8. `step_08_sequence_sorting.py`
9. `step_09_granularity_classification.py`
10. `step_10_tm_generation.py`
11. `step_10c_single_document_stabilize.py`
12. Optional cross-document path: `step_10b_cross_document_prealign.py` and `step_10d_crossdoc_temporal_graph.py`
13. `step_11_iso_normalization.py`
14. Optional presentation stage: `step_14_apply_summary.py`

Use the installed `uv run aih` entry point instead of invoking stages manually. Each stage receives a run-root directory and writes a separate intermediate output directory.

The pipeline uses OpenAI-compatible chat-completions APIs. Runtime credentials are supplied through environment variables or ignored local configuration files.
