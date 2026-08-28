# Shared human-study source data

These two JSON files preserve the original combined human-study export used by Experiments 1 and 2:

- `aih-standard-answers.json`: H-C1–H-C6 and T1–T3 reference units.
- `aih-participant-results.json`: six participants, with Experiment 1 and Experiment 2 stages retained together.

Experiment-specific directories contain the inputs required by their own scoring programs. Experiment 1 uses the derived range tables under `experiment-1/inputs/annotations/`; Experiment 2 uses T1–T3-only scoring and response files under `experiment-2/inputs/`.

Participant identifiers use `P1`–`P6`. The release records task responses, case/form assignments, condition labels, and elapsed time for reproducibility and comparison.

Authorization, pseudonymization fields, permitted research uses, and the
prohibition on re-identification are specified in
[`data-licenses.md`](../../../data-licenses.md).
