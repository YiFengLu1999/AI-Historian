# Minimal input example

The sample contains two short paragraphs of synthetic vernacular-Chinese demonstration text. Its filename uses lowercase kebab-case, while `input/manifest.json` records the person and title explicitly. Scored evaluation materials are indexed under `experiments/`.

Run the preprocessing stage:

```bash
uv run aih examples/input --output runs/example --through-step 1
```

A full run requires a configured model endpoint:

```bash
uv run aih examples/input --output runs/example_full
```
