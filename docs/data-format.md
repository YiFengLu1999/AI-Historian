# Data format

## Raw input

Place one or more UTF-8 `.txt` files in a directory. Filenames use lowercase
kebab-case and carry no structured metadata:

```text
liu-bang-synthetic-biography.txt
```

Declare the person and title of every text in a sibling `manifest.json`:

```json
{
  "schema": "ai_historian_input_v1",
  "documents": [
    {
      "file": "liu-bang-synthetic-biography.txt",
      "person": "刘邦",
      "title": "Synthetic Biography"
    }
  ]
}
```

Each `.txt` file must appear exactly once in `documents`, and every declared
file must exist. Blank lines delimit paragraphs. Chinese full stops,
exclamation marks, and question marks delimit sentences while preserving
quoted and parenthetical spans.

## Sentence identifier

AIH serializes a sentence identifier as:

```text
<book-uuid>.<chapter-id>.<paragraph-id>.<sentence-id>
```

The identifier is stable within a catalog/input configuration and is retained in every downstream object.

## Sentence object

Important fields are:

- `number`: stable sentence identifier.
- `sentence`: unmodified source sentence.
- `characters`: person-membership judgements.
- `Original_time_information`: whether an explicit temporal expression exists and its source wording.
- `sink`: classification of the sentence as descriptive/background or temporally locatable event content.
- `Interlude`: whether the sentence belongs to a passage displaced from the surrounding narrative timeline.
- `crossDocTransfer`: cross-document relation metadata.

## TimeBlock object

A TimeBlock groups usable sentences governed by one temporal context. Important fields are:

- `ID`: identifier of the first sentence in the block.
- `timeblock_range`: inclusive first/last sentence identifiers.
- `Conversion information`: original and context-completed temporal markers, their basis, and reasoning.
- `Granularity`: temporal precision class.
- `TM`: canonical temporal marker.
- `time_anchor`: anchor type, eligibility, canonical text, and reason.
- `cross_document_context`: evidence and constraints accepted from other documents, when enabled.
- `iso` and `iso_range`: normalized temporal coordinate and inferred range.
- `summary`: readable output generated after inference.

Open boundaries use `-infinity` and `+infinity`. BCE values follow astronomical year numbering, where year `0000` corresponds to 1 BCE.

## Evaluation separation

Experiment generation uses source packets. Gold and human files live under each experiment's `inputs/` directory and enter the scoring workflow after predictions have been finalized.
