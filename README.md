# Swap-Consistent, Still Wrong

This repository implements an auditable pilot for testing whether a pairwise
LLM judge can agree with itself under AB/BA candidate swapping while still
selecting the wrong response when a decisive error is located in the middle of
a long answer.

The pilot is a feasibility screen for one configured judge. A positive result
must be replicated across multiple judge families before supporting broad
claims about LLM judges.

## Experimental design

Each source item contains one canonical ordered list of correct sections and
one objectively wrong replacement for exactly one section. The condition
builder deterministically moves that section to early, middle, and late
positions, applies the same permutation to both candidates, and emits both
candidate orders at each configured context length.

The default design has:

- 40 base items;
- two prompt-length bands (4K and 16K);
- three internal error positions (early, middle, and late);
- two display orders (correct-first and flawed-first).

That produces 480 judge calls. At the nominal prompt targets, the run contains
about 4.9 million input tokens. The configured 512-token output cap permits at
most 245,760 output tokens, though the verdict contract should use far fewer.

The source data is intentionally not fabricated by the infrastructure. Create
`data/source/base_items.jsonl` with 40 independently verified items before a
real pilot. Each line must follow this shape:

```json
{
  "item_id": "item_0001",
  "domain": "arithmetic",
  "error_type": "calculation",
  "section_count": 20,
  "correct_sections": [
    {"section_id": "s01", "text": "First correct section."},
    {"section_id": "s02", "text": "Second correct section."}
  ],
  "flawed_section_id": "s11",
  "flawed_section_text": "The objectively incorrect replacement.",
  "gold_winner": "correct",
  "error_explanation": "Why the replacement is wrong.",
  "gold_source": "Machine-checkable or authoritative reference."
}
```

`section_count` must equal the number of entries in `correct_sections`.
Section IDs must be unique, and `flawed_section_id` must identify exactly one
of them. Modular sections should be self-contained enough that the fixed judge
prompt supplies all information needed to compare the responses.

## Metrics and decision rule

- Raw accuracy is the fraction of successfully parsed calls selecting the
  correct candidate.
- Swap consistency is the fraction of complete AB/BA pairs selecting the same
  content identity. Raw A/B letters are never compared directly.
- Consistent Error Rate (CER) is the fraction of swap-consistent pairs whose
  consistent winner is the flawed response.
- False Certification Rate (FCR) is the same wrong-given-consistent quantity
  in this design and is also reported by position.

At 16K, edge metrics pool the early and late observations before computing a
rate. The decision is `GO` only when every condition below passes:

- edge accuracy minus middle accuracy is at least 0.10;
- middle swap consistency is at least 0.80;
- edge swap consistency minus middle swap consistency is at most 0.05;
- middle CER minus edge CER is at least 0.10;
- every data-integrity gate passes.

Any failed or undefined check returns `KILL`. The thresholds live only in
`configs/pilot.yaml`.

## Installation

Python 3.10 or newer is required.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[test]"
```

For OpenAI judge execution, install the optional adapter and set the API key:

```powershell
python -m pip install -e ".[test,openai]"
Copy-Item .env.example .env
$env:OPENAI_API_KEY = "your-key"
```

Then change `judge.provider` from `mock` to `openai` and set an available
model in `configs/pilot.yaml`. The mock provider is deterministic and useful
only for testing the pipeline; its results are not research evidence.

## Reproducing a run

Review the source items and configuration, then run:

```powershell
python -m src.build_conditions --config configs/pilot.yaml
python -m src.validate_conditions --config configs/pilot.yaml
python -m src.run_judge --config configs/pilot.yaml --run-id pilot_001
python -m src.aggregate_pairs --run-id pilot_001
python -m src.analyze --run-id pilot_001
```

Validation is also rerun as a hard gate immediately before judge calls.
`run_judge` randomizes call order using the saved seed, writes every request
before sending it, saves each response immediately, and resumes a partially
completed run without recounting completed condition IDs. Choose a new run ID
for a genuinely new run.

The analysis writes:

```text
data/runs/pilot_001/
??? manifest.json
??? requests.jsonl
??? responses.jsonl
??? parsed.jsonl
??? pair_level.jsonl
??? metrics.json
??? pilot_decision.json
??? figures/
    ??? 01_accuracy_by_position.png
    ??? 02_swap_consistency_by_position.png
    ??? 03_accuracy_consistency_dissociation.png
    ??? 04_cer_by_position.png
    ??? 05_candidate_order_diagnostic.png
```

The manifest preserves the complete configuration, source/condition/prompt
hashes, random seed, provider/model settings, creation time, and Git commit
when the workspace is a Git repository. Bootstrap intervals use base-item
resampling so all matched conditions stay together.

## Integrity gates

Before a decision is valid, the implementation checks:

- at least 99% verdict parse success;
- complete candidate-order pairs and the expected call count;
- no duplicate responses counted;
- clearly separated token-depth distributions;
- configured candidate-length parity;
- unchanged non-target wording and section multisets across relocation;
- one verified flawed replacement and correct counterpart per source item.

Parser failures are retained. The parser accepts exactly one line matching
`VERDICT: A` or `VERDICT: B`; it never guesses from free-form reasoning.

## Tests

```powershell
python -m pytest
```

The suite covers deterministic condition construction, invariant validation,
all identity mappings, strict parsing, pooled metrics, each GO/KILL boundary,
and an offline end-to-end run that creates all analysis artifacts.
