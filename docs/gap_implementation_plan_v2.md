# Critical Implementation Plan for Codex

## Audit scope

Repository: `ArhaanShah/Judge`  
Branch: `main`  
Audited HEAD: `fc578d51471ab6753511d4c8feda1e2fc09908d9`  
Audit mode: **critical problems only**

The Gemini paper-analysis path is now sufficiently implemented for the current paper claims. Do not refactor or rerun Gemini unless a regression test fails.

The NVIDIA NIM robustness path is **not yet safe to run**. There are several hard runtime blockers and two scientific-integrity blockers that can make a run look valid when it is not.

Do not start the 240-call NIM main experiment until every item below is fixed and covered by tests.

---

# 1. CRITICAL: the current NIM main runner cannot execute with the current NIM configs

There are two deterministic config/runner mismatches.

## 1.1 `provider_preflight()` requires a legacy `budget` mapping that NIM configs do not define

Current `run_judge.provider_preflight()` does:

```python
budget = require_mapping(config.get("budget"), "budget")
required = int(budget["required_available_calls"])
```

but both current NIM configs define only:

```yaml
scheduler:
  max_concurrency: 1
  minimum_interval_seconds: 60
  max_transport_retries: 2
  base_backoff_seconds: 120
  hard_attempt_cap: 300
```

They have no `budget:` block.

Therefore a NIM main run will fail before the judge loop.

### Fix

Make `provider_preflight()` provider-aware.

For NIM providers:

```text
nvidia_nim
nim
nvidia
```

do not require the legacy Cohere/Gemini monthly-call-budget mapping.

Require only:

```text
NVIDIA_API_KEY present
explicit operator live-run confirmation
scheduler.hard_attempt_cap present
preflight/freeze gates already passed
```

The shared `AttemptBudget` is the authoritative local NIM cap.

Do not invent an NVIDIA quota value.

For legacy providers, preserve the existing budget behavior.

---

## 1.2 `run_judge()` indexes `judge.max_output_tokens`, but NIM configs only define `judge.max_tokens`

Current main loop calls:

```python
provider.judge(
    ...,
    max_output_tokens=int(judge_config["max_output_tokens"]),
)
```

Current NIM configs contain:

```yaml
max_tokens: 16384
```

for GLM and:

```yaml
max_tokens: 32768
```

for Nemotron.

They do not contain `max_output_tokens`.

This is a guaranteed `KeyError` before the first NIM main API call.

### Fix

Standardize one resolved output-token field.

Recommended:

```python
def resolved_max_tokens(judge_config):
    value = judge_config.get("max_tokens")
    if value is None:
        value = judge_config.get("max_output_tokens")
    if value is None:
        raise ValueError("judge config requires max_tokens")
    return int(value)
```

Use this helper in:

```text
run_judge.py
preflight_nim.py
freeze_nim.py
```

For NIM, `max_tokens` is canonical.

Do not keep two independently editable values in the NIM YAML files.

### Acceptance test

A mocked NIM main run must reach `provider.judge()` without:

```text
KeyError: budget
KeyError: max_output_tokens
```

---

# 2. CRITICAL: the current preflight can falsely mark an unverified model `ELIGIBLE`

This is the most important scientific bug.

The preflight currently sets:

```python
report["reasoning_verified"] = True
```

even if the OFF request throws an exception or continues to produce reasoning.

It also hardcodes:

```python
report["response_format_probe"] = "provider_schema"
report["selected_output_format_strategy"] = "provider_schema"
```

without performing a response-schema probe.

The six-condition smoke test counts a response as parsed if:

```python
resp.brief_reason is not None
```

but does **not** require a valid:

```text
winner ∈ {A, B, tie}
```

Therefore a model can currently pass preflight despite:

```text
no verified reasoning control
no verified output schema
no valid winner field
```

This invalidates the intended stronger-reasoning replication.

---

## 2.1 Reasoning verification must be real ON-vs-OFF verification

### Nemotron

The configured main request is:

```yaml
reasoning_effort: high
reasoning_budget: 16384
```

Preflight must compare it against a documented no-thinking control.

Require all of:

```text
ON request succeeds
OFF request succeeds

ON reasoning_present == true

OFF materially suppresses reasoning:
    reasoning_present == false
    OR reasoning token/character count is near-zero relative to ON

returned model is the expected Nemotron model family
```

If OFF errors because the control is unsupported:

```text
THINKING_UNVERIFIED
```

Do not ignore the exception.

If OFF still reasons materially:

```text
THINKING_UNVERIFIED
```

Do not mark verified.

---

## 2.2 GLM must not be eligible without an explicit configured reasoning control

Current GLM config has no reasoning field at all.

That means the current "ON" call is just the default model behavior.

For GLM:

```text
default reasoning appearing in a response != verified high thinking
```

### Required workflow

If probing GLM reasoning-control request shapes is desired:

1. Probe candidate request shapes.
2. If one works, update the GLM config to contain that exact request.
3. Rerun preflight from the beginning with the finalized config.
4. Compare the finalized configured ON request against a valid OFF request.
5. Only then permit `ELIGIBLE`.

Do not allow preflight to discover one request shape and then run main with a different static config.

If no explicit high/max control can be verified:

```text
GLM status = THINKING_UNVERIFIED
```

and use Nemotron fallback.

---

## 2.3 Perform a real response-format probe

The historical substantive prompt says:

```text
Return a JSON object only, following the required schema.
```

but does not spell the schema out because Gemini used provider-side schema enforcement.

The current NIM configs have no `response_format`.

Yet preflight currently records `provider_schema` anyway.

### Fix

Probe an actual provider-side schema request with:

```json
{
  "type": "object",
  "properties": {
    "winner": {
      "type": "string",
      "enum": ["A", "B", "tie"]
    },
    "brief_reason": {
      "type": "string"
    }
  },
  "required": ["winner", "brief_reason"],
  "additionalProperties": false
}
```

If provider-side structured output works:

```text
selected_output_format_strategy = provider_schema
```

and store/freeze the exact schema.

If it does not work, use a formatting-only prompt suffix:

```text
config/prompts/nim_schema_suffix.txt
```

that explicitly provides the JSON schema without changing judging criteria.

Then:

```text
selected_output_format_strategy = prompt_schema_suffix
```

Freeze the suffix hash.

Never hardcode `provider_schema` without an actual successful probe.

---

## 2.4 The six-condition smoke must validate the full verdict contract

For each of the six actual frozen 16K conditions require:

```text
successful response
expected returned model family
reasoning_present == true
normal/non-truncating finish reason

final JSON parses

winner in {"A", "B", "tie"}
brief_reason is a string
```

Do not use:

```python
brief_reason is not None
```

as the parse criterion.

Use the same parsing function that the main run uses, or a shared strict helper.

Preflight must fail if any of the six conditions fails this contract.

---

## 2.5 Smoke calls must use the intended main token budget

Current smoke calls default to roughly:

```text
2048 output tokens
```

because `max_output_tokens` is absent from NIM configs.

For a high-reasoning model, this can truncate reasoning and produce a false preflight failure or a non-representative smoke.

Use the canonical configured:

```text
max_tokens
```

for the real six-condition smoke.

Small budgets are fine only for trivial availability probes.

---

# 3. CRITICAL: bind the actual frozen 16K subset to its provenance manifest

The repository now contains:

```text
artifacts/nim_16k_subset_manifest.json
```

with a recorded:

```text
subset_sha256
```

but the actual generated:

```text
data/processed/pilot_conditions_16k.jsonl
```

is intentionally not committed.

That is acceptable only if preflight verifies the generated local subset against the committed manifest.

Current preflight verifies:

```text
parent_full_condition_sha256
n_packets
n_pairs
```

but does **not** verify:

```text
sha256(actual subset file) == manifest["subset_sha256"]
```

Therefore a modified local 240-row subset can pass preflight and be frozen as if it were the exact Gemini-derived subset.

That breaks the paired replication guarantee.

---

## 3.1 Add mandatory subset hash verification

Before any live NIM API call:

```python
actual_subset_sha = sha256_file(main_conditions)
expected_subset_sha = subset_manifest["subset_sha256"]

if actual_subset_sha != expected_subset_sha:
    LOCAL_INTEGRITY_FAILED
```

Also verify:

```text
n_conditions == 240
n_packets == 40
n_pairs == 120
all lengths == 16K
```

from the actual file, not only from manifest metadata.

---

## 3.2 Verify the source Gemini freeze-manifest hash

The subset manifest records:

```text
source_gemini_freeze_manifest_sha256
```

Verify it against the actual:

```text
artifacts/pilot_freeze_manifest_gemini.json
```

before live calls.

This closes the provenance chain:

```text
Gemini frozen full conditions
    -> exact filtered 16K subset
    -> NIM preflight
    -> NIM freeze
    -> NIM main run
```

---

## 3.3 Do not manually edit the generated subset

The only supported way to create the local file should be:

```bash
python -m src.prepare_nim_subset
```

Then validate it immediately.

If the subset hash differs from the committed manifest, stop and investigate rather than regenerating a new manifest automatically.

The committed manifest is the research artifact.

---

# 4. CRITICAL: freeze and main must use the exact reasoning/output request that passed preflight

Current `freeze_nim.py` builds its reasoning hash from static config fields:

```python
{
    "reasoning_effort": judge.get("reasoning_effort"),
    "reasoning_budget": judge.get("reasoning_budget")
}
```

but preflight does not currently record/verify a resolved exact reasoning request.

For GLM this is especially dangerous because the config currently has no reasoning fields.

The freeze can therefore claim an "eligible" preflight while freezing a different or empty reasoning request.

---

## 4.1 Preflight must write the exact resolved main request metadata

Add to `preflight.json`:

```text
selected_reasoning_request
selected_reasoning_request_sha256

selected_output_format_strategy
selected_response_format_or_suffix_sha256

configured_model
verified_returned_model_ids

configured_max_tokens
configured_temperature
configured_top_p
configured_seed
```

These values must describe the **actual request that main will send**.

---

## 4.2 Freeze must compare, not merely copy

`freeze_nim.py` must fail unless:

```text
preflight.config_sha256 == current config SHA

preflight.subset_sha256 == current subset SHA

preflight.prompt_sha256 == current prompt SHA

preflight.selected_reasoning_request_sha256
    == reasoning request derived from current config

preflight.selected_response_format_or_suffix_sha256
    == output-format request derived from current config/files
```

Then freeze those same hashes.

---

## 4.3 Main-run gate must verify the same hashes again

Before the first main request, `run_judge.py` must verify:

```text
config
subset
prompt
preflight report
reasoning request
output-format strategy/schema/suffix
```

against the NIM freeze manifest.

Do not rely only on the whole config hash if any provider request is dynamically resolved outside the config.

---

# 5. CRITICAL: the completed NIM run needs a hard integrity gate before it can be called replication evidence

Current `src/nim_analysis.py` mostly prints warnings.

For example:

```text
expected 240 calls -> warning only
parse success <98% -> warning only
```

It does not currently enforce:

```text
one returned model
reasoning present
normal finish reasons
no provider-error contamination
complete 120 pairs
per-position parse threshold
```

A damaged run could therefore still produce paper summaries.

That is not acceptable for the only independent robustness replication.

---

## 5.1 Convert integrity warnings into a hard inclusion decision

Write:

```text
results/nim/<model>/paper/audit.json
```

with:

```text
passed: true|false
paper_eligible: true|false
checks: {...}
```

Require for `paper_eligible == true`:

```text
exactly 240 intended conditions represented
exactly 120 complete AB/BA pairs
40 packets
40 complete pairs in each of early/middle/late

no duplicate condition IDs

overall parse success >= 98%
each position parse success >= 95%

one returned model family
returned model matches frozen model family

no unresolved provider errors in the completed run

all paper-included calls have required reasoning_present == true

no truncating/incomplete finish reasons

run manifest hashes match NIM freeze
```

If any fails:

```text
paper_eligible = false
```

The analysis may still produce a debugging summary, but it must print clearly:

```text
NOT PAPER-ELIGIBLE
```

and must not generate the normal paper-ready table under the standard filename.

---

## 5.2 Produce the actual robustness contrast with uncertainty

For a paper-eligible run compute, at 16K:

```text
edge = pooled early + late
```

and report:

```text
raw_accuracy_edge_minus_middle
position_consistency_edge_minus_middle
certification_coverage_edge_minus_middle
```

with 10,000 packet-cluster bootstrap draws.

Also compute the paired exploratory Gemini-vs-NIM difference using the same packet resample for both models.

This is the statistic the robustness experiment is being run to answer.

Do not stop at per-position point estimates.

---

# 6. CRITICAL OPERATIONAL: repair the corrupted NIM `.gitignore` entries

The current `.gitignore` contains NUL-separated/corrupted entries for:

```text
results/nim/*/raw/
results/nim/*/api_attempt_counter.json
```

They are not normal Git ignore patterns.

Replace them with ordinary UTF-8 text:

```gitignore
results/nim/*/raw/
results/nim/*/api_attempt_counter.json
```

Then verify with:

```bash
git check-ignore -v results/nim/glm52/raw/example.jsonl
git check-ignore -v results/nim/glm52/api_attempt_counter.json
git check-ignore -v results/nim/nemotron3super/raw/example.jsonl
```

This prevents accidental commits of full prompts/responses and local attempt-state files.

Do not ignore:

```text
preflight.json
paper aggregate outputs
freeze manifests
```

---

# 7. Minimal regression tests required before live calls

Do not add a broad test refactor. Add only tests that prevent the critical failures above.

Current `tests/test_nim.py` is effectively a smoke import test and does not exercise the experiment controls.

Add or replace with targeted tests.

## 7.1 Runner compatibility

```text
test_nim_config_does_not_require_legacy_budget_mapping

test_nim_main_uses_max_tokens_when_max_output_tokens_absent
```

---

## 7.2 Reasoning preflight

```text
test_preflight_fails_if_reasoning_off_request_errors

test_preflight_fails_if_reasoning_off_still_reasons

test_glm_without_explicit_verified_reasoning_control_is_not_eligible

test_preflight_requires_reasoning_on_all_six_real_smoke_calls
```

---

## 7.3 Output schema

```text
test_preflight_does_not_claim_provider_schema_without_probe

test_smoke_requires_valid_winner_and_brief_reason

test_invalid_winner_fails_preflight
```

---

## 7.4 Provenance

```text
test_preflight_fails_if_subset_sha_differs_from_manifest

test_preflight_fails_if_gemini_freeze_sha_differs_from_subset_manifest
```

---

## 7.5 Freeze/main binding

```text
test_freeze_fails_if_reasoning_request_differs_from_preflight

test_freeze_fails_if_output_format_differs_from_preflight

test_main_fails_if_reasoning_or_output_format_differs_from_freeze
```

---

## 7.6 Completed-run audit

```text
test_nim_analysis_not_paper_eligible_on_model_drift

test_nim_analysis_not_paper_eligible_on_missing_reasoning

test_nim_analysis_not_paper_eligible_on_truncation

test_nim_analysis_not_paper_eligible_on_incomplete_pairs

test_nim_analysis_not_paper_eligible_below_parse_threshold
```

All tests must be mocked and make zero live requests.

---

# 8. Codex execution order

Do this in exactly this order.

## Phase A — zero API calls

```text
1. Fix provider_preflight NIM budget handling.

2. Standardize NIM max_tokens handling across:
       run_judge
       preflight_nim
       freeze_nim

3. Fix preflight reasoning verification:
       real ON vs OFF
       no ignored OFF failures
       no unconditional reasoning_verified=true

4. Implement real response-format probing.

5. Make six-condition smoke require:
       expected model
       reasoning
       normal finish
       valid winner
       valid brief_reason

6. Bind smoke to configured main max_tokens.

7. Add subset SHA verification against committed manifest.

8. Verify Gemini freeze-manifest SHA from subset manifest.

9. Add exact selected reasoning/output hashes to preflight.

10. Make freeze compare those hashes.

11. Make main compare those hashes.

12. Convert NIM analysis integrity checks into a hard paper-eligibility gate.

13. Add edge-minus-middle bootstrap contrasts and paired Gemini comparison.

14. Repair .gitignore NIM entries.

15. Add the minimal critical regression tests above.

16. Run the full test suite.
```

Do not make a live request unless all tests pass.

---

## Phase B — materialize and validate the exact subset

```bash
python -m src.prepare_nim_subset
```

Then verify:

```text
actual subset SHA ==
9d64d695c2f0c0aa304becda482b3a160a2276a4f1489877ea64fe53a1890822
```

Then run PRMBench validation for the selected NIM config.

Do not rewrite `artifacts/nim_16k_subset_manifest.json` merely because the generated hash differs.

A mismatch is an error requiring investigation.

---

## Phase C — GLM preflight only

GLM may be used only if the finalized config contains an explicit high/max reasoning request and that exact request passes ON-vs-OFF verification.

If no explicit high-thinking control is verified:

```text
GLM -> THINKING_UNVERIFIED
```

Do not start GLM main.

Proceed to Nemotron.

---

## Phase D — Nemotron fallback preflight

Use the frozen intended configuration:

```text
model = nvidia/nemotron-3-super-120b-a12b
temperature = 1.0
top_p = 0.95
reasoning_effort = high
reasoning_budget = 16384
max_tokens = 32768
seed = 20260828
stream = true
```

Require:

```text
reasoning ON/OFF verified
output format verified
6/6 real frozen smoke conditions valid
returned model verified
```

Only then:

```text
status = ELIGIBLE
```

---

## Phase E — freeze, then main

Freeze the selected model only after an `ELIGIBLE` preflight.

Before the first main request verify:

```text
subset
prompt
config
reasoning request
output format
preflight
freeze
```

all match.

Then run exactly:

```text
40 packets × 3 positions × 2 orders = 240 calls
```

No mid-run model fallback.

---

# 9. Done criteria

The NIM experiment is ready only when all are true:

```text
[ ] Current NIM config reaches provider.judge without KeyError.

[ ] NIM does not require the legacy budget mapping.

[ ] Main and smoke use canonical max_tokens.

[ ] Reasoning ON-vs-OFF is actually verified.

[ ] GLM cannot become eligible from default/implicit reasoning alone.

[ ] Output schema strategy was actually tested.

[ ] 6/6 real smoke calls have valid winner + brief_reason.

[ ] 6/6 real smoke calls use required reasoning.

[ ] Returned model is verified.

[ ] Actual subset SHA matches committed subset manifest.

[ ] Source Gemini freeze SHA matches subset provenance.

[ ] Preflight, freeze, and main bind the same exact reasoning request.

[ ] Preflight, freeze, and main bind the same exact output-format strategy.

[ ] NIM completed-run audit can hard-fail paper eligibility.

[ ] Paper-eligible analysis reports packet-bootstrap edge-minus-middle contrasts.

[ ] Raw NIM run paths are actually gitignored.

[ ] All critical regression tests pass.
```

Until then, do **not** spend the 240-call NIM run budget.
