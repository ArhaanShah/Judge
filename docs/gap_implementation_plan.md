# Implementation Plan — Gap Remediation After Repo Audit

## Audit target

Repository: `ArhaanShah/Judge`  
Branch: `main`  
Audited HEAD: `d352c498a10aca6d23200b288c65411c2690b546`  
Audit date: 2026-08-30

This is a **gap-only plan**. Do not reimplement working Gemini paper-analysis code unless a task below explicitly requires a compatibility or correctness change.

The repository has implemented most of the Gemini-side reanalysis. The NVIDIA NIM path has substantial scaffolding, but it is **not yet safe to run as paper-quality evidence**.

Do not launch the 240-call NIM robustness experiment until every P0 acceptance criterion in this document passes.

---

# 1. Preserve what already works

## 1.1 Gemini historical-run audit

Keep the current `src/paper_analysis.py` audit behavior and committed artifact:

```text
results/gemini/paper/audit.json
```

The current audit correctly verifies:

```text
480 completed calls
240 complete AB/BA pairs
40 packets
40 pairs per length × position cell
100% parse success
one returned Gemini model
no duplicate condition IDs
condition hash matches frozen experiment
prompt hash matches frozen experiment
no incomplete responses
no missing prompt-token metadata
```

Do not rerun Gemini simply to reproduce this.

## 1.2 Tie-aware pair outcomes

Preserve the paper-facing five-way partition:

```text
certified_correct
certified_wrong
stable_tie
mixed_tie
order_disagreement
```

Preserve the distinction:

```text
position_consistency
```

where `tie/tie` counts as same-outcome consistency,

versus:

```text
certification_coverage
```

where only matching non-tie winners form a decisive certificate.

## 1.3 Selective-risk uncertainty

Keep:

```text
selective_risk =
    certified_wrong / decisively_certified
```

with exact two-sided Clopper–Pearson intervals.

When `n_certified == 0`:

```text
selective_risk = None
selective_risk_exact_ci = None
```

Do not revert to bootstrap-only CER intervals.

## 1.4 Gemini interaction analysis

Preserve the current packet-cluster bootstrap.

Current committed point estimates:

```text
raw-accuracy length × middle-position interaction:
    +0.1875
    +18.75 percentage points

decisive-certification-coverage interaction:
    +0.1500
    +15.0 percentage points

standard position-consistency interaction:
    -0.0500
    -5.0 percentage points
```

Current 95% packet-bootstrap intervals:

```text
raw accuracy:
    [0.0625, 0.30625]

decisive certification coverage:
    [-0.05, 0.35]

position consistency:
    [-0.30, 0.20]
```

Paper guardrail:

**The accuracy interaction is supported by the current interval.  
The +15 pp certification-coverage interaction is directional/exploratory because its current interval includes zero.**

Do not write that the certification-coverage interaction is statistically conclusive.

## 1.5 Token/position sanity checks

Preserve the historical Gemini native-token metadata and tokenizer-independent position checks.

Current Gemini prompt lengths are approximately:

```text
4K construction band:
    median total Gemini prompt tokens ≈ 8,873

16K construction band:
    median total Gemini prompt tokens ≈ 34,870
```

Continue calling 4K/16K **construction bands**, not literal Gemini-token lengths.

---

# 2. Update the scientific interpretation to match the new decomposition

The current reanalysis reveals a stronger and more precise mechanism than the original draft.

Gemini 16K-middle:

```text
raw accuracy                     = 31.25%
position consistency             = 60%
decisive certification coverage  = 15%

stable tie                       = 45%
mixed tie                        = 35%
tie-involved                     = 80%
pure clean↔flawed disagreement   = 5%

certified errors                 = 0/6
exact 95% selective-risk CI      ≈ [0%, 45.9%]
```

Among non-certified 16K-middle pairs:

```text
tie share of non-certification ≈ 94.1%
```

Therefore the current evidence does **not** support a primary story of AB↔BA winner flipping.

Preferred wording:

> Under long-context middle-position stress, the judge becomes substantially less decisive. Broad position consistency can remain high because stable ties count as consistent, while decisive certification coverage becomes very low.

Avoid:

```text
"swap consistency collapses in the middle"
"middle position causes large candidate-order bias"
"the swap gate mostly fails because the judge reverses its winner"
```

Prefer:

```text
"decisive certification coverage falls"
"non-certification is predominantly tie-mediated"
"the reliability procedure increasingly withholds a decisive certificate"
```

Update:

```text
docs/paper_claims.md
```

with these exact current facts and the coverage-CI caveat.

---

# 3. P0 — Repair the NIM condition/config path before any live request

The current NIM configs reference paths that do not exist or do not match the actual experiment layout:

```text
data/conditions/16k_subset.jsonl
config/prompts/judge_template.txt
results/gemini/raw/main_pilot_3/freeze_manifest.json
```

The actual frozen judge prompt is:

```text
config/prompts/pairwise_judge.txt
```

## 3.1 Add a deterministic 16K-subset preparation command

Create:

```text
src/prepare_nim_subset.py
```

Behavior:

1. Locate/read the **exact existing Gemini main condition file**.
2. Verify its SHA-256 equals the frozen Gemini condition hash:

```text
97afc9f46502ea9e9efe45804e4e73cc38481d1f6e34be7f7a0384298891f14a
```

3. Abort if the parent hash differs.
4. Filter only:

```text
length == "16K"
```

5. Do not regenerate candidate text from PRMBench.
6. Do not retokenize/rebuild packets.
7. Preserve exactly:
   - `condition_id`
   - `base_packet_id`
   - candidate texts
   - candidate identities
   - target ID
   - target index
   - filler IDs
   - position
   - candidate order
   - all other condition metadata
8. Write:

```text
data/processed/pilot_conditions_16k.jsonl
```

9. Verify:

```text
240 conditions
40 packets
3 positions per packet
2 orders per packet-position
120 complete AB/BA pairs
all length == 16K
```

10. Write:

```text
artifacts/nim_16k_subset_manifest.json
```

containing:

```text
parent_full_condition_sha256
subset_sha256
n_conditions
n_packets
n_pairs
source_gemini_freeze_manifest_sha256
generation_code_sha256
timestamp
```

The purpose is to prove the robustness model saw **the same 16K stimuli**, not a rebuilt approximation.

## 3.2 Repair both NIM YAML configs

Update:

```text
config/pilot_prmbench_nim_glm52.yaml
config/pilot_prmbench_nim_nemotron3super.yaml
```

They must contain all fields required by the PRMBench validation/run path.

Recommended shape:

```yaml
experiment_name: ...
seed: 20260828

paths:
  source_rows: data/source/prmbench_rows.jsonl
  source_manifest: data/source/prmbench_manifest.json

  main_conditions: data/processed/pilot_conditions_16k.jsonl
  main_validation_report: data/processed/main_validation_report_nim_<model>.json

  prompt_template: config/prompts/pairwise_judge.txt

  subset_manifest: artifacts/nim_16k_subset_manifest.json
  freeze_manifest: artifacts/nim_<model>_freeze_manifest.json
  preflight_report: results/nim/<model>/preflight.json
  attempt_counter: results/nim/<model>/api_attempt_counter.json

  runs_dir: results/nim/<model>/raw
  results_dir: results/nim/<model>/paper

tokenizer: CohereLabs/command-a-plus-05-2026-w4a4

lengths:
  16K:
    target: 16000
    min: 15200
    max: 16800

positions:
  early:  {target: 0.15, min: 0.10, max: 0.20}
  middle: {target: 0.50, min: 0.45, max: 0.55}
  late:   {target: 0.85, min: 0.80, max: 0.90}
```

Do not include a 4K length in a 16K-only NIM replication config.

The existing PRMBench validator should then expect:

```text
40 × 1 length × 3 positions × 2 orders = 240 conditions
```

---

# 4. P0 — Add a NIM-specific freeze manifest

Do not use the historical Gemini freeze manifest as the NIM run's own freeze manifest.

Create:

```text
src/freeze_nim.py
```

A separate freezer is safer than forcing the post-hoc robustness run through the preregistered pilot freezer.

## 4.1 Freeze manifest must contain

```text
git_commit
config_sha256

dataset_revision
dataset_fingerprint

parent_gemini_condition_sha256
nim_16k_subset_sha256
subset_manifest_sha256

prompt_sha256

provider
model

temperature
top_p
max_tokens
seed
stream

exact_reasoning_request_json
reasoning_request_sha256

output_format_strategy
response_schema_sha256_or_suffix_sha256

preflight_report_sha256

scheduler:
    max_concurrency
    minimum_interval_seconds
    max_transport_retries
    base_backoff_seconds
    hard_attempt_cap

analysis_code_sha256
timestamp
```

## 4.2 Freeze rules

Require:

```text
clean Git worktree
PRMBench 16K subset validation passed
preflight status == ELIGIBLE
all preflight hashes match current files/config
```

Once frozen, any changed:

```text
model
temperature
top_p
max_tokens
reasoning settings
prompt
schema/suffix
subset
scheduler
analysis code
```

requires a new freeze and new run ID.

---

# 5. P0 — Declare the NIM HTTP dependency

Current `src/providers/nvidia_nim_provider.py` imports `httpx`, but `pyproject.toml` does not declare it.

Add:

```toml
[project.optional-dependencies]
nim = ["httpx>=0.27,<1"]
```

Also include `httpx` in the recommended experiment install path if `pilot` remains the umbrella extra.

Update README install instructions accordingly.

---

# 6. P0 — Complete NIM response metadata

Keep the current direct-HTTP/SSE architecture.

It is a good choice because the experiment needs exact request control and raw rate-limit metadata.

Fix the missing pieces.

## 6.1 Capture returned model identity

Current NIM provider sets:

```text
returned_model = None
```

Fix it.

Synchronous response:

```text
returned_model = data.get("model")
request_id = data.get("id") or response.headers.get("x-request-id")
```

Streaming:

capture the first non-null:

```text
chunk.get("model")
chunk.get("id")
```

across SSE chunks.

Populate `JudgeResponse.returned_model`.

The completed-run integrity audit must be able to assert one returned model family.

## 6.2 Preserve usage

Capture when available:

```text
prompt_tokens
completion_tokens
total_tokens
reasoning_tokens
```

Keep `prompt_tokens` and `completion_tokens` in normal `JudgeResponse` fields and put additional usage metadata in `raw_response`.

## 6.3 Preserve all rate-limit headers

Capture every header whose lowercase name starts with:

```text
x-ratelimit-
```

plus:

```text
retry-after
```

Do not assume NVIDIA always uses one exact header set.

## 6.4 Propagate `Retry-After`

On a retriable response:

```python
err = RetriableProviderError(...)
err.retry_after = parsed_retry_after
err.response_headers = captured_rate_limit_headers
raise err
```

The runner already checks `retry_after`; the provider currently does not attach it.

## 6.5 Do not commit raw chain-of-thought

Continue storing only:

```text
reasoning_present
reasoning_character_count
reasoning_sha256
reasoning_tokens_if_reported
```

Do not require raw reasoning text in committed artifacts.

---

# 7. P0 — Make the NIM output contract protocol-equivalent

The historical judge prompt says:

```text
"Return a JSON object only, following the required schema."
```

Gemini received a provider-side schema.

The current NIM provider sends no response schema.

Therefore the NIM model is currently missing part of the output contract.

## 7.1 Probe `response_format`

During preflight, test NVIDIA's OpenAI-compatible `response_format` using the same schema:

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

Record whether the endpoint:

```text
accepts it
rejects it
accepts but ignores it
```

## 7.2 Preferred output-format strategy

If provider-side schema enforcement works:

```text
use exact historical substantive prompt
+
provider-side response schema
```

Freeze the exact schema JSON/hash.

## 7.3 Fallback output-format strategy

If NIM does not support provider-side schema enforcement, create:

```text
config/prompts/nim_schema_suffix.txt
```

containing only formatting instructions, for example:

```text
The required output schema is exactly:
{"winner":"A"|"B"|"tie","brief_reason":"string"}
Return no additional keys or surrounding prose.
```

Append this to the same substantive judge prompt.

Freeze/hash the suffix.

Document that:

```text
substantive judging instructions are unchanged
output-format enforcement differs by provider
```

Do not change mathematical/logical judging criteria.

---

# 8. P0 — Rewrite `src/preflight_nim.py`

The current preflight is insufficient for paper evidence.

Current issues include:

```text
"Hello" availability prompt
2+2 reasoning prompt
no ON/OFF reasoning verification
no returned-model check
two toy JSON calls only
toy output prompt differs from real judge prompt
~2,000 filler words instead of real long prompts
no real six-condition factorial smoke
minimal status/reason report
```

Replace it with the following staged preflight.

## 8.1 Stage 0 — local integrity, zero API calls

Before any network request:

```text
verify NVIDIA_API_KEY is available
verify config file
verify subset manifest
verify parent Gemini condition SHA
verify 240-condition subset SHA
verify actual prompt SHA
verify expected 40 packets / 120 pairs
verify no incompatible existing freeze
```

Failure status:

```text
LOCAL_INTEGRITY_FAILED
```

and send zero model calls.

## 8.2 Stage 1 — baseline endpoint availability

For GLM, do not inject an unverified `reasoning_effort` field into the baseline availability request.

Send only basic documented fields:

```text
model
messages
temperature
max_tokens
stream
seed
```

Classify errors accurately:

```text
401/403 -> AUTH_FAILED
404/model retired -> UNAVAILABLE
422 -> REQUEST_SCHEMA_FAILED
429 -> rate-limited/retry path
5xx -> transient
```

Do not label every error as `UNAVAILABLE`.

Verify returned model identity.

## 8.3 Stage 2 — GLM reasoning-control verification

GLM is the preferred model only if explicit high/max thinking can be verified.

Probe candidate request forms separately, for example:

```text
A:
reasoning_effort = "high"
vs
reasoning_effort = "none"

B:
chat_template_kwargs = {
    "enable_thinking": true,
    "clear_thinking": false
}
vs
enable_thinking = false
```

Use any newer official request shape if NVIDIA's docs have changed by execution time.

Eligibility requires:

```text
ON accepted
OFF accepted
ON produces observable reasoning metadata/content
OFF materially suppresses reasoning
```

If high/max thinking cannot be empirically verified:

```text
status = THINKING_UNVERIFIED
```

and GLM is not eligible for the main robustness run.

Do not silently run GLM at an unknown/default reasoning level.

## 8.4 Stage 3 — Nemotron reasoning verification

Fallback model:

```text
nvidia/nemotron-3-super-120b-a12b
```

Verify:

```text
reasoning_effort = high
reasoning_budget = 16384
```

against an OFF/no-thinking control.

Require observable ON/OFF differentiation.

## 8.5 Stage 4 — output-format/schema probe

Run the `response_format` probe from Section 7.

Choose/freeze either:

```text
provider_schema
```

or:

```text
prompt_schema_suffix
```

before smoke.

## 8.6 Stage 5 — exact real six-condition smoke

Choose one packet deterministically before looking at correctness, for example:

```text
lexicographically first base_packet_id
```

Run all six actual frozen 16K conditions:

```text
early clean-first
early flawed-first
middle clean-first
middle flawed-first
late clean-first
late flawed-first
```

Use the exact intended:

```text
substantive prompt
selected model
sampling settings
reasoning settings
output-format strategy
```

Require:

```text
6/6 successful service responses
6/6 intended returned model family
6/6 required reasoning active
6/6 normal finish
6/6 parse to winner ∈ {A,B,tie}
6/6 string brief_reason
```

Do not require the model to be correct.

## 8.7 Stage 6 — additional full-length capacity probes

Use at least one different frozen packet.

Run real:

```text
early
middle
late
```

conditions.

Do not synthesize filler.

Require no:

```text
context-length failure
truncation
persistent 429 state
model drift
reasoning failure
```

## 8.8 Rich machine-readable preflight report

Write:

```text
results/nim/<model>/preflight.json
```

with:

```text
status
timestamp

configured_model
returned_model_ids
endpoint

config_sha256
subset_sha256
prompt_sha256

reasoning_probe_attempts
selected_reasoning_request_json
reasoning_on_metadata
reasoning_off_metadata
reasoning_verified

response_format_probe
selected_output_format_strategy

smoke_condition_ids
smoke_success_count
smoke_parse_success_count
smoke_finish_reasons

capacity_condition_ids
capacity_success_count

rate_limit_headers_seen
transport_errors
total_http_attempts
```

Suggested terminal statuses:

```text
ELIGIBLE
UNAVAILABLE
AUTH_FAILED
REQUEST_SCHEMA_FAILED
THINKING_UNVERIFIED
OUTPUT_CONTRACT_FAILED
FULL_LENGTH_FAILED
RATE_LIMIT_UNUSABLE
LOCAL_INTEGRITY_FAILED
```

---

# 9. P0 — Make preflight a hard gate for NIM main runs

`run_judge.py` must refuse an NIM main run unless:

```text
preflight report exists
preflight status == ELIGIBLE
NIM freeze manifest exists
preflight hash matches freeze
config hash matches freeze
subset hash matches freeze
prompt hash matches freeze
reasoning request hash matches freeze
output-format hash matches freeze
```

No `--force` override for a paper-quality NIM run.

---

# 10. P0 — Fix NVIDIA API-key mapping

Current generic provider preflight derives an unknown provider key as:

```text
NVIDIA_NIM_API_KEY
```

but the NIM provider uses:

```text
NVIDIA_API_KEY
```

Add:

```python
"nvidia_nim": "NVIDIA_API_KEY",
"nim": "NVIDIA_API_KEY",
"nvidia": "NVIDIA_API_KEY",
```

to `provider_preflight()`.

Also call `load_env_file()` in `src/preflight_nim.py` before checking the environment so `.env` behaves consistently.

---

# 11. P0 — Correct retry semantics

`ProviderError` means non-retriable.

Do not retry it.

Runner behavior must be:

```python
except RetriableProviderError:
    retry_if_budget_remains()

except ProviderError:
    record_failure()
    break
```

Non-retriable examples:

```text
400
401
403
404
422
invalid frozen request field
invalid model slug
unsupported parameter
```

Retriable examples:

```text
429
500
502
503
504
network reset
read timeout
```

Malformed model output is also **not** a transport retry condition.

---

# 12. P0 — Actually use configured NIM backoff

Current runner reads:

```text
base_backoff_seconds
```

but still sleeps using the old small exponential delay.

Change retry delay to:

```text
max(
    parsed Retry-After if present,
    base_backoff_seconds * 2**attempt
)
```

Add modest jitter, e.g. ±10%.

For the robustness run freeze:

```text
max_concurrency = 1
minimum_interval_seconds = 60
max_transport_retries = 2
base_backoff_seconds = 120
hard_attempt_cap = 300
```

Do not dynamically speed up based on model answers.

---

# 13. P0 — Share one attempt counter across preflight and main run

Current main-run counter does not include preflight calls.

Add:

```text
src/attempt_budget.py
```

or equivalent helpers.

Use the config path:

```text
paths.attempt_counter
```

from both:

```text
preflight_nim.py
run_judge.py
```

Count:

```text
availability probes
reasoning ON/OFF probes
response-format probes
six-condition smoke
capacity probes
main calls
retries
```

Hard local cap:

```text
300 HTTP attempts
```

for the selected model experiment.

This is our safety cap, not a claim about NVIDIA's actual quota.

Persist:

```json
{
  "api_attempts": 17,
  "model": "...",
  "config_sha256": "...",
  "updated_at": "..."
}
```

---

# 14. P0 — Freeze all inference parameters on resume

Current PRMBench resume protection is insufficient.

A run must reject resume if any of these changed:

```text
config
prompt
subset
provider
model
temperature
top_p
max_tokens
seed
reasoning settings
exact reasoning request body
response-format strategy/schema
scheduler
freeze manifest
```

Persist hashes in each run manifest:

```text
config_sha256
prompt_sha256
subset_sha256
freeze_manifest_sha256
reasoning_request_sha256
response_format_sha256
scheduler_sha256
```

Recompute on resume and abort on mismatch.

---

# 15. P1 — Keep legacy `swap_consistent` but fix its semantics documentation

`src/aggregate_pairs.py` still uses the preregistered legacy definition:

```text
same non-tie content winner
```

under the name:

```text
swap_consistent
```

Do not destroy historical reproducibility.

Instead:

1. Add explicit new fields:
   ```text
   position_consistent
   decisively_certified
   pair_outcome
   certified_wrong
   ```
2. Keep `swap_consistent` only as a backward-compatible legacy field.
3. Add a comment/docstring:
   ```text
   Legacy preregistered metric: stable ties are excluded.
   Do not use this field as standard position consistency in paper-facing work.
   ```
4. Update README terminology.

All new paper/NIM analysis should use the explicit new names.

---

# 16. P1 — Add NIM result analysis

Create:

```text
src/nim_analysis.py
```

Reuse pair classification, exact CI, and packet-bootstrap helpers from `src/paper_analysis.py`.

Do not fork a second inconsistent implementation.

## 16.1 Completed-run integrity gate

Verify:

```text
240 intended conditions
120 complete AB/BA pairs
40 packets
40 pairs per early/middle/late cell
no duplicate condition IDs

one configured model
one returned model family
one frozen inference config

same prompt hash
same subset hash
same reasoning config
same output-format strategy

no truncation
reasoning present as required
no unexpected model drift
```

Predeclare paper-inclusion parse threshold:

```text
overall parse success >= 98%
and
each position parse success >= 95%
```

If below threshold:

```text
keep as exploratory/debugging
do not present as clean replication
```

Do not silently drop malformed outputs.

## 16.2 Per-position metrics

For each 16K position:

```text
n_calls
n_parse_valid_calls
n_pairs
n_parse_valid_pairs

raw_accuracy
packet-bootstrap 95% CI

position_consistency
packet-bootstrap 95% CI

certification_coverage
packet-bootstrap 95% CI

n_certified_correct
n_certified_wrong
selective_risk
exact Clopper–Pearson CI

stable_tie_fraction
mixed_tie_fraction
tie_involved_fraction
order_disagreement_fraction
```

## 16.3 Primary NIM contrasts

Pool early+late as edge.

Compute:

```text
raw_accuracy_edge_minus_middle
position_consistency_edge_minus_middle
certification_coverage_edge_minus_middle
```

Use 10,000 packet-cluster bootstrap draws.

Do not compute a length interaction for NIM because this robustness experiment is 16K-only.

## 16.4 Paired NIM-vs-Gemini analysis

Both models judge the same packet IDs.

For every bootstrap draw:

1. sample packet IDs once with replacement;
2. apply the same sampled packet multiset to Gemini and NIM;
3. compute each model's edge-minus-middle contrast.

Report exploratory:

```text
NIM_accuracy_middle_penalty
    - Gemini_accuracy_middle_penalty

NIM_coverage_middle_penalty
    - Gemini_coverage_middle_penalty

NIM_position_consistency_middle_penalty
    - Gemini_position_consistency_middle_penalty
```

Never pool raw model judgments.

## 16.5 Outputs

Generate:

```text
results/nim/<model>/paper/audit.json
results/nim/<model>/paper/pair_outcomes.csv
results/nim/<model>/paper/pair_outcomes.json
results/nim/<model>/paper/position_summary.csv
results/nim/<model>/paper/contrasts.json
results/nim/<model>/paper/gemini_paired_comparison.json
results/nim/<model>/paper/main_table.csv
results/nim/<model>/paper/main_table.md
results/nim/<model>/paper/figures/...
```

---

# 17. P1 — Strict model-selection workflow

Never implement request-level model fallback.

Correct workflow:

```text
GLM preflight ELIGIBLE
    -> freeze GLM
    -> run complete 240-condition GLM experiment

GLM preflight not ELIGIBLE
    -> do not start GLM main
    -> preflight Nemotron

Nemotron ELIGIBLE
    -> freeze Nemotron
    -> run complete 240-condition Nemotron experiment
```

If GLM main has already started:

```text
transient service issue -> checkpoint and resume GLM later
```

If GLM must be permanently abandoned:

```text
archive partial GLM run as incomplete
start a new complete Nemotron run
```

Never fill missing GLM rows with Nemotron.

---

# 18. P1 — Model-specific frozen configurations

## 18.1 GLM-5.2 preferred

Model:

```text
z-ai/glm-5.2
```

Current hosted availability must be empirically checked.

Baseline availability call should not include unverified reasoning controls.

If GLM passes explicit high-thinking verification, freeze:

```text
temperature = 0.0
seed = 20260828
max_tokens = 16384
stream = true
```

plus the exact verified reasoning request discovered by preflight.

If high/max reasoning cannot be verified:

```text
GLM is ineligible
```

## 18.2 Nemotron 3 Super fallback

Model:

```text
nvidia/nemotron-3-super-120b-a12b
```

Freeze:

```text
temperature = 1.0
top_p = 0.95

reasoning_effort = "high"
reasoning_budget = 16384

max_tokens = 32768
seed = 20260828
stream = true
```

Use the documented model-specific recommendation rather than forcing Gemini's sampling settings onto Nemotron.

---

# 19. P1 — Add dedicated NIM tests

Add:

```text
tests/test_nvidia_nim_provider.py
tests/test_nim_preflight.py
tests/test_nim_subset.py
tests/test_nim_runner.py
tests/test_nim_analysis.py
```

All unit tests must use mocks and make zero live requests.

## 19.1 Provider tests

At minimum:

```text
test_nim_requires_api_key

test_nim_stream_accumulates_reasoning_and_final_content
test_nim_stream_captures_returned_model
test_nim_stream_captures_generation_id
test_nim_stream_captures_usage

test_nim_sync_captures_model_and_usage

test_nim_429_is_retriable
test_nim_429_preserves_retry_after
test_nim_500_is_retriable
test_nim_422_is_nonretriable
test_nim_401_is_nonretriable

test_glm_baseline_does_not_inject_unverified_reasoning
test_nemotron_high_reasoning_payload
test_response_format_payload_when_enabled
```

Use `httpx.MockTransport` or equivalent.

## 19.2 Preflight tests

Test statuses:

```text
ELIGIBLE
UNAVAILABLE
AUTH_FAILED
REQUEST_SCHEMA_FAILED
THINKING_UNVERIFIED
OUTPUT_CONTRACT_FAILED
FULL_LENGTH_FAILED
RATE_LIMIT_UNUSABLE
LOCAL_INTEGRITY_FAILED
```

Also:

```text
test_glm_requires_on_off_reasoning_verification
test_nemotron_reasoning_on_off
test_preflight_uses_real_six_condition_smoke
test_preflight_uses_real_full_length_prompts
test_preflight_records_returned_model
test_preflight_records_selected_reasoning_request
test_preflight_counts_all_http_attempts
```

## 19.3 Runner tests

Add:

```text
test_nim_main_requires_eligible_preflight
test_nim_main_requires_matching_preflight_hash
test_nim_main_requires_freeze_manifest

test_resume_rejects_changed_temperature
test_resume_rejects_changed_top_p
test_resume_rejects_changed_reasoning
test_resume_rejects_changed_prompt
test_resume_rejects_changed_scheduler
test_resume_rejects_changed_output_format

test_provider_error_is_not_retried
test_retriable_error_honors_retry_after
test_backoff_uses_base_backoff_seconds
test_attempt_cap_includes_retries
test_no_mid_run_model_failover
```

Mock `time.sleep`.

## 19.4 Subset tests

Add:

```text
test_subset_requires_parent_gemini_hash
test_subset_contains_only_16k
test_subset_has_240_conditions
test_subset_has_40_packets
test_subset_has_two_orders_per_packet_position
test_subset_preserves_candidate_text_exactly
test_subset_is_deterministic
```

## 19.5 NIM analysis tests

Add:

```text
test_nim_edge_middle_accuracy_contrast
test_nim_edge_middle_position_consistency_contrast
test_nim_edge_middle_coverage_contrast

test_nim_packet_bootstrap_reproducible
test_paired_bootstrap_uses_same_packet_sample_for_both_models

test_nim_integrity_rejects_model_drift
test_nim_integrity_rejects_truncation
test_nim_integrity_rejects_config_drift
```

---

# 20. P1 — Strengthen Gemini golden-result tests

Current paper-analysis tests exercise formulas but do not pin the actual committed scientific results.

Add:

```text
tests/test_committed_paper_results.py
```

Assert:

```text
raw_accuracy interaction == 0.1875
certification_coverage interaction == 0.1500
position_consistency interaction == -0.05
```

For 16K-middle assert:

```text
n_pairs == 40
raw_accuracy == 0.3125
position_consistency == 0.60
certification_coverage == 0.15

stable_tie == 18
mixed_tie == 14
order_disagreement == 2

n_certified == 6
n_certified_wrong == 0
```

Also add exact-CI regression cases:

```text
0/6
0/11
1/17
1/24
1/29
0/0 -> None
```

Strengthen the packet-bootstrap clustering test; checking only that an output key exists does not prove packet-level resampling.

Use a fixture whose possible bootstrap support differs under call-level vs packet-level sampling.

Fix the tokenizer-position test fixture so condition IDs and response IDs are explicitly correct.

---

# 21. P1 — Update documentation

## 21.1 README

Add:

```text
Gemini paper reanalysis
NVIDIA NIM robustness workflow
```

Clarify terminology:

```text
legacy swap_consistent != standard position_consistency

paper coverage =
    decisive certification coverage
```

State that Cohere/Groq are not paper-facing robustness evidence.

## 21.2 AGENTS.md

Current agent instructions remain heavily centered on the old Cohere workflow.

Update so future Codex runs know:

```text
Gemini is primary paper evidence
NIM is post-hoc robustness only
GLM requires successful reasoning-control preflight
Nemotron is fallback selected before main
no request-level model fallback
no NIM main without ELIGIBLE preflight + freeze
```

## 21.3 `docs/paper_claims.md`

Add actual Gemini mechanism:

```text
16K-middle:
raw accuracy = 31.25%
position consistency = 60%
decisive certification coverage = 15%
stable ties = 45%
mixed ties = 35%
pure order disagreement = 5%
tie-involved = 80%
```

Add:

```text
The +15 pp decisive-coverage interaction has a current 95% packet-bootstrap
interval of approximately [-5 pp, +35 pp], so treat it as directional rather
than statistically conclusive.
```

---

# 22. P1 — Update `.gitignore`

Add:

```gitignore
results/nim/*/raw/
results/nim/*/api_attempt_counter.json
```

Do not ignore:

```text
preflight.json
NIM freeze manifests
paper aggregate outputs
paper figures
```

Avoid accidentally committing full long prompts/responses.

---

# 23. P2 — Mark legacy analysis clearly rather than deleting it

Keep:

```text
src/analyze.py
src/derive_coverage_table.py
historical GO/KILL outputs
```

because they document the preregistered analysis.

Add documentation that their old `swap_consistency` terminology excludes stable ties and should not be interpreted as standard position consistency.

Use:

```text
src/paper_analysis.py
src/nim_analysis.py
```

for all paper-facing results going forward.

---

# 24. Codex execution order

## Stage A — zero live API calls

```text
1. Run current test suite.

2. Reproduce committed Gemini paper artifacts locally.

3. Add Gemini golden-result tests.

4. Implement prepare_nim_subset.py.

5. Generate/validate the exact 240-condition 16K subset.

6. Repair GLM and Nemotron configs.

7. Add httpx dependency.

8. Fix NIM provider metadata and Retry-After propagation.

9. Implement/verify response-format strategy.

10. Rewrite NIM preflight.

11. Fix NVIDIA_API_KEY mapping.

12. Fix ProviderError retry semantics.

13. Implement configured backoff.

14. Add shared attempt counter.

15. Add strong resume fingerprints.

16. Add hard preflight/freeze gate.

17. Implement freeze_nim.py.

18. Implement nim_analysis.py.

19. Add all NIM tests.

20. Update README, AGENTS.md, paper_claims.md, .gitignore.

21. Run full pytest suite.

22. Confirm tests made zero network calls.
```

Do not proceed if tests fail.

## Stage B — GLM preflight only

Requires explicit operator authorization.

```text
23. Run local-integrity checks.

24. Make baseline GLM availability request without unverified reasoning fields.

25. Probe GLM reasoning ON/OFF controls.

26. Probe response_format.

27. Run six actual frozen 16K position×order conditions.

28. Run additional real long-prompt capacity probes.

29. Write detailed preflight.json.
```

If:

```text
status == ELIGIBLE
```

select GLM.

Otherwise do not start GLM main.

## Stage C — Nemotron fallback preflight

If GLM is not eligible:

```text
30. Run Nemotron high-vs-none reasoning verification.

31. Verify output-format strategy.

32. Run six real frozen smoke conditions.

33. Run additional long-prompt probes.

34. Require status == ELIGIBLE.
```

If Nemotron is also ineligible, stop.

Do not substitute a third model automatically.

## Stage D — freeze selected model

```text
35. Ensure clean Git worktree.

36. Freeze exact selected-model experiment.

37. Freeze preflight hash.

38. Freeze prompt/subset/reasoning/output-format/scheduler hashes.

39. Create a new run ID.
```

## Stage E — main robustness run

Run:

```text
40 packets × 3 positions × 2 orders = 240 calls
```

Sequentially.

Frozen scheduler:

```text
max_concurrency = 1
minimum_interval_seconds = 60
max_transport_retries = 2
base_backoff_seconds = 120
hard_attempt_cap = 300
```

Checkpoint every response.

Never switch models within the run.

## Stage F — analysis

```text
40. Parse the completed run.

41. Run NIM integrity audit.

42. If inclusion gate fails:
        retain as exploratory/debugging only

43. Generate per-position metrics.

44. Generate packet-bootstrap edge-minus-middle contrasts.

45. Generate paired NIM-vs-Gemini bootstrap comparison.

46. Generate paper-facing NIM table/figure.

47. Update paper claims using the predeclared interpretation matrix.
```

---

# 25. Predeclared interpretation matrix

## A — stronger NIM model reproduces a middle deficit

Write:

> The middle-position effect generalizes beyond the economical Gemini judge to an independent stronger reasoning judge.

## B — NIM improves overall accuracy but middle decisive coverage remains lower

Write:

> Greater judge capability improves baseline discrimination but does not fully remove positional sensitivity in decisive certification.

## C — NIM eliminates the middle effect

Write:

> The observed failure is capability/configuration dependent rather than universal.

Do not hide the non-replication.

## D — NIM is near-perfect in every position

Narrow the paper:

> The Gemini result characterizes an economical judge configuration rather than a universal limitation of swap validation.

## E — NIM cannot complete a clean hosted run

Do not report partial NIM results as a replication.

Gemini remains the primary experiment.

---

# 26. Acceptance checklist before authorizing 240 main calls

Every item must be true:

```text
[ ] Gemini paper artifacts reproduce.
[ ] Full test suite passes.
[ ] httpx is declared.
[ ] Exact 16K subset exists.
[ ] Parent Gemini condition hash matches freeze.
[ ] Subset has exactly 240 conditions / 40 packets / 120 pairs.
[ ] NIM configs reference real paths.
[ ] PRMBench validator passes the NIM subset.
[ ] NIM provider captures returned model identity.
[ ] NIM provider propagates Retry-After.
[ ] ProviderError is not retried.
[ ] base_backoff_seconds is actually used.
[ ] Shared attempt counter includes preflight.
[ ] Preflight uses actual frozen 16K conditions.
[ ] Reasoning ON/OFF has been verified.
[ ] Output-schema strategy is verified and frozen.
[ ] 6/6 real smoke conditions parse successfully.
[ ] Representative full-length prompts succeed.
[ ] Preflight status == ELIGIBLE.
[ ] NIM-specific freeze manifest exists.
[ ] Freeze hashes match current inputs/config.
[ ] Resume rejects changed inference parameters.
[ ] No request-level GLM↔Nemotron fallback exists.
[ ] Raw NIM runs are gitignored.
```

Only then run the 240-condition robustness experiment.

---

# 27. Short directive for Codex

```text
Do not redo the working Gemini paper analysis.

The Gemini side is largely implemented and currently shows:
  16K-middle raw accuracy = 31.25%
  standard position consistency = 60%
  decisive certification coverage = 15%
  stable ties = 45%
  mixed ties = 35%
  pure clean↔flawed disagreement = 5%
  tie-involved = 80%

The main remaining work is making the NVIDIA NIM robustness experiment
scientifically and operationally safe.

Before any NIM main call:
- repair broken condition/prompt/freeze paths;
- derive an exact 240-row 16K subset from the frozen Gemini conditions;
- add a NIM-specific freeze;
- declare httpx;
- capture returned model + request ID + usage + rate-limit metadata;
- propagate Retry-After;
- verify provider-side response schema or use a hashed formatting-only suffix;
- rewrite preflight using real frozen 16K prompts;
- verify reasoning ON vs OFF;
- hard-gate main on ELIGIBLE preflight + matching freeze;
- fix NVIDIA_API_KEY mapping;
- never retry non-retriable ProviderError;
- actually use the 120s exponential backoff;
- count preflight + main attempts under one 300-attempt cap;
- reject any resume after inference-config drift;
- add NIM-specific tests and NIM analysis;
- never switch GLM/Nemotron within a run.

GLM is preferred only if explicit high-thinking control is verifiable.
Otherwise select Nemotron 3 Super before the main run.

Nemotron frozen fallback:
  model = nvidia/nemotron-3-super-120b-a12b
  temperature = 1.0
  top_p = 0.95
  reasoning_effort = high
  reasoning_budget = 16384
  max_tokens = 32768
  seed = 20260828
  stream = true

Scientific guardrail:
  Gemini accuracy interaction = +18.75 pp,
      bootstrap 95% CI [6.25, 30.625] pp
  Gemini decisive-coverage interaction = +15 pp,
      bootstrap 95% CI [-5, 35] pp

Therefore do not call the coverage interaction statistically conclusive and
do not describe the Gemini result as primarily AB/BA winner flipping.
```
