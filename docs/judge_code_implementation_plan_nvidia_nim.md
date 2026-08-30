# JUDGe Experiment / Analysis Implementation Plan — Revised

## Purpose

This is the implementation brief for Codex.

The goal is to turn the existing experiment into the strongest defensible **2-page JUDGe workshop submission** with minimal additional experimentation.

The existing frozen Gemini experiment remains the primary evidence. The preregistered hypothesis was not supported; preserve that negative result. The new contribution is an analysis of how a downstream AB/BA reliability gate behaves under long-context internal-position stress.

The implementation must:

1. exhaust the existing Gemini data before spending model calls;
2. distinguish standard position consistency from decisive certification coverage;
3. fix uncertainty estimates for low-coverage conditional error rates;
4. decompose rejected AB/BA pairs into ties versus genuine order disagreement;
5. center the length × internal-position interaction rather than the overall long-context drop;
6. quarantine Cohere from paper-facing evidence;
7. optionally add one stronger independent reasoning-model replication through NVIDIA NIM;
8. prevent hosted-endpoint availability, rate-limit, reasoning-control, or model-switching artifacts from becoming new reviewer objections.

Do not silently change the original experiment, overwrite frozen artifacts, or reinterpret the preregistered `KILL` as a confirmatory success.

---

# 0. Scientific scope and non-negotiable constraints

## 0.1 Primary evidence

The existing frozen Gemini run is the **primary experiment**:

- model: `gemini-3.5-flash-lite`;
- 40 packets;
- two construction-length bands: 4K and 16K;
- three internal positions: early, middle, late;
- two candidate orders: clean-first and flawed-first;
- 480 calls total;
- 240 AB/BA pairs.

The original hypothesis predicted a false-certificate regime: middle-position errors would become harder to detect while AB/BA consistency remained high and consistent error rate increased.

That hypothesis was not supported. Preserve this as a negative preregistered result.

The paper's exploratory interpretation is:

> Under long-context internal-position stress, AB/BA validation may not become confidently wrong; instead, the rate at which it produces a usable decisive certificate can collapse.

## 0.2 Cohere

Keep all Cohere source code, tests, and historical results.

Do **not** use Cohere in paper-facing:

- tables;
- figures;
- abstract claims;
- cross-model generalization claims;
- wording such as "two independent proprietary judge families."

Treat Cohere as historical/exploratory only.

## 0.3 New model calls

Do not make any new model calls until the existing Gemini reanalysis is complete.

Any new NVIDIA NIM run is:

- a robustness replication;
- exploratory/post hoc relative to the original preregistration;
- analyzed separately by model;
- never pooled with Gemini;
- never used to rewrite the original hypothesis.

## 0.4 Frozen inputs

All replication runs must reuse the exact existing candidate texts for the selected conditions.

Do not rebuild PRMBench packets for a new model.

For every new run, freeze and record:

- candidate-condition file hash;
- prompt-template hash;
- Git commit;
- model slug;
- returned model identifier if available;
- provider endpoint;
- reasoning configuration;
- structured-output schema;
- rate-limit configuration.

## 0.5 Output directories

Do not overwrite the original result artifacts.

Use:

```text
results/gemini/paper/
results/nim/glm_5_2/
results/nim/nemotron_3_super/
```

---

# 1. JUDGe / literature framing guardrails

These are implementation-facing paper constraints. Generate a small `paper_claims.md` or equivalent so downstream drafting follows them.

## 1.1 What is already known

Do not claim novelty for:

- candidate-order bias in LLM judges;
- AB/BA swapping as a diagnostic/mitigation;
- "lost in the middle";
- long-context LLM-judge unreliability;
- within-document error/perturbation position affecting judge behavior;
- the generic fact that filtering inconsistent judgments reduces sample size.

The closest prior-work threat is work such as **Semantic Needles**, which already manipulates the internal location of a semantic alteration in document comparison.

The novelty boundary is the downstream systems question:

> How does internal evidence-position stress propagate through an AB/BA validation gate, and how often does that gate still yield a decisive usable judgment?

## 1.2 Avoid the quality-gap dilution trap

The 16K candidates contain more correct filler around one defective section, so the candidates become globally more similar as length grows.

Therefore:

- do not make the overall 4K → 16K accuracy/coverage decline the central causal result;
- emphasize the **within-length early/middle/late contrast**;
- make the **length × position interaction** the main statistic.

Within a fixed length, early/middle/late conditions have the same underlying candidate-quality gap and differ only in target location.

## 1.3 Avoid over-attributing rejection to "position bias"

One AB/BA observation per order does not identify the mechanism of every disagreement.

Even at temperature zero, proprietary/free hosted inference need not be perfectly deterministic.

Therefore:

- call the operational quantity `certification_coverage`;
- call clean/flawed reversals `order_disagreement`;
- do not claim every rejected pair demonstrates candidate-order bias;
- reserve `position_consistency` for the defined same-outcome-after-swap metric.

## 1.4 Selective-prediction framing

Use:

```text
certification_coverage = P(decisive swap certificate)
selective_risk         = P(wrong | decisive swap certificate)
```

Do not foreground:

```text
post_filter_accuracy - raw_accuracy
```

That is selected-subset accuracy minus population accuracy, not an intervention effect.

---

# 2. Phase A — Audit and freeze the existing Gemini evidence

Locate the completed Gemini main run under:

```text
results/gemini/raw/
```

Verify:

- exactly 480 completed conditions;
- exactly 240 complete AB/BA pairs;
- exactly 40 unique packets;
- every length × position cell contains 40 complete pairs;
- 100% parse success;
- exactly one returned Gemini model identifier;
- no duplicate condition IDs;
- no missing candidate-order member in a pair;
- condition hash matches the Gemini freeze manifest;
- prompt hash matches the Gemini freeze manifest;
- response records contain the A/B/tie information needed below;
- API prompt-token usage is present for most/all successful calls;
- no response is truncated or marked incomplete.

Generate:

```text
results/gemini/paper/audit.json
```

Include at least:

```text
run_id
git_sha
condition_hash
prompt_hash
configured_model
returned_model_ids
n_packets
n_calls
n_pairs
parse_rate
duplicate_count
incomplete_pair_count
missing_prompt_token_metadata
provider_error_count
```

### Acceptance rule

If this audit fails, stop paper analysis and diagnose the historical Gemini run.

Do not silently repair or rerun the old experiment.

---

# 3. Phase B — Separate position consistency from certification coverage

The current implementation conflates these ideas.

Implement both explicitly.

## 3.1 Standard position consistency

At content-outcome level, define an AB/BA pair as position-consistent when both orders produce the same semantic outcome.

Outcomes are:

```text
clean
flawed
tie
```

Therefore all of the following are position-consistent:

```text
clean / clean
flawed / flawed
tie / tie
```

Define:

```text
position_consistency =
    n_same_content_outcome_after_swap / n_parse_valid_complete_pairs
```

A stable tie counts as positionally consistent.

## 3.2 Decisive certification coverage

A usable certificate requires the same **non-tie** content winner in both orders.

Define:

```text
certified_correct = clean / clean
certified_wrong   = flawed / flawed

certification_coverage =
    (n_certified_correct + n_certified_wrong)
    / n_parse_valid_complete_pairs
```

Stable `tie/tie` does **not** count as decisive certification coverage.

## 3.3 Selective risk / CER

Define:

```text
selective_risk =
    n_certified_wrong
    / (n_certified_correct + n_certified_wrong)
```

This is identical to CER under this design.

When no decisive certificates exist, return `None`, never `0`.

## 3.4 Required per-cell metrics

For each:

```text
4K × early
4K × middle
4K × late
16K × early
16K × middle
16K × late
```

produce:

```text
n_pairs
n_calls

raw_accuracy
raw_accuracy_ci

position_consistency
position_consistency_ci

certification_coverage
certification_coverage_ci

n_certified
n_certified_correct
n_certified_wrong

selective_risk
selective_risk_exact_ci
```

The old Gemini point estimates for the previous non-tie consistency metric should still reproduce:

```text
16K early:  raw accuracy=.4500, decisive coverage=.275 = 11/40
16K middle: raw accuracy=.3125, decisive coverage=.150 =  6/40
16K late:   raw accuracy=.5375, decisive coverage=.275 = 11/40

4K early:   raw accuracy=.6375, decisive coverage=.425 = 17/40
4K middle:  raw accuracy=.7250, decisive coverage=.600 = 24/40
4K late:    raw accuracy=.8000, decisive coverage=.725 = 29/40
```

Do not assume the newly computed standard position-consistency values equal these.

---

# 4. Phase C — Decompose every AB/BA pair

Every parse-valid complete pair must map to exactly one mutually exclusive outcome:

```text
certified_correct
    clean / clean

certified_wrong
    flawed / flawed

stable_tie
    tie / tie

order_disagreement
    clean / flawed
    flawed / clean

mixed_tie
    exactly one order is tie
```

These five categories must sum to 40 in every length × position cell.

Optionally retain directional mixed-tie subtypes:

```text
tie / clean
clean / tie
tie / flawed
flawed / tie
```

Generate:

```text
results/gemini/paper/pair_outcomes.csv
results/gemini/paper/pair_outcomes.json
```

Include counts and fractions.

Also compute:

```text
rejection_rate =
    1 - certification_coverage

tie_involved_fraction =
    (stable_tie + mixed_tie) / n_pairs

pure_order_disagreement_fraction =
    order_disagreement / n_pairs

tie_share_of_noncertification =
    (stable_tie + mixed_tie)
    / (stable_tie + mixed_tie + order_disagreement)
```

### Paper interpretation rule

If tie-involved outcomes dominate:

> Long-context stress primarily induces non-decision/abstention under the validation protocol.

If clean/flawed reversals dominate:

> Long-context stress primarily induces order-sensitive instability.

If both matter:

> Certification loss is a mixture of explicit non-decision and order-sensitive disagreement.

Do not decide the wording before inspecting the data.

---

# 5. Phase D — Fix uncertainty for selective risk

The existing packet bootstrap is appropriate for repeated-measures contrasts, but not as the only uncertainty representation for a conditional error rate when zero errors were observed.

For selective risk/CER report an **exact two-sided Clopper-Pearson 95% interval** based on:

```text
k = n_certified_wrong
n = n_certified
```

Use `scipy.stats.beta.ppf` or an equivalent tested implementation.

Expected regression check:

```text
Gemini 16K middle:
k = 0
n = 6
point estimate = 0
two-sided exact 95% upper bound ≈ 0.459
```

Paper-facing representation:

```text
0/6 certified errors; selective risk 0%,
exact 95% CI approximately [0%, 45.9%]
```

For `n=0`:

```text
selective_risk = None
selective_risk_ci = None
```

Do not write `100% reliable` or imply that zero observed errors among six certificates establishes near-perfect reliability.

---

# 6. Phase E — Primary length × internal-position interactions

The main causal-looking statistic should be a difference-in-differences style interaction.

## 6.1 Accuracy interaction

For each length:

```text
edge_accuracy(L) =
    pooled accuracy across early and late calls at length L

middle_penalty_accuracy(L) =
    edge_accuracy(L) - middle_accuracy(L)
```

Then:

```text
accuracy_interaction =
    middle_penalty_accuracy(16K)
    - middle_penalty_accuracy(4K)
```

Expected point estimate:

```text
4K edge accuracy    = .71875
4K middle accuracy  = .72500
4K middle penalty   = -.00625

16K edge accuracy   = .49375
16K middle accuracy = .31250
16K middle penalty  = .18125

accuracy interaction = .18750
                     = +18.75 percentage points
```

## 6.2 Certification-coverage interaction

Analogously:

```text
edge_coverage(L) =
    pooled decisive-certification coverage across early and late

middle_penalty_coverage(L) =
    edge_coverage(L) - middle_coverage(L)

coverage_interaction =
    middle_penalty_coverage(16K)
    - middle_penalty_coverage(4K)
```

Expected point estimate under the current decisive-coverage definition:

```text
4K edge coverage    = .575
4K middle coverage  = .600
4K middle penalty   = -.025

16K edge coverage   = .275
16K middle coverage = .150
16K middle penalty  = .125

coverage interaction = .150
                     = +15.0 percentage points
```

## 6.3 Position-consistency interaction

Now that position consistency includes `tie/tie`, compute the analogous interaction for that metric too.

Do not assume it matches the decisive-coverage interaction.

This distinction may become one of the most interesting results:

```text
position consistency may remain relatively high
while decisive certification coverage collapses
```

if stable ties increase.

## 6.4 Bootstrap

Use 10,000 packet-level bootstrap resamples.

For each draw:

1. sample 40 packet IDs with replacement;
2. retain all observations for each selected packet;
3. recompute all cells;
4. recompute edge and middle metrics;
5. recompute the interaction.

Do not bootstrap individual calls independently.

Report percentile 95% CIs and valid draw count.

Generate:

```text
results/gemini/paper/interactions.json
```

---

# 7. Phase F — Candidate-order robustness

Using existing Gemini observations only, compute raw accuracy separately for:

```text
clean_first
flawed_first
```

by length × internal position.

For each order compute:

```text
(edge - middle at 16K)
-
(edge - middle at 4K)
```

Bootstrap at packet level.

This is secondary evidence.

The purpose is to show that the long-context middle effect is not solely driven by one presentation order.

Do not overinterpret this as a mechanistic decomposition of order bias.

---

# 8. Phase G — Native token-length sanity check

The construction uses the Cohere tokenizer even for Gemini.

Do not call that "4K Gemini tokens" and "16K Gemini tokens."

Use the Gemini API usage metadata already stored in the historical responses.

For each construction length report:

```text
n
mean
median
min
max
p05
p95
```

for total Gemini prompt tokens.

Also stratify by internal position and check that early/middle/late relocation does not substantially alter total prompt length.

Generate:

```text
results/gemini/paper/native_token_lengths.csv
```

Also add a tokenizer-independent target-position check using one or both of:

```text
character-offset midpoint fraction
whitespace-token midpoint fraction
```

Confirm that early < middle < late remains cleanly separated.

If native/tokenizer-independent location separation fails, stop and investigate.

---

# 9. Phase H — Difficulty / selection diagnostic

For each pair compute whether it becomes decisively certified.

Then compare:

```text
raw call accuracy among ultimately certified pairs
raw call accuracy among rejected pairs
```

This is descriptive only.

The goal is to verify the selective-prediction interpretation:

> the gate retains a subset of easier/more stable comparisons rather than improving the same comparisons.

Do not describe this as a causal performance improvement.

Keep this analysis out of the two-page paper if space is tight, but retain it as reviewer-defense evidence.

---

# 10. Phase I — Paper-facing Gemini outputs

Generate:

```text
results/gemini/paper/main_table.csv
results/gemini/paper/main_table.md
```

Recommended columns:

```text
Length
Position
Raw accuracy [95% CI]
Position consistency n/N [95% CI]
Decisive certification coverage n/N [95% CI]
Certified errors k/n [exact 95% CI]
Stable tie %
Mixed tie %
Order disagreement %
```

Do not include a `post-filter accuracy = 100%` headline column.

Prefer counts such as:

```text
0/6 certified errors
```

over visually impressive but statistically fragile percentages.

## Figure

Create one compact paper figure.

Recommended structure:

```text
Panel A — Raw accuracy by internal position
4K and 16K

Panel B — Decisive certification coverage by internal position
4K and 16K
```

Use packet-bootstrap 95% intervals.

If stable/mixed ties dominate noncertification, additionally generate an internal stacked outcome figure:

```text
certified correct
certified wrong
stable tie
mixed tie
order disagreement
```

The stacked figure only enters the paper if it communicates the mechanism more efficiently than the main table.

---

# 11. Phase J — Tests for the reanalysis

Add at minimum:

```text
test_pair_outcomes_partition_all_pairs

test_certified_correct
test_certified_wrong
test_stable_tie
test_mixed_tie_one_side
test_order_disagreement_clean_flawed
test_order_disagreement_flawed_clean

test_position_consistency_counts_stable_tie
test_certification_coverage_excludes_stable_tie

test_selective_risk_exact_ci_zero_of_six
test_selective_risk_none_at_zero_coverage

test_accuracy_interaction_formula
test_decisive_coverage_interaction_formula
test_position_consistency_interaction_formula

test_packet_bootstrap_preserves_cluster
test_bootstrap_interaction_reproducible_with_seed

test_native_token_summary_ignores_missing_values
test_tokenizer_independent_position_ordering
```

Regression checks:

```text
accuracy_interaction == 0.1875

decisive_coverage_interaction == 0.1500
```

within floating-point tolerance.

Do not hard-code a position-consistency interaction until the new tie-aware metric is computed from raw data.

---

# 12. Phase K — Cohere quarantine

Keep:

```text
results/cohere/
Cohere provider implementation
Cohere tests
historical combined outputs
```

Exclude from paper:

```text
Cohere result tables
Cohere figures
Cohere cross-model claims
```

Add a repository note:

```text
Cohere results are retained as exploratory historical runs and are not
used in the JUDGe submission analysis.
```

Do not spend additional experimental budget debugging Cohere for this submission.

---

# 13. Phase L — NVIDIA NIM robustness provider

The stronger-model replication should use NVIDIA's OpenAI-compatible NIM endpoint:

```text
https://integrate.api.nvidia.com/v1/chat/completions
```

Environment variable:

```text
NVIDIA_API_KEY
```

Preferred model:

```text
z-ai/glm-5.2
```

Fallback model:

```text
nvidia/nemotron-3-super-120b-a12b
```

The fallback is a **model-selection fallback before a main run**, not an automatic per-request failover. Never mix model identities within one experimental run.

## 13.1 Research-status snapshot to encode in the implementation

This plan was checked against NVIDIA's current documentation on 2026-08-30.

### GLM-5.2

Official NVIDIA documentation currently supports the following claims:

```text
model slug: z-ai/glm-5.2
endpoint: POST /v1/chat/completions
input context: 1,000,000 tokens
model size: 753B parameters
reasoning-capable
model card advertises multiple thinking-effort levels
model card advertises reasoning traces and structured output
NIM inference API exposes:
    temperature
    top_p
    max_tokens up to 32768
    seed
    stream
```

Important current inconsistency:

```text
- the NVIDIA API reference still documents the hosted GLM-5.2 endpoint;
- NVIDIA's model catalogs have recently labeled GLM-5.2 a free endpoint;
- the current GLM-5.2 build page also reports "Free Endpoint: Deprecated."
```

Therefore **do not assume hosted GLM-5.2 trial access exists**.

The code must perform a live availability preflight. A successful preflight is part of the experiment integrity record.

A second important caveat is that the current public GLM-5.2 inference schema does not document a `reasoning_effort`, `thinking`, or `chat_template_kwargs` field even though the model card/playground advertises reasoning controls.

Therefore **do not silently assume a high-thinking request field works**. The GLM run is paper-eligible only if thinking can be empirically verified before the main run.

### Nemotron 3 Super

NVIDIA currently documents:

```text
model slug: nvidia/nemotron-3-super-120b-a12b
context: up to 1,000,000 tokens
120B total / 12B active parameters
free hosted prototype endpoint available
reasoning_effort:
    none
    low
    high
default reasoning_effort:
    high
reasoning_budget:
    -1 to 32768
default reasoning_budget:
    16384
max_tokens:
    up to 32768
seed:
    supported, best-effort deterministic
```

NVIDIA's model card explicitly recommends:

```text
temperature = 1.0
top_p = 0.95
```

for this model across tasks/backends.

Nemotron is therefore the **clean fallback** if GLM hosted access or reasoning-control verification fails.

### Trial-rate limits

Do not encode a fixed NVIDIA free-tier RPM into the scientific configuration.

NVIDIA staff state that trial/free rate limits can depend on:

```text
model
use case
current overall traffic
```

Recent users often observe values around 40 RPM, but this is not a guaranteed contractual limit and some users report persistent 429s at much lower request counts.

Consequently:

- run sequentially;
- inspect rate-limit headers;
- start conservatively;
- honor server reset / Retry-After information;
- checkpoint after every condition;
- treat sustained rate limiting as an availability issue, not an experimental outcome.

---

# 14. Phase M — Generic NVIDIA NIM provider implementation

Implement:

```text
src/providers/nvidia_nim_provider.py
```

and register it in:

```text
src/providers/factory.py
```

Do not build separate HTTP stacks for GLM and Nemotron. Use one NIM provider plus explicit model profiles.

Recommended profile abstraction:

```text
NIMModelProfile
    model_slug
    temperature
    top_p
    max_tokens
    seed
    stream
    reasoning_mode
    reasoning_effort
    reasoning_budget
    extra_body
    reasoning_must_be_verified
```

Create two explicit configs:

```text
config/pilot_prmbench_nim_glm52.yaml
config/pilot_prmbench_nim_nemotron3super.yaml
```

Do not implement a single config that silently changes model at runtime.

## 14.1 Direct HTTP rather than opaque SDK behavior

Prefer a small direct HTTP client (`httpx` is suitable) over relying on an SDK to decide which unknown/provider-specific fields to keep or strip.

Reason:

- GLM thinking control on NIM is not fully represented in the current public schema;
- we need exact control of the request JSON;
- we need raw HTTP status and headers for rate-limit diagnostics;
- we need to preserve/inspect `reasoning_content` if returned;
- unknown request fields must not disappear inside a client abstraction.

The provider should send:

```text
Authorization: Bearer $NVIDIA_API_KEY
Content-Type: application/json
Accept: text/event-stream   # for the experiment stream path
```

to the single NIM base URL.

## 14.2 Streaming

Use streaming for the NIM robustness runs.

Reasoning-capable NIMs can emit reasoning separately from final content. The provider should parse SSE and independently accumulate:

```text
reasoning_content
final_content
finish_reason
usage_if_present
```

Do not concatenate reasoning text into the final verdict text.

The final parser only receives `final_content`.

For stored artifacts, it is sufficient to persist:

```text
reasoning_present
reasoning_character_count
reasoning_sha256
reasoning_tokens_if_reported
```

Raw reasoning text may remain in uncommitted raw logs if useful, but it is not required for paper analysis.

## 14.3 Output contract

The original judge prompt already requires a JSON object with:

```json
{
  "winner": "A | B | tie",
  "brief_reason": "one sentence"
}
```

Preserve the same semantic contract.

Do **not** make paper validity depend on NVIDIA accepting a `response_format` parameter, because the current model-specific NIM inference schemas do not document it consistently.

Instead:

1. keep the exact judge instruction;
2. parse the final content strictly;
3. permit stripping a single surrounding Markdown code fence;
4. require:
   - JSON object;
   - `winner` in `A`, `B`, `tie`;
   - string `brief_reason`;
5. treat malformed final output as a parse failure.

Do not retry a condition simply because the model returned malformed JSON. That would condition selection on output quality.

Only transport/service failures may be retried.

## 14.4 Request/response logging

Every NIM response record should contain at least:

```text
run_id
condition_id
base_packet_id
length
internal_position
candidate_order

configured_provider = nvidia_nim
base_url
configured_model
returned_model_if_present

request_timestamp
response_timestamp
http_status
request_id_if_present

temperature
top_p
max_tokens
seed

reasoning_requested
reasoning_configuration
reasoning_present
reasoning_character_count
reasoning_sha256
reasoning_tokens_if_reported

prompt_tokens_if_reported
completion_tokens_if_reported
total_tokens_if_reported

finish_reason
retry_count

raw_final_response_text
parsed_winner_A_B_tie
parsed_content_winner_clean_flawed_tie
error_status
```

Also persist response headers relevant to rate limiting when present:

```text
retry-after
x-ratelimit-*
```

Do not commit API keys.

---

# 15. Phase N — GLM-5.2 availability and high-thinking verification gate

GLM-5.2 is the preferred robustness model, but **the main run must not start simply because a one-line chat succeeds**.

Create a dedicated command, for example:

```text
python -m src.preflight_nim \
  --config config/pilot_prmbench_nim_glm52.yaml
```

The preflight should return a machine-readable report:

```text
results/nim/glm_5_2/preflight.json
```

with one of:

```text
ELIGIBLE
UNAVAILABLE
THINKING_UNVERIFIED
RATE_LIMIT_UNUSABLE
OUTPUT_CONTRACT_FAILED
FULL_LENGTH_FAILED
```

## 15.1 Stage 1 — endpoint availability

Send a tiny deterministic prompt.

Pass only if:

- HTTP request succeeds;
- the returned model is GLM-5.2 or an unambiguously matching identifier;
- the endpoint does not report model removal/deprecation;
- the final response is nonempty.

If the model is unavailable, skip all further GLM probing and select Nemotron before any main conditions are run.

## 15.2 Stage 2 — probe the reasoning control

The public NVIDIA model card says GLM-5.2 has multiple thinking-effort levels, but the current public inference schema does not specify the exact request field.

Therefore treat reasoning controls as **probe candidates**, not assumptions.

Probe on a trivial reasoning prompt using direct HTTP.

Candidate forms may include, in a controlled compatibility probe:

```text
top-level:
    reasoning_effort = "high"

top-level:
    reasoning_effort = "max"

extra/provider body:
    chat_template_kwargs = {
        "enable_thinking": true,
        "clear_thinking": false
    }
```

If NVIDIA changes the current schema, Codex may use the newly documented official field instead.

Never invent a field without recording the exact accepted request.

For each probe capture:

```text
HTTP status
response headers
whether reasoning_content was emitted
reasoning length/tokens
final answer
```

## 15.3 Stage 3 — verify that thinking is controllable, not merely present

A paper-quality GLM run requires evidence that the selected request genuinely enables reasoning.

Where the endpoint supports an OFF control, run matched tiny prompts with:

```text
thinking OFF
thinking ON / high
```

The ON configuration should:

- be accepted;
- produce reasoning content or reported reasoning tokens;
- differ observably from OFF in reasoning metadata.

If the hosted route produces reasoning but the effort level cannot be distinguished, record that precisely.

### GLM eligibility rule

Use GLM as the paper robustness model only if:

```text
hosted endpoint available
AND requested reasoning configuration accepted
AND reasoning can be verified as active
AND intended high/max level is accepted or otherwise verifiably mapped
```

If high/max reasoning cannot be verified, **do not downgrade silently to an unknown/default GLM thinking configuration**.

Select Nemotron 3 Super instead.

This is intentionally conservative: the robustness run exists to answer the "underpowered judge" objection.

## 15.4 Stage 4 — output-contract smoke

Using the frozen judge prompt, execute all six conditions for one 16K smoke packet:

```text
early × two orders
middle × two orders
late × two orders
```

Require:

```text
6/6 successful service responses
6/6 valid final JSON verdicts
no truncation
one model identity
reasoning active on all six
```

Do not tune temperature or prompt based on whether the judge is correct.

## 15.5 Stage 5 — full-length capacity smoke

Run at least three representative real 16K prompts sequentially:

```text
one early
one middle
one late
```

The purpose is service capacity, not accuracy.

Require:

```text
no context-length errors
no output truncation
valid verdicts
reasoning active
no persistent 429 state
```

Only then freeze the GLM main configuration.

---

# 16. Phase O — Frozen GLM-5.2 replication configuration

If GLM passes every preflight gate, create/freeze a dedicated run configuration.

Model:

```text
z-ai/glm-5.2
```

Subset:

```text
16K only
40 packets
3 positions
2 candidate orders
240 conditions
120 AB/BA pairs
```

Reuse the exact existing frozen 16K candidate texts and prompt template.

## 16.1 Sampling

Default experimental choice:

```text
temperature = 0.0
seed = 20260828
top_p = omit / provider default
max_tokens = 16384
stream = true
```

Rationale:

- the task is comparative evaluation, not creative generation;
- the original Gemini run used temperature zero;
- NVIDIA's GLM endpoint permits a seed and temperature control;
- lower sampling variance reduces one source of AB/BA disagreement.

If `temperature=0.0` is rejected by the hosted GLM route, use the lowest accepted temperature established during smoke and freeze it **before** observing main-run accuracy.

Do not tune sampling to improve results.

## 16.2 Reasoning

Use exactly the verified high/max-thinking configuration discovered in preflight.

Freeze its raw request fragment in the run manifest, e.g.:

```text
reasoning_request_json
reasoning_probe_evidence
```

Do not merely write `"high"` in YAML if the actual NIM field differs.

## 16.3 Main-run integrity

Before first main request, freeze:

```text
Git SHA
config SHA
16K-condition subset SHA
prompt SHA
model slug
reasoning request JSON
temperature
top_p
max_tokens
seed
scheduler configuration
```

A resumed run must refuse to continue if any frozen field changes.

---

# 17. Phase P — Nemotron 3 Super fallback configuration

If GLM is:

```text
unavailable
thinking-unverifiable
unable to process the real long prompts reliably
or otherwise fails preflight
```

use a **new full run** with:

```text
nvidia/nemotron-3-super-120b-a12b
```

Do not append Nemotron conditions to a partial GLM run.

## 17.1 Frozen inference parameters

Use the NVIDIA-documented reasoning configuration:

```text
temperature = 1.0
top_p = 0.95

reasoning_effort = "high"
reasoning_budget = 16384

max_tokens = 32768
seed = 20260828
stream = true
```

Why `max_tokens=32768`:

- NVIDIA allows up to 32768;
- high reasoning has a 16384-token reasoning budget by default;
- a larger generation ceiling reduces risk that the reasoning phase leaves insufficient space for the final JSON verdict;
- the model may stop naturally well before the ceiling.

Freeze this value before the main run.

Do not reduce the reasoning budget to make the service cheaper/faster after seeing results.

NVIDIA's model card recommends `temperature=1.0` and `top_p=0.95`; follow that model-specific recommendation instead of forcing Gemini's temperature-zero setup onto a different reasoning architecture.

## 17.2 Nemotron smoke

Use the same six-condition one-packet 16K smoke.

Require:

```text
6/6 valid final verdicts
reasoning_effort recorded as high
reasoning present/reported
no truncation
no provider/model drift
```

Then run three real-length capacity probes, one per position.

Only after these pass should the 240-condition main run begin.

---

# 18. Phase Q — Conservative NIM scheduler and retry policy

NVIDIA trial limits are dynamic; do not optimize for maximum throughput.

## 18.1 Concurrency

Use:

```text
max_concurrency = 1
```

for every paper-quality run.

Do not parallelize.

## 18.2 Initial pacing

Use a conservative starting interval:

```text
minimum_interval_seconds = 60
```

for full-length main calls.

The goal is service stability, not benchmark throughput.

The scheduler may shorten this only if an operator explicitly changes the frozen pre-run configuration **before the main run**.

Do not dynamically speed up based on observed model answers.

## 18.3 Rate-limit handling

On HTTP 429:

1. persist the failed attempt;
2. read `Retry-After` if present;
3. inspect `x-ratelimit-*` headers if present;
4. wait at least the server-requested interval;
5. otherwise use exponential backoff with jitter;
6. retry at most two times.

Recommended fallback delay logic:

```text
delay =
    max(
        server_retry_after_if_present,
        base_backoff_seconds * 2**attempt
    )
```

with:

```text
base_backoff_seconds = 120
max_transport_retries = 2
```

A persistent 429 is not a wrong judgment and must not be counted as such.

If the service remains rate-limited after the retry cap:

```text
checkpoint
exit non-zero
leave condition incomplete
```

Resume later with the same frozen model/config.

## 18.4 Transport/service retry classes

Retry only:

```text
429
500
502
503
504
connection reset
timeout
```

subject to the cap.

Do not retry:

```text
valid model verdict that is wrong
tie verdict
clean/flawed order disagreement
malformed final JSON
schema/422 caused by an invalid frozen request
```

Malformed output is research data / parse failure, not a transport failure.

## 18.5 Local attempt cap

Maintain a persistent NIM API-attempt counter.

Suggested hard cap for one 240-condition robustness run:

```text
300 total HTTP attempts
```

including:

- smoke calls;
- capacity probes;
- retries;
- main requests.

Exceeding the local cap must stop the run before sending another request.

This protects against accidentally burning through trial credits during an outage/retry loop.

Do not imply this is NVIDIA's quota; it is our own safety cap.

---

# 19. Phase R — No automatic mid-run model failover

This is a critical scientific constraint.

Do not implement:

```text
try GLM;
if request fails -> send same condition to Nemotron
```

That would create a mixed-model dataset where service availability determines which judge saw each condition.

Instead:

### Before the main run

```text
GLM eligible -> choose GLM and freeze GLM run
GLM ineligible -> choose Nemotron and freeze Nemotron run
```

### After a GLM main run has started

Transient errors:

```text
pause/checkpoint/resume GLM
```

Permanent GLM loss or abandonment:

```text
archive partial GLM run as exploratory/incomplete
start a NEW complete 240-condition Nemotron run
```

Never fill holes in one model with the other.

---

# 20. Phase S — Robustness-model analysis

The NIM replication is 16K-only, so do not compute a new-model length interaction.

For whichever NIM model completes the full integrity-gated run, compute for each internal position:

```text
n_calls
n_pairs

raw_accuracy
raw_accuracy_cluster_bootstrap_ci

position_consistency
position_consistency_cluster_bootstrap_ci

decisive_certification_coverage
decisive_certification_coverage_cluster_bootstrap_ci

n_certified_correct
n_certified_wrong
selective_risk
selective_risk_exact_ci

stable_tie_fraction
mixed_tie_fraction
order_disagreement_fraction
```

## 20.1 Primary 16K contrast

For accuracy:

```text
new_model_accuracy_middle_penalty =
    pooled_edge_accuracy - middle_accuracy
```

For decisive certification coverage:

```text
new_model_coverage_middle_penalty =
    pooled_edge_coverage - middle_coverage
```

For position consistency:

```text
new_model_position_consistency_middle_penalty =
    pooled_edge_position_consistency - middle_position_consistency
```

Use 10,000 packet-level bootstrap resamples across the 40 matched packets.

## 20.2 Cross-model paired comparison with Gemini

Because the NIM model sees the same 40 underlying packets, add an exploratory paired model contrast.

For each bootstrap draw, sample packet IDs once and apply the same sampled packet multiset to both:

```text
Gemini 16K
NIM 16K
```

Compute:

```text
difference_in_accuracy_middle_penalty =
    NIM edge-minus-middle
    - Gemini edge-minus-middle

difference_in_coverage_middle_penalty =
    NIM edge-minus-middle
    - Gemini edge-minus-middle
```

This asks whether stronger reasoning materially attenuates the phenomenon.

Do not pool raw judgments across models.

## 20.3 Paper interpretation matrix

Precommit to:

### A. NIM model reproduces middle-specific deficit

Interpretation:

> The phenomenon generalizes beyond the economical Gemini judge to an independent stronger reasoning judge.

### B. NIM model improves overall accuracy but still has lower middle coverage

Interpretation:

> Greater judge capability improves baseline discrimination but does not eliminate positional sensitivity in the downstream certification gate.

### C. NIM model has little/no middle-specific effect

Interpretation:

> The failure is capability/configuration dependent rather than universal.

### D. NIM model is near-perfect in all three positions

Interpretation:

> The original Gemini result is likely specific to an underpowered/economical judge configuration; narrow the paper claim accordingly.

### E. NIM service cannot support a clean full run

Interpretation:

> No new-model evidence. Do not substitute partial or mixed-provider observations for a replication.

The Gemini negative result and reanalysis remain the primary paper evidence either way.

---

# 21. Phase T — Forced-choice ablation remains lower priority

Do not run the old no-tie Gemini ablation before the NIM robustness attempt.

Only reconsider it if all are true:

```text
1. Existing Gemini decomposition shows tie-involved rejection dominates.
2. The paper specifically needs to distinguish explicit abstention
   from clean↔flawed order instability.
3. The completed NIM robustness run does not already resolve the practical claim.
4. Additional API budget/time remains.
```

If performed, call it exploratory and keep it separate from the original preregistered protocol.

---

# 22. Benchmark-contamination / ecological-validity notes

Add explicit limitations.

## PRMBench contamination

Newer models may have encountered PRMBench or related benchmark material during training.

Do not claim the experiment measures completely unseen reasoning items.

The key causal contrast is positional relocation of the **same target content**, which reduces but does not eliminate contamination concerns.

If a model has memorized item correctness, that could make the base discrimination easier; it does not itself explain a middle-only deficit.

Still list contamination as a limitation.

## Synthetic long-form construction

The packets concatenate independent reasoning problems.

This enables unusually clean relocation control but does not reproduce discourse dependencies in:

- essays;
- reports;
- proofs;
- agent trajectories.

Keep this limitation.

Do not add a new naturalistic benchmark for the 2-page submission unless all higher-priority work is complete.

---

# 23. Final paper-facing result hierarchy

The paper should prioritize evidence in this order.

## Result 1 — preregistered negative result

The original false-certificate hypothesis failed.

Do not obscure this.

## Result 2 — Gemini length × middle-position accuracy interaction

Expected point estimate:

```text
+18.75 percentage points
```

with packet-bootstrap uncertainty.

## Result 3 — Gemini length × decisive-certification-coverage interaction

Expected point estimate:

```text
+15.0 percentage points
```

with packet-bootstrap uncertainty.

## Result 4 — position consistency versus decisive coverage distinction

If stable ties are common, this can become a major conceptual result:

> a judge can be "consistent" in the broad sense while failing to produce a decisive certificate.

## Result 5 — rejection mechanism

Report whether noncertification is mainly:

```text
stable/mixed ties
clean↔flawed order disagreement
or both
```

## Result 6 — independent stronger-model replication

If GLM-5.2 and/or DeepSeek R1 pass the integrity gate, report them as exploratory robustness evidence.

Do not let a secondary model crowd the core Gemini story in a two-page paper.

---

# 24. Suggested paper thesis

If the existing and robustness results support it:

> **Long-context internal-position stress need not make swap validation confidently wrong. Instead, it can reduce the rate at which the validation protocol yields a decisive usable judgment. Reliability should therefore be reported as a risk–coverage tradeoff, with position consistency distinguished from decisive certification coverage.**

If stronger models do not reproduce it, narrow to:

> **For an economical Gemini judge, internal-position stress produces a middle-specific accuracy and certification deficit; stronger-model replication suggests this failure is capability dependent rather than universal.**

---

# 25. Deliverables

Codex should leave the repository with:

```text
results/gemini/paper/audit.json
results/gemini/paper/pair_outcomes.csv
results/gemini/paper/pair_outcomes.json
results/gemini/paper/interactions.json
results/gemini/paper/native_token_lengths.csv
results/gemini/paper/main_table.csv
results/gemini/paper/main_table.md
results/gemini/paper/figures/...

results/nim/glm_5_2/...
results/nim/nemotron_3_super/...

paper_claims.md
```

New/updated code should include:

```text
tie-aware pair aggregation
position consistency metric
decisive certification coverage metric
exact selective-risk CI
packet-level interaction bootstrap
NVIDIA NIM provider
GLM reasoning-control preflight
Nemotron high-reasoning profile
NIM sequential rate scheduler
local NIM attempt counter
NIM smoke/integrity checks
no-mid-run-model-failover guard
tests
```

---

# 26. Final execution order

Codex should execute in this order:

```text
1. Cleanly audit the existing Gemini run.
2. Implement tie-aware pair decomposition.
3. Separate position consistency from decisive certification coverage.
4. Fix exact CER/selective-risk uncertainty.
5. Compute Gemini interaction effects.
6. Compute order robustness.
7. Verify native token lengths / tokenizer-independent positions.
8. Generate Gemini paper table and figure.
9. Run all reanalysis tests.
10. Quarantine Cohere from paper-facing outputs.

11. Implement the generic NVIDIA NIM provider using direct HTTP/SSE.
12. Implement persistent rate-limit headers, retry, checkpoint, and attempt-cap logic.
13. Add separate frozen configs for:
        z-ai/glm-5.2
        nvidia/nemotron-3-super-120b-a12b

14. Run GLM endpoint-availability preflight.
15. Probe and VERIFY the exact GLM high/max-thinking request shape.
16. If GLM reasoning is not verifiable or hosted access is unavailable:
        mark GLM ineligible
        select Nemotron BEFORE any main run

17. For selected model, run one-packet 16K six-condition smoke.
18. Run three representative full-length capacity probes.
19. Freeze selected model/config/reasoning request/scheduler.

20. Run the complete 16K robustness experiment:
        40 packets × 3 positions × 2 orders = 240 calls
        sequentially
        no model fallback within the run

21. Analyze the selected NIM model separately.
22. Compute packet-bootstrap edge-minus-middle contrasts.
23. Compute exploratory paired NIM-vs-Gemini 16K contrasts.
24. Apply the paper-inclusion integrity gate.
25. Reassess whether a forced-choice ablation is still scientifically necessary.
26. Update paper-facing claims using the predeclared interpretation matrix.
```

---

# 27. Short Codex brief

```text
GOAL

Strengthen the JUDGe two-page submission around the existing frozen Gemini
experiment, preserving the preregistered negative result while testing the
exploratory claim that long-context internal-position stress reduces the
availability of decisive AB/BA certificates.

PRIMARY WORK — NO NEW MODEL CALLS

1. Audit the existing Gemini run.
2. Separate:
   a) position consistency, where clean/clean, flawed/flawed, and tie/tie count;
   b) decisive certification coverage, where only matching non-tie winners count.
3. Partition every pair into:
   certified_correct, certified_wrong, stable_tie, mixed_tie,
   or clean↔flawed order_disagreement.
4. Report selective risk/CER with exact binomial CIs and k/n counts.
5. Compute packet-bootstrap length×position interactions.
   Expected existing point estimates:
       accuracy interaction = +0.1875
       decisive-coverage interaction = +0.1500
6. Add order robustness, native-token sanity checks, paper table/figure, and tests.
7. Keep Cohere artifacts but exclude Cohere from paper-facing evidence.
8. Do not foreground post-filter-minus-raw accuracy.

NVIDIA NIM ROBUSTNESS — ONLY AFTER GEMINI REANALYSIS

Preferred:
    z-ai/glm-5.2

Fallback:
    nvidia/nemotron-3-super-120b-a12b

Use:
    https://integrate.api.nvidia.com/v1/chat/completions
    NVIDIA_API_KEY

Important GLM gate:
    NVIDIA currently has inconsistent public availability signals for hosted
    GLM-5.2, and the public GLM inference schema does not expose the reasoning
    knob even though the model card advertises multiple thinking levels.
    Therefore GLM is eligible only if:
        - the hosted endpoint succeeds,
        - the exact high/max thinking request is accepted,
        - reasoning can be empirically verified,
        - six-condition JSON smoke passes,
        - full-length 16K capacity probes pass.
    Otherwise choose Nemotron BEFORE the main run.

Never switch models condition-by-condition.

NEMOTRON FALLBACK CONFIG

    model = nvidia/nemotron-3-super-120b-a12b
    temperature = 1.0
    top_p = 0.95
    reasoning_effort = high
    reasoning_budget = 16384
    max_tokens = 32768
    seed = 20260828
    stream = true

NIM MAIN EXPERIMENT

    use exact frozen 16K candidate texts
    40 packets × 3 positions × 2 orders = 240 calls
    run sequentially
    initial minimum interval = 60 seconds
    max concurrency = 1
    retry only transport/rate-limit failures
    max transport retries = 2
    persist Retry-After / x-ratelimit headers
    hard local attempt cap = 300
    malformed JSON is a parse failure, NOT a retry condition
    checkpoint after every call

GLM sampling:
    default temperature=0, seed=20260828, max_tokens=16384;
    freeze the exact empirically verified high/max reasoning request before main.

If a selected model's main run starts, never fill failed/missing conditions
with the other model. Pause/resume the same model. If abandoned, archive that
partial run and start a new complete fallback-model experiment.

Analyze new NIM evidence separately from Gemini:
    raw accuracy
    position consistency
    decisive certification coverage
    exact certified-error risk
    stable ties / mixed ties / order disagreement
    16K pooled-edge-minus-middle contrasts
    paired NIM-vs-Gemini exploratory contrast

Treat all NIM evidence as post-hoc robustness relative to the original
preregistration. Do not pool models and do not hide a non-replication.
```


---

## Research provenance for the NIM implementation choices

Verified against current NVIDIA materials on 2026-08-30:

- NVIDIA NIM LLM APIs use the OpenAI-compatible
  `https://integrate.api.nvidia.com/v1/chat/completions` endpoint.
- NVIDIA's GLM-5.2 documentation lists a 1M-token context, 753B parameters,
  reasoning capability, reasoning traces, and multiple thinking-effort levels.
- The current GLM-5.2 inference API explicitly documents temperature, top_p,
  max_tokens up to 32768, seed, and streaming, but does not currently expose a
  reasoning field in that public schema. This is why reasoning activation must
  be verified rather than assumed.
- NVIDIA's current GLM-5.2 availability pages are inconsistent: the inference
  API remains documented while the current model build page reports the free
  endpoint as deprecated. This is why GLM availability is a preflight gate.
- NVIDIA Nemotron 3 Super has up to 1M context; its inference API explicitly
  documents `reasoning_effort=high` and `reasoning_budget` up to 32768.
- NVIDIA documents `reasoning_effort=high` as full reasoning for Nemotron 3
  Super and a default reasoning budget of 16384.
- NVIDIA's Nemotron model card recommends temperature 1.0 and top_p 0.95.
- NVIDIA staff state that free/trial NIM rate limits can vary with model,
  use case, and current traffic. Therefore no fixed 40-RPM assumption is used
  in the scientific configuration.
