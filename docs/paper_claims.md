# JUDGe Paper Claims Guardrails

Following the implementation plan, these are the constraints for drafting the 2-page workshop submission:

## 1. What is already known
Do not claim novelty for:
- candidate-order bias in LLM judges;
- AB/BA swapping as a diagnostic/mitigation;
- "lost in the middle";
- long-context LLM-judge unreliability;
- within-document error/perturbation position affecting judge behavior;
- the generic fact that filtering inconsistent judgments reduces sample size.

The novelty boundary is the downstream systems question:
> How does internal evidence-position stress propagate through an AB/BA validation gate, and how often does that gate still yield a decisive usable judgment?

## 2. Avoid the quality-gap dilution trap
Do not make the overall 4K → 16K accuracy/coverage decline the central causal result.
Emphasize the **within-length early/middle/late contrast**.
Make the **length × position interaction** the main statistic.

## 3. Avoid over-attributing rejection to "position bias"
Call the operational quantity `certification_coverage`.
Call clean/flawed reversals `order_disagreement`.
Do not claim every rejected pair demonstrates candidate-order bias.
Reserve `position_consistency` for the defined same-outcome-after-swap metric (including stable ties).

## 4. Selective-prediction framing
Use:
```text
certification_coverage = P(decisive swap certificate)
selective_risk         = P(wrong | decisive swap certificate)
```
Do not foreground post-filter minus raw accuracy, as that is selected-subset accuracy minus population accuracy.

## 5. Thesis
If the existing and robustness results support it:
> **Long-context internal-position stress need not make swap validation confidently wrong. Instead, it can reduce the rate at which the validation protocol yields a decisive usable judgment. Reliability should therefore be reported as a risk–coverage tradeoff, with position consistency distinguished from decisive certification coverage.**

If stronger models do not reproduce it, narrow to:
> **For an economical Gemini judge, internal-position stress produces a middle-specific accuracy and certification deficit; stronger-model replication suggests this failure is capability dependent rather than universal.**

## 6. Model Mentions
The existing frozen Gemini run is the **primary experiment** (gemini-3.5-flash-lite).
Cohere is quarantined from paper-facing evidence and should be treated as historical/exploratory only. Do not mention "two independent proprietary judge families."
NVIDIA NIM (GLM-5.2 or Nemotron 3 Super) is an exploratory robustness replication, analyzed separately and never pooled with Gemini.
