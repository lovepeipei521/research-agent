# Agent Evaluation Set v1

## Purpose

This resource defines a 120-case evaluation task set for this agent. It is an
evaluation design artifact, not a published benchmark result.  The cases have
not been executed against a fixed model or configuration.

## Scope

The task set is split into six groups of 20 cases:

| Group | What it checks | Primary measurable result after execution |
| --- | --- | --- |
| `research` | Evidence-backed research answers and source handling | task completion rate, citation coverage |
| `fallback` | Tool failure, circuit-breaker, and configured fallback behavior | fallback recovery rate |
| `context` | Context compression, externalization, and fact retention | required-fact retention rate |
| `memory` | Long-term memory recall and irrelevant-memory suppression | recall precision and recall |
| `orchestration` | Specialist selection, delegation, and result synthesis | specialist task completion rate |
| `quality` | L3 execution judgment, task-quality checks, and response contracts | contract pass rate and quality score |

Each case is intentionally expressed as a task contract rather than a
model-specific expected answer.  This lets the same set run against different
models without turning a model's wording into the ground truth.

## Execution Rules

Before reporting any performance number, run every case under one fixed
configuration: pinned model, prompt version, tool configuration, and date.
For each case, persist the trace, tool outcomes, final response, elapsed time,
and token usage.  A case passes only when every `acceptance` condition in the
JSON data is satisfied.

Report the following only after execution:

```text
task_completion_rate = passed_cases / executed_cases
fallback_recovery_rate = recovered_injected_failures / injected_failures
fact_retention_rate = retained_required_facts / required_facts
contract_pass_rate = contract_passes / executed_cases
latency = mean, p50, p95 wall-clock duration
cost = mean input tokens, output tokens, and provider cost per case
```

L3 LLM-judge scores must be labelled as LLM-judge scores and should be checked
against a manually reviewed sample.  They are not interchangeable with a
ground-truth accuracy rate.

## Resume-Safe Evidence

The following statements are true as soon as the files in this resource exist:

```text
- Designed a 120-case Agent evaluation task set spanning research, fallback,
  context governance, long-term memory, multi-agent orchestration, and L3
  response-quality contracts.
- Defined measurable pass criteria for task completion, fallback recovery,
  fact retention, response contracts, latency, and token cost.
```

Do not claim a success rate, latency percentile, token cost, or quality score
until the data set has been run and the raw traces are retained.

## Suggested Resume Wording

Use the wording that matches the evidence available at submission time:

```text
Before execution:
  Designed a 120-case Agent evaluation set across six capability dimensions,
  with task-level acceptance criteria for tool fallback, context retention,
  memory recall, specialist orchestration, and response contracts.

After execution:
  Evaluated this agent on 120 controlled Agent tasks; achieved <verified metric>
  task completion and <verified metric> fallback recovery under <fixed config>.
```

The second version requires the actual run artifact.  Replace only the angle
bracket placeholders with measured values.
