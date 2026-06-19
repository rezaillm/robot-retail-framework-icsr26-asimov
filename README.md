# robot-retail-framework-icsr26-asimov
Mental-Model-Aware Retail Robots: Neuro-Symbolic Explainable Assistance for Trustworthy Human–Robot Commerce

# Mental-Model-Aware Retail Robot Prototype

A lightweight Python prototype comparing a **real LLM-only retail robot** (Claude Haiku) against a **neuro-symbolic retail robot** in a trouser-recommendation scenario.

Designed for the ICSR 2026 ASIMOV workshop paper titled "Mental-Model-Aware Retail Robots: Neuro-Symbolic Explainable Assistance for Trustworthy Human–Robot Commerce". 
Intentionally small, reproducible, and inspectable.

## What is implemented?

Two conditions:

1. **LLM-only baseline** (`LLMOnlyRetailAssistantReal`)
   - Calls Claude Haiku directly with the user utterance and product catalog.
   - No symbolic planning or verified interaction state.
   - A stochastic heuristic fallback (`LLMOnlyRetailAssistant`) is available for offline runs without an API key.

2. **Neuro-symbolic condition** (`NeuroSymbolicRetailAssistant`)
   - Extracts a compact symbolic mental model from the user request (economic + non-economic parameters).
   - Compiles the interaction state into PDDL files (`generated_pddl/domain.pddl`, `generated_pddl/problem.pddl`).
   - Solves the planning problem with an embedded STRIPS planner.
   - Grounds the recommendation and explanation in the symbolic plan.
   - Supports a multi-turn **contestability loop** allowing users to correct the inferred mental model.

The generated PDDL files are human-readable and compatible with external planners (Fast Downward, pyperplan).

## Simulation results

200-interaction simulation (budgets 50–65 EUR, 3% upstream NLU extraction noise):

| Metric | LLM-only | Neuro-symbolic |
|--------|----------|----------------|
| Budget violation rate | 0.00 | 0.05 |
| Unavailable-product violation | 0.00 | 0.00 |
| Budget explanation rate | 1.00 | 0.995 |
| Contestability explanation rate | **0.185** | **0.685** |
| Mean recommended price (€) | 49.99 | 53.47 |
| Mean symbolic plan length | 0.0 | 2.095 |

The real LLM already handles budget constraints reliably. The symbolic layer's primary contribution is systematic contestability cues (3.7× higher rate) and an auditable plan trace.

## Mental model

`MentalModel` ([src/mm_retail_robot/models.py](src/mm_retail_robot/models.py)) captures inferred customer preferences. Every field is compiled into a typed PDDL predicate.

| Field | PDDL predicate | Notes |
|-------|----------------|-------|
| `budget` | `under-budget` | Hard price ceiling extracted from the utterance |
| `budget_sensitivity` | `budget-sensitive` | True when price is an explicit concern |
| `budget_flexibility` | `budget-flexible` + `near-budget` | Non-zero when hedging language detected ("a bit over", "around", "flexible") |
| `comfort_priority` | `comfort-priority` | Triggers comfort-aware recommendation action |
| `intended_use == "formal"` | `formal-use` | Triggers `recommend-formal-trousers` |
| `uncertain` | `uncertain` | Triggers `offer-comparison` instead of `offer-alternative` |
| `upsell_rejected` | `upsell-rejected` | Blocks `recommend-premium-trousers` |
| `prefers_low_pressure` | `low-pressure` | Softens alternative-suggestion phrasing |

## PDDL domain actions

| Action | Fires when |
|--------|-----------|
| `request-budget-clarification` | Budget parse failed; blocks all recommend actions until resolved |
| `recommend-budget-trousers` | Budget + comfort preference stated; product within budget |
| `recommend-any-budget-trousers` | Budget stated, no comfort/formal preference; product within budget |
| `recommend-flex-budget-trousers` | Budget flexibility expressed; product within extended range |
| `recommend-formal-trousers` | Formal/interview context specified |
| `recommend-premium-trousers` | Upsell not rejected, not budget-sensitive; commercial intent disclosed first |
| `offer-alternative` | Recommendation made; user not uncertain |
| `offer-comparison` | Recommendation made; user expressed uncertainty |
| `explain-recommendation` | Product recommended |
| `disclose-commercial-intent` | Premium product; must precede any premium recommendation |

## Error handling

When the NLU layer fails to extract the budget, `clarification-needed` is asserted in the PDDL init state. All `recommend-*` actions guard on `not (clarification-needed ?u)`, forcing `request-budget-clarification` before any product recommendation. The `RecommendationResult.metadata["clarification_needed"]` flag surfaces this to callers.

## Contestability loop

`ContestabilityEngine` ([src/mm_retail_robot/contestability.py](src/mm_retail_robot/contestability.py)) implements multi-turn correction:

```python
from mm_retail_robot.contestability import ContestabilityEngine
from mm_retail_robot.catalog import load_catalog

engine = ContestabilityEngine(load_catalog("data/product_catalog.json"))
turns = engine.run(
    "I need trousers under 60 euros",
    corrections=["Actually, for a job interview — formal is important"],
)
for i, turn in enumerate(turns):
    print(f"Turn {i}: {turn.result.plan}")
    print(f"  => {turn.result.response}")
```

Corrections are merged field-by-field: only explicitly changed fields are overwritten; everything else (e.g., budget from turn 0) is inherited.

## Scalability

`breadth_first_plan()` pre-filters the catalog to the top-K candidates (default 20) scored by budget fit, comfort, style, and upsell penalty before grounding. This bounds the planner's action space to O(K) regardless of catalog size. Actions are also sorted by priority type so BFS finds the shortest paths first.

## Repository structure

```
mma_retail_robot_framework/
├── requirements.txt
├── data/product_catalog.json
├── generated_pddl/          — written at runtime
├── results/                 — simulation outputs
├── results_llm/             — real-LLM simulation outputs
├── scripts/
│   ├── run_demo.py
│   └── run_simulated_evaluation.py
├── src/mm_retail_robot/
│   ├── assistant.py         — LLMOnlyRetailAssistant, NeuroSymbolicRetailAssistant
│   ├── llm_assistant.py     — LLMOnlyRetailAssistantReal (Anthropic API)
│   ├── catalog.py
│   ├── contestability.py    — ContestabilityEngine
│   ├── evaluator.py         — simulation and metrics
│   ├── explanation.py       — plan-grounded response generation
│   ├── models.py            — MentalModel, InteractionState, RecommendationResult
│   ├── pddl.py              — PDDL compilation, STRIPS planner, catalog filter
│   └── user_state.py        — utterance-to-state extraction
└── tests/test_pddl_planner.py
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`anthropic` is included in `requirements.txt` and needed only for `--real-llm` runs. Set `ANTHROPIC_API_KEY` in your environment before using it.

## Run a demonstration

```bash
python scripts/run_demo.py \
  --utterance "I need comfortable trousers for everyday wear, preferably under 60 euros. Please do not show premium options."
```

The neuro-symbolic condition prints a symbolic plan and writes PDDL files to `generated_pddl/`.

## Run the simulated evaluation

**Stochastic baseline (no API key needed):**

```bash
python scripts/run_simulated_evaluation.py --n-users 200 --output-dir results
```

**Real LLM baseline (requires `ANTHROPIC_API_KEY`):**

```bash
export ANTHROPIC_API_KEY=sk-ant-...
python scripts/run_simulated_evaluation.py --real-llm --n-users 200 --output-dir results_llm
```

Outputs: `simulated_trials.csv`, `condition_summary.csv`, `rate_metrics_noisy_pddl.pdf`, `mean_price_noisy_pddl.pdf`.

## Run the tests

```bash
python -m pytest tests/ -v
```

Covers: budget-constraint enforcement, PDDL file generation, flex-budget extraction and planning, non-economic predicate compilation, contestability loop state merging, upstream clarification handling, and catalog pre-filtering.
