"""Evaluation diagnostics for prototype simulations."""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from random import Random
from typing import Dict, List, Sequence

import pandas as pd

from .assistant import LLMOnlyRetailAssistant, NeuroSymbolicRetailAssistant
from .llm_assistant import LLMOnlyRetailAssistantReal
from .models import Product, RecommendationResult
from .user_state import infer_interaction_state


def result_to_metrics(result: RecommendationResult, budget: float | None) -> Dict[str, object]:
    """Convert a recommendation result into implementation-level metrics."""
    explanation_text = (result.explanation + " " + result.response).lower()
    budget_violation = bool(budget is not None and result.product.price > budget)
    unavailable_violation = not result.product.available
    budget_explanation = any(token in explanation_text for token in ["budget", "price", "euros"])
    contestability_explanation = any(token in explanation_text for token in ["alternative", "cheaper", "formal", "casual"])

    return {
        "condition": result.condition,
        "product_id": result.product.id,
        "product_name": result.product.name,
        "price": result.product.price,
        "premium": result.product.premium,
        "available": result.product.available,
        "budget": budget,
        "budget_violation": budget_violation,
        "unavailable_violation": unavailable_violation,
        "budget_explanation": budget_explanation,
        "contestability_explanation": contestability_explanation,
        "plan_length": len(result.plan),
        "plan": " | ".join(result.plan),
        "response": result.response,
    }


def run_simulation(
    catalog: Sequence[Product],
    n_users: int = 200,
    seed: int = 7,
    pddl_output_dir: str | Path = "generated_pddl",
    use_real_llm: bool = False,
) -> pd.DataFrame:
    """Run the LLM-only and neuro-symbolic conditions over simulated users.

    Parameters
    ----------
    use_real_llm:
        When True, the LLM-only condition calls the Anthropic API via
        LLMOnlyRetailAssistantReal instead of the stochastic heuristic.
        Requires ANTHROPIC_API_KEY in the environment.
    """
    rng = Random(seed)
    if use_real_llm:
        llm: LLMOnlyRetailAssistant | LLMOnlyRetailAssistantReal = LLMOnlyRetailAssistantReal(
            catalog, rng=Random(seed + 1)
        )
    else:
        llm = LLMOnlyRetailAssistant(catalog, rng=Random(seed + 1))
    neuro = NeuroSymbolicRetailAssistant(catalog, pddl_output_dir=pddl_output_dir, rng=Random(seed + 2))
    rows: List[Dict[str, object]] = []

    utterance_templates = [
        "I need comfortable trousers for everyday wear, preferably under {budget} euros. Please do not show premium options.",
        "I want relaxed trousers below {budget} euros. Comfort matters most.",
        "I am not sure what to buy, but I need comfortable everyday trousers under {budget} euros.",
        "I need trousers for daily use, maximum {budget} euros, and I prefer low-pressure advice.",
    ]

    for idx in range(n_users):
        budget = rng.choice([50, 55, 60, 65])
        utterance = rng.choice(utterance_templates).format(budget=budget)
        state = infer_interaction_state(utterance, rng=rng, budget_extraction_error_rate=0.03)

        for assistant in [llm, neuro]:
            result = assistant.recommend(state)
            metrics = result_to_metrics(result, budget=budget)
            metrics["trial"] = idx
            metrics["extracted_budget_correctly"] = state.extracted_budget_correctly
            rows.append(metrics)

    return pd.DataFrame(rows)


def summarize_conditions(trials: pd.DataFrame) -> pd.DataFrame:
    """Aggregate trial-level metrics by condition."""
    summary = (
        trials.groupby("condition")
        .agg(
            budget_violation_rate=("budget_violation", "mean"),
            unavailable_violation_rate=("unavailable_violation", "mean"),
            budget_explanation_rate=("budget_explanation", "mean"),
            contestability_explanation_rate=("contestability_explanation", "mean"),
            mean_price=("price", "mean"),
            mean_plan_length=("plan_length", "mean"),
        )
        .reset_index()
    )
    return summary
