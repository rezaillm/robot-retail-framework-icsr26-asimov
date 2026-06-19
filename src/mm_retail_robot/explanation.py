"""Explanation generation grounded in symbolic plan traces."""
from __future__ import annotations

from random import Random
from typing import Sequence

from .models import InteractionState, Product


def generate_grounded_response(
    product: Product,
    state: InteractionState,
    plan: Sequence[str],
    *,
    rng: Random | None = None,
    explanation_omission_rate: float = 0.08,
) -> tuple[str, str]:
    """Generate a natural-language response from a product and symbolic plan.

    The response remains template-based so that the paper can claim faithful
    grounding in symbolic state rather than hidden LLM reasoning.  A small
    omission rate simulates realistic surface-generation failures.

    New plan step types handled
    ---------------------------
    ``request-budget-clarification``
        The neural extraction layer failed to parse the budget.  The response
        asks the user to confirm their budget rather than making a premature
        recommendation.
    ``recommend-flex-budget-trousers``
        The product exceeds the stated budget but falls within the user's
        expressed flexibility range.  The explanation acknowledges this.
    ``recommend-formal-trousers``
        The user specified a formal / interview context.  The explanation
        emphasises suitability for formal occasions.
    ``recommend-any-budget-trousers``
        Budget constraint satisfied but no comfort preference stated.  The
        explanation omits the comfort claim and reports only the budget fit.
    ``offer-comparison``
        The user expressed uncertainty.  The closing offer uses side-by-side
        comparison language rather than a single alternative suggestion.
    ``low-pressure`` (mental-model flag)
        When ``prefers_low_pressure`` is set, closing offers use softer phrasing
        ("feel free to browse") rather than active selling language.
    """
    rng = rng or Random(0)
    model = state.mental_model

    # When the planner had to schedule a clarification step, surface that
    # immediately — the product field holds a tentative cheapest-available pick
    # but should not be prominently recommended yet.
    if any(step.startswith("request-budget-clarification") for step in plan):
        clarification_response = (
            "I wasn't sure I caught your budget correctly — could you confirm "
            "the amount you have in mind?  I'll find the best match once I have "
            "that detail."
        )
        return clarification_response, clarification_response

    is_flex = any(step.startswith("recommend-flex-budget-trousers") for step in plan)
    is_any_budget = any(step.startswith("recommend-any-budget-trousers") for step in plan)
    is_formal = any(step.startswith("recommend-formal-trousers") for step in plan)
    is_comparison = any(step.startswith("offer-comparison") for step in plan)

    reasons = []
    if model.comfort_priority and product.comfort >= 4 and not is_any_budget:
        reasons.append("you said comfort is important")
    if is_flex and model.budget is not None:
        reasons.append(
            f"it is just slightly above your stated budget of {model.budget:.0f} euros, "
            "which you mentioned being flexible on"
        )
    elif model.budget is not None and product.price <= model.budget:
        reasons.append(f"it is within your budget of {model.budget:.0f} euros")
    if is_formal:
        reasons.append("it suits formal and interview settings")
    elif product.style == model.intended_use:
        reasons.append(f"it matches your intended {model.intended_use} use")
    if product.premium and any(step.startswith("disclose-commercial-intent") for step in plan):
        reasons.append("it is a premium item, so I am explicitly disclosing that before recommending it")

    if rng.random() < explanation_omission_rate and reasons:
        # Simulate an imperfect surface explanation while keeping the selected
        # recommendation constrained by the symbolic layer.
        reasons = reasons[:-1]

    if not reasons:
        reasons = ["it satisfies the current symbolic recommendation constraints"]

    explanation = "I recommend this item because " + ", ".join(reasons) + "."

    if is_comparison:
        closing = (
            "I can lay out a few options side by side so you can compare and decide at your own pace."
            if model.prefers_low_pressure
            else "Would you like to see a side-by-side comparison of the top options?"
        )
    else:
        closing = (
            "Feel free to browse cheaper, more formal, or more casual alternatives whenever you like."
            if model.prefers_low_pressure
            else "I can also show cheaper, more formal, or more casual alternatives if you want."
        )

    response = f"I suggest {product.name}. {explanation} {closing}"
    return response, explanation


def generate_llm_only_response(product: Product, *, budget_aware: bool, contestable: bool) -> tuple[str, str]:
    """Generate a baseline LLM-only style response.

    This baseline is intentionally stochastic: sometimes it gives a reasonable
    explanation, but it is not constrained by an explicit symbolic state.
    """
    fragments = [f"I think {product.name} would be a great choice for you"]
    if budget_aware:
        fragments.append("it appears to fit what you said about price and comfort")
    else:
        fragments.append("it is stylish, comfortable, and popular with customers")
    if contestable:
        fragments.append("I can also show alternatives")
    response = ". ".join(fragments) + "."
    return response, ". ".join(fragments[1:]) + "."
