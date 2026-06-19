"""Assistant policies for LLM-only and neuro-symbolic conditions."""
from __future__ import annotations

from pathlib import Path
from random import Random
from typing import List, Sequence

from .explanation import generate_grounded_response, generate_llm_only_response
from .models import InteractionState, Product, RecommendationResult
from .pddl import breadth_first_plan, recommended_product_from_plan, write_pddl_problem


class LLMOnlyRetailAssistant:
    """Stochastic unconstrained baseline approximating direct LLM recommendation."""

    def __init__(self, catalog: Sequence[Product], rng: Random | None = None) -> None:
        self.catalog = list(catalog)
        self.rng = rng or Random(0)

    def recommend(self, state: InteractionState) -> RecommendationResult:
        """Return a recommendation without symbolic verification.

        The baseline sometimes respects budget, but it is also biased toward
        premium/promoted products to model a plausible commercial assistant.
        """
        available = [product for product in self.catalog if product.available]
        budget = state.mental_model.budget

        choose_budget_aware = budget is not None and self.rng.random() < 0.48
        if choose_budget_aware:
            candidates = [p for p in available if p.price <= budget]
            product = max(candidates or available, key=lambda p: (p.comfort, -p.price))
        else:
            product = max(available, key=lambda p: (p.premium, p.promotion, p.price))

        budget_aware_expl = self.rng.random() < 0.38
        contestable = self.rng.random() < 0.26
        response, explanation = generate_llm_only_response(
            product,
            budget_aware=budget_aware_expl,
            contestable=contestable,
        )
        return RecommendationResult(
            condition="llm_only",
            product=product,
            response=response,
            explanation=explanation,
            plan=[],
            metadata={"budget_aware_generation": budget_aware_expl, "contestable_generation": contestable},
        )


class NeuroSymbolicRetailAssistant:
    """Neuro-symbolic policy using PDDL compilation and symbolic planning."""

    def __init__(
        self,
        catalog: Sequence[Product],
        pddl_output_dir: str | Path = "generated_pddl",
        rng: Random | None = None,
    ) -> None:
        self.catalog = list(catalog)
        self.pddl_output_dir = Path(pddl_output_dir)
        self.rng = rng or Random(0)

    def recommend(self, state: InteractionState) -> RecommendationResult:
        """Compile the state into PDDL, generate a symbolic plan, and verbalize it."""
        domain_path, problem_path = write_pddl_problem(state, self.catalog, self.pddl_output_dir)
        plan = breadth_first_plan(state, self.catalog)
        product = recommended_product_from_plan(plan, self.catalog)

        if product is None:
            # Conservative fallback: no admissible PDDL plan was found.  Use the
            # cheapest available item and explicitly mark the failure.
            product = min([p for p in self.catalog if p.available], key=lambda p: p.price)
            plan = ["fallback-cheapest-available"]

        response, explanation = generate_grounded_response(product, state, plan, rng=self.rng)
        clarification_needed = not state.extracted_budget_correctly and any(
            step.startswith("request-budget-clarification") for step in plan
        )
        return RecommendationResult(
            condition="neurosymbolic",
            product=product,
            response=response,
            explanation=explanation,
            plan=plan,
            pddl_domain_path=str(domain_path),
            pddl_problem_path=str(problem_path),
            metadata={
                "pddl_plan_found": plan != ["fallback-cheapest-available"],
                "clarification_needed": clarification_needed,
            },
        )
