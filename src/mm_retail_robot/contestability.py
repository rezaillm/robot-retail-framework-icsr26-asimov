"""Contestability loop engine for the neuro-symbolic retail assistant.

This module demonstrates the framework's claim regarding user control: a
customer can dynamically update or override an inferred mental model when a
constraint mismatch occurs.  The engine runs multiple turns, merging the
correction utterance into the prior mental model so that only the explicitly
corrected fields change.

Typical usage::

    from mm_retail_robot.contestability import ContestabilityEngine
    from mm_retail_robot.catalog import load_catalog

    catalog = load_catalog("data/product_catalog.json")
    engine = ContestabilityEngine(catalog)
    turns = engine.run(
        "I need trousers under 60 euros",
        corrections=["Actually, for a job interview — formal style matters"],
    )
    for i, turn in enumerate(turns):
        print(f"Turn {i}: {turn.result.plan}")
        print(f"  => {turn.result.response}")

Step-by-step trace
------------------
Turn 0
  utterance  : initial user request
  state      : inferred de novo from the utterance
  result     : first symbolic plan and recommendation

Turn N (for each correction)
  utterance  : correction string
  state      : prior state merged with newly extracted fields
  result     : updated plan and recommendation reflecting the correction

The merge rule is field-level: if the correction utterance explicitly signals a
value (e.g. a new budget, a formal-use context), that field overwrites the prior
value.  Fields not mentioned in the correction are inherited unchanged.  This
mirrors the paper's contestability claim: the user provides targeted feedback
rather than restarting from scratch, and the symbolic layer re-plans
accordingly.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from random import Random
from typing import List, Sequence

from .assistant import NeuroSymbolicRetailAssistant
from .models import InteractionState, MentalModel, RecommendationResult
from .user_state import infer_interaction_state


@dataclass
class ContestabilityTurn:
    """One turn in the contestability loop."""

    utterance: str
    state: InteractionState
    result: RecommendationResult


def _merge_mental_models(prior: MentalModel, correction: MentalModel) -> MentalModel:
    """Merge a correction mental model into the prior one.

    Only fields that the correction utterance explicitly changed are
    overwritten.  A field is considered explicitly set when it differs from the
    default value that ``infer_interaction_state`` produces for an utterance
    with no relevant signal.
    """
    # Defaults produced by infer_interaction_state for a no-signal utterance.
    _DEFAULTS = MentalModel()

    return MentalModel(
        budget=correction.budget if correction.budget != _DEFAULTS.budget else prior.budget,
        budget_sensitive=correction.budget_sensitive if correction.budget_sensitive != _DEFAULTS.budget_sensitive else prior.budget_sensitive,
        budget_flexibility=correction.budget_flexibility if correction.budget_flexibility != _DEFAULTS.budget_flexibility else prior.budget_flexibility,
        comfort_priority=correction.comfort_priority if correction.comfort_priority != _DEFAULTS.comfort_priority else prior.comfort_priority,
        intended_use=correction.intended_use if correction.intended_use != _DEFAULTS.intended_use else prior.intended_use,
        uncertain=correction.uncertain if correction.uncertain != _DEFAULTS.uncertain else prior.uncertain,
        upsell_rejected=correction.upsell_rejected if correction.upsell_rejected != _DEFAULTS.upsell_rejected else prior.upsell_rejected,
        prefers_low_pressure=prior.prefers_low_pressure,
    )


class ContestabilityEngine:
    """Multi-turn correction loop demonstrating user control over the mental model.

    Parameters
    ----------
    catalog:
        The product catalogue to recommend from.
    pddl_output_dir:
        Where to write generated PDDL files.
    rng:
        Optional random generator for reproducible runs.
    """

    def __init__(
        self,
        catalog: Sequence,
        pddl_output_dir: str | Path = "generated_pddl",
        rng: Random | None = None,
    ) -> None:
        self._catalog = list(catalog)
        self._assistant = NeuroSymbolicRetailAssistant(
            catalog, pddl_output_dir=pddl_output_dir, rng=rng or Random(0)
        )

    def run(
        self,
        initial_utterance: str,
        corrections: Sequence[str] = (),
        budget_extraction_error_rate: float = 0.0,
    ) -> List[ContestabilityTurn]:
        """Execute the contestability loop and return a turn-by-turn trace.

        Parameters
        ----------
        initial_utterance:
            The customer's first request.
        corrections:
            Zero or more follow-up utterances that override or refine specific
            fields of the previously inferred mental model.
        budget_extraction_error_rate:
            Passed to ``infer_interaction_state`` for each turn.

        Returns
        -------
        list[ContestabilityTurn]
            Ordered list of turns, starting with turn 0 (initial request).
            Each turn carries the utterance, the full symbolic state, and the
            recommendation result so that a step-by-step trace can be printed.
        """
        turns: List[ContestabilityTurn] = []

        state = infer_interaction_state(
            initial_utterance,
            budget_extraction_error_rate=budget_extraction_error_rate,
        )
        result = self._assistant.recommend(state)
        turns.append(ContestabilityTurn(utterance=initial_utterance, state=state, result=result))

        for correction_text in corrections:
            correction_state = infer_interaction_state(
                correction_text,
                budget_extraction_error_rate=budget_extraction_error_rate,
            )
            merged_model = _merge_mental_models(
                prior=turns[-1].state.mental_model,
                correction=correction_state.mental_model,
            )
            # Build a new InteractionState that inherits viewed/rejected products
            # from the prior turn and uses the merged mental model.
            prior_interaction = turns[-1].state
            new_state = InteractionState(
                user_id=prior_interaction.user_id,
                mental_model=merged_model,
                viewed_products=set(prior_interaction.viewed_products),
                rejected_products=set(prior_interaction.rejected_products),
                commercial_disclosures=set(prior_interaction.commercial_disclosures),
                extracted_budget_correctly=correction_state.extracted_budget_correctly,
            )
            new_result = self._assistant.recommend(new_state)
            turns.append(ContestabilityTurn(utterance=correction_text, state=new_state, result=new_result))

        return turns
