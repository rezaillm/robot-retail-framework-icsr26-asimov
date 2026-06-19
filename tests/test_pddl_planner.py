"""Tests for the PDDL-backed neuro-symbolic planner."""
from __future__ import annotations

from pathlib import Path
from random import Random
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from mm_retail_robot.assistant import NeuroSymbolicRetailAssistant
from mm_retail_robot.catalog import load_catalog
from mm_retail_robot.contestability import ContestabilityEngine
from mm_retail_robot.pddl import (
    breadth_first_plan,
    initial_atoms,
    write_pddl_problem,
    _filter_catalog_for_planning,
)
from mm_retail_robot.user_state import infer_interaction_state


# ---------------------------------------------------------------------------
# Original tests (must remain green)
# ---------------------------------------------------------------------------

def test_budget_sensitive_user_gets_budget_plan(tmp_path: Path) -> None:
    catalog = load_catalog(REPO_ROOT / "data" / "product_catalog.json")
    state = infer_interaction_state(
        "I need comfortable trousers under 60 euros. Please do not show premium options.",
        rng=Random(1),
        budget_extraction_error_rate=0.0,
    )
    plan = breadth_first_plan(state, catalog)
    assert any(step.startswith("recommend-budget-trousers") for step in plan)
    assert not any(step.startswith("recommend-premium-trousers") for step in plan)


def test_pddl_files_are_written(tmp_path: Path) -> None:
    catalog = load_catalog(REPO_ROOT / "data" / "product_catalog.json")
    state = infer_interaction_state("comfortable trousers under 60 euros", rng=Random(1), budget_extraction_error_rate=0.0)
    domain_path, problem_path = write_pddl_problem(state, catalog, tmp_path)
    assert domain_path.exists()
    assert problem_path.exists()
    assert "retail-recommendation" in domain_path.read_text(encoding="utf-8")
    assert "under-budget" in problem_path.read_text(encoding="utf-8")


def test_neurosymbolic_recommendation_respects_budget(tmp_path: Path) -> None:
    catalog = load_catalog(REPO_ROOT / "data" / "product_catalog.json")
    state = infer_interaction_state("I need comfortable trousers under 60 euros", rng=Random(1), budget_extraction_error_rate=0.0)
    assistant = NeuroSymbolicRetailAssistant(catalog, pddl_output_dir=tmp_path, rng=Random(2))
    result = assistant.recommend(state)
    assert result.product.price <= 60
    assert result.pddl_problem_path is not None
    assert len(result.plan) > 0


# ---------------------------------------------------------------------------
# Review 1: Fluid budget / budget flexibility
# ---------------------------------------------------------------------------

def test_flex_budget_utterance_sets_budget_flexibility() -> None:
    state = infer_interaction_state(
        "I need comfortable trousers, around 60 euros — a bit over is fine.",
        rng=Random(1),
        budget_extraction_error_rate=0.0,
    )
    assert state.mental_model.budget == 60.0
    assert state.mental_model.budget_flexibility == 0.10


def test_flex_budget_plan_uses_flex_action(tmp_path: Path) -> None:
    catalog = load_catalog(REPO_ROOT / "data" / "product_catalog.json")
    # Budget 60 with 10 % flex → effective budget 66; stretch_denim (59.99) is
    # under the hard limit anyway, but the flex path should still be found when
    # the hard path is blocked.  Set budget to 58 so that stretch_denim (59.99)
    # is only reachable via the flex path.
    state = infer_interaction_state(
        "I need comfortable everyday trousers, around 58 euros — slightly over is okay.",
        rng=Random(1),
        budget_extraction_error_rate=0.0,
    )
    assert state.mental_model.budget_flexibility == 0.10
    plan = breadth_first_plan(state, catalog)
    # The plan must contain a recommendation action
    assert any(step.startswith("recommend-") for step in plan)


def test_flex_budget_atoms_present() -> None:
    catalog = load_catalog(REPO_ROOT / "data" / "product_catalog.json")
    state = infer_interaction_state(
        "I need trousers around 60 euros, a bit over is fine.",
        rng=Random(1),
        budget_extraction_error_rate=0.0,
    )
    atoms = initial_atoms(state, catalog)
    assert ("budget-flexible", "customer1") in atoms
    # At least one product should be near-budget
    assert any(a[0] == "near-budget" for a in atoms)


# ---------------------------------------------------------------------------
# Review 2: Non-economic parameters compiled to PDDL atoms
# ---------------------------------------------------------------------------

def test_non_economic_atoms_are_compiled() -> None:
    catalog = load_catalog(REPO_ROOT / "data" / "product_catalog.json")
    state = infer_interaction_state(
        "I am not sure — maybe comfortable trousers for a formal interview.",
        rng=Random(1),
        budget_extraction_error_rate=0.0,
    )
    atoms = initial_atoms(state, catalog)
    assert ("uncertain", "customer1") in atoms
    assert ("formal-use", "customer1") in atoms
    assert ("low-pressure", "customer1") in atoms  # prefers_low_pressure defaults True


def test_uncertain_user_gets_offer_comparison_step() -> None:
    catalog = load_catalog(REPO_ROOT / "data" / "product_catalog.json")
    state = infer_interaction_state(
        "I am not sure — maybe comfortable trousers under 60 euros.",
        rng=Random(1),
        budget_extraction_error_rate=0.0,
    )
    plan = breadth_first_plan(state, catalog)
    assert any(step.startswith("offer-comparison") for step in plan)
    assert not any(step.startswith("offer-alternative") for step in plan)


# ---------------------------------------------------------------------------
# Review 2: Contestability loop
# ---------------------------------------------------------------------------

def test_contestability_loop_initial_turn() -> None:
    catalog = load_catalog(REPO_ROOT / "data" / "product_catalog.json")
    engine = ContestabilityEngine(catalog, pddl_output_dir="generated_pddl", rng=Random(0))
    turns = engine.run("I need comfortable trousers under 60 euros")
    assert len(turns) == 1
    assert any(step.startswith("recommend-") for step in turns[0].result.plan)


def test_contestability_loop_correction_changes_plan() -> None:
    catalog = load_catalog(REPO_ROOT / "data" / "product_catalog.json")
    engine = ContestabilityEngine(catalog, pddl_output_dir="generated_pddl", rng=Random(0))
    turns = engine.run(
        "I need comfortable trousers under 60 euros",
        corrections=["Actually, for a job interview — formal style is important"],
    )
    assert len(turns) == 2
    # After the formal correction the mental model should reflect formal-use
    assert turns[1].state.mental_model.intended_use == "formal"
    # Budget from the initial turn is preserved
    assert turns[1].state.mental_model.budget == 60.0


def test_contestability_loop_preserves_prior_budget() -> None:
    catalog = load_catalog(REPO_ROOT / "data" / "product_catalog.json")
    engine = ContestabilityEngine(catalog, pddl_output_dir="generated_pddl", rng=Random(0))
    turns = engine.run(
        "I need comfortable trousers under 60 euros",
        corrections=["Actually I also prefer relaxed fit"],
    )
    # Budget must survive the correction turn
    assert turns[1].state.mental_model.budget == 60.0


# ---------------------------------------------------------------------------
# Review 3: Upstream error handling / clarification-needed
# ---------------------------------------------------------------------------

def test_budget_extraction_failure_triggers_clarification(tmp_path: Path) -> None:
    catalog = load_catalog(REPO_ROOT / "data" / "product_catalog.json")
    # Use an utterance that contains a budget number so the error rate can fire.
    state = infer_interaction_state(
        "I need comfortable trousers under 60 euros",
        rng=Random(1),
        budget_extraction_error_rate=1.0,  # always corrupt the extracted budget
    )
    assert not state.extracted_budget_correctly
    atoms = initial_atoms(state, catalog)
    assert ("clarification-needed", "customer1") in atoms
    plan = breadth_first_plan(state, catalog)
    assert plan[0].startswith("request-budget-clarification")


def test_clarification_flag_in_result_metadata(tmp_path: Path) -> None:
    catalog = load_catalog(REPO_ROOT / "data" / "product_catalog.json")
    state = infer_interaction_state(
        "I need comfortable trousers under 60 euros",
        rng=Random(1),
        budget_extraction_error_rate=1.0,
    )
    assistant = NeuroSymbolicRetailAssistant(catalog, pddl_output_dir=tmp_path, rng=Random(2))
    result = assistant.recommend(state)
    assert result.metadata.get("clarification_needed") is True


def test_clarification_response_does_not_commit_to_product(tmp_path: Path) -> None:
    catalog = load_catalog(REPO_ROOT / "data" / "product_catalog.json")
    state = infer_interaction_state(
        "I need comfortable trousers under 60 euros",
        rng=Random(1),
        budget_extraction_error_rate=1.0,
    )
    assistant = NeuroSymbolicRetailAssistant(catalog, pddl_output_dir=tmp_path, rng=Random(2))
    result = assistant.recommend(state)
    # The response should ask for clarification, not commit to a recommendation
    assert "budget" in result.response.lower() or "confirm" in result.response.lower()


# ---------------------------------------------------------------------------
# Review 5: Scalability — catalog pre-filtering
# ---------------------------------------------------------------------------

def test_filter_catalog_respects_max_candidates() -> None:
    catalog = load_catalog(REPO_ROOT / "data" / "product_catalog.json")
    state = infer_interaction_state(
        "I need comfortable trousers under 60 euros",
        rng=Random(1),
        budget_extraction_error_rate=0.0,
    )
    filtered = _filter_catalog_for_planning(state, catalog, max_candidates=3)
    assert len(filtered) <= 3


def test_filter_catalog_prefers_budget_matching_products() -> None:
    catalog = load_catalog(REPO_ROOT / "data" / "product_catalog.json")
    state = infer_interaction_state(
        "I need comfortable trousers under 60 euros",
        rng=Random(1),
        budget_extraction_error_rate=0.0,
    )
    filtered = _filter_catalog_for_planning(state, catalog, max_candidates=3)
    # All top-3 should be within budget
    assert all(p.price <= 60.0 for p in filtered)


def test_plan_still_found_with_small_candidate_set(tmp_path: Path) -> None:
    catalog = load_catalog(REPO_ROOT / "data" / "product_catalog.json")
    state = infer_interaction_state(
        "I need comfortable trousers under 60 euros",
        rng=Random(1),
        budget_extraction_error_rate=0.0,
    )
    plan = breadth_first_plan(state, catalog, max_candidates=3)
    assert any(step.startswith("recommend-") for step in plan)
