"""PDDL compilation and lightweight STRIPS planning.

The current prototype uses actual PDDL files as an intermediate symbolic
representation.  For reproducibility, this module also contains a very small
STRIPS planner tailored to the generated domain.  The generated domain and
problem files can be inspected, committed to GitHub, or replaced by calls to an
external planner in future work.

Scalability note
----------------
``breadth_first_plan`` grounds every action against the supplied catalogue.
With N products the action space is O(N), and worst-case BFS state space is
O(N^max_depth).  For realistic retail catalogues (hundreds or thousands of
SKUs) this is intractable at any non-trivial depth.

To bound the search, ``_filter_catalog_for_planning`` pre-selects the K most
relevant products before grounding (default K=20).  Products are scored by
budget fit, comfort match, style match, and upsell-rejection penalty so that
only the candidates most likely to appear in a valid plan enter the symbolic
layer.  The full catalogue remains available for display and for the LLM-only
baseline; only the planning catalogue is restricted.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, FrozenSet, Iterable, List, Sequence, Set, Tuple

from .models import InteractionState, Product


DOMAIN_TEXT = """(define (domain retail-recommendation)
  (:requirements :strips :typing :negative-preconditions)
  (:types customer product)
  (:predicates
    (customer ?u - customer)
    (product ?p - product)
    (trousers ?p - product)
    (available ?p - product)
    (premium-item ?p - product)

    ; --- Economic constraints ---
    (comfort-priority ?u - customer)
    (budget-sensitive ?u - customer)
    (under-budget ?p - product ?u - customer)
    (budget-flexible ?u - customer)
    (near-budget ?p - product ?u - customer)

    ; --- Ethical / interaction constraints ---
    (upsell-rejected ?u - customer)
    (low-pressure ?u - customer)
    (uncertain ?u - customer)
    (formal-use ?u - customer)
    (clarification-needed ?u - customer)
    (commercial-intent-disclosed ?p - product)

    ; --- Plan progress ---
    (recommended ?u - customer ?p - product)
    (recommendation-made ?u - customer)
    (explained ?u - customer ?p - product)
    (alternative-offered ?u - customer)
  )

  ; Request clarification when the budget could not be extracted from the
  ; user utterance.  All recommend-* actions guard on (not (clarification-needed
  ; ?u)) so the planner is forced to schedule this step first whenever the
  ; neural extraction layer failed.
  (:action request-budget-clarification
    :parameters (?u - customer)
    :precondition (and (customer ?u) (clarification-needed ?u))
    :effect (and (not (clarification-needed ?u)))
  )

  (:action disclose-commercial-intent
    :parameters (?u - customer ?p - product)
    :precondition (and (customer ?u) (product ?p) (premium-item ?p))
    :effect (and (commercial-intent-disclosed ?p))
  )

  (:action recommend-budget-trousers
    :parameters (?u - customer ?p - product)
    :precondition (and
      (customer ?u)
      (product ?p)
      (trousers ?p)
      (available ?p)
      (comfort-priority ?u)
      (under-budget ?p ?u)
      (not (clarification-needed ?u))
    )
    :effect (and (recommended ?u ?p) (recommendation-made ?u))
  )

  ; General budget path: fires when the user stated a budget but did not
  ; express a comfort or formal-use preference.  Lower priority than
  ; recommend-budget-trousers so the comfort-aware action is preferred whenever
  ; both are applicable, and it yields to recommend-formal-trousers when
  ; formal-use is set.
  (:action recommend-any-budget-trousers
    :parameters (?u - customer ?p - product)
    :precondition (and
      (customer ?u)
      (product ?p)
      (trousers ?p)
      (available ?p)
      (under-budget ?p ?u)
      (not (comfort-priority ?u))
      (not (formal-use ?u))
      (not (clarification-needed ?u))
    )
    :effect (and (recommended ?u ?p) (recommendation-made ?u))
  )

  ; Flex-budget path: fires when the user expressed willingness to go slightly
  ; over their stated budget (budget-flexible) and the product falls within
  ; budget * (1 + flexibility) (near-budget).
  (:action recommend-flex-budget-trousers
    :parameters (?u - customer ?p - product)
    :precondition (and
      (customer ?u)
      (product ?p)
      (trousers ?p)
      (available ?p)
      (budget-flexible ?u)
      (near-budget ?p ?u)
      (not (clarification-needed ?u))
    )
    :effect (and (recommended ?u ?p) (recommendation-made ?u))
  )

  ; Formal-use path: fires when intended_use == "formal", regardless of budget
  ; sensitivity.  Budget constraints still apply via clarification-needed guard.
  (:action recommend-formal-trousers
    :parameters (?u - customer ?p - product)
    :precondition (and
      (customer ?u)
      (product ?p)
      (trousers ?p)
      (available ?p)
      (formal-use ?u)
      (not (clarification-needed ?u))
    )
    :effect (and (recommended ?u ?p) (recommendation-made ?u))
  )

  (:action recommend-premium-trousers
    :parameters (?u - customer ?p - product)
    :precondition (and
      (customer ?u)
      (product ?p)
      (trousers ?p)
      (available ?p)
      (premium-item ?p)
      (not (budget-sensitive ?u))
      (not (upsell-rejected ?u))
      (commercial-intent-disclosed ?p)
      (not (clarification-needed ?u))
    )
    :effect (and (recommended ?u ?p) (recommendation-made ?u))
  )

  (:action explain-recommendation
    :parameters (?u - customer ?p - product)
    :precondition (and (recommended ?u ?p))
    :effect (and (explained ?u ?p))
  )

  ; Standard alternative offer for confident users.
  (:action offer-alternative
    :parameters (?u - customer)
    :precondition (and (recommendation-made ?u) (not (uncertain ?u)))
    :effect (and (alternative-offered ?u))
  )

  ; Side-by-side comparison offer for users who expressed uncertainty.  The
  ; low-pressure predicate is propagated to explanation generation so that the
  ; response avoids persuasive framing.
  (:action offer-comparison
    :parameters (?u - customer)
    :precondition (and (recommendation-made ?u) (uncertain ?u))
    :effect (and (alternative-offered ?u))
  )
)
"""

Atom = Tuple[str, ...]


@dataclass(frozen=True)
class GroundAction:
    """A grounded STRIPS action used by the lightweight planner."""

    name: str
    positive_preconditions: FrozenSet[Atom]
    negative_preconditions: FrozenSet[Atom]
    add_effects: FrozenSet[Atom]
    delete_effects: FrozenSet[Atom] = frozenset()

    def applicable(self, state: FrozenSet[Atom]) -> bool:
        """Return True iff the action is applicable in ``state``."""
        return self.positive_preconditions.issubset(state) and self.negative_preconditions.isdisjoint(state)

    def apply(self, state: FrozenSet[Atom]) -> FrozenSet[Atom]:
        """Apply the STRIPS action to a state."""
        return frozenset((set(state) - set(self.delete_effects)) | set(self.add_effects))


def product_object(product: Product) -> str:
    """Return a PDDL-safe object name."""
    return product.id.replace("-", "_")


def atom_to_pddl(atom: Atom) -> str:
    """Format an atom tuple as PDDL syntax."""
    return "(" + " ".join(atom) + ")"


def initial_atoms(state: InteractionState, catalog: Sequence[Product]) -> Set[Atom]:
    """Compile the current Python state into symbolic atoms.

    All MentalModel fields — economic and non-economic — are compiled to
    typed PDDL predicates so that the symbolic layer has a complete view of
    the customer's inferred preferences and interaction constraints.
    """
    user = state.user_id
    model = state.mental_model
    atoms: Set[Atom] = {("customer", user)}

    # Economic constraints
    if model.comfort_priority:
        atoms.add(("comfort-priority", user))
    if model.budget_sensitive:
        atoms.add(("budget-sensitive", user))
    if model.budget_flexibility > 0:
        atoms.add(("budget-flexible", user))

    # Ethical / interaction constraints
    if model.upsell_rejected:
        atoms.add(("upsell-rejected", user))
    if model.prefers_low_pressure:
        atoms.add(("low-pressure", user))
    if model.uncertain:
        atoms.add(("uncertain", user))
    if model.intended_use == "formal":
        atoms.add(("formal-use", user))

    # Upstream error guard: force a clarification step before any recommendation
    # when the neural extraction layer failed to parse the budget.
    if not state.extracted_budget_correctly:
        atoms.add(("clarification-needed", user))

    effective_budget = (
        model.budget * (1 + model.budget_flexibility) if model.budget is not None else None
    )

    for product in catalog:
        pid = product_object(product)
        atoms.update({("product", pid), ("trousers", pid)})
        if product.available:
            atoms.add(("available", pid))
        if product.premium:
            atoms.add(("premium-item", pid))
        if pid in state.commercial_disclosures:
            atoms.add(("commercial-intent-disclosed", pid))
        # Hard budget constraint
        if model.budget is not None and product.price <= model.budget:
            atoms.add(("under-budget", pid, user))
        # Soft budget constraint (within flexibility range)
        if effective_budget is not None and product.price <= effective_budget:
            atoms.add(("near-budget", pid, user))

    return atoms


def goal_atoms(user_id: str = "customer1") -> Set[Atom]:
    """Goal atoms for the prototype interaction plan."""
    return {("recommendation-made", user_id), ("alternative-offered", user_id)}


def write_pddl_problem(
    state: InteractionState,
    catalog: Sequence[Product],
    output_dir: str | Path,
    problem_name: str = "retail_problem",
) -> tuple[Path, Path]:
    """Write PDDL domain and problem files to disk.

    Returns
    -------
    tuple[Path, Path]
        Paths to ``domain.pddl`` and ``problem.pddl``.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    domain_path = out / "domain.pddl"
    problem_path = out / "problem.pddl"
    domain_path.write_text(DOMAIN_TEXT, encoding="utf-8")

    user = state.user_id
    products = " ".join(product_object(p) for p in catalog)
    init = "\n    ".join(sorted(atom_to_pddl(atom) for atom in initial_atoms(state, catalog)))
    goals = "\n      ".join(sorted(atom_to_pddl(atom) for atom in goal_atoms(user)))

    problem = f"""(define (problem {problem_name})
  (:domain retail-recommendation)
  (:objects
    {user} - customer
    {products} - product
  )
  (:init
    {init}
  )
  (:goal (and
      {goals}
  ))
)
"""
    problem_path.write_text(problem, encoding="utf-8")
    return domain_path, problem_path


def ground_actions(state: InteractionState, catalog: Sequence[Product]) -> List[GroundAction]:
    """Ground the generated retail PDDL domain for the finite catalogue."""
    user = state.user_id
    actions: List[GroundAction] = []

    # One clarification action per user (not per product).
    actions.append(
        GroundAction(
            name=f"request-budget-clarification {user}",
            positive_preconditions=frozenset({("customer", user), ("clarification-needed", user)}),
            negative_preconditions=frozenset(),
            add_effects=frozenset(),
            delete_effects=frozenset({("clarification-needed", user)}),
        )
    )

    for product in catalog:
        pid = product_object(product)

        actions.append(
            GroundAction(
                name=f"disclose-commercial-intent {user} {pid}",
                positive_preconditions=frozenset({("customer", user), ("product", pid), ("premium-item", pid)}),
                negative_preconditions=frozenset(),
                add_effects=frozenset({("commercial-intent-disclosed", pid)}),
            )
        )
        actions.append(
            GroundAction(
                name=f"recommend-budget-trousers {user} {pid}",
                positive_preconditions=frozenset({
                    ("customer", user),
                    ("product", pid),
                    ("trousers", pid),
                    ("available", pid),
                    ("comfort-priority", user),
                    ("under-budget", pid, user),
                }),
                negative_preconditions=frozenset({("clarification-needed", user)}),
                add_effects=frozenset({("recommended", user, pid), ("recommendation-made", user)}),
            )
        )
        actions.append(
            GroundAction(
                name=f"recommend-any-budget-trousers {user} {pid}",
                positive_preconditions=frozenset({
                    ("customer", user),
                    ("product", pid),
                    ("trousers", pid),
                    ("available", pid),
                    ("under-budget", pid, user),
                }),
                negative_preconditions=frozenset({
                    ("comfort-priority", user),
                    ("formal-use", user),
                    ("clarification-needed", user),
                }),
                add_effects=frozenset({("recommended", user, pid), ("recommendation-made", user)}),
            )
        )
        actions.append(
            GroundAction(
                name=f"recommend-flex-budget-trousers {user} {pid}",
                positive_preconditions=frozenset({
                    ("customer", user),
                    ("product", pid),
                    ("trousers", pid),
                    ("available", pid),
                    ("budget-flexible", user),
                    ("near-budget", pid, user),
                }),
                negative_preconditions=frozenset({("clarification-needed", user)}),
                add_effects=frozenset({("recommended", user, pid), ("recommendation-made", user)}),
            )
        )
        actions.append(
            GroundAction(
                name=f"recommend-formal-trousers {user} {pid}",
                positive_preconditions=frozenset({
                    ("customer", user),
                    ("product", pid),
                    ("trousers", pid),
                    ("available", pid),
                    ("formal-use", user),
                }),
                negative_preconditions=frozenset({("clarification-needed", user)}),
                add_effects=frozenset({("recommended", user, pid), ("recommendation-made", user)}),
            )
        )
        actions.append(
            GroundAction(
                name=f"recommend-premium-trousers {user} {pid}",
                positive_preconditions=frozenset({
                    ("customer", user),
                    ("product", pid),
                    ("trousers", pid),
                    ("available", pid),
                    ("premium-item", pid),
                    ("commercial-intent-disclosed", pid),
                }),
                negative_preconditions=frozenset({
                    ("budget-sensitive", user),
                    ("upsell-rejected", user),
                    ("clarification-needed", user),
                }),
                add_effects=frozenset({("recommended", user, pid), ("recommendation-made", user)}),
            )
        )
        actions.append(
            GroundAction(
                name=f"explain-recommendation {user} {pid}",
                positive_preconditions=frozenset({("recommended", user, pid)}),
                negative_preconditions=frozenset(),
                add_effects=frozenset({("explained", user, pid)}),
            )
        )

    actions.append(
        GroundAction(
            name=f"offer-alternative {user}",
            positive_preconditions=frozenset({("recommendation-made", user)}),
            negative_preconditions=frozenset({("uncertain", user)}),
            add_effects=frozenset({("alternative-offered", user)}),
        )
    )
    actions.append(
        GroundAction(
            name=f"offer-comparison {user}",
            positive_preconditions=frozenset({("recommendation-made", user), ("uncertain", user)}),
            negative_preconditions=frozenset(),
            add_effects=frozenset({("alternative-offered", user)}),
        )
    )
    return actions


# Priority order for action types in BFS.  Lower values are explored first,
# which shortens the average path to the goal and reduces states visited.
_ACTION_PREFIX_PRIORITY: Dict[str, int] = {
    "request-budget-clarification":   0,
    "recommend-budget-trousers":      1,   # comfort + budget
    "recommend-any-budget-trousers":  2,   # budget only (no comfort requirement)
    "recommend-flex-budget-trousers": 3,   # within flexibility range
    "recommend-formal-trousers":      4,   # formal context
    "recommend-premium-trousers":     5,
    "offer-comparison":               6,
    "offer-alternative":              6,
    "explain-recommendation":         7,
    "disclose-commercial-intent":     8,
}


def _action_priority(action: GroundAction) -> int:
    """Return the BFS exploration priority for a grounded action."""
    for prefix, priority in _ACTION_PREFIX_PRIORITY.items():
        if action.name.startswith(prefix):
            return priority
    return 9


def _filter_catalog_for_planning(
    state: InteractionState,
    catalog: Sequence[Product],
    max_candidates: int,
) -> List[Product]:
    """Pre-select the K most relevant products to bound planner branching factor.

    Grounding all N products produces O(N) actions per depth level.  For large
    retail catalogues this makes BFS intractable.  This function scores available
    products by how well they match the current mental model and returns only the
    top ``max_candidates``, keeping the symbolic search space manageable.

    The full catalogue is still used by the LLM-only baseline and for display.
    """
    model = state.mental_model
    effective_budget = (
        model.budget * (1 + model.budget_flexibility) if model.budget is not None else None
    )

    def score(p: Product) -> float:
        s = 0.0
        if effective_budget is not None:
            if p.price <= effective_budget:
                s += 10.0
            else:
                s -= 20.0
        if model.comfort_priority and p.comfort >= 4:
            s += 5.0
        if p.style == model.intended_use:
            s += 3.0
        if model.upsell_rejected and p.premium:
            s -= 15.0
        if p.premium and not model.budget_sensitive:
            s += 2.0
        return s

    available = [p for p in catalog if p.available]
    candidates = sorted(available, key=score, reverse=True)
    return candidates[:max_candidates]


def breadth_first_plan(
    state: InteractionState,
    catalog: Sequence[Product],
    max_depth: int = 5,
    max_candidates: int = 20,
) -> List[str]:
    """Find a short interaction plan with breadth-first search.

    For catalogues larger than ``max_candidates``, the search runs over a
    pre-filtered subset (see ``_filter_catalog_for_planning``) so that the
    action space remains bounded.  Actions are sorted by ``_action_priority``
    so that the most likely paths (budget-matching recommendations) are
    explored first, further reducing the number of states visited.
    """
    planning_catalog: Sequence[Product] = (
        _filter_catalog_for_planning(state, catalog, max_candidates)
        if len(catalog) > max_candidates
        else list(catalog)
    )

    init = frozenset(initial_atoms(state, planning_catalog))
    goals = goal_atoms(state.user_id)
    actions = sorted(ground_actions(state, planning_catalog), key=_action_priority)

    frontier: List[tuple[FrozenSet[Atom], List[str]]] = [(init, [])]
    visited = {init}

    while frontier:
        current_state, current_plan = frontier.pop(0)
        if goals.issubset(current_state):
            return current_plan
        if len(current_plan) >= max_depth:
            continue
        for action in actions:
            if action.applicable(current_state):
                next_state = action.apply(current_state)
                if next_state not in visited:
                    visited.add(next_state)
                    frontier.append((next_state, current_plan + [action.name]))
    return []


def recommended_product_from_plan(plan: Sequence[str], catalog: Sequence[Product]) -> Product | None:
    """Extract the recommended product from a generated plan."""
    product_by_id: Dict[str, Product] = {product_object(product): product for product in catalog}
    for step in plan:
        if step.startswith("recommend-"):
            pid = step.split()[-1]
            return product_by_id.get(pid)
    return None
