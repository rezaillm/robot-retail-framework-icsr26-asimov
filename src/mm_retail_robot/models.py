"""Core dataclasses for the retail robot prototype.

The module keeps the representation intentionally compact so that the
implementation can be understood and reproduced in a workshop paper.  The
symbolic state is later compiled into a small PDDL problem.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set


@dataclass(frozen=True)
class Product:
    """A product in the retail catalogue."""

    id: str
    name: str
    price: float
    style: str
    comfort: int
    available: bool
    premium: bool
    promotion: bool


@dataclass
class MentalModel:
    """A compact symbolic approximation of the customer's mental model.

    Economic constraints
    --------------------
    budget               : hard upper limit stated by the user (euros).
    budget_sensitivity   : True when price is an explicit concern.
    budget_flexibility   : fraction above the stated budget the user may
                           accept, e.g. 0.10 means up to +10 %.  Extracted
                           from hedging language ("a bit over", "flexible").
                           When non-zero the PDDL layer adds a ``near-budget``
                           predicate and enables ``recommend-flex-budget-trousers``.

    Non-economic / interaction constraints
    --------------------------------------
    comfort_priority     : comfort outweighs style or price.
    intended_use         : "everyday", "formal", "travel", "smart-casual".
                           Compiled to the ``formal-use`` PDDL predicate when
                           the value is "formal".
    uncertain            : user expressed ambiguity ("not sure", "maybe").
                           Triggers a ``offer-comparison`` plan step instead of
                           the default ``offer-alternative``.
    upsell_rejected      : user has explicitly declined premium products.
    prefers_low_pressure : user prefers minimal sales pressure.  Compiled to
                           the ``low-pressure`` PDDL predicate; influences how
                           alternatives are phrased.
    """

    budget: Optional[float] = None
    budget_sensitive: bool = False
    budget_flexibility: float = 0.0
    comfort_priority: bool = False
    intended_use: str = "everyday"
    uncertain: bool = False
    upsell_rejected: bool = False
    prefers_low_pressure: bool = True


@dataclass
class InteractionState:
    """Verified interaction state used by the symbolic layer."""

    user_id: str = "customer1"
    mental_model: MentalModel = field(default_factory=MentalModel)
    viewed_products: Set[str] = field(default_factory=set)
    rejected_products: Set[str] = field(default_factory=set)
    commercial_disclosures: Set[str] = field(default_factory=set)
    extracted_budget_correctly: bool = True
    utterance: str = ""


@dataclass
class RecommendationResult:
    """Output of either the LLM-only or neuro-symbolic policy."""

    condition: str
    product: Product
    response: str
    explanation: str
    plan: List[str] = field(default_factory=list)
    pddl_domain_path: Optional[str] = None
    pddl_problem_path: Optional[str] = None
    metadata: Dict[str, object] = field(default_factory=dict)
