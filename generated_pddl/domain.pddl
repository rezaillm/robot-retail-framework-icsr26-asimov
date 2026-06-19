(define (domain retail-recommendation)
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
