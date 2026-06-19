#!/usr/bin/env python3
"""Run one LLM-only versus neuro-symbolic demonstration."""
from __future__ import annotations

import argparse
from pathlib import Path
from random import Random
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from mm_retail_robot.assistant import LLMOnlyRetailAssistant, NeuroSymbolicRetailAssistant
from mm_retail_robot.catalog import load_catalog
from mm_retail_robot.user_state import infer_interaction_state


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", default=str(REPO_ROOT / "data" / "product_catalog.json"))
    parser.add_argument(
        "--utterance",
        default="I need comfortable trousers for everyday wear, preferably under 60 euros. Please do not show premium options.",
    )
    parser.add_argument("--pddl-output-dir", default=str(REPO_ROOT / "generated_pddl"))
    args = parser.parse_args()

    catalog = load_catalog(args.catalog)
    state = infer_interaction_state(args.utterance, rng=Random(1), budget_extraction_error_rate=0.0)

    assistants = [
        LLMOnlyRetailAssistant(catalog, rng=Random(2)),
        NeuroSymbolicRetailAssistant(catalog, pddl_output_dir=args.pddl_output_dir, rng=Random(3)),
    ]

    print(f"USER: {args.utterance}\n")
    for assistant in assistants:
        result = assistant.recommend(state)
        print("=" * 80)
        print(result.condition.upper())
        print(f"Product: {result.product.name} ({result.product.price:.2f} EUR)")
        if result.plan:
            print("Symbolic plan:")
            for idx, step in enumerate(result.plan, start=1):
                print(f"  {idx}. {step}")
        print(f"Response: {result.response}")
        if result.pddl_problem_path:
            print(f"PDDL problem: {result.pddl_problem_path}")


if __name__ == "__main__":
    main()
