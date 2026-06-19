#!/usr/bin/env python3
"""
Run simulated prototype diagnostics and plot prototype-level results with publication-readable Seaborn styling.

This script generates:
1. rate_metrics_noisy_pddl.pdf
2. mean_price_noisy_pddl.pdf
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from mm_retail_robot.catalog import load_catalog
from mm_retail_robot.evaluator import run_simulation, summarize_conditions


def configure_plot_style() -> None:
    """Configure Seaborn/Matplotlib style for paper-readable figures."""
    sns.set_theme(
        context="paper",
        style="whitegrid",
        font_scale=1.55,
        rc={
            "axes.labelsize": 16,
            "axes.titlesize": 16,
            "axes.labelweight": "bold",
            "axes.titleweight": "bold",
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "legend.fontsize": 12,
            "legend.title_fontsize": 12,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        },
    )


def save_rate_metrics(summary_csv: str, output_dir: str) -> None:
    """
    Plot diagnostic rate metrics as percentages.

    The figure size is chosen to match the height of the mean-price figure,
    so both plots can be placed side by side as LaTeX subfigures.
    """
    configure_plot_style()

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(summary_csv)

    rate_columns = [
        "budget_violation_rate",
        "unavailable_violation_rate",
        "budget_explanation_rate",
        "contestability_explanation_rate",
    ]

    long_df = df.melt(
        id_vars="condition",
        value_vars=rate_columns,
        var_name="metric",
        value_name="value",
    )

    long_df["value"] = long_df["value"] * 100.0
    long_df["metric"] = long_df["metric"].replace(
        {
            "budget_violation_rate": "Budget\nviol.",
            "unavailable_violation_rate": "Unavailable\nitem",
            "budget_explanation_rate": "Budget\nexpl.",
            "contestability_explanation_rate": "Contestability\nexpl.",
        }
    )

    fig, ax = plt.subplots(figsize=(6.8, 4.6))

    sns.barplot(
        data=long_df,
        x="metric",
        y="value",
        hue="condition",
        errorbar=None,
        ax=ax,
    )

    ax.set_xlabel("Diagnostic metric", fontweight="bold", labelpad=10)
    ax.set_ylabel("Rate (%)", fontweight="bold", labelpad=10)
    ax.set_ylim(0, 110)

    ax.legend(
        title="Condition",
        frameon=True,
        loc="upper left",
        fontsize=10,
        title_fontsize=10,
    )

    for container in ax.containers:
        ax.bar_label(
            container,
            fmt="%.1f",
            fontsize=9,
            fontweight="bold",
            padding=3,
        )

    ax.tick_params(axis="both", width=1.4, length=5)
    
    for tick_label in ax.get_xticklabels():
        tick_label.set_fontweight("bold")
        tick_label.set_fontsize(10)

    for tick_label in ax.get_yticklabels():
        tick_label.set_fontweight("bold")
        tick_label.set_fontsize(11)

    sns.despine()
    fig.tight_layout()

    fig.savefig(
        output_path / "rate_metrics_noisy_pddl.pdf",
        bbox_inches="tight",
        pad_inches=0.03,
    )
    fig.savefig(
        output_path / "rate_metrics_noisy_pddl.png",
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.03,
    )
    plt.close(fig)


def save_mean_price(summary_csv: str, output_dir: str) -> None:
    """
    Plot mean recommended price separately from rate metrics.

    The figure height matches the rate-metrics figure so both plots can be
    placed side by side as LaTeX subfigures.
    """
    configure_plot_style()

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(summary_csv)

    fig, ax = plt.subplots(figsize=(4.4, 4.6))

    sns.barplot(
        data=df,
        x="condition",
        y="mean_price",
        errorbar=None,
        ax=ax,
    )

    ax.set_xlabel("Condition", fontweight="bold", labelpad=10)
    ax.set_ylabel("Mean price (€)", fontweight="bold", labelpad=10)

    upper_limit = max(df["mean_price"]) * 1.18
    ax.set_ylim(0, upper_limit)

    for container in ax.containers:
        ax.bar_label(
            container,
            fmt="%.2f",
            fontsize=10,
            fontweight="bold",
            padding=3,
        )

    ax.tick_params(axis="both", width=1.4, length=5)
    for tick_label in ax.get_xticklabels() + ax.get_yticklabels():
        tick_label.set_fontweight("bold")

    sns.despine()
    fig.tight_layout()

    fig.savefig(
        output_path / "mean_price_noisy_pddl.pdf",
        bbox_inches="tight",
        pad_inches=0.03,
    )
    fig.savefig(
        output_path / "mean_price_noisy_pddl.png",
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.03,
    )
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", default=str(REPO_ROOT / "data" / "product_catalog.json"))
    parser.add_argument("--n-users", type=int, default=200)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "results"))
    parser.add_argument("--pddl-output-dir", default=str(REPO_ROOT / "generated_pddl"))
    parser.add_argument(
        "--real-llm",
        action="store_true",
        help=(
            "Use a live Anthropic API call for the LLM-only condition "
            "(requires ANTHROPIC_API_KEY). ~n-users API calls, < $0.10 with Haiku."
        ),
    )
    args = parser.parse_args()

    if args.real_llm:
        import os
        if not os.environ.get("ANTHROPIC_API_KEY"):
            parser.error("--real-llm requires ANTHROPIC_API_KEY to be set in the environment.")
        print(
            f"[real-llm] LLM-only condition will call the Anthropic API "
            f"({args.n_users} calls). Press Ctrl-C to abort."
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    catalog = load_catalog(args.catalog)
    trials = run_simulation(
        catalog,
        n_users=args.n_users,
        seed=args.seed,
        pddl_output_dir=args.pddl_output_dir,
        use_real_llm=args.real_llm,
    )
    summary = summarize_conditions(trials)

    trials.to_csv(output_dir / "simulated_trials.csv", index=False)
    summary.to_csv(output_dir / "condition_summary.csv", index=False)

    print(summary.to_string(index=False))

    save_rate_metrics(
        summary_csv="results/condition_summary.csv",
        output_dir="results",
    )
    save_mean_price(
        summary_csv="results/condition_summary.csv",
        output_dir="results",
    )


if __name__ == "__main__":
    main()
