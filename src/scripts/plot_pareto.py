#!/usr/bin/env uv run
"""Generate Pareto frontier plot of RAM vs accuracy from experiment results.

Usage:
    uv run src/scripts/plot_pareto.py [options]

Options:
    --input PATH       Path to JSONL results file (default: data/experiments/results.jsonl)
    --output PATH      Path to output plot (default: results/pareto_frontier.png)
    --show             Display plot interactively (requires GUI)
    --highlight ID     Highlight specific model IDs (can be repeated)

Example:
    uv run src/scripts/plot_pareto.py --highlight phase4b-small-kd --highlight phase4b-small-direct
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

from tiny_language_detection.experiments import ExperimentResult, load_results


def kb_to_readable(x: float, _pos: int = 0) -> str:
    """Format KB value as KB or MB for readability."""
    if x >= 1024:
        return f"{x / 1024:.1f} MB"
    return f"{x:.0f} KB"


def compute_pareto_frontier(results: list[ExperimentResult]) -> list[ExperimentResult]:
    """Compute Pareto-optimal models (best accuracy for given RAM or less).

    A model is Pareto-optimal if no other model has both:
    - Lower or equal RAM usage
    - Higher or equal accuracy

    Returns:
        List of Pareto-optimal models, sorted by RAM ascending
    """
    # Sort by RAM ascending
    sorted_results = sorted(results, key=lambda r: r.ram_kb)

    pareto_optimal = []
    max_accuracy = -float("inf")

    for result in sorted_results:
        # If this model has better accuracy than all models with <= RAM, it's Pareto-optimal
        if result.accuracy > max_accuracy:
            pareto_optimal.append(result)
            max_accuracy = result.accuracy

    return pareto_optimal


def create_plot(
    results: list[ExperimentResult],
    output_path: Path,
    highlight_ids: list[str] | None = None,
    show: bool = False,
) -> None:
    """Create Pareto frontier scatter plot.

    Args:
        results: List of experiment results
        output_path: Path to save plot
        highlight_ids: Model IDs to highlight with labels
        show: Whether to display interactively
    """
    # Set style
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(12, 8))

    # Extract data
    ram_values = [r.ram_kb for r in results]
    [r.accuracy for r in results]

    # Compute Pareto frontier
    pareto = compute_pareto_frontier(results)
    pareto_ram = [r.ram_kb for r in pareto]
    pareto_acc = [r.accuracy for r in pareto]

    # Color by architecture
    architectures = list(set(r.architecture for r in results))
    colors = plt.cm.tab10(np.linspace(0, 1, len(architectures)))
    arch_to_color = dict(zip(architectures, colors))

    # Plot each architecture with different marker
    markers = ["o", "s", "^", "D", "v", "<", ">", "p", "h", "*"]
    arch_to_marker = dict(zip(architectures, markers))

    for arch in architectures:
        arch_results = [r for r in results if r.architecture == arch]
        ax.scatter(
            [r.ram_kb for r in arch_results],
            [r.accuracy for r in arch_results],
            c=[arch_to_color[arch]],
            marker=arch_to_marker.get(arch, "o"),
            s=120,
            label=arch,
            alpha=0.7,
            edgecolors="black",
            linewidth=0.8,
        )

    # Plot Pareto frontier
    if len(pareto) > 1:
        ax.step(
            pareto_ram,
            pareto_acc,
            where="post",
            color="red",
            linewidth=2.5,
            label="Pareto Frontier",
            alpha=0.8,
        )
        # Mark Pareto-optimal points
        ax.scatter(
            pareto_ram,
            pareto_acc,
            s=200,
            c="red",
            marker="*",
            label="Pareto-Optimal Models",
            zorder=5,
            edgecolors="darkred",
            linewidth=1.5,
        )

    # Highlight specific models
    if highlight_ids:
        for result in results:
            if result.model_id in highlight_ids:
                ax.annotate(
                    result.model_name,
                    xy=(result.ram_kb, result.accuracy),
                    xytext=(10, 10),
                    textcoords="offset points",
                    fontsize=10,
                    fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.5", fc="yellow", alpha=0.8),
                    arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0"),
                )

    # Add model labels (avoid clutter by only labeling high-accuracy models)
    for result in results:
        if result.accuracy >= 90 and result.model_id not in (highlight_ids or []):
            ax.annotate(
                result.model_id,
                xy=(result.ram_kb, result.accuracy),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
                alpha=0.7,
            )

    # Formatting
    ax.set_xlabel("Runtime RAM Usage", fontsize=12, fontweight="bold")
    ax.set_ylabel("Accuracy (%)", fontsize=12, fontweight="bold")
    ax.set_title(
        "Model Selection: RAM vs Accuracy\n(YODAS-Granary Dataset)",
        fontsize=14,
        fontweight="bold",
    )

    # Custom x-axis formatter
    ax.xaxis.set_major_formatter(FuncFormatter(kb_to_readable))

    # Set axis limits with padding
    x_min = min(ram_values) * 0.8
    x_max = max(ram_values) * 1.1
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(50, 100)  # Accuracy range 50-100%

    # Add legend
    ax.legend(loc="lower right", fontsize=9, framealpha=0.9)

    # Add grid
    ax.grid(True, alpha=0.3, linestyle="--")

    # Add target regions
    ax.axvspan(0, 1024, alpha=0.1, color="green", label="Earbud Target (<1 MB)")
    ax.axvspan(1024, 2048, alpha=0.1, color="yellow", label="Headphone Target (1-2 MB)")

    # Tight layout
    plt.tight_layout()

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"✓ Plot saved to: {output_path}")

    if show:
        plt.show()
    else:
        plt.close()


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/experiments/results.jsonl"),
        help="Path to JSONL results file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/pareto_frontier.png"),
        help="Path to output plot",
    )
    parser.add_argument(
        "--show", action="store_true", help="Display plot interactively"
    )
    parser.add_argument(
        "--highlight",
        type=str,
        action="append",
        dest="highlight_ids",
        help="Model IDs to highlight (can be repeated)",
    )

    args = parser.parse_args()

    # Load results
    if not args.input.exists():
        print(f"✗ Results file not found: {args.input}")
        print("  Run experiments first to generate results.")
        return 1

    results = load_results(args.input)
    print(f"Loaded {len(results)} experiment results from {args.input}")

    # Show summary
    print("\nExperiments loaded:")
    if results:
        print(
            f"  RAM range: {min(r.ram_kb for r in results):.0f} KB - {max(r.ram_kb for r in results):.0f} KB"
        )
        print(
            f"  Accuracy range: {min(r.accuracy for r in results):.1f}% - {max(r.accuracy for r in results):.1f}%"
        )
        print(f"  Architectures: {', '.join(set(r.architecture for r in results))}")
    else:
        print("  No results to display.")

    # Compute Pareto
    pareto = compute_pareto_frontier(results)
    print(f"\nPareto-optimal models ({len(pareto)}):")
    for r in pareto:
        print(f"  {r.model_name}: {r.accuracy:.1f}% @ {r.ram_kb:.0f} KB")

    # Create plot
    create_plot(results, args.output, args.highlight_ids, args.show)

    return 0


if __name__ == "__main__":
    sys.exit(main())
