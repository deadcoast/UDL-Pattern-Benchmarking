#!/usr/bin/env python3
"""
Generate architecture diagrams for the UDL Rating Framework documentation.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch
import numpy as np


def create_high_level_architecture():
    """Create high-level architecture diagram."""

    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis("off")

    # Define colors
    input_color = "#E3F2FD"
    core_color = "#FFF3E0"
    output_color = "#E8F5E8"
    optional_color = "#F3E5F5"

    # Input Layer
    input_box = FancyBboxPatch(
        (0.5, 6.5),
        2,
        1,
        boxstyle="round,pad=0.1",
        facecolor=input_color,
        edgecolor="black",
        linewidth=1.5,
    )
    ax.add_patch(input_box)
    ax.text(
        1.5,
        7,
        "Input Processing\n• File Discovery\n• UDL Parsing\n• Tokenization",
        ha="center",
        va="center",
        fontsize=10,
        weight="bold",
    )

    # Core Engine
    core_box = FancyBboxPatch(
        (3.5, 4),
        3,
        3,
        boxstyle="round,pad=0.1",
        facecolor=core_color,
        edgecolor="black",
        linewidth=2,
    )
    ax.add_patch(core_box)
    ax.text(
        5,
        5.5,
        "Mathematical Engine",
        ha="center",
        va="center",
        fontsize=12,
        weight="bold",
    )

    # Individual metrics
    metrics = ["Consistency", "Completeness", "Expressiveness", "Structural\nCoherence"]
    metric_positions = [(4, 5.8), (6, 5.8), (4, 4.7), (6, 4.7)]

    for metric, pos in zip(metrics, metric_positions):
        metric_box = FancyBboxPatch(
            (pos[0] - 0.4, pos[1] - 0.2),
            0.8,
            0.4,
            boxstyle="round,pad=0.05",
            facecolor="white",
            edgecolor="gray",
        )
        ax.add_patch(metric_box)
        ax.text(pos[0], pos[1], metric, ha="center", va="center", fontsize=8)

    # Aggregation
    agg_box = FancyBboxPatch(
        (4.5, 4.2),
        1,
        0.3,
        boxstyle="round,pad=0.05",
        facecolor="lightblue",
        edgecolor="blue",
    )
    ax.add_patch(agg_box)
    ax.text(5, 4.35, "Aggregation", ha="center", va="center", fontsize=8, weight="bold")

    # CTM Layer (Optional)
    ctm_box = FancyBboxPatch(
        (7.5, 4.5),
        2,
        2,
        boxstyle="round,pad=0.1",
        facecolor=optional_color,
        edgecolor="purple",
        linewidth=1.5,
        linestyle="--",
    )
    ax.add_patch(ctm_box)
    ax.text(
        8.5,
        5.5,
        "CTM Model\n(Optional)\n• Neural Approximation\n• Fast Inference\n• Pattern Learning",
        ha="center",
        va="center",
        fontsize=10,
        weight="bold",
    )

    # Output Layer
    output_box = FancyBboxPatch(
        (3.5, 1.5),
        3,
        1.5,
        boxstyle="round,pad=0.1",
        facecolor=output_color,
        edgecolor="green",
        linewidth=1.5,
    )
    ax.add_patch(output_box)
    ax.text(
        5,
        2.25,
        "Output Generation\n• Quality Reports\n• Confidence Scores\n• Visualizations\n• Multiple Formats",
        ha="center",
        va="center",
        fontsize=10,
        weight="bold",
    )

    # Arrows
    # Input to Core
    arrow1 = ConnectionPatch(
        (1.5, 6.5),
        (5, 6.5),
        "data",
        "data",
        arrowstyle="->",
        shrinkA=5,
        shrinkB=5,
        mutation_scale=20,
        fc="black",
    )
    ax.add_patch(arrow1)

    # Core to Output
    arrow2 = ConnectionPatch(
        (5, 4),
        (5, 3),
        "data",
        "data",
        arrowstyle="->",
        shrinkA=5,
        shrinkB=5,
        mutation_scale=20,
        fc="black",
    )
    ax.add_patch(arrow2)

    # CTM connection (dashed)
    arrow3 = ConnectionPatch(
        (6.5, 5.5),
        (7.5, 5.5),
        "data",
        "data",
        arrowstyle="<->",
        shrinkA=5,
        shrinkB=5,
        mutation_scale=20,
        fc="purple",
        linestyle="--",
    )
    ax.add_patch(arrow3)

    # Title
    ax.text(
        5,
        7.5,
        "UDL Rating Framework - High-Level Architecture",
        ha="center",
        va="center",
        fontsize=16,
        weight="bold",
    )

    # Legend
    legend_elements = [
        patches.Patch(color=input_color, label="Input Processing"),
        patches.Patch(color=core_color, label="Core Engine"),
        patches.Patch(color=output_color, label="Output Generation"),
        patches.Patch(color=optional_color, label="Optional Components"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", bbox_to_anchor=(0.98, 0.98))

    plt.tight_layout()
    plt.savefig(
        "docs/_static/high_level_architecture.png", dpi=300, bbox_inches="tight"
    )
    plt.close()


def create_metric_computation_flow():
    """Create metric computation flow diagram."""

    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis("off")

    # UDL Input
    udl_box = FancyBboxPatch(
        (1, 7),
        2,
        0.8,
        boxstyle="round,pad=0.1",
        facecolor="lightblue",
        edgecolor="blue",
    )
    ax.add_patch(udl_box)
    ax.text(2, 7.4, "UDL Input", ha="center", va="center", fontsize=12, weight="bold")

    # Representation
    repr_box = FancyBboxPatch(
        (4, 7),
        2,
        0.8,
        boxstyle="round,pad=0.1",
        facecolor="lightgreen",
        edgecolor="green",
    )
    ax.add_patch(repr_box)
    ax.text(
        5,
        7.4,
        "UDL Representation\nT, G, S, R",
        ha="center",
        va="center",
        fontsize=10,
        weight="bold",
    )

    # Individual Metrics
    metrics_data = [
        ("Consistency", (1, 5.5), "Cycles &\nContradictions"),
        ("Completeness", (3.5, 5.5), "Construct\nCoverage"),
        ("Expressiveness", (6, 5.5), "Chomsky Level &\nComplexity"),
        ("Structural\nCoherence", (8.5, 5.5), "Shannon\nEntropy"),
    ]

    metric_scores = []
    for name, pos, desc in metrics_data:
        # Metric box
        metric_box = FancyBboxPatch(
            (pos[0] - 0.6, pos[1] - 0.4),
            1.2,
            0.8,
            boxstyle="round,pad=0.1",
            facecolor="lightyellow",
            edgecolor="orange",
        )
        ax.add_patch(metric_box)
        ax.text(
            pos[0], pos[1], name, ha="center", va="center", fontsize=9, weight="bold"
        )

        # Description
        ax.text(
            pos[0],
            pos[1] - 0.8,
            desc,
            ha="center",
            va="center",
            fontsize=8,
            style="italic",
        )

        # Score box
        score_box = FancyBboxPatch(
            (pos[0] - 0.3, pos[1] - 1.8),
            0.6,
            0.4,
            boxstyle="round,pad=0.05",
            facecolor="white",
            edgecolor="gray",
        )
        ax.add_patch(score_box)
        ax.text(
            pos[0],
            pos[1] - 1.6,
            f"m{len(metric_scores) + 1} ∈ [0,1]",
            ha="center",
            va="center",
            fontsize=8,
        )
        metric_scores.append((pos[0], pos[1] - 1.8))

        # Arrow from representation to metric
        arrow = ConnectionPatch(
            (5, 6.6),
            pos,
            "data",
            "data",
            arrowstyle="->",
            shrinkA=5,
            shrinkB=5,
            mutation_scale=15,
            fc="black",
        )
        ax.add_patch(arrow)

    # Aggregation
    agg_box = FancyBboxPatch(
        (4, 2.5),
        2,
        0.8,
        boxstyle="round,pad=0.1",
        facecolor="lightcoral",
        edgecolor="red",
    )
    ax.add_patch(agg_box)
    ax.text(
        5,
        2.9,
        "Weighted Aggregation\nQ = Σ wᵢ·mᵢ",
        ha="center",
        va="center",
        fontsize=10,
        weight="bold",
    )

    # Arrows from metrics to aggregation
    for score_pos in metric_scores:
        arrow = ConnectionPatch(
            score_pos,
            (5, 3.3),
            "data",
            "data",
            arrowstyle="->",
            shrinkA=5,
            shrinkB=5,
            mutation_scale=15,
            fc="red",
        )
        ax.add_patch(arrow)

    # Final Score
    final_box = FancyBboxPatch(
        (4, 1),
        2,
        0.8,
        boxstyle="round,pad=0.1",
        facecolor="lightgreen",
        edgecolor="green",
    )
    ax.add_patch(final_box)
    ax.text(
        5,
        1.4,
        "Overall Quality Score\nQ ∈ [0,1]",
        ha="center",
        va="center",
        fontsize=11,
        weight="bold",
    )

    # Arrow to final score
    arrow = ConnectionPatch(
        (5, 2.5),
        (5, 1.8),
        "data",
        "data",
        arrowstyle="->",
        shrinkA=5,
        shrinkB=5,
        mutation_scale=20,
        fc="green",
    )
    ax.add_patch(arrow)

    # Title
    ax.text(
        5,
        7.8,
        "Metric Computation Flow",
        ha="center",
        va="center",
        fontsize=16,
        weight="bold",
    )

    plt.tight_layout()
    plt.savefig(
        "docs/_static/metric_computation_flow.png", dpi=300, bbox_inches="tight"
    )
    plt.close()


def create_ctm_architecture():
    """Create CTM model architecture diagram."""

    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    ax.axis("off")

    # Input tokens
    tokens_box = FancyBboxPatch(
        (0.5, 6.5),
        2,
        1,
        boxstyle="round,pad=0.1",
        facecolor="lightblue",
        edgecolor="blue",
    )
    ax.add_patch(tokens_box)
    ax.text(
        1.5,
        7,
        "UDL Tokens\n[t₁, t₂, ..., tₙ]",
        ha="center",
        va="center",
        fontsize=10,
        weight="bold",
    )

    # Embedding layer
    embed_box = FancyBboxPatch(
        (3.5, 6.5),
        2,
        1,
        boxstyle="round,pad=0.1",
        facecolor="lightyellow",
        edgecolor="orange",
    )
    ax.add_patch(embed_box)
    ax.text(
        4.5,
        7,
        "Token Embedding\nE: Token → ℝᵈ",
        ha="center",
        va="center",
        fontsize=10,
        weight="bold",
    )

    # CTM Core
    ctm_box = FancyBboxPatch(
        (6.5, 4.5),
        3,
        3,
        boxstyle="round,pad=0.1",
        facecolor="lightcoral",
        edgecolor="red",
        linewidth=2,
    )
    ax.add_patch(ctm_box)
    ax.text(
        8,
        6,
        "Continuous Thought Machine",
        ha="center",
        va="center",
        fontsize=12,
        weight="bold",
    )

    # CTM internals
    ax.text(8, 5.5, "• T iterations", ha="center", va="center", fontsize=9)
    ax.text(8, 5.2, "• Neuron dynamics", ha="center", va="center", fontsize=9)
    ax.text(8, 4.9, "• Synchronization S(t)", ha="center", va="center", fontsize=9)

    # Rating head
    rating_box = FancyBboxPatch(
        (6.5, 2.5),
        3,
        1,
        boxstyle="round,pad=0.1",
        facecolor="lightgreen",
        edgecolor="green",
    )
    ax.add_patch(rating_box)
    ax.text(
        8,
        3,
        "Rating Head\nS(T) → [0,1]",
        ha="center",
        va="center",
        fontsize=10,
        weight="bold",
    )

    # Output
    output_box = FancyBboxPatch(
        (6.5, 0.5),
        3,
        1,
        boxstyle="round,pad=0.1",
        facecolor="lavender",
        edgecolor="purple",
    )
    ax.add_patch(output_box)
    ax.text(
        8,
        1,
        "Quality Score\n& Confidence",
        ha="center",
        va="center",
        fontsize=10,
        weight="bold",
    )

    # Arrows
    arrows = [
        ((2.5, 7), (3.5, 7)),
        ((5.5, 7), (6.5, 6.5)),
        ((8, 4.5), (8, 3.5)),
        ((8, 2.5), (8, 1.5)),
    ]

    for start, end in arrows:
        arrow = ConnectionPatch(
            start,
            end,
            "data",
            "data",
            arrowstyle="->",
            shrinkA=5,
            shrinkB=5,
            mutation_scale=20,
            fc="black",
        )
        ax.add_patch(arrow)

    # Training feedback (dashed)
    feedback_arrow = ConnectionPatch(
        (6.5, 1),
        (3, 4),
        "data",
        "data",
        arrowstyle="->",
        shrinkA=5,
        shrinkB=5,
        mutation_scale=15,
        fc="red",
        linestyle="--",
    )
    ax.add_patch(feedback_arrow)
    ax.text(
        4,
        2.5,
        "Training\nFeedback",
        ha="center",
        va="center",
        fontsize=9,
        style="italic",
        color="red",
    )

    # Mathematical ground truth
    math_box = FancyBboxPatch(
        (0.5, 3), 2.5, 2, boxstyle="round,pad=0.1", facecolor="wheat", edgecolor="brown"
    )
    ax.add_patch(math_box)
    ax.text(
        1.75,
        4,
        "Mathematical\nGround Truth\n• Consistency\n• Completeness\n• Expressiveness\n• Coherence",
        ha="center",
        va="center",
        fontsize=9,
        weight="bold",
    )

    # Title
    ax.text(
        6,
        7.5,
        "CTM Model Architecture for UDL Rating",
        ha="center",
        va="center",
        fontsize=16,
        weight="bold",
    )

    plt.tight_layout()
    plt.savefig("docs/_static/ctm_architecture.png", dpi=300, bbox_inches="tight")
    plt.close()


def create_all_diagrams():
    """Create all architecture diagrams."""

    # Create static directory if it doesn't exist
    import os

    os.makedirs("docs/_static", exist_ok=True)

    print("Creating high-level architecture diagram...")
    create_high_level_architecture()

    print("Creating metric computation flow diagram...")
    create_metric_computation_flow()

    print("Creating CTM architecture diagram...")
    create_ctm_architecture()

    print("All diagrams created successfully!")


if __name__ == "__main__":
    create_all_diagrams()
