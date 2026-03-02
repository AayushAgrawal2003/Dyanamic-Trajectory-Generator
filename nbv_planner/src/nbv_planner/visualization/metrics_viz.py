"""Metrics visualization using matplotlib."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from nbv_planner.metrics.evaluation import MethodEvaluation


def plot_coverage_vs_views(
    evaluations: list[MethodEvaluation],
    save_path: str | None = None,
) -> None:
    """Plot coverage percentage vs. number of views for all methods."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for ev in evaluations:
        views = np.arange(1, len(ev.coverage_curve) + 1)
        ax.plot(views, ev.coverage_curve * 100, marker="o", markersize=3,
                label=ev.method_name, linewidth=2)

    ax.set_xlabel("Number of Views", fontsize=12)
    ax.set_ylabel("Coverage (%)", fontsize=12)
    ax.set_title("Coverage vs. Number of Views", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 105)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved coverage plot to {save_path}")
    plt.show()


def plot_new_points_per_frame(
    evaluations: list[MethodEvaluation],
    save_path: str | None = None,
) -> None:
    """Plot new unique points per frame for each method."""
    fig, axes = plt.subplots(
        1, len(evaluations), figsize=(5 * len(evaluations), 5), sharey=True
    )
    if len(evaluations) == 1:
        axes = [axes]

    for ax, ev in zip(axes, evaluations):
        frames = np.arange(1, len(ev.new_points_curve) + 1)
        ax.bar(frames, ev.new_points_curve, alpha=0.7, color="steelblue")
        ax.set_xlabel("Frame", fontsize=11)
        ax.set_title(ev.method_name, fontsize=12)
        ax.grid(True, alpha=0.3, axis="y")

    axes[0].set_ylabel("New Unique Points", fontsize=11)
    fig.suptitle("New Unique Points per Frame", fontsize=14, y=1.02)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved new-points plot to {save_path}")
    plt.show()


def plot_density_per_frame(
    evaluations: list[MethodEvaluation],
    save_path: str | None = None,
) -> None:
    """Plot total points per frame over time."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for ev in evaluations:
        frames = np.arange(1, len(ev.density_curve) + 1)
        ax.plot(frames, ev.density_curve, marker="s", markersize=3,
                label=ev.method_name, linewidth=2)

    ax.set_xlabel("Frame", fontsize=12)
    ax.set_ylabel("Points in Frame", fontsize=12)
    ax.set_title("Points per Frame over Time", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved density plot to {save_path}")
    plt.show()


def plot_coverage_vs_trajectory(
    evaluations: list[MethodEvaluation],
    save_path: str | None = None,
) -> None:
    """Plot coverage vs. cumulative trajectory length."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for ev in evaluations:
        ax.bar(ev.method_name, ev.total_coverage * 100, alpha=0.7)
        ax.text(
            ev.method_name, ev.total_coverage * 100 + 1,
            f"traj: {ev.trajectory_length:.2f}m",
            ha="center", fontsize=9,
        )

    ax.set_ylabel("Final Coverage (%)", fontsize=12)
    ax.set_title("Coverage and Trajectory Length Comparison", fontsize=14)
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(0, 110)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved efficiency plot to {save_path}")
    plt.show()


def generate_all_plots(
    evaluations: list[MethodEvaluation],
    output_dir: str = "results/plots",
) -> None:
    """Generate all comparison plots and save to output directory."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    plot_coverage_vs_views(
        evaluations, str(output_path / "coverage_vs_views.png")
    )
    plot_new_points_per_frame(
        evaluations, str(output_path / "new_points_per_frame.png")
    )
    plot_density_per_frame(
        evaluations, str(output_path / "density_per_frame.png")
    )
    plot_coverage_vs_trajectory(
        evaluations, str(output_path / "coverage_vs_trajectory.png")
    )
    print(f"\nAll plots saved to {output_path}")
