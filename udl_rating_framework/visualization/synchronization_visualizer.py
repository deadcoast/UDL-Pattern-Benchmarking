"""
Synchronization Evolution Visualization Utilities.

Provides tools for visualizing synchronization matrix evolution over time.
"""

from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from ..models.ctm_adapter import TrackingData


class SynchronizationVisualizer:
    """
    Visualizer for CTM synchronization evolution.

    Provides methods to create various visualizations of synchronization
    matrices and their evolution over time.
    """

    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        """
        Initialize visualizer.

        Args:
            figsize: Default figure size for plots
        """
        self.figsize = figsize
        plt.style.use(
            "seaborn-v0_8" if "seaborn-v0_8" in plt.style.available else "default"
        )

    def plot_synchronization_evolution(
        self,
        tracking_data: TrackingData,
        synch_type: str = "out",
        batch_idx: int = 0,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Plot evolution of synchronization values over time.

        Args:
            tracking_data: TrackingData object with recorded synchronization
            synch_type: 'out' or 'action' synchronization
            batch_idx: Which batch sample to visualize
            save_path: Optional path to save the figure

        Returns:
            matplotlib Figure object
        """
        if synch_type == "out":
            synch_data = tracking_data.synch_out[
                :, batch_idx, :
            ]  # [iterations, synch_dim]
        elif synch_type == "action":
            synch_data = tracking_data.synch_action[
                :, batch_idx, :
            ]  # [iterations, synch_dim]
        else:
            raise ValueError("synch_type must be 'out' or 'action'")

        fig, ax = plt.subplots(figsize=self.figsize)

        # Create heatmap
        im = ax.imshow(
            synch_data.T, aspect="auto", cmap="viridis", interpolation="nearest"
        )

        # Customize plot
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Synchronization Dimension")
        ax.set_title(
            f"{synch_type.capitalize()} Synchronization Evolution (Batch {batch_idx})"
        )

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Synchronization Value")

        # Set ticks
        ax.set_xticks(
            range(0, tracking_data.iterations, max(
                1, tracking_data.iterations // 10))
        )
        synch_dim = synch_data.shape[1]
        ax.set_yticks(range(0, synch_dim, max(1, synch_dim // 10)))

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig

    def plot_synchronization_time_series(
        self,
        tracking_data: TrackingData,
        synch_indices: Optional[List[int]] = None,
        synch_type: str = "out",
        batch_idx: int = 0,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Plot synchronization time series for selected dimensions.

        Args:
            tracking_data: TrackingData object with recorded synchronization
            synch_indices: List of synchronization indices to plot (default: first 5)
            synch_type: 'out' or 'action' synchronization
            batch_idx: Which batch sample to visualize
            save_path: Optional path to save the figure

        Returns:
            matplotlib Figure object
        """
        if synch_type == "out":
            synch_data = tracking_data.synch_out[
                :, batch_idx, :
            ]  # [iterations, synch_dim]
            max_dim = tracking_data.n_synch_out
        elif synch_type == "action":
            synch_data = tracking_data.synch_action[
                :, batch_idx, :
            ]  # [iterations, synch_dim]
            max_dim = tracking_data.n_synch_action
        else:
            raise ValueError("synch_type must be 'out' or 'action'")

        if synch_indices is None:
            synch_indices = list(range(min(5, max_dim)))

        fig, ax = plt.subplots(figsize=self.figsize)

        iterations = range(tracking_data.iterations)

        for synch_idx in synch_indices:
            if synch_idx < max_dim:
                ax.plot(
                    iterations,
                    synch_data[:, synch_idx],
                    label=f"Synch {synch_idx}",
                    linewidth=2,
                )

        ax.set_xlabel("Iteration")
        ax.set_ylabel("Synchronization Value")
        ax.set_title(
            f"{synch_type.capitalize()} Synchronization Time Series (Batch {batch_idx})"
        )
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig

    def plot_synchronization_convergence(
        self,
        tracking_data: TrackingData,
        synch_type: str = "out",
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Plot convergence analysis of synchronization values.

        Args:
            tracking_data: TrackingData object with recorded synchronization
            synch_type: 'out' or 'action' synchronization
            save_path: Optional path to save the figure

        Returns:
            matplotlib Figure object
        """
        if synch_type == "out":
            # [iterations, batch, synch_dim]
            synch_data = tracking_data.synch_out
        elif synch_type == "action":
            # [iterations, batch, synch_dim]
            synch_data = tracking_data.synch_action
        else:
            raise ValueError("synch_type must be 'out' or 'action'")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.figsize)

        # Compute convergence metrics
        iterations = range(tracking_data.iterations)

        # 1. Change rate over time (L2 norm of differences)
        if tracking_data.iterations > 1:
            differences = np.diff(
                synch_data, axis=0
            )  # [iterations-1, batch, synch_dim]
            change_rates = np.linalg.norm(
                differences, axis=-1)  # [iterations-1, batch]
            mean_change_rates = np.mean(change_rates, axis=1)  # [iterations-1]
            std_change_rates = np.std(change_rates, axis=1)  # [iterations-1]

            ax1.plot(iterations[1:], mean_change_rates,
                     "b-", linewidth=2, label="Mean")
            ax1.fill_between(
                iterations[1:],
                mean_change_rates - std_change_rates,
                mean_change_rates + std_change_rates,
                alpha=0.3,
                color="blue",
                label="±1 Std",
            )
            ax1.set_xlabel("Iteration")
            ax1.set_ylabel("Change Rate (L2 Norm)")
            ax1.set_title(
                f"{synch_type.capitalize()} Synchronization Change Rate")
            ax1.legend()
            ax1.grid(True, alpha=0.3)

        # 2. Distance from final state
        final_state = synch_data[-1]  # [batch, synch_dim]
        distances_to_final = []
        for t in range(tracking_data.iterations):
            current_state = synch_data[t]  # [batch, synch_dim]
            distances = np.linalg.norm(
                current_state - final_state, axis=-1)  # [batch]
            distances_to_final.append(np.mean(distances))

        ax2.plot(iterations, distances_to_final, "r-", linewidth=2)
        ax2.set_xlabel("Iteration")
        ax2.set_ylabel("Distance to Final State")
        ax2.set_title(f"{synch_type.capitalize()} Convergence to Final State")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig

    def plot_attention_weights_evolution(
        self,
        tracking_data: TrackingData,
        head_idx: int = 0,
        batch_idx: int = 0,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Plot evolution of attention weights over time.

        Args:
            tracking_data: TrackingData object with recorded attention weights
            head_idx: Which attention head to visualize
            batch_idx: Which batch sample to visualize
            save_path: Optional path to save the figure

        Returns:
            matplotlib Figure object
        """
        # attention_weights shape: [iterations, batch, heads, seq_len]
        attention_data = tracking_data.attention_weights[
            :, batch_idx, head_idx, :
        ]  # [iterations, seq_len]

        fig, ax = plt.subplots(figsize=self.figsize)

        # Create heatmap
        im = ax.imshow(
            attention_data.T, aspect="auto", cmap="Blues", interpolation="nearest"
        )

        # Customize plot
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Sequence Position")
        ax.set_title(
            f"Attention Weights Evolution (Head {head_idx}, Batch {batch_idx})"
        )

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Attention Weight")

        # Set ticks
        ax.set_xticks(
            range(0, tracking_data.iterations, max(
                1, tracking_data.iterations // 10))
        )
        ax.set_yticks(
            range(0, tracking_data.seq_len, max(
                1, tracking_data.seq_len // 10))
        )

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig

    def plot_attention_normalization_check(
        self, tracking_data: TrackingData, save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot attention weight normalization verification.

        Args:
            tracking_data: TrackingData object with recorded attention weights
            save_path: Optional path to save the figure

        Returns:
            matplotlib Figure object
        """
        # attention_weights shape: [iterations, batch, heads, seq_len]
        # Sum over seq_len dimension (last axis)
        attention_sums = np.sum(
            tracking_data.attention_weights, axis=-1
        )  # [iterations, batch, heads]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.figsize)

        # Plot distribution of attention sums
        ax1.hist(attention_sums.flatten(), bins=50,
                 alpha=0.7, edgecolor="black")
        ax1.axvline(
            x=1.0, color="red", linestyle="--", linewidth=2, label="Expected (1.0)"
        )
        ax1.set_xlabel("Attention Sum")
        ax1.set_ylabel("Frequency")
        ax1.set_title("Distribution of Attention Weight Sums")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot deviation from 1.0 over time
        deviations = np.abs(attention_sums - 1.0)  # [iterations, batch, heads]
        mean_deviations = np.mean(deviations, axis=(1, 2))  # [iterations]
        max_deviations = np.max(deviations, axis=(1, 2))  # [iterations]

        iterations = range(tracking_data.iterations)
        ax2.plot(iterations, mean_deviations, "b-",
                 linewidth=2, label="Mean Deviation")
        ax2.plot(iterations, max_deviations, "r-",
                 linewidth=2, label="Max Deviation")
        ax2.set_xlabel("Iteration")
        ax2.set_ylabel("Deviation from 1.0")
        ax2.set_title("Attention Normalization Deviation Over Time")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_yscale("log")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig

    def create_synchronization_summary_report(
        self, tracking_data: TrackingData, save_dir: Optional[str] = None
    ) -> Dict[str, plt.Figure]:
        """
        Create comprehensive synchronization analysis report.

        Args:
            tracking_data: TrackingData object with recorded synchronization
            save_dir: Optional directory to save all figures

        Returns:
            Dictionary mapping plot names to Figure objects
        """
        figures = {}

        # Synchronization evolution plots
        figures["synch_out_evolution"] = self.plot_synchronization_evolution(
            tracking_data,
            "out",
            save_path=f"{save_dir}/synch_out_evolution.png" if save_dir else None,
        )

        figures["synch_action_evolution"] = self.plot_synchronization_evolution(
            tracking_data,
            "action",
            save_path=f"{save_dir}/synch_action_evolution.png" if save_dir else None,
        )

        # Time series plots
        figures["synch_out_timeseries"] = self.plot_synchronization_time_series(
            tracking_data,
            synch_type="out",
            save_path=f"{save_dir}/synch_out_timeseries.png" if save_dir else None,
        )

        figures["synch_action_timeseries"] = self.plot_synchronization_time_series(
            tracking_data,
            synch_type="action",
            save_path=f"{save_dir}/synch_action_timeseries.png" if save_dir else None,
        )

        # Convergence analysis
        figures["synch_out_convergence"] = self.plot_synchronization_convergence(
            tracking_data,
            "out",
            save_path=f"{save_dir}/synch_out_convergence.png" if save_dir else None,
        )

        figures["synch_action_convergence"] = self.plot_synchronization_convergence(
            tracking_data,
            "action",
            save_path=f"{save_dir}/synch_action_convergence.png" if save_dir else None,
        )

        # Attention analysis
        figures["attention_evolution"] = self.plot_attention_weights_evolution(
            tracking_data,
            save_path=f"{save_dir}/attention_evolution.png" if save_dir else None,
        )

        figures["attention_normalization"] = self.plot_attention_normalization_check(
            tracking_data,
            save_path=f"{save_dir}/attention_normalization.png" if save_dir else None,
        )

        return figures
