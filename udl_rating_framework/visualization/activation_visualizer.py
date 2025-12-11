"""
Activation Pattern Visualization Utilities.

Provides tools for visualizing neuron activation patterns from CTM processing.
"""

from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from ..models.ctm_adapter import TrackingData


class ActivationVisualizer:
    """
    Visualizer for CTM activation patterns.

    Provides methods to create various visualizations of neuron activations
    including heatmaps, time series, and statistical distributions.
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

    def plot_activation_heatmap(
        self,
        tracking_data: TrackingData,
        activation_type: str = "post",
        batch_idx: int = 0,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Create heatmap of activation patterns over time.

        Args:
            tracking_data: TrackingData object with recorded activations
            activation_type: 'pre' or 'post' activations
            batch_idx: Which batch sample to visualize
            save_path: Optional path to save the figure

        Returns:
            matplotlib Figure object
        """
        if activation_type == "pre":
            activations = tracking_data.pre_activations[
                :, batch_idx, :
            ]  # [iterations, neurons]
        elif activation_type == "post":
            activations = tracking_data.post_activations[
                :, batch_idx, :
            ]  # [iterations, neurons]
        else:
            raise ValueError("activation_type must be 'pre' or 'post'")

        fig, ax = plt.subplots(figsize=self.figsize)

        # Create heatmap
        im = ax.imshow(
            activations.T, aspect="auto", cmap="RdBu_r", interpolation="nearest"
        )

        # Customize plot
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Neuron Index")
        ax.set_title(
            f"{activation_type.capitalize()} Activation Patterns (Batch {batch_idx})"
        )

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Activation Value")

        # Set ticks
        ax.set_xticks(
            range(0, tracking_data.iterations, max(
                1, tracking_data.iterations // 10))
        )
        ax.set_yticks(
            range(0, tracking_data.n_neurons, max(
                1, tracking_data.n_neurons // 10))
        )

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig

    def plot_activation_time_series(
        self,
        tracking_data: TrackingData,
        neuron_indices: Optional[List[int]] = None,
        activation_type: str = "post",
        batch_idx: int = 0,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Plot activation time series for selected neurons.

        Args:
            tracking_data: TrackingData object with recorded activations
            neuron_indices: List of neuron indices to plot (default: first 5)
            activation_type: 'pre' or 'post' activations
            batch_idx: Which batch sample to visualize
            save_path: Optional path to save the figure

        Returns:
            matplotlib Figure object
        """
        if activation_type == "pre":
            activations = tracking_data.pre_activations[
                :, batch_idx, :
            ]  # [iterations, neurons]
        elif activation_type == "post":
            activations = tracking_data.post_activations[
                :, batch_idx, :
            ]  # [iterations, neurons]
        else:
            raise ValueError("activation_type must be 'pre' or 'post'")

        if neuron_indices is None:
            neuron_indices = list(range(min(5, tracking_data.n_neurons)))

        fig, ax = plt.subplots(figsize=self.figsize)

        iterations = range(tracking_data.iterations)

        for neuron_idx in neuron_indices:
            if neuron_idx < tracking_data.n_neurons:
                ax.plot(
                    iterations,
                    activations[:, neuron_idx],
                    label=f"Neuron {neuron_idx}",
                    linewidth=2,
                )

        ax.set_xlabel("Iteration")
        ax.set_ylabel("Activation Value")
        ax.set_title(
            f"{activation_type.capitalize()} Activation Time Series (Batch {batch_idx})"
        )
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig

    def plot_activation_distribution(
        self,
        tracking_data: TrackingData,
        activation_type: str = "post",
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Plot distribution of activation values.

        Args:
            tracking_data: TrackingData object with recorded activations
            activation_type: 'pre' or 'post' activations
            save_path: Optional path to save the figure

        Returns:
            matplotlib Figure object
        """
        if activation_type == "pre":
            activations = tracking_data.pre_activations.flatten()
        elif activation_type == "post":
            activations = tracking_data.post_activations.flatten()
        else:
            raise ValueError("activation_type must be 'pre' or 'post'")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.figsize)

        # Histogram
        ax1.hist(activations, bins=50, alpha=0.7,
                 density=True, edgecolor="black")
        ax1.set_xlabel("Activation Value")
        ax1.set_ylabel("Density")
        ax1.set_title(
            f"{activation_type.capitalize()} Activation Distribution")
        ax1.grid(True, alpha=0.3)

        # Box plot
        ax2.boxplot(activations, vert=True)
        ax2.set_ylabel("Activation Value")
        ax2.set_title(f"{activation_type.capitalize()} Activation Box Plot")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig

    def plot_neuron_correlation_matrix(
        self,
        tracking_data: TrackingData,
        activation_type: str = "post",
        batch_idx: int = 0,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Plot correlation matrix between neurons.

        Args:
            tracking_data: TrackingData object with recorded activations
            activation_type: 'pre' or 'post' activations
            batch_idx: Which batch sample to analyze
            save_path: Optional path to save the figure

        Returns:
            matplotlib Figure object
        """
        if activation_type == "pre":
            activations = tracking_data.pre_activations[
                :, batch_idx, :
            ]  # [iterations, neurons]
        elif activation_type == "post":
            activations = tracking_data.post_activations[
                :, batch_idx, :
            ]  # [iterations, neurons]
        else:
            raise ValueError("activation_type must be 'pre' or 'post'")

        # Compute correlation matrix
        correlation_matrix = np.corrcoef(activations.T)  # [neurons, neurons]

        fig, ax = plt.subplots(figsize=self.figsize)

        # Create heatmap
        im = ax.imshow(
            correlation_matrix, cmap="RdBu_r", vmin=-1, vmax=1, interpolation="nearest"
        )

        # Customize plot
        ax.set_xlabel("Neuron Index")
        ax.set_ylabel("Neuron Index")
        ax.set_title(
            f"{activation_type.capitalize()} Neuron Correlation Matrix (Batch {batch_idx})"
        )

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Correlation Coefficient")

        # Set ticks
        n_ticks = min(10, tracking_data.n_neurons)
        tick_indices = np.linspace(
            0, tracking_data.n_neurons - 1, n_ticks, dtype=int)
        ax.set_xticks(tick_indices)
        ax.set_yticks(tick_indices)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig

    def create_activation_summary_report(
        self, tracking_data: TrackingData, save_dir: Optional[str] = None
    ) -> Dict[str, plt.Figure]:
        """
        Create comprehensive activation analysis report.

        Args:
            tracking_data: TrackingData object with recorded activations
            save_dir: Optional directory to save all figures

        Returns:
            Dictionary mapping plot names to Figure objects
        """
        figures = {}

        # Create various visualizations
        figures["pre_heatmap"] = self.plot_activation_heatmap(
            tracking_data,
            "pre",
            save_path=f"{save_dir}/pre_activation_heatmap.png" if save_dir else None,
        )

        figures["post_heatmap"] = self.plot_activation_heatmap(
            tracking_data,
            "post",
            save_path=f"{save_dir}/post_activation_heatmap.png" if save_dir else None,
        )

        figures["pre_timeseries"] = self.plot_activation_time_series(
            tracking_data,
            activation_type="pre",
            save_path=f"{save_dir}/pre_activation_timeseries.png" if save_dir else None,
        )

        figures["post_timeseries"] = self.plot_activation_time_series(
            tracking_data,
            activation_type="post",
            save_path=(
                f"{save_dir}/post_activation_timeseries.png" if save_dir else None
            ),
        )

        figures["pre_distribution"] = self.plot_activation_distribution(
            tracking_data,
            "pre",
            save_path=(
                f"{save_dir}/pre_activation_distribution.png" if save_dir else None
            ),
        )

        figures["post_distribution"] = self.plot_activation_distribution(
            tracking_data,
            "post",
            save_path=(
                f"{save_dir}/post_activation_distribution.png" if save_dir else None
            ),
        )

        figures["correlation_matrix"] = self.plot_neuron_correlation_matrix(
            tracking_data,
            "post",
            save_path=f"{save_dir}/neuron_correlation_matrix.png" if save_dir else None,
        )

        return figures
