"""
Training prompt builder for PromptBuilder.

This module provides the TrainingPromptBuilder for creating prompts
from TrainingArtifacts instances.
"""

from typing import Optional, Dict, Any, List
from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import math

from .base import BasePromptBuilder
from ..artifacts import TrainingArtifacts, Artifacts


class CurvePlotter:
    def __init__(self):
        self.fig, self.ax = None, None

    def _set_df_metrics(self, metrics_values: pd.DataFrame):
        # Copy the dataframe
        df_metrics = metrics_values.copy()
        # Create the is_train and key columns
        df_metrics["is_train"] = df_metrics["key"].str.contains("train")
        df_metrics["key"] = df_metrics["key"].apply(
            lambda x: "loss" if "loss" in x else x
        )
        # Create the figure and axes
        self.fig, self.ax = plt.subplots(
            figsize=(15, 10), nrows=math.ceil(len(df_metrics) / 2), ncols=2, sharex=True
        )
        self.ax = self.ax.flatten()

    def plot(self, df_metrics: pd.DataFrame, save_path=None):
        self._set_df_metrics(df_metrics)

        for i, (name, df) in enumerate(df_metrics.groupby("key")):
            if name == "loss":
                self._plot_loss_curves(df, y_col="value", x_col="step", ax=self.ax[i])
            elif name == "val_f1score":
                self._plot_val_curves(
                    df, y_col="value", x_col="step", ax=self.ax[i], title="F1 Score"
                )
            elif name == "val_precision":
                self._plot_val_curves(
                    df, y_col="value", x_col="step", ax=self.ax[i], title="Precision"
                )
            elif name == "val_recall":
                self._plot_val_curves(
                    df, y_col="value", x_col="step", ax=self.ax[i], title="Recall"
                )

        if save_path:
            self.fig.savefig(save_path)
        return self.fig, self.ax

    def _plot_loss_curves(self, df, y_col, x_col, ax):
        sns.scatterplot(x=x_col, y=y_col, data=df, ax=ax, hue="is_train")
        ax.set_title("Loss Curves")
        ax.set_xlabel("Step")
        ax.set_ylabel("Loss")

    def _plot_val_curves(self, df, y_col, x_col, ax, title):
        sns.scatterplot(x=x_col, y=y_col, data=df, ax=ax)
        ax.set_title(title)
        ax.set_xlabel("Step")
        ax.set_ylabel(y_col)
        ax.set_ylim(0, 1)


class TrainingPromptBuilder(BasePromptBuilder):
    """Builds prompts for training artifact analysis."""

    def can_build(self, artifact: Artifacts) -> bool:
        """Check if this builder can handle TrainingArtifacts."""
        return isinstance(artifact, TrainingArtifacts)

    def build_prompt(
        self,
        artifact: TrainingArtifacts,
        plot_curves: bool = False,
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Build structured prompt from TrainingArtifacts."""
        prompt_parts = []
        # Add metrics analysis if available
        if artifact.metrics_values is not None and plot_curves:
            raise NotImplementedError("Plotting curves is not implemented yet")
            plotter = CurvePlotter()
            plot_path = (
                Path(artifact.metrics_path)
                .with_suffix(".png")
                .with_stem("training_curves")
            )
            plotter.plot(artifact.metrics_values, save_path=plot_path)
            prompt_parts.append(f"\nTraining curves:")
            prompt_parts.append(
                f"- Open the image file to see the training curves: {plot_path}"
            )
        else:
            prompt_parts.append(f"\nTraining curves:")
            flat_metrics = {}
            for name, df in artifact.metrics_values.groupby("key"):
                flat_metrics[name] = df.sort_values(by=["step"])["value"].to_list()
            prompt_parts.append(
                f"- Parse the training curves from the metrics values:\n {flat_metrics}"
            )

        # Add training parameters if available
        if artifact.params:
            prompt_parts.append(f"\nTraining parameters:")
            for key, value in artifact.params.items():
                prompt_parts.append(f"- {key}: {value}")

        # Add context if provided
        if context:
            context_str = self._format_context(context)
            if context_str:
                prompt_parts.append(f"\nAdditional context:\n{context_str}")

        # Combine and truncate if necessary
        full_prompt = "\n".join(prompt_parts)
        return full_prompt
