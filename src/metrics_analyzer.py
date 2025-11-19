"""Metrics analysis and visualization"""

import logging
from pathlib import Path
from typing import List
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)

sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)


class MetricsAnalyzer:
    """Analysis and visualization"""

    def __init__(self, plots_dir: str = "results/plots"):
        self.plots_dir = Path(plots_dir)
        self.plots_dir.mkdir(parents=True, exist_ok=True)

    def plot_ttft_vs_input_length(self, df: pd.DataFrame):
        """TTFT vs Input Length"""
        if "input_len" not in df.columns or "median_ttft_ms" not in df.columns:
            logger.warning("Missing columns for TTFT plot")
            return

        plt.figure(figsize=(12, 6))

        if "concurrency" in df.columns and df["concurrency"].nunique() > 1:
            for conc in sorted(df["concurrency"].dropna().unique()):
                subset = df[df["concurrency"] == conc]
                plt.plot(
                    subset["input_len"],
                    subset["median_ttft_ms"],
                    marker="o",
                    label=f"Concurrency={int(conc)}",
                )
        else:
            plt.plot(df["input_len"], df["median_ttft_ms"], marker="o")

        plt.xlabel("Input Length (tokens)")
        plt.ylabel("Median TTFT (ms)")
        plt.title("Time To First Token vs Input Length")
        plt.legend()
        plt.grid(True, alpha=0.3)

        output = self.plots_dir / "ttft_vs_input_length.png"
        plt.savefig(output, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved: {output}")

    def plot_itl_vs_output_length(self, df: pd.DataFrame):
        """ITL vs Output Length"""
        if "output_len" not in df.columns or "median_itl_ms" not in df.columns:
            logger.warning("Missing columns for ITL plot")
            return

        plt.figure(figsize=(12, 6))

        if "concurrency" in df.columns and df["concurrency"].nunique() > 1:
            for conc in sorted(df["concurrency"].dropna().unique()):
                subset = df[df["concurrency"] == conc]
                plt.plot(
                    subset["output_len"],
                    subset["median_itl_ms"],
                    marker="o",
                    label=f"Concurrency={int(conc)}",
                )
        else:
            plt.plot(df["output_len"], df["median_itl_ms"], marker="o")

        plt.xlabel("Output Length (tokens)")
        plt.ylabel("Median ITL (ms)")
        plt.title("Inter-Token Latency vs Output Length")
        plt.legend()
        plt.grid(True, alpha=0.3)

        output = self.plots_dir / "itl_vs_output_length.png"
        plt.savefig(output, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved: {output}")

    def plot_latency_vs_concurrency(self, df: pd.DataFrame):
        """Latency vs Concurrency"""
        if "concurrency" not in df.columns:
            logger.warning("No concurrency column")
            return

        plt.figure(figsize=(12, 6))

        grouped = (
            df.groupby("concurrency")
            .agg({"median_ttft_ms": "mean", "p99_ttft_ms": "mean"})
            .reset_index()
        )

        plt.plot(
            grouped["concurrency"],
            grouped["median_ttft_ms"],
            marker="o",
            label="Median TTFT",
            linewidth=2,
        )
        plt.plot(
            grouped["concurrency"],
            grouped["p99_ttft_ms"],
            marker="s",
            label="P99 TTFT",
            linewidth=2,
        )

        plt.xlabel("Concurrency Level")
        plt.ylabel("TTFT (ms)")
        plt.title("Latency vs Concurrency")
        plt.legend()
        plt.grid(True, alpha=0.3)

        output = self.plots_dir / "latency_vs_concurrency.png"
        plt.savefig(output, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved: {output}")

    def plot_throughput_vs_concurrency(self, df: pd.DataFrame):
        """Throughput vs Concurrency"""
        if "concurrency" not in df.columns or "throughput" not in df.columns:
            logger.warning("Missing columns for throughput plot")
            return

        plt.figure(figsize=(12, 6))

        grouped = df.groupby("concurrency")["throughput"].mean().reset_index()

        plt.plot(
            grouped["concurrency"],
            grouped["throughput"],
            marker="o",
            linewidth=2,
            markersize=8,
        )

        plt.xlabel("Concurrency Level")
        plt.ylabel("Throughput (tokens/s)")
        plt.title("Throughput vs Concurrency")
        plt.grid(True, alpha=0.3)

        output = self.plots_dir / "throughput_vs_concurrency.png"
        plt.savefig(output, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved: {output}")

    def generate_all_plots(self, df: pd.DataFrame, plot_types: List[str]):
        """Generate all plots"""
        plot_methods = {
            "ttft_vs_input_length": self.plot_ttft_vs_input_length,
            "itl_vs_output_length": self.plot_itl_vs_output_length,
            "latency_vs_concurrency": self.plot_latency_vs_concurrency,
            "throughput_vs_concurrency": self.plot_throughput_vs_concurrency,
        }

        for plot_type in plot_types:
            if plot_type in plot_methods:
                logger.info(f"Generating: {plot_type}")
                try:
                    plot_methods[plot_type](df)
                except Exception as e:
                    logger.error(f"Failed {plot_type}: {e}")
            else:
                logger.warning(f"Unknown plot: {plot_type}")
