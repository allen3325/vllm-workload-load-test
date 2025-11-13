"""Result aggregator"""

import json
import logging
from pathlib import Path
from typing import List, Dict
import pandas as pd

logger = logging.getLogger(__name__)


class ResultAggregator:
    """Aggregate vllm bench results"""

    def __init__(self, output_dir: str = "results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def aggregate_results(self, results: List[Dict]) -> pd.DataFrame:
        """Aggregate results"""
        aggregated_data = []

        for item in results:
            if item["status"] != "success" or not item["result"]:
                continue

            config = item["config"]
            result = item["result"]

            row = {
                "run_id": Path(config.output_json).stem,
                "model": config.model,
                "concurrency": config.concurrency,
                "request_rate": config.request_rate,
                "input_len": config.input_len,
                "output_len": config.output_len,
                "num_prompts": config.num_prompts,
            }

            # Extract vllm bench metrics
            if "summary" in result:
                summary = result["summary"]
                row.update(
                    {
                        "total_time": summary.get("total_time"),
                        "throughput": summary.get("throughput"),
                        "mean_ttft_ms": summary.get("mean_ttft_ms"),
                        "median_ttft_ms": summary.get("median_ttft_ms"),
                        "p99_ttft_ms": summary.get("p99_ttft_ms"),
                        "mean_tpot_ms": summary.get("mean_tpot_ms"),
                        "median_tpot_ms": summary.get("median_tpot_ms"),
                        "p99_tpot_ms": summary.get("p99_tpot_ms"),
                        "mean_itl_ms": summary.get("mean_itl_ms"),
                        "median_itl_ms": summary.get("median_itl_ms"),
                        "p99_itl_ms": summary.get("p99_itl_ms"),
                    }
                )

            aggregated_data.append(row)

        df = pd.DataFrame(aggregated_data)
        logger.info(f"Aggregated {len(df)} experiments")

        return df

    def save_aggregated_results(self, df: pd.DataFrame, filepath: str):
        """Save aggregated results"""
        output_path = self.output_dir / filepath
        output_path.parent.mkdir(parents=True, exist_ok=True)

        df.to_csv(output_path, index=False)
        logger.info(f"Saved to {output_path}")

    def generate_summary(self, df: pd.DataFrame) -> Dict:
        """Generate statistical summary"""
        summary = {
            "total_experiments": len(df),
            "models": df["model"].unique().tolist(),
            "concurrency_levels": sorted(df["concurrency"].dropna().unique().tolist()),
            "input_lengths": sorted(df["input_len"].unique().tolist()),
            "output_lengths": sorted(df["output_len"].unique().tolist()),
        }

        if "throughput" in df.columns:
            summary["overall_throughput"] = {
                "mean": float(df["throughput"].mean()),
                "min": float(df["throughput"].min()),
                "max": float(df["throughput"].max()),
            }

        if "mean_ttft_ms" in df.columns:
            summary["overall_ttft"] = {
                "mean": float(df["mean_ttft_ms"].mean()),
                "min": float(df["mean_ttft_ms"].min()),
                "max": float(df["mean_ttft_ms"].max()),
            }

        return summary

    def save_summary(self, summary: Dict, filepath: str):
        """Save summary"""
        output_path = self.output_dir / filepath
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved summary to {output_path}")
