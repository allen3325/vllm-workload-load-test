#!/usr/bin/env python3
"""vLLM Workload Load Test - based on vllm bench"""

import argparse
import logging
import sys
from pathlib import Path

from src.config_loader import load_config
from src.experiment_runner import ExperimentRunner


def setup_logging(log_file: str = None, verbose: bool = False):
    """Configure logging"""
    level = logging.DEBUG if verbose else logging.INFO
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(level)
    console.setFormatter(formatter)

    root = logging.getLogger()
    root.setLevel(level)
    root.addHandler(console)

    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)


def main():
    parser = argparse.ArgumentParser(
        description="vLLM Workload Load Test (powered by vllm bench)"
    )

    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    try:
        print(f"📋 Loading: {args.config}")
        config = load_config(args.config)

        setup_logging(config.output.log_file, args.verbose)
        logger = logging.getLogger(__name__)

        logger.info(f"Model: {config.vllm_service.model}")
        logger.info(f"Service: {config.vllm_service.host}:{config.vllm_service.port}")

        runner = ExperimentRunner(config)

        if args.dry_run:
            print("\n🔍 DRY RUN\n")
            runner.print_experiment_matrix()
        else:
            print("\n🚀 Starting...\n")
            runner.run_all_experiments()
            print("\n✅ Completed!\n")
            print("📊 Results:")
            print(f"   - CSV: {config.output.aggregated_csv}")
            print(f"   - Summary: {config.output.summary_json}")
            print(f"   - Plots: {config.output.plots_dir}")

    except Exception as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
