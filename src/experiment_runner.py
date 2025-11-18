"""Experiment runner"""

import logging
from typing import List
from itertools import product

from .config_loader import Config
from .bench_executor import BenchExecutor, BenchCommand
from .result_aggregator import ResultAggregator
from .metrics_analyzer import MetricsAnalyzer

logger = logging.getLogger(__name__)


class ExperimentRunner:
    """Experiment runner"""

    def __init__(self, config: Config):
        self.config = config

        base_url = f"http://{config.vllm_service.host}:{config.vllm_service.port}"
        self.executor = BenchExecutor(base_url)
        self.aggregator = ResultAggregator(config.output.results_dir)
        self.analyzer = MetricsAnalyzer(config.output.plots_dir)

    def _expand_length_config(self, length_config) -> List[int]:
        """Expand length configuration"""
        if length_config.type == "list":
            return length_config.values
        elif length_config.type == "fixed":
            return [length_config.value]
        elif length_config.type == "range":
            return list(range(length_config.min, length_config.max + 1, length_config.step or 1))
        raise ValueError(f"Unknown type: {length_config.type}")

    def build_experiment_matrix(self) -> List[BenchCommand]:
        """Build experiment matrix"""
        sweep = self.config.sweep_variables

        input_lengths = self._expand_length_config(sweep.input_lengths)
        output_lengths = self._expand_length_config(sweep.output_lengths)

        use_concurrency = sweep.concurrency_levels is not None
        load_levels = sweep.concurrency_levels if use_concurrency else sweep.request_rates

        commands = []
        run_counter = 1

        for load, input_len, output_len in product(load_levels, input_lengths, output_lengths):
            run_id = f"run_{run_counter:03d}"
            output_json = self.config.output.raw_results_pattern.format(run_id=run_id)

            cmd = BenchCommand(
                model=self.config.vllm_service.model,
                tokenizer=self.config.vllm_service.tokenizer,
                host=self.config.vllm_service.host,
                port=self.config.vllm_service.port,
                dataset_name=self.config.benchmark.dataset.name,
                dataset_path=self.config.benchmark.dataset.path,
                num_prompts=self.config.benchmark.num_prompts,
                input_len=input_len,
                output_len=output_len,
                concurrency=int(load) if use_concurrency else None,
                request_rate=float(load) if not use_concurrency else None,
                seed=self.config.benchmark.seed,
                trust_remote_code=self.config.benchmark.trust_remote_code,
                output_json=output_json,
            )

            commands.append(cmd)
            run_counter += 1

        logger.info(f"Built {len(commands)} experiments")
        return commands

    def print_experiment_matrix(self):
        """Print experiment matrix"""
        commands = self.build_experiment_matrix()

        print(f"\n{'='*80}")
        print(f"Experiment Matrix ({len(commands)} experiments)")
        print(f"{'='*80}\n")

        for idx, cmd in enumerate(commands, 1):
            strategy = f"concurrency={cmd.concurrency}" if cmd.concurrency else f"rps={cmd.request_rate}"
            print(f"Run {idx:03d}: {strategy}, input={cmd.input_len}, output={cmd.output_len}")

        print(f"\n{'='*80}\n")

    def run_all_experiments(self):
        """Run all experiments"""
        commands = self.build_experiment_matrix()
        logger.info(f"Starting {len(commands)} experiments")

        # Execute in batches
        results = self.executor.execute_batch(commands)

        # Aggregate results
        logger.info("Aggregating results...")
        df = self.aggregator.aggregate_results(results)

        if df.empty:
            logger.error("No successful experiments")
            return

        # Save
        self.aggregator.save_aggregated_results(df, self.config.output.aggregated_csv)

        summary = self.aggregator.generate_summary(df)
        self.aggregator.save_summary(summary, self.config.output.summary_json)

        # Generate plots
        if self.config.analysis.plots:
            logger.info("Generating plots...")
            self.analyzer.generate_all_plots(df, self.config.analysis.plots)

        logger.info("Completed!")
