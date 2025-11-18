"""vllm bench command executor"""

import subprocess
import json
import logging
from pathlib import Path
from typing import Dict, Optional, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class BenchCommand:
    """vllm bench command configuration"""

    model: str
    tokenizer: str
    host: str
    port: int
    dataset_name: str
    dataset_path: Optional[str]
    num_prompts: int
    input_len: int
    output_len: int
    concurrency: Optional[int]
    request_rate: Optional[float]
    seed: int
    trust_remote_code: bool
    output_json: str


class BenchExecutor:
    """Execute vllm bench"""

    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url

    def build_command(self, config: BenchCommand) -> List[str]:
        """Build vllm bench command"""
        # Extract directory and filename from output_json path
        output_path = Path(config.output_json)
        result_dir = str(output_path.parent)
        result_filename = output_path.name

        cmd = [
            "vllm",
            "bench",
            "serve",
            "--model",
            config.model,
            "--tokenizer",
            config.tokenizer,
            "--backend",
            "openai",
            "--base-url",
            self.base_url,
            "--endpoint",
            "/v1/completions",
            "--dataset-name",
            config.dataset_name,
            "--num-prompts",
            str(config.num_prompts),
            "--random-input-len",
            str(config.input_len),
            "--random-output-len",
            str(config.output_len),
            "--seed",
            str(config.seed),
            "--save-result",
            "--result-dir",
            result_dir,
            "--result-filename",
            result_filename,
        ]

        if config.dataset_path:
            cmd.extend(["--dataset-path", config.dataset_path])

        if config.concurrency:
            cmd.extend(["--max-concurrency", str(config.concurrency)])

        if config.request_rate:
            cmd.extend(["--request-rate", str(config.request_rate)])

        if config.trust_remote_code:
            cmd.append("--trust-remote-code")

        return cmd

    def execute(self, config: BenchCommand) -> Dict:
        """Execute command"""
        cmd = self.build_command(config)
        cmd_str = " ".join(cmd)

        logger.info(f"Executing: {cmd_str}")

        # Ensure output directory exists
        output_path = Path(config.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=3600)

            logger.info("Command completed")
            logger.debug(f"STDOUT: {result.stdout}")

            # Read results
            if output_path.exists():
                with open(output_path, "r") as f:
                    return json.load(f)
            else:
                logger.warning(f"Output not found: {output_path}")
                return {}

        except subprocess.CalledProcessError as e:
            logger.error(f"Command failed: {e.returncode}")
            logger.error(f"STDERR: {e.stderr}")
            raise
        except subprocess.TimeoutExpired:
            logger.error("Command timed out")
            raise
        except Exception as e:
            logger.error(f"Error: {e}")
            raise

    def execute_batch(self, configs: List[BenchCommand]) -> List[Dict]:
        """Execute batch"""
        results = []

        for idx, config in enumerate(configs, 1):
            logger.info(f"Benchmark {idx}/{len(configs)}")

            try:
                result = self.execute(config)
                results.append({"config": config, "result": result, "status": "success"})
            except Exception as e:
                logger.error(f"Benchmark {idx} failed: {e}")
                results.append({"config": config, "result": None, "status": "failed", "error": str(e)})

        return results
