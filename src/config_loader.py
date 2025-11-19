"""Configuration loader - vllm bench based version"""

from typing import Literal, Optional, List
from pathlib import Path
import yaml
from pydantic import BaseModel, Field, field_validator


class VLLMServeArgs(BaseModel):
    """vLLM serve startup arguments"""

    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    max_num_seqs: int = 4


class VLLMServiceConfig(BaseModel):
    """vLLM service configuration"""

    model: str
    tokenizer: str
    host: str = "localhost"
    port: int = 8000
    auto_start: bool = False
    serve_args: VLLMServeArgs = Field(default_factory=VLLMServeArgs)


class DatasetConfig(BaseModel):
    """Dataset configuration"""

    name: Literal["sharegpt", "sonnet", "random", "custom"] = "sharegpt"
    path: Optional[str] = None


class BenchmarkConfig(BaseModel):
    """Benchmark configuration"""

    dataset: DatasetConfig
    num_prompts: int = Field(gt=0, default=100)
    request_rate: Optional[float] = None
    seed: int = 42
    trust_remote_code: bool = True
    output_format: Literal["json", "text"] = "json"
    save_results: bool = True


class LengthConfig(BaseModel):
    """Length configuration"""

    type: Literal["list", "fixed", "range"]
    values: Optional[List[int]] = None
    value: Optional[int] = None
    min: Optional[int] = None
    max: Optional[int] = None
    step: Optional[int] = None

    @field_validator("values")
    @classmethod
    def validate_values(cls, v, info):
        if info.data.get("type") == "list" and not v:
            raise ValueError("values required when type='list'")
        return v

    @field_validator("value")
    @classmethod
    def validate_value(cls, v, info):
        if info.data.get("type") == "fixed" and v is None:
            raise ValueError("value required when type='fixed'")
        return v

    @field_validator("min")
    @classmethod
    def validate_range(cls, v, info):
        if info.data.get("type") == "range":
            if info.data.get("min") is None or info.data.get("max") is None:
                raise ValueError("min and max required when type='range'")
        return v


class SweepVariablesConfig(BaseModel):
    """Sweep variables configuration"""

    concurrency_levels: Optional[List[int]] = None
    request_rates: Optional[List[float]] = None
    input_lengths: LengthConfig
    output_lengths: LengthConfig

    @field_validator("request_rates")
    @classmethod
    def validate_strategy(cls, v, info):
        concurrency = info.data.get("concurrency_levels")
        if concurrency and v:
            raise ValueError(
                "concurrency_levels and request_rates are mutually exclusive"
            )
        if not concurrency and not v:
            raise ValueError("Must specify either concurrency_levels or request_rates")
        return v


class OutputConfig(BaseModel):
    """Output configuration"""

    results_dir: str = "results"
    raw_results_pattern: str = "results/raw/run_{run_id}.json"
    aggregated_csv: str = "results/aggregated_results.csv"
    summary_json: str = "results/summary.json"
    plots_dir: str = "results/plots"
    log_file: str = "results/benchmark.log"


class AnalysisConfig(BaseModel):
    """Analysis configuration"""

    plots: List[str] = Field(default_factory=list)
    percentiles: List[int] = Field(default=[50, 90, 95, 99])
    detailed_report: bool = True


class Config(BaseModel):
    """Main configuration"""

    vllm_service: VLLMServiceConfig
    benchmark: BenchmarkConfig
    sweep_variables: SweepVariablesConfig
    output: OutputConfig
    analysis: AnalysisConfig


def load_config(config_path: str) -> Config:
    """Load configuration"""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(path, "r", encoding="utf-8") as f:
        config_dict = yaml.safe_load(f)

    return Config(**config_dict)
