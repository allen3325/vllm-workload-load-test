# vLLM Workload Load Test Tool

> **Load testing tool based on `vllm bench`** - config generator + results analyzer

## 🎯 Why use vllm bench?

This tool is built on top of the official vLLM `vllm bench` tool for the following reasons:

✅ **Official support & maintenance** - stays in sync with vLLM releases  
✅ **More accurate measurements** - deeper understanding of vLLM internals  
✅ **Rich built-in features** - multiple load strategies and dataset support  
✅ **Lower maintenance burden** - focus on configuration management and data analysis  

## 📦 Installation

```bash
# Clone the repo
git clone https://github.com/allen3325/vllm-workload-load-test.git
cd vllm-workload-load-test

# Install dependencies
pip install -r requirements.txt
```

## 🚀 Quickstart

### 1. Start vLLM service

```bash
vllm serve Qwen/Qwen2-0.5B --host 0.0.0.0 --port 8000
```

### 2. Configure test parameters

Edit `config.yaml`:

```yaml
vllm_service:
  model: "Qwen/Qwen2-0.5B"
  host: "localhost"
  port: 8000

benchmark:
  dataset:
    name: "sharegpt"
  num_prompts: 100

sweep_variables:
  concurrency_levels: [1, 2, 4, 8]
  input_lengths:
    type: list
    values: [128, 512, 1024]
  output_lengths:
    type: fixed
    value: 256
```

### 3. Preview experiment matrix

```bash
python main.py --dry-run
```

### 4. Run tests

```bash
python main.py
```

### 5. View results

```bash
# Aggregated data
cat results/aggregated_results.csv

# Statistical summary
cat results/summary.json

# Plots
ls results/plots/
```

## 📋 Configuration

### Load strategies

**Fixed concurrency**
```yaml
sweep_variables:
  concurrency_levels: [1, 2, 4, 8, 16]
```

**Fixed request rate**
```yaml
sweep_variables:
  request_rates: [1, 2, 5, 10, 20]
```

### Length configuration

**List mode**
```yaml
input_lengths:
  type: list
  values: [128, 512, 1024, 2048]
```

**Fixed value**
```yaml
output_lengths:
  type: fixed
  value: 256
```

**Range mode**
```yaml
input_lengths:
  type: range
  min: 128
  max: 2048
  step: 256
```

## 📊 Output format

### Aggregated data (CSV)

| Field | Description |
|------|-------------|
| run_id | Experiment ID |
| model | Model name |
| concurrency | Concurrency level |
| input_len | Input length |
| output_len | Output length |
| median_ttft_ms | Median TTFT |
| p99_ttft_ms | TTFT P99 |
| median_itl_ms | Median ITL |
| throughput | Throughput |

### Statistical summary (JSON)

```json
{
  "total_experiments": 32,
  "models": ["Qwen/Qwen2-0.5B"],
  "concurrency_levels": [1, 2, 4, 8],
  "overall_throughput": {
    "mean": 125.3,
    "min": 45.2,
    "max": 256.7
  }
}
```

### Visualization plots

- `ttft_vs_input_length.png` - TTFT vs input length
- `itl_vs_output_length.png` - ITL vs output length
- `latency_vs_concurrency.png` - Latency vs concurrency
- `throughput_vs_concurrency.png` - Throughput vs concurrency

## 🔧 Advanced usage

### Custom config file

```bash
python main.py --config custom.yaml
```

### Verbose logging

```bash
python main.py --verbose
```

### Use custom dataset

```yaml
benchmark:
  dataset:
    name: "sharegpt"
    path: "/path/to/custom/dataset.jsonl"
```

## 🐛 Troubleshooting

### Cannot connect to vLLM service

```bash
# Check service status
curl http://localhost:8000/v1/models
```

### vllm bench command not found

```bash
# Ensure vLLM is installed
pip install vllm>=0.3.0
```

## 📝 License

MIT License

## 🤝 Contributing

Contributions welcome via Issues and PRs!