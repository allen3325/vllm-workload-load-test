import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

try:
    df = pd.read_csv("vllm_full_benchmark.csv")
except FileNotFoundError:
    print(
        "Error: 'vllm_full_benchmark.csv' not found. Please run the benchmark script first."
    )
    exit()

# Set plot style
sns.set_theme(style="whitegrid")
plt.figure(figsize=(16, 7))

# Left Figure: TTFT Analysis (P50/P90/P99 vs Input Length)
plt.subplot(1, 2, 1)

# Filter Phase 1 data
ttft_data = df[df["phase"] == "TTFT"]

if not ttft_data.empty:
    # Calculate percentiles
    stats = (
        ttft_data.groupby("input_len")["ttft_ms"].quantile([0.5, 0.9, 0.99]).unstack()
    )
    stats.columns = ["P50", "P90", "P99"]
    stats = stats.reset_index()

    # Draw SLO reference line (5s)
    # SLO_TARGET = 5000
    # plt.axhline(
    #     y=SLO_TARGET,
    #     color="r",
    #     linestyle="--",
    #     label=f"TTFT SLO ({SLO_TARGET}s)",
    #     alpha=0.7,
    # )

    plt.plot(
        stats["input_len"],
        stats["P99"],
        marker="^",
        color="#e74c3c",
        label="P99 (Worst Case)",
        linewidth=2,
        linestyle="--",
    )
    plt.plot(
        stats["input_len"],
        stats["P90"],
        marker="s",
        color="#f39c12",
        label="P90",
        linewidth=2,
    )
    plt.plot(
        stats["input_len"],
        stats["P50"],
        marker="o",
        color="#27ae60",
        label="P50 (Median)",
        linewidth=2,
    )

    # Fill the area between P50 and P99 (Variance Area)
    plt.fill_between(
        stats["input_len"],
        stats["P50"],
        stats["P99"],
        color="#f39c12",
        alpha=0.15,
        label="Variance Range",
    )

    # Annotate values
    for i, row in stats.iterrows():
        plt.text(
            row["input_len"],
            row["P99"],
            f"{int(row['P99'])}",
            fontsize=9,
            va="bottom",
            ha="center",
            color="#c0392b",
            fontweight="bold",
        )

    plt.title("SLO Definition: TTFT vs. Input Length", fontsize=14, fontweight="bold")
    plt.xlabel("Input Token Length", fontsize=12)
    plt.ylabel("Time To First Token (ms)", fontsize=12)
    plt.legend(title="Latency Percentiles")
    plt.xticks(stats["input_len"])
else:
    plt.text(0.5, 0.5, "No TTFT Data Found", ha="center", va="center")

# Right Figure: ITL Analysis (Batch Size vs ITL)
plt.subplot(1, 2, 2)

# Filter Phase 2 data
itl_data = df[df["phase"] == "ITL"]

if not itl_data.empty:
    # Focus on average ITL and P99 ITL for each batch size
    itl_stats = (
        itl_data.groupby("concurrency")["itl_ms"]
        .agg(["mean", lambda x: x.quantile(0.99)])
        .reset_index()
    )
    itl_stats.columns = ["concurrency", "Mean_ITL", "P99_ITL"]

    # Draw P99 ITL
    sns.lineplot(
        data=itl_stats,
        x="concurrency",
        y="P99_ITL",
        marker="o",
        color="#8e44ad",
        linewidth=2.5,
        label="P99 ITL",
    )

    # Plot SLO reference line (e.g., 50ms = 20 tokens/s)
    SLO_TARGET = 50
    plt.axhline(
        y=SLO_TARGET,
        color="r",
        linestyle="--",
        label=f"Readability SLO ({SLO_TARGET}ms)",
        alpha=0.7,
    )

    plt.title("Capacity Planning: ITL vs. Batch Size", fontsize=14, fontweight="bold")
    plt.xlabel("Concurrency (Batch Size)", fontsize=12)
    plt.ylabel("Inter-Token Latency (ms)", fontsize=12)
    plt.legend()

    # Mark points exceeding SLO
    over_slo = itl_stats[itl_stats["P99_ITL"] > SLO_TARGET]
    if not over_slo.empty:
        first_fail = over_slo.iloc[0]
        plt.annotate(
            f"SLO Breached\n@ Batch {int(first_fail['concurrency'])}",
            xy=(first_fail["concurrency"], first_fail["P99_ITL"]),
            xytext=(first_fail["concurrency"], first_fail["P99_ITL"] + 20),
            arrowprops=dict(facecolor="black", shrink=0.05),
            ha="center",
        )

else:
    plt.text(0.5, 0.5, "No ITL Data Found", ha="center", va="center")

plt.tight_layout()
plt.savefig("vllm_workload_slo_analysis.png")
print("Graph saved to 'vllm_workload_slo_analysis.png'")
plt.show()
