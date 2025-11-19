import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 讀取數據
cases = ["round_robin", "zipfian", "bursty"]
for case in cases:
    file_name = f"benchmark_results_{case}"
    df = pd.read_csv(f'{file_name}.csv')

    # 預處理：將開始時間標準化為從 0 開始
    df['relative_start_time'] = df['start_time'] - df['start_time'].min()

    # 設定畫布
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))

    # 1. 系統穩定性 (Time vs TTFT)
    sns.scatterplot(data=df, x='relative_start_time', y='ttft_ms', hue='model', ax=axes[0, 0])
    axes[0, 0].set_title('1. System Stability: TTFT over Time')
    axes[0, 0].set_xlabel('Time Elapsed (s)')
    axes[0, 0].grid(True)

    # 2. 分佈檢測 (TTFT Distribution)
    sns.histplot(data=df, x='ttft_ms', kde=True, hue='scenario', element="step", ax=axes[0, 1])
    axes[0, 1].set_title('2. Latency Distribution (Check for Bimodal)')
    axes[0, 1].grid(True)

    # 3. 瓶頸分析 (TTFT vs ITL)
    sns.scatterplot(data=df, x='ttft_ms', y='avg_itl_ms', hue='model', ax=axes[1, 0])
    axes[1, 0].set_title('3. Bottleneck Analysis: Scheduling(TTFT) vs Compute(ITL)')
    axes[1, 0].set_ylim(bottom=0) # 確保看到 0
    axes[1, 0].grid(True)

    # 4. 總延遲比較 (Total Latency by Model)
    sns.boxplot(data=df, x='model', y='total_latency_ms', ax=axes[1, 1])
    axes[1, 1].set_title('4. Total Latency Overview per Model')
    axes[1, 1].grid(True)

    plt.tight_layout()
    # plt.show()
    print(f"save figure {file_name}.png")
    plt.savefig(f'{file_name}.png')