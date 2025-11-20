import asyncio
import aiohttp
import time
import random
import numpy as np
import pandas as pd
from datetime import datetime
from tqdm import tqdm

API_ENDPOINT = "http://localhost:8001/v1/completions" 
MODELS = ['Chatbot-A-large', 'Chatbot-B', 'Chatbot-C']

# 測試持續時間 (秒)
TEST_DURATION = 60

# 基礎並發量 (Requests Per Second)
TARGET_RPS = 3

# 輸入的 Prompt (長度最好固定，以排除 input length 的干擾，專注測切換)
PROMPT_TEXT = "Define the Service Level Objective in one sentence."
MAX_TOKENS = 50

async def send_request(session, request_id, model_name, scenario_name):
    """
    發送單一請求，使用 Streaming 模式以測量 TTFT 和 ITL
    """
    url = API_ENDPOINT
    payload = {
        "model": model_name,
        "prompt": PROMPT_TEXT,
        "max_tokens": MAX_TOKENS,
        "temperature": 0.7,
        "stream": True
    }
    
    start_time = time.time()
    ttft = 0
    total_latency = 0
    token_count = 0
    status = "FAIL"
    
    try:
        async with session.post(url, json=payload) as response:
            if response.status == 200:
                first_token_time = None
                last_token_time = None
                
                async for line in response.content:
                    if line:
                        current_time = time.time()
                        
                        # 捕捉 TTFT (第一次收到數據的時間)
                        if first_token_time is None:
                            first_token_time = current_time
                            ttft = (first_token_time - start_time) * 1000 # ms
                        
                        last_token_time = current_time
                        token_count += 1
                
                # 計算總延遲
                if last_token_time:
                    end_time = last_token_time
                    total_latency = (end_time - start_time) * 1000 # ms
                    status = "SUCCESS"
                else:
                    # 有 200 OK 但沒內容的情況
                    status = "EMPTY_RESPONSE"
            else:
                # 使用 tqdm.write 避免破壞進度條
                tqdm.write(f"Error {response.status}") 
                pass
                
    except Exception as e:
        tqdm.write(f"Request failed: {e}")
        pass
    
    avg_itl = 0
    if token_count > 1 and status == "SUCCESS":
        avg_itl = (total_latency - ttft) / (token_count - 1)

    return {
        "request_id": request_id,
        "scenario": scenario_name,
        "model": model_name,
        "start_time": start_time,
        "ttft_ms": ttft,
        "total_latency_ms": total_latency,
        "avg_itl_ms": avg_itl,
        "token_count": token_count,
        "status": status
    }

async def run_scenario_round_robin(session, results):
    """場景一：極限切換 (Round Robin)"""
    print(f"--- Starting Scenario: Round Robin (Worst Case) ---")
    start_test = time.time()
    req_id = 0
    
    # 使用 tqdm 建立進度條，總量為測試秒數
    with tqdm(total=TEST_DURATION, desc="Round Robin Progress", unit="s") as pbar:
        last_update_time = start_test
        
        while time.time() - start_test < TEST_DURATION:
            model = MODELS[req_id % len(MODELS)]
            task = asyncio.create_task(send_request(session, f"RR-{req_id}", model, "round_robin"))
            results.append(task)
            req_id += 1
            
            await asyncio.sleep(1.0 / TARGET_RPS)
            
            now = time.time()
            pbar.update(now - last_update_time)
            last_update_time = now

async def run_scenario_zipfian(session, results):
    """場景二：真實分佈 (Zipfian / Weighted)"""
    print(f"--- Starting Scenario: Zipfian (Real World) ---")
    # 設定權重：第一個模型 80%，其餘平分剩下的 20%
    weights = [0.8] + [(0.2 / (len(MODELS)-1))] * (len(MODELS)-1)
    
    start_test = time.time()
    req_id = 0
    
    with tqdm(total=TEST_DURATION, desc="Zipfian Progress", unit="s") as pbar:
        last_update_time = start_test
        
        while time.time() - start_test < TEST_DURATION:
            # 根據權重隨機選擇
            model = random.choices(MODELS, weights=weights, k=1)[0]
            task = asyncio.create_task(send_request(session, f"ZIPF-{req_id}", model, "zipfian"))
            results.append(task)
            req_id += 1
            # 使用泊松過程的間隔時間 (更接近真實流量)
            sleep_time = np.random.exponential(1.0 / TARGET_RPS)
            await asyncio.sleep(sleep_time)
            
            now = time.time()
            pbar.update(now - last_update_time)
            last_update_time = now

async def run_scenario_bursty(session, results):
    """場景三：突發流量 (Bursty)"""
    print(f"--- Starting Scenario: Bursty (Stress Test) ---")
    start_test = time.time()
    req_id = 0
    
    with tqdm(total=TEST_DURATION, desc="Bursty Progress", unit="s") as pbar:
        last_update_time = start_test
        
        while time.time() - start_test < TEST_DURATION:
            current_elapsed = time.time() - start_test
            
            # 每 10 秒產生一次突發流量 (Burst)
            if int(current_elapsed) % 10 == 0 and int(current_elapsed) > 0:
                # 使用 tqdm.write 來打印日誌，避免進度條錯亂
                tqdm.write(f"!!! BURST INCOMING at {int(current_elapsed)}s !!!")
                
                burst_size = 10  # 一次灌入 10 個請求
                # 突發流量通常是混合的，這裡隨機混合
                burst_models = random.choices(MODELS, k=burst_size)
                
                for model in burst_models:
                    task = asyncio.create_task(send_request(session, f"BURST-{req_id}", model, "bursty"))
                    results.append(task)
                    req_id += 1
                
                # 突發後稍微休息一下，避免瞬間過載導致 Client 當機
                await asyncio.sleep(1) 
            else:
                # 背景流量 (低負載)
                model = random.choice(MODELS)
                task = asyncio.create_task(send_request(session, f"BG-{req_id}", model, "bursty"))
                results.append(task)
                req_id += 1
                await asyncio.sleep(1.0 / (TARGET_RPS / 2)) # 背景流量設為目標的一半
            
            now = time.time()
            pbar.update(now - last_update_time)
            last_update_time = now

async def run_single_test(session, test_case):
    all_tasks = []
    case_name = ""
    
    if test_case == 1:
        case_name = "round_robin"
        await run_scenario_round_robin(session, all_tasks)
    elif test_case == 2:
        case_name = "zipfian"
        await run_scenario_zipfian(session, all_tasks)
    elif test_case == 3:
        case_name = "bursty"
        await run_scenario_bursty(session, all_tasks)
    
    print(f"\nAll requests dispatched for {case_name}. Waiting for pending responses...")
    
    responses = []
    for f in tqdm(asyncio.as_completed(all_tasks), total=len(all_tasks), desc="Collecting Responses", unit="req"):
        responses.append(await f)
    
    df = pd.DataFrame(responses)
    filename = f"benchmark_results_{case_name}.csv"
    df.to_csv(filename, index=False)
    print(f"Done! Results saved to {filename}")
    
    print(f"\n=== Quick Summary ({case_name}) ===")
    if not df.empty:
        print(df.groupby("model")["total_latency_ms"].describe())
    else:
        print("No data collected.")

async def main():
    print("input test_case number:")
    print("1. Round Robin")
    print("2. Zipfian (真實分佈)")
    print("3. Bursty")
    print("4. Run All")
    
    try:
        test_case = int(input().strip())
    except ValueError:
        print("Invalid input")
        return

    if test_case not in [1, 2, 3, 4]:
        print(f"請輸入 1, 2, 3, 或 4 選項")
        return

    async with aiohttp.ClientSession() as session:
        if test_case == 4:
            scenarios = [1, 2, 3]
            for i, scenario in enumerate(scenarios):
                await run_single_test(session, scenario)
                if i < len(scenarios) - 1:
                    print("\nWaiting 10 seconds before next test...")
                    await asyncio.sleep(10)
        else:
            await run_single_test(session, test_case)

if __name__ == "__main__":
    asyncio.run(main())