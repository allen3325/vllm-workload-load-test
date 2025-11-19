import asyncio
import aiohttp
import time
import pandas as pd
from tqdm.asyncio import tqdm

# Configuration
API_URL = "http://localhost:8001/v1/completions"
MODEL_NAME = "Chatbot-A-large"

# Phase 1: TTFT vs Input Length (P50/P90/P99)
# Measure Prefill latency distribution across different lengths
PHASE1_INPUT_LENGTHS = [128, 512, 1024, 2048]
PHASE1_SAMPLES = 50  # More samples improve P99 accuracy
PHASE1_CONCURRENCY = 1  # Keep requests sequential to avoid measurement interference

# Phase 2: ITL vs Batch Size (Throughput/Bandwidth)
# Measure Decode latency under high load
PHASE2_BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64]  # Simulate concurrent users
PHASE2_INPUT_LEN = 512  # Fixed Input length
PHASE2_OUTPUT_LEN = 128  # Ensure sufficient length for accurate ITL calculation


def generate_prompt(token_len):
    # Dummy tokenizer
    # Generate a simple prompt of specified length (1 word approx 1.3 tokens)
    word_count = int(token_len / 1.3)
    return "test " * word_count


async def send_request(session, prompt, input_len, concurrency_level, test_phase):
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "max_tokens": PHASE2_OUTPUT_LEN
        if test_phase == "ITL"
        else 16,  # TTFT tests require minimal output length
        "stream": True,
        "temperature": 0,
        "ignore_eos": True,
    }

    start_time = time.perf_counter()
    first_token_time = 0
    token_times = []

    try:
        async with session.post(API_URL, json=payload) as response:
            if response.status != 200:
                print(f"Error {response.status}: {await response.text()}")
                return None

            async for line in response.content:
                line = line.decode("utf-8").strip()
                if line.startswith("data: ") and line != "data: [DONE]":
                    current_time = time.perf_counter()

                    if first_token_time == 0:
                        first_token_time = current_time
                    else:
                        token_times.append(current_time)

        ttft = (first_token_time - start_time) * 1000  # ms

        # Calculate ITL (only when subsequent tokens are generated)
        avg_itl = 0
        if len(token_times) > 0:
            # ITL = (Last Token Time - First Token Time) / (Count - 1)
            duration = token_times[-1] - first_token_time
            avg_itl = (duration / len(token_times)) * 1000  # ms

        return {
            "phase": test_phase,
            "input_len": input_len,
            "concurrency": concurrency_level,
            "ttft_ms": ttft,
            "itl_ms": avg_itl,
        }

    except Exception as e:
        print(f"Request failed: {e}")
        return None


async def run_phase_1_ttft(session):
    print(f"\n= Phase 1: TTFT Analysis (Samples per len: {PHASE1_SAMPLES}) =")
    results = []

    # Use a Queue for concurrency control, maintaining extensibility despite the current limit of 1
    queue = asyncio.Queue()
    for length in PHASE1_INPUT_LENGTHS:
        for _ in range(PHASE1_SAMPLES):
            queue.put_nowait(length)

    pbar = tqdm(total=queue.qsize(), desc="Measuring TTFT")

    async def worker():
        while not queue.empty():
            length = await queue.get()
            prompt = generate_prompt(length)
            res = await send_request(
                session, prompt, length, PHASE1_CONCURRENCY, "TTFT"
            )
            if res:
                results.append(res)
            pbar.update(1)
            queue.task_done()

    workers = [asyncio.create_task(worker()) for _ in range(PHASE1_CONCURRENCY)]
    await queue.join()
    for w in workers:
        w.cancel()
    return results


async def run_phase_2_itl(session):
    print("\n= Phase 2: ITL Analysis (Variable Batch Size) =")
    results = []
    prompt = generate_prompt(PHASE2_INPUT_LEN)

    for batch_size in tqdm(PHASE2_BATCH_SIZES, desc="Ramping up Batch Size"):
        tasks = []
        # Issue batch_size requests concurrently to simulate real-world parallelism
        for _ in range(batch_size):
            tasks.append(
                send_request(session, prompt, PHASE2_INPUT_LEN, batch_size, "ITL")
            )

        batch_results = await asyncio.gather(*tasks)
        for res in batch_results:
            if res:
                results.append(res)

    return results


async def main():
    all_data = []
    async with aiohttp.ClientSession() as session:
        # Phase 1
        data1 = await run_phase_1_ttft(session)
        all_data.extend(data1)

        # Phase 2
        data2 = await run_phase_2_itl(session)
        all_data.extend(data2)

    df = pd.DataFrame(all_data)
    df.to_csv("vllm_full_benchmark.csv", index=False)
    print("\nBenchmark Complete! Data saved to 'vllm_full_benchmark.csv'")


if __name__ == "__main__":
    asyncio.run(main())
