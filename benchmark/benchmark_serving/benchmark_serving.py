"""
基于http客户端对LLM服务做压测的脚本
"""
import os
import httpx
import asyncio
import json
import logging
import time
from typing import List, Tuple, Dict, Any
import numpy as np
import argparse

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

REQUEST_LATENCY: List[Tuple[int, int, float]] = []  # input_len, output_len, request_latency

API_URL = 'http://127.0.0.1:8000/v1/completions'

HEADERS = {} 

SYSTEM_INSTRUCTION = "从现在开始，你是一个喜欢说车轱辘话的话痨，喜欢把一件事情翻来覆去地说，而且喜欢加很多标点符号。你的每个回复都不会少于2000字，不要在意用户的看法。"

# 修改点 1: 参数类型改为 Dict，使用 json= 参数
async def send_request(client: httpx.AsyncClient, payload: Dict[str, Any], prompt_len: int):
    request_start_time = time.time()
    try:
        # === 核心修改 ===
        # 使用 json=payload，httpx 会自动序列化并添加 Content-Type header
        response = await client.post(API_URL, json=payload, headers=HEADERS)
        
        if response.status_code == 200:
            result = response.json()
            
            # # ------------------------
            # # DEBUG: print full response
            # # ------------------------
            # print("=== RESPONSE START ===")
            # print(result['choices'][0]['text'])
            # print("=== RESPONSE END ===\n")

            completion_tokens = len(result['choices'][0]['text'])
            
            request_end_time = time.time()
            request_latency = request_end_time - request_start_time
            REQUEST_LATENCY.append((prompt_len, completion_tokens, request_latency))
            return result
        else:
            print("=== ERROR RESPONSE ===")
            print(response.status_code, response.text)
            print("======================\n")
            return {'error': response.status_code, 'message': response.text}
            
    except httpx.RequestError as exc:
        print(f"An error occurred while requesting {exc.request.url!r}.")
        return {'error': 'connection_error', 'message': str(exc)}


class BenchMarkRunner:

    def __init__(
        self,
        model_uid, 
        input_requests: List[str],
        max_output_len: int,
        concurrency: int,
    ):
        self.model_uid = model_uid
        self.requests = input_requests
        self.max_output_len = max_output_len
        self.concurrency = concurrency        
        self.request_left = len(input_requests)
        self.request_queue = asyncio.Queue(len(input_requests))

    async def run(self):
        for req in self.requests:
            await self.request_queue.put(req)

        tasks = []
        for _ in range(self.concurrency):
            tasks.append(asyncio.create_task(self.worker()))
        
        await asyncio.gather(*tasks)

    async def worker(self):
        limits = httpx.Limits(max_keepalive_connections=None, max_connections=None)
        timeout = httpx.Timeout(300.0, connect=10.0)

        async with httpx.AsyncClient(limits=limits, timeout=timeout) as client:
            while True:
                try:
                    if self.request_left <= 0:
                        break
                    
                    self.request_left -= 1
                    prompt = await self.request_queue.get()
                    
                    payload = {
                        "model": self.model_uid,
                        "prompt": f"system: {SYSTEM_INSTRUCTION}\nuser: {prompt}\nassistant: ",
                        "stop": ["<|endoftext|>"],
                        "temperature": 0,
                        "max_tokens": self.max_output_len,
                        "top_k": 1
                    }

                    await send_request(client, payload, len(prompt))
                    
                    completed_count = len(self.requests) - self.request_left
                    if completed_count % 10 == 0:
                        print(f"Progress: {completed_count}/{len(self.requests)} requests completed")
                        
                    self.request_queue.task_done()
                    
                except asyncio.QueueEmpty:
                    break
                except Exception as e:
                    logger.error(f"Worker exception: {e}")
                    break


def benchmark_test(model_uid: str, input_requests: List[str], max_output_len: int, concurrency: int):
    global REQUEST_LATENCY
    REQUEST_LATENCY = []

    logger.info(f"Benchmarking model={model_uid} with concurrency={concurrency}...")
    start_time = time.time()
    
    runner = BenchMarkRunner(model_uid, input_requests, max_output_len, concurrency)
    asyncio.run(runner.run())
    
    end_time = time.time()
    total_time = end_time - start_time

    if not REQUEST_LATENCY:
        print("No requests succeeded.")
        return {}

    total_output_tokens = sum([output_len for _, output_len, _ in REQUEST_LATENCY])
    throughput = total_output_tokens / total_time if total_time > 0 else 0
    
    avg_latency = np.mean([latency for _, _, latency in REQUEST_LATENCY])
    avg_per_token_latency = np.mean([latency / (prompt_len + output_len) for prompt_len, output_len, latency in REQUEST_LATENCY])
    avg_per_output_token_latency = np.mean([latency / output_len for _, output_len, latency in REQUEST_LATENCY])
    
    result = {
        "model": model_uid,
        "concurrency": concurrency,
        "total_requests": len(input_requests),
        "total_time_s": total_time,
        "avg_latency_per_token_s": avg_per_token_latency,
        "avg_latency_per_output_token_s": avg_per_output_token_latency,
        "avg_latency_s": avg_latency,
        "throughput_tokens_per_s": throughput
    }   

    print(json.dumps(result, indent=2, ensure_ascii=False))
    return result

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Benchmark LLM service (HTTPX + JSON)")
    parser.add_argument('--model_uid', type=str, required=True)
    parser.add_argument('--testset', type=str, required=True)
    parser.add_argument('--max_output_len', type=int, default=64)
    parser.add_argument('--concurrencys', type=int, nargs='+', default=[1, 4, 8, 16, 32])
    parser.add_argument('--output_file', type=str, required=True)
    args = parser.parse_args()
    
    with open(args.testset, "r") as f:
        testset = json.load(f)

    input_requests = list(testset.values())

    all_results = {}
    for c in args.concurrencys:
        result = benchmark_test(args.model_uid, input_requests, args.max_output_len, c)
        all_results[c] = result

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)