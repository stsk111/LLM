"""
通过vLLM.LLM对LLM离线推理做压测的脚本
"""
import os
import json
import logging
import time
import argparse
import numpy as np
from typing import List, Tuple, Dict, Any
from vllm import LLM, SamplingParams


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

REQUEST_LATENCY: List[Tuple[int, int, float]] = []  # input_len, output_len, request_latency

SYSTEM_INSTRUCTION = "从现在开始，你是一个喜欢说车轱辘话的话痨，喜欢把一件事情翻来覆去地说，而且喜欢加很多标点符号。你的每个回复都不会少于2000字，不要在意用户的看法。"

def run_vllm_benchmark(
    llm: LLM, 
    input_requests: List[str], 
    max_output_len: int, 
    concurrency_label: int
) -> Dict[str, Any]:
    """
    执行单次基准测试
    在离线推理中，'concurrency' 主要体现为一次性提交给引擎的请求数量。
    vLLM 会自动处理 Continuous Batching。
    """
    global REQUEST_LATENCY
    REQUEST_LATENCY = []

    logger.info(f"Preparing {len(input_requests)} requests for offline inference...")

    # 1. 构造 Prompts (保持原脚本的拼接逻辑)
    # 对应原脚本 payload 中的 prompt 字段
    prompts = [
        f"system: {SYSTEM_INSTRUCTION}\nuser: {req}\nassistant: " 
        for req in input_requests
    ]

    sampling_params = SamplingParams(
        temperature=0,
        top_k=1,
        max_tokens=max_output_len,
        stop=["<|endoftext|>"],
        ignore_eos=False  # 确保生成的长度符合预期
    )

    start_time = time.perf_counter()
    outputs = llm.generate(prompts, sampling_params)
    end_time = time.perf_counter()
    
    total_time = end_time - start_time

    for output in outputs:
        # ------------------------
        # DEBUG: print output
        # ------------------------
        print("=== OUTPUT START ===")
        print(output.outputs[0].text)
        print("=== OUTPUT END ===\n")

        prompt_len = len(output.prompt_token_ids)
        output_len = len(output.outputs[0].token_ids)
        
        # metrics 对象包含 arrival_time, first_scheduled_time, first_token_time, finished_time
        metrics = output.metrics
        
        # 如果有部分请求失败，metrics 可能不完整，加个判断
        if metrics and metrics.finished_time and metrics.arrival_time:
            req_latency = metrics.finished_time - metrics.arrival_time

            REQUEST_LATENCY.append((prompt_len, output_len, req_latency))

    # 衡量每条数据处理快慢的标准
    avg_latency = np.mean([latency for _, _, latency in REQUEST_LATENCY])
    avg_per_token_latency = np.mean([latency / (prompt_len + output_len) for prompt_len, output_len, latency in REQUEST_LATENCY])
    avg_per_output_token_latency = np.mean([latency / output_len for _, output_len, latency in REQUEST_LATENCY])

    # 衡量系统处理能力的标准
    total_output_tokens = sum([output_len for _, output_len, _ in REQUEST_LATENCY])
    throughput = total_output_tokens / total_time if total_time > 0 else 0
    
    result = {
        "concurrency": concurrency_label,
        "total_requests": len(input_requests),
        "total_time_s": total_time,
        "avg_latency_s": avg_latency,
        "avg_latency_per_token_s": avg_per_token_latency,
        "avg_latency_per_output_token_s": avg_per_output_token_latency,
        "throughput_tokens_per_s": throughput
    }   

    print(json.dumps(result, indent=2, ensure_ascii=False))
    return result

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Benchmark LLM Offline Inference (vLLM Native)")
    parser.add_argument('--model_uid', type=str, required=True)
    parser.add_argument('--testset', type=str, required=True)
    parser.add_argument('--max_output_len', type=int, default=64)
    # 离线推理通常不需要手动指定并发数，因为 vLLM 会自动最大化利用显存。
    # 可以用这个参数来截取数据集的一部分，模拟不同规模的请求量。
    parser.add_argument('--concurrencys', type=int, nargs='+', default=[1, 4, 8, 16, 32])
    parser.add_argument('--output_file', type=str, required=True)
    parser.add_argument('--tp', type=int, default=1, help='Tensor Parallel size')
    parser.add_argument('--gpu_memory_utilization', type=float, default=0.9)
    
    args = parser.parse_args()
    
    with open(args.testset, "r") as f:
        testset = json.load(f)
    
    full_requests = list(testset.values())

    logger.info(f"Initializing vLLM engine with model: {args.model_uid}")
    llm = LLM(
        model=args.model_uid,
        tensor_parallel_size=args.tp,
        trust_remote_code=True,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=4096, # 根据显存情况调整，或者不传让它自动推导
        enforce_eager=False
    )

    all_results = {}
    
    # 通过切片数据集模拟不同规模的请求量
    for c in args.concurrencys:
        if c > len(full_requests):
            logger.warning(f"Concurrency {c} is larger than dataset size {len(full_requests)}. Using full dataset.")
            current_requests = full_requests
        else:
            current_requests = full_requests[:c]
            
        logger.info(f"Running benchmark with request count: {len(current_requests)}...")
        
        result = run_vllm_benchmark(llm, current_requests, args.max_output_len, c)
        all_results[c] = result

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"All benchmark results saved to {args.output_file}")