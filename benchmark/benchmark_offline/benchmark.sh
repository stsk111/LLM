python benchmark_offline/benchmark_offline.py \
    --model_uid /root/autodl-tmp/models/Qwen/Qwen2.5-7B-Instruct \
    --testset dataset/bench_data.json \
    --max_output_len 64 \
    --concurrencys 1 4 8 16 32 \
    --tp 1 \
    --gpu_memory_utilization 0.9 \
    --output_file output/Qwen2.5-7B-Instruct_1GPU_offline.json