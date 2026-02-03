python benchmark_serving/benchmark_serving.py \
    --model_uid /root/autodl-tmp/models/Qwen/Qwen2.5-7B-Instruct \
    --testset dataset/bench_data.json \
    --concurrencys 1 4 8 16 32 \
    --output_file output/Qwen2.5-7B-Instruct_1GPU_serving.json