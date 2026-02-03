# PYTHONUNBUFFERED=1强制显示输出
PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 vllm serve /root/autodl-tmp/models/Qwen/Qwen2.5-7B-Instruct \
    --port 8000 \
    --max-model-len 32000 \
    --tensor-parallel-size 1 