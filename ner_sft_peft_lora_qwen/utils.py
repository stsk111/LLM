from transformers import TrainerCallback, TrainerState, TrainingArguments
import os
import json
import numpy as np

# 定义一个处理 Numpy 类型的辅助类, 防止出现 TypeError: Object of type float32 is not JSON serializable
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

class AutoSaveLogCallback(TrainerCallback):
    """
    一个自定义回调，用于在每次 logging 时自动保存最新的日志到 json 文件。
    防止训练意外中断导致日志丢失。
    """
    def __init__(self, output_dir, file_name="training_logs.json"):
        self.output_path = os.path.join(output_dir, file_name)
        os.makedirs(output_dir, exist_ok=True)

    def on_log(self, args: TrainingArguments, state: TrainerState, control, logs=None, **kwargs):
        """
        每当 Trainer 触发日志记录（打印 loss 或 eval 结果）时调用
        """
        # 只在主进程（Process 0）进行写入，防止多卡训练时文件冲突
        if state.is_world_process_zero:
            # 获取当前的完整日志历史
            log_history = state.log_history
            
            # 使用 NumpyEncoder 写入文件
            try:
                with open(self.output_path, "w", encoding="utf-8") as f:
                    json.dump(log_history, f, indent=4, ensure_ascii=False, cls=NumpyEncoder)
            except Exception as e:
                print(f"⚠️ 自动保存日志失败: {e}")