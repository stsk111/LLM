#!/usr/bin/env python3
"""
DocQA Pro - 主程序入口
智能文档检索问答助手
"""

import sys
import os
from pathlib import Path

# 设置环境变量 - 绕过 PyTorch 安全检查（本地可信环境）
os.environ['TORCH_LOAD_WEIGHTS_ONLY'] = '0'
os.environ['TORCH_ALLOW_VULNERABLE_LOAD'] = '1'
os.environ['TRANSFORMERS_NO_ADVISORY_CHECK'] = '1'

# Monkey patch: 绕过 transformers 的 PyTorch 版本检查
# 这是针对本地可信模型文件的临时解决方案
try:
    import torch
    import transformers.modeling_utils as modeling_utils
    
    # 保存原始的 torch.load 函数
    _original_torch_load = torch.load
    
    # 创建一个包装函数，强制使用 weights_only=False
    def patched_torch_load(*args, **kwargs):
        # 移除或覆盖 weights_only 参数
        kwargs.pop('weights_only', None)
        # 强制设置为 False
        return _original_torch_load(*args, weights_only=False, **kwargs)
    
    # 替换 torch.load
    torch.load = patched_torch_load
    print("✅ 已应用 torch.load 补丁（本地可信环境）")
    
except Exception as e:
    print(f"⚠️  补丁应用失败，可能仍会遇到加载问题: {e}")

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from ui.app import main

if __name__ == "__main__":
    print("🤖 DocQA - 智能文档检索问答助手")
    print("=" * 50)
    
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 应用已停止")
    except Exception as e:
        print(f"\n❌ 启动失败: {e}")
        sys.exit(1)