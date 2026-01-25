"""
DocQA Pro - LLM引擎
vLLM封装的ChatLLM类，支持流式输出
"""

from typing import Iterator, List, Dict, Any, Optional
import torch
from vllm import LLM, SamplingParams

from config import LLM_MODEL_PATH, LLM_TENSOR_PARALLEL_SIZE, SAMPLING_PARAMS


class ChatLLM:
    """基于vLLM的聊天LLM封装"""
    
    def __init__(
        self, 
        model_path: str = str(LLM_MODEL_PATH),
        tensor_parallel_size: int = LLM_TENSOR_PARALLEL_SIZE,
        **kwargs
    ):
        """
        初始化vLLM模型
        
        Args:
            model_path: 模型路径
            tensor_parallel_size: 张量并行大小
            **kwargs: 其他vLLM参数
        """
        self.model_path = model_path
        self.tensor_parallel_size = tensor_parallel_size
        self.llm = None
        
        # 合并默认参数
        self.vllm_kwargs = {
            'tensor_parallel_size': tensor_parallel_size,
            'trust_remote_code': True,
            'dtype': 'float16' if torch.cuda.is_available() else 'float32',
            'gpu_memory_utilization': 0.90,  # 使用90%的GPU内存
            'max_model_len': 8192,  # 限制最大序列长度，节省KV cache
            **kwargs
        }
        
        self._load_model()
    
    def _load_model(self):
        """加载vLLM模型"""
        try:
            print(f"🔄 加载LLM模型: {self.model_path}")
            print(f"   张量并行大小: {self.tensor_parallel_size}")
            
            self.llm = LLM(
                model=self.model_path,
                **self.vllm_kwargs
            )
            
            print("✅ LLM模型加载成功")
            
        except Exception as e:
            print(f"❌ LLM模型加载失败: {e}")
            raise
    
    def _format_messages(
        self, 
        user_message: str, 
        system_prompt: str = "",
        chat_history: List[Dict[str, str]] = None
    ) -> str:
        """
        格式化消息为Qwen格式
        
        Args:
            user_message: 用户消息
            system_prompt: 系统提示
            chat_history: 聊天历史 [{"role": "user", "content": "..."}, ...]
            
        Returns:
            格式化的提示文本
        """
        messages = []
        
        # 添加系统提示
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        # 添加聊天历史
        if chat_history:
            messages.extend(chat_history)
        
        # 添加当前用户消息
        messages.append({"role": "user", "content": user_message})
        
        # 使用Qwen格式化
        formatted_prompt = ""
        for message in messages:
            role = message["role"]
            content = message["content"]
            
            if role == "system":
                formatted_prompt += f"<|im_start|>system\n{content}<|im_end|>\n"
            elif role == "user":
                formatted_prompt += f"<|im_start|>user\n{content}<|im_end|>\n"
            elif role == "assistant":
                formatted_prompt += f"<|im_start|>assistant\n{content}<|im_end|>\n"
        
        # 添加assistant开始标记
        formatted_prompt += "<|im_start|>assistant\n"
        
        return formatted_prompt
    
    def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        chat_history: List[Dict[str, str]] = None,
        sampling_params: Dict[str, Any] = None,
        stream: bool = True
    ) -> Iterator[str]:
        """
        生成回答（支持流式输出）
        
        Args:
            prompt: 用户输入
            system_prompt: 系统提示
            chat_history: 聊天历史
            sampling_params: 采样参数
            stream: 是否流式输出
            
        Yields:
            生成的文本片段
        """
        if not self.llm:
            raise RuntimeError("LLM模型未加载")
        
        try:
            # 格式化输入
            formatted_prompt = self._format_messages(
                prompt, system_prompt, chat_history
            )
            
            # 设置采样参数
            if sampling_params is None:
                sampling_params = SAMPLING_PARAMS.copy()
            
            sampling_config = SamplingParams(**sampling_params)
            
            if stream:
                # 流式生成（vLLM暂不直接支持流式，模拟实现）
                outputs = self.llm.generate(
                    [formatted_prompt], 
                    sampling_config
                )
                
                # 返回完整结果（vLLM限制）
                if outputs and outputs[0].outputs:
                    generated_text = outputs[0].outputs[0].text
                    # 模拟流式输出
                    for i in range(0, len(generated_text), 10):
                        yield generated_text[i:i+10]
                else:
                    yield ""
            else:
                # 非流式生成
                outputs = self.llm.generate(
                    [formatted_prompt], 
                    sampling_config
                )
                
                if outputs and outputs[0].outputs:
                    yield outputs[0].outputs[0].text
                else:
                    yield ""
                    
        except Exception as e:
            print(f"❌ LLM生成失败: {e}")
            yield f"生成失败: {str(e)}"
    
    def generate_simple(
        self,
        prompt: str,
        system_prompt: str = "",
        **kwargs
    ) -> str:
        """
        简单生成（非流式）
        
        Args:
            prompt: 用户输入
            system_prompt: 系统提示
            **kwargs: 采样参数
            
        Returns:
            生成的完整文本
        """
        sampling_params = SAMPLING_PARAMS.copy()
        sampling_params.update(kwargs)
        
        result = ""
        for chunk in self.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            sampling_params=sampling_params,
            stream=False
        ):
            result += chunk
        
        return result.strip()
    
    def batch_generate(
        self,
        prompts: List[str],
        system_prompts: List[str] = None,
        sampling_params: Dict[str, Any] = None
    ) -> List[str]:
        """
        批量生成
        
        Args:
            prompts: 提示列表
            system_prompts: 系统提示列表
            sampling_params: 采样参数
            
        Returns:
            生成结果列表
        """
        if not self.llm:
            raise RuntimeError("LLM模型未加载")
        
        # 格式化所有提示
        formatted_prompts = []
        for i, prompt in enumerate(prompts):
            system_prompt = ""
            if system_prompts and i < len(system_prompts):
                system_prompt = system_prompts[i]
            
            formatted_prompt = self._format_messages(prompt, system_prompt)
            formatted_prompts.append(formatted_prompt)
        
        # 设置采样参数
        if sampling_params is None:
            sampling_params = SAMPLING_PARAMS.copy()
        
        sampling_config = SamplingParams(**sampling_params)
        
        # 批量生成
        try:
            outputs = self.llm.generate(formatted_prompts, sampling_config)
            
            results = []
            for output in outputs:
                if output.outputs:
                    results.append(output.outputs[0].text.strip())
                else:
                    results.append("")
            
            return results
            
        except Exception as e:
            print(f"❌ 批量生成失败: {e}")
            return [""] * len(prompts)


def create_chat_llm(model_path: str = None, **kwargs) -> ChatLLM:
    """
    创建ChatLLM实例的便捷函数
    
    Args:
        model_path: 模型路径，为None时使用配置中的路径
        **kwargs: 其他参数
        
    Returns:
        ChatLLM实例
    """
    if model_path is None:
        model_path = str(LLM_MODEL_PATH)
    
    return ChatLLM(model_path=model_path, **kwargs)