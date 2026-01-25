"""
DocQA Pro - Gradio界面
左右分栏布局，支持PDF上传和流式问答
"""

import os
import gradio as gr
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any, Generator
import tempfile

# 导入核心模块
from core.ingestion import PDFIngestionPipeline
from core.retrieval import create_retrieval_system, HybridRetriever
from core.reranker import create_reranker
from llm_engine.chat_llm import create_chat_llm
from core.qa_chain import create_qa_chain
from core.cache_manager import create_cache_manager

from config import (
    GRADIO_THEME, CHATBOT_HEIGHT, ENABLE_STREAMING,
    RETRIEVAL_TOP_K, RERANK_TOP_N, RERANK_SCORE_THRESHOLD, ENABLE_CACHE
)


class DocQAApp:
    """DocQA应用主类"""
    
    def __init__(self):
        """初始化应用"""
        self.pdf_pipeline = None
        self.qa_chain = None
        self.current_chunks = []
        self.chat_history = []
        self.processing = False
        
        # 全局单例模型（避免重复加载）
        self.embedding_engine = None
        self.reranker = None 
        self.chat_llm = None
        self._models_loaded = False
        
        # 缓存管理器
        self.cache_manager = create_cache_manager() if ENABLE_CACHE else None
        
        print("🚀 DocQA应用初始化...")
    
    def _update_progress(self, message: str, current: int = 0, total: int = 100):
        """进度更新回调"""
        progress = current / total if total > 0 else 0
        print(f"进度 {current}/{total}: {message}")
    
    def _format_sources_display(self, sources: List[Dict]) -> str:
        """
        格式化来源信息用于UI展示
        
        Args:
            sources: 来源信息列表
            
        Returns:
            格式化的来源文本
        """
        if not sources:
            return ""
        
        sources_text = "\n\n---\n**📚 参考来源:**\n\n"
        
        for source in sources:
            index = source['index']
            page = source['page']
            score = source['rerank_score']
            content = source.get('content', source.get('content_preview', ''))
            
            # 限制每个片段的显示长度（避免过长）
            if len(content) > 300:
                content = content[:300] + "..."
            
            # 格式化单个来源
            sources_text += f"**[{index}]** 第{page}页 (相关性: {score:.2f})\n"
            sources_text += f"> {content}\n\n"
        
        return sources_text
    
    def _initialize_models(self):
        """初始化模型（全局单例，避免显存重复占用）"""
        if self._models_loaded:
            print("♻️  复用已加载的模型")
            return
        
        try:
            print("🔄 首次加载模型...")
            
            # 加载Reranker模型（单例）
            if self.reranker is None:
                self.reranker = create_reranker()
            
            # 加载LLM模型（单例）
            if self.chat_llm is None:
                self.chat_llm = create_chat_llm()
            
            self._models_loaded = True
            print("✅ 模型初始化完成（全局单例）")
            
        except Exception as e:
            error_msg = f"模型初始化失败: {str(e)}"
            print(f"❌ {error_msg}")
            raise RuntimeError(error_msg)
    
    def process_pdf(self, file) -> Tuple[str, str, gr.update]:
        """
        处理上传的PDF文件（支持缓存加速）
        
        Args:
            file: 上传的文件对象
            
        Returns:
            (状态消息, 文档信息, 聊天区域更新)
        """
        if self.processing:
            return "正在处理中，请稍候...", "", gr.update()
        
        if file is None:
            return "请选择PDF文件", "", gr.update()
        
        self.processing = True
        from_cache = False
        
        try:
            # 获取文件路径
            file_path = file if isinstance(file, str) else file.name
            
            # 初始化模型（全局单例，避免重复加载）
            self._initialize_models()
            
            # 尝试从缓存加载
            if self.cache_manager and self.cache_manager.cache_exists(file_path):
                print("🎯 检测到缓存，尝试快速加载...")
                
                # 确保embedding_engine已初始化
                if self.embedding_engine is None:
                    from core.retrieval import EmbeddingEngine
                    self.embedding_engine = EmbeddingEngine()
                
                cache_result = self.cache_manager.load_cache(
                    file_path,
                    self.embedding_engine.embeddings
                )
                
                if cache_result:
                    faiss_index, chunks, metadata = cache_result
                    self.current_chunks = chunks
                    
                    # 重建BM25索引（BM25索引很轻量，重建很快）
                    from langchain_community.retrievers import BM25Retriever
                    from config import RETRIEVAL_TOP_K
                    sparse_retriever = BM25Retriever.from_documents(chunks)
                    sparse_retriever.k = RETRIEVAL_TOP_K
                    
                    # 创建混合检索器
                    hybrid_retriever = HybridRetriever(faiss_index, chunks)
                    
                    # 创建问答链（复用全局模型）
                    self.qa_chain = create_qa_chain(self.chat_llm, hybrid_retriever, self.reranker)
                    
                    # 清空聊天历史
                    self.chat_history = []
                    
                    from_cache = True
                    stats = metadata
                    
                    print("✅ 从缓存加载成功！")
            
            # 缓存不存在或加载失败，正常处理
            if not from_cache:
                print("📄 缓存不存在，开始处理PDF...")
                
                # 初始化PDF处理管道
                if self.pdf_pipeline is None:
                    self.pdf_pipeline = PDFIngestionPipeline(
                        progress_callback=self._update_progress
                    )
                
                # 处理PDF
                result = self.pdf_pipeline.process_pdf(file_path)
                
                if not result["success"]:
                    return f"处理失败: {result['error']}", "", gr.update()
                
                # 保存文档片段
                self.current_chunks = result["chunks"]
                
                # 构建检索系统
                status_msg = "构建检索系统..."
                print(status_msg)
                
                # 创建检索组件（只重建索引，复用模型）
                embedding_engine, index_builder, hybrid_retriever = create_retrieval_system(
                    self.current_chunks
                )
                
                # 保存embedding_engine供缓存使用
                if self.embedding_engine is None:
                    self.embedding_engine = embedding_engine
                
                # 创建问答链（复用全局模型）
                self.qa_chain = create_qa_chain(self.chat_llm, hybrid_retriever, self.reranker)
                
                # 清空聊天历史
                self.chat_history = []
                
                stats = result["stats"]
                
                # 保存到缓存
                if self.cache_manager:
                    print("💾 保存到缓存...")
                    self.cache_manager.save_cache(
                        file_path,
                        index_builder.vector_store,
                        self.current_chunks,
                        stats
                    )
            
            # 生成文档信息
            cache_indicator = "⚡ **从缓存加载**\n\n" if from_cache else ""
            doc_info = f"""
{cache_indicator}📊 **文档处理完成**

- **总页数**: {stats['total_pages']}
- **文本块数**: {stats['total_chunks']}
- **平均每页块数**: {stats['avg_chunks_per_page']:.1f}
- **块大小**: {stats['chunk_size']} tokens
- **块重叠**: {stats['chunk_overlap']} tokens

✅ **系统就绪，可以开始问答！**
            """.strip()
            
            status_prefix = "⚡ 从缓存加载完成，系统就绪" if from_cache else "✅ PDF处理完成，系统就绪"
            return status_prefix, doc_info, gr.update(value=[])
            
        except FileNotFoundError:
            return "❌ 文件未找到，请重新上传PDF文件", "", gr.update()
        except ValueError as e:
            return f"❌ 文件格式错误: {str(e)}", "", gr.update()
        except RuntimeError as e:
            return f"❌ 系统错误: {str(e)}", "", gr.update()
        except MemoryError:
            return "❌ 内存不足，请尝试上传更小的文件或重启应用", "", gr.update()
        except Exception as e:
            error_msg = f"❌ 处理失败: {str(e)}"
            print(f"❌ {error_msg}")
            import traceback
            traceback.print_exc()
            return "❌ 文件处理失败，请检查文件格式并重试", "", gr.update()
        
        finally:
            self.processing = False
    
    def chat_response(
        self, 
        message: str, 
        history: List,
        top_k: int,
        top_n: int,
        threshold: float
    ) -> Tuple[str, List]:
        """
        处理聊天消息（非流式）
        
        Args:
            message: 用户消息
            history: 聊天历史（Gradio 6.0格式）
            top_k: 检索数量
            top_n: 重排数量
            threshold: 分数阈值
            
        Returns:
            ("", 更新的历史)
        """
        if not self.qa_chain:
            error_msg = "请先上传并处理PDF文件"
            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": error_msg})
            return "", history
        
        if not message.strip():
            return "", history
        
        try:
            # 转换历史格式 - 兼容Gradio 6.0的字典格式
            chat_history = []
            for i in range(0, len(history), 2):
                if i + 1 < len(history):
                    user_msg = history[i].get("content", "") if isinstance(history[i], dict) else history[i][0]
                    assistant_msg = history[i+1].get("content", "") if isinstance(history[i+1], dict) else history[i+1][1]
                    chat_history.append((user_msg, assistant_msg))
            
            # 执行问答
            result = self.qa_chain.ask(
                question=message,
                chat_history=chat_history,
                top_n=top_n,
                score_threshold=threshold,
                stream=False
            )
            
            # 构建回答
            answer = result.get("answer", "无法生成回答")
            sources = result.get("sources", [])
            
            # 添加来源信息（使用新的格式化函数）
            if sources:
                answer += self._format_sources_display(sources)
            
            # 更新历史 - Gradio 6.0格式
            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": answer})
            
        except MemoryError:
            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": "❌ 显存不足，请重启应用或尝试更简单的问题。"})
        except TimeoutError:
            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": "❌ 响应超时，请稍后重试。"})
        except Exception as e:
            error_msg = f"回答生成失败: {str(e)}"
            print(f"❌ {error_msg}")
            history.append({"role": "user", "content": message})
            if "model" in str(e).lower() or "cuda" in str(e).lower():
                history.append({"role": "assistant", "content": "❌ 模型加载异常，请重启应用。"})
            else:
                history.append({"role": "assistant", "content": "❌ 抱歉，处理您的问题时出现了错误，请重试。"})
        
        return "", history
    
    def chat_response_stream(
        self,
        message: str,
        history: List, 
        top_k: int,
        top_n: int,
        threshold: float
    ) -> Generator[Tuple[str, List], None, None]:
        """
        处理聊天消息（流式）
        
        Args:
            message: 用户消息
            history: 聊天历史（Gradio 6.0格式）
            top_k: 检索数量
            top_n: 重排数量  
            threshold: 分数阈值
            
        Yields:
            ("", 更新的历史)
        """
        if not self.qa_chain:
            error_msg = "请先上传并处理PDF文件"
            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": error_msg})
            yield "", history
            return
        
        if not message.strip():
            yield "", history
            return
        
        try:
            # 转换历史格式 - 兼容Gradio 6.0
            chat_history = []
            for i in range(0, len(history), 2):
                if i + 1 < len(history):
                    user_msg = history[i].get("content", "") if isinstance(history[i], dict) else history[i][0]
                    assistant_msg = history[i+1].get("content", "") if isinstance(history[i+1], dict) else history[i+1][1]
                    chat_history.append((user_msg, assistant_msg))
            
            # 执行问答（流式）
            result = self.qa_chain.ask(
                question=message,
                chat_history=chat_history,
                top_n=top_n,
                score_threshold=threshold,
                stream=True
            )
            
            # 初始化回答 - Gradio 6.0格式
            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": ""})
            answer_stream = result.get("answer_stream")
            sources = result.get("sources", [])
            
            if answer_stream:
                # 流式更新回答
                current_answer = ""
                for chunk in answer_stream:
                    current_answer += chunk
                    history[-1]["content"] = current_answer
                    yield "", history
                
                # 添加来源信息（使用新的格式化函数）
                if sources:
                    sources_text = self._format_sources_display(sources)
                    history[-1]["content"] = current_answer + sources_text
                    yield "", history
            else:
                # 非流式回退
                answer = result.get("answer", "无法生成回答")
                if sources:
                    answer += self._format_sources_display(sources)
                
                history[-1]["content"] = answer
                yield "", history
                
        except MemoryError:
            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": "❌ 显存不足，请重启应用或尝试更简单的问题。"})
            yield "", history
        except TimeoutError:
            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": "❌ 响应超时，请稍后重试。"})
            yield "", history
        except Exception as e:
            error_msg = f"回答生成失败: {str(e)}"
            print(f"❌ {error_msg}")
            history.append({"role": "user", "content": message})
            if "model" in str(e).lower() or "cuda" in str(e).lower():
                history.append({"role": "assistant", "content": "❌ 模型加载异常，请重启应用。"})
            else:
                history.append({"role": "assistant", "content": "❌ 抱歉，处理您的问题时出现了错误，请重试。"})
            yield "", history
    
    def clear_chat(self):
        """清空聊天历史"""
        self.chat_history = []
        return []
    
    def build_interface(self) -> gr.Blocks:
        """构建Gradio界面"""
        with gr.Blocks(
            title="DocQA Pro - 智能文档问答助手"
        ) as demo:
            
            gr.Markdown("# 🤖 DocQA Pro - 智能文档问答助手")
            gr.Markdown("基于RAG技术的本地文档问答系统，支持PDF上传和智能问答")
            
            with gr.Row():
                # 左侧栏 - 控制面板
                with gr.Column(scale=1):
                    gr.Markdown("## 📁 文档上传")
                    
                    # 文件上传
                    file_upload = gr.File(
                        label="选择PDF文件",
                        file_types=[".pdf"],
                        type="filepath"
                    )
                    
                    process_btn = gr.Button("📊 处理文档", variant="primary")
                    
                    # 处理状态
                    status_box = gr.Textbox(
                        label="处理状态",
                        value="等待上传PDF文件...",
                        interactive=False,
                        lines=2
                    )
                    
                    # 文档信息
                    doc_info = gr.Markdown(
                        value="",
                        label="文档信息"
                    )
                    
                    gr.Markdown("## ⚙️ 参数设置")
                    
                    # 参数控制
                    top_k = gr.Slider(
                        minimum=1, maximum=20, value=RETRIEVAL_TOP_K,
                        step=1, label="检索数量 (Top K)"
                    )
                    
                    top_n = gr.Slider(
                        minimum=1, maximum=10, value=RERANK_TOP_N,
                        step=1, label="重排数量 (Top N)"
                    )
                    
                    threshold = gr.Slider(
                        minimum=0.0, maximum=1.0, value=RERANK_SCORE_THRESHOLD,
                        step=0.1, label="相关性阈值"
                    )
                    
                    # 清除按钮
                    clear_btn = gr.Button("🗑️ 清空对话", variant="secondary")
                
                # 右侧栏 - 对话区域  
                with gr.Column(scale=2):
                    gr.Markdown("## 💬 智能问答")
                    
                    # 聊天界面
                    chatbot = gr.Chatbot(
                        height=CHATBOT_HEIGHT,
                        label="对话历史",
                        show_label=False
                    )
                    
                    # 输入框
                    msg = gr.Textbox(
                        label="输入您的问题",
                        placeholder="请输入关于文档的问题...",
                        lines=2
                    )
                    
                    with gr.Row():
                        submit_btn = gr.Button("发送", variant="primary")
                        gr.Button("示例问题", variant="secondary", visible=False)
            
            # 事件绑定
            process_btn.click(
                fn=self.process_pdf,
                inputs=[file_upload],
                outputs=[status_box, doc_info, chatbot]
            )
            
            # 选择流式或非流式
            if ENABLE_STREAMING:
                msg.submit(
                    fn=self.chat_response_stream,
                    inputs=[msg, chatbot, top_k, top_n, threshold],
                    outputs=[msg, chatbot]
                )
                submit_btn.click(
                    fn=self.chat_response_stream,
                    inputs=[msg, chatbot, top_k, top_n, threshold],
                    outputs=[msg, chatbot]
                )
            else:
                msg.submit(
                    fn=self.chat_response,
                    inputs=[msg, chatbot, top_k, top_n, threshold],
                    outputs=[msg, chatbot]
                )
                submit_btn.click(
                    fn=self.chat_response,
                    inputs=[msg, chatbot, top_k, top_n, threshold],
                    outputs=[msg, chatbot]
                )
            
            clear_btn.click(
                fn=self.clear_chat,
                outputs=[chatbot]
            )
        
        return demo
    
    def launch(self, **kwargs):
        """启动应用"""
        demo = self.build_interface()
        # 将theme参数传递给launch方法（Gradio 6.0）
        launch_kwargs = {
            'server_name': kwargs.get('server_name', '0.0.0.0'),
            'server_port': kwargs.get('server_port', 7860),
            'share': kwargs.get('share', False),
            'debug': kwargs.get('debug', False)
        }
        demo.launch(**launch_kwargs)


def main():
    """主函数"""
    app = DocQAApp()
    
    print("🚀 启动DocQA Pro应用...")
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        debug=False
    )


if __name__ == "__main__":
    main()