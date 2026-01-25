"""
DocQA Pro - 问答链路
整合检索、重排、生成的完整问答管道
"""

from typing import List, Dict, Any, Optional, Tuple
from langchain_core.documents import Document

from core.retrieval import HybridRetriever
from core.reranker import BGEReranker
from llm_engine.chat_llm import ChatLLM
from config import (
    SYSTEM_PROMPT_TEMPLATE, QUERY_REWRITE_PROMPT,
    RERANK_TOP_N, RERANK_SCORE_THRESHOLD
)


class DocQAChain:
    """文档问答链路"""
    
    def __init__(
        self,
        llm: ChatLLM,
        retriever: HybridRetriever,
        reranker: BGEReranker
    ):
        """
        初始化问答链
        
        Args:
            llm: 语言模型
            retriever: 混合检索器
            reranker: 重排模型
        """
        self.llm = llm
        self.retriever = retriever
        self.reranker = reranker
        
        print("✅ DocQA问答链初始化完成")
    
    def _rewrite_query(
        self,
        question: str,
        chat_history: List[Tuple[str, str]] = None
    ) -> str:
        """
        基于历史对话重写查询
        
        Args:
            question: 当前问题
            chat_history: 历史对话 [(user_msg, assistant_msg), ...]
            
        Returns:
            重写后的查询
        """
        if not chat_history:
            return question
        
        try:
            # 格式化历史对话
            history_text = ""
            for user_msg, assistant_msg in chat_history[-3:]:  # 只用最近3轮对话
                history_text += f"用户: {user_msg}\n助手: {assistant_msg}\n"
            
            # 构造重写提示
            rewrite_prompt = QUERY_REWRITE_PROMPT.format(
                chat_history=history_text,
                question=question
            )
            
            # 生成重写查询
            rewritten = self.llm.generate_simple(
                rewrite_prompt,
                system_prompt="你是一个查询重写助手，帮助将不完整的问题改写为完整的搜索查询。",
                temperature=0.3,
                max_tokens=200
            )
            
            # 清理重写结果
            rewritten = rewritten.strip()
            if rewritten and len(rewritten) > 10:
                print(f"🔄 查询重写: '{question}' -> '{rewritten}'")
                return rewritten
            else:
                return question
                
        except Exception as e:
            print(f"⚠️  查询重写失败: {e}")
            return question
    
    def _format_sources(self, ranked_docs: List[Tuple[Document, float]]) -> Tuple[str, List[Dict]]:
        """
        格式化参考来源
        
        Args:
            ranked_docs: 重排后的文档和分数列表
            
        Returns:
            (格式化的上下文文本, 来源信息列表)
        """
        if not ranked_docs:
            return "无相关文档。", []
        
        context_parts = []
        sources_info = []
        
        for i, (doc, score) in enumerate(ranked_docs):
            # 提取元数据
            page = doc.metadata.get('page', '未知')
            chunk_id = doc.metadata.get('chunk_id', f'chunk_{i}')
            
            # 格式化文档内容
            content = doc.page_content.strip()
            if len(content) > 500:  # 限制长度
                content = content[:500] + "..."
            
            # 添加到上下文
            context_parts.append(f"[文档{i+1}]\n{content}\n")
            
            # 记录来源信息（保存完整内容用于UI展示）
            sources_info.append({
                'index': i + 1,
                'page': page,
                'chunk_id': chunk_id,
                'content': content,  # 完整内容
                'content_preview': content[:100] + "..." if len(content) > 100 else content,
                'rerank_score': score,
                'fusion_score': doc.metadata.get('fusion_score', 0.0)
            })
        
        context_text = "\n".join(context_parts)
        return context_text, sources_info
    
    def _format_chat_history(self, chat_history: List[Tuple[str, str]]) -> str:
        """
        格式化聊天历史
        
        Args:
            chat_history: 历史对话
            
        Returns:
            格式化的历史文本
        """
        if not chat_history:
            return "无历史对话。"
        
        history_parts = []
        for i, (user_msg, assistant_msg) in enumerate(chat_history[-5:]):  # 最多5轮
            history_parts.append(f"第{i+1}轮:")
            history_parts.append(f"用户: {user_msg}")
            history_parts.append(f"助手: {assistant_msg}")
            history_parts.append("")
        
        return "\n".join(history_parts)
    
    def ask(
        self,
        question: str,
        chat_history: List[Tuple[str, str]] = None,
        top_n: int = RERANK_TOP_N,
        score_threshold: float = RERANK_SCORE_THRESHOLD,
        stream: bool = True
    ) -> Dict[str, Any]:
        """
        执行问答
        
        Args:
            question: 用户问题
            chat_history: 聊天历史 [(user_msg, assistant_msg), ...]
            top_n: 重排后保留的文档数
            score_threshold: 相关性分数阈值
            stream: 是否流式返回
            
        Returns:
            问答结果字典
        """
        try:
            # Step 1: 查询重写
            rewritten_query = self._rewrite_query(question, chat_history)
            
            # Step 2: 混合检索
            print(f"🔍 执行检索: {rewritten_query}")
            retrieved_docs = self.retriever.retrieve(rewritten_query)
            
            if not retrieved_docs:
                return {
                    "answer": "根据提供的文档，我无法回答这个问题。",
                    "sources": [],
                    "rewritten_query": rewritten_query,
                    "retrieval_count": 0,
                    "rerank_count": 0
                }
            
            # Step 3: 重排
            print(f"🔄 重排 {len(retrieved_docs)} 个文档...")
            ranked_docs = self.reranker.rerank(
                rewritten_query, 
                retrieved_docs,
                top_n=top_n,
                score_threshold=score_threshold
            )
            
            if not ranked_docs:
                return {
                    "answer": "根据提供的文档，我无法回答这个问题。",
                    "sources": [],
                    "rewritten_query": rewritten_query,
                    "retrieval_count": len(retrieved_docs),
                    "rerank_count": 0
                }
            
            # Step 4: 格式化上下文和来源
            context_text, sources_info = self._format_sources(ranked_docs)
            history_text = self._format_chat_history(chat_history)
            
            # Step 5: 构建最终提示
            final_prompt = SYSTEM_PROMPT_TEMPLATE.format(
                context=context_text,
                chat_history=history_text,
                question=question
            )
            
            # Step 6: 生成回答
            print("🤖 生成回答...")
            
            if stream:
                # 流式生成
                answer_generator = self.llm.generate(
                    prompt="请根据上述信息回答问题。",
                    system_prompt=final_prompt,
                    stream=True
                )
                
                return {
                    "answer_stream": answer_generator,
                    "sources": sources_info,
                    "rewritten_query": rewritten_query,
                    "retrieval_count": len(retrieved_docs),
                    "rerank_count": len(ranked_docs),
                    "context": context_text
                }
            else:
                # 非流式生成
                answer = self.llm.generate_simple(
                    prompt="请根据上述信息回答问题。",
                    system_prompt=final_prompt
                )
                
                return {
                    "answer": answer,
                    "sources": sources_info,
                    "rewritten_query": rewritten_query,
                    "retrieval_count": len(retrieved_docs),
                    "rerank_count": len(ranked_docs),
                    "context": context_text
                }
                
        except Exception as e:
            error_msg = f"问答处理失败: {str(e)}"
            print(f"❌ {error_msg}")
            
            return {
                "answer": "抱歉，处理您的问题时出现了错误，请稍后重试。",
                "sources": [],
                "error": error_msg,
                "rewritten_query": question,
                "retrieval_count": 0,
                "rerank_count": 0
            }
    
    def batch_ask(
        self,
        questions: List[str],
        chat_histories: List[List[Tuple[str, str]]] = None
    ) -> List[Dict[str, Any]]:
        """
        批量问答
        
        Args:
            questions: 问题列表
            chat_histories: 对应的聊天历史列表
            
        Returns:
            问答结果列表
        """
        if chat_histories is None:
            chat_histories = [None] * len(questions)
        
        results = []
        for i, (question, history) in enumerate(zip(questions, chat_histories)):
            print(f"\n处理问题 {i+1}/{len(questions)}: {question}")
            result = self.ask(question, history, stream=False)
            results.append(result)
        
        return results
    
    def get_chat_response(
        self,
        question: str,
        chat_history: List[Tuple[str, str]] = None
    ) -> Tuple[str, List[Dict], str]:
        """
        获取聊天响应（简化接口）
        
        Args:
            question: 问题
            chat_history: 聊天历史
            
        Returns:
            (回答, 来源列表, 重写查询)
        """
        result = self.ask(question, chat_history, stream=False)
        return (
            result.get("answer", "无法生成回答"),
            result.get("sources", []),
            result.get("rewritten_query", question)
        )


def create_qa_chain(
    llm: ChatLLM,
    retriever: HybridRetriever,
    reranker: BGEReranker
) -> DocQAChain:
    """
    创建问答链的便捷函数
    
    Args:
        llm: 语言模型
        retriever: 检索器
        reranker: 重排器
        
    Returns:
        问答链实例
    """
    return DocQAChain(llm, retriever, reranker)