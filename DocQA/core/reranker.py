"""
DocQA Pro - 重排模块
BGE-Reranker模型封装，对检索结果进行重新排序
"""

from typing import List, Tuple, Dict, Any
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from langchain_core.documents import Document

from config import RERANKER_MODEL_PATH, RERANKER_DEVICE, RERANKER_BATCH_SIZE


class BGEReranker:
    """BGE Reranker模型封装"""
    
    def __init__(self, model_path: str = str(RERANKER_MODEL_PATH)):
        """
        初始化Reranker模型
        
        Args:
            model_path: 模型路径
        """
        self.model_path = model_path
        self.device = RERANKER_DEVICE
        self.batch_size = RERANKER_BATCH_SIZE
        
        self.tokenizer = None
        self.model = None
        
        self._load_model()
    
    def _load_model(self):
        """加载Reranker模型和tokenizer"""
        try:
            print(f"🔄 加载Reranker模型: {self.model_path}")
            
            # 加载tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True
            )
            
            # 加载模型
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                torch_dtype=torch.float16 if self.device == 'cuda' else torch.float32,
                weights_only=False  # 显式关闭仅权重加载，绕过版本检查
            )
            
            # 移动到指定设备
            if torch.cuda.is_available() and self.device == 'cuda':
                self.model = self.model.to(self.device)
            
            self.model.eval()
            print("✅ Reranker模型加载成功")
            
        except Exception as e:
            print(f"❌ Reranker模型加载失败: {e}")
            raise
    
    def _compute_scores(self, query: str, texts: List[str]) -> List[float]:
        """
        计算query与文本列表的相关性分数
        
        Args:
            query: 查询文本
            texts: 文档文本列表
            
        Returns:
            相关性分数列表
        """
        if not self.model or not self.tokenizer:
            raise RuntimeError("Reranker模型未加载")
        
        scores = []
        
        # 批处理计算分数
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            batch_pairs = [(query, text) for text in batch_texts]
            
            try:
                # Tokenize输入对
                inputs = self.tokenizer(
                    batch_pairs,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                    max_length=512
                )
                
                # 移动到设备
                if torch.cuda.is_available() and self.device == 'cuda':
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # 推理
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    batch_scores = outputs.logits.squeeze().cpu().numpy()
                
                # 处理单个样本的情况
                if len(batch_texts) == 1:
                    batch_scores = [float(batch_scores)]
                else:
                    batch_scores = batch_scores.tolist()
                
                scores.extend(batch_scores)
                
            except Exception as e:
                print(f"⚠️  批次 {i//self.batch_size + 1} 处理失败: {e}")
                # 为失败的批次添加默认分数
                scores.extend([0.0] * len(batch_texts))
        
        return scores
    
    def rerank(
        self, 
        query: str, 
        documents: List[Document],
        top_n: int = 5,
        score_threshold: float = 0.0
    ) -> List[Tuple[Document, float]]:
        """
        对文档重新排序
        
        Args:
            query: 查询文本
            documents: 文档列表
            top_n: 返回前N个结果
            score_threshold: 分数阈值，低于此分数的结果将被过滤
            
        Returns:
            排序后的(文档, 分数)列表
        """
        if not documents:
            return []
        
        try:
            print(f"🔄 重排 {len(documents)} 个文档...")
            
            # 提取文档文本
            texts = [doc.page_content for doc in documents]
            
            # 计算相关性分数
            scores = self._compute_scores(query, texts)
            
            # 创建(文档, 分数)对
            doc_score_pairs = list(zip(documents, scores))
            
            # 按分数降序排序
            doc_score_pairs.sort(key=lambda x: x[1], reverse=True)
            
            # 应用分数阈值过滤
            filtered_pairs = [
                (doc, score) for doc, score in doc_score_pairs 
                if score >= score_threshold
            ]
            
            # 取Top-N
            top_results = filtered_pairs[:top_n]
            
            print(f"✅ 重排完成，返回 {len(top_results)} 个结果")
            
            # 在文档metadata中添加重排分数
            for doc, score in top_results:
                doc.metadata['rerank_score'] = score
            
            return top_results
            
        except Exception as e:
            print(f"❌ 重排失败: {e}")
            # 返回原始文档，但限制数量
            return [(doc, 0.0) for doc in documents[:top_n]]
    
    def batch_rerank(
        self,
        queries: List[str],
        document_lists: List[List[Document]],
        top_n: int = 5,
        score_threshold: float = 0.0
    ) -> List[List[Tuple[Document, float]]]:
        """
        批量重排多个查询的文档
        
        Args:
            queries: 查询列表
            document_lists: 每个查询对应的文档列表
            top_n: 每个查询返回的结果数
            score_threshold: 分数阈值
            
        Returns:
            每个查询的重排结果列表
        """
        results = []
        
        for query, docs in zip(queries, document_lists):
            reranked = self.rerank(query, docs, top_n, score_threshold)
            results.append(reranked)
        
        return results
    
    def get_relevance_scores(self, query: str, documents: List[Document]) -> Dict[str, float]:
        """
        获取查询与所有文档的相关性分数字典
        
        Args:
            query: 查询文本
            documents: 文档列表
            
        Returns:
            文档ID -> 相关性分数的映射
        """
        if not documents:
            return {}
        
        texts = [doc.page_content for doc in documents]
        scores = self._compute_scores(query, texts)
        
        # 创建文档ID到分数的映射
        score_dict = {}
        for i, (doc, score) in enumerate(zip(documents, scores)):
            # 使用chunk_id或者索引作为key
            doc_id = doc.metadata.get('chunk_id', f'doc_{i}')
            score_dict[doc_id] = score
        
        return score_dict


def create_reranker(model_path: str = None) -> BGEReranker:
    """
    创建Reranker实例的便捷函数
    
    Args:
        model_path: 模型路径，为None时使用配置中的路径
        
    Returns:
        Reranker实例
    """
    if model_path is None:
        model_path = str(RERANKER_MODEL_PATH)
    
    return BGEReranker(model_path)