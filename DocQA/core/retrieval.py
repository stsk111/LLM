"""
DocQA Pro - 检索模块
Embedding、向量索引、混合检索功能
"""

import os
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from config import (
    EMBEDDING_MODEL_PATH, EMBEDDING_DEVICE, EMBEDDING_BATCH_SIZE,
    FAISS_INDEX_TYPE, FAISS_USE_GPU, DENSE_WEIGHT, SPARSE_WEIGHT,
    RETRIEVAL_TOP_K, CACHE_DIR
)


class LangChainEmbeddingsWrapper(Embeddings):
    """LangChain Embeddings接口包装器"""
    
    def __init__(self, model):
        """
        初始化包装器
        
        Args:
            model: SentenceTransformer模型实例
        """
        self.model = model
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """对文本列表进行向量化"""
        embeddings = self.model.encode(
            texts,
            batch_size=EMBEDDING_BATCH_SIZE,
            normalize_embeddings=True,
            show_progress_bar=False
        )
        return embeddings.tolist()
    
    def embed_query(self, text: str) -> List[float]:
        """对单个查询进行向量化"""
        embedding = self.model.encode(
            text,
            normalize_embeddings=True,
            show_progress_bar=False
        )
        return embedding.tolist()


class EmbeddingEngine:
    """Embedding模型引擎"""
    
    def __init__(self, model_path: str = str(EMBEDDING_MODEL_PATH)):
        """
        初始化Embedding模型
        
        Args:
            model_path: 模型路径
        """
        self.model_path = model_path
        self.embeddings = None
        self._load_model()
    
    def _load_model(self):
        """加载本地Embedding模型"""
        try:
            print(f"🔄 加载Embedding模型: {self.model_path}")
            
            # 在加载前再次确保环境变量生效
            import os
            os.environ['TORCH_LOAD_WEIGHTS_ONLY'] = '0'
            os.environ['TORCH_ALLOW_VULNERABLE_LOAD'] = '1'
            
            # 使用sentence-transformers直接加载
            from sentence_transformers import SentenceTransformer
            
            # 加载模型，显式传递权重加载参数（如果库支持）
            self.model = SentenceTransformer(
                self.model_path,
                device=EMBEDDING_DEVICE,
                trust_remote_code=True,
                model_kwargs={"weights_only": False}
            )
            
            # 创建LangChain兼容的embeddings包装器
            self.embeddings = LangChainEmbeddingsWrapper(self.model)
            
            print("✅ Embedding模型加载成功")
            
        except Exception as e:
            print(f"❌ Embedding模型加载失败: {e}")
            raise
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """对文本列表进行向量化（直接调用接口）"""
        return self.embeddings.embed_documents(texts)
    
    def embed_query(self, text: str) -> List[float]:
        """对单个查询进行向量化（直接调用接口）"""
        return self.embeddings.embed_query(text)
    


class FAISSIndexBuilder:
    """FAISS向量索引构建器"""
    
    def __init__(self, embedding_engine: EmbeddingEngine):
        """
        初始化索引构建器
        
        Args:
            embedding_engine: Embedding引擎
        """
        self.embedding_engine = embedding_engine
        self.vector_store = None
    
    def create_index(self, chunks: List[Document]) -> FAISS:
        """
        构建FAISS向量索引
        
        Args:
            chunks: 文档片段列表
            
        Returns:
            FAISS向量存储
        """
        if not chunks:
            raise ValueError("文档片段列表不能为空")
        
        try:
            print(f"🔄 构建FAISS索引，共 {len(chunks)} 个文档片段...")
            
            # 使用FAISS构建向量存储
            self.vector_store = FAISS.from_documents(
                chunks, 
                self.embedding_engine.embeddings
            )
            
            print("✅ FAISS索引构建成功")
            return self.vector_store
            
        except Exception as e:
            print(f"❌ FAISS索引构建失败: {e}")
            raise
    
    def save_index(self, index: FAISS, save_path: str):
        """
        保存索引到本地
        
        Args:
            index: FAISS索引
            save_path: 保存路径
        """
        try:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            index.save_local(save_path)
            print(f"✅ 索引已保存到: {save_path}")
        except Exception as e:
            print(f"❌ 索引保存失败: {e}")
            raise
    
    def load_index(self, load_path: str) -> FAISS:
        """
        从本地加载索引
        
        Args:
            load_path: 加载路径
            
        Returns:
            FAISS索引
        """
        try:
            if not Path(load_path).exists():
                raise FileNotFoundError(f"索引文件不存在: {load_path}")
            
            self.vector_store = FAISS.load_local(
                load_path, 
                self.embedding_engine.embeddings
            )
            print(f"✅ 索引已加载: {load_path}")
            return self.vector_store
            
        except Exception as e:
            print(f"❌ 索引加载失败: {e}")
            raise


class HybridRetriever:
    """混合检索器 (Dense + Sparse)"""
    
    def __init__(self, faiss_index: FAISS, documents: List[Document]):
        """
        初始化混合检索器
        
        Args:
            faiss_index: FAISS向量索引
            documents: 原始文档列表（用于BM25）
        """
        self.faiss_index = faiss_index
        self.documents = documents
        
        # 初始化稠密检索器（FAISS）
        self.dense_retriever = faiss_index.as_retriever(
            search_kwargs={"k": RETRIEVAL_TOP_K}
        )
        self.faiss_index = faiss_index  # 保存原始索引用于直接查询
        
        # 初始化稀疏检索器（BM25）
        self.sparse_retriever = BM25Retriever.from_documents(documents)
        self.sparse_retriever.k = RETRIEVAL_TOP_K
        
        print("✅ 混合检索器初始化完成")
    
    def retrieve(self, query: str, top_k: int = RETRIEVAL_TOP_K) -> List[Document]:
        """
        执行混合检索
        
        Args:
            query: 查询文本
            top_k: 返回结果数量
            
        Returns:
            检索结果文档列表
        """
        try:
            # 稠密检索（向量相似度）- 使用invoke代替get_relevant_documents
            dense_results = self.dense_retriever.invoke(query)
            
            # 稀疏检索（BM25关键词匹配）
            sparse_results = self.sparse_retriever.invoke(query)
            
            # 使用RRF（Reciprocal Rank Fusion）融合结果
            fused_results = self._fuse_results(
                dense_results, sparse_results, 
                DENSE_WEIGHT, SPARSE_WEIGHT
            )
            
            # 返回Top-K结果
            return fused_results[:top_k]
            
        except Exception as e:
            print(f"❌ 混合检索失败: {e}")
            raise
    
    def _fuse_results(
        self, 
        dense_results: List[Document], 
        sparse_results: List[Document],
        dense_weight: float = 0.5,
        sparse_weight: float = 0.5
    ) -> List[Document]:
        """
        融合稠密和稀疏检索结果
        
        Args:
            dense_results: 稠密检索结果
            sparse_results: 稀疏检索结果
            dense_weight: 稠密检索权重
            sparse_weight: 稀疏检索权重
            
        Returns:
            融合后的结果
        """
        # 使用RRF算法融合结果
        doc_scores = {}
        
        # 计算稠密检索分数
        for i, doc in enumerate(dense_results):
            doc_id = self._get_doc_id(doc)
            rrf_score = dense_weight / (60 + i + 1)  # RRF公式，k=60
            doc_scores[doc_id] = doc_scores.get(doc_id, 0) + rrf_score
        
        # 计算稀疏检索分数
        for i, doc in enumerate(sparse_results):
            doc_id = self._get_doc_id(doc)
            rrf_score = sparse_weight / (60 + i + 1)
            doc_scores[doc_id] = doc_scores.get(doc_id, 0) + rrf_score
        
        # 按分数排序
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        
        # 根据doc_id找回原文档
        id_to_doc = {}
        for doc in dense_results + sparse_results:
            doc_id = self._get_doc_id(doc)
            if doc_id not in id_to_doc:
                id_to_doc[doc_id] = doc
        
        # 返回排序后的文档
        fused_results = []
        for doc_id, score in sorted_docs:
            if doc_id in id_to_doc:
                doc = id_to_doc[doc_id]
                # 添加融合分数到metadata
                doc.metadata['fusion_score'] = score
                fused_results.append(doc)
        
        return fused_results
    
    def _get_doc_id(self, doc: Document) -> str:
        """获取文档唯一标识"""
        # 使用chunk_id或者页码+内容hash作为唯一标识
        if 'chunk_id' in doc.metadata:
            return doc.metadata['chunk_id']
        else:
            # 备用方案：使用页码和内容hash
            page = doc.metadata.get('page', 0)
            content_hash = hash(doc.page_content[:100])  # 使用前100字符的hash
            return f"page_{page}_hash_{content_hash}"


def create_retrieval_system(chunks: List[Document]) -> Tuple[EmbeddingEngine, FAISSIndexBuilder, HybridRetriever]:
    """
    创建完整的检索系统
    
    Args:
        chunks: 文档片段列表
        
    Returns:
        (Embedding引擎, 索引构建器, 混合检索器)
    """
    # 创建Embedding引擎
    embedding_engine = EmbeddingEngine()
    
    # 创建FAISS索引
    index_builder = FAISSIndexBuilder(embedding_engine)
    faiss_index = index_builder.create_index(chunks)
    
    # 创建混合检索器
    hybrid_retriever = HybridRetriever(faiss_index, chunks)
    
    return embedding_engine, index_builder, hybrid_retriever