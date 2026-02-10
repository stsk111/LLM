"""
DocQA Pro - 文档摄取处理模块
PDF解析与文本切分功能 (支持缓存)
"""

import os
import json
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from config import CHUNK_SIZE, CHUNK_OVERLAP, MAX_FILE_SIZE_MB

# 缓存目录配置
CACHE_DIR = ".cache/ingestion_chunks"

class PDFIngestionPipeline:
    """PDF文档处理管道"""
    
    def __init__(
        self,
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
        progress_callback: Optional[Callable[[str, int, int], None]] = None,
        use_cache: bool = True  # 新增缓存开关
    ):
        """
        初始化PDF处理管道
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.progress_callback = progress_callback
        self.use_cache = use_cache
        
        # 确保缓存目录存在
        if self.use_cache:
            os.makedirs(CACHE_DIR, exist_ok=True)
        
        # 初始化文本切分器
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
    
    def _update_progress(self, message: str, current: int = 0, total: int = 100):
        """更新进度"""
        if self.progress_callback:
            self.progress_callback(message, current, total)

    def _calculate_file_hash(self, file_path: str) -> str:
        """计算文件的MD5哈希值，用于缓存键"""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        # 将分块参数也加入哈希，如果参数变了，缓存也应该失效
        params = f"{self.chunk_size}_{self.chunk_overlap}"
        hash_md5.update(params.encode('utf-8'))
        return hash_md5.hexdigest()

    def _save_to_cache(self, file_hash: str, result: Dict[str, Any]):
        """将处理结果保存到磁盘 JSON"""
        cache_path = os.path.join(CACHE_DIR, f"{file_hash}.json")
        
        # Document 对象不能直接 JSON 序列化，需要转 dict
        serializable_result = result.copy()
        if "chunks" in serializable_result:
            serializable_result["chunks"] = [
                {
                    "page_content": doc.page_content,
                    "metadata": doc.metadata,
                    "type": "Document"
                } 
                for doc in serializable_result["chunks"]
            ]
            
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_result, f, ensure_ascii=False, indent=2)

    def _load_from_cache(self, file_hash: str) -> Optional[Dict[str, Any]]:
        """从磁盘加载缓存"""
        cache_path = os.path.join(CACHE_DIR, f"{file_hash}.json")
        if not os.path.exists(cache_path):
            return None
            
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 将 dict 重新转回 LangChain Document 对象
            if "chunks" in data:
                data["chunks"] = [
                    Document(page_content=item["page_content"], metadata=item["metadata"])
                    for item in data["chunks"]
                ]
            return data
        except Exception as e:
            print(f"⚠️ 缓存读取失败 (将重新处理): {e}")
            return None

    def validate_pdf(self, file_path: str) -> Dict[str, Any]:
        """验证PDF文件 (逻辑保持不变)"""
        path = Path(file_path)
        if not path.exists():
            return {"valid": False, "error": f"文件不存在: {file_path}"}
        if path.suffix.lower() != '.pdf':
            return {"valid": False, "error": "只支持PDF文件格式"}
        try:
            file_size_mb = path.stat().st_size / (1024 * 1024)
            if file_size_mb > MAX_FILE_SIZE_MB:
                return {"valid": False, "error": f"文件过大: {file_size_mb:.1f}MB"}
        except OSError:
            return {"valid": False, "error": "无法读取文件信息"}
        return {"valid": True, "size_mb": file_size_mb, "name": path.name}
    
    def load_pdf(self, file_path: str) -> List[Document]:
        """加载PDF并提取文本 (逻辑保持不变)"""
        # ... (此处省略未改动的代码，与你原版一致，直接复用即可) ...
        # 为了代码简洁，请将原来的 load_pdf 完整代码保留在这里
        validation = self.validate_pdf(file_path)
        if not validation["valid"]:
            raise ValueError(validation["error"])
        
        self._update_progress("开始加载PDF文件...", 0, 100)
        try:
            loader = PyPDFLoader(file_path)
            pages = loader.load()
            self._update_progress("PDF加载完成，开始处理页面...", 30, 100)
            processed_docs = []
            for i, doc in enumerate(pages):
                doc.metadata.update({
                    "page": i + 1, "source": file_path, "total_pages": len(pages)
                })
                processed_docs.append(doc)
            self._update_progress("文档加载完成", 100, 100)
            return processed_docs
        except Exception as e:
            self._update_progress(f"PDF加载失败: {str(e)}", 0, 100)
            raise RuntimeError(str(e))

    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """切分文档 (逻辑保持不变)"""
        # ... (此处省略未改动的代码，与你原版一致，直接复用即可) ...
        # 请将原来的 chunk_documents 完整代码保留在这里
        if not documents: return []
        self._update_progress("开始文本切分...", 0, 100)
        try:
            chunked_docs = []
            total_docs = len(documents)
            for i, doc in enumerate(documents):
                chunks = self.text_splitter.split_documents([doc])
                for j, chunk in enumerate(chunks):
                    chunk.metadata.update({
                        "chunk_id": f"page_{doc.metadata['page']}_chunk_{j}",
                        "chunk_index": j,
                        "total_chunks_in_page": len(chunks)
                    })
                chunked_docs.extend(chunks)
                self._update_progress(f"正在切分第 {i+1}/{total_docs} 页...", int((i+1)*100/total_docs), 100)
            self._update_progress(f"文本切分完成", 100, 100)
            return chunked_docs
        except Exception as e:
            self._update_progress(f"文本切分失败: {e}", 0, 100)
            raise RuntimeError(str(e))

    def process_pdf(self, file_path: str) -> Dict[str, Any]:
        """
        完整处理PDF文档 (已集成缓存逻辑)
        """
        try:
            # 1. 计算哈希，尝试读取缓存
            if self.use_cache:
                file_hash = self._calculate_file_hash(file_path)
                cached_result = self._load_from_cache(file_hash)
                
                if cached_result:
                    self._update_progress("🚀 命中缓存，直接加载处理结果...", 100, 100)
                    return cached_result

            # 2. 缓存未命中，执行常规流程
            documents = self.load_pdf(file_path)
            chunks = self.chunk_documents(documents)
            
            stats = {
                "total_pages": len(documents),
                "total_chunks": len(chunks),
                "avg_chunks_per_page": len(chunks) / len(documents) if documents else 0,
                "chunk_size": self.chunk_size,
                "chunk_overlap": self.chunk_overlap
            }
            
            result = {
                "success": True,
                "chunks": chunks,
                "stats": stats,
                "message": f"PDF处理完成：{stats['total_pages']} 页 -> {stats['total_chunks']} 个文本块"
            }

            # 3. 保存到缓存
            if self.use_cache:
                self._save_to_cache(file_hash, result)

            return result
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "chunks": [],
                "stats": {}
            }

def create_pdf_pipeline(progress_callback: Optional[Callable] = None, use_cache: bool = True) -> PDFIngestionPipeline:
    return PDFIngestionPipeline(progress_callback=progress_callback, use_cache=use_cache)