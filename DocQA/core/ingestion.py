"""
DocQA Pro - 文档摄取处理模块
PDF解析与文本切分功能
"""

import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from config import CHUNK_SIZE, CHUNK_OVERLAP, MAX_FILE_SIZE_MB


class PDFIngestionPipeline:
    """PDF文档处理管道"""
    
    def __init__(
        self,
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
        progress_callback: Optional[Callable[[str, int, int], None]] = None
    ):
        """
        初始化PDF处理管道
        
        Args:
            chunk_size: 文本块大小
            chunk_overlap: 文本块重叠大小  
            progress_callback: 进度回调函数 (message, current, total)
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.progress_callback = progress_callback
        
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
    
    def validate_pdf(self, file_path: str) -> Dict[str, Any]:
        """
        验证PDF文件
        
        Args:
            file_path: PDF文件路径
            
        Returns:
            验证结果字典
        """
        path = Path(file_path)
        
        # 检查文件是否存在
        if not path.exists():
            return {
                "valid": False,
                "error": f"文件不存在: {file_path}"
            }
        
        # 检查文件扩展名
        if path.suffix.lower() != '.pdf':
            return {
                "valid": False, 
                "error": "只支持PDF文件格式"
            }
        
        # 检查文件大小
        try:
            file_size_mb = path.stat().st_size / (1024 * 1024)
            if file_size_mb > MAX_FILE_SIZE_MB:
                return {
                    "valid": False,
                    "error": f"文件过大: {file_size_mb:.1f}MB (最大支持 {MAX_FILE_SIZE_MB}MB)"
                }
        except OSError:
            return {
                "valid": False,
                "error": "无法读取文件信息，请检查文件权限"
            }
        
        return {
            "valid": True,
            "size_mb": file_size_mb,
            "name": path.name
        }
    
    def load_pdf(self, file_path: str) -> List[Document]:
        """
        加载PDF并提取文本
        
        Args:
            file_path: PDF文件路径
            
        Returns:
            文档列表，每个文档包含页面内容和元数据
        """
        # 验证文件
        validation = self.validate_pdf(file_path)
        if not validation["valid"]:
            raise ValueError(validation["error"])
        
        self._update_progress("开始加载PDF文件...", 0, 100)
        
        try:
            # 使用PyPDFLoader加载PDF
            loader = PyPDFLoader(file_path)
            pages = loader.load()
            
            self._update_progress("PDF加载完成，开始处理页面...", 30, 100)
            
            # 处理每一页，确保元数据包含页码信息
            processed_docs = []
            for i, doc in enumerate(pages):
                # 确保元数据包含必要信息
                doc.metadata.update({
                    "page": i + 1,
                    "source": file_path,
                    "total_pages": len(pages)
                })
                processed_docs.append(doc)
            
            self._update_progress("文档加载完成", 100, 100)
            return processed_docs
            
        except FileNotFoundError:
            error_msg = "PDF文件未找到"
            self._update_progress(error_msg, 0, 100)
            raise FileNotFoundError(error_msg)
        except PermissionError:
            error_msg = "PDF文件权限不足，无法读取"
            self._update_progress(error_msg, 0, 100)
            raise PermissionError(error_msg)
        except Exception as e:
            if "encrypted" in str(e).lower() or "password" in str(e).lower():
                error_msg = "PDF文件已加密，请提供无密码保护的文件"
            elif "corrupt" in str(e).lower() or "damaged" in str(e).lower():
                error_msg = "PDF文件已损坏，请检查文件完整性"
            else:
                error_msg = f"PDF加载失败: {str(e)}"
            
            self._update_progress(error_msg, 0, 100)
            raise RuntimeError(error_msg)
    
    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """
        切分文档为小片段
        
        Args:
            documents: 文档列表
            
        Returns:
            切分后的文档片段列表
        """
        if not documents:
            return []
        
        self._update_progress("开始文本切分...", 0, 100)
        
        try:
            chunked_docs = []
            total_docs = len(documents)
            
            for i, doc in enumerate(documents):
                # 切分当前文档
                chunks = self.text_splitter.split_documents([doc])
                
                # 为每个chunk添加额外的元数据
                for j, chunk in enumerate(chunks):
                    chunk.metadata.update({
                        "chunk_id": f"page_{doc.metadata['page']}_chunk_{j}",
                        "chunk_index": j,
                        "total_chunks_in_page": len(chunks)
                    })
                
                chunked_docs.extend(chunks)
                
                # 更新进度
                progress = int((i + 1) * 100 / total_docs)
                self._update_progress(
                    f"正在切分第 {i+1}/{total_docs} 页...", 
                    progress, 100
                )
            
            self._update_progress(
                f"文本切分完成，共生成 {len(chunked_docs)} 个文本块", 
                100, 100
            )
            
            return chunked_docs
            
        except Exception as e:
            error_msg = f"文本切分失败: {str(e)}"
            self._update_progress(error_msg, 0, 100)
            raise RuntimeError(error_msg)
    
    def process_pdf(self, file_path: str) -> Dict[str, Any]:
        """
        完整处理PDF文档
        
        Args:
            file_path: PDF文件路径
            
        Returns:
            处理结果，包含文档片段和统计信息
        """
        try:
            # 加载PDF
            documents = self.load_pdf(file_path)
            
            # 切分文档
            chunks = self.chunk_documents(documents)
            
            # 统计信息
            stats = {
                "total_pages": len(documents),
                "total_chunks": len(chunks),
                "avg_chunks_per_page": len(chunks) / len(documents) if documents else 0,
                "chunk_size": self.chunk_size,
                "chunk_overlap": self.chunk_overlap
            }
            
            return {
                "success": True,
                "chunks": chunks,
                "stats": stats,
                "message": f"PDF处理完成：{stats['total_pages']} 页 -> {stats['total_chunks']} 个文本块"
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "chunks": [],
                "stats": {}
            }


def create_pdf_pipeline(progress_callback: Optional[Callable] = None) -> PDFIngestionPipeline:
    """
    创建PDF处理管道的便捷函数
    
    Args:
        progress_callback: 进度回调函数
        
    Returns:
        PDF处理管道实例
    """
    return PDFIngestionPipeline(progress_callback=progress_callback)