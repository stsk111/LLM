"""
DocQA Pro - 缓存管理模块
实现索引和文档的持久化缓存
"""

import hashlib
import pickle
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

from config import CACHE_DIR


class CacheManager:
    """索引和文档缓存管理器"""
    
    def __init__(self, cache_dir: Path = CACHE_DIR):
        """
        初始化缓存管理器
        
        Args:
            cache_dir: 缓存目录路径
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"📦 缓存管理器初始化，缓存目录: {self.cache_dir}")
    
    def calculate_file_hash(self, file_path: str) -> str:
        """
        计算文件的MD5哈希值
        
        Args:
            file_path: 文件路径
            
        Returns:
            文件的MD5哈希值
        """
        md5_hash = hashlib.md5()
        
        try:
            with open(file_path, "rb") as f:
                # 分块读取文件，避免大文件占用过多内存
                for chunk in iter(lambda: f.read(4096), b""):
                    md5_hash.update(chunk)
            
            return md5_hash.hexdigest()
        
        except Exception as e:
            print(f"❌ 计算文件哈希失败: {e}")
            raise
    
    def get_cache_path(self, file_hash: str) -> Path:
        """
        获取缓存文件路径
        
        Args:
            file_hash: 文件哈希值
            
        Returns:
            缓存目录路径
        """
        return self.cache_dir / file_hash
    
    def cache_exists(self, file_path: str) -> bool:
        """
        检查缓存是否存在
        
        Args:
            file_path: 原始文件路径
            
        Returns:
            缓存是否存在
        """
        try:
            file_hash = self.calculate_file_hash(file_path)
            cache_path = self.get_cache_path(file_hash)
            
            # 检查所有必需的缓存文件是否存在
            required_files = [
                cache_path / "index.faiss",
                cache_path / "index.pkl",
                cache_path / "chunks.pkl",
                cache_path / "metadata.json"
            ]
            
            return all(f.exists() for f in required_files)
        
        except Exception as e:
            print(f"⚠️  检查缓存失败: {e}")
            return False
    
    def save_cache(
        self,
        file_path: str,
        faiss_index: FAISS,
        chunks: List[Document],
        metadata: Dict[str, Any]
    ) -> bool:
        """
        保存索引和文档到缓存
        
        Args:
            file_path: 原始文件路径
            faiss_index: FAISS索引
            chunks: 文档片段列表
            metadata: 元数据（页数、块数等统计信息）
            
        Returns:
            是否保存成功
        """
        try:
            file_hash = self.calculate_file_hash(file_path)
            cache_path = self.get_cache_path(file_hash)
            cache_path.mkdir(parents=True, exist_ok=True)
            
            print(f"💾 保存缓存到: {cache_path}")
            
            # 1. 保存FAISS索引
            faiss_index.save_local(str(cache_path))
            print(f"  ✓ FAISS索引已保存")
            
            # 2. 保存chunks（用于重建BM25索引）
            chunks_file = cache_path / "chunks.pkl"
            with open(chunks_file, 'wb') as f:
                pickle.dump(chunks, f)
            print(f"  ✓ 文档片段已保存 ({len(chunks)} 个)")
            
            # 3. 保存元数据
            metadata_file = cache_path / "metadata.json"
            metadata_to_save = {
                **metadata,
                'file_path': str(file_path),
                'file_hash': file_hash,
                'cache_version': '1.0'
            }
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata_to_save, f, ensure_ascii=False, indent=2)
            print(f"  ✓ 元数据已保存")
            
            print(f"✅ 缓存保存成功！")
            return True
        
        except Exception as e:
            print(f"❌ 保存缓存失败: {e}")
            return False
    
    def load_cache(
        self,
        file_path: str,
        embeddings
    ) -> Optional[Tuple[FAISS, List[Document], Dict[str, Any]]]:
        """
        从缓存加载索引和文档
        
        Args:
            file_path: 原始文件路径
            embeddings: Embedding模型（用于加载FAISS索引）
            
        Returns:
            (FAISS索引, 文档片段列表, 元数据) 或 None
        """
        try:
            if not self.cache_exists(file_path):
                return None
            
            file_hash = self.calculate_file_hash(file_path)
            cache_path = self.get_cache_path(file_hash)
            
            print(f"📂 从缓存加载: {cache_path}")
            
            # 1. 加载FAISS索引
            faiss_index = FAISS.load_local(
                str(cache_path),
                embeddings,
                allow_dangerous_deserialization=True  # 允许反序列化本地文件
            )
            print(f"  ✓ FAISS索引已加载")
            
            # 2. 加载chunks
            chunks_file = cache_path / "chunks.pkl"
            with open(chunks_file, 'rb') as f:
                chunks = pickle.load(f)
            print(f"  ✓ 文档片段已加载 ({len(chunks)} 个)")
            
            # 3. 加载元数据
            metadata_file = cache_path / "metadata.json"
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            print(f"  ✓ 元数据已加载")
            
            print(f"✅ 缓存加载成功！")
            return faiss_index, chunks, metadata
        
        except Exception as e:
            print(f"❌ 加载缓存失败: {e}")
            return None
    
    def clear_cache(self, file_path: Optional[str] = None) -> bool:
        """
        清除缓存
        
        Args:
            file_path: 要清除的文件路径，如果为None则清除所有缓存
            
        Returns:
            是否清除成功
        """
        try:
            if file_path:
                # 清除指定文件的缓存
                file_hash = self.calculate_file_hash(file_path)
                cache_path = self.get_cache_path(file_hash)
                
                if cache_path.exists():
                    import shutil
                    shutil.rmtree(cache_path)
                    print(f"🗑️  已清除缓存: {cache_path}")
            else:
                # 清除所有缓存
                if self.cache_dir.exists():
                    import shutil
                    for item in self.cache_dir.iterdir():
                        if item.is_dir():
                            shutil.rmtree(item)
                    print(f"🗑️  已清除所有缓存")
            
            return True
        
        except Exception as e:
            print(f"❌ 清除缓存失败: {e}")
            return False
    
    def list_caches(self) -> List[Dict[str, Any]]:
        """
        列出所有缓存
        
        Returns:
            缓存信息列表
        """
        caches = []
        
        try:
            if not self.cache_dir.exists():
                return caches
            
            for cache_dir in self.cache_dir.iterdir():
                if cache_dir.is_dir():
                    metadata_file = cache_dir / "metadata.json"
                    if metadata_file.exists():
                        with open(metadata_file, 'r', encoding='utf-8') as f:
                            metadata = json.load(f)
                            caches.append({
                                'hash': cache_dir.name,
                                'file_path': metadata.get('file_path', 'Unknown'),
                                'total_pages': metadata.get('total_pages', 0),
                                'total_chunks': metadata.get('total_chunks', 0),
                                'cache_dir': str(cache_dir)
                            })
        
        except Exception as e:
            print(f"⚠️  列出缓存失败: {e}")
        
        return caches
    
    def get_cache_size(self) -> int:
        """
        获取缓存总大小（字节）
        
        Returns:
            缓存总大小
        """
        total_size = 0
        
        try:
            if not self.cache_dir.exists():
                return 0
            
            for item in self.cache_dir.rglob('*'):
                if item.is_file():
                    total_size += item.stat().st_size
        
        except Exception as e:
            print(f"⚠️  计算缓存大小失败: {e}")
        
        return total_size
    
    def format_cache_size(self, size_bytes: int) -> str:
        """
        格式化缓存大小
        
        Args:
            size_bytes: 字节数
            
        Returns:
            格式化的大小字符串
        """
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.2f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.2f} TB"


def create_cache_manager(cache_dir: Path = CACHE_DIR) -> CacheManager:
    """
    创建缓存管理器的便捷函数
    
    Args:
        cache_dir: 缓存目录路径
        
    Returns:
        缓存管理器实例
    """
    return CacheManager(cache_dir)
