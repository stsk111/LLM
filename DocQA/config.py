"""
DocQA Pro - 配置文件
统一管理模型路径、参数配置等
"""
from pathlib import Path


# 项目根目录
PROJECT_ROOT = Path(__file__).parent.absolute()

# ==================== 模型路径配置 ====================
MODELS_DIR = PROJECT_ROOT / "models"

# LLM 模型配置
LLM_MODEL_PATH = MODELS_DIR / "Qwen" / "Qwen2___5-7B-Instruct"
LLM_TENSOR_PARALLEL_SIZE = 1  # 多卡并行数量，单卡设为1

# Embedding 模型配置
EMBEDDING_MODEL_PATH = MODELS_DIR / "BAAI" / "bge-m3"
EMBEDDING_DEVICE = "cpu"  # 'cuda' 或 'cpu' - 使用CPU避免与vLLM争抢显存
EMBEDDING_BATCH_SIZE = 16  # CPU模式下减小batch size

# Reranker 模型配置
RERANKER_MODEL_PATH = MODELS_DIR / "BAAI" / "bge-reranker-large"
RERANKER_DEVICE = "cpu"  # 使用CPU避免显存不足
RERANKER_BATCH_SIZE = 8   # CPU模式下减小batch size

# ==================== 文档处理配置 ====================
# 文本切分参数
CHUNK_SIZE = 600  # 单个文本块的Token数量
CHUNK_OVERLAP = 100  # 相邻文本块的重叠Token数
TEXT_SPLITTER_TYPE = "recursive"  # 'recursive' 或 'semantic'

# PDF处理
SUPPORTED_FILE_TYPES = [".pdf"]
MAX_FILE_SIZE_MB = 100  # 最大文件大小限制

# ==================== 检索配置 ====================
# 向量检索参数
FAISS_INDEX_TYPE = "Flat"  # 'Flat', 'IVF', 'HNSW'
FAISS_USE_GPU = True  # 是否使用GPU加速FAISS

# 混合检索参数
DENSE_WEIGHT = 0.5  # 向量检索权重
SPARSE_WEIGHT = 0.5  # BM25检索权重
RETRIEVAL_TOP_K = 10  # 初始召回数量

# 重排参数
EVAL_ENABLE_RERANK = False
RERANK_TOP_N = 5  # 重排后保留的数量
RERANK_SCORE_THRESHOLD = 0.0  # 相关性分数阈值，低于此值的结果将被过滤

# ==================== 生成配置 ====================
# vLLM 推理参数
SAMPLING_PARAMS = {
    "temperature": 0.7,
    "top_p": 0.8,
    "top_k": 50,
    "max_tokens": 2048,
    "repetition_penalty": 1.1,
}

# System Prompt 模板
SYSTEM_PROMPT_TEMPLATE = """你是一个专业的文档智能助手。请严格根据下方的【参考文档】回答用户的问题。

要求：
1. 回答需准确、简洁，并基于文档内容。
2. 即使有历史对话，回答当前问题时也必须优先依据【参考文档】。
3. 如果【参考文档】中没有包含回答问题所需的信息，请直接回答"根据提供的文档，我无法回答这个问题"，不要编造信息。
4. **重要**：当你引用参考文档中的内容时，必须在引用内容的句子后面标注来源编号，格式为 [1] 或 [2][3]。
   - 例如：混合专家模型是一种机器学习模型架构[1]，其中MoE层由门控网络和多个专家网络组成[2][3]。
   - 如果一句话引用了多个文档，可以连续标注，如 [1][2]。
5. 只需要在正文中标注编号，不要在回答末尾再次列出来源列表（系统会自动添加）。

【参考文档】：
{context}

【历史对话】：
{chat_history}

【当前问题】：
{question}"""

# Query重写Prompt
QUERY_REWRITE_PROMPT = """基于以下对话历史，将最新的问题改写为一个独立的、完整的搜索查询。
如果问题中有代词（如"它"、"这个"、"那个"），请用具体的名词替换。

对话历史：
{chat_history}

最新问题：{question}

改写后的查询："""

# ==================== UI 配置 ====================
GRADIO_THEME = "soft"  # Gradio主题
GRADIO_SERVER_PORT = 7860
GRADIO_SHARE = False  # 是否生成公开链接

# 聊天窗口配置
CHATBOT_HEIGHT = 600
ENABLE_STREAMING = True  # 是否启用流式输出

# ==================== 性能配置 ====================
# 缓存配置
ENABLE_CACHE = True
CACHE_DIR = PROJECT_ROOT / ".cache"

# 日志配置
LOG_LEVEL = "INFO"  # 'DEBUG', 'INFO', 'WARNING', 'ERROR'
LOG_DIR = PROJECT_ROOT / "logs"

# ==================== 评估配置 ====================
EVAL_SIMILARITY_THRESHOLD = 0.8
EVAL_MRR_MAX_DEPTH = 20

