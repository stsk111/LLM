"""
DocQA 评估模块
实现检索质量和回答质量的评估指标
"""

import json
import logging
import time
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path

# 第三方库依赖
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import jieba
from sentence_transformers import SentenceTransformer

# 本地模块
import config
from core.qa_chain import DocQAChain
from core.retrieval import HybridRetriever
from core.reranker import BGEReranker
from llm_engine.chat_llm import ChatLLM
from core.ingestion import create_pdf_pipeline
from core.cache_manager import create_cache_manager
from core.retrieval import EmbeddingEngine, FAISSIndexBuilder

# 配置日志（尽量简洁，避免评估时刷屏）
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)


logger = logging.getLogger(__name__)


class DocQAEvaluator:
    """DocQA系统评估器"""
    
    def __init__(self, docqa_chain: DocQAChain):
        """
        初始化评估器
        
        Args:
            docqa_chain: DocQA问答链实例
        """
        self.docqa_chain = docqa_chain
        self._rouge_scorer = rouge_scorer.RougeScorer(
            ['rouge1', 'rouge2', 'rougeL'], 
            use_stemmer=False
        )
        self._smoothing_function = SmoothingFunction().method4
        
        # 初始化语义模型（用于计算相似度）
        from config import EMBEDDING_MODEL_PATH

        self._sentence_model = None
        model_path = str(EMBEDDING_MODEL_PATH)

        try:
            from config import EMBEDDING_DEVICE
            device = str(EMBEDDING_DEVICE) if EMBEDDING_DEVICE else "cpu"

            logger.info(f"开始加载语义模型: {model_path} (device={device})")
            self._sentence_model = SentenceTransformer(
                model_path,
                device=device,
                trust_remote_code=True,
                model_kwargs={"weights_only": False}
            )
            logger.info(f"语义模型加载完成: {model_path} (device={device})")
        except OSError as e:
            logger.error(f"语义模型加载失败（疑似本地模型路径不存在或不完整）: {e}")
        except ValueError as e:
            logger.error(f"语义模型加载失败（配置或参数不正确）: {e}")
    
    def _tokenize_chinese(self, text: str) -> List[str]:
        """
        中文分词
        
        Args:
            text: 输入文本
            
        Returns:
            分词结果
        """
        text = (text or "").strip()
        if not text:
            return []
        return [t for t in jieba.cut(text) if t and t.strip()]

    def _tokenize_for_rouge(self, text: str) -> str:
        tokens = self._tokenize_chinese(text)
        return " ".join(tokens)
    
    def _calculate_recall_at_k(
        self, 
        retrieved_contexts: List[str], 
        ground_truth_contexts: List[str], 
        k: int,
        similarity_threshold: float = 0.8
    ) -> float:
        """
        计算Recall@K
        
        Args:
            retrieved_contexts: 检索到的上下文列表
            ground_truth_context: 真实上下文
            k: 前K个结果
            
        Returns:
            Recall@K分数
        """
        if not retrieved_contexts or not ground_truth_contexts:
            return 0.0

        # 取前k个检索结果
        top_k_contexts = retrieved_contexts[:k]

        # 命中任意 gt 即算命中；仅使用 gt in ctx + 语义相似度阈值
        for ctx in top_k_contexts:
            if not isinstance(ctx, str):
                continue
            ctx = ctx.strip()
            if not ctx:
                continue

            for gt in ground_truth_contexts:
                if not isinstance(gt, str):
                    continue
                gt = gt.strip()
                if not gt:
                    continue

                if gt in ctx:
                    return 1.0

                similarity = self._calculate_text_similarity(gt, ctx)
                if similarity > similarity_threshold:
                    return 1.0

        return 0.0
    
    def _calculate_mrr(
        self, 
        retrieved_contexts: List[str], 
        ground_truth_contexts: List[str],
        similarity_threshold: float = 0.8,
        max_depth: int = 20
    ) -> float:
        """
        计算MRR (Mean Reciprocal Rank)
        
        Args:
            retrieved_contexts: 检索到的上下文列表
            ground_truth_context: 真实上下文
            
        Returns:
            MRR分数
        """
        if not retrieved_contexts or not ground_truth_contexts:
            return 0.0

        # 限制最大检索深度，避免无意义遍历导致评估过慢
        limited_contexts = retrieved_contexts[:max_depth] if max_depth and max_depth > 0 else retrieved_contexts

        for i, ctx in enumerate(limited_contexts, 1):
            if not isinstance(ctx, str):
                continue
            ctx = ctx.strip()
            if not ctx:
                continue

            for gt in ground_truth_contexts:
                if not isinstance(gt, str):
                    continue
                gt = gt.strip()
                if not gt:
                    continue

                if gt in ctx:
                    return 1.0 / i

                similarity = self._calculate_text_similarity(gt, ctx)
                if similarity > similarity_threshold:
                    return 1.0 / i

        return 0.0
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """
        计算文本相似度
        
        Args:
            text1: 文本1
            text2: 文本2
            
        Returns:
            相似度分数
        """
        if not text1 or not text2:
            return 0.0
        
        # 使用语义模型计算相似度
        if self._sentence_model:
            try:
                embeddings = self._sentence_model.encode([text1, text2])
                similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
                return float(similarity)
            except (OSError, ValueError, RuntimeError) as e:
                logger.warning(f"语义相似度计算失败: {e}")
        
        # 退回到简单的词汇重叠
        words1 = set(self._tokenize_chinese(text1))
        words2 = set(self._tokenize_chinese(text2))
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0
    
    def _calculate_rouge_scores(self, prediction: str, reference: str) -> Dict[str, float]:
        """
        计算ROUGE分数
        
        Args:
            prediction: 预测答案
            reference: 参考答案
            
        Returns:
            ROUGE分数字典
        """
        try:
            # rouge-score 默认按空格切分 token；中文若不分词会导致分数异常偏低
            reference_tok = self._tokenize_for_rouge(reference)
            prediction_tok = self._tokenize_for_rouge(prediction)

            scores = self._rouge_scorer.score(reference_tok, prediction_tok)
            return {
                'rouge1': scores['rouge1'].fmeasure,
                'rouge2': scores['rouge2'].fmeasure,
                'rougeL': scores['rougeL'].fmeasure
            }
        except (ValueError, TypeError) as e:
            logger.error(f"ROUGE计算失败: {e}")
            return {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}
    
    def _calculate_bleu_score(self, prediction: str, reference: str) -> float:
        """
        计算BLEU分数
        
        Args:
            prediction: 预测答案
            reference: 参考答案
            
        Returns:
            BLEU分数
        """
        try:
            prediction = (prediction or "").strip()
            reference = (reference or "").strip()
            if not prediction or not reference:
                return 0.0

            reference_tokens = [self._tokenize_chinese(reference)]
            prediction_tokens = self._tokenize_chinese(prediction)

            if not reference_tokens[0] or not prediction_tokens:
                return 0.0
            
            bleu_score = sentence_bleu(
                reference_tokens,
                prediction_tokens,
                smoothing_function=self._smoothing_function
            )
            return float(bleu_score)
        except (ValueError, IndexError, ZeroDivisionError) as e:
            logger.error(f"BLEU计算失败: {e}")
            return 0.0
    
    def _calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """
        计算语义相似度
        
        Args:
            text1: 文本1
            text2: 文本2
            
        Returns:
            语义相似度分数
        """
        return self._calculate_text_similarity(text1, text2)
    
    def evaluate_single_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        评估单个样本
        
        Args:
            sample: 测试样本
            
        Returns:
            评估结果
        """
        question = sample['question']
        ground_truth = sample['ground_truth']
        ground_contexts = sample['contexts']
        
        logger.info(f"评估问题: {question}")
        
        start_time = time.time()
        
        # 调用DocQA系统（不捕获异常，便于查看完整堆栈）
        result = self.docqa_chain.ask(question, stream=False)
        
        end_time = time.time()
        response_time = end_time - start_time
        
        # 提取检索结果和答案
        # sources 当前结构为 List[dict]（由 core/qa_chain.py::_format_sources 生成）
        sources = result.get('sources', [])

        retrieved_contexts = [
            (s.get('content') or s.get('content_preview') or '').strip()
            for s in sources
            if isinstance(s, dict) and (s.get('content') or s.get('content_preview'))
        ]
        predicted_answer = result.get('answer', '')
        
        # 计算检索指标
        
        similarity_threshold = float(getattr(config, 'EVAL_SIMILARITY_THRESHOLD', 0.8))
        max_mrr_depth = int(getattr(config, 'EVAL_MRR_MAX_DEPTH', 20))

        recall_1 = self._calculate_recall_at_k(retrieved_contexts, ground_contexts, 1, similarity_threshold=similarity_threshold)
        recall_3 = self._calculate_recall_at_k(retrieved_contexts, ground_contexts, 3, similarity_threshold=similarity_threshold)
        recall_5 = self._calculate_recall_at_k(retrieved_contexts, ground_contexts, 5, similarity_threshold=similarity_threshold)
        mrr = self._calculate_mrr(retrieved_contexts, ground_contexts, similarity_threshold=similarity_threshold, max_depth=max_mrr_depth)
        
        # 计算回答质量指标
        rouge_scores = self._calculate_rouge_scores(predicted_answer, ground_truth)
        bleu_score = self._calculate_bleu_score(predicted_answer, ground_truth)
        semantic_sim = self._calculate_semantic_similarity(predicted_answer, ground_truth)
        
        evaluation_result = {
            'sample_id': sample.get('id', ''),
            'question': question,
            'ground_truth': ground_truth,
            'predicted_answer': predicted_answer,
            'ground_context': ground_contexts,
            'retrieved_contexts': retrieved_contexts,
            'response_time': response_time,
            'retrieval_metrics': {
                'recall@1': recall_1,
                'recall@3': recall_3,
                'recall@5': recall_5,
                'mrr': mrr
            },
            'answer_quality_metrics': {
                'rouge1': rouge_scores['rouge1'],
                'rouge2': rouge_scores['rouge2'],
                'rougeL': rouge_scores['rougeL'],
                'bleu': bleu_score,
                'semantic_similarity': semantic_sim
            }
        }

        logger.info(json.dumps(evaluation_result, ensure_ascii=False, indent=2))
        logger.info(f"样本 {sample.get('id', '')} 评估完成")
        return evaluation_result
    
    def evaluate_dataset(
        self, 
        testset_path: str, 
        output_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        评估整个测试集
        
        Args:
            testset_path: 测试集路径
            output_path: 输出路径（可选）
            
        Returns:
            评估结果汇总
        """
        logger.info(f"开始评估测试集: {testset_path}")
        
        # 加载测试集
        try:
            with open(testset_path, 'r', encoding='utf-8') as f:
                testset = json.load(f)
            logger.info(f"加载测试集成功，共 {len(testset)} 个样本")
        except Exception as e:
            logger.error(f"加载测试集失败: {e}")
            raise
        
        # 逐个评估
        all_results = []
        successful_evaluations = 0
        
        for i, sample in enumerate(testset):
            logger.info(f"评估进度: {i+1}/{len(testset)}")
            result = self.evaluate_single_sample(sample)
            all_results.append(result)
            
            if 'error' not in result:
                successful_evaluations += 1
        
        # 计算平均指标
        avg_metrics = self._calculate_average_metrics(all_results)
        
        # 构建最终报告
        final_report = {
            'evaluation_summary': {
                'total_samples': len(testset),
                'successful_evaluations': successful_evaluations,
                'success_rate': successful_evaluations / len(testset) if testset else 0,
                'evaluation_time': time.strftime('%Y-%m-%d %H:%M:%S')
            },
            'average_metrics': avg_metrics,
            'detailed_results': all_results
        }
        
        # 保存结果
        if output_path:
            self._save_results(final_report, output_path)
        
        # 打印摘要
        self._print_summary(final_report)
        
        return final_report
    
    def _calculate_average_metrics(self, all_results: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        计算平均指标
        
        Args:
            all_results: 所有评估结果
            
        Returns:
            平均指标字典
        """
        successful_results = [r for r in all_results if 'error' not in r]
        
        if not successful_results:
            return {}
        
        metrics = {
            'avg_recall@1': 0.0,
            'avg_recall@3': 0.0,
            'avg_recall@5': 0.0,
            'avg_mrr': 0.0,
            'avg_rouge_1': 0.0,
            'avg_rouge_2': 0.0,
            'avg_rouge_l': 0.0,
            'avg_bleu': 0.0,
            'avg_semantic_similarity': 0.0,
            'avg_response_time': 0.0
        }
        
        for result in successful_results:
            retrieval_metrics = result.get('retrieval_metrics', {})
            answer_metrics = result.get('answer_quality_metrics', {})
            
            metrics['avg_recall@1'] += retrieval_metrics.get('recall@1', 0)
            metrics['avg_recall@3'] += retrieval_metrics.get('recall@3', 0)
            metrics['avg_recall@5'] += retrieval_metrics.get('recall@5', 0)
            metrics['avg_mrr'] += retrieval_metrics.get('mrr', 0)
            metrics['avg_rouge_1'] += answer_metrics.get('rouge1', 0)
            metrics['avg_rouge_2'] += answer_metrics.get('rouge2', 0)
            metrics['avg_rouge_l'] += answer_metrics.get('rougeL', 0)
            metrics['avg_bleu'] += answer_metrics.get('bleu', 0)
            metrics['avg_semantic_similarity'] += answer_metrics.get('semantic_similarity', 0)
            metrics['avg_response_time'] += result.get('response_time', 0)
        
        # 计算平均值
        num_successful = len(successful_results)
        for key in metrics:
            metrics[key] /= num_successful
        
        return metrics
    
    def _save_results(self, results: Dict[str, Any], output_path: str) -> None:
        """
        保存评估结果
        
        Args:
            results: 评估结果
            output_path: 输出路径
        """
        try:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            
            logger.info(f"评估结果已保存至: {output_path}")
        except (OSError, TypeError, ValueError) as e:
            logger.error(f"保存结果失败: {e}")
    
    def _print_summary(self, results: Dict[str, Any]) -> None:
        """
        打印评估摘要
        
        Args:
            results: 评估结果
        """
        summary = results['evaluation_summary']
        metrics = results['average_metrics']
        
        logger.info(
            json.dumps(
                {
                    "event": "evaluation_summary",
                    "evaluation_summary": summary,
                    "average_metrics": metrics,
                },
                ensure_ascii=False,
            )
        )


class NoReranker:
    def rerank(
        self,
        query: str,
        documents: List[Any],
        top_n: int = 5,
        score_threshold: float = 0.0,
    ):
        if not documents:
            return []
        return [(doc, 0.0) for doc in documents[:top_n]]


def create_docqa_chain(pdf_path: str, enable_rerank: bool = True) -> DocQAChain:
    """
    创建DocQA链实例

    Args:
        pdf_path: PDF文档路径

    Returns:
        DocQA链实例

    Raises:
        FileNotFoundError: PDF路径不存在
        ValueError: 文档处理失败
    """
    logger.info(f"初始化DocQA系统，PDF路径: {pdf_path}")

    pdf_file = Path(pdf_path)
    if not pdf_file.exists():
        raise FileNotFoundError(f"PDF文件不存在: {pdf_path}")

    try:
        # 初始化缓存管理器
        from config import ENABLE_CACHE
        cache_manager = create_cache_manager() if ENABLE_CACHE else None
        
        # 初始化核心组件模型（单例模式思想）
        llm = ChatLLM()
        embedding_engine = EmbeddingEngine()
        reranker = BGEReranker() if enable_rerank else NoReranker()

        documents = None
        faiss_index = None
        stats = None
        from_cache = False

        # 尝试从缓存加载
        if cache_manager and cache_manager.cache_exists(pdf_path):
            logger.info("🎯 检测到文档缓存，尝试快速加载...")
            cache_result = cache_manager.load_cache(pdf_path, embedding_engine.embeddings)
            if cache_result:
                faiss_index, documents, stats = cache_result
                from_cache = True
                logger.info("✅ 成功从缓存加载索引和文档片段")

        if not from_cache:
            logger.info("📄 缓存不存在或加载失败，开始处理PDF...")
            # 处理PDF文档
            pipeline = create_pdf_pipeline()
            result = pipeline.process_pdf(str(pdf_file))
            
            if not result["success"]:
                raise ValueError(f"文档处理失败: {result.get('error')}")
            
            documents = result["chunks"]
            stats = result["stats"]
            logger.info(f"文档处理完成，共 {len(documents)} 个片段")
            
            # 创建向量索引
            index_builder = FAISSIndexBuilder(embedding_engine)
            faiss_index = index_builder.create_index(documents)
            
            # 保存到缓存
            if cache_manager:
                logger.info("💾 保存处理结果到缓存...")
                cache_manager.save_cache(
                    pdf_path,
                    index_builder.vector_store,
                    documents,
                    stats
                )
        
        # 创建检索器
        retriever = HybridRetriever(faiss_index, documents)
        
        # 创建问答链
        docqa_chain = DocQAChain(llm, retriever, reranker)
        
        logger.info("DocQA系统初始化成功")
        return docqa_chain
        
    except Exception as e:
        logger.error(f"DocQA系统初始化失败: {e}")
        raise


if __name__ == "__main__":
    # 固定参数配置
    pdf_path = 'dataset/raw/doc.pdf'
    testset_path = 'dataset/testset/testset_final.json'
    output_path = 'output/evaluation_report_no_reranker.json'

    try:
        # 创建DocQA链
        enable_rerank = bool(getattr(config, "EVAL_ENABLE_RERANK", True))
        logger.info(f"评估开关: enable_rerank={enable_rerank}")
        docqa_chain = create_docqa_chain(pdf_path, enable_rerank=enable_rerank)

        # 创建评估器
        evaluator = DocQAEvaluator(docqa_chain)

        # 执行评估
        results = evaluator.evaluate_dataset(testset_path, output_path)

        logger.info("评估完成!")

    except Exception as e:
        logger.error(f"评估失败: {e}")
        raise