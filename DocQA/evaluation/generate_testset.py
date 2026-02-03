import sys
import os
import dotenv
import pandas as pd
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas.testset.generator import TestsetGenerator
from ragas.testset.evolutions import simple, reasoning, multi_context
from core.ingestion import create_pdf_pipeline
    

dotenv.load_dotenv()
# --- 2. 配置参数 ---
# 请修改这里为你实际想要测试的 PDF 文件路径
INPUT_PDF_PATH = "dataset/docs/doc.pdf"
OUTPUT_CSV_PATH = "dataset/testset/testset.csv"
TEST_SIZE = 10  # 生成的问题数量，建议先设为 10 进行测试

def main():
    # --- 步骤 A: 复用你的 ingestion 模块加载文档 ---
    print(f"📂 正在加载文档: {INPUT_PDF_PATH}")
    
    if not os.path.exists(INPUT_PDF_PATH):
        print(f"❌ 错误: 文件不存在 -> {INPUT_PDF_PATH}")
        return

    # 实例化你的管道
    pipeline = create_pdf_pipeline(
        progress_callback=lambda msg, cur, tot: print(f"   [处理中] {msg} ({cur}%)")
    )
    
    # 执行处理 (包含加载、验证、切分)
    result = pipeline.process_pdf(INPUT_PDF_PATH)
    
    if not result["success"]:
        print(f"❌ 文档处理失败: {result.get('error')}")
        return

    # 获取切分好的文档块 (chunks)
    # Ragas 需要这些 chunks 来生成相关的问题
    documents = result["chunks"]
    print(f"✅ 文档加载成功！共生成 {len(documents)} 个文本块 (Chunk Size: {result['stats']['chunk_size']})")

    # --- 步骤 B: 初始化 Ragas 生成器 ---
    print("🤖 正在初始化 Ragas (GPT-4o)...")
    
    # 建议使用 GPT-4o 以获得最佳的数据生成质量
    # 确保环境变量中已设置 OPENAI_API_KEY
    generator_llm = ChatOpenAI(basemodel="gpt-4o")
    critic_llm = ChatOpenAI(model="gpt-4o")
    embeddings = OpenAIEmbeddings()

    generator = TestsetGenerator.from_langchain(
        generator_llm,
        critic_llm,
        embeddings
    )

    # --- 步骤 C: 生成测试集 ---
    print(f"🚀 开始生成测试集 (目标: {TEST_SIZE} 个问题)...")
    print("   提示: 这需要消耗一定的 Token 并花费几分钟时间。")

    # 限制用于生成的 chunks 数量以节省 Token (可选)
    # 如果 PDF 非常大，建议只取前 20-30 个 chunk 进行生成
    # docs_to_process = documents[:30] 
    docs_to_process = documents 

    try:
        testset = generator.generate_with_langchain_docs(
            docs_to_process,
            test_size=TEST_SIZE,
            distributions={
                simple: 0.5,        # 50% 简单直接检索问题
                reasoning: 0.25,    # 25% 需要逻辑推理的问题
                multi_context: 0.25 # 25% 需要综合多段内容的问题
            },
            raise_exceptions=False # 遇到个别错误继续执行
        )
    except Exception as e:
        print(f"❌ Ragas 生成过程中出错: {e}")
        return

    # --- 步骤 D: 保存结果 ---
    df = testset.to_pandas()
    
    # 简单清洗：去除生成的 NaN 行
    df = df.dropna(subset=['question', 'ground_truth'])
    
    df.to_csv(OUTPUT_CSV_PATH, index=False)
    print(f"🎉 成功！测试数据集已保存至: {OUTPUT_CSV_PATH}")
    print("\n数据预览:")
    print(df[['question', 'ground_truth']].head(3))

if __name__ == "__main__":
    main()