import os
import json
import random
import time
import re
import glob
from typing import List, Dict
import dotenv
from openai import OpenAI
from tqdm import tqdm

# 复用你的核心模块
from core.ingestion import create_pdf_pipeline
from langchain_core.documents import Document

# 加载环境变量
dotenv.load_dotenv()

# 配置
API_KEY = os.getenv("API_KEY")
BASE_URL = os.getenv("BASE_URL")

PDF_DIR = "dataset/split_docs" 
QUESTIONS_PER_DOC = 15 # 这里的配置主要用于计算总目标，或者你可以根据文件数量自动计算
OUTPUT_FILE = "dataset/testset/testset_final.jsonl"

# 问题分布配置
DISTRIBUTION = {
    "simple": 9,        # 事实类
    "reasoning": 3,     # 推理类
    "multi_context": 3  # 多跳类
}

# 初始化 OpenAI 客户端
client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

def get_page_offset(filename: str) -> int:
    """
    解析文件名获取起始页码
    期望格式: docXX_start_end.pdf (例如: doc00_1_20.pdf)
    如果解析失败，默认返回 1
    """
    # 正则匹配：匹配文件名中的两个数字部分 _(d+)_(d+)
    # group(1) 是起始页，group(2) 是结束页
    match = re.search(r'_(\d+)_(\d+)\.pdf$', filename)

    return int(match.group(1))
    

def generate_questions_by_llm(chunks: List[Document], doc_name: str, page_offset: int) -> List[Dict]:
    """
    核心生成逻辑：根据chunks生成指定分布的问题
    Args:
        page_offset: 当前文档在原书中的起始页码
    """
    generated_data = []
    
    # 定义一个内部函数来计算绝对页码
    def get_abs_page(chunk):
        # 相对页码 (从1开始)
        relative_page = chunk.metadata.get('page', 1) 
        # 绝对页码 = 起始页 + 相对页 - 1
        return page_offset + relative_page - 1

    # --- 1. 生成 Simple (事实类) ---
    target_chunks = random.sample(chunks, k=min(len(chunks), DISTRIBUTION["simple"]))
    for chunk in target_chunks:
        abs_page = get_abs_page(chunk)
        prompt = f"""
        基于以下文档片段，生成 1 个【简单事实类(simple)】问答对。
        要求：
        1. 答案必须能直接从片段中找到。
        2. "contexts" 字段必须严格摘录原文句子。
        3. 输出纯 JSON 格式。

        文档片段 (原书第 {abs_page} 页):
        {chunk.page_content[:1500]}
        """
        data = _call_llm(prompt, abs_page, "simple", doc_name) # 传入绝对页码
        if data: generated_data.append(data)

    # --- 2. 生成 Reasoning (推理类) ---
    target_chunks = random.sample(chunks, k=min(len(chunks), DISTRIBUTION["reasoning"]))
    for chunk in target_chunks:
        abs_page = get_abs_page(chunk)
        prompt = f"""
        基于以下文档片段，生成 1 个【深度推理类(reasoning)】问答对。
        要求：
        1. 问题包含"为什么"、"如何影响"或"对比"。
        2. 需要结合片段中的逻辑进行推断才能回答。
        3. 输出纯 JSON 格式。

        文档片段 (原书第 {abs_page} 页):
        {chunk.page_content[:1500]}
        """
        data = _call_llm(prompt, abs_page, "reasoning", doc_name)
        if data: generated_data.append(data)

    # --- 3. 生成 Multi_context (多跳类) ---
    for _ in range(DISTRIBUTION["multi_context"]):
        if len(chunks) < 2: break
        c1, c2 = random.sample(chunks, 2)
        
        abs_page_1 = get_abs_page(c1)
        abs_page_2 = get_abs_page(c2)

        prompt = f"""
        基于以下两个不同的文档片段，生成 1 个【多文档综合类(multi_context)】问答对。
        要求：
        1. 问题必须需要同时结合片段A和片段B的信息才能回答。
        2. "contexts" 字段需包含两个片段中的关键句。
        3. 输出纯 JSON 格式。

        片段A (原书第 {abs_page_1} 页):
        {c1.page_content[:800]}
        
        片段B (原书第 {abs_page_2} 页):
        {c2.page_content[:800]}
        """
        # 多跳问题页码通常记录主要来源的页码，或者记录为列表。
        # 这里为了保持 schema 一致，我们记录片段A的页码
        data = _call_llm(prompt, abs_page_1, "multi_context", doc_name, extra_chunk=c2)
        if data: generated_data.append(data)

    return generated_data

def _call_llm(prompt: str, abs_page_num: int, q_type: str, doc_name: str, extra_chunk=None) -> Dict:
    """LLM 调用与 JSON 清洗通用函数"""
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini", 
            messages=[
                {"role": "system", "content": "你是一个严谨的数据集生成专家。请只返回 JSON，不要包含 Markdown 格式标记。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )
        content = response.choices[0].message.content.strip()
        
        if content.startswith("```"):
            content = content.replace("```json", "").replace("```", "")
            
        parsed = json.loads(content)
        if isinstance(parsed, list): parsed = parsed[0]
        
        return {
            "question": parsed.get("question", ""),
            "ground_truth": parsed.get("ground_truth") or parsed.get("answer", ""),
            "contexts": parsed.get("contexts", []),
            "question_type": q_type,
            "page_num": abs_page_num,  # ✅ 这里现在是绝对页码
            "source_file": doc_name    # 这里可以记录原始拆分文件名，或者你可以改为记录原书名
        }
    except Exception as e:
        print(f"  [Error] 生成失败: {e}")
        return None

def clean_and_convert_jsonl(input_path: str):
    """
    读取 .jsonl 文件，执行以下操作：
    1. 数据清洗：剔除任何包含空值（None, 空字符串, 空列表）的记录。
    2. ID 重置：对保留下来的有效数据，重新生成连续的 ID (000, 001...)。
    3. 格式转换：保存为标准 .json 格式。

    额外清洗：
    - 将 "contexts" 字段统一为 list[str]。

    Args:
        input_path (str): 输入的 .jsonl 文件路径
    """
    if not os.path.exists(input_path):
        print(f"❌ 错误：文件不存在 - {input_path}")
        return

    output_path = input_path.rsplit('.', 1)[0] + ".json"
    final_data = []

    total_lines = 0
    dropped_lines = 0

    print(f"🔄 正在清洗并转换: {input_path} ...")

    def _clean_contexts(value):
        if value is None:
            return []
        if isinstance(value, str):
            v = value.strip()
            return [v] if v else []
        if isinstance(value, list):
            cleaned = []
            for x in value:
                if x is None:
                    continue
                s = str(x).strip()
                if s:
                    cleaned.append(s)
            return cleaned
        s = str(value).strip()
        return [s] if s else []

    try:
        with open(input_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue

                total_lines += 1

                try:
                    item = json.loads(line)

                    # 先清洗 contexts，保证类型统一
                    item["contexts"] = _clean_contexts(item.get("contexts"))

                    # 再执行空值过滤
                    if any(not v for _, v in item.items()):
                        dropped_lines += 1
                        continue

                    # --- 2. 插入连续 ID ---
                    new_item = {
                        "id": f"{len(final_data):04d}",
                        **item
                    }
                    final_data.append(new_item)

                except json.JSONDecodeError:
                    print(f"❌ 第 {line_num+1} 行 JSON 格式错误，跳过。")
                    dropped_lines += 1

        # --- 3. 保存结果 ---
        if final_data:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(final_data, f, ensure_ascii=False, indent=2)

            print(f"🎉 处理完成！")
            print(f"📊 统计：原数据 {total_lines} 条 -> 有效数据 {len(final_data)} 条 (剔除 {dropped_lines} 条)")
            print(f"💾 输出文件: {output_path}")
        else:
            print("⚠️ 警告：没有提取到任何有效数据，未生成输出文件。")

    except Exception as e:
        print(f"❌ 处理过程中发生未知错误: {e}")

def main():
    pipeline = create_pdf_pipeline()

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    # ✅ 修改 1: 扫描目录下所有的 PDF 文件，而不是用 range 猜测
    # 假设文件名格式: doc00_1_20.pdf, doc01_21_40.pdf
    pdf_files = sorted(glob.glob(os.path.join(PDF_DIR, "*.pdf")))

    if not pdf_files:
        print(f"❌ 错误: 在 {PDF_DIR} 目录下没有找到 PDF 文件")
        return

    print(f"🚀 开始批量生成任务：扫描到 {len(pdf_files)} 个文件")

    for file_path in tqdm(pdf_files, desc="处理文档"):
        filename = os.path.basename(file_path)

        # ✅ 修改 2: 获取页码偏移量
        page_offset = get_page_offset(filename)

        try:
            # 1. 使用 ingestion.py 处理 PDF
            result = pipeline.process_pdf(file_path)
            if not result['success']:
                print(f"❌ 解析失败: {filename} - {result.get('error')}")
                continue

            chunks = result['chunks']
            if not chunks:
                print(f"⚠️ 警告: {filename} 未提取到文本")
                continue

            # 2. 调用 LLM 生成数据
            # ✅ 修改 3: 传入 page_offset
            print(f"  正在生成 {filename} (起始页: {page_offset})...")
            doc_questions = generate_questions_by_llm(chunks, filename, page_offset)

            with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
                for item in doc_questions:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")

            print(f"  ✅ {filename} 完成，数据已追加")
            time.sleep(1)

        except Exception as e:
            print(f"❌ 处理 {filename} 时发生未知错误: {e}")

    print(f"\n💾 LLM生成的数据集已保存至: {OUTPUT_FILE}")

    clean_and_convert_jsonl(OUTPUT_FILE)

if __name__ == "__main__":
    # main()
    clean_and_convert_jsonl(OUTPUT_FILE)
