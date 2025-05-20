# -*- coding: utf-8 -*-
from pathlib import Path


class Config:
    """全局配置类"""
    # 模型路径
    EMBED_MODEL_PATH = r"E:\xuexiziliao\llm\text2vec-base-chinese-sentence"
    RERANK_MODEL_PATH = r"E:\xuexiziliao\llm\BAAI\bge-reranker-large"
    LLM_MODEL_PATH = r"E:\xuexiziliao\llm\Qwen\Qwen1.5-4B-Chat"

    # openai api接口
    API_KEY = "fake"
    API_BASE = "http://localhost:23333/v1"

    # 路径配置
    DATA_DIR = "./data"
    VECTOR_DB_DIR = "./chroma_db"
    PERSIST_DIR = "./storage"

    # 检索配置
    COLLECTION_NAME = "chinese_labor_laws"
    # TOP_K = 3  # 初始检索数量
    TOP_K = 20  # 初始检索数量
    RERANK_TOP_K = 5  # 重排序保留数量
    MIN_RERANK_SCORE = 0.4  # 重排序分数阈值

    # 提示词模板
    QA_TEMPLATE = (
        "<|im_start|>system\n"
        "您是中国劳动法领域专业助手，必须严格遵循以下规则：\n"
        "1.仅使用提供的法律条文回答问题\n"
        "2.若问题与劳动法无关或超出知识库范围，明确告知无法回答\n"
        "3.引用条文时标注出处\n\n"
        "可用法律条文（共{context_count}条）：\n{context_str}\n<|im_end|>\n"
        "<|im_start|>user\n问题：{query_str}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )