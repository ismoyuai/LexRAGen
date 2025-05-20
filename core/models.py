# -*- coding: utf-8 -*-
from core.config import Config
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai_like import OpenAILike
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core import Settings


class ModelInitializer:
    """模型初始化器"""

    @staticmethod
    def init_models():
        """初始化并验证所有模型"""
        # 初始化嵌入模型
        embed_model = HuggingFaceEmbedding(
            model_name=Config.EMBED_MODEL_PATH,
        )

        # LLM
        llm = HuggingFaceLLM(
            model_name=Config.LLM_MODEL_PATH,
            tokenizer_name=Config.LLM_MODEL_PATH,
            model_kwargs={
                "trust_remote_code": True,
            },
            tokenizer_kwargs={"trust_remote_code": True},
            generate_kwargs={"temperature": 0.3}
        )

        # 初始化LLM（使用OpenAILike）
        # llm = OpenAILike(
        #     model=Config.LLM_MODEL_PATH,
        #     api_base=config.API_BASE,
        #     api_key=Config.API_KEY,
        #     context_window=4096,
        #     is_chat_model=True,
        #     is_function_calling_model=False,
        # )

        # 初始化重排序器
        reranker = SentenceTransformerRerank(
            model=Config.RERANK_MODEL_PATH,
            top_n=Config.RERANK_TOP_K
        )

        # 全局设置
        Settings.embed_model = embed_model
        Settings.llm = llm

        # 模型验证
        test_embedding = embed_model.get_text_embedding("测试文本")
        print(f"嵌入维度验证：{len(test_embedding)}")

        return embed_model, llm, reranker