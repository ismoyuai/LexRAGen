# -*- coding: utf-8 -*-
import time
import numpy as np

from core.config import Config
from core.models import ModelInitializer
from core.data_processor import DataProcessor
from core.vector_store import VectorStoreManager
from core.evaluators import RecallEvaluator, E2EEvaluator
from core.benchmark_data import RETRIEVAL_BENCHMARK, E2E_BENCHMARK
from pathlib import Path
from llama_index.core import get_response_synthesizer, PromptTemplate


class LegalAssistant:
    """法律助手主程序"""
    def __init__(self):
        # 初始化组件
        self.embed_model, self.llm, self.reranker = ModelInitializer.init_models()

        # 加载或创建索引
        if not Path(Config.VECTOR_DB_DIR).exists():
            raw_data = DataProcessor.load_and_validate_json(Config.DATA_DIR)
            nodes = DataProcessor.create_nodes(raw_data)
        else:
            nodes = None

        self.vector_mgr = VectorStoreManager()
        self.index = self.vector_mgr.init_index(nodes)

        # 初始化引擎
        self.retriever = self.index.as_retriever(
            similarity_top_k=Config.TOP_K
        )
        self.query_engine = self.index.as_query_engine(
            similarity_top_k=Config.TOP_K,
            node_postprocessors=[self.reranker]
        )

        # 响应合成器
        self.response_template = PromptTemplate(Config.QA_TEMPLATE)
        self.response_synthesizer = get_response_synthesizer(
            text_qa_template=self.response_template,
            verbose=True)

    def run_evaluation(self):
        """运行评估模式"""
        # 召回率评估
        print("\n=== 开始召回率评估 ===")
        recall_evaluator = RecallEvaluator(self.retriever, self.reranker)
        recall_result = recall_evaluator.evaluate(RETRIEVAL_BENCHMARK)

        # 端到端评估
        print("\n=== 开始端到端评估 ===")
        e2e_evaluator = E2EEvaluator(self.query_engine)
        e2e_results = e2e_evaluator.evaluate(E2E_BENCHMARK)

        # 生成报告
        print("\n=== 最终评估报告 ===")
        print(f"重排序召回率：{recall_result:.1%}")
        print(f"端到端条款命中率：{np.mean([r['clause_score'] for r in e2e_results]):.1%}")
        return

    def run_qa(self):
        """运行问答模式"""
        while True:
            question = input("\n请输入劳动法问题（输入q退出）: ")
            if question.lower() == 'q':
                break

            start_time = time.time()

            # 检索与处理
            initial_nodes = self.retriever.retrieve(question)
            reranked_nodes = self.reranker.postprocess_nodes(initial_nodes, query_str=question)

            # 结果过滤
            filtered_nodes = [
                node for node in reranked_nodes
                if node.score > Config.MIN_RERANK_SCORE
            ]

            if not filtered_nodes:
                print("未找到相关法律条款")
                continue

            # 生成回答
            response = self.response_synthesizer.synthesize(
                question,
                nodes=filtered_nodes
            )

            # 显示结果
            self._display_response(response, reranked_nodes, start_time)

    def _display_response(self, response, nodes, start_time):
        """显示回答结果"""
        print(f"\n回答：\n{response.response}")
        print("\n支持依据：")
        for idx, node in enumerate(nodes, 1):
            meta = node.node.metadata
            print(f"\n[{idx}] {meta['full_title']}")
            print(f"  来源文件：{meta['source_file']}")
            print(f"  法律名称：{meta['law_name']}")
            print(f"  初始相关度：{node.node.metadata.get('initial_score', 0):.4f}")  # 安全访问
            print(f"  重排序得分：{getattr(node, 'score', 0):.4f}")  # 兼容属性访问
            print(f"  条款内容：{node.node.text[:100]}...")
            # print(f"\n[{idx}] {meta['full_title']}")
            # print(f"  来源：{meta['source_file']}")
            # print(f"  相关度：{node.score:.4f}")
            # print(f"  内容：{node.node.text[:100]}...")

        total_time = time.time() - start_time
        print(f"\n总耗时：{total_time:.2f}s")


if __name__ == "__main__":
    assistant = LegalAssistant()
    mode = input("请选择模式：1-问答 2-评估\n")

    if mode == "2":
        assistant.run_evaluation()
    else:
        assistant.run_qa()